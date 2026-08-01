from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from prepare_phase_b import ARTIFACT_ROOT, EVIDENCE_ROOT, EXPERIMENT_ROOT


FINAL_ROOT = EVIDENCE_ROOT / "matched_pair_result_v1"
PLAN_SHA256 = "4e92dc821d7e3ea334c18b7278c8e918fced82bf0a833d29d55307faea9cdc7e"
RUNS = {
    "A_smoke": {
        "condition": "A",
        "steps": 500,
        "config": "train_A_smoke_500.json",
        "samples": {"real": 32_000, "simulation": 0},
        "domains": ["real"],
    },
    "B_smoke": {
        "condition": "B",
        "steps": 500,
        "config": "train_B_smoke_500.json",
        "samples": {"real": 16_000, "simulation": 16_000},
        "domains": ["real", "simulation"],
    },
    "A_full": {
        "condition": "A",
        "steps": 20_000,
        "config": "train_A_full_20000.json",
        "samples": {"real": 1_280_000, "simulation": 0},
        "domains": ["real"],
    },
    "B_full": {
        "condition": "B",
        "steps": 20_000,
        "config": "train_B_full_20000.json",
        "samples": {"real": 640_000, "simulation": 640_000},
        "domains": ["real", "simulation"],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def output_inventory(root: Path) -> list[dict]:
    rows = []
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def final_metrics(log_path: Path, steps: int) -> dict:
    marker = "step:500" if steps == 500 else "step:20K"
    lines = [line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if marker in line]
    if not lines:
        raise RuntimeError(f"Missing final metric line in {log_path}")
    line = lines[-1]
    match = re.search(
        r"loss:(?P<loss>[0-9.]+).*?grdn:(?P<grad>[0-9.]+).*?lr:(?P<lr>[0-9.eE+-]+).*?mem_gb:(?P<mem>[0-9.]+)",
        line,
    )
    if match is None:
        raise RuntimeError(f"Cannot parse final metric line: {line}")
    return {
        "loss": float(match.group("loss")),
        "gradient_norm": float(match.group("grad")),
        "learning_rate": float(match.group("lr")),
        "reported_peak_memory_gib": float(match.group("mem")),
        "interpretation": "training diagnostic only; not a model-performance result",
    }


def verify_run(run_id: str, spec: dict) -> tuple[dict, list[Path]]:
    evidence_dir = EVIDENCE_ROOT / "runs" / run_id
    result_path = evidence_dir / "run_result.json"
    validation_path = evidence_dir / "offline_validation.json"
    result = load_json(result_path)
    validation = load_json(validation_path)
    config_path = EXPERIMENT_ROOT / spec["config"]
    config = load_json(config_path)
    run_root = Path(config["output_dir"])
    checkpoint = run_root / "checkpoints" / f"{spec['steps']:06d}" / "pretrained_model"
    model_path = checkpoint / "model.safetensors"
    log_path = Path(result["log"])

    assert result["status"] == "pass"
    assert result["condition"] == spec["condition"]
    assert result["optimizer_steps"] == spec["steps"]
    assert result["batch_size"] == 64
    assert result["sample_consumption"] == spec["samples"]
    assert result["hardware_accessed"] is False
    assert result["rollout_started"] is False
    assert result["push"] is False
    assert config["resume"] is False
    assert config["steps"] == spec["steps"]
    assert config["batch_size"] == 64
    assert config["seed"] == 1000
    assert sha256_file(config_path) == result["config_sha256"]
    assert sha256_file(log_path) == result["log_sha256"]
    assert sha256_file(validation_path) == result["offline_validation_sha256"]
    assert output_inventory(run_root) == result["output_inventory"]
    assert sorted(
        path.name for path in (run_root / "checkpoints").iterdir() if not path.is_symlink()
    ) == [f"{spec['steps']:06d}"]
    last_link = run_root / "checkpoints" / "last"
    assert last_link.is_symlink() and last_link.resolve() == (run_root / "checkpoints" / f"{spec['steps']:06d}").resolve()
    assert sha256_file(model_path) == result["checkpoint_model_sha256"] == validation["model_sha256"]
    assert validation["status"] == "pass"
    assert validation["steps"] == spec["steps"]
    assert validation["condition"] == spec["condition"]
    assert validation["hardware_accessed"] is False
    assert validation["rollout_started"] is False
    assert [sample["domain"] for sample in validation["samples"]] == spec["domains"]
    assert all(sample["action_finite"] and sample["action_shape"] == [1, 6] for sample in validation["samples"])

    primary_paths = [result_path, validation_path, config_path, log_path, model_path]
    if spec["condition"] == "B":
        counts_path = run_root / "domain_sampling_counts.json"
        counts = load_json(counts_path)
        assert counts["optimizer_steps"] == spec["steps"]
        assert counts["batch_size_per_process"] == 64
        assert counts["actual_samples_seen_by_main_process"] == spec["samples"]
        primary_paths.append(counts_path)

    summary = {
        "run_id": run_id,
        "condition": spec["condition"],
        "status": "pass",
        "optimizer_steps": spec["steps"],
        "duration_s": result["duration_s"],
        "start_utc": result["start_utc"],
        "end_utc": result["end_utc"],
        "sample_consumption": result["sample_consumption"],
        "checkpoint": str(checkpoint),
        "model_sha256": result["checkpoint_model_sha256"],
        "offline_validation_sha256": result["offline_validation_sha256"],
        "log_sha256": result["log_sha256"],
        "final_training_metrics": final_metrics(log_path, spec["steps"]),
    }
    return summary, primary_paths


def main() -> None:
    if FINAL_ROOT.exists():
        raise RuntimeError(f"Refusing to overwrite frozen result: {FINAL_ROOT}")
    plan_path = EXPERIMENT_ROOT / "research_phase_b_plan.json"
    if sha256_file(plan_path) != PLAN_SHA256:
        raise RuntimeError("Frozen Phase B plan mismatch")
    plan = load_json(plan_path)
    assert plan["authorization_token"] == "GO_TASK1_SMOLVLA_MATCHED_PAIR_PHASE_B_V1"
    assert plan["predeclared_execution_order"] == sorted(plan["predeclared_execution_order"], key=lambda row: row["order"])

    runs = {}
    primary_paths = [
        plan_path,
        EXPERIMENT_ROOT / "research_phase_a_manifest.json",
        EXPERIMENT_ROOT / "phase_a_preflight_result.json",
        EXPERIMENT_ROOT / "phase_b_preparation_index.json",
        EVIDENCE_ROOT / "preparation_result.json",
        EVIDENCE_ROOT / "pre_optimizer_gate.json",
        EVIDENCE_ROOT / "deep_pre_optimizer_gate.json",
        EVIDENCE_ROOT / "exact_base_binding_v1" / "binding_manifest.json",
    ]
    for run_id, spec in RUNS.items():
        runs[run_id], paths = verify_run(run_id, spec)
        primary_paths.extend(paths)

    phase_a = load_json(EXPERIMENT_ROOT / "phase_a_preflight_result.json")
    summary = {
        "schema": "task1_smolvla_matched_pair_offline_result_v1",
        "experiment_id": "task1_picklift_real24_vs_questsim48_v7_smolvla_v1",
        "status": "offline_phase_b_complete_matched_pair_frozen",
        "completed_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "research_commit": "680e1e2f1b9c007f6d75c07e3a0b758fc94b3f67",
        "authorization_token": "GO_TASK1_SMOLVLA_MATCHED_PAIR_PHASE_B_V1",
        "plan_sha256": PLAN_SHA256,
        "execution_order": ["A_smoke", "B_smoke", "A_full", "B_full"],
        "runtime": {
            "python": phase_a["dependency_freeze"]["python"],
            "packages": phase_a["dependency_freeze"]["packages"],
            "gpu": phase_a["cuda_non_training_probe"]["gpu"],
            "gpu_total_mib": phase_a["cuda_non_training_probe"]["gpu_total_mib"],
            "gpu_driver": phase_a["cuda_non_training_probe"]["driver"],
        },
        "common_contract": {
            "base_model_sha256": "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb",
            "vlm_model_sha256": "b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3",
            "seed": 1000,
            "batch_size": 64,
            "selected_checkpoint_step": 20000,
            "chunk_size": 50,
            "n_action_steps": 50,
            "input_features": ["observation.state", "observation.images.front"],
            "output_features": ["action"],
            "independent_initialization_from_same_base": True,
        },
        "runs": runs,
        "selected_models": {
            "A_real24_only": {
                "checkpoint": runs["A_full"]["checkpoint"],
                "model_sha256": runs["A_full"]["model_sha256"],
            },
            "B_real24_plus_questsim48_v7": {
                "checkpoint": runs["B_full"]["checkpoint"],
                "model_sha256": runs["B_full"]["model_sha256"],
            },
        },
        "confirmed_engineering_facts": [
            "Both 500-step smoke runs and both independently initialized 20000-step runs completed in the frozen order.",
            "Each selected step-20000 checkpoint reloaded through its saved processor and produced finite action[1,6] offline outputs.",
            "Condition A consumed 1280000 Real samples; condition B consumed exactly 640000 Real and 640000 Quest-Sim samples.",
            "No hardware, Quest, Remote, MuJoCo, LocalSim, or real-robot rollout was accessed and nothing was pushed.",
        ],
        "research_hypothesis": "Aligned Quest-Sim48-v7 demonstrations may improve downstream real Task1 performance under SmolVLA relative to Real24-only.",
        "requires_future_evidence": [
            "A separately frozen matched evaluation is required before any model-performance comparison.",
            "Paper-scale real-robot trials are required before any paper claim.",
        ],
        "paper_result": False,
        "hardware_accessed": False,
        "rollout_started": False,
        "pushed": False,
    }

    temp_root = Path(tempfile.mkdtemp(prefix="matched_pair_result_v1.tmp-", dir=EVIDENCE_ROOT))
    try:
        write_json(temp_root / "matched_pair_summary.json", summary)
        unique_paths = sorted(set(primary_paths), key=lambda path: str(path))
        inventory = [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in unique_paths
        ]
        write_json(temp_root / "primary_inventory.json", {"schema": "task1_smolvla_phase_b_primary_inventory_v1", "files": inventory})
        (temp_root / "primary_hashes.sha256").write_text(
            "".join(f"{row['sha256']}  {row['path']}\n" for row in inventory), encoding="utf-8"
        )
        temp_root.rename(FINAL_ROOT)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    print(json.dumps({"status": summary["status"], "final_root": str(FINAL_ROOT), "selected_models": summary["selected_models"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
