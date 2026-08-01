from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent
EVIDENCE_ROOT = Path("/home/ubuntu24/Teleop/artifacts/evidence/task1_picklift_real24_vs_questsim48_v7_smolvla_v1/phase_b_v1")
FINAL_ROOT = EVIDENCE_ROOT / "matched_pair_result_v1"
RESULT_INDEX = EXPERIMENT_ROOT / "phase_b_result_index.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_output_inventory(result: dict) -> None:
    config = load_json(Path(result["config"]))
    root = Path(config["output_dir"])
    for row in result["output_inventory"]:
        path = root / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert sha256_file(path) == row["sha256"]
    actual = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    assert actual == [row["path"] for row in result["output_inventory"]]


def main() -> None:
    summary_path = FINAL_ROOT / "matched_pair_summary.json"
    inventory_path = FINAL_ROOT / "primary_inventory.json"
    primary_hashes_path = FINAL_ROOT / "primary_hashes.sha256"
    validation_path = FINAL_ROOT / "independent_validation.json"
    evidence_hashes_path = FINAL_ROOT / "evidence_hashes.sha256"
    for path in (validation_path, evidence_hashes_path, RESULT_INDEX):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite {path}")

    summary = load_json(summary_path)
    inventory = load_json(inventory_path)
    assert summary["status"] == "offline_phase_b_complete_matched_pair_frozen"
    assert summary["execution_order"] == ["A_smoke", "B_smoke", "A_full", "B_full"]
    assert summary["paper_result"] is False
    assert summary["hardware_accessed"] is False
    assert summary["rollout_started"] is False
    assert summary["pushed"] is False

    lines = [line for line in primary_hashes_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == len(inventory["files"])
    for row, line in zip(inventory["files"], lines, strict=True):
        assert line == f"{row['sha256']}  {row['path']}"
        path = Path(row["path"])
        assert path.stat().st_size == row["bytes"]
        assert sha256_file(path) == row["sha256"]

    expected = {
        "A_smoke": (500, {"real": 32_000, "simulation": 0}),
        "B_smoke": (500, {"real": 16_000, "simulation": 16_000}),
        "A_full": (20_000, {"real": 1_280_000, "simulation": 0}),
        "B_full": (20_000, {"real": 640_000, "simulation": 640_000}),
    }
    verified_models = {}
    for run_id, (steps, samples) in expected.items():
        result_path = EVIDENCE_ROOT / "runs" / run_id / "run_result.json"
        validation_source = EVIDENCE_ROOT / "runs" / run_id / "offline_validation.json"
        result = load_json(result_path)
        offline = load_json(validation_source)
        assert result["status"] == "pass" and result["optimizer_steps"] == steps
        assert result["sample_consumption"] == samples
        assert result["hardware_accessed"] is False and result["rollout_started"] is False and result["push"] is False
        assert sha256_file(Path(result["log"])) == result["log_sha256"]
        assert sha256_file(validation_source) == result["offline_validation_sha256"]
        assert offline["status"] == "pass" and offline["steps"] == steps
        assert all(row["action_finite"] and row["action_shape"] == [1, 6] for row in offline["samples"])
        verify_output_inventory(result)
        checkpoint_model = Path(offline["checkpoint"]) / "model.safetensors"
        assert sha256_file(checkpoint_model) == result["checkpoint_model_sha256"] == offline["model_sha256"]
        verified_models[run_id] = result["checkpoint_model_sha256"]

    assert summary["selected_models"]["A_real24_only"]["model_sha256"] == verified_models["A_full"]
    assert summary["selected_models"]["B_real24_plus_questsim48_v7"]["model_sha256"] == verified_models["B_full"]
    validation = {
        "schema": "task1_smolvla_phase_b_independent_validation_v1",
        "status": "pass",
        "validated_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "primary_files_verified": len(inventory["files"]),
        "run_results_verified": list(expected),
        "selected_models_verified": {
            "A_real24_only": verified_models["A_full"],
            "B_real24_plus_questsim48_v7": verified_models["B_full"],
        },
        "sample_consumption_verified": {run_id: samples for run_id, (_, samples) in expected.items()},
        "hardware_accessed": False,
        "rollout_started": False,
        "paper_result": False,
    }
    write_new_json(validation_path, validation)
    index = {
        "schema": "task1_smolvla_phase_b_result_index_v1",
        "experiment_id": summary["experiment_id"],
        "status": summary["status"],
        "evidence_root": str(FINAL_ROOT),
        "matched_pair_summary_sha256": sha256_file(summary_path),
        "primary_inventory_sha256": sha256_file(inventory_path),
        "primary_hashes_sha256": sha256_file(primary_hashes_path),
        "independent_validation_sha256": sha256_file(validation_path),
        "selected_models": summary["selected_models"],
        "hardware_accessed": False,
        "rollout_started": False,
        "pushed": False,
        "paper_result": False,
    }
    write_new_json(RESULT_INDEX, index)
    evidence_files = [summary_path, inventory_path, primary_hashes_path, validation_path, RESULT_INDEX]
    evidence_hashes_path.write_text(
        "".join(f"{sha256_file(path)}  {path}\n" for path in evidence_files), encoding="utf-8"
    )
    for line in evidence_hashes_path.read_text(encoding="utf-8").splitlines():
        expected_sha, path_text = line.split("  ", 1)
        assert sha256_file(Path(path_text)) == expected_sha
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
