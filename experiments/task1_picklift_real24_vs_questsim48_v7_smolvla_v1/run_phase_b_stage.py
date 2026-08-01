from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from prepare_phase_b import ARTIFACT_ROOT, EVIDENCE_ROOT, EXPERIMENT_ROOT

RUNS = {
    "A_smoke": {"config": "train_A_smoke_500.json", "condition": "A", "steps": 500, "predecessors": []},
    "B_smoke": {"config": "train_B_smoke_500.json", "condition": "B", "steps": 500, "predecessors": ["A_smoke"]},
    "A_full": {"config": "train_A_full_20000.json", "condition": "A", "steps": 20000, "predecessors": ["A_smoke", "B_smoke"]},
    "B_full": {"config": "train_B_full_20000.json", "condition": "B", "steps": 20000, "predecessors": ["A_full"]},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iso_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def inventory(root: Path) -> list[dict]:
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


def gate(run_id: str) -> None:
    for predecessor in RUNS[run_id]["predecessors"]:
        result_path = EVIDENCE_ROOT / "runs" / predecessor / "run_result.json"
        if not result_path.is_file() or json.loads(result_path.read_text())["status"] != "pass":
            raise RuntimeError(f"Predecessor gate not passed: {predecessor}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", choices=RUNS)
    args = parser.parse_args()
    spec = RUNS[args.run_id]
    gate(args.run_id)
    config_path = EXPERIMENT_ROOT / spec["config"]
    config = json.loads(config_path.read_text())
    run_root = Path(config["output_dir"])
    evidence_dir = EVIDENCE_ROOT / "runs" / args.run_id
    log_path = ARTIFACT_ROOT / "logs" / f"{args.run_id}.log"
    if run_root.exists() or evidence_dir.exists() or log_path.exists():
        raise RuntimeError(f"Refusing to rerun or overwrite frozen run {args.run_id}")
    evidence_dir.mkdir(parents=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONHASHSEED": "1000",
        }
    )
    command = [
        sys.executable,
        str(EXPERIMENT_ROOT / "train_phase_b.py"),
        "--config_path",
        str(config_path),
    ]
    start_utc = iso_now()
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(json.dumps({"command": command, "environment": {key: env[key] for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED")}}, sort_keys=True) + "\n")
        process = subprocess.Popen(
            command,
            cwd=EXPERIMENT_ROOT.parents[1],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        returncode = process.wait()
    end_utc = iso_now()
    duration_s = time.perf_counter() - start
    if returncode != 0:
        failure = {
            "schema": "task1_smolvla_phase_b_run_result_v1",
            "run_id": args.run_id,
            "status": "failed_engineering_error",
            "returncode": returncode,
            "start_utc": start_utc,
            "end_utc": end_utc,
            "duration_s": duration_s,
            "config_sha256": sha256_file(config_path),
            "log_sha256": sha256_file(log_path),
        }
        (evidence_dir / "run_result.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise SystemExit(returncode)

    validation_path = evidence_dir / "offline_validation.json"
    validation_command = [
        sys.executable,
        str(EXPERIMENT_ROOT / "validate_phase_b_checkpoint.py"),
        "--condition",
        spec["condition"],
        "--steps",
        str(spec["steps"]),
        "--run-root",
        str(run_root),
        "--output",
        str(validation_path),
    ]
    subprocess.run(validation_command, cwd=EXPERIMENT_ROOT.parents[1], env=env, check=True)
    validation = json.loads(validation_path.read_text())
    sampling = None
    if spec["condition"] == "B":
        sampling_path = run_root / "domain_sampling_counts.json"
        sampling = json.loads(sampling_path.read_text())
        expected = spec["steps"] * 32
        if sampling["actual_samples_seen_by_main_process"] != {"real": expected, "simulation": expected}:
            raise RuntimeError("Actual B sample consumption is not exact 32+32")

    output_inventory = inventory(run_root)
    result = {
        "schema": "task1_smolvla_phase_b_run_result_v1",
        "run_id": args.run_id,
        "status": "pass",
        "condition": spec["condition"],
        "optimizer_steps": spec["steps"],
        "batch_size": 64,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "duration_s": duration_s,
        "command": command,
        "environment": {key: env[key] for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED")},
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
        "checkpoint_model_sha256": validation["model_sha256"],
        "offline_validation_sha256": sha256_file(validation_path),
        "sample_consumption": (
            sampling["actual_samples_seen_by_main_process"]
            if sampling is not None
            else {"real": spec["steps"] * 64, "simulation": 0}
        ),
        "output_inventory": output_inventory,
        "output_file_count": len(output_inventory),
        "output_bytes": sum(row["bytes"] for row in output_inventory),
        "hardware_accessed": False,
        "rollout_started": False,
        "push": False,
    }
    (evidence_dir / "run_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("run_id", "status", "duration_s", "checkpoint_model_sha256", "sample_consumption")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
