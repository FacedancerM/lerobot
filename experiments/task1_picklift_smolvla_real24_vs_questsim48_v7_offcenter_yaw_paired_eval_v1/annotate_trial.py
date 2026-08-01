from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from paired_evaluator import BASE, load_frozen_plan, sha256_file


FAILURE_CATEGORIES = {
    "missed_grasp",
    "insufficient_lift",
    "dropped_before_0p5s",
    "timeout",
    "unknown",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write one immutable SmolVLA matched-eval operator label.")
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--operator-label", choices=("success", "failure"), required=True)
    parser.add_argument("--failure-category", choices=sorted(FAILURE_CATEGORIES))
    parser.add_argument("--placement-confirmed", action="store_true", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    plan = load_frozen_plan()
    trial = BASE.find_trial(plan, args.trial_id)
    stem = trial["spawn_region"]
    evidence_path = Path(plan["evidence_root"]) / "trials" / f"{stem}.json"
    if not evidence_path.exists():
        raise RuntimeError(f"trial evidence does not exist: {evidence_path}")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "completed_pending_operator_annotation" or evidence.get("run_error") is not None:
        raise RuntimeError("trial evidence is not a completed valid trial")
    if args.operator_label == "success" and args.failure_category is not None:
        raise RuntimeError("successful label cannot have a failure category")
    if args.operator_label == "failure" and args.failure_category is None:
        raise RuntimeError("failed label requires a failure category")

    labels_root = Path(plan["evidence_root"]) / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)
    output = labels_root / f"{stem}.operator.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite immutable label: {output}")
    label = {
        "schema_version": 1,
        "schema_id": "task1_smolvla_matched_real_operator_annotation_v1",
        "evaluation_id": plan["evaluation_id"],
        "trial_id": trial["trial_id"],
        "artifact_stem": stem,
        "cell_id": trial["cell_id"],
        "model_id": trial["model_id"],
        "engine_evidence_path": str(evidence_path),
        "engine_evidence_sha256": sha256_file(evidence_path),
        "operator_label": args.operator_label,
        "failure_category": args.failure_category or "none",
        "placement_confirmed": args.placement_confirmed,
        "annotation_utc": datetime.now(UTC).isoformat(),
        "notes": args.notes,
        "notes_are_operator_observation_not_measurement_truth": True,
        "success_definition": plan["success_contract"]["criterion"],
        "canonical_video_review": {"status": "pending"},
    }
    output.write_text(json.dumps(label, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved immutable operator label to {output}")


if __name__ == "__main__":
    main()
