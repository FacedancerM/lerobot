from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parent
PLAN_PATH = EXP / "evaluation_plan.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> None:
    plan = read(PLAN_PATH)
    root = Path(plan["evidence_root"])
    manifest_path = root / "operator_manifest_v1.json"
    summary_path = root / "operator_summary_v1.json"
    hashes_path = root / "operator_evidence_hashes_v1.sha256"
    require(not any(path.exists() for path in (manifest_path, summary_path, hashes_path)), "operator freeze already exists")
    files: set[Path] = {PLAN_PATH}
    scored = []
    for trial in plan["trials"]:
        stem = trial["spawn_region"]
        evidence_path = root / "trials" / f"{stem}.json"
        sidecar_path = root / "trials" / f"{stem}.paired_evalv2.json"
        label_path = root / "labels" / f"{stem}.operator.json"
        for path in (evidence_path, sidecar_path, label_path):
            require(path.exists(), f"missing {path}")
        evidence, sidecar, label = read(evidence_path), read(sidecar_path), read(label_path)
        require(evidence["status"] == "completed_pending_operator_annotation", f"incomplete {stem}")
        require(evidence["termination"] == "maximum_duration" and evidence["run_error"] is None, f"invalid termination {stem}")
        require(evidence["evaluation_plan_sha256"] == sha(PLAN_PATH), f"plan mismatch {stem}")
        require(evidence["evaluation_profile_sha256"] == sha(EXP / "real_evaluation_profile_return3s_v1.json"), f"profile mismatch {stem}")
        require(evidence["model_sha256"] == plan["models"][trial["model_id"]]["model_sha256"], f"model mismatch {stem}")
        require(evidence["paired_evalv2_trial"] == trial and sidecar["trial"] == trial, f"trial mismatch {stem}")
        require(evidence["ready_pose_alignment"]["result"]["status"] == "ready_pose_observed", f"ready failure {stem}")
        require(evidence["automatic_return"]["result"]["status"] == "ready_pose_observed", f"return failure {stem}")
        require(evidence["torque_disable_verified"] is True, f"torque not disabled {stem}")
        require(evidence["video"]["exists"] is True, f"video missing {stem}")
        require(evidence["video"]["frames"] == evidence["steps_jsonl"]["lines"], f"tick mismatch {stem}")
        require(evidence["upstream_action_modified_events"] == 0, f"action modified {stem}")
        require(label["engine_evidence_sha256"] == sha(evidence_path), f"label binding mismatch {stem}")
        require(label["operator_label"] in {"success", "failure"}, f"label incomplete {stem}")
        related = [
            evidence_path, sidecar_path, label_path,
            Path(evidence["video"]["path"]), Path(evidence["steps_jsonl"]["path"]),
            Path(evidence["ready_pose_alignment"]["trajectory"]["path"]),
            Path(evidence["automatic_return"]["trajectory"]["path"]),
            Path(sidecar["pre_action_frame"]["path"]),
        ]
        for path in related:
            require(path.exists(), f"missing related artifact {path}")
            files.add(path)
        scored.append({
            "order": trial["order"], "trial_id": trial["trial_id"], "source_pose_order": trial["source_pose_order"],
            "source_pose_id": trial["source_pose_id"], "cell_id": trial["cell_id"], "model_id": trial["model_id"],
            "model_sha256": evidence["model_sha256"], "artifact_stem": stem,
            "operator_success": label["operator_label"] == "success", "failure_category": label["failure_category"],
            "operator_notes": label["notes"], "policy_ticks": evidence["steps_jsonl"]["lines"],
            "video_path": evidence["video"]["path"], "video_sha256": evidence["video"]["sha256"],
            "steps_path": evidence["steps_jsonl"]["path"], "steps_sha256": evidence["steps_jsonl"]["sha256"],
            "pre_action_frame_path": sidecar["pre_action_frame"]["path"],
            "pre_action_frame_sha256": sidecar["pre_action_frame"]["sha256"],
            "evidence_path": str(evidence_path), "evidence_sha256": sha(evidence_path),
            "ready_maximum_absolute_error_degrees": evidence["ready_pose_alignment"]["result"]["maximum_absolute_error"],
            "return_maximum_absolute_error_degrees": evidence["automatic_return"]["result"]["maximum_absolute_error"],
            "canonical_video_review_status": "pending",
        })
    require(len(scored) == 24, "expected 24 trials")
    by_model = {}
    for model_id in plan["models"]:
        rows = [row for row in scored if row["model_id"] == model_id]
        successes = sum(row["operator_success"] for row in rows)
        by_model[model_id] = {"trials": 12, "operator_successes": successes, "operator_failures": 12 - successes, "operator_success_rate": successes / 12, "failure_categories": dict(Counter(row["failure_category"] for row in rows if not row["operator_success"]))}
    paired, paired_counts = [], Counter()
    a_id, b_id = "A_real24_only", "B_real24_plus_questsim48_v7"
    for pose in range(1, 13):
        rows = [row for row in scored if row["source_pose_order"] == pose]
        require(len(rows) == 2, f"incomplete pose {pose}")
        by_id = {row["model_id"]: row for row in rows}
        a, b = by_id[a_id]["operator_success"], by_id[b_id]["operator_success"]
        outcome = "both_success" if a and b else "both_failure" if not a and not b else "A_only" if a else "B_only"
        paired_counts[outcome] += 1
        paired.append({"source_pose_order": pose, "source_pose_id": rows[0]["source_pose_id"], "cell_id": rows[0]["cell_id"], "A_success": a, "B_success": b, "paired_outcome": outcome})
    manifest = {"schema_version": 1, "evaluation_id": plan["evaluation_id"], "result_scope": "operator labels only; canonical-video review pending", "plan_sha256": sha(PLAN_PATH), "profile_sha256": sha(EXP / "real_evaluation_profile_return3s_v1.json"), "scored_trials": scored, "infrastructure_invalid_originals": []}
    summary = {"schema_version": 1, "evaluation_id": plan["evaluation_id"], "result_scope": "operator labels only; canonical-video review pending", "by_model": by_model, "paired_pose_results": paired, "paired_outcomes": dict(paired_counts), "infrastructure_invalid_original_count": 0, "canonical_video_review_status": "pending"}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files.update({manifest_path, summary_path})
    hashes_path.write_text("".join(f"{sha(path)}  {path}\n" for path in sorted(files)), encoding="utf-8")
    print(json.dumps({"manifest_sha256": sha(manifest_path), "summary_sha256": sha(summary_path), "hashes_sha256": sha(hashes_path), "by_model": by_model, "paired_outcomes": dict(paired_counts)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
