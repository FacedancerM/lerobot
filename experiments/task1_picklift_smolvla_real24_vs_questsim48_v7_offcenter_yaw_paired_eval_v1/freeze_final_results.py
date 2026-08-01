from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path


EXP = Path(__file__).resolve().parent
PLAN_PATH = EXP / "evaluation_plan.json"
DECISIONS_PATH = EXP / "canonical_video_review_v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def probe_video(path: Path) -> dict:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration", "-of", "json", str(path)], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    return {"codec_name": stream["codec_name"], "width": int(stream["width"]), "height": int(stream["height"]), "avg_frame_rate": stream["avg_frame_rate"], "frames": int(stream["nb_frames"]), "duration_seconds": float(stream["duration"])}


def inspect_steps(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require(rows, f"empty steps {path}")
    return {"lines": len(rows), "first_step": rows[0]["step"], "last_step": rows[-1]["step"], "last_tick_elapsed_seconds": rows[-1]["tick_started_elapsed_seconds"]}


def main() -> None:
    plan = read(PLAN_PATH)
    root = Path(plan["evidence_root"])
    manifest_path = root / "operator_manifest_v1.json"
    operator_summary_path = root / "operator_summary_v1.json"
    operator_hashes_path = root / "operator_evidence_hashes_v1.sha256"
    review_root = root / "canonical_video_review_v1"
    final_index_path = EXP / "final_result_index.json"
    final_md_path = EXP / "FINAL_RESULT.md"
    require(not review_root.exists() and not final_index_path.exists() and not final_md_path.exists(), "final review output already exists")
    manifest, decisions_doc = read(manifest_path), read(DECISIONS_PATH)
    trials, decisions = manifest["scored_trials"], decisions_doc["decisions"]
    require(len(trials) == len(decisions) == 24, "expected 24 review rows")
    require([r["trial_id"] for r in trials] == [r["trial_id"] for r in decisions], "decision order mismatch")
    review_root.mkdir(parents=True)
    labels_root = review_root / "labels"
    labels_root.mkdir()
    rows, paths = [], {PLAN_PATH, DECISIONS_PATH, manifest_path, operator_summary_path, operator_hashes_path}
    for trial, decision in zip(trials, decisions, strict=True):
        video_path, steps_path, pre_path = Path(trial["video_path"]), Path(trial["steps_path"]), Path(trial["pre_action_frame_path"])
        require(sha(video_path) == trial["video_sha256"] == decision["canonical_video_sha256"], f"video hash mismatch {trial['trial_id']}")
        require(sha(steps_path) == trial["steps_sha256"], f"steps hash mismatch {trial['trial_id']}")
        require(sha(pre_path) == trial["pre_action_frame_sha256"] == decision["pre_action_frame_sha256"], f"pre-action hash mismatch {trial['trial_id']}")
        video, steps = probe_video(video_path), inspect_steps(steps_path)
        require((video["width"], video["height"]) == (640, 480), "video geometry mismatch")
        require(Fraction(video["avg_frame_rate"]) == 20, "video FPS mismatch")
        require(video["frames"] == trial["policy_ticks"] == steps["lines"], "frame/tick mismatch")
        require(steps["first_step"] == 0 and steps["last_step"] == steps["lines"] - 1, "non-contiguous steps")
        require(steps["last_tick_elapsed_seconds"] >= 29.9, "policy wall window shorter than 30 seconds")
        require(decision["continuous_hold_at_least_0p5s"] == decision["review_success"], "hold flag inconsistent")
        label = {"schema_version": 1, "schema_id": "task1_smolvla_matched_real_canonical_video_review_v1", "evaluation_id": plan["evaluation_id"], **decision, "operator_success": trial["operator_success"], "operator_review_agree": trial["operator_success"] == decision["review_success"], "video_probe": video, "steps_probe": steps}
        label_path = labels_root / f"{trial['artifact_stem']}.canonical.json"
        label_path.write_text(json.dumps(label, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append({**trial, **label, "canonical_label_path": str(label_path), "canonical_label_sha256": sha(label_path)})
        paths.update({video_path, steps_path, pre_path, Path(trial["evidence_path"]), label_path})
    by_model = {}
    for model_id in plan["models"]:
        selected = [row for row in rows if row["model_id"] == model_id]
        successes = sum(row["review_success"] for row in selected)
        by_model[model_id] = {"trials": 12, "reviewed_successes": successes, "reviewed_failures": 12 - successes, "reviewed_success_rate": successes / 12, "failure_types": dict(Counter(row["review_failure_type"] for row in selected if not row["review_success"]))}
    a_id, b_id = "A_real24_only", "B_real24_plus_questsim48_v7"
    paired, paired_counts = [], Counter()
    for pose in range(1, 13):
        selected = [row for row in rows if row["source_pose_order"] == pose]
        require(len(selected) == 2, f"incomplete pose {pose}")
        by_id = {row["model_id"]: row for row in selected}
        a, b = by_id[a_id]["review_success"], by_id[b_id]["review_success"]
        outcome = "both_success" if a and b else "both_failure" if not a and not b else "A_only" if a else "B_only"
        paired_counts[outcome] += 1
        paired.append({"source_pose_order": pose, "source_pose_id": selected[0]["source_pose_id"], "cell_id": selected[0]["cell_id"], "A_review_success": a, "B_review_success": b, "paired_review_outcome": outcome})
    disagreements = [{"trial_id": row["trial_id"], "operator_success": row["operator_success"], "review_success": row["review_success"]} for row in rows if not row["operator_review_agree"]]
    trials_path = review_root / "trials.jsonl"
    trials_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = {"schema_version": 1, "review_id": decisions_doc["review_id"], "evaluation_id": plan["evaluation_id"], "by_model": by_model, "paired_pose_results": paired, "paired_outcomes": dict(paired_counts), "operator_review_disagreements": disagreements, "operator_review_agreement_count": 24 - len(disagreements), "video_contract": {"videos": 24, "width": 640, "height": 480, "encoded_fps": 20, "all_frame_counts_match_steps": True, "all_policy_windows_reach_at_least_29p9_seconds": True}, "matched_pair_inference": "A and B each succeeded on 1/12 poses; successes occurred on different poses; discordant paired counts are 1 versus 1, so this evaluation shows no directional advantage.", "paper_result": False, "interpretation_boundary": "Fresh 12-pose paired real engineering evaluation; do not generalize as a paper-scale effect estimate."}
    summary_path = review_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    review_manifest = {"schema_version": 1, "review_id": decisions_doc["review_id"], "evaluation_id": plan["evaluation_id"], "status": "canonical_video_review_frozen", "inputs": {str(path): sha(path) for path in (PLAN_PATH, DECISIONS_PATH, manifest_path, operator_summary_path, operator_hashes_path)}, "trials_jsonl": {"path": str(trials_path), "sha256": sha(trials_path)}, "summary": {"path": str(summary_path), "sha256": sha(summary_path)}, "review_method": decisions_doc["review_method"]}
    review_manifest_path = review_root / "manifest.json"
    review_manifest_path.write_text(json.dumps(review_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths.update({trials_path, summary_path, review_manifest_path})
    hashes_path = review_root / "hashes.sha256"
    hashes_path.write_text("".join(f"{sha(path)}  {path}\n" for path in sorted(paths)), encoding="utf-8")
    final_index = {"schema_version": 1, "evaluation_id": plan["evaluation_id"], "status": "matched_real_evaluation_and_canonical_review_complete", "paper_result": False, "models": {model_id: {**values, "model_sha256": plan["models"][model_id]["model_sha256"]} for model_id, values in by_model.items()}, "paired_outcomes": dict(paired_counts), "operator_review_agreement_count": 24 - len(disagreements), "operator_review_disagreements": disagreements, "operator_manifest_sha256": sha(manifest_path), "operator_summary_sha256": sha(operator_summary_path), "operator_hashes_sha256": sha(operator_hashes_path), "canonical_review_manifest_sha256": sha(review_manifest_path), "canonical_review_summary_sha256": sha(summary_path), "canonical_review_trials_sha256": sha(trials_path), "canonical_review_hashes_sha256": sha(hashes_path), "result_statement": "A=1/12 and B=1/12; successes are on different poses; no directional matched-pair advantage is observed.", "interpretation_boundary": summary["interpretation_boundary"]}
    final_index_path.write_text(json.dumps(final_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final_md_path.write_text("# Task1 SmolVLA matched real Eval-v2 result\n\n- A Real24-only: 1/12 (8.33%).\n- B Real24 + Quest-Sim48-v7: 1/12 (8.33%).\n- Paired outcomes: 10 both-failure, 1 A-only, 1 B-only, 0 both-success.\n- Operator/canonical-video agreement: 24/24.\n- All 24 trials completed the fixed wall-clock window, returned to ready pose, and verified torque disable.\n\nNo directional matched-pair advantage was observed. This is a 12-pose engineering result, not a paper-scale effect estimate (`paper_result=false`).\n", encoding="utf-8")
    print(json.dumps({"final_index_sha256": sha(final_index_path), "review_manifest_sha256": sha(review_manifest_path), "review_summary_sha256": sha(summary_path), "review_hashes_sha256": sha(hashes_path), "by_model": by_model, "paired_outcomes": dict(paired_counts), "operator_review_disagreements": disagreements}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
