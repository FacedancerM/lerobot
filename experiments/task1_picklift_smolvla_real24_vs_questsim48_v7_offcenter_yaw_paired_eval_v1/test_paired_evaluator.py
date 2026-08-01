from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

import paired_evaluator as evaluator


def plan() -> dict:
    return evaluator.load_frozen_plan()


def test_fresh_identity_and_no_hardware_authorization() -> None:
    value = plan()
    assert value["protocol_revision"]["fresh_evidence"] is True
    assert value["authorization"] == {"hardware_authorized": False, "first_hardware_action_requires_later_explicit_go": True, "serial_accessed_during_preparation": False, "camera_accessed_during_preparation": False, "robot_accessed_during_preparation": False, "torque_accessed_during_preparation": False, "rollout_executed_during_preparation": False}


def test_exact_source_pose_pairs_and_balanced_first_model() -> None:
    trials = plan()["trials"]
    assert len(trials) == 24
    assert [row["model_id"] for row in trials[:4]] == ["B_real24_plus_questsim48_v7", "A_real24_only", "A_real24_only", "B_real24_plus_questsim48_v7"]
    assert sum(trials[index]["model_id"] == "A_real24_only" for index in range(0, 24, 2)) == 6
    assert sum(trials[index]["model_id"] == "B_real24_plus_questsim48_v7" for index in range(0, 24, 2)) == 6


def test_static_checkpoint_processor_and_prompt_contract() -> None:
    static = evaluator.verify_static_files(plan())
    assert set(static["models"]) == set(evaluator.MODEL_IDS)
    assert all(row["chunk_size"] == 50 and row["n_action_steps"] == 50 for row in static["models"].values())
    assert all(row["task_prompt"] == evaluator.TASK_TEXT for row in static["models"].values())


def test_official_engine_imports_without_hardware_access() -> None:
    engine = evaluator.load_engine(plan())
    assert engine.POLICY_CHUNK_SIZE == 50
    assert engine.POLICY_ACTION_STEPS == 50
    assert engine.CONTROL_FPS == 20


def test_official_engine_loads_frozen_contract_before_device_preflight() -> None:
    value = plan()
    engine = evaluator.load_engine(value)
    loaded_plan, profile, trial = engine.load_frozen_contract(
        argparse.Namespace(
            plan=evaluator.PLAN_PATH,
            profile=evaluator.PROFILE_PATH,
            maximum_trial_seconds=30.0,
            spawn_region=value["trials"][0]["spawn_region"],
        )
    )
    assert loaded_plan["evaluation_id"] == value["evaluation_id"]
    assert profile["profile_id"] == value["evaluation_profile"]["profile_id"]
    assert trial["trial_id"] == "t01"


def test_fake_24_trial_protocol_has_no_devices() -> None:
    result = evaluator.BASE.run_fake_protocol(plan())
    assert result["real_device_accessed"] is False
    assert result["trials_exercised"] == 24
    assert result["policy_reset_calls"] == {"A_real24_only": 12, "B_real24_plus_questsim48_v7": 12}
    assert result["all_official_sent_equals_requested"] is True


def test_ready_return_and_no_catchup() -> None:
    ready = evaluator.BASE.run_fake_interpolated_ready_probe(plan())
    assert ready["commands_sent"] == 60
    assert ready["all_official_sent_equals_requested"] is True

    class Clock:
        value = 0.0
        @classmethod
        def now(cls): return cls.value
        @classmethod
        def sleep(cls, seconds): cls.value += seconds
    def tick(step, tick_started, loop_started):
        del step, tick_started, loop_started
        Clock.value += 0.08
        return {}
    rows = evaluator.BASE.run_paced_ticks(0.22, tick, lambda row: None, period=0.05, now_fn=Clock.now, sleep_fn=Clock.sleep)
    assert len(rows) == 3
    assert [row["scheduled_sleep_seconds"] for row in rows] == [0.0, 0.0, 0.0]


def test_execution_order_fails_closed(tmp_path: Path) -> None:
    value = copy.deepcopy(plan())
    value["evidence_root"] = str(tmp_path)
    evaluator.BASE.validate_execution_order(value, value["trials"][0], replacement=False)
    with pytest.raises(RuntimeError, match="next missing trial"):
        evaluator.BASE.validate_execution_order(value, value["trials"][1], replacement=False)


def test_success_and_evidence_schema_unchanged() -> None:
    value = plan()
    assert value["success_contract"]["changes_policy_action_window"] is False
    assert value["success_contract"]["must_remain_held_until_timeout"] is False
    assert "operator_label" in value["evidence_contract"]["per_trial"]
    assert "canonical_video_review_label" in value["evidence_contract"]["per_trial"]


def test_plan_tamper_fails_closed(tmp_path: Path, monkeypatch) -> None:
    tampered = tmp_path / "evaluation_plan.json"
    tampered.write_bytes(evaluator.PLAN_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(evaluator, "PLAN_PATH", tampered)
    with pytest.raises(RuntimeError, match="frozen plan_sha256 mismatch"):
        evaluator.load_frozen_plan()


def test_explicit_operator_and_canonical_review_schemas() -> None:
    operator = json.loads((evaluator.EXPERIMENT_DIR / "operator_annotation_schema_v1.json").read_text())
    review = json.loads((evaluator.EXPERIMENT_DIR / "canonical_video_review_schema_v1.json").read_text())
    assert operator["immutable_write"] is True
    assert {"operator_label", "failure_category", "placement_confirmed"} <= set(operator["required"])
    assert review["immutable_write"] is True
    assert {"canonical_video_sha256", "pre_action_frame_sha256", "continuous_hold_at_least_0p5s"} <= set(review["required"])
    assert review["paired_result_rule"].startswith("Freeze all 24 canonical reviews")
