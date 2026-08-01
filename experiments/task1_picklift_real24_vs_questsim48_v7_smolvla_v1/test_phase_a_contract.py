from __future__ import annotations

import json
from pathlib import Path

import pytest

import verify_phase_a as verify


@pytest.fixture(scope="module")
def plan() -> dict:
    return json.loads((Path(__file__).parent / "candidate_plan.json").read_text())


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads((Path(__file__).parent / "phase_a_preflight_result.json").read_text())


def test_static_matched_contract(plan: dict, result: dict) -> None:
    verify.verify_static_contract(plan, result)


def test_front_only_feature_and_task_contract(plan: dict) -> None:
    assert set(plan["task_contract"]["inputs"]) == {
        "observation.state",
        "observation.images.front",
    }
    assert plan["task_contract"]["output"] == {
        "action": {"dtype": "float32", "shape": [6]}
    }
    assert plan["task_contract"]["task_text"] == (
        "Task 1 PickLift v1: grasp the 2 cm red cube and lift it >=5 cm "
        "with bilateral finger hold"
    )


def test_exact_base_and_vlm_are_frozen(result: dict) -> None:
    verify.verify_file_inventory(
        Path(result["official_base"]["cache_path"]), result["official_base"]["files"]
    )
    verify.verify_file_inventory(
        Path(result["transitive_vlm"]["cache_path"]), result["transitive_vlm"]["files"]
    )


def test_dependency_and_implementation_freeze(result: dict) -> None:
    verify.verify_dependencies(result)


def test_no_training_or_hardware_claim(result: dict) -> None:
    negative = result["negative_evidence"]
    assert negative["training_steps_executed"] == 0
    assert negative["optimizer_steps_executed"] == 0
    assert negative["checkpoints_created"] == 0
    assert not any(
        negative[key]
        for key in (
            "hardware_accessed",
            "serial_accessed",
            "hardware_camera_accessed",
            "robot_or_torque_accessed",
            "quest_or_remote_service_started",
            "mujoco_rollout_started",
            "dataset_modified",
            "push",
        )
    )


@pytest.fixture(scope="module")
def loaded_datasets(plan: dict):
    real, real_identity = verify.verify_dataset(
        plan["conditions"]["A_real24_only"], include_size=False
    )
    combined, combined_identity = verify.verify_dataset(
        plan["conditions"]["B_real24_plus_questsim48_v7"], include_size=True
    )
    return real, combined, real_identity, combined_identity


def test_dataset_identities_and_official_loader(loaded_datasets) -> None:
    real, combined, real_identity, combined_identity = loaded_datasets
    assert real_identity["sha256"] == "251cbdc079b304425ccdfbd7a08f15d34858ea0dd8c19345544b8da9f3adb9f2"
    assert combined_identity["sha256"] == "957ef4a8dc7ec3c75d8b4febbebbdd839de47b9def3647eb716a28401929bedd"
    assert real.num_episodes == 24 and real.num_frames == 3790
    assert combined.num_episodes == 72 and combined.num_frames == 10147


def test_combined_sampler_is_exact_32_plus_32(plan: dict, loaded_datasets) -> None:
    _, combined, _, _ = loaded_datasets
    evidence = verify.verify_sampler(combined, plan)
    assert evidence == {
        "complete_batches": 118,
        "samples": 7552,
        "per_domain_per_batch": 32,
        "duplicates": 0,
        "deterministic_repeat_equal": True,
    }

