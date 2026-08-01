from __future__ import annotations

import json
from pathlib import Path

import verify_phase_b_contract as verify
from train_phase_b import FullBatchEpisodeAwareSampler


def test_research_inputs_match_freeze() -> None:
    verify.verify_research_inputs()


def test_exact_local_binding_is_front_only() -> None:
    binding = verify.verify_binding()
    assert set(binding["features"]) == {"observation.state", "observation.images.front"}
    assert binding["policy_vlm_model_name"] == binding["tokenizer_name"]


def test_four_configs_share_one_policy_recipe() -> None:
    configs = {key: verify.verify_config(name, require_absent_output=True) for key, name in verify.CONFIGS.items()}
    verify.verify_config_pair(configs)


def test_exact_execution_order_is_frozen() -> None:
    plan = json.loads((Path(__file__).parent / "research_phase_b_plan.json").read_text())
    assert [row["run_id"] for row in plan["predeclared_execution_order"]] == [
        "A_real24_only_smoke_500",
        "B_real24_plus_questsim48_v7_smoke_500",
        "A_real24_only_full_20000",
        "B_real24_plus_questsim48_v7_full_20000",
    ]
    assert plan["authorization_token"] == "GO_TASK1_SMOLVLA_MATCHED_PAIR_PHASE_B_V1"


def test_phase_a_inputs_and_balanced_sampler() -> None:
    evidence = verify.verify_phase_a_inputs()
    assert evidence["real24"]["sha256"] == "251cbdc079b304425ccdfbd7a08f15d34858ea0dd8c19345544b8da9f3adb9f2"
    assert evidence["combined72"]["sha256"] == "957ef4a8dc7ec3c75d8b4febbebbdd839de47b9def3647eb716a28401929bedd"
    assert evidence["B_sampler"] == {
        "complete_batches": 118,
        "samples": 7552,
        "per_domain_per_batch": 32,
        "duplicates": 0,
        "deterministic_repeat_equal": True,
    }
    assert evidence["A_sampler"] == {
        "complete_batches": 59,
        "samples": 3776,
        "batch_size": 64,
        "duplicates": 0,
        "deterministic_repeat_equal": True,
    }


def test_real_sampler_emits_only_complete_deterministic_batches() -> None:
    dataset = verify_phase_a_dataset()
    kwargs = {
        "dataset_from_indices": dataset.meta.episodes["dataset_from_index"],
        "dataset_to_indices": dataset.meta.episodes["dataset_to_index"],
        "episode_indices_to_use": dataset.episodes,
        "shuffle": True,
        "seed": 1000,
        "absolute_to_relative_idx": dataset.absolute_to_relative_idx,
    }
    first = list(FullBatchEpisodeAwareSampler(**kwargs))
    repeat = list(FullBatchEpisodeAwareSampler(**kwargs))
    assert len(first) == 3776 == 59 * 64
    assert first == repeat
    assert len(set(first)) == len(first)


def verify_phase_a_dataset():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        "local/task1_picklift_formal24_s03_20260728",
        root="/home/ubuntu24/Teleop/artifacts/task1_picklift_formal24_s03_20260728",
        video_backend="pyav",
    )
