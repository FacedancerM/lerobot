from __future__ import annotations

import json
from pathlib import Path

import finalize_phase_b_results as finalize


def test_final_result_exists_and_is_scoped() -> None:
    summary = json.loads((finalize.FINAL_ROOT / "matched_pair_summary.json").read_text())
    assert summary["status"] == "offline_phase_b_complete_matched_pair_frozen"
    assert summary["paper_result"] is False
    assert summary["hardware_accessed"] is False
    assert summary["rollout_started"] is False


def test_selected_models_are_only_step_20000() -> None:
    summary = json.loads((finalize.FINAL_ROOT / "matched_pair_summary.json").read_text())
    for model in summary["selected_models"].values():
        checkpoint = Path(model["checkpoint"])
        assert checkpoint.parent.name == "020000"
        assert finalize.sha256_file(checkpoint / "model.safetensors") == model["model_sha256"]


def test_exact_sample_consumption() -> None:
    summary = json.loads((finalize.FINAL_ROOT / "matched_pair_summary.json").read_text())
    assert summary["runs"]["A_full"]["sample_consumption"] == {"real": 1_280_000, "simulation": 0}
    assert summary["runs"]["B_full"]["sample_consumption"] == {"real": 640_000, "simulation": 640_000}


def test_independent_validation_passed() -> None:
    validation = json.loads((finalize.FINAL_ROOT / "independent_validation.json").read_text())
    assert validation["status"] == "pass"
    assert validation["run_results_verified"] == ["A_smoke", "B_smoke", "A_full", "B_full"]
