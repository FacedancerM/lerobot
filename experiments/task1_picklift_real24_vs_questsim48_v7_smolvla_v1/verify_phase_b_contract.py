from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

import verify_phase_a
from prepare_phase_b import BINDING_ROOT, EVIDENCE_ROOT, EXPERIMENT_ROOT, TASK_TEXT, VLM_ROOT
from train_phase_b import FullBatchEpisodeAwareSampler

RESEARCH_FILES = {
    "research_phase_b_plan.json": "4e92dc821d7e3ea334c18b7278c8e918fced82bf0a833d29d55307faea9cdc7e",
    "research_phase_a_manifest.json": "181054c01265f52281e2e2349cd611740939edf0264f1c6b7f4866a11e9d0687",
    "research_phase_b_decision.md": "254f5dd8edc65155a37e9538fc136cb341e5b36326aefd6dc640d022f3ab53c8",
}
CONFIGS = {
    "A_smoke": "train_A_smoke_500.json",
    "B_smoke": "train_B_smoke_500.json",
    "A_full": "train_A_full_20000.json",
    "B_full": "train_B_full_20000.json",
}
FORBIDDEN_FEATURE_TOKENS = (
    "camera1",
    "camera2",
    "camera3",
    "observation.image\"",
    "observation.image2",
    "observation.image3",
    "wrist",
    "handeye",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_research_inputs() -> None:
    for relative, expected in RESEARCH_FILES.items():
        if sha256_file(EXPERIMENT_ROOT / relative) != expected:
            raise RuntimeError(f"Research input mismatch: {relative}")


def verify_binding() -> dict:
    manifest = load(BINDING_ROOT / "binding_manifest.json")
    for relative, expected in manifest["derived_files"].items():
        path = BINDING_ROOT / relative
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"Binding file mismatch: {relative}")
    config = load(BINDING_ROOT / "config.json")
    expected_inputs = {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
    }
    if config["input_features"] != expected_inputs:
        raise RuntimeError("Binding is not exact front-only + state")
    if config["output_features"] != {"action": {"type": "ACTION", "shape": [6]}}:
        raise RuntimeError("Binding action feature mismatch")
    if config["vlm_model_name"] != str(VLM_ROOT):
        raise RuntimeError("Policy VLM is not bound to frozen local snapshot")
    preprocessor = load(BINDING_ROOT / "policy_preprocessor.json")
    tokenizer_names = [
        step["config"]["tokenizer_name"]
        for step in preprocessor["steps"]
        if step["registry_name"] == "tokenizer_processor"
    ]
    if tokenizer_names != [str(VLM_ROOT)]:
        raise RuntimeError("Tokenizer is not bound to frozen local snapshot")
    serialized = json.dumps({"policy": config, "preprocessor": preprocessor}, sort_keys=True)
    if any(token in serialized for token in FORBIDDEN_FEATURE_TOKENS):
        raise RuntimeError("Forbidden placeholder/wrist feature remains in binding")
    return {
        "manifest_sha256": sha256_file(BINDING_ROOT / "binding_manifest.json"),
        "config_sha256": sha256_file(BINDING_ROOT / "config.json"),
        "preprocessor_sha256": sha256_file(BINDING_ROOT / "policy_preprocessor.json"),
        "policy_vlm_model_name": config["vlm_model_name"],
        "tokenizer_name": tokenizer_names[0],
        "features": config["input_features"],
    }


def verify_config(name: str, *, require_absent_output: bool) -> dict:
    config = load(EXPERIMENT_ROOT / name)
    policy = config["policy"]
    if config["seed"] != 1000 or config["batch_size"] != 64:
        raise RuntimeError(f"Seed/batch mismatch in {name}")
    if policy["pretrained_path"] != str(BINDING_ROOT) or policy["vlm_model_name"] != str(VLM_ROOT):
        raise RuntimeError(f"Initialization binding mismatch in {name}")
    if policy["chunk_size"] != 50 or policy["n_action_steps"] != 50:
        raise RuntimeError(f"Action horizon mismatch in {name}")
    if policy["input_features"] != {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
    }:
        raise RuntimeError(f"Front-only feature mismatch in {name}")
    if policy["normalization_mapping"] != {
        "VISUAL": "IDENTITY",
        "STATE": "MEAN_STD",
        "ACTION": "MEAN_STD",
    }:
        raise RuntimeError(f"Normalization mismatch in {name}")
    if config["env"] is not None or config["env_eval_freq"] != 0 or config["wandb"]["enable"]:
        raise RuntimeError(f"Offline-only boundary mismatch in {name}")
    if config["resume"] or policy["push_to_hub"] or config["save_checkpoint_to_hub"]:
        raise RuntimeError(f"Resume/push mismatch in {name}")
    if require_absent_output and Path(config["output_dir"]).exists():
        raise RuntimeError(f"Fresh output already exists: {config['output_dir']}")
    return config


def verify_config_pair(configs: dict[str, dict]) -> None:
    if configs["A_smoke"]["steps"] != 500 or configs["B_smoke"]["steps"] != 500:
        raise RuntimeError("Smoke step mismatch")
    if configs["A_full"]["steps"] != 20000 or configs["B_full"]["steps"] != 20000:
        raise RuntimeError("Full step mismatch")
    if configs["A_full"]["save_freq"] != 20000 or configs["B_full"]["save_freq"] != 20000:
        raise RuntimeError("Only-20k checkpoint rule mismatch")
    for key in ("A_smoke", "A_full"):
        if configs[key]["dataset"]["repo_id"] != "local/task1_picklift_formal24_s03_20260728":
            raise RuntimeError("Condition A dataset mismatch")
        if configs[key]["dataset"]["domain_balanced_episode_groups"] is not None:
            raise RuntimeError("Condition A unexpectedly domain-balanced")
    expected_groups = {"real": list(range(24)), "simulation": list(range(24, 72))}
    for key in ("B_smoke", "B_full"):
        if configs[key]["dataset"]["domain_balanced_episode_groups"] != expected_groups:
            raise RuntimeError("Condition B is not exact 32+32 source partition")
    policy_hashes = {json.dumps(config["policy"], sort_keys=True) for config in configs.values()}
    if len(policy_hashes) != 1:
        raise RuntimeError("Policy recipes differ across the four runs")


def verify_phase_a_inputs() -> dict:
    plan = verify_phase_a.load_json(verify_phase_a.PLAN_PATH)
    result = verify_phase_a.load_json(verify_phase_a.RESULT_PATH)
    verify_phase_a.verify_static_contract(plan, result)
    verify_phase_a.verify_dependencies(result)
    verify_phase_a.verify_file_inventory(Path(result["official_base"]["cache_path"]), result["official_base"]["files"])
    verify_phase_a.verify_file_inventory(Path(result["transitive_vlm"]["cache_path"]), result["transitive_vlm"]["files"])
    real, real_identity = verify_phase_a.verify_dataset(plan["conditions"]["A_real24_only"], include_size=False)
    combined, combined_identity = verify_phase_a.verify_dataset(plan["conditions"]["B_real24_plus_questsim48_v7"], include_size=True)
    sampler = verify_phase_a.verify_sampler(combined, plan)
    if list(real.meta.tasks.index) != [TASK_TEXT] or list(combined.meta.tasks.index) != [TASK_TEXT]:
        raise RuntimeError("Exact training task text mismatch")
    real_sampler_kwargs = {
        "dataset_from_indices": real.meta.episodes["dataset_from_index"],
        "dataset_to_indices": real.meta.episodes["dataset_to_index"],
        "episode_indices_to_use": real.episodes,
        "shuffle": True,
        "seed": 1000,
        "absolute_to_relative_idx": real.absolute_to_relative_idx,
    }
    real_epoch = list(FullBatchEpisodeAwareSampler(**real_sampler_kwargs))
    real_repeat = list(FullBatchEpisodeAwareSampler(**real_sampler_kwargs))
    if len(real_epoch) != 3776 or real_epoch != real_repeat or len(set(real_epoch)) != len(real_epoch):
        raise RuntimeError("Condition A is not deterministic 59 x 64 full-batch sampling")
    return {
        "real24": real_identity,
        "combined72": combined_identity,
        "A_sampler": {
            "complete_batches": 59,
            "samples": 3776,
            "batch_size": 64,
            "duplicates": 0,
            "deterministic_repeat_equal": True,
        },
        "B_sampler": sampler,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-existing-outputs", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verify_research_inputs()
    binding = verify_binding()
    configs = {
        key: verify_config(name, require_absent_output=not args.allow_existing_outputs)
        for key, name in CONFIGS.items()
    }
    verify_config_pair(configs)
    datasets = verify_phase_a_inputs()
    result = {
        "schema": "task1_smolvla_phase_b_pre_optimizer_gate_v1",
        "status": "pass",
        "research_commit": "680e1e2f1b9c007f6d75c07e3a0b758fc94b3f67",
        "research_files": RESEARCH_FILES,
        "binding": binding,
        "configs": {name: sha256_file(EXPERIMENT_ROOT / filename) for name, filename in CONFIGS.items()},
        "datasets": datasets,
        "training_entrypoint": {
            "path": str(EXPERIMENT_ROOT / "train_phase_b.py"),
            "sha256": sha256_file(EXPERIMENT_ROOT / "train_phase_b.py"),
            "purpose": "Ensure Condition A uses only complete deterministic batches of 64 without changing frozen LeRobot core files.",
        },
        "training_steps": 0,
        "optimizer_steps": 0,
        "hardware_accessed": False,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
