from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from lerobot.datasets import DomainBalancedSampler
from lerobot.datasets.lerobot_dataset import LeRobotDataset

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
PLAN_PATH = EXPERIMENT_ROOT / "candidate_plan.json"
RESULT_PATH = EXPERIMENT_ROOT / "phase_a_preflight_result.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_identity(root: Path, *, include_size: bool) -> dict[str, int | str]:
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for path in sorted((entry for entry in root.rglob("*") if entry.is_file()), key=lambda p: p.relative_to(root).as_posix()):
        file_hash = sha256_file(path)
        size = path.stat().st_size
        relative = path.relative_to(root).as_posix()
        row = f"{file_hash}  {size}  {relative}\n" if include_size else f"{file_hash}  {relative}\n"
        digest.update(row.encode("utf-8"))
        files += 1
        total_bytes += size
    return {"sha256": digest.hexdigest(), "files": files, "bytes": total_bytes}


def verify_file_inventory(root: Path, inventory: dict[str, dict]) -> None:
    for relative, expected in inventory.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"Missing frozen file: {path}")
        if path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"Size mismatch: {path}")
        if sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"SHA mismatch: {path}")


def verify_sample(sample: dict) -> None:
    expected = {
        "observation.state": (6,),
        "action": (6,),
        "observation.images.front": (3, 480, 640),
    }
    for key, shape in expected.items():
        tensor = sample[key]
        if tuple(tensor.shape) != shape:
            raise RuntimeError(f"{key} shape mismatch: {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{key} contains non-finite values")
    if sample["observation.state"].dtype != torch.float32 or sample["action"].dtype != torch.float32:
        raise RuntimeError("State/action dtype is not float32")


def verify_dataset(condition: dict, *, include_size: bool) -> tuple[LeRobotDataset, dict]:
    root = Path(condition["dataset_root"])
    identity = tree_identity(root, include_size=include_size)
    expected = {
        "sha256": condition["tree_sha256"],
        "files": 56 if not include_size else 104,
        "bytes": 123884807 if not include_size else 159543656,
    }
    if identity != expected:
        raise RuntimeError(f"Dataset identity mismatch: {identity} != {expected}")
    dataset = LeRobotDataset(condition["repo_id"], root=root, video_backend="pyav")
    if dataset.num_episodes != condition["episodes"] or dataset.num_frames != condition["frames"]:
        raise RuntimeError("Official loader episode/frame counts mismatch")
    expected_task = load_json(PLAN_PATH)["task_contract"]["task_text"]
    if list(dataset.meta.tasks.index) != [expected_task]:
        raise RuntimeError(f"Task text mismatch: {list(dataset.meta.tasks.index)}")
    verify_sample(dataset[0])
    verify_sample(dataset[len(dataset) - 1])
    return dataset, identity


def verify_sampler(dataset: LeRobotDataset, plan: dict) -> dict:
    sampler_contract = plan["conditions"]["B_real24_plus_questsim48_v7"]["sampler"]
    groups = {"real": list(range(24)), "simulation": list(range(24, 72))}
    kwargs = {
        "dataset_from_indices": dataset.meta.episodes["dataset_from_index"],
        "dataset_to_indices": dataset.meta.episodes["dataset_to_index"],
        "episode_indices": dataset.meta.episodes["episode_index"],
        "domain_episode_groups": groups,
        "batch_size": sampler_contract["batch_size"],
        "episode_indices_to_use": dataset.episodes,
        "seed": sampler_contract["seed"],
        "absolute_to_relative_idx": dataset.absolute_to_relative_idx,
    }
    sampler = DomainBalancedSampler(**kwargs)
    epoch = list(sampler)
    repeat = list(DomainBalancedSampler(**kwargs))
    if epoch != repeat:
        raise RuntimeError("Domain-balanced epoch is not deterministic")
    if len(epoch) != 7552 or len(set(epoch)) != len(epoch):
        raise RuntimeError("Domain-balanced sample count or uniqueness mismatch")
    for offset in range(0, len(epoch), 64):
        batch = epoch[offset : offset + 64]
        if len(batch) != 64 or sum(index < 3790 for index in batch) != 32:
            raise RuntimeError(f"Batch {offset // 64} is not exactly 32 Real + 32 Sim")
    return {
        "complete_batches": sampler.num_batches,
        "samples": len(epoch),
        "per_domain_per_batch": sampler.samples_per_domain_per_batch,
        "duplicates": len(epoch) - len(set(epoch)),
        "deterministic_repeat_equal": True,
    }


def verify_static_contract(plan: dict, result: dict) -> None:
    if plan["status"] != "phase_a_candidate_frozen_pending_research_contract_and_training_go":
        raise RuntimeError("Plan status changed")
    recipe = plan["common_recipe"]
    required = {"seed": 1000, "batch_size": 64, "smoke_steps": 500, "full_steps": 20000, "selected_checkpoint_step": 20000}
    if any(recipe[key] != value for key, value in required.items()):
        raise RuntimeError("Common matched recipe changed")
    order = [row["run"] for row in plan["predeclared_execution_order"]]
    if order != [
        "A_real24_only_smoke_500",
        "B_real24_plus_questsim48_v7_smoke_500",
        "A_real24_only_full_20000",
        "B_real24_plus_questsim48_v7_full_20000",
    ]:
        raise RuntimeError("Predeclared run order changed")
    if plan["phase_authorization"] != {
        "phase_a_preflight": True,
        "model_training": False,
        "hardware_or_rollout": False,
        "push": False,
    }:
        raise RuntimeError("Phase boundary changed")
    if result["negative_evidence"]["training_steps_executed"] != 0 or result["cuda_non_training_probe"]["optimizer_steps"] != 0:
        raise RuntimeError("Phase A evidence claims an optimizer/training step")


def verify_dependencies(result: dict) -> None:
    freeze = result["dependency_freeze"]
    if sha256_file(REPO_ROOT / "pyproject.toml") != freeze["pyproject_sha256"]:
        raise RuntimeError("pyproject.toml hash mismatch")
    if sha256_file(REPO_ROOT / "uv.lock") != freeze["uv_lock_sha256"]:
        raise RuntimeError("uv.lock hash mismatch")
    for relative, expected in freeze["source_files"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"Pinned implementation file changed: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Task1 SmolVLA Phase A frozen inputs")
    parser.add_argument("--skip-datasets", action="store_true")
    args = parser.parse_args()

    plan = load_json(PLAN_PATH)
    result = load_json(RESULT_PATH)
    verify_static_contract(plan, result)
    verify_dependencies(result)

    base = result["official_base"]
    verify_file_inventory(Path(base["cache_path"]), base["files"])
    vlm = result["transitive_vlm"]
    verify_file_inventory(Path(vlm["cache_path"]), vlm["files"])

    details: dict[str, object] = {
        "status": "pass",
        "plan_sha256": sha256_file(PLAN_PATH),
        "preflight_result_sha256": sha256_file(RESULT_PATH),
        "base_inventory": "pass",
        "vlm_inventory": "pass",
        "dependency_freeze": "pass",
    }
    if not args.skip_datasets:
        real, real_identity = verify_dataset(plan["conditions"]["A_real24_only"], include_size=False)
        combined, combined_identity = verify_dataset(plan["conditions"]["B_real24_plus_questsim48_v7"], include_size=True)
        details.update(
            {
                "real24_identity": real_identity,
                "combined72_identity": combined_identity,
                "sampler": verify_sampler(combined, plan),
                "official_loader": "pass",
                "task_text_equal": list(real.meta.tasks.index) == list(combined.meta.tasks.index),
            }
        )
    print(json.dumps(details, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

