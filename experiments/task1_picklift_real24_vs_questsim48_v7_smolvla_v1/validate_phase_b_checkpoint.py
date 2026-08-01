from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from prepare_phase_b import BINDING_ROOT, TASK_TEXT, VLM_ROOT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("A", "B"), required=True)
    parser.add_argument("--steps", type=int, choices=(500, 20000), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.run_root / "checkpoints" / f"{args.steps:06d}" / "pretrained_model"
    if not checkpoint.is_dir():
        raise RuntimeError(f"Missing fixed checkpoint: {checkpoint}")
    config = json.loads((checkpoint / "config.json").read_text())
    if set(config["input_features"]) != {"observation.state", "observation.images.front"}:
        raise RuntimeError("Saved policy is not front-only")
    if config["vlm_model_name"] != str(VLM_ROOT):
        raise RuntimeError("Saved policy VLM binding drifted")
    preprocessor_config = json.loads((checkpoint / "policy_preprocessor.json").read_text())
    tokenizer_names = [
        step["config"]["tokenizer_name"]
        for step in preprocessor_config["steps"]
        if step["registry_name"] == "tokenizer_processor"
    ]
    if tokenizer_names != [str(VLM_ROOT)]:
        raise RuntimeError("Saved tokenizer binding drifted")
    serialized = json.dumps({"config": config, "preprocessor": preprocessor_config})
    if any(token in serialized for token in ("camera1", "camera2", "camera3", "observation.image2", "observation.image3", "wrist", "handeye")):
        raise RuntimeError("Saved policy contains forbidden feature")

    if args.condition == "A":
        repo_id = "local/task1_picklift_formal24_s03_20260728"
        dataset_root = Path("/home/ubuntu24/Teleop/artifacts/task1_picklift_formal24_s03_20260728")
        sample_indices = [0]
    else:
        repo_id = "local/task1_picklift_real24_questsim48_v7_combined72_v1"
        dataset_root = Path(
            "/home/ubuntu24/Teleop/artifacts/datasets/"
            "task1_picklift_real24_questsim48_v7_act_v1/combined72_v1"
        )
        sample_indices = [0, 3790]

    dataset = LeRobotDataset(repo_id, root=dataset_root, video_backend="pyav")
    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.to("cuda")
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    samples = []
    for sample_index in sample_indices:
        sample = dataset[sample_index]
        inputs = {
            "observation.state": sample["observation.state"].unsqueeze(0),
            "observation.images.front": sample["observation.images.front"].unsqueeze(0),
            "task": [TASK_TEXT],
        }
        policy.reset()
        with torch.inference_mode():
            action = postprocessor(policy.select_action(preprocessor(inputs)))
        values = action.detach().cpu().numpy()
        if values.shape != (1, 6) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid offline action for sample {sample_index}: {values.shape}")
        episode_index = int(sample["episode_index"])
        samples.append(
            {
                "sample_index": sample_index,
                "episode_index": episode_index,
                "domain": "real" if episode_index < 24 else "simulation",
                "input_shapes": {
                    "state": list(inputs["observation.state"].shape),
                    "front": list(inputs["observation.images.front"].shape),
                },
                "action_shape": list(values.shape),
                "action_finite": True,
                "action": values[0].tolist(),
            }
        )

    result = {
        "schema": "task1_smolvla_phase_b_offline_checkpoint_validation_v1",
        "status": "pass",
        "condition": args.condition,
        "steps": args.steps,
        "checkpoint": str(checkpoint),
        "model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "config_sha256": sha256_file(checkpoint / "config.json"),
        "train_config_sha256": sha256_file(checkpoint / "train_config.json"),
        "preprocessor_sha256": sha256_file(checkpoint / "policy_preprocessor.json"),
        "postprocessor_sha256": sha256_file(checkpoint / "policy_postprocessor.json"),
        "initialization_binding": str(BINDING_ROOT),
        "vlm_path": str(VLM_ROOT),
        "tokenizer_path": tokenizer_names[0],
        "samples": samples,
        "hardware_accessed": False,
        "rollout_started": False,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

