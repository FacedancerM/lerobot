from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from prepare_phase_b import BINDING_ROOT, EXPERIMENT_ROOT, VLM_ROOT
from verify_phase_b_contract import CONFIGS, sha256_file, verify_binding, verify_config, verify_config_pair


def stats_sha(stats: dict) -> str:
    digest = hashlib.sha256()
    for feature in sorted(stats):
        for name in sorted(stats[feature]):
            tensor = torch.as_tensor(stats[feature][name]).detach().cpu().contiguous()
            digest.update(feature.encode())
            digest.update(b"\0")
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(b"\0")
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(b"\0")
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def processor_gate(config: SmolVLAConfig, repo_id: str, root: Path) -> dict:
    dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    features = {**config.input_features, **config.output_features}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(BINDING_ROOT),
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": features,
                "norm_map": config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": config.output_features,
                "norm_map": config.normalization_mapping,
            }
        },
    )
    normalizer = next(step for step in preprocessor.steps if step.__class__.__name__ == "NormalizerProcessorStep")
    unnormalizer = next(step for step in postprocessor.steps if step.__class__.__name__ == "UnnormalizerProcessorStep")
    tokenizer = next(step for step in preprocessor.steps if step.__class__.__name__ == "TokenizerProcessorStep")
    if set(normalizer.features) != {"observation.state", "observation.images.front", "action"}:
        raise RuntimeError("Actual loaded normalizer is not state/front/action only")
    if set(unnormalizer.features) != {"action"}:
        raise RuntimeError("Actual loaded postprocessor is not action only")
    if tokenizer.tokenizer_name != str(VLM_ROOT):
        raise RuntimeError("Actual loaded tokenizer path drifted")
    for key in ("observation.state", "action"):
        for stat_name, expected in dataset.meta.stats[key].items():
            actual = normalizer.stats[key][stat_name]
            if not torch.equal(torch.as_tensor(actual), torch.as_tensor(expected)):
                raise RuntimeError(f"Actual loaded normalizer stats drifted: {key}.{stat_name}")
    for stat_name, expected in dataset.meta.stats["action"].items():
        actual = unnormalizer.stats["action"][stat_name]
        if not torch.equal(torch.as_tensor(actual), torch.as_tensor(expected)):
            raise RuntimeError(f"Actual loaded postprocessor stats drifted: action.{stat_name}")
    return {
        "repo_id": repo_id,
        "episodes": dataset.num_episodes,
        "frames": dataset.num_frames,
        "normalizer_features": sorted(normalizer.features),
        "postprocessor_features": sorted(unnormalizer.features),
        "tokenizer_name": tokenizer.tokenizer_name,
        "dataset_stats_sha256": stats_sha(dataset.meta.stats),
        "normalizer_stats_sha256": stats_sha(normalizer.stats),
        "postprocessor_action_stats_sha256": stats_sha({"action": unnormalizer.stats["action"]}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_binding()
    configs = {key: verify_config(name, require_absent_output=True) for key, name in CONFIGS.items()}
    verify_config_pair(configs)
    parsed = {}
    for key, filename in CONFIGS.items():
        cfg = TrainPipelineConfig.from_pretrained(EXPERIMENT_ROOT / filename)
        if cfg.batch_size != 64 or cfg.seed != 1000 or cfg.policy.pretrained_path != BINDING_ROOT:
            raise RuntimeError(f"Materialized TrainPipelineConfig round-trip mismatch: {key}")
        parsed[key] = {
            "config_sha256": sha256_file(EXPERIMENT_ROOT / filename),
            "output_dir": str(cfg.output_dir),
            "steps": cfg.steps,
            "batch_size": cfg.batch_size,
            "seed": cfg.seed,
        }

    model_config = PreTrainedConfig.from_pretrained(BINDING_ROOT)
    if not isinstance(model_config, SmolVLAConfig):
        raise RuntimeError(f"Derived binding resolved to {type(model_config).__name__}, not SmolVLAConfig")
    if model_config.input_features != {
        "observation.images.front": model_config.input_features["observation.images.front"],
        "observation.state": model_config.input_features["observation.state"],
    }:
        raise RuntimeError("Constructed model feature contract drifted")
    if model_config.empty_cameras != 0 or model_config.adapt_to_pi_aloha or model_config.use_delta_joint_actions_aloha:
        raise RuntimeError("Constructed model unexpectedly enables camera/action adaptation")
    if model_config.vlm_model_name != str(VLM_ROOT):
        raise RuntimeError("Constructed model VLM path drifted")
    policy = SmolVLAPolicy(model_config)
    state = load_file(BINDING_ROOT / "model.safetensors", device="cpu")
    incompatible = policy.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Base state mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    parameter_counts = {
        "total": sum(parameter.numel() for parameter in policy.parameters()),
        "trainable": sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad),
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
    }
    del state, policy

    processor_a = processor_gate(
        model_config,
        "local/task1_picklift_formal24_s03_20260728",
        Path("/home/ubuntu24/Teleop/artifacts/task1_picklift_formal24_s03_20260728"),
    )
    processor_b = processor_gate(
        model_config,
        "local/task1_picklift_real24_questsim48_v7_combined72_v1",
        Path(
            "/home/ubuntu24/Teleop/artifacts/datasets/"
            "task1_picklift_real24_questsim48_v7_act_v1/combined72_v1"
        ),
    )
    result = {
        "schema": "task1_smolvla_phase_b_deep_pre_optimizer_gate_v1",
        "status": "pass",
        "configs_round_trip": parsed,
        "model_load": parameter_counts,
        "condition_A_processor": processor_a,
        "condition_B_processor": processor_b,
        "cross_condition_stats_equal": processor_a["dataset_stats_sha256"] == processor_b["dataset_stats_sha256"],
        "expected_cross_condition_stats_equal": False,
        "optimizer_steps": 0,
        "hardware_accessed": False,
    }
    if result["cross_condition_stats_equal"]:
        raise RuntimeError("A and B unexpectedly share one stats payload")
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
