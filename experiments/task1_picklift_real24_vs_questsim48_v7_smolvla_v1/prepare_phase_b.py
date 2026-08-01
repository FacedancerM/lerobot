from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
ARTIFACT_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/training/"
    "task1_picklift_real24_vs_questsim48_v7_smolvla_v1"
)
EVIDENCE_ROOT = Path(
    "/home/ubuntu24/Teleop/artifacts/evidence/"
    "task1_picklift_real24_vs_questsim48_v7_smolvla_v1/phase_b_v1"
)
BASE_ROOT = Path(
    "/home/ubuntu24/.cache/huggingface/hub/models--lerobot--smolvla_base/"
    "snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
)
VLM_ROOT = Path(
    "/home/ubuntu24/.cache/huggingface/hub/"
    "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/"
    "snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
)
BINDING_ROOT = EVIDENCE_ROOT / "exact_base_binding_v1"
TASK_TEXT = (
    "Task 1 PickLift v1: grasp the 2 cm red cube and lift it >=5 cm "
    "with bilateral finger hold"
)

BASE_FILES = {
    "README.md": "adb98360b162fe40b6f7995d21c1a4250bd51ddd094973234e290482a35ea4ce",
    "config.json": "650584b56c104720f7a3c91d1ec6bec9e8de8ac11e60c92ba2fa82d93eda147d",
    "model.safetensors": "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb",
    "policy_postprocessor.json": "2b78bb742065288df2ec63b0ee35f97a1f6950171cf2272f7e34b1fe3873b17b",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": (
        "490ab239d96e263687c0b2e386a0afbc235a2eceb9857c36ed32f2f162a3e7c8"
    ),
    "policy_preprocessor.json": "7683d648280a72f0d51e869fb8dd1596beed90b7f84849330cf1bc26634a4bd6",
    "policy_preprocessor_step_5_normalizer_processor.safetensors": (
        "490ab239d96e263687c0b2e386a0afbc235a2eceb9857c36ed32f2f162a3e7c8"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Refusing to overwrite different frozen file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def exact_features() -> tuple[dict, dict]:
    inputs = {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
    }
    outputs = {"action": {"type": "ACTION", "shape": [6]}}
    return inputs, outputs


def verify_source_base() -> None:
    for relative, expected in BASE_FILES.items():
        path = BASE_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen base mismatch: {path}")
    vlm_model = VLM_ROOT / "model.safetensors"
    if sha256_file(vlm_model) != "b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3":
        raise RuntimeError("Frozen VLM model mismatch")


def materialize_binding() -> dict:
    if BINDING_ROOT.exists():
        manifest = json.loads((BINDING_ROOT / "binding_manifest.json").read_text())
        for relative, expected in manifest["derived_files"].items():
            if sha256_file(BINDING_ROOT / relative) != expected["sha256"]:
                raise RuntimeError(f"Existing binding mismatch: {relative}")
        return manifest

    BINDING_ROOT.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="exact_base_binding_v1.tmp-", dir=BINDING_ROOT.parent))
    try:
        shutil.copy2(BASE_ROOT / "README.md", temp_root / "README.md")
        for filename in (
            "model.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
        ):
            os.link((BASE_ROOT / filename).resolve(), temp_root / filename)

        inputs, outputs = exact_features()
        policy_config = json.loads((BASE_ROOT / "config.json").read_text())
        policy_config["input_features"] = inputs
        policy_config["output_features"] = outputs
        policy_config["vlm_model_name"] = str(VLM_ROOT)
        policy_config["push_to_hub"] = False
        policy_config["repo_id"] = None
        (temp_root / "config.json").write_text(
            json.dumps(policy_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        preprocessor = json.loads((BASE_ROOT / "policy_preprocessor.json").read_text())
        for step in preprocessor["steps"]:
            if step["registry_name"] == "tokenizer_processor":
                step["config"]["tokenizer_name"] = str(VLM_ROOT)
            if step["registry_name"] == "normalizer_processor":
                step["config"]["features"] = {**inputs, **outputs}
        (temp_root / "policy_preprocessor.json").write_text(
            json.dumps(preprocessor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        shutil.copy2(BASE_ROOT / "policy_postprocessor.json", temp_root / "policy_postprocessor.json")
        derived_files = {}
        for path in sorted(temp_root.iterdir()):
            if path.is_file():
                derived_files[path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        manifest = {
            "schema": "task1_smolvla_exact_local_base_binding_v1",
            "policy_source": {
                "repo": "lerobot/smolvla_base",
                "revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
                "snapshot": str(BASE_ROOT),
                "model_sha256": BASE_FILES["model.safetensors"],
            },
            "vlm_source": {
                "repo": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                "revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
                "snapshot": str(VLM_ROOT),
                "model_sha256": "b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3",
            },
            "binding": {
                "policy_vlm_model_name": str(VLM_ROOT),
                "preprocessor_tokenizer_name": str(VLM_ROOT),
                "input_features": inputs,
                "output_features": outputs,
                "base_model_weights_modified": False,
            },
            "derived_files": derived_files,
        }
        (temp_root / "binding_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temp_root.rename(BINDING_ROOT)
        return manifest
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def dataset_config(condition: str) -> dict:
    common = {
        "image_transforms": {
            "enable": False,
            "max_num_transforms": 3,
            "random_order": False,
            "tfs": {},
        },
        "revision": None,
        "use_imagenet_stats": True,
        "video_backend": "pyav",
        "return_uint8": False,
        "depth_output_unit": "m",
        "streaming": False,
        "eval_split": 0.0,
    }
    if condition == "A":
        return {
            **common,
            "repo_id": "local/task1_picklift_formal24_s03_20260728",
            "root": "/home/ubuntu24/Teleop/artifacts/task1_picklift_formal24_s03_20260728",
            "episodes": list(range(24)),
            "domain_balanced_episode_groups": None,
        }
    return {
        **common,
        "repo_id": "local/task1_picklift_real24_questsim48_v7_combined72_v1",
        "root": (
            "/home/ubuntu24/Teleop/artifacts/datasets/"
            "task1_picklift_real24_questsim48_v7_act_v1/combined72_v1"
        ),
        "episodes": list(range(72)),
        "domain_balanced_episode_groups": {
            "real": list(range(24)),
            "simulation": list(range(24, 72)),
        },
    }


def training_config(condition: str, stage: str) -> dict:
    steps = 500 if stage == "smoke" else 20000
    condition_id = "A_real24_only" if condition == "A" else "B_real24_plus_questsim48_v7"
    run_name = f"{condition_id}_{'smoke_500' if stage == 'smoke' else 'full_20000'}"
    policy = json.loads((BINDING_ROOT / "config.json").read_text())
    policy["pretrained_path"] = str(BINDING_ROOT)
    policy["pretrained_revision"] = None
    return {
        "dataset": dataset_config(condition),
        "env": None,
        "policy": policy,
        "reward_model": None,
        "output_dir": str(ARTIFACT_ROOT / run_name),
        "job_name": f"task1_smolvla_{run_name}",
        "resume": False,
        "seed": 1000,
        "cudnn_deterministic": False,
        "num_workers": 4,
        "batch_size": 64,
        "prefetch_factor": 4,
        "persistent_workers": True,
        "steps": steps,
        "env_eval_freq": 0,
        "log_freq": 20 if stage == "smoke" else 200,
        "eval_steps": 0,
        "max_eval_samples": 0,
        "tolerance_s": 0.0001,
        "save_checkpoint": True,
        "save_freq": steps,
        "use_policy_training_preset": True,
        "optimizer": None,
        "scheduler": None,
        "wandb": {
            "enable": False,
            "disable_artifact": False,
            "project": "lerobot",
            "entity": None,
            "notes": None,
            "run_id": None,
            "mode": None,
            "add_tags": True,
        },
        "peft": None,
        "job": {
            "target": None,
            "image": "huggingface/lerobot-gpu:latest",
            "timeout": "2d",
            "detach": False,
            "tags": [],
        },
        "save_checkpoint_to_hub": False,
        "sample_weighting": None,
        "rename_map": {},
    }


def main() -> None:
    verify_source_base()
    manifest = materialize_binding()
    configs = {}
    for condition in ("A", "B"):
        for stage in ("smoke", "full"):
            name = f"train_{condition}_{stage}_{500 if stage == 'smoke' else 20000}.json"
            path = EXPERIMENT_ROOT / name
            write_json(path, training_config(condition, stage))
            configs[name] = sha256_file(path)
    result = {
        "schema": "task1_smolvla_phase_b_preparation_result_v1",
        "status": "prepared_no_optimizer_step",
        "research_commit": "680e1e2f1b9c007f6d75c07e3a0b758fc94b3f67",
        "authorization_token": "GO_TASK1_SMOLVLA_MATCHED_PAIR_PHASE_B_V1",
        "binding_root": str(BINDING_ROOT),
        "binding_manifest_sha256": sha256_file(BINDING_ROOT / "binding_manifest.json"),
        "binding": manifest["binding"],
        "train_configs": configs,
        "training_steps": 0,
        "optimizer_steps": 0,
        "hardware_accessed": False,
    }
    write_json(EVIDENCE_ROOT / "preparation_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

