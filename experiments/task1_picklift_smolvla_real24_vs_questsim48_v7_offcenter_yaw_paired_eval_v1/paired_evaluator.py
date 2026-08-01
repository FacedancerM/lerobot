from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
PLAN_PATH = EXPERIMENT_DIR / "evaluation_plan.json"
PROFILE_PATH = EXPERIMENT_DIR / "real_evaluation_profile_return3s_v1.json"
INDEX_PATH = EXPERIMENT_DIR / "software_freeze_index.json"
RESULT_MANIFEST = EXPERIMENT_DIR / "research_result_manifest_snapshot.json"
BASE_PATH = REPO_ROOT / "experiments/task1_picklift_real24_vs_questsim48_v7_offcenter_yaw_paired_eval_replication_v1/paired_evaluator.py"
MODEL_IDS = ("A_real24_only", "B_real24_plus_questsim48_v7")
TASK_TEXT = "Task 1 PickLift v1: grasp the 2 cm red cube and lift it >=5 cm with bilateral finger hold"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base():
    spec = importlib.util.spec_from_file_location("frozen_evalv2_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen Eval-v2 base")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_IDS = MODEL_IDS
    return module


BASE = load_base()


def load_frozen_plan() -> dict:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    for name, path in (("plan_sha256", PLAN_PATH), ("profile_sha256", PROFILE_PATH), ("engine_sha256", EXPERIMENT_DIR / "smolvla_official_send_engine.py"), ("research_manifest_sha256", RESULT_MANIFEST)):
        if sha256_file(path) != index[name]:
            raise RuntimeError(f"frozen {name} mismatch")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if plan["evaluation_id"] != "task1_picklift_smolvla_real24_vs_questsim48_v7_offcenter_yaw_paired_eval24_v1":
        raise RuntimeError("unexpected evaluation identity")
    if plan["runtime_prompt"]["text"] != TASK_TEXT or not plan["runtime_prompt"]["identical_for_both_models"]:
        raise RuntimeError("frozen SmolVLA prompt changed")
    setup = plan["setup"]
    if setup["control_fps"] != 20 or setup["maximum_trial_seconds"] != 30 or setup["stop_on_success"] is not False:
        raise RuntimeError("frozen timing changed")
    if setup["policy_chunk_size"] != 50 or setup["policy_n_action_steps"] != 50:
        raise RuntimeError("SmolVLA action horizon changed")
    if setup["max_relative_target"] is not None or setup["custom_absolute_action_clamp"] is not False or setup["custom_relative_step_limit_degrees"] is not None:
        raise RuntimeError("action path gained a custom limiter")
    if any(plan["authorization"][key] for key in ("hardware_authorized", "serial_accessed_during_preparation", "camera_accessed_during_preparation", "robot_accessed_during_preparation", "torque_accessed_during_preparation", "rollout_executed_during_preparation")):
        raise RuntimeError("software freeze claims hardware access")
    trials = plan["trials"]
    if len(trials) != 24 or [row["order"] for row in trials] != list(range(1, 25)):
        raise RuntimeError("paired order changed")
    expected = []
    for pose in range(1, 13):
        expected.extend((MODEL_IDS[1], MODEL_IDS[0]) if pose % 2 else MODEL_IDS)
    if [row["model_id"] for row in trials] != expected:
        raise RuntimeError("alternating first-model order changed")
    source = json.loads((BASE_PATH.parent / "evaluation_plan.json").read_text(encoding="utf-8"))
    fields = ("source_pose_order", "source_pose_id", "source_order_sha256", "cell_id", "quadrant", "nominal_x_forward_m", "nominal_y_lateral_m", "nominal_yaw_degrees_modulo_90", "operator_placement_prompt_zh")
    for pair_index, (left, right) in enumerate(zip(trials[::2], trials[1::2], strict=True)):
        source_pose = source["trials"][pair_index * 2]
        if any(left[key] != right[key] or left[key] != source_pose[key] for key in fields):
            raise RuntimeError("pose differs from frozen Eval-v2 source")
    return plan


def verify_static_files(plan: dict) -> dict:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    engine_path = EXPERIMENT_DIR / "smolvla_official_send_engine.py"
    engine_source = engine_path.read_text(encoding="utf-8")
    required = ("max_relative_target=None", "requested_action = raw_action.copy()", "robot.send_action(action_dict(requested_action))", "steps = run_paced_ticks", "SmolVLAPolicy.from_pretrained", TASK_TEXT)
    if any(fragment not in engine_source for fragment in required):
        raise RuntimeError("SmolVLA official-send adapter drifted")
    models = {}
    expected_inputs = {"observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]}, "observation.state": {"type": "STATE", "shape": [6]}}
    expected_outputs = {"action": {"type": "ACTION", "shape": [6]}}
    for model_id, model in plan["models"].items():
        checkpoint = Path(model["checkpoint"])
        paths = {
            "model_sha256": checkpoint / "model.safetensors",
            "config_sha256": checkpoint / "config.json",
            "train_config_sha256": checkpoint / "train_config.json",
            "policy_preprocessor_sha256": checkpoint / "policy_preprocessor.json",
            "policy_postprocessor_sha256": checkpoint / "policy_postprocessor.json",
            "normalizer_stats_sha256": checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors",
            "unnormalizer_stats_sha256": checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        }
        for key, path in paths.items():
            if sha256_file(path) != model[key]:
                raise RuntimeError(f"{model_id} {key} mismatch")
        config = json.loads(paths["config_sha256"].read_text(encoding="utf-8"))
        processor = json.loads(paths["policy_preprocessor_sha256"].read_text(encoding="utf-8"))
        if config["type"] != "smolvla" or config["chunk_size"] != 50 or config["n_action_steps"] != 50:
            raise RuntimeError(f"{model_id} SmolVLA config mismatch")
        if config["input_features"] != expected_inputs or config["output_features"] != expected_outputs:
            raise RuntimeError(f"{model_id} feature contract mismatch")
        tokenizer = [step["config"]["tokenizer_name"] for step in processor["steps"] if step["registry_name"] == "tokenizer_processor"]
        if tokenizer != [config["vlm_model_name"]]:
            raise RuntimeError(f"{model_id} tokenizer/VLM binding mismatch")
        if config["normalization_mapping"] != {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}:
            raise RuntimeError(f"{model_id} normalization changed")
        models[model_id] = {"checkpoint": str(checkpoint), "model_sha256": model["model_sha256"], "chunk_size": 50, "n_action_steps": 50, "input_features": config["input_features"], "output_features": config["output_features"], "task_prompt": TASK_TEXT}
    return {"plan_sha256": index["plan_sha256"], "profile_sha256": index["profile_sha256"], "engine_sha256": index["engine_sha256"], "official_send_contract": {"max_relative_target_none": True, "runner_absolute_clamp": False, "runner_step_limiter": None, "no_catch_up_pacing": True, "ready_return_trajectory": "linear_3s_20hz_then_hold_target"}, "models": models}


def load_engine(plan: dict):
    path = EXPERIMENT_DIR / "smolvla_official_send_engine.py"
    deployment_safety_dir = REPO_ROOT / "experiments/task1_picklift_real24_act_v1"
    if str(deployment_safety_dir) not in sys.path:
        sys.path.insert(0, str(deployment_safety_dir))
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("smolvla_official_send_engine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load SmolVLA official-send engine")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    module.EXPECTED_PLAN_SHA256 = index["plan_sha256"]
    module.EXPECTED_PROFILE_SHA256 = index["profile_sha256"]
    module.EXPECTED_EVALUATION_ID = plan["evaluation_id"]
    return module


def checkpoint_adapter_smoke(plan: dict) -> dict:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    dataset = LeRobotDataset("local/task1_picklift_formal24_s03_20260728", root=Path("/home/ubuntu24/Teleop/artifacts/task1_picklift_formal24_s03_20260728"), video_backend="pyav")
    sample = dataset[0]
    results = {}
    for model_id, record in plan["models"].items():
        checkpoint = Path(record["checkpoint"])
        policy = SmolVLAPolicy.from_pretrained(checkpoint).to("cuda").eval()
        pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(checkpoint), preprocessor_overrides={"device_processor": {"device": "cuda"}})
        policy.reset()
        inputs = {"observation.state": sample["observation.state"].unsqueeze(0), "observation.images.front": sample["observation.images.front"].unsqueeze(0), "task": [TASK_TEXT]}
        with torch.inference_mode():
            action = post(policy.select_action(pre(inputs))).detach().cpu().numpy()
        results[model_id] = {"shape": list(action.shape), "finite": bool(np.isfinite(action).all()), "model_sha256": sha256_file(checkpoint / "model.safetensors"), "task_prompt": TASK_TEXT}
        del policy, pre, post
        torch.cuda.empty_cache()
    if any(row["shape"] != [1, 6] or not row["finite"] for row in results.values()):
        raise RuntimeError("SmolVLA adapter smoke failed")
    return results


def software_dry_run(plan: dict) -> dict:
    static = verify_static_files(plan)
    fake = BASE.run_fake_protocol(plan)
    ready = BASE.run_fake_interpolated_ready_probe(plan)
    adapter = checkpoint_adapter_smoke(plan)
    if fake["trials_exercised"] != 24 or fake["policy_reset_calls"] != {MODEL_IDS[0]: 12, MODEL_IDS[1]: 12}:
        raise RuntimeError("fake paired protocol failed")
    if not all((fake["all_ready_before_policy"], fake["all_ready_after_trial"], fake["all_canonical_rgb_640x480"], fake["all_pre_action_frames_before_policy_send"], fake["all_official_sent_equals_requested"], fake["all_torque_disabled"], ready["commands_sent"] == 60, ready["all_official_sent_equals_requested"])):
        raise RuntimeError("fake Eval-v2 contract failed")
    return {"schema_version": 1, "evaluation_id": plan["evaluation_id"], "created_at_utc": datetime.now(UTC).isoformat(), "status": "software_dry_run_passed_hardware_not_accessed", "hardware_access": {"serial": False, "camera": False, "robot": False, "torque": False, "rollout": False}, "static_verification": static, "fake_protocol": fake, "fake_interpolated_ready_return": ready, "checkpoint_adapter_smoke": adapter, "next_gate": "Stop before the user turns on Follower 12 V."}


def write_software_evidence(plan: dict, dry_run: dict) -> dict:
    root = Path(plan["evidence_root"]) / "software_preparation_v1"
    if root.exists():
        raise RuntimeError(f"refusing to overwrite frozen evidence: {root}")
    root.mkdir(parents=True)
    copies = {"evaluation_plan.json": PLAN_PATH, "software_freeze_index.json": INDEX_PATH, "research_result_manifest_snapshot.json": RESULT_MANIFEST, "real_evaluation_profile_return3s_v1.json": PROFILE_PATH, "operator_annotation_schema_v1.json": EXPERIMENT_DIR / "operator_annotation_schema_v1.json", "canonical_video_review_schema_v1.json": EXPERIMENT_DIR / "canonical_video_review_schema_v1.json"}
    for name, source in copies.items():
        shutil.copyfile(source, root / name)
    (root / "dry_run.json").write_text(json.dumps(dry_run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    inventory = {path.name: sha256_file(path) for path in sorted(root.iterdir())}
    (root / "hashes.sha256").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(inventory.items())), encoding="utf-8")
    return {"root": str(root), "files": inventory, "hashes_sha256": sha256_file(root / "hashes.sha256")}


def execute_hardware(args: argparse.Namespace, plan: dict) -> None:
    if not args.operator_confirmed_ready or args.trial_id is None:
        raise RuntimeError("hardware execution requires the single-trial operator gate")
    verify_static_files(plan)
    trial = BASE.find_trial(plan, args.trial_id)
    stem, replacement_for = BASE.validate_execution_order(plan, trial, replacement=args.replacement)
    engine = load_engine(plan)
    model = plan["models"][trial["model_id"]]
    engine.EXPECTED_MODEL_SHA256 = model["model_sha256"]
    engine_args = argparse.Namespace(execute_hardware=True, operator_confirmed_ready=True, spawn_region=trial["spawn_region"], follower_port=args.follower_port, camera_device=args.camera_device, checkpoint=Path(model["checkpoint"]), calibration=args.calibration, plan=PLAN_PATH, profile=PROFILE_PATH, evidence_dir=Path(plan["evidence_root"]) / "trials", maximum_trial_seconds=30.0)
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    engine.READY_MOVE_TOLERANCE = 3.0
    engine.READY_MOVE_PROFILE_ID = profile["ready_pose_movement"]["profile_id"]
    engine.move_to_frozen_ready_pose = BASE.build_interpolated_ready_move(engine, profile)
    preflight = engine.preflight(engine_args)
    preflight.update({"paired_evalv2_evaluation_id": plan["evaluation_id"], "paired_evalv2_trial": trial, "replacement_for": replacement_for, "success_contract": plan["success_contract"], "operator_label": {"status": "pending"}, "canonical_video_review_label": {"status": "pending"}})
    engine_args.spawn_region = stem
    try:
        engine.run_hardware_trial(engine_args, preflight)
    finally:
        BASE.write_evalv2_sidecar(plan, trial, stem, replacement_for)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--software-dry-run", action="store_true")
    mode.add_argument("--execute-hardware", action="store_true")
    parser.add_argument("--freeze-software-evidence", action="store_true")
    parser.add_argument("--trial-id")
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--operator-confirmed-ready", action="store_true")
    parser.add_argument("--follower-port", default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110904-if00")
    parser.add_argument("--camera-device", default="/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0")
    parser.add_argument("--calibration", type=Path, default=Path("/home/ubuntu24/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower_main.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_frozen_plan()
    if args.software_dry_run:
        if args.operator_confirmed_ready or args.trial_id or args.replacement:
            raise RuntimeError("hardware-only arguments are invalid in dry-run mode")
        result = software_dry_run(plan)
        if args.freeze_software_evidence:
            result["frozen_evidence"] = write_software_evidence(plan, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        print("DRY RUN ONLY: no serial, camera, robot, torque, 12 V, or rollout was accessed.")
        return
    if args.freeze_software_evidence:
        raise RuntimeError("freeze is dry-run only")
    execute_hardware(args, plan)


if __name__ == "__main__":
    main()
