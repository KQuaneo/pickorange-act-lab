"""Evaluate the joint-6 ACT PickOrange A0--A3 ablation groups."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from contextlib import nullcontext
from pathlib import Path

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--group", choices=("a0", "a1", "a2", "a3"), required=True)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--checkpoint_001", type=Path)
parser.add_argument("--checkpoint_002", type=Path)
parser.add_argument("--checkpoint_003", type=Path)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--action_horizon", type=int, default=100)
parser.add_argument("--policy_steps_per_orange", type=int, default=420)
parser.add_argument("--total_policy_steps", type=int, default=1020, help="A0 total policy actions; default preserves the formal protocol")
parser.add_argument(
    "--horizon_protocol",
    choices=("native_horizon", "matched_horizon"),
    default="native_horizon",
    help="native keeps A0=1020/A1=1260; matched sets A0=1260 and keeps A1=420x3",
)
parser.add_argument("--sim_steps_per_action", type=int, default=2)
parser.add_argument("--stable_plate_steps", type=int, default=10)
parser.add_argument("--dataset_label", default=None)
parser.add_argument("--strategy_label", default=None)
parser.add_argument("--checkpoint_label", default=None)
parser.add_argument("--stage_reference_json", type=Path, default=None, help="optional expert stage-start joint mean/std JSON")
parser.add_argument("--stage_state_jsonl", type=Path, default=None, help="raw stage-start records; defaults under output_dir")
parser.add_argument("--record_video", action="store_true")
parser.add_argument("--output_dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

import leisaac.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from leisaac.utils.robot_utils import (
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
)


TASK = "LeIsaac-SO101-PickOrange-v0"
ORANGES = ("Orange001", "Orange002", "Orange003")
TOTAL_POLICY_STEPS = 1020
SIM_STEPS_PER_ACTION = 2
FPS = 30
STAGE_RANGES = (
    (0, 250),
    (250, 340),
    (340, 590),
    (590, 680),
    (680, 930),
    (930, 1020),
)


def policy_image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 4:
        image = image[0]
    if image.shape[-1] in (3, 4):
        image = image[..., :3].permute(2, 0, 1)
    image = image.to(dtype=torch.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image.unsqueeze(0)


def video_image(image: torch.Tensor) -> np.ndarray:
    if image.ndim == 4:
        image = image[0]
    array = image.detach().cpu().numpy()
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    array = array[..., :3]
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def video_frame(observation: dict) -> np.ndarray:
    return np.concatenate(
        [video_image(observation["policy"]["front"]), video_image(observation["policy"]["wrist"])],
        axis=1,
    )


def stage_index(step: int) -> int:
    for index, (start, end) in enumerate(STAGE_RANGES):
        if start <= step < end:
            return index
    raise ValueError(f"No stage for policy step {step}")


def stage_end(step: int) -> int:
    return STAGE_RANGES[stage_index(step)][1]


def policy_observation(observation: dict, device: torch.device, stage: int | None = None) -> dict:
    state = np.asarray(convert_leisaac_action_to_lerobot(observation["policy"]["joint_pos"]), dtype=np.float32)
    if state.ndim == 1:
        state = state[None, :]
    if stage is not None:
        one_hot = np.zeros((state.shape[0], 6), dtype=np.float32)
        one_hot[:, stage] = 1.0
        state = np.concatenate([state, one_hot], axis=1)
    return {
        "observation.state": torch.as_tensor(state, device=device),
        "observation.images.front": policy_image(observation["policy"]["front"]).to(device),
        "observation.images.wrist": policy_image(observation["policy"]["wrist"]).to(device),
    }


def load_policy(checkpoint: Path, device: torch.device):
    checkpoint = checkpoint.resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(checkpoint)
    print(f"Loading {checkpoint}", flush=True)
    policy = ACTPolicy.from_pretrained(checkpoint).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor, checkpoint


def predict_actions(bundle, observation: dict, device: torch.device, count: int, stage: int | None) -> torch.Tensor:
    policy, preprocessor, postprocessor, _ = bundle
    model_input = policy_observation(observation, device, stage)
    amp_context = (lambda: torch.autocast(device_type="cuda")) if device.type == "cuda" else nullcontext
    with torch.inference_mode(), amp_context():
        raw = policy.predict_action_chunk(preprocessor(model_input))
    count = min(count, int(raw.shape[1]))
    motor_actions = torch.cat([postprocessor(raw[:, index, :]) for index in range(count)], dim=0).float()
    joint_actions = convert_lerobot_action_to_leisaac(motor_actions)
    return torch.as_tensor(joint_actions, dtype=torch.float32, device=device)


def orange_in_plate(env, orange: str) -> bool:
    delta = env.scene[orange].data.root_pos_w[0] - env.scene["Plate"].data.root_pos_w[0]
    return abs(float(delta[0])) < 0.10 and abs(float(delta[1])) < 0.10 and abs(float(delta[2])) < 0.07


def update_metrics(env, initial_z: dict[str, float], metrics: dict, stable: dict[str, int]) -> None:
    for orange in ORANGES:
        z = float(env.scene[orange].data.root_pos_w[0, 2].item())
        metrics[orange]["max_lift_m"] = max(metrics[orange]["max_lift_m"], z - initial_z[orange])
        if orange_in_plate(env, orange):
            stable[orange] += 1
        else:
            stable[orange] = 0
        if stable[orange] >= args.stable_plate_steps:
            metrics[orange]["ever_stably_placed"] = True


def step_action(env, robot, action: torch.Tensor):
    observation = None
    for _ in range(args.sim_steps_per_action):
        robot.write_joint_damping_to_sim(damping=10.0)
        if env.cfg.dynamic_reset_gripper_effort_limit:
            dynamic_reset_gripper_effort_limit_sim(env, "so101leader")
        observation, _, _, _, _ = env.step(action.unsqueeze(0))
    return observation


def scene_state(env, observation: dict) -> dict:
    joint = np.asarray(convert_leisaac_action_to_lerobot(observation["policy"]["joint_pos"]), dtype=np.float32).reshape(-1)
    objects = {
        name: env.scene[name].data.root_pos_w[0].detach().cpu().tolist()
        for name in (*ORANGES, "Plate")
    }
    ee = None
    try:
        ee = env.scene["robot"].data.body_pos_w[0, -1].detach().cpu().tolist()
    except Exception:
        pass
    return {"joint_position": joint.tolist(), "gripper_position": float(joint[-1]), "object_positions": objects, "end_effector_position": ee}


def initial_state_id(seed: int, episode: int, state: dict) -> str:
    from pick_orange_analysis import stable_state_id

    values = list(state["joint_position"])
    for name in (*ORANGES, "Plate"):
        values.extend(state["object_positions"][name])
    return stable_state_id(seed, episode, values)


def reference_distance(stage: str, state: dict, references: dict) -> dict:
    from pick_orange_analysis import normalized_l2

    ref = references.get(stage) or references.get(stage.lower())
    if not ref:
        return {
            "joint_normalized_l2": None,
            "joint_reason": "expert stage reference unavailable",
            "object_normalized_l2": None,
            "object_reason": "LeRobot phase parquet has no object-position field",
        }
    try:
        return {
            "joint_normalized_l2": normalized_l2(state["joint_position"], ref["joint_mean"], ref["joint_std"]),
            "joint_reason": None,
            "object_normalized_l2": None,
            "object_reason": "LeRobot phase parquet has no object-position field",
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "joint_normalized_l2": None,
            "joint_reason": f"invalid expert stage reference: {exc}",
            "object_normalized_l2": None,
            "object_reason": "LeRobot phase parquet has no object-position field",
        }


def aggregate_stage_deviation(rows: list[dict]) -> dict:
    output = {}
    for stage in ORANGES:
        values = [
            row["expert_reference"]["joint_normalized_l2"]
            for row in rows
            if row["stage"] == stage and row["expert_reference"]["joint_normalized_l2"] is not None
        ]
        output[stage] = {
            "records": sum(row["stage"] == stage for row in rows),
            "joint_reference_available": len(values),
            "joint_normalized_l2_mean": sum(values) / len(values) if values else None,
            "joint_normalized_l2_max": max(values) if values else None,
            "object_reference_available": 0,
            "object_reference_reason": "LeRobot phase parquet has no object-position field",
        }
    return output


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean, y_mean = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_scale = sum((x - x_mean) ** 2 for x in xs) ** 0.5
    y_scale = sum((y - y_mean) ** 2 for y in ys) ** 0.5
    return numerator / (x_scale * y_scale) if x_scale and y_scale else None


def aggregate_stage_diagnostics(rows: list[dict]) -> dict:
    stages = {}
    for stage in ORANGES:
        stage_rows = [row for row in rows if row["stage_id"] == stage]
        overruns = [float(row["post_success_overrun"]) for row in stage_rows if row["post_success_overrun"] is not None]
        successful = [float(row["post_success_overrun"]) for row in stage_rows if row["stage_success"] and row["post_success_overrun"] is not None]
        failed = [float(row["post_success_overrun"]) for row in stage_rows if not row["stage_success"] and row["post_success_overrun"] is not None]
        successful_episodes = [float(row["post_success_overrun"]) for row in stage_rows if row.get("episode_success") and row["post_success_overrun"] is not None]
        failed_episodes = [float(row["post_success_overrun"]) for row in stage_rows if row.get("episode_success") is False and row["post_success_overrun"] is not None]
        stages[stage] = {
            "episodes": len(stage_rows),
            "overrun_available": len(overruns),
            "mean": sum(overruns) / len(overruns) if overruns else None,
            "median": percentile(overruns, 0.5),
            "q25": percentile(overruns, 0.25),
            "q75": percentile(overruns, 0.75),
            "q90": percentile(overruns, 0.90),
            "successful_stage_mean": sum(successful) / len(successful) if successful else None,
            "failed_stage_mean": sum(failed) / len(failed) if failed else None,
            "successful_episode_mean": sum(successful_episodes) / len(successful_episodes) if successful_episodes else None,
            "failed_episode_mean": sum(failed_episodes) / len(failed_episodes) if failed_episodes else None,
            "prefix_destroyed_during_overrun": sum(
                bool(row.get("prefix_destroyed_during_overrun"))
                for row in stage_rows
            ),
        }
    relation_rows = [
        row for row in rows
        if row["post_success_overrun"] is not None and row.get("next_stage_start_joint_normalized_l2") is not None
    ]
    xs = [float(row["post_success_overrun"]) for row in relation_rows]
    ys = [float(row["next_stage_start_joint_normalized_l2"]) for row in relation_rows]
    return {
        "by_stage": stages,
        "overrun_vs_next_stage_start_deviation": {
            "pairs": len(relation_rows),
            "pearson_r": pearson_correlation(xs, ys),
            "interpretation": "descriptive correlation only; no causal claim",
        },
    }


def validate_args() -> list[Path]:
    horizon_flag_explicit = any(value == "--horizon_protocol" or value.startswith("--horizon_protocol=") for value in sys.argv)
    total_steps_explicit = any(value == "--total_policy_steps" or value.startswith("--total_policy_steps=") for value in sys.argv)
    if horizon_flag_explicit and args.horizon_protocol not in str(args.output_dir):
        parser.error("explicit --horizon_protocol requires its name in --output_dir to prevent result overwrite")
    if args.horizon_protocol == "matched_horizon":
        if args.group == "a0":
            if total_steps_explicit and args.total_policy_steps != 1260:
                parser.error("matched_horizon fixes A0 at 1260; remove the conflicting --total_policy_steps")
            args.total_policy_steps = 1260
        elif args.group == "a1" and args.policy_steps_per_orange != 420:
            parser.error("matched_horizon requires A1 --policy_steps_per_orange 420")
    elif horizon_flag_explicit:
        if args.group == "a0" and args.total_policy_steps != 1020:
            parser.error("native_horizon requires A0 --total_policy_steps 1020")
        if args.group == "a1" and args.policy_steps_per_orange != 420:
            parser.error("native_horizon requires A1 --policy_steps_per_orange 420")
    if args.group in ("a0", "a2"):
        if args.checkpoint is None:
            parser.error(f"--checkpoint is required for {args.group}")
        return [args.checkpoint]
    checkpoints = [args.checkpoint_001, args.checkpoint_002, args.checkpoint_003]
    if any(path is None for path in checkpoints):
        parser.error(f"--checkpoint_001/002/003 are required for {args.group}")
    return checkpoints


def main() -> None:
    checkpoint_args = validate_args()
    device = torch.device(args.device)
    bundles = [load_policy(path, device) for path in checkpoint_args]

    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    env_cfg.use_teleop_device("so101leader")
    # Expert joint targets were recorded while gravity was disabled and with
    # damping=10. Match that executor while bypassing IK at inference.
    env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = 40.0
    env_cfg.recorders = None
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    env = gym.make(TASK, cfg=env_cfg).unwrapped

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_state_path = args.stage_state_jsonl or (args.output_dir / "stage_start_states.jsonl")
    stage_diagnostics_path = args.output_dir / "post_success_overrun.jsonl"
    references = {}
    if args.stage_reference_json is not None:
        references = json.loads(args.stage_reference_json.read_text(encoding="utf-8"))
    results = []
    raw_stage_states = []
    raw_stage_diagnostics = []

    for episode in range(args.episodes):
        observation, _ = env.reset()
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)
        for bundle in bundles:
            bundle[0].reset()

        writer = None
        video_path = None
        if args.record_video:
            video_path = args.output_dir / f"episode_{episode:03d}.mp4"
            writer = imageio.get_writer(video_path, fps=FPS, codec="libx264")
            writer.append_data(video_frame(observation))

        initial_z = {orange: float(env.scene[orange].data.root_pos_w[0, 2].item()) for orange in ORANGES}
        initial_state = scene_state(env, observation)
        state_id = initial_state_id(args.seed, episode, initial_state)
        metrics = {
            orange: {"max_lift_m": 0.0, "ever_stably_placed": False} for orange in ORANGES
        }
        stable = {orange: 0 for orange in ORANGES}
        total_steps = 0
        switch_steps = []
        failure_reason = None
        detector_first_success_step = None
        detector_switch_count = 0
        detector_stage_steps = {orange: 0 for orange in ORANGES}
        stage_1_timeout = False
        stage_results = []
        episode_stage_diagnostics = []

        try:
            if args.group in ("a0", "a2"):
                bundle = bundles[0]
                target_total_steps = args.total_policy_steps if args.group == "a0" else TOTAL_POLICY_STEPS
                while total_steps < target_total_steps:
                    stage = stage_index(total_steps) if args.group == "a2" else None
                    remaining = target_total_steps - total_steps
                    if args.group == "a2":
                        remaining = min(remaining, stage_end(total_steps) - total_steps)
                    count = min(args.action_horizon, remaining)
                    actions = predict_actions(bundle, observation, device, count, stage)
                    for action in actions:
                        observation = step_action(env, robot, action)
                        total_steps += 1
                        update_metrics(env, initial_z, metrics, stable)
                        if writer is not None:
                            writer.append_data(video_frame(observation))

            elif args.group == "a1":
                for orange_index, bundle in enumerate(bundles):
                    orange = ORANGES[orange_index]
                    bundle[0].reset()
                    start_state = scene_state(env, observation)
                    distance = reference_distance(orange, start_state, references)
                    stage_start_step = total_steps
                    prefix_intact_at_start = all(orange_in_plate(env, name) for name in ORANGES[:orange_index])
                    raw_stage_states.append({
                        "seed": args.seed, "episode_index": episode, "initial_state_id": state_id,
                        "stage_index": orange_index, "stage": orange, "policy_step": total_steps,
                        "state": start_state, "expert_reference": distance,
                    })
                    phase_steps = 0
                    target_first_satisfied_step = stage_start_step if orange_in_plate(env, orange) else None
                    target_first_stably_satisfied_step = stage_start_step if stable[orange] >= args.stable_plate_steps else None
                    prefix_first_broken_step = None if prefix_intact_at_start else stage_start_step
                    while phase_steps < args.policy_steps_per_orange:
                        count = min(args.action_horizon, args.policy_steps_per_orange - phase_steps)
                        actions = predict_actions(bundle, observation, device, count, None)
                        for action in actions:
                            observation = step_action(env, robot, action)
                            phase_steps += 1
                            total_steps += 1
                            update_metrics(env, initial_z, metrics, stable)
                            if target_first_satisfied_step is None and orange_in_plate(env, orange):
                                target_first_satisfied_step = total_steps
                            if target_first_stably_satisfied_step is None and stable[orange] >= args.stable_plate_steps:
                                target_first_stably_satisfied_step = total_steps
                            if prefix_first_broken_step is None and not all(orange_in_plate(env, name) for name in ORANGES[:orange_index]):
                                prefix_first_broken_step = total_steps
                            if writer is not None:
                                writer.append_data(video_frame(observation))
                    switch_steps.append(total_steps)
                    at_end = {name: orange_in_plate(env, name) for name in ORANGES}
                    target_success = at_end[orange]
                    prefix_intact = all(at_end[name] for name in ORANGES[:orange_index])
                    stage_switch_step = total_steps
                    from pick_orange_analysis import post_success_overrun as compute_overrun

                    post_success_overrun = compute_overrun(stage_switch_step, target_first_stably_satisfied_step)
                    outcome = "success" if target_success and prefix_intact else "prefix_lost" if not prefix_intact else "target_not_placed"
                    if target_first_satisfied_step is None:
                        stage_failure_reason = "target_never_satisfied"
                    elif target_first_stably_satisfied_step is None:
                        stage_failure_reason = "target_never_stably_satisfied"
                    elif not target_success:
                        stage_failure_reason = "target_lost_before_switch"
                    elif not prefix_intact:
                        stage_failure_reason = "prefix_lost_before_switch"
                    else:
                        stage_failure_reason = None
                    diagnostic = {
                        "episode_index": episode,
                        "seed": args.seed,
                        "initialization_id": state_id,
                        "initial_state_id": state_id,
                        "stage_index": orange_index,
                        "stage_id": orange,
                        "stage_start_step": stage_start_step,
                        "target_first_satisfied_step": target_first_satisfied_step,
                        "target_first_stably_satisfied_step": target_first_stably_satisfied_step,
                        "stage_switch_step": stage_switch_step,
                        "post_success_overrun": post_success_overrun,
                        "target_in_plate_before_switch": target_success,
                        "prefix_intact_at_start": prefix_intact_at_start,
                        "prefix_intact_before_switch": prefix_intact,
                        "prefix_first_broken_step": prefix_first_broken_step,
                        "prefix_destroyed_during_overrun": (
                            target_first_stably_satisfied_step is not None
                            and prefix_first_broken_step is not None
                            and prefix_first_broken_step >= target_first_stably_satisfied_step
                        ),
                        "stage_success": outcome == "success",
                        "stage_outcome": outcome,
                        "failure_reason": stage_failure_reason,
                        "stage_start_deviation": distance,
                        "next_stage_start_joint_normalized_l2": None,
                    }
                    episode_stage_diagnostics.append(diagnostic)
                    stage_results.append({
                        "stage_index": orange_index, "target": orange, "reached": True,
                        "start_policy_step": total_steps - phase_steps, "end_policy_step": total_steps,
                        "target_in_plate_at_end": target_success, "prefix_intact_at_end": prefix_intact,
                        "in_plate_at_end": at_end,
                        "target_first_satisfied_step": target_first_satisfied_step,
                        "target_first_stably_satisfied_step": target_first_stably_satisfied_step,
                        "stage_switch_step": stage_switch_step,
                        "post_success_overrun": post_success_overrun,
                        "failure_reason": stage_failure_reason,
                        "stage_start_deviation": distance,
                        "outcome": outcome,
                    })

                for index, diagnostic in enumerate(episode_stage_diagnostics[:-1]):
                    diagnostic["next_stage_start_joint_normalized_l2"] = episode_stage_diagnostics[index + 1]["stage_start_deviation"].get("joint_normalized_l2")
                raw_stage_diagnostics.extend(episode_stage_diagnostics)

            else:  # A3: switch only after the current orange is stably placed.
                for orange_index, (orange, bundle) in enumerate(zip(ORANGES, bundles, strict=True)):
                    bundle[0].reset()
                    phase_steps = 0
                    phase_success = False
                    while phase_steps < args.policy_steps_per_orange and not phase_success:
                        count = min(args.action_horizon, args.policy_steps_per_orange - phase_steps)
                        actions = predict_actions(bundle, observation, device, count, None)
                        for action in actions:
                            observation = step_action(env, robot, action)
                            phase_steps += 1
                            total_steps += 1
                            detector_stage_steps[orange] = phase_steps
                            update_metrics(env, initial_z, metrics, stable)
                            if writer is not None:
                                writer.append_data(video_frame(observation))
                            if stable[orange] >= args.stable_plate_steps:
                                phase_success = True
                                switch_steps.append(total_steps)
                                detector_switch_count += 1
                                if detector_first_success_step is None:
                                    detector_first_success_step = total_steps
                                break
                    if not phase_success:
                        failure_reason = f"{orange}_not_stably_placed"
                        if orange_index == 0:
                            stage_1_timeout = True
                        break
        finally:
            if writer is not None:
                writer.close()

        final_in_plate = {orange: orange_in_plate(env, orange) for orange in ORANGES}
        success = all(final_in_plate.values())
        for diagnostic in raw_stage_diagnostics:
            if diagnostic["episode_index"] == episode:
                diagnostic["episode_success"] = success
        if failure_reason is None and not success:
            failed_stage = next((row for row in stage_results if row["outcome"] != "success"), None)
            failure_reason = failed_stage["outcome"] if failed_stage else "final_configuration_incomplete"
        result = {
            "episode": episode,
            "episode_index": episode,
            "seed": args.seed,
            "initial_state_id": state_id,
            "initialization_id": state_id,
            "dataset_label": args.dataset_label,
            "strategy_label": args.strategy_label or args.group,
            "checkpoint_label": args.checkpoint_label,
            "success": success,
            "policy_steps": total_steps,
            "switch_steps": switch_steps,
            "failure_reason": failure_reason,
            "detector_first_success_step": detector_first_success_step,
            "detector_switch_count": detector_switch_count,
            "time_spent_in_stage_1": detector_stage_steps["Orange001"],
            "stage_1_timeout": stage_1_timeout,
            "detector_stage_steps": detector_stage_steps,
            "oranges": metrics,
            "final_in_plate": final_in_plate,
            "stage_results": stage_results,
            "video": str(video_path.resolve()) if video_path is not None else None,
        }
        results.append(result)
        print(f"Episode {episode + 1}/{args.episodes}: {result}", flush=True)

    successes = sum(int(result["success"]) for result in results)
    stage_success = {
        orange: sum(int(result["oranges"][orange]["ever_stably_placed"]) for result in results)
        for orange in ORANGES
    }
    summary = {
        "schema_version": 3,
        "group": args.group,
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / args.episodes,
        "stage_successes": stage_success,
        "stage_success_rates": {orange: count / args.episodes for orange, count in stage_success.items()},
        "seed": args.seed,
        "action_horizon": args.action_horizon,
        "policy_steps_per_orange": args.policy_steps_per_orange,
        "horizon_protocol": args.horizon_protocol,
        "horizon_protocol_definition": {
            "native_horizon": {"a0_policy_actions": 1020, "a1_policy_actions": 1260},
            "matched_horizon": {"a0_policy_actions": 1260, "a1_policy_actions": 1260},
        },
        "a0_a1_same_total_horizon": args.horizon_protocol == "matched_horizon",
        "horizon_protocol_conformant": (
            args.total_policy_steps == (1260 if args.horizon_protocol == "matched_horizon" else 1020)
            if args.group == "a0" else args.policy_steps_per_orange == 420 if args.group == "a1" else None
        ),
        "total_policy_steps": args.total_policy_steps if args.group == "a0" else TOTAL_POLICY_STEPS if args.group == "a2" else args.policy_steps_per_orange * 3,
        "sim_steps_per_action": args.sim_steps_per_action,
        "simulation_steps": (args.total_policy_steps if args.group == "a0" else TOTAL_POLICY_STEPS if args.group == "a2" else args.policy_steps_per_orange * 3) * args.sim_steps_per_action,
        "sim_dt_s": float(env.cfg.sim.dt),
        "decimation": int(getattr(env.cfg, "decimation", 1)),
        "theoretical_duration_s": (args.total_policy_steps if args.group == "a0" else TOTAL_POLICY_STEPS if args.group == "a2" else args.policy_steps_per_orange * 3) * args.sim_steps_per_action * float(env.cfg.sim.dt) * int(getattr(env.cfg, "decimation", 1)),
        "scheduler_classification": "fixed-time external multi-policy scheduler" if args.group == "a1" else "single-policy fixed-horizon" if args.group == "a0" else "ground-truth success oracle" if args.group == "a3" else "single-policy fixed-stage schedule",
        "success_definition": "all three orange centers inside evaluator plate tolerances at final step; robot rest is not required",
        "dataset_label": args.dataset_label,
        "strategy_label": args.strategy_label or args.group,
        "checkpoint_label": args.checkpoint_label,
        "stable_plate_steps": args.stable_plate_steps,
        "executor": "6D direct joint-position targets; gravity disabled; joint damping 10",
        "checkpoints": [str(bundle[3]) for bundle in bundles],
        "episode_results": results,
        "stage_start_deviation_summary": aggregate_stage_deviation(raw_stage_states),
        "post_success_overrun_summary": aggregate_stage_diagnostics(raw_stage_diagnostics),
    }
    with stage_state_path.open("w", encoding="utf-8") as stream:
        for row in raw_stage_states:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["stage_state_jsonl"] = str(stage_state_path.resolve())
    with stage_diagnostics_path.open("w", encoding="utf-8") as stream:
        for row in raw_stage_diagnostics:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["post_success_overrun_jsonl"] = str(stage_diagnostics_path.resolve())
    summary_path = args.output_dir / "summary.json"
    protocol_summary_path = args.output_dir / f"summary.{args.horizon_protocol}.json"
    summary["protocol_summary"] = str(protocol_summary_path.resolve())
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    protocol_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Result: {successes}/{args.episodes} = {successes / args.episodes:.1%}", flush=True)
    print(f"Summary: {summary_path.resolve()}", flush=True)
    print(f"Protocol summary: {protocol_summary_path.resolve()}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
