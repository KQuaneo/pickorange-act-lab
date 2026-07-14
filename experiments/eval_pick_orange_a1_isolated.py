"""Isolated evaluation for the three A1 PickOrange sub-policies.

B1: Orange001 policy from a normal environment reset.
B2: Orange002 policy from a synthetic oracle state where Orange001 is already on the plate.
B3: Orange003 policy from a synthetic oracle state where Orange001/002 are already on the plate.

The synthetic oracle state avoids mixing the 8D IK state-machine controller with
the 6D direct-joint ACT executor.  For B2/B3, previous oranges are teleported to
stable plate locations and the robot joints are initialized from the first
training frame of the corresponding subtask dataset.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from contextlib import nullcontext
from pathlib import Path

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=("b1", "b2", "b3"), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument(
    "--phase_dataset",
    type=Path,
    default=None,
    help="Optional LeRobot parquet for the phase start state. Defaults to the legacy A1 dataset mapping.",
)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--action_horizon", type=int, default=100)
parser.add_argument(
    "--policy_steps",
    type=int,
    default=420,
    help="Maximum policy actions. 420 covers the longest Gate3 B2/B3 expert slices and settle time.",
)
parser.add_argument("--sim_steps_per_action", type=int, default=2)
parser.add_argument("--dataset_label", default=None)
parser.add_argument("--strategy_label", default="a1_isolated_oracle_init")
parser.add_argument("--checkpoint_label", default=None)
parser.add_argument("--stage_state_jsonl", type=Path, default=None)
parser.add_argument("--stable_plate_steps", type=int, default=10)
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
import pandas as pd
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


ROOT = Path(__file__).resolve().parents[1]
TASK = "LeIsaac-SO101-PickOrange-v0"
ORANGES = ("Orange001", "Orange002", "Orange003")
PHASE_TO_TARGET = {"b1": "Orange001", "b2": "Orange002", "b3": "Orange003"}
PHASE_TO_DATASET = {
    "b1": ROOT / "data/lerobot/local/so101_pick_orange_a1_orange001_joint6_v0/data/chunk-000/file-000.parquet",
    "b2": ROOT / "data/lerobot/local/so101_pick_orange_a1_orange002_joint6_v0/data/chunk-000/file-000.parquet",
    "b3": ROOT / "data/lerobot/local/so101_pick_orange_a1_orange003_joint6_v0/data/chunk-000/file-000.parquet",
}
PHASE_TO_COMPLETED = {"b1": (), "b2": ("Orange001",), "b3": ("Orange001", "Orange002")}
SIM_STEPS_PER_ACTION = 2
FPS = 30


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


def policy_observation(observation: dict, device: torch.device) -> dict:
    state = np.asarray(convert_leisaac_action_to_lerobot(observation["policy"]["joint_pos"]), dtype=np.float32)
    if state.ndim == 1:
        state = state[None, :]
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


def predict_actions(bundle, observation: dict, device: torch.device, count: int) -> torch.Tensor:
    policy, preprocessor, postprocessor, _ = bundle
    model_input = policy_observation(observation, device)
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


def captured_state(env, observation: dict) -> dict:
    joint = np.asarray(convert_leisaac_action_to_lerobot(observation["policy"]["joint_pos"]), dtype=np.float32).reshape(-1)
    return {
        "joint_position": joint.tolist(),
        "gripper_position": float(joint[-1]),
        "object_positions": {name: env.scene[name].data.root_pos_w[0].detach().cpu().tolist() for name in (*ORANGES, "Plate")},
        "end_effector_position": env.scene["robot"].data.body_pos_w[0, -1].detach().cpu().tolist(),
    }


def phase_dataset_path(phase: str) -> Path:
    return args.phase_dataset if args.phase_dataset is not None else PHASE_TO_DATASET[phase]


def first_subtask_state(phase: str) -> torch.Tensor:
    table = pd.read_parquet(phase_dataset_path(phase), columns=["episode_index", "frame_index", "observation.state"])
    row = table[(table["episode_index"] == 0) & (table["frame_index"] == 0)].iloc[0]
    return torch.as_tensor(np.asarray(row["observation.state"], dtype=np.float32))


ORACLE_PLATE_OFFSETS_M = {
    "Orange001": [0.045, 0.000, 0.045],
    "Orange002": [-0.025, 0.040, 0.045],
    "Orange003": [-0.025, -0.040, 0.045],
}


def place_oracle_oranges(env, completed: tuple[str, ...]) -> dict:
    placed = {}
    if not completed:
        return placed
    plate_pose = env.scene["Plate"].data.root_pose_w.clone()
    for orange in completed:
        offset = torch.tensor(ORACLE_PLATE_OFFSETS_M[orange], device=env.device)
        pose = env.scene[orange].data.root_pose_w.clone()
        pose[:, :3] = plate_pose[:, :3] + offset.unsqueeze(0)
        env.scene[orange].write_root_pose_to_sim(pose)
        env.scene[orange].write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
        placed[orange] = {
            "plate_offset_m": ORACLE_PLATE_OFFSETS_M[orange],
            "world_pos_m": [float(value) for value in pose[0, :3].detach().cpu().tolist()],
        }
    return placed


def set_robot_subtask_start(env, robot, phase: str) -> dict:
    if phase == "b1":
        return {"source": "standard_env_reset", "target_lerobot_state": None}
    lerobot_state = first_subtask_state(phase).to(env.device).unsqueeze(0)
    joint_pos = torch.as_tensor(convert_lerobot_action_to_leisaac(lerobot_state), dtype=torch.float32, device=env.device)
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    return {
        "source": f"{phase_dataset_path(phase)} episode=0 frame=0 observation.state",
        "target_lerobot_state": [float(value) for value in lerobot_state[0].detach().cpu().tolist()],
    }


def synthesize_oracle_state(env, observation: dict, phase: str) -> tuple[dict, dict[str, bool], dict]:
    robot = env.scene["robot"]
    robot_check = set_robot_subtask_start(env, robot, phase)
    placed_oranges = place_oracle_oranges(env, PHASE_TO_COMPLETED[phase])
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    # Step one no-op action to refresh manager observations and camera tensors.
    noop = robot.data.joint_pos[0].clone()
    observation = step_action(env, robot, noop)
    prefix_valid = {orange: orange_in_plate(env, orange) for orange in PHASE_TO_COMPLETED[phase]}
    observed_lerobot_state = np.asarray(
        convert_leisaac_action_to_lerobot(observation["policy"]["joint_pos"]), dtype=np.float32
    )
    if robot_check["target_lerobot_state"] is None:
        max_abs_joint_state_error = None
    else:
        target = np.asarray(robot_check["target_lerobot_state"], dtype=np.float32)
        max_abs_joint_state_error = float(np.max(np.abs(observed_lerobot_state - target)))
    oracle_check = {
        "type": "oracle_initialized_isolated_evaluation",
        "interpretation": "upper-bound primitive test, not a real sequential rollout",
        "completed_oranges": list(PHASE_TO_COMPLETED[phase]),
        "completed_orange_placements": placed_oranges,
        "robot_start_state": robot_check,
        "observed_lerobot_state_after_refresh": [float(value) for value in observed_lerobot_state.reshape(-1).tolist()],
        "max_abs_lerobot_joint_state_error_after_refresh": max_abs_joint_state_error,
        "camera_refresh": "front/wrist observations are rendered after teleporting prior oranges and refreshing one no-op step",
        "distribution_caveat": "synthetic prior-orange placement may be easier or harder than real sequential rollout; use B2/B3 as ideal-prefix upper-bound tests",
    }
    return observation, prefix_valid, oracle_check


def main() -> None:
    device = torch.device(args.device)
    target_orange = PHASE_TO_TARGET[args.phase]
    bundle = load_policy(args.checkpoint, device)

    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    env_cfg.use_teleop_device("so101leader")
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
    results = []
    raw_stage_states = []

    for episode in range(args.episodes):
        observation, _ = env.reset()
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)
        bundle[0].reset()
        observation, prefix_valid, oracle_check = synthesize_oracle_state(env, observation, args.phase)
        start_state = captured_state(env, observation)
        from pick_orange_analysis import stable_state_id

        id_values = list(start_state["joint_position"])
        for name in (*ORANGES, "Plate"):
            id_values.extend(start_state["object_positions"][name])
        state_id = stable_state_id(args.seed, episode, id_values)
        raw_stage_states.append({
            "seed": args.seed,
            "episode_index": episode,
            "initial_state_id": state_id,
            "initialization_id": state_id,
            "phase": args.phase,
            "target_orange": target_orange,
            "state": start_state,
            "initialization": "standard reset" if args.phase == "b1" else "oracle initialized",
        })

        writer = None
        video_path = None
        if args.record_video:
            video_path = args.output_dir / f"episode_{episode:03d}.mp4"
            writer = imageio.get_writer(video_path, fps=FPS, codec="libx264")
            writer.append_data(video_frame(observation))

        initial_z = {orange: float(env.scene[orange].data.root_pos_w[0, 2].item()) for orange in ORANGES}
        metrics = {orange: {"max_lift_m": 0.0, "ever_stably_placed": False} for orange in ORANGES}
        stable = {orange: 0 for orange in ORANGES}
        total_steps = 0

        try:
            while total_steps < args.policy_steps:
                count = min(args.action_horizon, args.policy_steps - total_steps)
                actions = predict_actions(bundle, observation, device, count)
                for action in actions:
                    observation = step_action(env, robot, action)
                    total_steps += 1
                    update_metrics(env, initial_z, metrics, stable)
                    if writer is not None:
                        writer.append_data(video_frame(observation))
        finally:
            if writer is not None:
                writer.close()

        final_in_plate = {orange: orange_in_plate(env, orange) for orange in ORANGES}
        target_success = bool(metrics[target_orange]["ever_stably_placed"] or final_in_plate[target_orange])
        result = {
            "episode": episode,
            "episode_index": episode,
            "seed": args.seed,
            "initial_state_id": state_id,
            "dataset_label": args.dataset_label,
            "strategy_label": args.strategy_label,
            "checkpoint_label": args.checkpoint_label,
            "phase": args.phase,
            "target_orange": target_orange,
            "success": target_success,
            "policy_steps": total_steps,
            "prefix_valid": prefix_valid,
            "oracle_check": oracle_check,
            "oranges": metrics,
            "final_in_plate": final_in_plate,
            "video": str(video_path.resolve()) if video_path is not None else None,
        }
        results.append(result)
        print(f"Episode {episode + 1}/{args.episodes}: {result}", flush=True)

    successes = sum(int(result["success"]) for result in results)
    prefix_valid_count = sum(int(all(result["prefix_valid"].values())) for result in results)
    summary = {
        "schema_version": 2,
        "phase": args.phase,
        "target_orange": target_orange,
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / args.episodes,
        "prefix_valid_episodes": prefix_valid_count,
        "prefix_valid_rate": prefix_valid_count / args.episodes,
        "seed": args.seed,
        "action_horizon": args.action_horizon,
        "policy_steps": args.policy_steps,
        "sim_steps_per_action": args.sim_steps_per_action,
        "simulation_steps": args.policy_steps * args.sim_steps_per_action,
        "sim_dt_s": float(env.cfg.sim.dt),
        "decimation": int(getattr(env.cfg, "decimation", 1)),
        "theoretical_duration_s": args.policy_steps * args.sim_steps_per_action * float(env.cfg.sim.dt) * int(getattr(env.cfg, "decimation", 1)),
        "stable_plate_steps": args.stable_plate_steps,
        "oracle_state": "oracle initialized isolated evaluation; synthetic previous oranges on plate; robot joints set from subtask dataset frame 0 when applicable",
        "oracle_plate_offsets_m": ORACLE_PLATE_OFFSETS_M,
        "interpretation": "B2/B3 measure sub-policy ability under idealized prefix conditions; they are not estimates of real sequential rollout performance",
        "scheduler_classification": "oracle initialization for isolated primitive evaluation; no sequential scheduler",
        "success_definition": "target orange is ever stably placed or lies inside evaluator plate tolerances at final step",
        "dataset_label": args.dataset_label,
        "strategy_label": args.strategy_label,
        "checkpoint_label": args.checkpoint_label,
        "checkpoint": str(bundle[3]),
        "phase_dataset": str(args.phase_dataset) if args.phase_dataset is not None else str(PHASE_TO_DATASET[args.phase]),
        "episode_results": results,
    }
    with stage_state_path.open("w", encoding="utf-8") as stream:
        for row in raw_stage_states:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["stage_state_jsonl"] = str(stage_state_path.resolve())
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Result: {successes}/{args.episodes} = {successes / args.episodes:.1%}", flush=True)
    print(f"Summary: {summary_path.resolve()}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
