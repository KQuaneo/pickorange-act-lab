#!/usr/bin/env python3
"""Paired B1 inference experiment for ACT temporal aggregation.

The evaluator keeps K=100 and executes one action per policy call.  For each
manifest entry it runs the no-aggregation condition, restores the exact same
physical state, verifies the refreshed initial state numerically, and only then
runs LeRobot's native ACTTemporalEnsembler with coeff=0.01.

This file never trains, deletes, or mutates a checkpoint.  A formal run creates
COMPLETED only after all paired episodes finish and the checkpoint SHA256 is
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
TASK = "LeIsaac-SO101-PickOrange-v0"
ORANGES = ("Orange001", "Orange002", "Orange003")
OBJECTS = (*ORANGES, "Plate")
TARGET = "Orange001"
CHUNK_SIZE = 100
POLICY_STEPS = 420
SIM_STEPS_PER_ACTION = 2
TEMPORAL_COEFF = 0.01
FORBIDDEN_ROOT = Path("/workspace/Octopus-SingleArm")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--policy-steps", type=int, default=POLICY_STEPS)
    parser.add_argument("--sim-steps-per-action", type=int, default=SIM_STEPS_PER_ACTION)
    parser.add_argument("--stable-plate-steps", type=int, default=10)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.0)
    parser.add_argument("--pair-atol", type=float, default=1e-6)
    parser.add_argument("--generate-manifest-only", action="store_true")
    parser.add_argument("--deletion-pause-confirmed", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def check_paths(args: argparse.Namespace) -> None:
    for path in (Path.cwd(), args.checkpoint, args.output_root, args.manifest):
        if is_under(path, FORBIDDEN_ROOT):
            raise RuntimeError(f"refusing to touch protected path: {path}")
    if not args.deletion_pause_confirmed:
        raise RuntimeError("execution gated: pass --deletion-pause-confirmed only after the server deletion flow is paused")
    if args.policy_steps != POLICY_STEPS:
        raise ValueError(f"formal protocol requires policy_steps={POLICY_STEPS}")
    if args.sim_steps_per_action != SIM_STEPS_PER_ACTION:
        raise ValueError(f"formal protocol requires sim_steps_per_action={SIM_STEPS_PER_ACTION}")
    if not 1 <= args.episodes <= 20:
        raise ValueError("episodes must be in [1, 20]")


def check_gpu_idle(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"cannot verify GPU safety: {completed.stderr.strip()}")
    active = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if active:
        raise RuntimeError("GPU has active compute processes; protected-task safety stop: " + " | ".join(active))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_state_id(seed: int, episode: int, vector: list[float]) -> str:
    rounded = [round(float(value), 7) for value in vector]
    raw = json.dumps([seed, episode, rounded], separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonl_append(stream: TextIO, payload: dict) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()


class MultiTee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


parser = build_parser()
args = parser.parse_args()
check_paths(args)
check_gpu_idle(args.device)
args.enable_cameras = True
app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.factory import make_pre_post_processors

import leisaac.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac


def policy_image(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 4:
        image = image[0]
    if image.shape[-1] in (3, 4):
        image = image[..., :3].permute(2, 0, 1)
    image = image.to(dtype=torch.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image.unsqueeze(0)


def policy_observation(observation: dict, device: torch.device) -> dict:
    state = np.asarray(convert_leisaac_action_to_lerobot(observation["policy"]["joint_pos"]), dtype=np.float32)
    if state.ndim == 1:
        state = state[None, :]
    return {
        "observation.state": torch.as_tensor(state, device=device),
        "observation.images.front": policy_image(observation["policy"]["front"]).to(device),
        "observation.images.wrist": policy_image(observation["policy"]["wrist"]).to(device),
    }


def observation_digest(observation: dict) -> dict:
    joint = np.asarray(convert_leisaac_action_to_lerobot(observation["policy"]["joint_pos"]), dtype=np.float32).reshape(-1)
    front = observation["policy"]["front"].detach().float()
    wrist = observation["policy"]["wrist"].detach().float()
    return {
        "joint_sha256": hashlib.sha256(joint.tobytes()).hexdigest()[:16],
        "front_mean": float(front.mean().item()),
        "wrist_mean": float(wrist.mean().item()),
    }


def capture_state(env) -> dict:
    robot = env.scene["robot"]
    return {
        "robot_joint_position": robot.data.joint_pos[0].detach().cpu().tolist(),
        "robot_joint_velocity": robot.data.joint_vel[0].detach().cpu().tolist(),
        "objects": {
            name: {
                "root_pose_w": env.scene[name].data.root_pose_w[0].detach().cpu().tolist(),
                "root_velocity_w": env.scene[name].data.root_vel_w[0].detach().cpu().tolist(),
            }
            for name in OBJECTS
        },
    }


def state_vector(state: dict) -> list[float]:
    values = [*state["robot_joint_position"], *state["robot_joint_velocity"]]
    for name in OBJECTS:
        values.extend(state["objects"][name]["root_pose_w"])
        values.extend(state["objects"][name]["root_velocity_w"])
    return [float(value) for value in values]


def restore_state(env, state: dict) -> None:
    robot = env.scene["robot"]
    joint_pos = torch.as_tensor(state["robot_joint_position"], dtype=torch.float32, device=env.device).unsqueeze(0)
    joint_vel = torch.as_tensor(state["robot_joint_velocity"], dtype=torch.float32, device=env.device).unsqueeze(0)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    for name in OBJECTS:
        root_pose = torch.as_tensor(state["objects"][name]["root_pose_w"], dtype=torch.float32, device=env.device).unsqueeze(0)
        root_velocity = torch.as_tensor(state["objects"][name]["root_velocity_w"], dtype=torch.float32, device=env.device).unsqueeze(0)
        env.scene[name].write_root_pose_to_sim(root_pose)
        env.scene[name].write_root_velocity_to_sim(root_velocity)


def step_action(env, robot, action: torch.Tensor) -> dict:
    observation = None
    for _ in range(args.sim_steps_per_action):
        robot.write_joint_damping_to_sim(damping=10.0)
        if env.cfg.dynamic_reset_gripper_effort_limit:
            dynamic_reset_gripper_effort_limit_sim(env, "so101leader")
        observation, _, _, _, _ = env.step(action.unsqueeze(0))
    assert observation is not None
    return observation


def restore_and_refresh(env, state: dict) -> tuple[dict, dict]:
    observation, _ = env.reset()
    robot = env.scene["robot"]
    robot.write_joint_damping_to_sim(damping=10.0)
    restore_state(env, state)
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    noop = robot.data.joint_pos[0].clone()
    observation = step_action(env, robot, noop)
    refreshed = capture_state(env)
    return observation, refreshed


def orange_in_plate(env, orange: str) -> bool:
    delta = env.scene[orange].data.root_pos_w[0] - env.scene["Plate"].data.root_pos_w[0]
    return abs(float(delta[0])) < 0.10 and abs(float(delta[1])) < 0.10 and abs(float(delta[2])) < 0.07


def failure_category(success: bool, max_lift_m: float, first_close: int | None) -> str:
    if success:
        return "final_success"
    if first_close is None:
        return "no_gripper_close_command"
    if max_lift_m < 0.005:
        return "no_effect_after_close"
    if max_lift_m < 0.03:
        return "low_lift_without_placement"
    return "high_lift_without_placement"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - index) + ordered[hi] * (index - lo))


@dataclass
class PolicyBundle:
    policy: ACTPolicy
    preprocessor: Any
    postprocessor: Any
    checkpoint: Path


def load_policy(checkpoint: Path, device: torch.device) -> PolicyBundle:
    checkpoint = checkpoint.resolve()
    model_path = checkpoint / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    policy = ACTPolicy.from_pretrained(checkpoint).to(device).eval()
    if int(policy.config.chunk_size) != CHUNK_SIZE:
        raise ValueError(f"checkpoint chunk_size={policy.config.chunk_size}, expected {CHUNK_SIZE}")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return PolicyBundle(policy, preprocessor, postprocessor, checkpoint)


def postprocess_action(bundle: PolicyBundle, raw_action: torch.Tensor, device: torch.device) -> torch.Tensor:
    motor_action = bundle.postprocessor(raw_action).float()
    joint_action = convert_lerobot_action_to_leisaac(motor_action)
    return torch.as_tensor(joint_action, dtype=torch.float32, device=device).reshape(-1)


@dataclass
class GroupArtifacts:
    name: str
    directory: Path
    episode_stream: TextIO
    initial_stream: TextIO
    policy_stream: TextIO
    aggregation_stream: TextIO
    results: list[dict]

    @classmethod
    def create(cls, name: str, directory: Path, config: dict, command: str, checkpoint_sha: str, checkpoint_path: Path):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        json_dump(directory / "run_config.json", config)
        (directory / "command.txt").write_text(command + "\n", encoding="utf-8")
        hash_line = f"{checkpoint_sha}  {checkpoint_path}\n"
        (directory / "checkpoint_sha256_before.txt").write_text(hash_line, encoding="utf-8")
        return cls(
            name=name,
            directory=directory,
            episode_stream=(directory / "episode_results.jsonl").open("w", encoding="utf-8"),
            initial_stream=(directory / "initial_states.jsonl").open("w", encoding="utf-8"),
            policy_stream=(directory / "policy_calls.jsonl").open("w", encoding="utf-8"),
            aggregation_stream=(directory / "aggregation_diagnostics.jsonl").open("w", encoding="utf-8"),
            results=[],
        )

    def close(self) -> None:
        for stream in (self.episode_stream, self.initial_stream, self.policy_stream, self.aggregation_stream):
            stream.close()


def load_manifest(path: Path, episodes: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < episodes:
        raise ValueError(f"manifest has {len(rows)} states, need {episodes}")
    selected = rows[:episodes]
    for index, row in enumerate(selected):
        if row.get("episode_index") != index or row.get("seed") != seed:
            raise ValueError(f"manifest row {index} does not match seed/index protocol")
    return selected


def generate_manifest(env, path: Path, episodes: int, seed: int) -> None:
    if path.exists():
        existing = load_manifest(path, episodes, seed)
        print(f"Reusing manifest with {len(existing)} validated states: {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for episode in range(episodes):
            env.reset()
            state = capture_state(env)
            vector = state_vector(state)
            row = {
                "schema_version": 1,
                "seed": seed,
                "episode_index": episode,
                "initialization_id": stable_state_id(seed, episode, vector),
                "state": state,
                "capture_protocol": "standard env.reset; capture robot q/qd and Orange001/2/3 + Plate root pose/velocity before paired refresh",
            }
            jsonl_append(stream, row)
    print(f"Generated manifest: {path}", flush=True)


def run_rollout(
    env,
    bundle: PolicyBundle,
    device: torch.device,
    artifacts: GroupArtifacts,
    episode: int,
    initialization_id: str,
    observation: dict,
    initial_state: dict,
    aggregation: bool,
) -> dict:
    robot = env.scene["robot"]
    bundle.policy.reset()
    ensembler = ACTTemporalEnsembler(TEMPORAL_COEFF, CHUNK_SIZE) if aggregation else None
    initial_z = float(env.scene[TARGET].data.root_pos_w[0, 2].item())
    max_lift_m = 0.0
    stable_count = 0
    ever_stably_placed = False
    first_close: int | None = None
    selected_actions: list[list[float]] = []
    policy_times: list[float] = []
    amp_context = (lambda: torch.autocast(device_type="cuda")) if device.type == "cuda" else nullcontext
    start_wall = time.perf_counter()

    jsonl_append(artifacts.initial_stream, {
        "episode_index": episode,
        "seed": args.seed,
        "initialization_id": initialization_id,
        "state": initial_state,
        "refresh_protocol": "restore manifest; sim.step(render=False); scene.update; one no-op env action with 2 sim steps; use returned observation",
    })

    for step in range(args.policy_steps):
        digest = observation_digest(observation)
        model_input = policy_observation(observation, device)
        call_start = time.perf_counter()
        with torch.inference_mode(), amp_context():
            raw_chunk = bundle.policy.predict_action_chunk(bundle.preprocessor(model_input))
        call_time = time.perf_counter() - call_start
        policy_times.append(call_time)
        if tuple(raw_chunk.shape[:2]) != (1, CHUNK_SIZE):
            raise RuntimeError(f"unexpected action chunk shape: {tuple(raw_chunk.shape)}")

        latest_raw = raw_chunk[:, 0]
        latest_action = postprocess_action(bundle, latest_raw, device)
        if aggregation:
            assert ensembler is not None
            previous_count = 0 if ensembler.ensembled_actions_count is None else int(ensembler.ensembled_actions_count[0].item())
            ensemble_size = previous_count + 1
            aggregated_raw = ensembler.update(raw_chunk)
            selected_action = postprocess_action(bundle, aggregated_raw, device)
            weight_sum = float(ensembler.ensemble_weights_cumsum[ensemble_size - 1].item())
        else:
            ensemble_size = 1
            weight_sum = 1.0
            selected_action = latest_action

        selected_list = [float(value) for value in selected_action.detach().cpu().tolist()]
        latest_list = [float(value) for value in latest_action.detach().cpu().tolist()]
        difference = float(torch.linalg.vector_norm(selected_action - latest_action).item())
        selected_actions.append(selected_list)
        if first_close is None and selected_list[-1] <= args.gripper_close_threshold:
            first_close = step

        jsonl_append(artifacts.policy_stream, {
            "episode_index": episode,
            "step": step,
            "policy_call_index": step,
            "initialization_id": initialization_id,
            "policy_call_wall_time_s": call_time,
            "chunk_size": CHUNK_SIZE,
            "observation_refresh_serial": step,
            "observation_digest": digest,
        })
        jsonl_append(artifacts.aggregation_stream, {
            "episode_index": episode,
            "step": step,
            "initialization_id": initialization_id,
            "aggregation_enabled": aggregation,
            "temporal_ensemble_coeff": TEMPORAL_COEFF if aggregation else None,
            "ensemble_size_per_step": ensemble_size,
            "ensemble_weight_sum": weight_sum,
            "aggregated_action": selected_list,
            "latest_chunk_action": latest_list,
            "aggregation_action_difference_l2": difference,
            "action_space": "postprocessed LeRobot action converted to LeIsaac 6D direct joint target",
        })

        observation = step_action(env, robot, selected_action)
        lift = float(env.scene[TARGET].data.root_pos_w[0, 2].item()) - initial_z
        max_lift_m = max(max_lift_m, lift)
        if orange_in_plate(env, TARGET):
            stable_count += 1
        else:
            stable_count = 0
        ever_stably_placed = ever_stably_placed or stable_count >= args.stable_plate_steps

    final_in_plate = orange_in_plate(env, TARGET)
    success = bool(ever_stably_placed or final_in_plate)
    deltas = [
        math.sqrt(sum((b - a) ** 2 for a, b in zip(selected_actions[index - 1], selected_actions[index], strict=True)))
        for index in range(1, len(selected_actions))
    ]
    result = {
        "episode_index": episode,
        "seed": args.seed,
        "initialization_id": initialization_id,
        "success": success,
        "max_lift_m": max_lift_m,
        "failure_category": failure_category(success, max_lift_m, first_close),
        "first_gripper_close_step": first_close,
        "policy_call_count": len(policy_times),
        "rollout_wall_time": time.perf_counter() - start_wall,
        "final_in_plate": final_in_plate,
        "ever_stably_placed": ever_stably_placed,
        "action_jitter_mean_l2": sum(deltas) / len(deltas) if deltas else 0.0,
        "action_jitter_p95_l2": percentile(deltas, 0.95),
        "policy_call_time_mean_s": sum(policy_times) / len(policy_times),
        "policy_call_time_p95_s": percentile(policy_times, 0.95),
    }
    artifacts.results.append(result)
    jsonl_append(artifacts.episode_stream, result)
    print(f"{artifacts.name} episode {episode + 1}/{args.episodes}: {result}", flush=True)
    return result


def summarize(artifacts: GroupArtifacts, checkpoint: Path, checkpoint_sha: str, aggregation: bool, env) -> dict:
    successes = sum(int(row["success"]) for row in artifacts.results)
    categories = Counter(row["failure_category"] for row in artifacts.results)
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "condition": artifacts.name,
        "episodes": len(artifacts.results),
        "successes": successes,
        "success_rate": successes / len(artifacts.results),
        "seed": args.seed,
        "prediction_chunk_size_k": CHUNK_SIZE,
        "execution_horizon_h": 1,
        "n_action_steps": 1,
        "policy_steps": args.policy_steps,
        "sim_steps_per_action": args.sim_steps_per_action,
        "policy_calls_per_episode": args.policy_steps,
        "temporal_aggregation": aggregation,
        "temporal_ensemble_coeff": TEMPORAL_COEFF if aggregation else None,
        "temporal_ensemble_implementation": "lerobot.policies.act.modeling_act.ACTTemporalEnsembler.update on raw model chunk before postprocessing",
        "failure_taxonomy": dict(sorted(categories.items())),
        "max_lift_mean_m": sum(row["max_lift_m"] for row in artifacts.results) / len(artifacts.results),
        "action_jitter_mean_l2": sum(row["action_jitter_mean_l2"] for row in artifacts.results) / len(artifacts.results),
        "rollout_wall_time_mean_s": sum(row["rollout_wall_time"] for row in artifacts.results) / len(artifacts.results),
        "policy_call_time_mean_s": sum(row["policy_call_time_mean_s"] for row in artifacts.results) / len(artifacts.results),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "manifest": str(args.manifest.resolve()),
        "initial_observation_refresh_protocol": "identical restore + one no-op action refresh for both conditions",
        "sim_dt_s": float(env.cfg.sim.dt),
        "decimation": int(getattr(env.cfg, "decimation", 1)),
        "claim_boundary": "K=100, H=1, G4 A1-B1 14k checkpoint, paired manifest initializations only",
        "episode_results": artifacts.results,
    }


def main() -> int:
    device = torch.device(args.device)
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

    try:
        generate_manifest(env, args.manifest, 20, args.seed)
        if args.generate_manifest_only:
            return 0

        manifest = load_manifest(args.manifest, args.episodes, args.seed)
        bundle = load_policy(args.checkpoint, device)
        model_path = bundle.checkpoint / "model.safetensors"
        checkpoint_sha_before = sha256_file(model_path)
        command = shlex.join([sys.executable, *sys.argv])
        common_config = {
            "checkpoint": str(bundle.checkpoint),
            "checkpoint_model_file": str(model_path),
            "seed": args.seed,
            "episodes": args.episodes,
            "chunk_size": CHUNK_SIZE,
            "n_action_steps": 1,
            "policy_steps": args.policy_steps,
            "sim_steps_per_action": args.sim_steps_per_action,
            "stable_plate_steps": args.stable_plate_steps,
            "gripper_close_threshold": args.gripper_close_threshold,
            "pair_atol": args.pair_atol,
            "manifest": str(args.manifest.resolve()),
            "training": False,
            "additional_action_smoothing": False,
            "controller_change": False,
        }
        no_dir = args.output_root / "h1_no_aggregation"
        agg_dir = args.output_root / "h1_temporal_aggregation_001"
        no_artifacts = GroupArtifacts.create(
            "h1_no_aggregation", no_dir, {**common_config, "temporal_aggregation": False}, command, checkpoint_sha_before, model_path
        )
        agg_artifacts = GroupArtifacts.create(
            "h1_temporal_aggregation_001", agg_dir,
            {**common_config, "temporal_aggregation": True, "temporal_ensemble_coeff": TEMPORAL_COEFF},
            command, checkpoint_sha_before, model_path,
        )
        no_log = (no_dir / "stdout.log").open("w", encoding="utf-8")
        agg_log = (agg_dir / "stdout.log").open("w", encoding="utf-8")
        sys.stdout = MultiTee(sys.__stdout__, no_log, agg_log)
        sys.stderr = MultiTee(sys.__stderr__, no_log, agg_log)

        pairing_rows = []
        for episode, manifest_row in enumerate(manifest):
            expected_id = manifest_row["initialization_id"]
            observation_a, state_a = restore_and_refresh(env, manifest_row["state"])
            vector_a = state_vector(state_a)
            run_rollout(env, bundle, device, no_artifacts, episode, expected_id, observation_a, state_a, False)

            observation_b, state_b = restore_and_refresh(env, manifest_row["state"])
            vector_b = state_vector(state_b)
            max_abs = max(abs(a - b) for a, b in zip(vector_a, vector_b, strict=True))
            pair_ok = max_abs <= args.pair_atol
            pairing_row = {
                "episode_index": episode,
                "initialization_id": expected_id,
                "max_abs_refreshed_state_difference": max_abs,
                "pair_atol": args.pair_atol,
                "paired": pair_ok,
            }
            pairing_rows.append(pairing_row)
            if not pair_ok:
                json_dump(args.output_root / "PAIRING_FAILED.json", pairing_row)
                raise RuntimeError(f"paired initialization failed at episode {episode}: max_abs={max_abs}")
            run_rollout(env, bundle, device, agg_artifacts, episode, expected_id, observation_b, state_b, True)

        checkpoint_sha_after = sha256_file(model_path)
        if checkpoint_sha_after != checkpoint_sha_before:
            raise RuntimeError("checkpoint SHA256 changed during inference")
        hash_line = f"{checkpoint_sha_after}  {model_path}\n"
        for artifacts, aggregation in ((no_artifacts, False), (agg_artifacts, True)):
            json_dump(artifacts.directory / "summary.json", summarize(artifacts, bundle.checkpoint, checkpoint_sha_after, aggregation, env))
            (artifacts.directory / "checkpoint_sha256_after.txt").write_text(hash_line, encoding="utf-8")
            (artifacts.directory / "COMPLETED").write_text("paired evaluation complete; checkpoint unchanged\n", encoding="utf-8")
        json_dump(args.output_root / "pairing_diagnostics.json", {
            "status": "PAIRING_CONFIRMED",
            "episodes": args.episodes,
            "all_paired": all(row["paired"] for row in pairing_rows),
            "rows": pairing_rows,
        })
        print(f"Completed paired temporal aggregation evaluation: {args.output_root}", flush=True)
        return 0
    finally:
        try:
            if "no_artifacts" in locals():
                no_artifacts.close()
            if "agg_artifacts" in locals():
                agg_artifacts.close()
        finally:
            env.close()
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
