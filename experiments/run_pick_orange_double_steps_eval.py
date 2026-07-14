#!/usr/bin/env python3
"""Parallel, resumable evaluation for the doubled-step PickOrange runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Job:
    name: str
    command: list[str]
    summary: Path
    checkpoints: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("count", type=int, choices=(30, 50))
    parser.add_argument("--max-parallel", type=int, default=int(os.environ.get("EVAL_MAX_PARALLEL", "3")))
    parser.add_argument("--launch-stagger", type=int, default=int(os.environ.get("EVAL_LAUNCH_STAGGER", "45")))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--policy-steps", type=int, default=420)
    return parser.parse_args()


def unique_dir(pattern: str) -> Path:
    matches = sorted((ROOT / "outputs/train").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one training directory for {pattern}, found {matches}")
    return matches[0]


def checkpoint(root: Path, step: int) -> Path:
    path = root / "checkpoints" / f"{step:06d}" / "pretrained_model"
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def validate_required_checkpoints(root: Path, required: set[int]) -> None:
    checkpoint_root = root / "checkpoints"
    actual = {int(path.name) for path in checkpoint_root.iterdir() if path.is_dir() and path.name.isdigit()}
    missing = required - actual
    if missing:
        raise RuntimeError(f"Required checkpoints missing for {root}: {sorted(missing)}")
    extras = actual - required
    if extras:
        print(f"Preserve extra checkpoints (read-only policy): {root}: {sorted(extras)}", flush=True)


def summary_matches(job: Job) -> bool:
    if not job.summary.is_file():
        return False
    try:
        text = job.summary.read_text(encoding="utf-8")
        return all(str(path) in text for path in job.checkpoints)
    except Exception:
        return False


def make_jobs(args: argparse.Namespace) -> tuple[list[Job], list[Job], list[Job], Path]:
    tag = f"gate_exp{args.count}"
    train_tag = f"{tag}_double"
    output_root = ROOT / f"outputs/eval/pick_orange_{tag}_double"
    checkpoint_root = output_root / "a0_a1_checkpoint_eval"
    isolated_root = output_root / "a1_isolated_014000_policy420"
    legacy_isolated_root = output_root / "a1_isolated_legacy007000_policy420"
    log_root = ROOT / f"outputs/logs/pick_orange_{tag}_double/eval"
    for path in (checkpoint_root, isolated_root, legacy_isolated_root, log_root):
        path.mkdir(parents=True, exist_ok=True)

    a0_root = unique_dir(f"{train_tag}_a0_full_act_ch100_b64_s42000_train*_noval")
    a1_roots = {
        orange: unique_dir(
            f"{train_tag}_a1_prefixstrict_orange{orange}_act_ch100_b64_s14000_train*_noval"
        )
        for orange in ("001", "002", "003")
    }
    validate_required_checkpoints(a0_root, {30000, 36000, 42000})
    for root in a1_roots.values():
        validate_required_checkpoints(root, {10000, 12000, 14000})
    python = sys.executable
    common = [
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--action_horizon",
        "100",
        "--policy_steps_per_orange",
        str(args.policy_steps),
        "--stable_plate_steps",
        "10",
        "--device",
        "cuda",
        "--headless",
    ]

    checkpoint_jobs: list[Job] = []
    if args.count == 30:
        legacy_a0_root = unique_dir("gate_exp30_a0_full_act_ch100_b64_s21000_train30_noval")
        legacy_a1_roots = {
            orange: unique_dir(f"gate_exp30_a1_orange{orange}_act_ch100_b64_s7000_train*_noval")
            for orange in ("001", "002", "003")
        }
        legacy_a0 = checkpoint(legacy_a0_root, 21000)
        legacy_a0_output = checkpoint_root / "a0_legacy21k"
        checkpoint_jobs.append(
            Job(
                "a0_legacy21k",
                [
                    python,
                    "experiments/eval_pick_orange_joint6_ablation.py",
                    "--group",
                    "a0",
                    *common,
                    "--output_dir",
                    str(legacy_a0_output),
                    "--checkpoint",
                    str(legacy_a0),
                ],
                legacy_a0_output / "summary.json",
                (legacy_a0,),
            )
        )
        legacy_a1 = tuple(checkpoint(legacy_a1_roots[orange], 7000) for orange in ("001", "002", "003"))
        legacy_a1_output = checkpoint_root / "a1_legacy7k"
        checkpoint_jobs.append(
            Job(
                "a1_legacy7k",
                [
                    python,
                    "experiments/eval_pick_orange_joint6_ablation.py",
                    "--group",
                    "a1",
                    *common,
                    "--output_dir",
                    str(legacy_a1_output),
                    "--checkpoint_001",
                    str(legacy_a1[0]),
                    "--checkpoint_002",
                    str(legacy_a1[1]),
                    "--checkpoint_003",
                    str(legacy_a1[2]),
                ],
                legacy_a1_output / "summary.json",
                legacy_a1,
            )
        )
    for label, a0_step, a1_step in (("s30k", 30000, 10000), ("s36k", 36000, 12000), ("s42k", 42000, 14000)):
        a0_ckpt = checkpoint(a0_root, a0_step)
        a0_output = checkpoint_root / f"a0_{label}"
        checkpoint_jobs.append(
            Job(
                f"a0_{label}",
                [
                    python,
                    "experiments/eval_pick_orange_joint6_ablation.py",
                    "--group",
                    "a0",
                    *common,
                    "--output_dir",
                    str(a0_output),
                    "--checkpoint",
                    str(a0_ckpt),
                ],
                a0_output / "summary.json",
                (a0_ckpt,),
            )
        )
        a1_ckpts = tuple(checkpoint(a1_roots[orange], a1_step) for orange in ("001", "002", "003"))
        a1_output = checkpoint_root / f"a1_{label}"
        checkpoint_jobs.append(
            Job(
                f"a1_{label}",
                [
                    python,
                    "experiments/eval_pick_orange_joint6_ablation.py",
                    "--group",
                    "a1",
                    *common,
                    "--output_dir",
                    str(a1_output),
                    "--checkpoint_001",
                    str(a1_ckpts[0]),
                    "--checkpoint_002",
                    str(a1_ckpts[1]),
                    "--checkpoint_003",
                    str(a1_ckpts[2]),
                ],
                a1_output / "summary.json",
                a1_ckpts,
            )
        )

    final_jobs: list[Job] = []
    a0_final = checkpoint(a0_root, 42000)
    a0_output = checkpoint_root / "a0_final_video"
    final_jobs.append(
        Job(
            "a0_final_video",
            [
                python,
                "experiments/eval_pick_orange_joint6_ablation.py",
                "--group",
                "a0",
                "--episodes",
                "1",
                "--seed",
                str(args.seed),
                "--action_horizon",
                "100",
                "--policy_steps_per_orange",
                str(args.policy_steps),
                "--stable_plate_steps",
                "10",
                "--device",
                "cuda",
                "--headless",
                "--record_video",
                "--output_dir",
                str(a0_output),
                "--checkpoint",
                str(a0_final),
            ],
            a0_output / "summary.json",
            (a0_final,),
        )
    )
    a1_final = tuple(checkpoint(a1_roots[orange], 14000) for orange in ("001", "002", "003"))
    a1_output = checkpoint_root / "a1_final_video"
    final_jobs.append(
        Job(
            "a1_final_video",
            [
                python,
                "experiments/eval_pick_orange_joint6_ablation.py",
                "--group",
                "a1",
                "--episodes",
                "1",
                "--seed",
                str(args.seed),
                "--action_horizon",
                "100",
                "--policy_steps_per_orange",
                str(args.policy_steps),
                "--stable_plate_steps",
                "10",
                "--device",
                "cuda",
                "--headless",
                "--record_video",
                "--output_dir",
                str(a1_output),
                "--checkpoint_001",
                str(a1_final[0]),
                "--checkpoint_002",
                str(a1_final[1]),
                "--checkpoint_003",
                str(a1_final[2]),
            ],
            a1_output / "summary.json",
            a1_final,
        )
    )

    isolated_jobs: list[Job] = []
    for phase, orange in (("b1", "001"), ("b2", "002"), ("b3", "003")):
        ckpt = checkpoint(a1_roots[orange], 14000)
        dataset = ROOT / (
            f"data/lerobot/local/so101_pick_orange_{tag}_a1_event_prefixstrict_"
            f"orange{orange}_joint6_v0/data/chunk-000/file-000.parquet"
        )
        if not dataset.is_file():
            raise FileNotFoundError(dataset)
        output = isolated_root / phase
        isolated_jobs.append(
            Job(
                phase,
                [
                    python,
                    "experiments/eval_pick_orange_a1_isolated.py",
                    "--phase",
                    phase,
                    "--checkpoint",
                    str(ckpt),
                    "--phase_dataset",
                    str(dataset),
                    "--episodes",
                    str(args.episodes),
                    "--seed",
                    str(args.seed),
                    "--action_horizon",
                    "100",
                    "--policy_steps",
                    str(args.policy_steps),
                    "--stable_plate_steps",
                    "10",
                    "--device",
                    "cuda",
                    "--headless",
                    "--output_dir",
                    str(output),
                ],
                output / "summary.json",
                (ckpt,),
            )
        )
    if args.count == 30:
        legacy_a1_roots = {
            orange: unique_dir(f"gate_exp30_a1_orange{orange}_act_ch100_b64_s7000_train*_noval")
            for orange in ("001", "002", "003")
        }
        for phase, orange in (("b1", "001"), ("b2", "002"), ("b3", "003")):
            ckpt = checkpoint(legacy_a1_roots[orange], 7000)
            dataset = ROOT / (
                f"data/lerobot/local/so101_pick_orange_{tag}_a1_event_prefixstrict_"
                f"orange{orange}_joint6_v0/data/chunk-000/file-000.parquet"
            )
            output = legacy_isolated_root / phase
            isolated_jobs.append(
                Job(
                    f"legacy_{phase}",
                    [
                        python,
                        "experiments/eval_pick_orange_a1_isolated.py",
                        "--phase",
                        phase,
                        "--checkpoint",
                        str(ckpt),
                        "--phase_dataset",
                        str(dataset),
                        "--episodes",
                        str(args.episodes),
                        "--seed",
                        str(args.seed),
                        "--action_horizon",
                        "100",
                        "--policy_steps",
                        str(args.policy_steps),
                        "--stable_plate_steps",
                        "10",
                        "--device",
                        "cuda",
                        "--headless",
                        "--output_dir",
                        str(output),
                    ],
                    output / "summary.json",
                    (ckpt,),
                )
            )
    reference_path = ROOT / f"outputs/audits/pick_orange_{tag}_effective_data/stage_start_references.json"
    for job in (*checkpoint_jobs, *final_jobs):
        group = job.command[job.command.index("--group") + 1]
        job.command += ["--dataset_label", tag, "--strategy_label", group, "--checkpoint_label", job.name]
        if group == "a1" and reference_path.is_file():
            job.command += ["--stage_reference_json", str(reference_path)]
    for job in isolated_jobs:
        job.command += ["--dataset_label", tag, "--strategy_label", "a1_isolated_oracle_init", "--checkpoint_label", job.name]
    return checkpoint_jobs, final_jobs, isolated_jobs, output_root


def tail(path: Path, limit: int = 80) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-limit:])
    except Exception as exc:
        return f"Unable to read {path}: {exc}"


def run_pool(jobs: list[Job], max_parallel: int, stagger: int, log_root: Path) -> None:
    pending = [job for job in jobs if not summary_matches(job)]
    for job in jobs:
        if summary_matches(job):
            print(f"Skip completed {job.name}", flush=True)
    active: dict[subprocess.Popen, tuple[Job, object, Path]] = {}
    failures: list[tuple[Job, int, Path]] = []
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    while pending or active:
        while pending and len(active) < max_parallel:
            job = pending.pop(0)
            log_path = log_root / f"{job.name}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            log_handle.write(f"\n===== launch {time.strftime('%F %T')} =====\n{' '.join(job.command)}\n")
            log_handle.flush()
            print(f"Launch {job.name}; active={len(active) + 1}/{max_parallel}", flush=True)
            process = subprocess.Popen(
                job.command,
                cwd=ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active[process] = (job, log_handle, log_path)
            if pending and len(active) < max_parallel and stagger:
                time.sleep(stagger)

        time.sleep(10)
        for process, (job, handle, log_path) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            handle.close()
            del active[process]
            if status == 0 and summary_matches(job):
                print(f"Complete {job.name}", flush=True)
            else:
                failures.append((job, status, log_path))
                print(f"FAILED {job.name} exit={status}\n{tail(log_path)}", flush=True)
    if failures:
        names = ", ".join(f"{job.name}(exit={status})" for job, status, _ in failures)
        raise RuntimeError(f"Evaluation jobs failed: {names}")


def main() -> None:
    args = parse_args()
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")
    if args.count == 30 and os.environ.get("FRONTLOAD_50_BEFORE_EVAL30", "1") == "1":
        print(
            "Front-load requested: complete 50-demo collection/gates/training before 30-demo evaluation.",
            flush=True,
        )
        subprocess.run(
            [sys.executable, "experiments/frontload_pick_orange_50.py"],
            cwd=ROOT,
            env=os.environ.copy(),
            check=True,
        )
    subprocess.run(
        [sys.executable, "experiments/audit_pick_orange_effective_data.py", "--expert-count", str(args.count)],
        cwd=ROOT,
        env=os.environ.copy(),
        check=True,
    )
    checkpoint_jobs, final_jobs, isolated_jobs, output_root = make_jobs(args)
    log_root = ROOT / f"outputs/logs/pick_orange_gate_exp{args.count}_double/eval"
    run_pool(checkpoint_jobs, args.max_parallel, args.launch_stagger, log_root)
    run_pool(final_jobs, min(args.max_parallel, 2), args.launch_stagger, log_root)
    run_pool(isolated_jobs, args.max_parallel, args.launch_stagger, log_root)
    marker = output_root / "DONE"
    marker.write_text(
        json.dumps(
            {
                "count": args.count,
                "batch_size": 64,
                "a0_steps": 42000,
                "a1_steps": 14000,
                "isolated_policy_steps": args.policy_steps,
                "episodes": args.episodes,
                "seed": args.seed,
                "max_parallel": args.max_parallel,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Evaluation complete: {marker}")


if __name__ == "__main__":
    main()
