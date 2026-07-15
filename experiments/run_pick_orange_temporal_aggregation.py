#!/usr/bin/env python3
"""Smoke-gated orchestration for the paired B1 temporal aggregation experiment.

Sequence:
1. strict safety preflight;
2. create/reuse a 20-state manifest;
3. paired 1-episode smoke;
4. paired 3-episode smoke;
5. paired 20-episode formal comparison;
6. dependency-light analysis and incremental tar.gz archive.

The runner stops on any failed smoke, pairing mismatch, active GPU process, or
checkpoint hash change.  It never trains and never deletes existing data.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORMAL_ROOT = ROOT / "outputs/eval/pick_orange_temporal_aggregation"
DEFAULT_SMOKE_ROOT = ROOT / "outputs/smoke/pick_orange_temporal_aggregation"
DEFAULT_ARCHIVE_ROOT = ROOT / "outputs/archive"
FORBIDDEN_ROOT = Path("/workspace/Octopus-SingleArm")
CONDITIONS = ("h1_no_aggregation", "h1_temporal_aggregation_001")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="G4 A1-B1 14k pretrained_model directory")
    p.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    p.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    p.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--pair-atol", type=float, default=1e-6)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--deletion-pause-confirmed", action="store_true")
    return p


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def fail(message: str) -> None:
    raise RuntimeError(message)


def gpu_processes() -> list[str]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        fail(f"cannot verify GPU occupancy: {completed.stderr.strip()}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def destructive_processes() -> list[str]:
    completed = subprocess.run(["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        fail("cannot inspect server processes")
    needles = ("rm -rf", "find /", "-delete", "checkpoint-prune", "cleanup_outputs", "server_delete")
    rows = []
    for line in completed.stdout.splitlines():
        if str(os.getpid()) in line:
            continue
        if any(token in line for token in needles):
            rows.append(line.strip())
    return rows


def preflight(args: argparse.Namespace) -> dict:
    if not args.execute:
        fail("dry-run only: pass --execute after reviewing paths and resource safety")
    if not args.deletion_pause_confirmed:
        fail("server deletion flow must be paused; then pass --deletion-pause-confirmed")
    for path in (Path.cwd(), args.checkpoint, args.formal_root, args.smoke_root, args.archive_root):
        if is_under(path, FORBIDDEN_ROOT):
            fail(f"protected path refusal: {path}")
    model = args.checkpoint / "model.safetensors"
    if not model.is_file():
        fail(f"checkpoint missing: {model}")
    active_gpu = gpu_processes() if args.device.startswith("cuda") else []
    if active_gpu:
        fail("GPU occupied; protected-task safety stop: " + " | ".join(active_gpu))
    destructive = destructive_processes()
    if destructive:
        fail("destructive/server-deletion process still detected; do not run: " + " | ".join(destructive))
    if args.formal_root.exists():
        existing_groups = [str(args.formal_root / name) for name in CONDITIONS if (args.formal_root / name).exists()]
        if existing_groups:
            fail("formal condition output already exists; refusing overwrite: " + ", ".join(existing_groups))
    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "formal_root": str(args.formal_root.resolve()),
        "smoke_root": str(args.smoke_root.resolve()),
        "device": args.device,
        "gpu_idle": True,
        "deletion_pause_confirmed": True,
        "protected_path_untouched": str(FORBIDDEN_ROOT),
    }


def run(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("$ " + shlex.join(command) + "\n")
        stream.flush()
        completed = subprocess.run(command, cwd=cwd, text=True, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        fail(f"command failed exit={completed.returncode}; see {log_path}")


def evaluator_command(args: argparse.Namespace, output_root: Path, manifest: Path, episodes: int, manifest_only: bool = False) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "experiments/eval_pick_orange_temporal_aggregation.py"),
        "--checkpoint", str(args.checkpoint),
        "--output-root", str(output_root),
        "--manifest", str(manifest),
        "--episodes", str(episodes),
        "--seed", str(args.seed),
        "--policy-steps", "420",
        "--sim-steps-per-action", "2",
        "--pair-atol", str(args.pair_atol),
        "--device", args.device,
        "--headless",
        "--deletion-pause-confirmed",
    ]
    if manifest_only:
        command.append("--generate-manifest-only")
    return command


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_run(root: Path, episodes: int, formal: bool) -> dict:
    checks = []
    initial_ids = []
    for condition in CONDITIONS:
        directory = root / condition
        required = {
            "summary.json", "episode_results.jsonl", "initial_states.jsonl", "policy_calls.jsonl",
            "aggregation_diagnostics.jsonl", "run_config.json", "command.txt", "stdout.log",
            "checkpoint_sha256_before.txt", "checkpoint_sha256_after.txt", "COMPLETED",
        }
        missing = sorted(name for name in required if not (directory / name).exists())
        if missing:
            fail(f"{condition} missing outputs: {missing}")
        before = (directory / "checkpoint_sha256_before.txt").read_text(encoding="utf-8")
        after = (directory / "checkpoint_sha256_after.txt").read_text(encoding="utf-8")
        if before != after:
            fail(f"checkpoint hash changed in {condition}")
        episodes_rows = read_jsonl(directory / "episode_results.jsonl")
        calls = read_jsonl(directory / "policy_calls.jsonl")
        diagnostics = read_jsonl(directory / "aggregation_diagnostics.jsonl")
        initials = read_jsonl(directory / "initial_states.jsonl")
        if len(episodes_rows) != episodes:
            fail(f"{condition}: episode count {len(episodes_rows)} != {episodes}")
        if len(calls) != episodes * 420 or len(diagnostics) != episodes * 420:
            fail(f"{condition}: expected {episodes * 420} calls/diagnostics")
        if any(int(row["policy_call_count"]) != 420 for row in episodes_rows):
            fail(f"{condition}: not every episode has 420 policy calls")
        if any(int(row["policy_call_index"]) != int(row["step"]) for row in calls):
            fail(f"{condition}: per-step policy call sequence invalid")
        if any(int(row["observation_refresh_serial"]) != int(row["step"]) for row in calls):
            fail(f"{condition}: observation refresh serial invalid")
        if condition == CONDITIONS[0]:
            if any(int(row["ensemble_size_per_step"]) != 1 for row in diagnostics):
                fail("no-aggregation smoke used an ensemble")
            if any(float(row["aggregation_action_difference_l2"]) > 1e-12 for row in diagnostics):
                fail("no-aggregation selected action differs from latest chunk action")
        else:
            expected = [min(int(row["step"]) + 1, 100) for row in diagnostics]
            actual = [int(row["ensemble_size_per_step"]) for row in diagnostics]
            if actual != expected:
                fail("temporal aggregation overlap pattern invalid")
            first_rows = [row for row in diagnostics if int(row["step"]) == 0]
            if len(first_rows) != episodes or any(int(row["ensemble_size_per_step"]) != 1 for row in first_rows):
                fail("temporal ensemble was not reset at every episode")
            if not any(float(row["aggregation_action_difference_l2"]) > 1e-12 for row in diagnostics if int(row["step"]) > 0):
                fail("aggregation never differed from the latest prediction")
        initial_ids.append([row["initialization_id"] for row in initials])
        checks.append({
            "condition": condition,
            "episodes": len(episodes_rows),
            "policy_calls": len(calls),
            "completed": True,
            "checkpoint_unchanged": True,
        })
    if initial_ids[0] != initial_ids[1]:
        fail("condition initialization IDs are not paired")
    pairing = json.loads((root / "pairing_diagnostics.json").read_text(encoding="utf-8"))
    if not pairing.get("all_paired") or pairing.get("episodes") != episodes:
        fail("numeric refreshed-state pairing failed")
    if formal and episodes != 20:
        fail("formal validation requires 20 episodes")
    return {"status": "PASS", "episodes": episodes, "conditions": checks, "paired": True}


def archive_results(formal_root: Path, archive_root: Path) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = archive_root / f"pick_orange_temporal_aggregation_{stamp}.tar.gz"
    with tarfile.open(archive_path, "x:gz") as archive:
        archive.add(formal_root, arcname=formal_root.name, recursive=True)
    return archive_path


def main() -> int:
    args = parser().parse_args()
    manifest = args.manifest or (args.formal_root / "initial_state_manifest.jsonl")
    args.formal_root.mkdir(parents=True, exist_ok=True)
    safety = preflight(args)
    lock = args.formal_root / "DO_NOT_DELETE.lock"
    lock.write_text(json.dumps({**safety, "pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
    status = {"schema_version": 1, "status": "RUNNING", "safety": safety, "steps": []}
    (args.formal_root / "orchestration_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    try:
        manifest_command = evaluator_command(args, args.formal_root, manifest, 20, manifest_only=True)
        run(manifest_command, ROOT, args.formal_root / "manifest_generation.log")
        status["steps"].append({"name": "manifest", "status": "PASS", "path": str(manifest.resolve())})

        for episodes in (1, 3):
            smoke_output = args.smoke_root / f"episodes_{episodes}"
            if smoke_output.exists() and any(smoke_output.iterdir()):
                fail(f"refusing to overwrite smoke output: {smoke_output}")
            run(evaluator_command(args, smoke_output, manifest, episodes), ROOT, smoke_output / "orchestrator.log")
            validation = validate_run(smoke_output, episodes, formal=False)
            (smoke_output / "SMOKE_PASS.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
            status["steps"].append({"name": f"smoke_{episodes}", "status": "PASS", "output": str(smoke_output.resolve())})

        if gpu_processes():
            fail("GPU became occupied after smoke; formal run stopped")
        run(evaluator_command(args, args.formal_root, manifest, 20), ROOT, args.formal_root / "formal_orchestrator.log")
        formal_validation = validate_run(args.formal_root, 20, formal=True)
        status["steps"].append({"name": "formal_20", "status": "PASS", "validation": formal_validation})

        analysis_command = [
            sys.executable,
            str(ROOT / "experiments/analyze_pick_orange_temporal_aggregation.py"),
            "--root", str(args.formal_root),
            "--seed", str(args.seed),
        ]
        run(analysis_command, ROOT, args.formal_root / "analysis.log")
        status["steps"].append({"name": "analysis", "status": "PASS", "report": str((args.formal_root / "REPORT.md").resolve())})

        archive_path = archive_results(args.formal_root, args.archive_root)
        status["steps"].append({"name": "incremental_archive", "status": "PASS", "path": str(archive_path.resolve())})
        status["status"] = "COMPLETE"
        status["incremental_archive"] = str(archive_path.resolve())
        (args.formal_root / "orchestration_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(status, indent=2))
        return 0
    except Exception as exc:
        status["status"] = "FAILED"
        status["error"] = str(exc)
        (args.formal_root / "orchestration_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        lock.write_text(lock.read_text(encoding="utf-8") + "finished_at=" + datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
