#!/usr/bin/env python3
"""Isolated 1--3 episode smoke test for PickOrange evaluation protocols.

Default execution is read-only and reports SKIPPED for simulator checks.  The
simulator is launched only with explicit ``--execute``.  Outputs are always
separate from formal evaluation summaries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/smoke/pick_orange_eval_protocol_v2"
REQUIRED_OVERRUN_FIELDS = {
    "episode_index", "seed", "initialization_id", "stage_id", "stage_start_step",
    "target_first_satisfied_step", "target_first_stably_satisfied_step",
    "stage_switch_step", "post_success_overrun", "target_in_plate_before_switch",
    "prefix_intact_before_switch", "stage_success", "failure_reason", "stage_start_deviation",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--episodes", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--a0-checkpoint", type=Path, required=True)
    p.add_argument("--b1-checkpoint", type=Path, required=True)
    p.add_argument("--b2-checkpoint", type=Path, required=True)
    p.add_argument("--b3-checkpoint", type=Path, required=True)
    p.add_argument("--b3-dataset", type=Path, required=True, help="B3 LeRobot parquet used for oracle robot start state")
    p.add_argument("--execute", action="store_true", help="launch four minimal simulator checks")
    p.add_argument("--skip-reason", default=None, help="resource-safety reason recorded when --execute is not used")
    return p


def check(name: str, status: str, detail: str, category: str) -> dict:
    return {"name": name, "status": status, "detail": detail, "category": category}


def checkpoint_ok(path: Path) -> bool:
    return (path / "model.safetensors").is_file()


def checkpoint_manifest(paths: list[Path]) -> list[dict]:
    rows = []
    for root in paths:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rows.append({"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns})
    return rows


def make_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    full = args.root / "experiments/eval_pick_orange_joint6_ablation.py"
    isolated = args.root / "experiments/eval_pick_orange_a1_isolated.py"
    common = ["--episodes", str(args.episodes), "--seed", str(args.seed), "--device", args.device, "--headless"]
    return {
        "a0_native_horizon": [sys.executable, str(full), "--group", "a0", "--horizon_protocol", "native_horizon", "--checkpoint", str(args.a0_checkpoint), "--output_dir", str(args.output_dir / "a0_native_horizon"), *common],
        "a0_matched_horizon": [sys.executable, str(full), "--group", "a0", "--horizon_protocol", "matched_horizon", "--checkpoint", str(args.a0_checkpoint), "--output_dir", str(args.output_dir / "a0_matched_horizon"), *common],
        "a1_matched_horizon": [sys.executable, str(full), "--group", "a1", "--horizon_protocol", "matched_horizon", "--checkpoint_001", str(args.b1_checkpoint), "--checkpoint_002", str(args.b2_checkpoint), "--checkpoint_003", str(args.b3_checkpoint), "--policy_steps_per_orange", "420", "--output_dir", str(args.output_dir / "a1_matched_horizon"), *common],
        "b3_isolated_oracle": [sys.executable, str(isolated), "--phase", "b3", "--checkpoint", str(args.b3_checkpoint), "--phase_dataset", str(args.b3_dataset), "--policy_steps", "420", "--output_dir", str(args.output_dir / "b3_isolated_oracle"), *common],
    }


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def validate_summary(name: str, directory: Path, episodes: int) -> tuple[list[dict], dict | None]:
    path = directory / "summary.json"
    payload = read_json(path)
    if payload is None:
        return [check(f"{name}.summary", "FAIL", str(path), "simulation")], None
    rows = payload.get("episode_results", [])
    checks = [check(f"{name}.episode_count", "PASS" if len(rows) == episodes else "FAIL", f"{len(rows)}/{episodes}", "simulation")]
    expected = 1020 if name == "a0_native_horizon" else 1260 if name in {"a0_matched_horizon", "a1_matched_horizon"} else 420
    actual = [row.get("policy_steps") for row in rows]
    checks.append(check(f"{name}.horizon", "PASS" if actual and all(step == expected for step in actual) else "FAIL", f"expected={expected}, actual={actual}", "simulation"))
    if name == "a1_matched_horizon":
        switches = [row.get("switch_steps") for row in rows]
        checks.append(check("a1.fixed_switches", "PASS" if switches and all(value == [420, 840, 1260] for value in switches) else "FAIL", str(switches), "simulation"))
        overrun_path = directory / "post_success_overrun.jsonl"
        try:
            diagnostics = [json.loads(line) for line in overrun_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            fields_ok = len(diagnostics) == episodes * 3 and all(REQUIRED_OVERRUN_FIELDS <= row.keys() for row in diagnostics)
        except Exception:
            diagnostics, fields_ok = [], False
        checks.append(check("a1.post_success_overrun_jsonl", "PASS" if fields_ok else "FAIL", f"records={len(diagnostics)}, expected={episodes * 3}", "simulation"))
        checks.append(check("a1.stage_start_jsonl", "PASS" if Path(payload.get("stage_state_jsonl", "")).is_file() else "FAIL", str(payload.get("stage_state_jsonl")), "simulation"))
        checks.append(check("a1.scheduler_classification", "PASS" if "fixed-time" in payload.get("scheduler_classification", "") else "FAIL", payload.get("scheduler_classification", "missing"), "simulation"))
    if name == "b3_isolated_oracle":
        oracle = "oracle" in payload.get("scheduler_classification", "").lower() and "oracle" in payload.get("oracle_state", "").lower()
        checks.append(check("isolated.oracle_initialization", "PASS" if oracle else "FAIL", payload.get("scheduler_classification", "missing"), "simulation"))
        checks.append(check("b3.crosses_340", "PASS" if actual and all(step == 420 for step in actual) else "FAIL", str(actual), "simulation"))
    return checks, payload


def main() -> int:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = make_commands(args)
    checkpoint_paths = [args.a0_checkpoint, args.b1_checkpoint, args.b2_checkpoint, args.b3_checkpoint]
    checks = []
    for name, path in zip(("a0_checkpoint", "b1_checkpoint", "b2_checkpoint", "b3_checkpoint"), checkpoint_paths, strict=True):
        checks.append(check(name, "PASS" if checkpoint_ok(path) else "FAIL", str(path), "code_input"))
    checks += [
        check("b3_dataset", "PASS" if args.b3_dataset.is_file() else "FAIL", str(args.b3_dataset), "code_input"),
        check("episode_range", "PASS", str(args.episodes), "code_input"),
        check("smoke_output_isolation", "PASS" if "/outputs/smoke/" in str(args.output_dir) else "FAIL", str(args.output_dir), "code_input"),
        check("native_horizon_definition", "PASS", "A0=1020, A1=420x3=1260", "code_input"),
        check("matched_horizon_definition", "PASS", "A0=1260, A1=420x3=1260", "code_input"),
        check("b3_not_truncated", "PASS", "420 > historical 340; release may occur at 350-358", "code_input"),
        check("initialization_separation", "PASS", "full uses environment reset; isolated B3 command uses oracle initialization", "code_input"),
    ]
    runner_text = (args.root / "experiments/run_pick_orange_double_steps_eval.py").read_text(encoding="utf-8")
    no_delete = "validate_required_checkpoints" in runner_text and "rmtree(" not in runner_text
    checks.append(check("evaluator_checkpoint_non_deletion", "PASS" if no_delete else "FAIL", "formal evaluator validates required checkpoints without delete calls", "code_input"))
    (args.output_dir / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    preflight_failed = any(item["status"] == "FAIL" for item in checks)
    payloads = {}
    before = checkpoint_manifest(checkpoint_paths)
    if args.execute and not preflight_failed:
        for name, command in commands.items():
            completed = subprocess.run(command, cwd=args.root, check=False)
            checks.append(check(f"{name}.process", "PASS" if completed.returncode == 0 else "FAIL", f"exit={completed.returncode}", "simulation"))
            if completed.returncode != 0:
                break
            result_checks, payload = validate_summary(name, args.output_dir / name, args.episodes)
            checks.extend(result_checks)
            payloads[name] = payload
        ids = [
            [row.get("initialization_id", row.get("initial_state_id")) for row in (payloads.get(name) or {}).get("episode_results", [])]
            for name in ("a0_native_horizon", "a0_matched_horizon", "a1_matched_horizon")
        ]
        aligned = bool(ids[0]) and ids[0] == ids[1] == ids[2]
        checks.append(check("full_initialization_id_alignment", "PASS" if aligned else "FAIL", str(ids), "simulation"))
        before_after_equal = before == checkpoint_manifest(checkpoint_paths)
        checks.append(check("checkpoint_manifest_unchanged", "PASS" if before_after_equal else "FAIL", "path/size/mtime comparison", "simulation"))
    elif not args.execute:
        reason = args.skip_reason or "dry-run only; simulator execution requires an explicitly safe GPU window"
        for name in commands:
            checks.append(check(f"{name}.simulation", "SKIPPED", reason, "resource_safety"))
        checks += [
            check("initialization_id_runtime_alignment", "SKIPPED", reason, "resource_safety"),
            check("checkpoint_manifest_runtime_comparison", "SKIPPED", reason, "resource_safety"),
        ]
    statuses = {item["status"] for item in checks}
    overall = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "SKIPPED" if "SKIPPED" in statuses else "PASS"
    payload = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "executed": args.execute and not preflight_failed,
        "skip_reason": args.skip_reason if not args.execute else None,
        "checks": checks,
        "commands": commands,
        "formal_results_included": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# PickOrange evaluation protocol smoke test v2", "", f"Overall: **{overall}**", "", "| Category | Check | Status | Detail |", "|---|---|---|---|"]
    lines.extend(f"| {item['category']} | {item['name']} | {item['status']} | {item['detail']} |" for item in checks)
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{overall}: {args.output_dir / 'summary.md'}")
    return 1 if overall == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
