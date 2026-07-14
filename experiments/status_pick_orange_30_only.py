#!/usr/bin/env python3
"""Live Markdown/JSON status for the PickOrange 30-demo-only evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "outputs/pipeline/pick_orange_30_only"
EVAL = ROOT / "outputs/eval/pick_orange_gate_exp30_double"
LOGS = ROOT / "outputs/logs/pick_orange_gate_exp30_double/eval"
EPISODE_RE = re.compile(r"Episode\s+(\d+)/(\d+):")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=int, default=0)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def command(command: list[str]) -> str:
    try:
        return subprocess.run(command, text=True, capture_output=True, timeout=10, check=False).stdout.strip()
    except Exception:
        return ""


def log_progress(path: Path, expected: int) -> tuple[int, int]:
    if not path.is_file():
        return 0, expected
    completed = 0
    total = expected
    try:
        with path.open("r", errors="replace") as stream:
            for line in stream:
                match = EPISODE_RE.search(line)
                if match:
                    completed = max(completed, int(match.group(1)))
                    total = int(match.group(2))
    except Exception:
        pass
    return completed, total


def job(name: str, summary: Path, expected: int) -> dict:
    log = LOGS / f"{name}.log"
    payload = read_json(summary)
    if payload:
        completed = int(payload.get("episodes", expected))
        successes = payload.get("successes")
        status = "DONE"
    else:
        completed, expected = log_progress(log, expected)
        successes = None
        status = "RUN" if name in command(["pgrep", "-af", "eval_pick_orange"]) else "WAIT"
        if completed and status == "WAIT":
            status = "QUEUED/RESUME"
    return {
        "name": name,
        "completed": completed,
        "total": expected,
        "successes": successes,
        "status": status,
        "summary": str(summary),
        "log": str(log),
    }


def collect() -> dict:
    full_root = EVAL / "a0_a1_checkpoint_eval"
    full = [
        job("a0_legacy21k", full_root / "a0_legacy21k/summary.json", 20),
        job("a1_legacy7k", full_root / "a1_legacy7k/summary.json", 20),
    ]
    for label in ("s30k", "s36k", "s42k"):
        for group in ("a0", "a1"):
            name = f"{group}_{label}"
            full.append(job(name, full_root / name / "summary.json", 20))
    final_video = [
        job("a0_final_video", full_root / "a0_final_video/summary.json", 1),
        job("a1_final_video", full_root / "a1_final_video/summary.json", 1),
    ]
    isolated = []
    for phase in ("b1", "b2", "b3"):
        isolated.append(job(phase, EVAL / f"a1_isolated_014000_policy420/{phase}/summary.json", 20))
    for phase in ("b1", "b2", "b3"):
        isolated.append(job(f"legacy_{phase}", EVAL / f"a1_isolated_legacy007000_policy420/{phase}/summary.json", 20))
    disk = shutil.disk_usage(ROOT)
    return {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pipeline": read_json(PIPELINE / "state.json"),
        "tmux_running": "po_30_only:" in command(["tmux", "list-sessions"]),
        "active_processes": command(["pgrep", "-af", "eval_pick_orange"]),
        "full": full,
        "final_video": final_video,
        "isolated": isolated,
        "eval_done": (EVAL / "DONE").is_file(),
        "report_done": (ROOT / "outputs/reports/pick_orange_30_only/DONE").is_file(),
        "all_done": (PIPELINE / "ALL_DONE").is_file(),
        "gpu": command(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"]),
        "disk_free_gb": disk.free / 1024**3,
    }


def table(lines: list[str], title: str, rows: list[dict]) -> None:
    lines += ["", f"## {title}", "", "| Job | Episode | Result | Status |", "|---|---:|---:|---|"]
    for row in rows:
        result = f"{row['successes']}/{row['total']}" if row["successes"] is not None else "-"
        lines.append(f"| {row['name']} | {row['completed']}/{row['total']} | {result} | {row['status']} |")


def markdown(data: dict) -> str:
    state = data["pipeline"]
    lines = [
        "# PickOrange 30-Only Live Status", "",
        f"Last update: `{data['updated_at']}`", "",
        "- 50-demo extension: **CANCELLED**",
        f"- tmux `po_30_only`: **{'RUNNING' if data['tmux_running'] else 'STOPPED'}**",
        f"- Pipeline stage: **{state.get('stage', '-')}**",
        f"- Pipeline status: **{state.get('status', 'unknown')}**",
        f"- Evaluation parallelism: **{state.get('eval_parallel', '-')}**",
        f"- Attempt: **{state.get('attempt', '-')}**",
    ]
    table(lines, "Full-task checkpoint evaluation", data["full"])
    table(lines, "Final checkpoint videos", data["final_video"])
    table(lines, "Isolated B1/B2/B3 evaluation", data["isolated"])
    active = data["active_processes"].replace("\n", "<br>") or "none"
    lines += [
        "", "## Resources", "",
        f"- GPU: `{data['gpu']}`",
        f"- Free disk: `{data['disk_free_gb']:.1f} GB`",
        f"- Evaluation DONE: **{'YES' if data['eval_done'] else 'NO'}**",
        f"- 30-only report DONE: **{'YES' if data['report_done'] else 'NO'}**",
        f"- Entire 30-only pipeline: **{'DONE' if data['all_done'] else 'RUNNING'}**",
        "", "## Active evaluator processes", "", active, "",
        "This file is generated by `status_pick_orange_30_only.py` every 20 seconds.",
    ]
    return "\n".join(lines) + "\n"


def atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    options = args()
    while True:
        data = collect()
        atomic(options.json, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        atomic(options.markdown, markdown(data))
        if options.watch <= 0:
            return
        time.sleep(options.watch)


if __name__ == "__main__":
    main()
