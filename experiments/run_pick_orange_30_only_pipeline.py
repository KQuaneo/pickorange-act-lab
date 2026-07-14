#!/usr/bin/env python3
"""Persistent, resumable 30-demo evaluation and report pipeline.

This intentionally disables the historical 50-demo frontload.  Completed
evaluation summaries are reused by the underlying evaluator on every retry.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs/pipeline/pick_orange_30_only"
STATE = RUN_ROOT / "state.json"
LOG = ROOT / "outputs/logs/pick_orange_30_only/pipeline.log"
ALL_DONE = RUN_ROOT / "ALL_DONE"
EVAL_DONE = ROOT / "outputs/eval/pick_orange_gate_exp30_double/DONE"
REPORT_DONE = ROOT / "outputs/reports/pick_orange_30_only/DONE"


def save(payload: dict) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, STATE)


def run_logged(command: list[str], environment: dict) -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(f"\n===== {time.strftime('%F %T')} {' '.join(command)} =====\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        return process.wait()


def main() -> None:
    state = {
        "status": "starting",
        "stage": None,
        "attempt": 0,
        "eval_parallel": 3,
        "fifty_demo_cancelled": True,
        "updated_at": time.strftime("%FT%T%z"),
    }
    if STATE.is_file():
        try:
            state.update(json.loads(STATE.read_text(encoding="utf-8")))
        except Exception:
            pass
    environment = os.environ.copy()
    environment["FRONTLOAD_50_BEFORE_EVAL30"] = "0"
    environment["PYTHONUNBUFFERED"] = "1"

    while not EVAL_DONE.is_file():
        free_gb = shutil.disk_usage(ROOT).free / 1024**3
        if free_gb < 20:
            state.update(status="waiting_for_disk", stage="eval30", free_gb=round(free_gb, 1), updated_at=time.strftime("%FT%T%z"))
            save(state)
            time.sleep(300)
            continue
        state["attempt"] = int(state.get("attempt", 0)) + 1
        state.update(status="running", stage="eval30", free_gb=round(free_gb, 1), updated_at=time.strftime("%FT%T%z"))
        save(state)
        environment["EVAL_MAX_PARALLEL"] = str(state["eval_parallel"])
        status = run_logged([sys.executable, "experiments/run_pick_orange_double_steps_eval.py", "30"], environment)
        if status == 0 and EVAL_DONE.is_file():
            break
        if state["eval_parallel"] > 1:
            state["eval_parallel"] -= 1
        state.update(status="retry_wait", last_exit=status, updated_at=time.strftime("%FT%T%z"))
        save(state)
        time.sleep(min(600, 30 * 2 ** min(state["attempt"] - 1, 4)))

    while not REPORT_DONE.is_file():
        state.update(status="running", stage="report30", updated_at=time.strftime("%FT%T%z"))
        save(state)
        status = run_logged([sys.executable, "experiments/report_pick_orange_30_only.py"], environment)
        if status == 0 and REPORT_DONE.is_file():
            break
        state.update(status="retry_wait", last_exit=status, updated_at=time.strftime("%FT%T%z"))
        save(state)
        time.sleep(30)

    state.update(status="complete", stage=None, completed_at=time.strftime("%FT%T%z"), updated_at=time.strftime("%FT%T%z"))
    save(state)
    ALL_DONE.write_text(f"completed_at={state['completed_at']}\nfifty_demo_cancelled=true\n", encoding="utf-8")


if __name__ == "__main__":
    main()
