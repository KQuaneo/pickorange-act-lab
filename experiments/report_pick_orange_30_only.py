#!/usr/bin/env python3
"""Generate a focused report for the 30-demo PickOrange experiment only."""

from __future__ import annotations

import json
from pathlib import Path

from pick_orange_analysis import wilson_interval


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs/eval/pick_orange_gate_exp30_double"
OUTPUT = ROOT / "outputs/reports/pick_orange_30_only"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def result_row(name: str, path: Path) -> dict:
    payload = load(path)
    episodes = int(payload["episodes"])
    successes = int(payload["successes"])
    low, high = wilson_interval(successes, episodes)
    final_counts = [sum(bool(value) for value in (row.get("final_in_plate") or {}).values()) for row in payload.get("episode_results", [])]
    return {
        "name": name,
        "successes": successes,
        "episodes": episodes,
        "rate": successes / episodes,
        "wilson_95": [low, high],
        "mean_final_oranges": sum(final_counts) / len(final_counts) if final_counts else None,
        "horizon_protocol": payload.get("horizon_protocol", "native_horizon"),
        "policy_actions": payload.get("total_policy_steps"),
        "simulation_steps": payload.get("simulation_steps"),
        "theoretical_duration_s": payload.get("theoretical_duration_s"),
        "summary": str(path),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    full_root = EVAL / "a0_a1_checkpoint_eval"
    full_specs = [
        ("A0 legacy21k", full_root / "a0_legacy21k/summary.json"),
        ("A1 legacy7k", full_root / "a1_legacy7k/summary.json"),
    ] + [
        (f"{group.upper()} {label}", full_root / f"{group}_{label}/summary.json")
        for label in ("s30k", "s36k", "s42k") for group in ("a0", "a1")
    ]
    full_rows = [result_row(name, path) for name, path in full_specs]
    isolated_specs = [
        (f"legacy7k {phase.upper()}", EVAL / f"a1_isolated_legacy007000_policy420/{phase}/summary.json")
        for phase in ("b1", "b2", "b3")
    ] + [
        (f"14k {phase.upper()}", EVAL / f"a1_isolated_014000_policy420/{phase}/summary.json")
        for phase in ("b1", "b2", "b3")
    ]
    isolated_rows = [result_row(name, path) for name, path in isolated_specs]
    effective = load(ROOT / "outputs/audits/pick_orange_gate_exp30_effective_data/effective_data.json")
    b3 = load(ROOT / "outputs/audits/pick_orange_gate_exp30_b3/audit.json")
    b3_total = int(b3.get("count") or b3.get("orange003_slices") or len(b3.get("episodes", [])))
    final_a1 = load(full_root / "a1_s42k/summary.json")
    payload = {
        "schema_version": 1,
        "scope": "30_demo_only",
        "fifty_demo_cancelled": True,
        "full_task": full_rows,
        "isolated": isolated_rows,
        "effective_data": effective,
        "b3_audit": b3,
        "post_success_overrun": final_a1.get("post_success_overrun_summary"),
    }
    (OUTPUT / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# PickOrange 30-demo experiment report", "",
        "The 50-demo extension was cancelled by user request. This report contains only the completed 30-demo protocol.", "",
        "## Protocol", "",
        "- Batch size: 64", "- A0: 42k training steps; checkpoints 30k/36k/42k plus legacy21k",
        "- A1: 14k per sub-policy; checkpoints 10k/12k/14k plus legacy7k",
        "- Formal evaluator: native_horizon, 20 episodes, seed 2026",
        "- A1 scheduler: fixed-time 420 actions per stage; isolated B2/B3 use oracle initialization",
        "", "## Full-task results", "",
        "| Method | Success | Rate | Wilson 95% | Mean final oranges | Actions | Sim steps | Time |", "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in full_rows:
        lo, hi = row["wilson_95"]
        lines.append(f"| {row['name']} | {row['successes']}/{row['episodes']} | {100*row['rate']:.1f}% | {100*lo:.1f}–{100*hi:.1f}% | {row['mean_final_oranges']:.2f} | {row['policy_actions']} | {row['simulation_steps']} | {row['theoretical_duration_s']:.1f}s |")
    lines += ["", "## Isolated A1 results", "", "| Checkpoint | Success | Rate | Wilson 95% |", "|---|---:|---:|---|"]
    for row in isolated_rows:
        lo, hi = row["wilson_95"]
        lines.append(f"| {row['name']} | {row['successes']}/{row['episodes']} | {100*row['rate']:.1f}% | {100*lo:.1f}–{100*hi:.1f}% |")
    lines += ["", "## Effective data", "", "| Dataset | Episodes | Frames/anchors | Approx exposure |", "|---|---:|---:|---:|"]
    for row in effective.get("datasets", []):
        if row.get("status") == "ok":
            lines.append(f"| {row['name']} | {row['episodes']} | {row['frames']} | {row['approx_anchor_exposures']:.2f}× |")
    lines += ["", "## B3 audit", "", f"- Integrity: {b3.get('integrity_passes')}/{b3_total}", f"- Target success: {b3.get('target_successes')}/{b3_total}", f"- Strict-prefix success: {b3.get('strict_successes')}/{b3_total}", "", "## Fixed-time overrun", ""]
    overrun = final_a1.get("post_success_overrun_summary") or {}
    for stage, row in (overrun.get("by_stage") or {}).items():
        lines.append(f"- {stage}: n={row['overrun_available']}, mean={row['mean']}, median={row['median']}, Q25/Q75={row['q25']}/{row['q75']}, prefix destroyed={row['prefix_destroyed_during_overrun']}")
    relation = overrun.get("overrun_vs_next_stage_start_deviation") or {}
    lines += ["", f"Overrun vs next-stage start deviation: pairs={relation.get('pairs', 0)}, Pearson r={relation.get('pearson_r')}. Descriptive correlation only; no causal claim.", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT / "DONE").write_text("30-demo report complete\n", encoding="utf-8")
    print(OUTPUT / "report.md")


if __name__ == "__main__":
    main()
