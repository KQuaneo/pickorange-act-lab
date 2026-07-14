#!/usr/bin/env python3
"""Cross-result protocol, paired, and isolated-vs-sequential analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pick_orange_analysis import (
    align_paired_results,
    exact_mcnemar,
    horizon_fairness,
    horizon_protocol_spec,
    isolated_sequential_gap,
    paired_bootstrap_difference,
)


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def paired(left_path: Path, right_path: Path) -> dict:
    left, right = load(left_path), load(right_path)
    if not left or not right:
        return {"status": "missing", "left": str(left_path), "right": str(right_path)}
    aligned = align_paired_results(left.get("episode_results", []), right.get("episode_results", []))
    pairs = aligned["pairs"]
    left_success = [bool(pair[0].get("success")) for pair in pairs]
    right_success = [bool(pair[1].get("success")) for pair in pairs]
    left_partial = [sum(bool(v) for v in (pair[0].get("final_in_plate") or {}).values()) for pair in pairs]
    right_partial = [sum(bool(v) for v in (pair[1].get("final_in_plate") or {}).values()) for pair in pairs]
    return {
        "status": "ok" if pairs else "unaligned",
        "left": str(left_path), "right": str(right_path),
        "aligned_pairs": len(pairs), "left_only": len(aligned["left_only"]), "right_only": len(aligned["right_only"]),
        "mcnemar_success": exact_mcnemar(left_success, right_success) if pairs else None,
        "paired_bootstrap_final_oranges": paired_bootstrap_difference(left_partial, right_partial) if pairs else None,
        "difference_direction": "right_minus_left",
    }


def sequential_gap(root: Path, count: int, phase: str) -> dict:
    base = root / f"outputs/eval/pick_orange_gate_exp{count}_double"
    full_path = base / "a0_a1_checkpoint_eval/a1_s42k/summary.json"
    isolated_path = base / f"a1_isolated_014000_policy420/{phase}/summary.json"
    full, isolated = load(full_path), load(isolated_path)
    if not full or not isolated:
        return {"count": count, "phase": phase, "status": "missing", "full": str(full_path), "isolated": str(isolated_path)}
    index = {"b1": 0, "b2": 1, "b3": 2}[phase]
    rows = full.get("episode_results", [])
    sequential = 0
    reached = 0
    for row in rows:
        stages = row.get("stage_results") or []
        if len(stages) > index:
            reached += int(bool(stages[index].get("reached")))
            sequential += int(stages[index].get("outcome") == "success")
    episodes = min(reached, int(isolated.get("episodes", 0)))
    return {
        "count": count, "phase": phase, "status": "ok" if episodes else "legacy_summary_missing_stage_fields",
        "sequential_definition": "target in plate and prior prefix intact at fixed-time stage end",
        "isolated_definition": "oracle-initialized primitive target success",
        **isolated_sequential_gap(sequential, int(isolated.get("successes", 0)), episodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.root / "outputs/reports/pick_orange_double30_vs50/protocol_analysis"
    comparisons = []
    for group in ("a0", "a1"):
        for label in ("s30k", "s36k", "s42k"):
            left = args.root / f"outputs/eval/pick_orange_gate_exp30_double/a0_a1_checkpoint_eval/{group}_{label}/summary.json"
            right = args.root / f"outputs/eval/pick_orange_gate_exp50_double/a0_a1_checkpoint_eval/{group}_{label}/summary.json"
            comparisons.append({"comparison": f"{group.upper()} {label}: 30 vs 50", **paired(left, right)})
    gaps = [sequential_gap(args.root, count, phase) for count in (30, 50) for phase in ("b1", "b2", "b3")]
    overrun = {}
    for count in (30, 50):
        summary_path = args.root / f"outputs/eval/pick_orange_gate_exp{count}_double/a0_a1_checkpoint_eval/a1_s42k/summary.json"
        summary = load(summary_path) or {}
        overrun[str(count)] = {
            "status": "ok" if summary.get("post_success_overrun_summary") else "missing_or_legacy_result",
            "summary": summary.get("post_success_overrun_summary"),
            "raw_jsonl": summary.get("post_success_overrun_jsonl"),
        }
    payload = {
        "schema_version": 1,
        "horizon_audit": horizon_fairness(),
        "horizon_protocols": {
            name: horizon_protocol_spec(name) for name in ("native_horizon", "matched_horizon")
        },
        "scheduler_classification": {
            "A1_full": "fixed-time external multi-policy scheduler (420 actions per stage); not success-oracle",
            "A1_isolated_B2_B3": "oracle-initialized primitive tests",
            "A3_unused": "ground-truth stable-placement oracle scheduler; outside the current formal comparison",
        },
        "two_level_evaluation": {
            "level_1": "isolated B1/B2/B3 primitive competence under documented initialization",
            "level_2": "full sequential rollout with fixed-time scheduler and prefix preservation",
            "rule": "Do not interpret isolated oracle results as end-to-end success rates.",
        },
        "paired_30_vs_50": comparisons,
        "isolated_vs_sequential": gaps,
        "post_success_overrun": overrun,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    horizon = payload["horizon_audit"]
    lines = [
        "# PickOrange protocol and paired analysis", "", "## Horizon fairness", "",
        f"Formal A0: **{horizon['a0']['policy_steps']}** policy steps / **{horizon['a0']['theoretical_duration_s']:.1f}s** theoretical simulation time.", "",
        f"Formal A1: **{horizon['a1']['policy_steps']}** policy steps / **{horizon['a1']['theoretical_duration_s']:.1f}s**. A0 therefore receives {100*horizon['a0_over_a1']:.1f}% of A1's horizon.", "",
        "Keep the formal result unchanged; if needed, add an explicitly secondary A0-1260 comparable-horizon result.", "",
        "| Protocol | Method | Policy actions | Simulation steps | Sim steps/action | Theoretical time | Same A0/A1 horizon |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for protocol_name in ("native_horizon", "matched_horizon"):
        protocol = payload["horizon_protocols"][protocol_name]
        for method in ("a0", "a1"):
            item = protocol[method]
            lines.append(f"| {protocol_name} | {method.upper()} | {item['policy_steps']} | {item['simulation_steps']} | {item['sim_steps_per_action']} | {item['theoretical_duration_s']:.1f}s | {protocol['same_total_horizon']} |")
    lines += ["",
        "## Paired 30 vs 50", "", "| Comparison | Aligned | McNemar p | Partial-orange Δ (right-left) | 95% paired bootstrap |", "|---|---:|---:|---:|---|",
    ]
    for row in comparisons:
        mc = row.get("mcnemar_success") or {}
        pb = row.get("paired_bootstrap_final_oranges") or {}
        ci = pb.get("ci_95") or [None, None]
        lines.append(f"| {row['comparison']} | {row.get('aligned_pairs', 0)} | {mc.get('exact_two_sided_p', '-')} | {pb.get('difference', '-')} | {ci} |")
    lines += ["", "## Isolated vs sequential gap", "", "| Experts | Phase | Status | Sequential | Isolated | Gap (pp) |", "|---:|---|---|---:|---:|---:|"]
    for row in gaps:
        lines.append(f"| {row['count']} | {row['phase'].upper()} | {row['status']} | {row.get('sequential_rate', '-')} | {row.get('isolated_rate', '-')} | {row.get('gap_pp', '-')} |")
    lines += ["", "## Fixed-time post-success overrun", "", "`overrun = stage_switch_step - target_first_stably_satisfied_step`; missing stable success remains null.", "", "| Experts | Stage | N | Mean | Median | Q25 | Q75 | Q90 | Prefix destroyed |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for count in (30, 50):
        by_stage = ((overrun[str(count)].get("summary") or {}).get("by_stage") or {})
        if not by_stage:
            lines.append(f"| {count} | - | - | - | - | - | - | - | result pending/legacy |")
            continue
        for stage, row in by_stage.items():
            lines.append(f"| {count} | {stage} | {row['overrun_available']} | {row['mean']} | {row['median']} | {row['q25']} | {row['q75']} | {row['q90']} | {row['prefix_destroyed_during_overrun']} |")
        relation = (overrun[str(count)]["summary"] or {}).get("overrun_vs_next_stage_start_deviation", {})
        lines.append(f"| {count} | overrun vs next-start deviation | {relation.get('pairs', 0)} | r={relation.get('pearson_r')} | correlation only | - | - | - | no causal claim |")
    lines += ["", "## Interpretation", "", "Level 1 is isolated primitive competence. Level 2 is sequential competence including stage-start distribution shift and prefix preservation. The two levels must be reported separately.", ""]
    (output / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "analysis.md")


if __name__ == "__main__":
    main()
