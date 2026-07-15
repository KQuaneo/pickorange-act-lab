#!/usr/bin/env python3
"""Dependency-light paired analysis for the B1 ACT temporal aggregation run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from pick_orange_analysis import exact_mcnemar, paired_bootstrap_difference, wilson_interval


CONDITIONS = ("h1_no_aggregation", "h1_temporal_aggregation_001")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def summarize_values(values: Iterable[float]) -> dict:
    data = [float(value) for value in values]
    if not data:
        return {"n": 0, "mean": None, "median": None, "p95": None}
    return {
        "n": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "p95": percentile(data, 0.95),
    }


def keyed(rows: list[dict]) -> dict[tuple[int, str], dict]:
    return {(int(row["episode_index"]), str(row["initialization_id"])): row for row in rows}


def action_jitter(rows: list[dict]) -> dict[int, float]:
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode_index"])].append(row)
    result = {}
    for episode, items in by_episode.items():
        items.sort(key=lambda row: int(row["step"]))
        actions = [row["aggregated_action"] for row in items]
        deltas = [
            math.sqrt(sum((float(b) - float(a)) ** 2 for a, b in zip(actions[index - 1], actions[index], strict=True)))
            for index in range(1, len(actions))
        ]
        result[episode] = statistics.fmean(deltas) if deltas else 0.0
    return result


def require_complete(root: Path) -> None:
    for condition in CONDITIONS:
        directory = root / condition
        required = {
            "summary.json", "episode_results.jsonl", "initial_states.jsonl", "policy_calls.jsonl",
            "aggregation_diagnostics.jsonl", "run_config.json", "command.txt", "stdout.log",
            "checkpoint_sha256_before.txt", "checkpoint_sha256_after.txt", "COMPLETED",
        }
        missing = sorted(name for name in required if not (directory / name).exists())
        if missing:
            raise FileNotFoundError(f"{condition} incomplete; missing {missing}")
        before = (directory / "checkpoint_sha256_before.txt").read_text(encoding="utf-8")
        after = (directory / "checkpoint_sha256_after.txt").read_text(encoding="utf-8")
        if before != after:
            raise RuntimeError(f"checkpoint hash mismatch in {condition}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    require_complete(args.root)

    no_rows = read_jsonl(args.root / CONDITIONS[0] / "episode_results.jsonl")
    agg_rows = read_jsonl(args.root / CONDITIONS[1] / "episode_results.jsonl")
    no_map, agg_map = keyed(no_rows), keyed(agg_rows)
    if set(no_map) != set(agg_map) or len(no_map) != 20:
        raise RuntimeError(f"formal paired analysis requires the same 20 initialization keys; no={len(no_map)}, agg={len(agg_map)}")
    keys = sorted(no_map)

    no_success = [bool(no_map[key]["success"]) for key in keys]
    agg_success = [bool(agg_map[key]["success"]) for key in keys]
    transitions = Counter(
        "success_to_success" if a and b else
        "success_to_failure" if a and not b else
        "failure_to_success" if not a and b else
        "failure_to_failure"
        for a, b in zip(no_success, agg_success, strict=True)
    )
    no_count, agg_count = sum(no_success), sum(agg_success)
    no_lift = [float(no_map[key]["max_lift_m"]) for key in keys]
    agg_lift = [float(agg_map[key]["max_lift_m"]) for key in keys]
    lift_deltas = [b - a for a, b in zip(no_lift, agg_lift, strict=True)]

    no_diag = read_jsonl(args.root / CONDITIONS[0] / "aggregation_diagnostics.jsonl")
    agg_diag = read_jsonl(args.root / CONDITIONS[1] / "aggregation_diagnostics.jsonl")
    no_jitter = action_jitter(no_diag)
    agg_jitter = action_jitter(agg_diag)
    jitter_left = [no_jitter[key[0]] for key in keys]
    jitter_right = [agg_jitter[key[0]] for key in keys]
    aggregation_differences = [float(row["aggregation_action_difference_l2"]) for row in agg_diag]
    ensemble_sizes = [int(row["ensemble_size_per_step"]) for row in agg_diag]

    no_calls = read_jsonl(args.root / CONDITIONS[0] / "policy_calls.jsonl")
    agg_calls = read_jsonl(args.root / CONDITIONS[1] / "policy_calls.jsonl")
    policy_counts_ok = len(no_calls) == len(agg_calls) == 20 * 420
    ensemble_pattern_ok = all(
        int(row["ensemble_size_per_step"]) == min(int(row["step"]) + 1, 100) for row in agg_diag
    )
    no_aggregation_identity_ok = all(float(row["aggregation_action_difference_l2"]) <= 1e-12 for row in no_diag)

    report = {
        "schema_version": 1,
        "status": "PAIRED_ANALYSIS_COMPLETE",
        "claim_boundary": "Only K=100, H=1, current G4 A1-B1 14k checkpoint, seed 2026 and the fixed paired 20-state manifest.",
        "sample_size_warning": "Twenty paired episodes describe the observed sample; they are not a precise estimate of the true success rate.",
        "implementation_validation": {
            "policy_calls_420_per_episode_each_condition": policy_counts_ok,
            "aggregation_ensemble_size_pattern_1_to_100": ensemble_pattern_ok,
            "no_aggregation_selected_equals_latest_chunk": no_aggregation_identity_ok,
            "native_logic": "LeRobot ACTTemporalEnsembler.update; exponential weights exp(-0.01*i), oldest prediction receives w0.",
        },
        "pairing": json.loads((args.root / "pairing_diagnostics.json").read_text(encoding="utf-8")),
        "success": {
            CONDITIONS[0]: {
                "successes": no_count,
                "episodes": 20,
                "rate": no_count / 20,
                "wilson95": list(wilson_interval(no_count, 20)),
            },
            CONDITIONS[1]: {
                "successes": agg_count,
                "episodes": 20,
                "rate": agg_count / 20,
                "wilson95": list(wilson_interval(agg_count, 20)),
            },
            "paired_transitions": dict(transitions),
            "exact_mcnemar": exact_mcnemar(no_success, agg_success),
        },
        "max_lift_m": {
            CONDITIONS[0]: summarize_values(no_lift),
            CONDITIONS[1]: summarize_values(agg_lift),
            "paired_aggregation_minus_no_aggregation": {
                **summarize_values(lift_deltas),
                "bootstrap_mean_difference_ci95": paired_bootstrap_difference(
                    no_lift, agg_lift, samples=args.bootstrap_samples, seed=args.seed
                ),
            },
        },
        "failure_taxonomy": {
            CONDITIONS[0]: dict(sorted(Counter(row["failure_category"] for row in no_rows).items())),
            CONDITIONS[1]: dict(sorted(Counter(row["failure_category"] for row in agg_rows).items())),
        },
        "action_jitter": {
            CONDITIONS[0]: summarize_values(jitter_left),
            CONDITIONS[1]: summarize_values(jitter_right),
            "paired_aggregation_minus_no_aggregation": paired_bootstrap_difference(
                jitter_left, jitter_right, samples=args.bootstrap_samples, seed=args.seed
            ),
            "decreased_observed_mean": statistics.fmean(jitter_right) < statistics.fmean(jitter_left),
        },
        "aggregation_vs_latest_prediction": {
            "difference_l2": summarize_values(aggregation_differences),
            "nonzero_steps": sum(value > 1e-12 for value in aggregation_differences),
            "ensemble_size": summarize_values(ensemble_sizes),
        },
        "inference_cost": {
            CONDITIONS[0]: {
                "rollout_wall_time_s": summarize_values(row["rollout_wall_time"] for row in no_rows),
                "policy_call_time_s": summarize_values(row["policy_call_wall_time_s"] for row in no_calls),
            },
            CONDITIONS[1]: {
                "rollout_wall_time_s": summarize_values(row["rollout_wall_time"] for row in agg_rows),
                "policy_call_time_s": summarize_values(row["policy_call_wall_time_s"] for row in agg_calls),
            },
        },
        "rhc_interpretation": {
            "changes_original_rhc_conclusion": False,
            "reason": "The original RHC conclusion concerned replanning horizon without temporal aggregation. This experiment tests an additional inference mechanism at H=1; even a gain would be an H=1×aggregation interaction, not evidence that replanning alone beat H=100.",
        },
    }
    observed_gain = agg_count - no_count
    report["followup"] = {
        "worth_full_task": bool(observed_gain > 0 and statistics.fmean(agg_lift) >= statistics.fmean(no_lift)),
        "criterion": "exploratory only: observed success gain plus non-worse mean max lift; no general claim from n=20",
    }
    (args.root / "paired_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    mcnemar = report["success"]["exact_mcnemar"]
    jitter_delta = report["action_jitter"]["paired_aggregation_minus_no_aggregation"]["difference"]
    lines = [
        "# PickOrange B1 ACT Temporal Aggregation — Paired Analysis",
        "",
        "> Claim boundary: K=100, H=1, G4 A1-B1 14k, coeff=0.01, fixed paired 20-state manifest.",
        "> Twenty episodes are a small observed sample, not a precise true success rate.",
        "",
        f"- No aggregation: **{no_count}/20 ({no_count / 20:.1%})**, Wilson 95% {wilson_interval(no_count, 20)[0]:.1%}–{wilson_interval(no_count, 20)[1]:.1%}.",
        f"- Temporal aggregation: **{agg_count}/20 ({agg_count / 20:.1%})**, Wilson 95% {wilson_interval(agg_count, 20)[0]:.1%}–{wilson_interval(agg_count, 20)[1]:.1%}.",
        f"- Paired transitions: {dict(transitions)}; exact McNemar p={mcnemar['exact_two_sided_p']:.6g}.",
        f"- Mean paired max-lift difference (aggregation − no aggregation): {statistics.fmean(lift_deltas):.6f} m.",
        f"- Mean paired action-jitter difference (aggregation − no aggregation): {jitter_delta:.6f}.",
        f"- Mean rollout wall time: {statistics.fmean(row['rollout_wall_time'] for row in no_rows):.3f}s vs {statistics.fmean(row['rollout_wall_time'] for row in agg_rows):.3f}s.",
        "- Original RHC conclusion is not replaced: this isolates temporal aggregation at H=1, not replanning horizon alone.",
        f"- Full-task follow-up worth doing under the exploratory gate: **{report['followup']['worth_full_task']}**.",
    ]
    (args.root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.root / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
