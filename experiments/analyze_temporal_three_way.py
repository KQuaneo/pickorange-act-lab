#!/usr/bin/env python3
"""Strictly paired three-way analysis for B1 ACT execution semantics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


GROUPS = ("h100_off", "h1_off", "h1_aggregation_001")
PAIRS = (
    ("h100_off", "h1_off"),
    ("h100_off", "h1_aggregation_001"),
    ("h1_off", "h1_aggregation_001"),
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson(successes: int, episodes: int, z: float = 1.959963984540054) -> list[float]:
    rate = successes / episodes
    denominator = 1 + z * z / episodes
    center = (rate + z * z / (2 * episodes)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / episodes + z * z / (4 * episodes**2)) / denominator
    return [center - margin, center + margin]


def exact_mcnemar(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def load_group(root: Path, horizon: int) -> dict:
    required = (
        "summary.json", "episode_results.jsonl", "initial_states.jsonl", "policy_calls.jsonl",
        "run_config.json", "command.txt", "checkpoint_sha256_before.txt",
        "checkpoint_sha256_after.txt", "COMPLETED",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete group {root}: {missing}")
    episodes = read_jsonl(root / "episode_results.jsonl")
    starts = read_jsonl(root / "initial_states.jsonl")
    calls = read_jsonl(root / "policy_calls.jsonl")
    expected_calls = 20 * math.ceil(420 / horizon)
    if len(episodes) != 20 or len(starts) != 20 or len(calls) != expected_calls:
        raise RuntimeError(
            f"invalid counts for {root}: episodes={len(episodes)}, starts={len(starts)}, calls={len(calls)}"
        )
    if (root / "checkpoint_sha256_before.txt").read_text() != (root / "checkpoint_sha256_after.txt").read_text():
        raise RuntimeError(f"checkpoint changed during {root}")
    expected_starts = list(range(0, 420, horizon))
    for episode in range(20):
        episode_calls = [row for row in calls if int(row["episode_index"]) == episode]
        if [int(row["environment_step"]) for row in episode_calls] != expected_starts:
            raise RuntimeError(f"policy-call boundaries failed in {root}, episode {episode}")
        if int(episodes[episode]["rhc"]["total_executed_actions"]) != 420:
            raise RuntimeError(f"executed-action count failed in {root}, episode {episode}")
    return {"root": root, "episodes": episodes, "starts": starts, "calls": calls}


def group_summary(group: dict) -> dict:
    rows = group["episodes"]
    successes = sum(bool(row["success"]) for row in rows)
    lifts = [float(row["oranges"]["Orange001"]["max_lift_m"]) for row in rows]
    failures = Counter(row["failure_category"] for row in rows)
    return {
        "successes": successes,
        "episodes": 20,
        "success_rate": successes / 20,
        "wilson_95": wilson(successes, 20),
        "max_lift_mean_m": statistics.fmean(lifts),
        "max_lift_median_m": statistics.median(lifts),
        "failure_taxonomy": dict(failures),
        "contact_or_better": 20 - int(failures.get("no_effect", 0)),
        "contact_or_better_rate": (20 - int(failures.get("no_effect", 0))) / 20,
        "action_jitter_l2_mean": statistics.fmean(float(row["rhc"]["executed_action_delta_l2_mean"]) for row in rows),
        "policy_calls_total": sum(int(row["rhc"]["policy_call_count"]) for row in rows),
        "policy_calls_per_episode": [int(row["rhc"]["policy_call_count"]) for row in rows],
        "policy_latency_total_mean_s": statistics.fmean(float(row["rhc"]["policy_latency_total_s"]) for row in rows),
        "rollout_wall_time_mean_s": statistics.fmean(float(row["rhc"]["rollout_wall_time_s"]) for row in rows),
    }


def svg_chart(summaries: dict[str, dict]) -> str:
    labels = {"h100_off": "H=100 off", "h1_off": "H=1 off", "h1_aggregation_001": "H=1 agg .01"}
    colors = {"h100_off": "#64748b", "h1_off": "#ef4444", "h1_aggregation_001": "#22c55e"}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="430" viewBox="0 0 920 430">',
        '<rect width="920" height="430" fill="#ffffff"/>',
        '<text x="40" y="38" font-family="sans-serif" font-size="22" font-weight="700">Paired B1: success and mean max lift</text>',
        '<line x1="70" y1="350" x2="860" y2="350" stroke="#334155"/>',
    ]
    for index, name in enumerate(GROUPS):
        x = 120 + index * 250
        success_height = summaries[name]["success_rate"] * 260
        lift_height = min(summaries[name]["max_lift_mean_m"] / 0.16, 1.0) * 260
        parts.extend([
            f'<rect x="{x}" y="{350-success_height:.2f}" width="70" height="{success_height:.2f}" fill="{colors[name]}"/>',
            f'<rect x="{x+85}" y="{350-lift_height:.2f}" width="70" height="{lift_height:.2f}" fill="{colors[name]}" opacity="0.45"/>',
            f'<text x="{x+77}" y="385" text-anchor="middle" font-family="sans-serif" font-size="15">{labels[name]}</text>',
            f'<text x="{x+35}" y="{340-success_height:.2f}" text-anchor="middle" font-family="sans-serif" font-size="14">{summaries[name]["successes"]}/20</text>',
            f'<text x="{x+120}" y="{340-lift_height:.2f}" text-anchor="middle" font-family="sans-serif" font-size="14">{summaries[name]["max_lift_mean_m"]:.3f}m</text>',
        ])
    parts.extend([
        '<rect x="680" y="55" width="18" height="18" fill="#334155"/><text x="705" y="70" font-family="sans-serif" font-size="14">success rate (solid)</text>',
        '<rect x="680" y="82" width="18" height="18" fill="#334155" opacity="0.45"/><text x="705" y="97" font-family="sans-serif" font-size="14">mean max lift (light)</text>',
        '</svg>',
    ])
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h100-off", type=Path, required=True)
    parser.add_argument("--h1-off", type=Path, required=True)
    parser.add_argument("--h1-aggregation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    groups = {
        "h100_off": load_group(args.h100_off, 100),
        "h1_off": load_group(args.h1_off, 1),
        "h1_aggregation_001": load_group(args.h1_aggregation, 1),
    }
    pairing_rows = []
    for episode in range(20):
        starts = {name: groups[name]["starts"][episode] for name in GROUPS}
        ids = {name: starts[name]["initialization_id"] for name in GROUPS}
        paired = (
            len(set(ids.values())) == 1
            and all(starts[name]["manifest_state"] == starts["h1_off"]["manifest_state"] for name in GROUPS)
            and all(starts[name]["state"] == starts["h1_off"]["state"] for name in GROUPS)
            and all(starts[name]["initialization"] == starts["h1_off"]["initialization"] for name in GROUPS)
        )
        pairing_rows.append({"episode_index": episode, "initialization_ids": ids, "strict_physical_pairing": paired})
    if not all(row["strict_physical_pairing"] for row in pairing_rows):
        (args.output / "PAIRING_BLOCKER.json").write_text(json.dumps(pairing_rows, indent=2), encoding="utf-8")
        raise RuntimeError("three-way physical-state pairing failed")

    summaries = {name: group_summary(groups[name]) for name in GROUPS}
    details = []
    for episode in range(20):
        row = {"episode_index": episode, "initialization_id": groups["h1_off"]["starts"][episode]["initialization_id"]}
        for name in GROUPS:
            result = groups[name]["episodes"][episode]
            row[name] = {
                "success": bool(result["success"]),
                "max_lift_m": float(result["oranges"]["Orange001"]["max_lift_m"]),
                "failure_category": result["failure_category"],
                "policy_call_count": int(result["rhc"]["policy_call_count"]),
                "action_jitter_l2_mean": float(result["rhc"]["executed_action_delta_l2_mean"]),
                "rollout_wall_time_s": float(result["rhc"]["rollout_wall_time_s"]),
            }
        details.append(row)

    pairwise = {}
    for left, right in PAIRS:
        transitions = Counter()
        lift_differences = []
        for row in details:
            left_success = int(row[left]["success"])
            right_success = int(row[right]["success"])
            transitions[f"{left_success}_to_{right_success}"] += 1
            lift_differences.append(row[right]["max_lift_m"] - row[left]["max_lift_m"])
        b = transitions["1_to_0"]
        c = transitions["0_to_1"]
        pairwise[f"{left}_vs_{right}"] = {
            "transition_left_to_right": dict(transitions),
            "mcnemar_exact_two_sided_p": exact_mcnemar(b, c),
            "left_success_right_failure": b,
            "left_failure_right_success": c,
            "paired_max_lift_difference_right_minus_left_m": {
                "mean": statistics.fmean(lift_differences),
                "median": statistics.median(lift_differences),
                "values": lift_differences,
            },
        }

    triple_transitions = Counter(
        f"{int(row['h100_off']['success'])}{int(row['h1_off']['success'])}{int(row['h1_aggregation_001']['success'])}"
        for row in details
    )
    result = {
        "status": "THREE_WAY_PAIRED_COMPARISON_COMPLETE",
        "pairing_status": "STRICT_PHYSICAL_PAIRING_20_OF_20",
        "protocol": {"chunk_size": 100, "episodes": 20, "seed": 2026, "policy_steps": 420, "sim_steps_per_action": 2},
        "groups": summaries,
        "triple_success_transition_h100off_h1off_h1agg": dict(triple_transitions),
        "pairwise": pairwise,
        "pairing_rows": pairing_rows,
        "claim_boundary": "Only this G4 A1-B1 14k checkpoint, 20 paired simulator starts, K=100, and the tested H/aggregation settings.",
    }
    (args.output / "comparison_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output / "paired_episode_details.jsonl").open("w", encoding="utf-8") as stream:
        for row in details:
            stream.write(json.dumps(row) + "\n")
    (args.output / "three_way_success_and_lift.svg").write_text(svg_chart(summaries), encoding="utf-8")

    lines = [
        "# Strictly Paired B1 Three-Way Comparison", "",
        "All 20 episodes share identical initialization IDs, serialized task state, restored physical state, and refresh protocol.", "",
        "| Group | Success | Wilson 95% | Mean/median lift | Contact-or-better | Jitter | Calls | Wall time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"h100_off": "H=100 off", "h1_off": "H=1 off", "h1_aggregation_001": "H=1 aggregation .01"}
    for name in GROUPS:
        item = summaries[name]
        lines.append(
            f"| {labels[name]} | {item['successes']}/20 | {item['wilson_95'][0]:.3f}–{item['wilson_95'][1]:.3f} | "
            f"{item['max_lift_mean_m']:.4f}/{item['max_lift_median_m']:.4f} m | {item['contact_or_better']}/20 | "
            f"{item['action_jitter_l2_mean']:.5f} | {item['policy_calls_total']} | {item['rollout_wall_time_mean_s']:.2f} s |"
        )
    lines.extend(["", f"Triple transitions (H100 off, H1 off, H1 agg): `{dict(triple_transitions)}`.", ""])
    for name, item in pairwise.items():
        lines.append(f"- `{name}`: transitions {item['transition_left_to_right']}; exact McNemar p={item['mcnemar_exact_two_sided_p']:.6g}.")
    lines.extend(["", "These 20 paired episodes do not establish a generally optimal controller.", ""])
    (args.output / "THREE_WAY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output / "COMPLETED").write_text("PASS\n", encoding="utf-8")


if __name__ == "__main__":
    main()
