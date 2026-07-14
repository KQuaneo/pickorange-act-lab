#!/usr/bin/env python3
"""Tighten Gate3 event-slice success to preserve completed prefix stages.

The original Gate3 checker marks a slice successful when its target orange is
placed.  For A1 training that is insufficient: an Orange003 demonstration is
only valid if Orange001 and Orange002 remain in the plate at the end of the
slice.  This tool writes a filtered summary that can be consumed by
build_a1_event_lerobot_datasets.py with --allow_failed_slices.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path


PREFIX_REQUIREMENTS = {
    "Orange001": (),
    "Orange002": ("Orange001",),
    "Orange003": ("Orange001", "Orange002"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = copy.deepcopy(source)
    rejected: list[dict] = []
    counts: dict[str, Counter] = defaultdict(Counter)

    for item in result.get("slice_results", []):
        orange = item["orange"]
        final_in_plate = item.get("final_in_plate") or {}
        target_success = bool(item.get("success"))
        required_prefix = PREFIX_REQUIREMENTS[orange]
        prefix_intact = all(bool(final_in_plate.get(name)) for name in required_prefix)
        strict_success = target_success and prefix_intact

        item["target_success"] = target_success
        item["prefix_requirements"] = list(required_prefix)
        item["prefix_intact"] = prefix_intact
        item["strict_success"] = strict_success
        item["success"] = strict_success
        if not target_success:
            item["strict_rejection_reason"] = "target_not_stably_placed"
        elif not prefix_intact:
            missing = [name for name in required_prefix if not final_in_plate.get(name)]
            item["strict_rejection_reason"] = "prefix_not_intact:" + ",".join(missing)
        else:
            item["strict_rejection_reason"] = None

        counts[orange]["total"] += 1
        counts[orange]["target_success"] += int(target_success)
        counts[orange]["prefix_intact"] += int(prefix_intact)
        counts[orange]["strict_success"] += int(strict_success)
        if not strict_success:
            rejected.append(
                {
                    "lerobot_episode": item.get("lerobot_episode"),
                    "hdf5_episode": item.get("hdf5_episode"),
                    "orange": orange,
                    "reason": item["strict_rejection_reason"],
                    "final_in_plate": final_in_plate,
                    "target_max_lift_m": item.get("target_max_lift_m"),
                }
            )

    slices = result.get("slice_results", [])
    successful = sum(bool(item.get("success")) for item in slices)
    per_episode: dict[int, list[bool]] = defaultdict(list)
    for item in slices:
        per_episode[int(item["lerobot_episode"])].append(bool(item.get("success")))

    result["successful_slices"] = successful
    result["total_slices"] = len(slices)
    result["slice_success_rate"] = successful / len(slices) if slices else 0.0
    result["episodes_all_slices_successful"] = sum(all(values) for values in per_episode.values())
    result["strict_prefix_filter"] = {
        "enabled": True,
        "requirements": {key: list(value) for key, value in PREFIX_REQUIREMENTS.items()},
        "counts": {key: dict(value) for key, value in counts.items()},
        "rejected_slices": rejected,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Gate3 strict-prefix audit",
        "",
        f"Source: `{args.input}`",
        "",
        "| Phase | Total | Target success | Prefix intact | Strict success |",
        "|---|---:|---:|---:|---:|",
    ]
    for orange in PREFIX_REQUIREMENTS:
        row = counts[orange]
        lines.append(
            f"| {orange} | {row['total']} | {row['target_success']} | "
            f"{row['prefix_intact']} | {row['strict_success']} |"
        )
    lines += ["", "## Rejected slices", ""]
    if rejected:
        lines += [
            "| LeRobot episode | HDF5 episode | Phase | Reason | Final in plate |",
            "|---:|---:|---|---|---|",
        ]
        for item in rejected:
            lines.append(
                f"| {item['lerobot_episode']} | {item['hdf5_episode']} | {item['orange']} | "
                f"{item['reason']} | `{json.dumps(item['final_in_plate'], ensure_ascii=False)}` |"
            )
    else:
        lines.append("No slices were rejected by the strict-prefix rule.")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(result["strict_prefix_filter"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
