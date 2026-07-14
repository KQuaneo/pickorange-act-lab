#!/usr/bin/env python3
"""Audit nominal, gated, strict, and ACT-effective PickOrange data counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pick_orange_analysis import dataset_sampling_stats, exclusion_summary


ROOT = Path(__file__).resolve().parents[1]


def load_optional(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def parquet_files(dataset: Path) -> list[Path]:
    return sorted((dataset / "data").glob("chunk-*/*.parquet"))


def dataset_row(name: str, dataset: Path, steps: int, chunk_size: int, batch_size: int) -> tuple[dict, dict]:
    info = load_optional(dataset / "meta/info.json")
    files = parquet_files(dataset)
    if not info or not files:
        return {"name": name, "path": str(dataset), "status": "missing"}, {}
    table = pd.concat([pd.read_parquet(path, columns=["episode_index", "frame_index", "observation.state"]) for path in files], ignore_index=True)
    lengths = table.groupby("episode_index").size().astype(int).tolist()
    stats = dataset_sampling_stats(int(info["total_episodes"]), int(info["total_frames"]), lengths, chunk_size, batch_size, steps)
    starts = table.sort_values(["episode_index", "frame_index"]).groupby("episode_index", as_index=False).first()
    states = np.stack(starts["observation.state"].map(lambda value: np.asarray(value, dtype=float)).tolist())
    reference = {
        "joint_mean": states.mean(axis=0).tolist(),
        "joint_std": states.std(axis=0).tolist(),
        "episodes": int(len(states)),
        "definition": "first observation.state frame of every retained episode",
    }
    return {"name": name, "path": str(dataset), "status": "ok", **stats}, reference


def collect_exclusions(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        rows = []
        for key, value in payload.items():
            if key in {"rejected_slices", "excluded_slices", "exclusions"} and isinstance(value, list):
                rows.extend(item if isinstance(item, dict) else {"reason": str(item)} for item in value)
            else:
                rows.extend(collect_exclusions(value))
        return rows
    if isinstance(payload, list):
        rows = []
        for value in payload:
            rows.extend(collect_exclusions(value))
        return rows
    return []


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--expert-count", type=int, choices=(30, 50), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    tag = f"gate_exp{args.expert_count}"
    output = args.output_dir or args.root / f"outputs/audits/pick_orange_{tag}_effective_data"
    datasets = {
        "A0": (args.root / f"data/lerobot/local/so101_pick_orange_{tag}_a0_joint6_v0", 42000),
        "Orange001": (args.root / f"data/lerobot/local/so101_pick_orange_{tag}_a1_event_prefixstrict_orange001_joint6_v0", 14000),
        "Orange002": (args.root / f"data/lerobot/local/so101_pick_orange_{tag}_a1_event_prefixstrict_orange002_joint6_v0", 14000),
        "Orange003": (args.root / f"data/lerobot/local/so101_pick_orange_{tag}_a1_event_prefixstrict_orange003_joint6_v0", 14000),
    }
    rows, references = [], {}
    for name, (path, steps) in datasets.items():
        row, reference = dataset_row(name, path, steps, args.chunk_size, args.batch_size)
        rows.append(row)
        if reference:
            references[name] = reference
    gate2 = load_optional(args.root / f"outputs/eval/pick_orange_{tag}/gate2_full_replay/summary.json")
    gate3 = load_optional(args.root / f"outputs/eval/pick_orange_{tag}/gate3_event_slice_replay/summary.json")
    b3 = load_optional(args.root / f"outputs/audits/pick_orange_{tag}_b3/audit.json")
    strict = load_optional(args.root / f"outputs/audits/pick_orange_{tag}_b3/gate3_strict_prefix_summary.json")
    exclusions = collect_exclusions(strict)
    payload = {
        "schema_version": 1,
        "expert_count_nominal": args.expert_count,
        "gate2": gate2,
        "gate3": gate3,
        "b3_audit": b3,
        "strict_prefix": strict,
        "exclusions": exclusion_summary(exclusions),
        "datasets": rows,
        "act_sampling_note": "ACT training indexes every frame as an anchor and boundary-pads future chunk positions; full_unpadded_windows is reported separately.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "effective_data.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "stage_start_references.json").write_text(json.dumps(references, indent=2), encoding="utf-8")
    lines = [
        f"# PickOrange effective data audit ({args.expert_count} nominal experts)", "",
        "| Dataset | Retained episodes | Frames / anchors | Full unpadded windows | Steps | Approx. anchor exposures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"| {row['name']} | missing | - | - | - | - |")
        else:
            lines.append(f"| {row['name']} | {row['episodes']} | {row['frames']} | {row['full_unpadded_windows']} | {row['optimizer_steps']} | {row['approx_anchor_exposures']:.2f}× |")
    lines += ["", f"Strict-prefix exclusions found: **{payload['exclusions']['excluded']}**", "", f"Reasons: `{json.dumps(payload['exclusions']['reasons'], ensure_ascii=False)}`", ""]
    (output / "effective_data.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "effective_data.md")


if __name__ == "__main__":
    main()
