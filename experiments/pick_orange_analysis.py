#!/usr/bin/env python3
"""Pure, dependency-light analysis helpers for PickOrange experiments.

This module intentionally does not import Isaac Lab or LeRobot.  It can be used
while training is running to audit protocols and completed JSON summaries.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def protocol_horizon(
    policy_steps: int,
    sim_steps_per_action: int = 2,
    sim_dt_s: float = 1.0 / 60.0,
    decimation: int = 1,
) -> dict:
    sim_steps = policy_steps * sim_steps_per_action
    return {
        "policy_steps": policy_steps,
        "sim_steps_per_action": sim_steps_per_action,
        "simulation_steps": sim_steps,
        "sim_dt_s": sim_dt_s,
        "decimation": decimation,
        "theoretical_duration_s": sim_steps * sim_dt_s * decimation,
    }


def horizon_fairness(a0_steps: int = 1020, a1_steps_per_stage: int = 420, stages: int = 3) -> dict:
    a1_steps = a1_steps_per_stage * stages
    a0 = protocol_horizon(a0_steps)
    a1 = protocol_horizon(a1_steps)
    return {
        "a0": a0,
        "a1": a1,
        "equal_total_horizon": a0_steps == a1_steps,
        "a0_over_a1": a0_steps / a1_steps if a1_steps else None,
        "comparable_a0_policy_steps": a1_steps,
        "note": "The comparable horizon is an additional analysis condition; it does not replace the formal protocol.",
    }


def horizon_protocol_spec(name: str, sim_steps_per_action: int = 2) -> dict:
    if name not in {"native_horizon", "matched_horizon"}:
        raise ValueError(f"unknown horizon protocol: {name}")
    a0_actions = 1020 if name == "native_horizon" else 1260
    a1_actions = 420 * 3
    return {
        "name": name,
        "a0": protocol_horizon(a0_actions, sim_steps_per_action),
        "a1": protocol_horizon(a1_actions, sim_steps_per_action),
        "same_total_horizon": a0_actions == a1_actions,
    }


def post_success_overrun(stage_switch_step: int, first_stable_step: int | None) -> int | None:
    return None if first_stable_step is None else stage_switch_step - first_stable_step


def dataset_sampling_stats(
    episodes: int,
    frames: int,
    episode_lengths: Sequence[int] | None,
    chunk_size: int,
    batch_size: int,
    training_steps: int,
) -> dict:
    lengths = list(episode_lengths or [])
    full_windows = sum(max(0, length - chunk_size + 1) for length in lengths) if lengths else None
    anchors = frames  # LeRobot indexes every frame; future actions are boundary-padded.
    draws = batch_size * training_steps
    return {
        "episodes": episodes,
        "frames": frames,
        "chunk_size": chunk_size,
        "sample_anchor_windows": anchors,
        "full_unpadded_windows": full_windows,
        "optimizer_steps": training_steps,
        "batch_size": batch_size,
        "sample_draws": draws,
        "approx_anchor_exposures": draws / anchors if anchors else None,
        "sampling_assumption": "one anchor per frame; future chunk positions crossing episode end are boundary-padded",
    }


def matched_exposure_steps(reference: Mapping, target_anchor_windows: int) -> int | None:
    anchors = int(reference.get("sample_anchor_windows") or 0)
    steps = int(reference.get("optimizer_steps") or 0)
    if anchors <= 0 or target_anchor_windows <= 0:
        return None
    return max(1, round(steps * target_anchor_windows / anchors))


def isolated_sequential_gap(sequential_successes: int, isolated_successes: int, episodes: int) -> dict:
    if episodes <= 0:
        return {"episodes": 0, "gap_pp": None}
    return {
        "episodes": episodes,
        "sequential_rate": sequential_successes / episodes,
        "isolated_rate": isolated_successes / episodes,
        "gap_pp": 100.0 * (isolated_successes - sequential_successes) / episodes,
    }


def episode_key(row: Mapping) -> tuple:
    return (
        row.get("seed"),
        row.get("episode_index", row.get("episode")),
        row.get("initial_state_id"),
    )


def align_paired_results(left: Iterable[Mapping], right: Iterable[Mapping]) -> dict:
    left_map = {episode_key(row): row for row in left}
    right_map = {episode_key(row): row for row in right}
    shared = sorted(set(left_map) & set(right_map), key=str)
    return {
        "pairs": [(left_map[key], right_map[key]) for key in shared],
        "shared_keys": shared,
        "left_only": sorted(set(left_map) - set(right_map), key=str),
        "right_only": sorted(set(right_map) - set(left_map), key=str),
    }


def exact_mcnemar(left: Iterable[bool], right: Iterable[bool]) -> dict:
    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values):
        raise ValueError("McNemar test requires equal-length inputs")
    pairs = list(zip(left_values, right_values))
    left_only = sum(bool(a) and not bool(b) for a, b in pairs)
    right_only = sum(not bool(a) and bool(b) for a, b in pairs)
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(left_only, right_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "pairs": len(pairs),
        "left_success_right_failure": left_only,
        "left_failure_right_success": right_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def paired_bootstrap_difference(
    left: Sequence[float], right: Sequence[float], samples: int = 10000, seed: int = 2026
) -> dict:
    if len(left) != len(right):
        raise ValueError("paired bootstrap requires equal-length inputs")
    if not left:
        return {"pairs": 0, "difference": None, "ci_95": [None, None]}
    deltas = [float(b) - float(a) for a, b in zip(left, right)]
    rng = random.Random(seed)
    draws = [sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas) for _ in range(samples)]
    draws.sort()
    lo = draws[int(0.025 * (samples - 1))]
    hi = draws[int(0.975 * (samples - 1))]
    return {"pairs": len(deltas), "difference": sum(deltas) / len(deltas), "ci_95": [lo, hi], "samples": samples, "seed": seed}


def normalized_l2(value: Sequence[float], mean: Sequence[float], std: Sequence[float], eps: float = 1e-6) -> float:
    if not (len(value) == len(mean) == len(std)):
        raise ValueError("value, mean and std dimensions must match")
    if not value:
        return 0.0
    terms = [((float(v) - float(m)) / max(abs(float(s)), eps)) ** 2 for v, m, s in zip(value, mean, std)]
    return math.sqrt(sum(terms) / len(terms))


def stable_state_id(seed: int, episode_index: int, values: Sequence[float]) -> str:
    rounded = [round(float(value), 6) for value in values]
    raw = json.dumps([seed, episode_index, rounded], separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def exclusion_summary(rows: Iterable[Mapping]) -> dict:
    rows = list(rows)
    reasons = Counter(str(row.get("reason") or row.get("failure_reason") or "unspecified") for row in rows)
    return {"excluded": len(rows), "reasons": dict(sorted(reasons.items()))}
