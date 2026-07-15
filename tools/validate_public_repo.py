#!/usr/bin/env python3
"""Validate the curated public repository without simulator dependencies."""

from __future__ import annotations

import json
import py_compile
import re
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TEXT = [
    re.compile(r"/home/[A-Za-z0-9_.-]+/"),
    re.compile(r"192\.168\."),
    re.compile(r"(?i)(password|api[_-]?key|secret)\s*[:=]\s*[^\s$<{]+"),
]
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".mp4"}
MAX_PUBLIC_FILE = 10 * 1024 * 1024


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def check_summary(errors: list[str]) -> None:
    path = ROOT / "results/summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    best = max(data["final_full_task"], key=lambda row: row["rate"])
    if (best["method"], best["checkpoint"], best["successes"], best["episodes"]) != ("A1", "14k", 3, 20):
        fail("Unexpected best final result", errors)
    if data["b3_data_audit"]["strict_prefix_success"] != [28, 30]:
        fail("Unexpected B3 strict-prefix count", errors)
    if data["evaluation"]["native_horizon"]["same_total_horizon"]:
        fail("Native horizons must remain explicitly unmatched", errors)
    if data["evaluation"]["matched_horizon"]["enabled_by_default"]:
        fail("Matched horizon must remain opt-in", errors)
    if data["evaluation"]["matched_horizon"]["formal_20_episode_result_available"]:
        fail("No formal matched-horizon 20-episode result should be claimed", errors)
    if data["evaluated_rollout_inventory"]["total"] != 1160:
        fail("Unexpected historical rollout inventory", errors)
    if len(data["final_full_task"]) != 6 or any(row["checkpoint"] in {"21k", "7k"} for row in data["final_full_task"]):
        fail("Primary full-task results must contain only the six G4 cells", errors)
    if len(data["isolated_primitives"]) != 3 or any(row["checkpoint_family"] != "14k" for row in data["isolated_primitives"]):
        fail("Primary isolated results must contain only the three G4 14k cells", errors)
    if not data["historical_reference"]["not_a_direct_comparator"]:
        fail("Historical G3 results must be marked as non-direct comparators", errors)
    gate30 = next(item for item in data["training_generations"] if item["id"] == "G3")
    if gate30["legacy_full_task_results"]["A1"] != {"5k": [1, 20], "6k": [1, 20], "7k": [0, 20]}:
        fail("Unexpected Gate30 legacy checkpoint results", errors)
    rhc = data["rhc_b1_execution_horizon_ablation"]
    if rhc["selected_horizon_h_star"] != 100:
        fail("Unexpected RHC H* selection", errors)
    if [(row["execution_horizon"], row["successes"]) for row in rhc["results"]] != [
        (100, 5), (25, 3), (10, 2), (1, 1)
    ]:
        fail("Unexpected RHC B1 results", errors)
    if rhc["pairing"]["status"] != "PAIRING_INVALID":
        fail("RHC comparison must remain marked as unpaired", errors)
    if rhc["full_task_followup"]["executed"]:
        fail("Redundant full-task RHC follow-up must not be claimed", errors)
    temporal = data["temporal_aggregation_b1_paired"]
    if temporal["status"] != "STRICT_PHYSICAL_PAIRING_20_OF_20":
        fail("Temporal-aggregation comparison must remain strictly paired", errors)
    if [(row["name"], row["successes"]) for row in temporal["groups"]] != [
        ("H100_off", 5), ("H1_off", 0), ("H1_aggregation_001", 5)
    ]:
        fail("Unexpected paired temporal-aggregation results", errors)
    if temporal["pairwise_exact_mcnemar"]["H100_off_vs_H1_aggregation_001"]["p_two_sided"] != 1.0:
        fail("Unexpected H=100 versus aggregation paired result", errors)
    paired_rows = [
        json.loads(line)
        for line in (ROOT / "results/raw/temporal_aggregation_paired_episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(paired_rows) != 20 or len({row["initialization_id"] for row in paired_rows}) != 20:
        fail("Expected 20 unique paired temporal-aggregation episode rows", errors)


def check_python(errors: list[str]) -> None:
    paths = sorted((ROOT / "experiments").rglob("*.py")) + sorted((ROOT / "tools").rglob("*.py"))
    with tempfile.TemporaryDirectory(prefix="pickorange-pycompile-") as temporary:
        for index, path in enumerate(paths):
            try:
                py_compile.compile(str(path), cfile=str(Path(temporary) / f"{index}.pyc"), doraise=True)
            except py_compile.PyCompileError as exc:
                fail(f"Python compile failed: {path.relative_to(ROOT)}: {exc}", errors)


def check_files(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_PUBLIC_FILE:
            fail(f"File exceeds 10 MiB: {relative}", errors)
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in FORBIDDEN_TEXT:
            if pattern.search(content):
                fail(f"Potential private value in {relative}: {pattern.pattern}", errors)


def check_markdown_links(errors: list[str]) -> None:
    pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
    for path in ROOT.rglob("*.md"):
        content = path.read_text(encoding="utf-8")
        for target in pattern.findall(content):
            target = target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            if ROOT.resolve() not in resolved.parents and resolved != ROOT.resolve():
                fail(f"Link escapes repository: {path.relative_to(ROOT)} -> {target}", errors)
            elif not resolved.exists():
                fail(f"Broken local link: {path.relative_to(ROOT)} -> {target}", errors)


def main() -> int:
    errors: list[str] = []
    check_summary(errors)
    check_python(errors)
    check_files(errors)
    check_markdown_links(errors)
    if errors:
        print("FAIL")
        for item in errors:
            print(f"- {item}")
        return 1
    print("PASS: summary, Python, file size, privacy patterns and Markdown links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
