#!/usr/bin/env python3
"""Render repository result charts with only the Python standard library."""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summary.json"
ASSETS = ROOT / "assets"


def text(x: float, y: float, value: str, *, size: int = 15, anchor: str = "middle", weight: int = 400, fill: str = "#e5e7eb") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Inter,Arial,sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escape(value)}</text>'
    )


def document(title: str, description: str, body: list[str], width: int = 960, height: int = 520) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f"<title id=\"title\">{escape(title)}</title>",
            f"<desc id=\"desc\">{escape(description)}</desc>",
            f'<rect width="{width}" height="{height}" rx="18" fill="#0b1220"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def full_task(data: dict) -> str:
    rows = data["final_full_task"]
    a0 = [row for row in rows if row["method"] == "A0"]
    a1 = [row for row in rows if row["method"] == "A1"]
    width, height = 960, 520
    left, top, plot_w, plot_h = 92, 92, 808, 330
    ymax = 0.40
    out = [text(left, 42, "G4 full-task success (20 episodes per checkpoint)", size=22, anchor="start", weight=500)]
    out.append(text(left, 68, "Primary G4 benchmark · Wilson 95% intervals · native horizon", size=13, anchor="start", fill="#9ca3af"))
    for tick in range(0, 41, 10):
        y = top + plot_h * (1 - tick / 40)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#334155" stroke-width="1"/>')
        out.append(text(left - 14, y + 5, f"{tick}%", size=12, anchor="end", fill="#9ca3af"))
    group_w = plot_w / len(a0)
    bar_w = 44
    for idx, (r0, r1) in enumerate(zip(a0, a1)):
        center = left + group_w * (idx + 0.5)
        for row, x, color in [(r0, center - 30, "#f59e0b"), (r1, center + 30, "#38bdf8")]:
            rate = row["rate"]
            y = top + plot_h * (1 - rate / ymax)
            base = top + plot_h
            bar_h = max(2, base - y)
            out.append(f'<rect x="{x - bar_w/2:.1f}" y="{base - bar_h:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="5" fill="{color}"/>')
            low, high = row["wilson95"]
            low_y = top + plot_h * (1 - low / ymax)
            high_y = top + plot_h * (1 - high / ymax)
            out.append(f'<line x1="{x}" y1="{high_y:.1f}" x2="{x}" y2="{low_y:.1f}" stroke="#f8fafc" stroke-width="2"/>')
            out.append(f'<line x1="{x-8}" y1="{high_y:.1f}" x2="{x+8}" y2="{high_y:.1f}" stroke="#f8fafc" stroke-width="2"/>')
            out.append(f'<line x1="{x-8}" y1="{low_y:.1f}" x2="{x+8}" y2="{low_y:.1f}" stroke="#f8fafc" stroke-width="2"/>')
            out.append(text(x, max(top + 15, y - 10), f'{row["successes"]}/20', size=13, weight=500))
        out.append(text(center, top + plot_h + 30, f'{r0["checkpoint"]} / {r1["checkpoint"]}', size=13))
    out.append(text(left + plot_w / 2, height - 42, "A0 checkpoint / A1 per-primitive checkpoint", size=13, fill="#9ca3af"))
    out.append(f'<rect x="{left + 560}" y="44" width="14" height="14" rx="3" fill="#f59e0b"/>')
    out.append(text(left + 582, 56, "A0 monolithic", size=13, anchor="start"))
    out.append(f'<rect x="{left + 690}" y="44" width="14" height="14" rx="3" fill="#38bdf8"/>')
    out.append(text(left + 712, 56, "A1 3-policy", size=13, anchor="start"))
    return document(
        "Final full-task success rates",
        "A0 observed zero successes at every checkpoint. A1 observed 2 of 20 at 10k and 3 of 20 at 14k. Wilson intervals are wide.",
        out,
        width,
        height,
    )


def isolated(data: dict) -> str:
    rows = data["isolated_primitives"]
    final = {row["stage"]: row for row in rows if row["checkpoint_family"] == "14k"}
    width, height = 960, 500
    left, top, plot_w, plot_h = 92, 94, 808, 300
    ymax = 0.60
    out = [text(left, 42, "G4 isolated primitive success — 14k", size=22, anchor="start", weight=500)]
    out.append(text(left, 68, "20 episodes per stage · B2/B3 use oracle initialization", size=13, anchor="start", fill="#9ca3af"))
    for tick in range(0, 61, 10):
        y = top + plot_h * (1 - tick / 60)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#334155" stroke-width="1"/>')
        out.append(text(left - 14, y + 5, f"{tick}%", size=12, anchor="end", fill="#9ca3af"))
    group_w = plot_w / 3
    for idx, stage in enumerate(["B1", "B2", "B3"]):
        center = left + group_w * (idx + 0.5)
        row = final[stage]
        rate = row["rate"]
        y = top + plot_h * (1 - rate / ymax)
        base = top + plot_h
        out.append(f'<rect x="{center-32:.1f}" y="{y:.1f}" width="64" height="{base-y:.1f}" rx="5" fill="#34d399"/>')
        out.append(text(center, y - 10, f'{row["successes"]}/20', size=14, weight=500))
        label = stage if stage == "B1" else f"{stage} (oracle init)"
        out.append(text(center, top + plot_h + 32, label, size=14))
    out.append(f'<rect x="{left + 716}" y="45" width="14" height="14" rx="3" fill="#34d399"/>')
    out.append(text(left + 738, 57, "G4 14k", size=13, anchor="start"))
    out.append(text(left + plot_w / 2, height - 42, "Primitive policy", size=13, fill="#9ca3af"))
    return document(
        "Isolated primitive success rates",
        "G4 14k primitive checkpoints only. B2 and B3 use oracle initialization and are capability upper bounds.",
        out,
        width,
        height,
    )


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "final-full-task-results.svg").write_text(full_task(data), encoding="utf-8")
    (ASSETS / "isolated-primitive-results.svg").write_text(isolated(data), encoding="utf-8")
    print("Rendered 2 charts from results/summary.json")


if __name__ == "__main__":
    main()
