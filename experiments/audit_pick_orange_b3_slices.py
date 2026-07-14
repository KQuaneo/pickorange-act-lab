#!/usr/bin/env python3
"""Audit every Gate3 Orange003 slice numerically and visually.

The report checks target success, prefix preservation, frame/action integrity,
source-video coverage, and produces side-by-side front/wrist clips plus contact
sheets for all Orange003 slices (including rejected slices).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


VIDEO_KEYS = ("observation.images.front", "observation.images.wrist")
PREFIX = ("Orange001", "Orange002")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, choices=(30, 50), required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--contact-points", type=int, default=5)
    parser.add_argument("--episodes-per-page", type=int, default=4)
    parser.add_argument("--skip-clips", action="store_true")
    return parser.parse_args()


def max_abs_delta(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(values, axis=0))))


def decode_segment(path: Path, start: float, end: float) -> list[tuple[float, np.ndarray]]:
    frames: list[tuple[float, np.ndarray]] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        seek_time = max(0.0, start - 1.0)
        container.seek(int(seek_time / float(stream.time_base)), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            if timestamp + 1e-4 < start:
                continue
            if timestamp >= end - 1e-4:
                break
            frames.append((timestamp, frame.to_ndarray(format="rgb24")))
    return frames


def resize_rgb(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(array).resize(size, Image.Resampling.LANCZOS))


def write_clip(path: Path, front: list[tuple[float, np.ndarray]], wrist: list[tuple[float, np.ndarray]], fps: int) -> int:
    count = min(len(front), len(wrist))
    if count == 0:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.mp4")
    if temporary.exists():
        temporary.unlink()
    with av.open(str(temporary), mode="w") as output:
        stream = output.add_stream("libx264", rate=fps)
        stream.width = 1280
        stream.height = 480
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "24", "preset": "veryfast"}
        for index in range(count):
            left = resize_rgb(front[index][1], (640, 480))
            right = resize_rgb(wrist[index][1], (640, 480))
            canvas = np.concatenate([left, right], axis=1)
            frame = av.VideoFrame.from_ndarray(canvas, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    os.replace(temporary, path)
    return count


def label_font(size: int = 22) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def write_contact_sheet(
    path: Path,
    episode: int,
    status: str,
    front: list[tuple[float, np.ndarray]],
    wrist: list[tuple[float, np.ndarray]],
    points: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 320, 240
    header = 42
    sheet = Image.new("RGB", (points * width, header + 2 * height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), f"source episode {episode:03d} | {status} | front / wrist", fill="black", font=label_font())
    count = min(len(front), len(wrist))
    indices = np.linspace(0, max(0, count - 1), points).round().astype(int) if count else np.zeros(points, dtype=int)
    for column, index in enumerate(indices):
        if count:
            front_image = Image.fromarray(front[int(index)][1]).resize((width, height), Image.Resampling.LANCZOS)
            wrist_image = Image.fromarray(wrist[int(index)][1]).resize((width, height), Image.Resampling.LANCZOS)
            sheet.paste(front_image, (column * width, header))
            sheet.paste(wrist_image, (column * width, header + height))
            draw.text((column * width + 4, header + 4), f"{column + 1}/{points}", fill="white", stroke_fill="black", stroke_width=2)
    sheet.save(path)


def make_overview_pages(contact_paths: list[Path], output_dir: Path, per_page: int) -> list[Path]:
    pages = []
    for page_index in range(math.ceil(len(contact_paths) / per_page)):
        chunk = contact_paths[page_index * per_page : (page_index + 1) * per_page]
        images = [Image.open(path).convert("RGB") for path in chunk]
        if not images:
            continue
        width = max(image.width for image in images)
        height = sum(image.height for image in images)
        page = Image.new("RGB", (width, height), "white")
        cursor = 0
        for image in images:
            page.paste(image, (0, cursor))
            cursor += image.height
        output = output_dir / f"overview_page_{page_index + 1:02d}.jpg"
        page.save(output, quality=88)
        pages.append(output)
        for image in images:
            image.close()
    return pages


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    tag = f"gate_exp{args.count}"
    gate3_path = root / f"outputs/eval/pick_orange_{tag}/gate3_event_slice_replay/summary.json"
    output_root = root / f"outputs/audits/pick_orange_{tag}_b3"
    output_root.mkdir(parents=True, exist_ok=True)
    clips_dir = output_root / "clips"
    contacts_dir = output_root / "contact_sheets"

    gate3 = json.loads(gate3_path.read_text(encoding="utf-8"))
    source_parquet = root / gate3["lerobot_parquet"]
    source_root = source_parquet.parents[2]
    source_table = pq.read_table(source_parquet)
    source_episodes = pd.read_parquet(source_root / "meta/episodes/chunk-000/file-000.parquet")
    source_info = json.loads((source_root / "meta/info.json").read_text(encoding="utf-8"))
    fps = int(source_info["fps"])

    b3_rows = sorted(
        (row for row in gate3["slice_results"] if row["orange"] == "Orange003"),
        key=lambda row: int(row["lerobot_episode"]),
    )
    audits: list[dict[str, Any]] = []
    contact_paths: list[Path] = []

    for row in b3_rows:
        episode = int(row["lerobot_episode"])
        start = int(row["lerobot_start"])
        end = int(row["lerobot_end_exclusive"])
        expected_frames = end - start
        source_row = source_episodes[source_episodes["episode_index"] == episode].iloc[0]
        mask = pc.and_(
            pc.equal(source_table["episode_index"], episode),
            pc.and_(pc.greater_equal(source_table["frame_index"], start), pc.less(source_table["frame_index"], end)),
        )
        segment = source_table.filter(mask)
        actions = np.asarray(segment["action"].to_pylist(), dtype=np.float32)
        states = np.asarray(segment["observation.state"].to_pylist(), dtype=np.float32)
        timestamps = np.asarray(segment["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64)
        target_success = bool(row.get("success"))
        final_in_plate = row.get("final_in_plate") or {}
        prefix_intact = all(bool(final_in_plate.get(name)) for name in PREFIX)
        strict_success = target_success and prefix_intact
        if not target_success:
            status = "REJECT target"
        elif not prefix_intact:
            status = "REJECT prefix"
        else:
            status = "STRICT PASS"

        decoded: dict[str, list[tuple[float, np.ndarray]]] = {}
        video_paths: dict[str, str] = {}
        video_windows: dict[str, list[float]] = {}
        for key in VIDEO_KEYS:
            prefix = f"videos/{key}"
            file_index = int(source_row[f"{prefix}/file_index"])
            chunk_index = int(source_row[f"{prefix}/chunk_index"])
            source_start = float(source_row[f"{prefix}/from_timestamp"])
            window_start = source_start + start / fps
            window_end = source_start + end / fps
            video = source_root / f"videos/{key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            video_paths[key] = str(video)
            video_windows[key] = [window_start, window_end]
            decoded[key] = decode_segment(video, window_start, window_end)

        front = decoded[VIDEO_KEYS[0]]
        wrist = decoded[VIDEO_KEYS[1]]
        contact_path = contacts_dir / f"episode_{episode:03d}.jpg"
        write_contact_sheet(contact_path, episode, status, front, wrist, args.contact_points)
        contact_paths.append(contact_path)

        clip_path = clips_dir / f"episode_{episode:03d}.mp4"
        clip_frames = 0
        if not args.skip_clips:
            clip_frames = write_clip(clip_path, front, wrist, fps)

        timestamp_step_error = 0.0
        if len(timestamps) > 1:
            timestamp_step_error = float(np.max(np.abs(np.diff(timestamps) - 1.0 / fps)))
        audit = {
            "lerobot_episode": episode,
            "hdf5_episode": int(row["hdf5_episode"]),
            "status": status,
            "target_success": target_success,
            "prefix_intact": prefix_intact,
            "strict_success": strict_success,
            "final_in_plate": final_in_plate,
            "target_max_lift_m": row.get("target_max_lift_m"),
            "expected_frames": expected_frames,
            "parquet_frames": int(segment.num_rows),
            "front_decoded_frames": len(front),
            "wrist_decoded_frames": len(wrist),
            "clip_frames": clip_frames,
            "finite_actions": bool(np.isfinite(actions).all()),
            "finite_states": bool(np.isfinite(states).all()),
            "max_action_step_delta": max_abs_delta(actions),
            "max_state_step_delta": max_abs_delta(states),
            "timestamp_step_max_error": timestamp_step_error,
            "video_paths": video_paths,
            "video_windows": video_windows,
            "contact_sheet": str(contact_path),
            "clip": str(clip_path) if not args.skip_clips else None,
        }
        audit["integrity_pass"] = bool(
            audit["parquet_frames"] == expected_frames
            and audit["finite_actions"]
            and audit["finite_states"]
            and abs(audit["front_decoded_frames"] - expected_frames) <= 2
            and abs(audit["wrist_decoded_frames"] - expected_frames) <= 2
            and audit["timestamp_step_max_error"] < 1e-4
        )
        audits.append(audit)
        print(
            f"episode={episode:03d} {status:13s} frames={segment.num_rows}/{expected_frames} "
            f"video={len(front)}/{len(wrist)} lift={row.get('target_max_lift_m')}"
        )

    pages = make_overview_pages(contact_paths, output_root, args.episodes_per_page)
    strict_passes = sum(item["strict_success"] for item in audits)
    integrity_passes = sum(item["integrity_pass"] for item in audits)
    payload = {
        "count": args.count,
        "gate3_summary": str(gate3_path),
        "source_lerobot": str(source_root),
        "orange003_slices": len(audits),
        "target_successes": sum(item["target_success"] for item in audits),
        "prefix_intact": sum(item["prefix_intact"] for item in audits),
        "strict_successes": strict_passes,
        "integrity_passes": integrity_passes,
        "status_counts": dict(Counter(item["status"] for item in audits)),
        "overview_pages": [str(path) for path in pages],
        "episodes": audits,
    }
    (output_root / "audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# Orange003 / B3 Gate3 audit ({tag})",
        "",
        f"- Gate3 Orange003 slices: **{len(audits)}**",
        f"- Target-only successes: **{payload['target_successes']}**",
        f"- Strict prefix-preserving successes: **{strict_passes}**",
        f"- Numeric/video integrity passes: **{integrity_passes}/{len(audits)}**",
        "",
        "A strict B3 slice requires Orange003 success and Orange001/Orange002 still in the plate.",
        "",
        "| Episode | Status | Final in plate | Lift (m) | Frames parquet/front/wrist | Max action delta |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for item in audits:
        lines.append(
            f"| {item['lerobot_episode']} | {item['status']} | "
            f"`{json.dumps(item['final_in_plate'], ensure_ascii=False)}` | "
            f"{float(item['target_max_lift_m'] or 0):.4f} | "
            f"{item['parquet_frames']}/{item['front_decoded_frames']}/{item['wrist_decoded_frames']} | "
            f"{item['max_action_step_delta']:.4f} |"
        )
    lines += ["", "## Visual pages", ""]
    lines.extend(f"- `{path}`" for path in pages)
    (output_root / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if integrity_passes != len(audits):
        raise SystemExit(f"B3 integrity audit failed: {integrity_passes}/{len(audits)}")
    if strict_passes <= 0:
        raise SystemExit("B3 strict-prefix audit left no usable slices")
    (output_root / "DONE").write_text(
        f"count={args.count}\nstrict_successes={strict_passes}\nintegrity_passes={integrity_passes}\n",
        encoding="utf-8",
    )
    print(f"Audit complete: {output_root}")


if __name__ == "__main__":
    main()
