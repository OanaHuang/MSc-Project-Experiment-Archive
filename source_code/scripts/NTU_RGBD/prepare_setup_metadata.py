from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.NTU_RGBD.core.config import (
    NTU_DEFAULT_FPS, NTU_DEFAULT_RGB_HEIGHT, NTU_DEFAULT_RGB_WIDTH,
)
from scripts.NTU_RGBD.core.skeleton_reader import read_skeleton_summary


SAMPLE_PATTERN = re.compile(
    r"S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<performer>\d{3})"
    r"R(?P<replication>\d{3})A(?P<action>\d{3})"
)
FIELDS = (
    "sample_id", "setup", "camera", "performer", "replication", "action",
    "rgb_path", "skeleton_path", "rgb_frames", "skeleton_frames",
    "frame_difference", "frame_count_match", "fps", "width", "height",
    "max_bodies", "empty_frames", "is_single_person",
)
EXCLUSION_FIELDS = (
    "sample_id", "reason", "detail", "frame_dir", "skeleton_path",
)


def parse_sample_id(sample_id: str) -> dict[str, int]:
    match = SAMPLE_PATTERN.fullmatch(sample_id)
    if match is None:
        raise ValueError(f"Invalid NTU sample id: {sample_id}")
    return {key: int(value) for key, value in match.groupdict().items()}


def read_frame_marker(sample_dir: Path) -> dict:
    marker = sample_dir / ".frames_complete.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Missing or invalid frame marker: {marker}") from error
    if int(payload.get("saved_frames", 0)) < 1:
        raise RuntimeError(f"Frame marker contains no frames: {marker}")
    return payload


def exclusion_row(sample_id: str, reason: str, detail: str,
                  frame_dir: Path, skeleton_path: Path) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "reason": reason,
        "detail": detail,
        "frame_dir": str(frame_dir.resolve()),
        "skeleton_path": str(skeleton_path.resolve()),
    }


def build_rows(frame_root: Path, video_root: Path, skeleton_root: Path,
               setup: str, tolerance: int) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    exclusions: list[dict] = []
    for sample_dir in sorted(path for path in frame_root.glob(f"{setup}*")
                             if path.is_dir()):
        sample_id = sample_dir.name
        identity = parse_sample_id(sample_id)
        frame_marker = read_frame_marker(sample_dir)
        rgb_frames = int(frame_marker.get("source_frames",
                        frame_marker["saved_frames"]))
        skeleton_path = skeleton_root / f"{sample_id}.skeleton"
        if not skeleton_path.is_file():
            exclusions.append(exclusion_row(
                sample_id, "missing_skeleton", "Skeleton file does not exist",
                sample_dir, skeleton_path,
            ))
            continue
        try:
            summary = read_skeleton_summary(skeleton_path)
        except (OSError, UnicodeError, ValueError) as error:
            exclusions.append(exclusion_row(
                sample_id, "invalid_skeleton",
                f"{type(error).__name__}: {error}", sample_dir, skeleton_path,
            ))
            continue
        skeleton_frames = int(summary["num_frames"])
        if skeleton_frames <= 0:
            exclusions.append(exclusion_row(
                sample_id, "empty_skeleton", "Skeleton contains zero frames",
                sample_dir, skeleton_path,
            ))
            continue
        if int(summary["empty_frames"]) >= skeleton_frames:
            exclusions.append(exclusion_row(
                sample_id, "empty_skeleton", "All skeleton frames contain no body",
                sample_dir, skeleton_path,
            ))
            continue
        difference = abs(rgb_frames - skeleton_frames)
        rows.append({
            "sample_id": sample_id,
            **identity,
            "rgb_path": str((video_root / f"{sample_id}_rgb.avi").resolve()),
            "skeleton_path": str(skeleton_path.resolve()),
            "rgb_frames": rgb_frames,
            "skeleton_frames": skeleton_frames,
            "frame_difference": difference,
            "frame_count_match": difference <= tolerance,
            "fps": NTU_DEFAULT_FPS,
            "width": NTU_DEFAULT_RGB_WIDTH,
            "height": NTU_DEFAULT_RGB_HEIGHT,
            "max_bodies": summary["max_bodies"],
            "empty_frames": summary["empty_frames"],
            "is_single_person": summary["is_single_person"],
        })
    if not rows:
        raise RuntimeError(
            f"No skeleton-valid completed {setup} samples found in {frame_root}"
        )
    return rows, exclusions


def filter_usable_rows(rows: list[dict], frame_root: Path) \
        -> tuple[list[dict], list[dict]]:
    usable: list[dict] = []
    exclusions: list[dict] = []
    for row in rows:
        reasons = []
        if not row["is_single_person"]:
            reasons.append("multiple_bodies")
        if not row["frame_count_match"]:
            reasons.append("frame_count_mismatch")
        if reasons:
            sample_id = str(row["sample_id"])
            exclusions.append(exclusion_row(
                sample_id, ";".join(reasons),
                (
                    f"rgb_frames={row['rgb_frames']}, "
                    f"skeleton_frames={row['skeleton_frames']}, "
                    f"max_bodies={row['max_bodies']}"
                ),
                frame_root / sample_id, Path(str(row["skeleton_path"])),
            ))
        else:
            usable.append(row)
    return usable, exclusions


def write_csv(path: Path, rows: list[dict],
              fieldnames: tuple[str, ...] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build video-level NTU setup splits")
    parser.add_argument("--setup", default="S010")
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--skeleton-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-count-tolerance", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, skeleton_exclusions = build_rows(
        args.frame_root.resolve(), args.video_root.resolve(),
        args.skeleton_root.resolve(), args.setup, args.frame_count_tolerance,
    )
    usable, quality_exclusions = filter_usable_rows(rows, args.frame_root.resolve())
    exclusions = skeleton_exclusions + quality_exclusions
    if not usable:
        raise RuntimeError("No usable single-person frame-matched samples")
    random.Random(args.seed).shuffle(usable)
    train_end = int(len(usable) * 0.70)
    validation_end = train_end + int(len(usable) * 0.15)
    splits = {
        "train_split.csv": usable[:train_end],
        "val_split.csv": usable[train_end:validation_end],
        "test_split.csv": usable[validation_end:],
    }
    write_csv(args.output_dir / "matched_samples.csv", rows)
    write_csv(
        args.output_dir / "excluded_samples.csv", exclusions, EXCLUSION_FIELDS,
    )
    for name, split in splits.items():
        write_csv(args.output_dir / name, split)
    exclusion_counts: dict[str, int] = {}
    for item in exclusions:
        reason = str(item["reason"])
        exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
    summary = {
        "setup": args.setup,
        "all_samples": len(rows) + len(skeleton_exclusions),
        "skeleton_valid_samples": len(rows),
        "excluded_samples": len(exclusions),
        "exclusion_counts": exclusion_counts,
        "usable_single_person_samples": len(usable),
        "train_samples": len(splits["train_split.csv"]),
        "validation_samples": len(splits["val_split.csv"]),
        "test_samples": len(splits["test_split.csv"]),
        "split_unit": "video",
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
