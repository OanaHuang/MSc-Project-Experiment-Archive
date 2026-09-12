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
    NTU60_CROSS_SUBJECT_TRAIN_PERFORMERS,
    NTU60_CROSS_VIEW_TRAIN_CAMERAS,
    NTU_DEFAULT_FPS,
    NTU_DEFAULT_RGB_HEIGHT,
    NTU_DEFAULT_RGB_WIDTH,
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


def identity(sample_id: str) -> dict[str, int]:
    match = SAMPLE_PATTERN.fullmatch(sample_id)
    if match is None:
        raise ValueError(f"invalid NTU60 sample id: {sample_id}")
    return {key: int(value) for key, value in match.groupdict().items()}


def project_relative(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError as error:
        raise ValueError(f"dataset path must be inside project root: {path}") from error


def sampled_records(frame_root: Path, clip_subdir: str) -> list[dict]:
    records: list[dict] = []
    for manifest_path in sorted(frame_root.glob(f"S*/{clip_subdir}/samples.jsonl")):
        with manifest_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"no {clip_subdir} sample manifests found below {frame_root}")
    ids = [str(item["sample_id"]) for item in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate sample ids across contiguous manifests")
    return records


def skeleton_index(skeleton_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in skeleton_root.rglob("*.skeleton"):
        if path.stem in result:
            raise RuntimeError(f"duplicate skeleton file for {path.stem}")
        result[path.stem] = path
    if not result:
        raise RuntimeError(f"no skeleton files found below {skeleton_root}")
    return result


def build_rows(
    records: list[dict], skeletons: dict[str, Path], project_root: Path,
    frame_tolerance: int,
) -> tuple[list[dict], list[dict]]:
    usable: list[dict] = []
    excluded: list[dict] = []
    for index, record in enumerate(records, 1):
        sample_id = str(record["sample_id"])
        skeleton_path = skeletons.get(sample_id)
        if skeleton_path is None:
            excluded.append({"sample_id": sample_id, "reason": "missing_skeleton"})
            continue
        try:
            summary = read_skeleton_summary(skeleton_path)
        except (OSError, UnicodeError, ValueError) as error:
            excluded.append({
                "sample_id": sample_id,
                "reason": f"invalid_skeleton:{type(error).__name__}",
            })
            continue
        source_frames = int(record["source_frames"])
        skeleton_frames = int(summary["num_frames"])
        difference = abs(source_frames - skeleton_frames)
        reasons = []
        if not bool(summary["is_single_person"]):
            reasons.append("multiple_bodies")
        if skeleton_frames <= 0 or int(summary["empty_frames"]) >= skeleton_frames:
            reasons.append("empty_skeleton")
        if difference > frame_tolerance:
            reasons.append("frame_count_mismatch")
        if reasons:
            excluded.append({"sample_id": sample_id, "reason": ";".join(reasons)})
            continue
        parsed = identity(sample_id)
        usable.append({
            "sample_id": sample_id,
            **parsed,
            "rgb_path": "",
            "skeleton_path": project_relative(skeleton_path, project_root),
            "rgb_frames": source_frames,
            "skeleton_frames": skeleton_frames,
            "frame_difference": difference,
            "frame_count_match": True,
            "fps": NTU_DEFAULT_FPS,
            "width": NTU_DEFAULT_RGB_WIDTH,
            "height": NTU_DEFAULT_RGB_HEIGHT,
            "max_bodies": summary["max_bodies"],
            "empty_frames": summary["empty_frames"],
            "is_single_person": True,
        })
        if index == 1 or index % 5000 == 0 or index == len(records):
            print(f"[metadata] {index}/{len(records)}", flush=True)
    return usable, excluded


def grouped_protocol_split(
    rows: list[dict], protocol: str, validation_fraction: float, seed: int,
) -> dict[str, list[dict]]:
    if protocol == "xsub":
        official_train = [
            row for row in rows
            if int(row["performer"]) in NTU60_CROSS_SUBJECT_TRAIN_PERFORMERS
        ]
        test = [
            row for row in rows
            if int(row["performer"]) not in NTU60_CROSS_SUBJECT_TRAIN_PERFORMERS
        ]
    elif protocol == "xview":
        official_train = [
            row for row in rows
            if int(row["camera"]) in NTU60_CROSS_VIEW_TRAIN_CAMERAS
        ]
        test = [
            row for row in rows
            if int(row["camera"]) not in NTU60_CROSS_VIEW_TRAIN_CAMERAS
        ]
    else:
        raise ValueError(f"unsupported protocol: {protocol}")

    performers = sorted({int(row["performer"]) for row in official_train})
    validation_count = max(1, round(len(performers) * validation_fraction))
    validation_performers = set(random.Random(seed).sample(performers, validation_count))
    validation = [
        row for row in official_train if int(row["performer"]) in validation_performers
    ]
    train = [
        row for row in official_train if int(row["performer"]) not in validation_performers
    ]
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "official_train": official_train,
        "validation_performers": sorted(validation_performers),
    }


def write_csv(path: Path, rows: list[dict], fields=FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build portable official-protocol metadata for contiguous NTU60 clips",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--skeleton-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-subdir", default="contiguous_2x16")
    parser.add_argument("--protocol", choices=("xsub", "xview"), default="xsub")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-count-tolerance", type=int, default=2)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between zero and one")

    project_root = args.project_root.resolve()
    records = sampled_records(args.frame_root.resolve(), args.clip_subdir)
    skeletons = skeleton_index(args.skeleton_root.resolve())
    usable, excluded = build_rows(
        records, skeletons, project_root, args.frame_count_tolerance,
    )
    split = grouped_protocol_split(
        usable, args.protocol, args.validation_fraction, args.seed,
    )
    output = args.output_dir.resolve()
    write_csv(output / "matched_samples.csv", usable)
    write_csv(output / "train_split.csv", split["train"])
    write_csv(output / "val_split.csv", split["validation"])
    write_csv(output / "test_split.csv", split["test"])
    write_csv(output / "official_train_split.csv", split["official_train"])
    write_csv(output / "excluded_samples.csv", excluded, ("sample_id", "reason"))
    summary = {
        "dataset": "NTU RGB+D 60",
        "protocol": args.protocol,
        "split_unit": "performer for validation; official protocol for test",
        "clip_subdir": args.clip_subdir,
        "sampled_videos": len(records),
        "usable_single_person_videos": len(usable),
        "excluded_videos": len(excluded),
        "train_videos": len(split["train"]),
        "validation_videos": len(split["validation"]),
        "test_videos": len(split["test"]),
        "official_train_videos": len(split["official_train"]),
        "validation_performers": split["validation_performers"],
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "paths": "project-relative",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
