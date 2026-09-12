#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from spikepose_thesis.data.ntu.core.config import (
    NTU60_CROSS_SUBJECT_TRAIN_PERFORMERS, NTU60_CROSS_VIEW_TRAIN_CAMERAS,
)
from spikepose_thesis.data.ntu.core.person_selector import count_body_occurrences
from spikepose_thesis.data.ntu.core.skeleton_reader import read_skeleton_file


def _ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        line.strip().split(",", 1)[0] for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lower().startswith("sample_id")
    }


def _validation(sample_id: str, fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def _resolve_skeleton(row: dict, root: Path) -> Path:
    original = Path(row["skeleton_path"])
    if original.is_file():
        return original.resolve()
    matches = list(root.glob(f"**/{original.name}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not uniquely locate {original.name} under {root}")
    return matches[0].resolve()


def prepare(args) -> None:
    invalid = set().union(*(_ids(path) for path in args.exclude))
    with args.metadata.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = {"train": [], "validation": [], "test": []}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in invalid:
            continue
        performer, camera = int(row["performer"]), int(row["camera"])
        in_train_pool = (
            performer in NTU60_CROSS_SUBJECT_TRAIN_PERFORMERS
            if args.protocol == "cross_subject"
            else camera in NTU60_CROSS_VIEW_TRAIN_CAMERAS
        )
        split = "test" if not in_train_pool else (
            "validation" if _validation(sample_id, args.validation_fraction, args.seed)
            else "train"
        )
        if args.prevalidated_single_person:
            output[split].append({
                **row, "body_id": "", "person_index": 0,
            })
            continue
        skeleton = _resolve_skeleton(row, args.skeleton_root)
        occurrences = count_body_occurrences(read_skeleton_file(skeleton))
        body_ids = [
            body_id for body_id, count in sorted(occurrences.items())
            if count >= args.minimum_body_frames
        ]
        for person_index, body_id in enumerate(body_ids):
            output[split].append({
                **row, "skeleton_path": str(skeleton), "body_id": body_id,
                "person_index": person_index,
            })
    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(next(iter(rows)).keys()) + ["body_id", "person_index"]
    for split, values in output.items():
        with (args.output / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
        print(f"{split}: {len(values)} sequence-person rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--skeleton-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=("cross_subject", "cross_view"), required=True)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[])
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--minimum-body-frames", type=int, default=2)
    parser.add_argument(
        "--prevalidated-single-person", action="store_true",
        help=(
            "Trust metadata that has already excluded empty and multi-body "
            "sequences; preserve the primary-person convention without "
            "rescanning every skeleton."
        ),
    )
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
