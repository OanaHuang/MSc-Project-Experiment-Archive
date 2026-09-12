#!/usr/bin/env python3
"""Freeze and prepare a frame-budgeted full-video NTU60 test subset.

The selection unit is an entire official Cross-Subject test video.  The
selection preserves setup and action proportions, then performs deterministic
within-action swaps to approach the requested frame budget.  Preparation
reuses the existing two contiguous clips with hard links and decodes only the
missing frames from the per-setup NTU RGB archives.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import time
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA = (
    PROJECT_ROOT / "Datasets/NTU_RGBD/metadata/contiguous_xsub/test_split.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "Datasets/NTU_RGBD/metadata/fullvideo_xsub35"
)
DEFAULT_FULL_ROOT = PROJECT_ROOT / "Datasets/NTU_RGBD/extracted_frames_full"
DEFAULT_SAMPLED_ROOT = PROJECT_ROOT / "Datasets/NTU_RGBD/frames"
DEFAULT_ARCHIVE_ROOT = Path("/extra2/yunhao/ntu60_archives")
DEFAULT_TARGET_FRAMES = 419_968
DEFAULT_SEED = 20_260_826
VIDEO_SUFFIX = "_rgb.avi"
FRAME_PATTERN = re.compile(r"frame_(\d{6})\.jpg")
FRAME_MARKER = ".frames_complete.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Metadata is empty: {path}")
    required = {
        "sample_id", "setup", "camera", "performer", "replication",
        "action", "rgb_frames", "skeleton_frames", "frame_difference",
        "frame_count_match", "is_single_person",
    }
    missing = required - rows[0].keys()
    if missing:
        raise RuntimeError(f"Metadata lacks required fields: {sorted(missing)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Metadata contains duplicate sample_id values")
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_rank(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def _largest_remainder(weights: dict[object, int], total: int) -> dict[object, int]:
    weight_sum = sum(weights.values())
    if weight_sum <= 0 or total < 0:
        raise ValueError("Largest-remainder allocation requires positive weights")
    exact = {key: total * value / weight_sum for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(
        weights,
        key=lambda key: (-(exact[key] - result[key]), str(key)),
    )
    for key in order[:remaining]:
        result[key] += 1
    return result


def _largest_remainder_with_coverage(
    weights: dict[object, int], total: int,
) -> dict[object, int]:
    """Allocate proportionally while selecting every group when possible."""
    if total < len(weights):
        return _largest_remainder(weights, total)
    result = {key: 1 for key in weights}
    remaining = total - len(weights)
    exact = {
        key: total * value / sum(weights.values())
        for key, value in weights.items()
    }
    for _ in range(remaining):
        candidates = [key for key in weights if result[key] < weights[key]]
        if not candidates:
            raise ValueError("Allocation exceeds the available group capacity")
        key = min(candidates, key=lambda item: (result[item] - exact[item], str(item)))
        result[key] += 1
    return result


def _distribution_delta(
    counts: dict[str, Counter], expected: dict[str, dict[str, float]],
    outgoing: dict[str, str], incoming: dict[str, str],
) -> float:
    delta = 0.0
    for field in ("camera", "performer", "replication"):
        left = outgoing[field]
        right = incoming[field]
        if left == right:
            continue
        current = counts[field]
        target = expected[field]
        before = (
            abs(current[left] - target[left])
            + abs(current[right] - target[right])
        )
        after = (
            abs(current[left] - 1 - target[left])
            + abs(current[right] + 1 - target[right])
        )
        delta += after - before
    return delta


def _balanced_action_selection(
    setup: int, rows: list[dict[str, str]],
    by_action: dict[str, list[dict[str, str]]],
    action_targets: dict[str, int], selected_count: int, seed: int,
) -> set[str]:
    """Fill fixed action quotas while balancing other metadata marginals."""
    fields = ("camera", "performer", "replication")
    totals = {field: Counter(row[field] for row in rows) for field in fields}
    expected = {
        field: {
            value: selected_count * count / len(rows)
            for value, count in totals[field].items()
        }
        for field in fields
    }
    counts = {field: Counter() for field in fields}
    selected: set[str] = set()
    actions = sorted(
        by_action,
        key=lambda action: (
            len(by_action[action]),
            _stable_rank(seed, f"setup={setup}:action-order", action),
        ),
    )
    for action in actions:
        candidates = list(by_action[action])
        for _ in range(action_targets[action]):
            def score(row: dict[str, str]) -> tuple:
                marginal = sum(
                    counts[field][row[field]] / expected[field][row[field]]
                    for field in fields
                )
                return (
                    marginal,
                    _stable_rank(
                        seed, f"setup={setup}:action={action}", row["sample_id"],
                    ),
                )

            chosen = min(candidates, key=score)
            candidates.remove(chosen)
            selected.add(chosen["sample_id"])
            for field in fields:
                counts[field][chosen[field]] += 1
    return selected


def _improve_frame_budget(
    rows: list[dict[str, str]], selected_ids: set[str], target_frames: int,
    seed: int, max_swaps: int = 500,
) -> tuple[set[str], int]:
    by_action_selected: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_action_other: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        destination = (
            by_action_selected if row["sample_id"] in selected_ids
            else by_action_other
        )
        destination[row["action"]].append(row)

    selected_frames = sum(
        int(row["rgb_frames"])
        for row in rows if row["sample_id"] in selected_ids
    )
    selected_count = len(selected_ids)
    counts = {
        field: Counter(
            row[field] for row in rows if row["sample_id"] in selected_ids
        )
        for field in ("camera", "performer", "replication")
    }
    totals = {
        field: Counter(row[field] for row in rows)
        for field in ("camera", "performer", "replication")
    }
    expected = {
        field: {
            value: selected_count * count / len(rows)
            for value, count in totals[field].items()
        }
        for field in totals
    }

    rng = random.Random(seed)
    for _ in range(max_swaps):
        current_error = abs(selected_frames - target_frames)
        if current_error == 0:
            break
        best: tuple[int, float, float, str, dict, dict] | None = None
        for action in sorted(by_action_selected):
            for outgoing in by_action_selected[action]:
                outgoing_frames = int(outgoing["rgb_frames"])
                for incoming in by_action_other[action]:
                    candidate_frames = (
                        selected_frames - outgoing_frames
                        + int(incoming["rgb_frames"])
                    )
                    candidate_error = abs(candidate_frames - target_frames)
                    if candidate_error >= current_error:
                        continue
                    distribution = _distribution_delta(
                        counts, expected, outgoing, incoming,
                    )
                    tie = _stable_rank(
                        seed, "swap",
                        outgoing["sample_id"] + ":" + incoming["sample_id"],
                    )
                    candidate = (
                        candidate_error, distribution, rng.random(), tie,
                        outgoing, incoming,
                    )
                    if best is None or candidate[:4] < best[:4]:
                        best = candidate
        if best is None:
            break
        _, _, _, _, outgoing, incoming = best
        selected_ids.remove(outgoing["sample_id"])
        selected_ids.add(incoming["sample_id"])
        selected_frames += (
            int(incoming["rgb_frames"]) - int(outgoing["rgb_frames"])
        )
        action = outgoing["action"]
        by_action_selected[action].remove(outgoing)
        by_action_selected[action].append(incoming)
        by_action_other[action].remove(incoming)
        by_action_other[action].append(outgoing)
        for field in counts:
            counts[field][outgoing[field]] -= 1
            counts[field][incoming[field]] += 1
    return selected_ids, selected_frames


def freeze_selection(args: argparse.Namespace) -> dict:
    metadata_path = args.test_metadata.resolve()
    output_dir = args.output_dir.resolve()
    rows = _read_csv(metadata_path)
    fields = list(rows[0].keys())
    total_frames = sum(int(row["rgb_frames"]) for row in rows)
    if not 0 < args.target_frames < total_frames:
        raise ValueError(
            f"target-frames must lie between 1 and {total_frames - 1}"
        )
    ratio = args.target_frames / total_frames
    target_videos = round(len(rows) * ratio)

    by_setup: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_setup[int(row["setup"])].append(row)
    setup_video_targets = _largest_remainder(
        {setup: len(group) for setup, group in by_setup.items()},
        target_videos,
    )
    setup_frame_targets = _largest_remainder(
        {
            setup: sum(int(row["rgb_frames"]) for row in group)
            for setup, group in by_setup.items()
        },
        args.target_frames,
    )

    selected_ids: set[str] = set()
    setup_reports = []
    for setup in sorted(by_setup):
        setup_rows = by_setup[setup]
        target_count = setup_video_targets[setup]
        by_action: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in setup_rows:
            by_action[row["action"]].append(row)
        action_targets = _largest_remainder_with_coverage(
            {action: len(group) for action, group in by_action.items()},
            target_count,
        )
        setup_selected = _balanced_action_selection(
            setup, setup_rows, by_action, action_targets, target_count, args.seed,
        )
        setup_selected, selected_frames = _improve_frame_budget(
            setup_rows, setup_selected, setup_frame_targets[setup],
            args.seed + setup,
        )
        selected_ids.update(setup_selected)
        setup_reports.append({
            "setup": setup,
            "available_videos": len(setup_rows),
            "selected_videos": len(setup_selected),
            "video_ratio": len(setup_selected) / len(setup_rows),
            "available_frames": sum(
                int(row["rgb_frames"]) for row in setup_rows
            ),
            "target_frames": setup_frame_targets[setup],
            "selected_frames": selected_frames,
            "frame_error": selected_frames - setup_frame_targets[setup],
            "frame_ratio": selected_frames / sum(
                int(row["rgb_frames"]) for row in setup_rows
            ),
            "available_actions": len(by_action),
            "selected_actions": len({
                row["action"] for row in setup_rows
                if row["sample_id"] in setup_selected
            }),
        })

    selected = sorted(
        (row for row in rows if row["sample_id"] in selected_ids),
        key=lambda row: row["sample_id"],
    )
    excluded = sorted(
        (row for row in rows if row["sample_id"] not in selected_ids),
        key=lambda row: row["sample_id"],
    )
    if len(selected) != target_videos:
        raise RuntimeError(
            f"Selection count mismatch: {len(selected)} != {target_videos}"
        )
    selected_path = output_dir / "test_split.csv"
    excluded_path = output_dir / "not_selected.csv"
    audit_path = output_dir / "setup_audit.csv"
    _write_csv(selected_path, selected, fields)
    _write_csv(excluded_path, excluded, fields)
    audit_fields = list(setup_reports[0].keys())
    _write_csv(
        audit_path,
        [{key: str(value) for key, value in row.items()} for row in setup_reports],
        audit_fields,
    )
    selected_frames = sum(int(row["rgb_frames"]) for row in selected)
    report = {
        "protocol": "official Cross-Subject stratified full-video subset",
        "selection_unit": "whole source video",
        "selection_seed": args.seed,
        "selection_constraints": [
            "proportional setup counts",
            "proportional action counts within setup with full action coverage",
            "camera/performer/replication marginal balancing",
            "deterministic within-action frame-budget swaps",
        ],
        "source_metadata": str(metadata_path),
        "source_metadata_sha256": _sha256(metadata_path),
        "source_videos": len(rows),
        "source_frames": total_frames,
        "target_frames": args.target_frames,
        "target_ratio": ratio,
        "selected_videos": len(selected),
        "selected_video_ratio": len(selected) / len(rows),
        "selected_frames": selected_frames,
        "selected_frame_ratio": selected_frames / total_frames,
        "frame_budget_error": selected_frames - args.target_frames,
        "test_split": str(selected_path),
        "test_split_sha256": _sha256(selected_path),
        "not_selected": str(excluded_path),
        "not_selected_sha256": _sha256(excluded_path),
        "setup_audit": str(audit_path),
        "setups": setup_reports,
    }
    _write_json(output_dir / "selection_manifest.json", report)
    return report


def _frame_numbers(path: Path) -> list[int]:
    values = []
    for candidate in path.glob("frame_*.jpg"):
        match = FRAME_PATTERN.fullmatch(candidate.name)
        if match is not None and candidate.stat().st_size > 0:
            values.append(int(match.group(1)))
    return sorted(values)


def _sampled_sources(sampled_root: Path, setup: int, sample_id: str) -> list[Path]:
    root = (
        sampled_root / f"S{setup:03d}" / "contiguous_2x16" / sample_id
    )
    return sorted(root.glob("clip_*/frame_*.jpg"))


_ZIP_CACHE: dict[str, zipfile.ZipFile] = {}


def _zip_handle(path: str) -> zipfile.ZipFile:
    handle = _ZIP_CACHE.get(path)
    if handle is None:
        handle = zipfile.ZipFile(path)
        _ZIP_CACHE[path] = handle
    return handle


def _prepare_one(task: dict) -> dict:
    import cv2

    cv2.setNumThreads(1)
    sample_id = task["sample_id"]
    expected = int(task["expected_frames"])
    output_dir = Path(task["full_root"]) / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in output_dir.glob("frame_*.jpg"):
        if candidate.stat().st_size == 0:
            candidate.unlink()
    expected_numbers = list(range(expected))
    existing = _frame_numbers(output_dir)
    if existing == expected_numbers:
        status = "reused_complete"
        linked_new = 0
        written_new = 0
        decoded = expected
    else:
        unexpected = [value for value in existing if value >= expected]
        if unexpected:
            raise RuntimeError(
                f"{sample_id} contains out-of-range frames: {unexpected[:5]}"
            )
        linked_new = 0
        for source_text in task["sampled_sources"]:
            source = Path(source_text)
            match = FRAME_PATTERN.fullmatch(source.name)
            if match is None:
                continue
            number = int(match.group(1))
            if number >= expected:
                raise RuntimeError(
                    f"Sampled frame {source} exceeds expected length {expected}"
                )
            target = output_dir / source.name
            if not target.exists():
                os.link(source, target)
                linked_new += 1

        handle = _zip_handle(task["archive"])
        info = handle.getinfo(task["member"])
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f"{sample_id}_", suffix=".avi", delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                with handle.open(info) as source:
                    shutil.copyfileobj(source, temporary, length=4 * 1024 * 1024)
            capture = cv2.VideoCapture(str(temporary_path))
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Could not decode archive video for {sample_id}")
            decoded = 0
            written_new = 0
            try:
                while True:
                    ok, image = capture.read()
                    if not ok or image is None:
                        break
                    target = output_dir / f"frame_{decoded:06d}.jpg"
                    if not target.exists():
                        if not cv2.imwrite(
                            str(target), image,
                            [cv2.IMWRITE_JPEG_QUALITY, int(task["jpeg_quality"])],
                        ):
                            raise RuntimeError(f"Could not write {target}")
                        written_new += 1
                    decoded += 1
            finally:
                capture.release()
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        if decoded != expected:
            raise RuntimeError(
                f"Decoded frame mismatch for {sample_id}: {decoded} != {expected}"
            )
        final_numbers = _frame_numbers(output_dir)
        if final_numbers != expected_numbers:
            raise RuntimeError(
                f"Prepared frame set is incomplete for {sample_id}: "
                f"{len(final_numbers)} != {expected}"
            )
        status = "prepared"

    marker = {
        "sample_id": sample_id,
        "setup": int(task["setup"]),
        "source": task["archive"],
        "archive_member": task["member"],
        "archive_crc32": task["archive_crc32"],
        "archive_member_bytes": int(task["archive_member_bytes"]),
        "stride": 1,
        "jpeg_quality": int(task["jpeg_quality"]),
        "saved_frames": expected,
        "reused_sampled_hardlinks": len(task["sampled_sources"]),
        "new_hardlinks": linked_new,
        "new_jpegs": written_new,
        "selection_sha256": task["selection_sha256"],
    }
    _write_json(output_dir / FRAME_MARKER, marker)
    return {**marker, "status": status}


def _archive_members(
    archive_root: Path, rows: list[dict[str, str]],
) -> dict[str, tuple[Path, zipfile.ZipInfo]]:
    requested: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        requested[int(row["setup"])].add(row["sample_id"])
    result: dict[str, tuple[Path, zipfile.ZipInfo]] = {}
    for setup in sorted(requested):
        archive = archive_root / f"nturgbd_rgb_s{setup:03d}.zip"
        if not archive.is_file():
            raise FileNotFoundError(f"Missing archive: {archive}")
        with zipfile.ZipFile(archive) as handle:
            by_basename = {Path(info.filename).name: info for info in handle.infolist()}
            for sample_id in requested[setup]:
                filename = f"{sample_id}{VIDEO_SUFFIX}"
                info = by_basename.get(filename)
                if info is None:
                    raise FileNotFoundError(f"{filename} is absent from {archive}")
                result[sample_id] = (archive, info)
    return result


def prepare_frames(args: argparse.Namespace) -> dict:
    selection_path = args.output_dir.resolve() / "test_split.csv"
    rows = _read_csv(selection_path)
    selection_sha256 = _sha256(selection_path)
    full_root = args.full_root.resolve()
    sampled_root = args.sampled_root.resolve()
    archives = _archive_members(args.archive_root.resolve(), rows)
    tasks = []
    for row in rows:
        sample_id = row["sample_id"]
        setup = int(row["setup"])
        archive, info = archives[sample_id]
        sources = _sampled_sources(sampled_root, setup, sample_id)
        if len(sources) != 32:
            raise RuntimeError(
                f"Expected 32 sampled frames for {sample_id}, found {len(sources)}"
            )
        tasks.append({
            "sample_id": sample_id,
            "setup": setup,
            "expected_frames": int(row["rgb_frames"]),
            "archive": str(archive),
            "member": info.filename,
            "archive_crc32": f"{info.CRC:08x}",
            "archive_member_bytes": info.file_size,
            "full_root": str(full_root),
            "sampled_sources": [str(path) for path in sources],
            "jpeg_quality": args.jpeg_quality,
            "selection_sha256": selection_sha256,
        })

    workers = args.workers or (os.cpu_count() or 1)
    started = time.monotonic()
    status_counts: Counter = Counter()
    linked = 0
    written = 0
    frames = 0
    print(
        f"[prepare] videos={len(tasks)} workers={workers} "
        f"target_frames={sum(task['expected_frames'] for task in tasks)}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(_prepare_one, task): task for task in tasks}
        for index, future in enumerate(as_completed(pending), 1):
            task = pending[future]
            try:
                report = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"Preparation failed for {task['sample_id']}: {error}"
                ) from error
            status_counts[report["status"]] += 1
            linked += int(report["new_hardlinks"])
            written += int(report["new_jpegs"])
            frames += int(report["saved_frames"])
            if index == 1 or index % 25 == 0 or index == len(tasks):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[prepare] {index}/{len(tasks)} "
                    f"({index / elapsed:.2f} videos/s) "
                    f"status={dict(status_counts)}",
                    flush=True,
                )
    report = {
        "selection": str(selection_path),
        "selection_sha256": selection_sha256,
        "full_root": str(full_root),
        "videos": len(tasks),
        "frames": frames,
        "new_hardlinks": linked,
        "new_jpegs": written,
        "statuses": dict(status_counts),
        "workers": workers,
        "seconds": time.monotonic() - started,
    }
    _write_json(args.output_dir.resolve() / "preparation_complete.json", report)
    return report


def audit_prepared(args: argparse.Namespace) -> dict:
    selection_path = args.output_dir.resolve() / "test_split.csv"
    rows = _read_csv(selection_path)
    full_root = args.full_root.resolve()
    missing = []
    incomplete = []
    frames = 0
    for row in rows:
        sample_id = row["sample_id"]
        expected = int(row["rgb_frames"])
        directory = full_root / sample_id
        if not directory.is_dir():
            missing.append(sample_id)
            continue
        actual = _frame_numbers(directory)
        if actual != list(range(expected)):
            incomplete.append({
                "sample_id": sample_id,
                "expected": expected,
                "actual": len(actual),
            })
            continue
        frames += expected
    report = {
        "selection": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "videos": len(rows),
        "complete_videos": len(rows) - len(missing) - len(incomplete),
        "complete_frames": frames,
        "missing_videos": missing,
        "incomplete_videos": incomplete,
    }
    _write_json(args.output_dir.resolve() / "preparation_audit.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("select", "prepare", "audit", "all"),
        default="all",
    )
    parser.add_argument("--test-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--sampled-root", type=Path, default=DEFAULT_SAMPLED_ROOT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--target-frames", type=int, default=DEFAULT_TARGET_FRAMES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Decoder processes; zero uses every detected CPU",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must lie between 1 and 100")
    if args.stage in {"select", "all"}:
        print(json.dumps(freeze_selection(args), indent=2), flush=True)
    if args.stage in {"prepare", "all"}:
        print(json.dumps(prepare_frames(args), indent=2), flush=True)
    if args.stage == "audit":
        print(json.dumps(audit_prepared(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
