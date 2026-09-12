from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
import re

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.core.paths import PROJECT_ROOT, resolve_project_path


def _csv_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    lines = [line for line in path.read_text(
        encoding="utf-8", errors="replace",
    ).splitlines() if line.strip()]
    return max(len(lines) - 1, 0)


def _csv_sample_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", errors="replace") as handle:
        return {
            str(row.get("sample_id", "")).strip()
            for row in csv.DictReader(handle)
            if str(row.get("sample_id", "")).strip()
        }


def _split_overlap(paths: list[Path]) -> dict[str, int]:
    train, validation, test = (_csv_sample_ids(path) for path in paths)
    return {
        "train_validation": len(train & validation),
        "train_test": len(train & test),
        "validation_test": len(validation & test),
    }


_NTU_SAMPLE = re.compile(
    r"S(?P<setup>\d{3})C\d{3}P(?P<subject>\d{3})"
    r"R(?P<repetition>\d{3})A(?P<action>\d{3})"
)


def _performance_groups(path: Path) -> set[str]:
    result = set()
    for sample_id in _csv_sample_ids(path):
        match = _NTU_SAMPLE.search(sample_id)
        if match:
            result.add(
                "S{setup}P{subject}R{repetition}A{action}".format(**match.groupdict())
            )
    return result


def _performance_group_overlap(paths: list[Path]) -> dict[str, int]:
    train, validation, test = (_performance_groups(path) for path in paths)
    return {
        "train_validation": len(train & validation),
        "train_test": len(train & test),
        "validation_test": len(validation & test),
    }


def _clip_layout(frames_root: Path, clip_subdir: str,
                 clips_per_sequence: int, clip_length: int,
                 included_sample_ids: set[str] | None = None) -> dict[str, object]:
    total = audited = valid = wrong_count = wrong_length = nonconsecutive = overlap = 0
    overlapping_sample_ids = []
    for setup_dir in frames_root.glob("S[0-9][0-9][0-9]"):
        root = setup_dir / clip_subdir
        if not root.is_dir():
            continue
        for sequence_dir in root.glob("S*"):
            if not sequence_dir.is_dir():
                continue
            total += 1
            if (included_sample_ids is not None
                    and sequence_dir.name not in included_sample_ids):
                continue
            audited += 1
            clip_dirs = sorted(
                item for item in sequence_dir.glob("clip_*") if item.is_dir()
            )
            if len(clip_dirs) != clips_per_sequence:
                wrong_count += 1
                continue
            indices = []
            sequence_valid = True
            for clip_dir in clip_dirs:
                frame_indices = []
                for path in clip_dir.glob("frame_*.jpg"):
                    try:
                        frame_indices.append(int(path.stem.rsplit("_", 1)[1]))
                    except (IndexError, ValueError):
                        sequence_valid = False
                frame_indices.sort()
                if len(frame_indices) != clip_length:
                    wrong_length += 1
                    sequence_valid = False
                if frame_indices and any(
                    current != previous + 1
                    for previous, current in zip(frame_indices, frame_indices[1:])
                ):
                    nonconsecutive += 1
                    sequence_valid = False
                indices.append(set(frame_indices))
            if len(indices) == 2 and indices[0] & indices[1]:
                overlap += 1
                overlapping_sample_ids.append(sequence_dir.name)
                sequence_valid = False
            if sequence_valid:
                valid += 1
    return {
        "sequences": total,
        "audited_sequences": audited,
        "excluded_sequences_not_audited": total - audited,
        "valid_sequences": valid,
        "wrong_clip_count": wrong_count,
        "wrong_clip_length": wrong_length,
        "nonconsecutive_clips": nonconsecutive,
        "overlapping_clips": overlap,
        "overlapping_sample_ids": sorted(overlapping_sample_ids),
    }


def audit_data(
    require_preflight: bool = True,
    use_preflight_cache: bool = True,
    require_runtime_cache: bool = True,
) -> dict:
    mpii = load_experiment("confirm140_m_s3_u2")["data"]
    ntu = load_experiment("confirm140_ntu_spikepose_frame")["data"]
    mpii_images = resolve_project_path(mpii["images_dir"])
    mpii_train = resolve_project_path(mpii["train_metadata"])
    mpii_validation = resolve_project_path(mpii["validation_metadata"])
    mpii_hb_calibration = resolve_project_path(mpii["head_bone_calibration"])
    skeleton_root = PROJECT_ROOT / "Datasets" / "NTU_RGBD" / "skeletons"
    frames_root = resolve_project_path(ntu["frames_dir"])
    clip_subdir = str(ntu.get("frame_clip_subdir", "contiguous_2x16"))
    exclusion_path = resolve_project_path(ntu["exclusion_metadata"])
    expected = int(ntu["expected_raw_sequences"])
    expected_exclusions = int(ntu["expected_quality_exclusions"])
    expected_usable = int(ntu["expected_usable_sequences"])
    contiguous_clip_length = int(ntu.get("contiguous_clip_length", 0))
    clips_per_sequence = int(ntu.get("clips_per_sequence", 0))
    split_keys = ("train_metadata", "validation_metadata", "test_metadata")
    cross_subject_paths = [resolve_project_path(ntu[key]) for key in split_keys]
    cross_subject_rows = [_csv_row_count(path) for path in cross_subject_paths]
    cross_subject_overlap = _split_overlap(cross_subject_paths)
    cross_subject_group_overlap = _performance_group_overlap(cross_subject_paths)
    usable_sample_ids = set().union(*(
        _csv_sample_ids(path) for path in cross_subject_paths
    ))
    cached_report = None
    if use_preflight_cache:
        from spikepose_thesis.data.ntu.preflight import cached_audit_report

        cached_report = cached_audit_report(ntu)
    if cached_report is not None:
        cached_counts = cached_report["counts"]
        clip_layout = deepcopy(cached_counts["clip_layout"])
        skeleton_files = int(cached_counts["skeleton_files"])
    else:
        clip_layout = _clip_layout(
            frames_root, clip_subdir, clips_per_sequence, contiguous_clip_length,
            usable_sample_ids,
        )
        skeleton_files = sum(1 for _ in skeleton_root.glob("**/*.skeleton"))
    from spikepose_thesis.data.ntu.runtime_cache import runtime_cache_status

    runtime_status = runtime_cache_status(ntu)
    counts = {
        "skeleton_files": skeleton_files,
        "frame_sequences": clip_layout["sequences"],
        "clip_layout": clip_layout,
        "quality_exclusions": _csv_row_count(exclusion_path),
        "cross_subject_split_rows": cross_subject_rows,
        "cross_subject_split_overlap": cross_subject_overlap,
        "cross_subject_performance_group_overlap": cross_subject_group_overlap,
    }
    checks = {
        "mpii_images": mpii_images.is_dir(),
        "mpii_train_metadata": mpii_train.is_file(),
        "mpii_validation_metadata": mpii_validation.is_file(),
        "mpii_head_bone_calibration": mpii_hb_calibration.is_file(),
        "ntu_skeletons_complete": counts["skeleton_files"] >= expected,
        "ntu_frames_complete": counts["frame_sequences"] >= expected,
        "ntu_clip_layout_valid": (
            clip_layout["audited_sequences"] == expected_usable
            and clip_layout["wrong_clip_count"] == 0
            and clip_layout["wrong_clip_length"] == 0
            and clip_layout["nonconsecutive_clips"] == 0
        ),
        "ntu_clips_nonoverlapping": (
            clip_layout["overlapping_clips"] == 0
            or bool(ntu.get("exclude_overlapping_clips", False))
        ),
        "ntu_quality_exclusions_complete": (
            counts["quality_exclusions"] == expected_exclusions
        ),
        "ntu_cross_subject_metadata": (
            all(path.is_file() for path in cross_subject_paths)
            and all(cross_subject_rows)
            and sum(cross_subject_rows) == expected_usable
        ),
        "ntu_cross_subject_disjoint": not any(cross_subject_overlap.values()),
        "ntu_cross_subject_performance_groups_disjoint": not any(
            cross_subject_group_overlap.values()
        ),
        "ntu_pose_preflight_ready": cached_report is not None,
        "ntu_runtime_cache_ready": bool(runtime_status["ready"]),
    }
    required_common = (
        "mpii_images", "mpii_train_metadata", "mpii_validation_metadata",
        "ntu_skeletons_complete", "ntu_frames_complete",
        "ntu_clip_layout_valid", "ntu_clips_nonoverlapping",
        "ntu_quality_exclusions_complete",
        *(("ntu_pose_preflight_ready",) if require_preflight else tuple()),
        *(("ntu_runtime_cache_ready",) if require_runtime_cache else tuple()),
    )
    required_cs = (*required_common, "ntu_cross_subject_metadata",
                   "ntu_cross_subject_disjoint",
                   "ntu_cross_subject_performance_groups_disjoint")
    return {
        "ready": all(checks[key] for key in required_cs),
        "ready_cross_subject": all(checks[key] for key in required_cs),
        "checks": checks, "counts": counts,
        "expected_ntu_sequences": expected,
        "expected_quality_exclusions": expected_exclusions,
        "expected_usable_ntu_sequences": expected_usable,
        "effective_usable_ntu_sequences": (
            expected_usable - int(clip_layout["overlapping_clips"])
        ),
        "contiguous_clip_length": contiguous_clip_length,
        "quality_exclusion_file": str(exclusion_path),
        "preflight_cached": cached_report is not None,
        "runtime_cache": {
            key: value for key, value in runtime_status.items() if key != "manifest"
        },
    }
