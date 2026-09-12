from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from spikepose_thesis.evaluation.runner import summarize_predictions
from spikepose_thesis.evaluation.temporal import sequence_groups, temporal_summary


STRATA = ("low", "medium", "high")
SEQUENCE_ID_KEYS = ("video_id", "person_id", "clip_id")
REQUIRED_KEYS = {
    "prediction", "target", "visibility", "scale_hb", "sample_id",
    "person_id", "frame_index", "video_id", "clip_id",
    "frame_position_in_clip",
}


def load_prediction_archive(path) -> dict[str, np.ndarray]:
    """Load one prediction archive without permitting pickled payloads."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def validate_prediction_archive(values: dict[str, np.ndarray]) -> None:
    missing = sorted(REQUIRED_KEYS - values.keys())
    if missing:
        raise ValueError(f"Prediction archive is missing {missing}")
    frames = len(values["prediction"])
    if frames == 0:
        raise ValueError("Prediction archive contains no frames")
    for key in REQUIRED_KEYS:
        if len(values[key]) != frames:
            raise ValueError(
                f"Prediction archive field {key!r} has {len(values[key])} "
                f"rows; expected {frames}"
            )
    if values["target"].ndim != 3 or values["target"].shape[-1] != 2:
        raise ValueError("target must have shape frames x joints x 2")
    if values["prediction"].shape != values["target"].shape:
        raise ValueError("prediction and target shapes differ")
    if values["visibility"].shape != values["target"].shape[:2]:
        raise ValueError("visibility shape differs from target")


def validate_shared_ground_truth(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray],
) -> None:
    """Require a candidate method to use the exact reference rows and GT."""
    validate_prediction_archive(reference)
    validate_prediction_archive(candidate)
    keys = (
        "target", "visibility", "scale_hb", "sample_id", "person_id",
        "frame_index", "video_id", "clip_id", "frame_position_in_clip",
    )
    for key in keys:
        left, right = reference[key], candidate[key]
        equal = (
            np.array_equal(left, right, equal_nan=True)
            if np.issubdtype(left.dtype, np.number)
            else np.array_equal(left, right)
        )
        if not equal:
            raise ValueError(f"Candidate rows or ground truth differ at {key}")


def _sequence_key(values: dict[str, np.ndarray], indices: np.ndarray) -> tuple[str, ...]:
    return tuple(str(values[key][indices[0]]) for key in SEQUENCE_ID_KEYS)


def _pose_bbox_diagonal(
    target: np.ndarray, visibility: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(target).all(axis=-1)
    valid = (visibility > 0) & finite
    lower = np.where(valid[..., None], target, np.inf).min(axis=1)
    upper = np.where(valid[..., None], target, -np.inf).max(axis=1)
    diagonal = np.linalg.norm(upper - lower, axis=-1)
    diagonal[valid.sum(axis=1) < 2] = np.nan
    diagonal[~np.isfinite(diagonal) | (diagonal <= 0)] = np.nan
    return diagonal


def video_speed_scores(
    values: dict[str, np.ndarray], *,
    normalization: str = "pose_bbox_diagonal", fps: float = 30.0,
) -> list[dict[str, object]]:
    """Compute GT-only normalized joint-movement scores for complete videos.

    The default follows the movement-scale idea used by MTPose: visible-joint
    displacement is normalized for person size.  Here, the person-size proxy is
    the diagonal of the visible GT joint box because it is available in every
    frozen per-frame prediction archive.  Scores are averaged over all valid
    joint transitions in each video and never use model predictions.
    """
    validate_prediction_archive(values)
    if normalization not in {"pose_bbox_diagonal", "head_bone"}:
        raise ValueError(f"Unknown movement normalization: {normalization}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    rows = []
    for indices in sequence_groups(values):
        if len(indices) < 2:
            raise ValueError(f"Video {_sequence_key(values, indices)} is too short")
        positions = values["frame_position_in_clip"][indices].astype(np.int64)
        if not np.array_equal(positions, np.arange(len(indices))):
            raise ValueError(
                f"Video {_sequence_key(values, indices)} is not complete and contiguous"
            )
        target = values["target"][indices].astype(np.float64, copy=False)
        visibility = values["visibility"][indices] > 0
        displacement = np.linalg.norm(np.diff(target, axis=0), axis=-1)
        valid = visibility[1:] & visibility[:-1] & np.isfinite(displacement)
        if normalization == "pose_bbox_diagonal":
            frame_scale = _pose_bbox_diagonal(target, visibility)
        else:
            frame_scale = values["scale_hb"][indices].astype(np.float64, copy=False)
            frame_scale = frame_scale.copy()
            frame_scale[~np.isfinite(frame_scale) | (frame_scale <= 0)] = np.nan
        pair_scale = 0.5 * (frame_scale[1:] + frame_scale[:-1])
        normalized = displacement / pair_scale[:, None]
        valid &= np.isfinite(normalized)
        if not valid.any():
            raise ValueError(
                f"Video {_sequence_key(values, indices)} has no valid joint transitions"
            )
        score = float(normalized[valid].mean())
        raw = float(displacement[valid].mean())
        key = _sequence_key(values, indices)
        rows.append({
            "video_id": key[0],
            "person_id": key[1],
            "clip_id": key[2],
            "frames": int(len(indices)),
            "valid_joint_transitions": int(valid.sum()),
            "gt_movement_per_frame": score,
            "gt_movement_per_second": score * float(fps),
            "gt_displacement_px_per_frame": raw,
        })
    if len(rows) < 3:
        raise ValueError("At least three videos are required for speed tertiles")
    return rows


def assign_speed_tertiles(
    rows: Iterable[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Assign deterministic, equal-count low/medium/high video strata."""
    ordered = [dict(row) for row in rows]
    if len(ordered) < 3:
        raise ValueError("At least three videos are required for speed tertiles")
    ordered.sort(key=lambda row: (
        float(row["gt_movement_per_frame"]), str(row["video_id"]),
        str(row["person_id"]), str(row["clip_id"]),
    ))
    first = len(ordered) // 3
    second = 2 * len(ordered) // 3
    boundaries = (0, first, second, len(ordered))
    for stratum, start, end in zip(STRATA, boundaries[:-1], boundaries[1:]):
        for rank, row in enumerate(ordered[start:end], start=start + 1):
            row["stratum"] = stratum
            row["speed_rank"] = rank

    scores = np.asarray(
        [float(row["gt_movement_per_frame"]) for row in ordered],
        dtype=np.float64,
    )
    protocol = {
        "assignment": "equal_count_rank_tertiles",
        "tie_break": "video_id_then_person_id_then_clip_id",
        "videos": len(ordered),
        "counts": {
            label: int(sum(row["stratum"] == label for row in ordered))
            for label in STRATA
        },
        "boundaries": {
            "low_max": float(scores[first - 1]),
            "medium_min": float(scores[first]),
            "medium_max": float(scores[second - 1]),
            "high_min": float(scores[second]),
            "low_medium_tie_split": bool(scores[first - 1] == scores[first]),
            "medium_high_tie_split": bool(scores[second - 1] == scores[second]),
        },
    }
    return ordered, protocol


def speed_stratum_metrics(
    values: dict[str, np.ndarray], assignments: Iterable[dict[str, object]],
) -> dict[str, dict[str, float | int]]:
    """Summarize paper pose and motion metrics within fixed video strata."""
    validate_prediction_archive(values)
    label_by_key = {
        tuple(str(row[key]) for key in SEQUENCE_ID_KEYS): str(row["stratum"])
        for row in assignments
    }
    grouped_indices: dict[str, list[np.ndarray]] = {label: [] for label in STRATA}
    seen = set()
    for indices in sequence_groups(values):
        key = _sequence_key(values, indices)
        if key not in label_by_key:
            raise ValueError(f"No speed assignment for video {key}")
        seen.add(key)
        grouped_indices[label_by_key[key]].append(indices)
    if seen != set(label_by_key):
        raise ValueError("Speed manifest contains videos absent from predictions")

    result = {}
    frames = len(values["prediction"])
    for label in STRATA:
        if not grouped_indices[label]:
            raise ValueError(f"Speed stratum {label!r} is empty")
        indices = np.concatenate(grouped_indices[label])
        subset = {
            key: value[indices] if isinstance(value, np.ndarray) and len(value) == frames
            else value
            for key, value in values.items()
        }
        pose = summarize_predictions({**subset, "scale": subset["scale_hb"]})
        temporal = temporal_summary(subset, strict=False)
        result[label] = {
            "videos": int(len(grouped_indices[label])),
            "frames": int(len(indices)),
            "pck_hb_0_5_all_frames": float(pose["pck_0.5"]),
            "pck_hb_0_5_video_macro": float(pose["sequence_equal"]["pck_0.5"]),
            "vel_e_px_video_macro": float(temporal["vel_e"]),
            "acc_e_px_video_macro": float(temporal["acc_e"]),
            "amr_video_macro": float(temporal["amr"]),
        }
    return result


__all__ = [
    "STRATA",
    "assign_speed_tertiles",
    "load_prediction_archive",
    "speed_stratum_metrics",
    "validate_prediction_archive",
    "validate_shared_ground_truth",
    "video_speed_scores",
]
