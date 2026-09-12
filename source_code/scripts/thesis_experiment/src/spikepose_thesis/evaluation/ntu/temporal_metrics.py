from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def compute_nacce(prediction: np.ndarray, target: np.ndarray,
                  visibility: np.ndarray, head_length: np.ndarray,
                  sample_ids: np.ndarray, frame_indices: np.ndarray,
                  joint_names: tuple[str, ...] | list[str]) -> dict:
    """Compute head-normalized acceleration error on consecutive triplets."""
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    visibility = np.asarray(visibility)
    head_length = np.asarray(head_length, dtype=np.float32)
    sample_ids = np.asarray(sample_ids).astype(str)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [N, J, 2]")
    n, joints, _ = prediction.shape
    if visibility.shape != (n, joints):
        raise ValueError("visibility must have shape [N, J]")
    if any(array.shape != (n,) for array in (head_length, sample_ids, frame_indices)):
        raise ValueError("sequence metadata must have shape [N]")
    if len(joint_names) != joints:
        raise ValueError("joint_names must match the prediction joint dimension")

    centers: list[int] = []
    for sample_id in np.unique(sample_ids):
        indices = np.flatnonzero(sample_ids == sample_id)
        indices = indices[np.argsort(frame_indices[indices], kind="stable")]
        gaps = np.diff(frame_indices[indices])
        centers.extend(indices[1:-1][(gaps[:-1] == 1) & (gaps[1:] == 1)].tolist())
    center = np.asarray(centers, dtype=np.int64)
    previous, following = center - 1, center + 1
    if len(center) and not (
        np.all(sample_ids[previous] == sample_ids[center])
        and np.all(sample_ids[following] == sample_ids[center])
        and np.all(frame_indices[previous] + 1 == frame_indices[center])
        and np.all(frame_indices[center] + 1 == frame_indices[following])
    ):
        raise ValueError("prediction rows are not ordered as contiguous video frames")

    pred_accel = prediction[following] - 2 * prediction[center] + prediction[previous]
    gt_accel = target[following] - 2 * target[center] + target[previous]
    valid = (
        (visibility[previous] > 0) & (visibility[center] > 0)
        & (visibility[following] > 0)
        & np.isfinite(head_length[center, None])
        & (head_length[center, None] > 1e-6)
    )
    scale = np.maximum(head_length[center, None], 1e-6)
    error = np.linalg.norm(pred_accel - gt_accel, axis=2) / scale
    pixel_error = np.linalg.norm(pred_accel - gt_accel, axis=2)
    pred_magnitude = np.linalg.norm(pred_accel, axis=2) / scale
    gt_magnitude = np.linalg.norm(gt_accel, axis=2) / scale
    pixel_acceleration = np.linalg.norm(pred_accel, axis=2)
    gt_pixel_acceleration = np.linalg.norm(gt_accel, axis=2)

    def mean(values: np.ndarray, mask: np.ndarray) -> float | None:
        return float(values[mask].mean()) if np.any(mask) else None

    per_joint = {}
    for joint, name in enumerate(joint_names):
        mask = valid[:, joint]
        pred_value, gt_value = mean(pred_magnitude[:, joint], mask), mean(gt_magnitude[:, joint], mask)
        per_joint[name] = {
            "valid_triplets": int(mask.sum()),
            "acce": mean(pixel_error[:, joint], mask),
            "nacce": mean(error[:, joint], mask),
            "accel": mean(pixel_acceleration[:, joint], mask),
            "gt_accel": mean(gt_pixel_acceleration[:, joint], mask),
            "naccel": pred_value,
            "gt_naccel": gt_value,
            "naccel_to_gt_ratio": (
                pred_value / gt_value if pred_value is not None and gt_value is not None and gt_value > 1e-12 else None
            ),
            "predicted_normalized_acceleration": pred_value,
            "gt_normalized_acceleration": gt_value,
            "predicted_to_gt_acceleration_ratio": (
                pred_value / gt_value if pred_value is not None and gt_value is not None and gt_value > 1e-12 else None
            ),
        }
    pred_mean, gt_mean = mean(pred_magnitude, valid), mean(gt_magnitude, valid)
    accel_mean = mean(pixel_acceleration, valid)
    gt_accel_mean = mean(gt_pixel_acceleration, valid)
    return {
        "metric": "NAccE",
        "metric_name": "head_normalized_acceleration_error",
        "lower_is_better": True,
        "coordinate_space": "original_image_pixels",
        "temporal_difference": "central_second_difference",
        "frame_gap": 1,
        "normalization": "center_frame_gt_head_to_neck",
        "aggregation": "micro_average_over_valid_joint_triplets",
        "videos": int(len(np.unique(sample_ids))),
        "frames": int(n),
        "consecutive_frame_triplets": int(len(center)),
        "valid_joint_triplets": int(valid.sum()),
        "acce": mean(pixel_error, valid),
        "acce_unit": "original_image_pixels_per_frame_squared",
        "nacce": mean(error, valid),
        "accel": accel_mean,
        "accel_unit": "original_image_pixels_per_frame_squared",
        "gt_accel": gt_accel_mean,
        "naccel": pred_mean,
        "gt_naccel": gt_mean,
        "naccel_to_gt_ratio": (
            pred_mean / gt_mean if pred_mean is not None and gt_mean is not None and gt_mean > 1e-12 else None
        ),
        "predicted_normalized_acceleration": pred_mean,
        "gt_normalized_acceleration": gt_mean,
        "predicted_to_gt_acceleration_ratio": (
            pred_mean / gt_mean if pred_mean is not None and gt_mean is not None and gt_mean > 1e-12 else None
        ),
        "per_joint": per_joint,
    }


def save_nacce(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nacce_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "nacce_per_joint.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("joint", "valid_triplets", "acce", "nacce", "accel", "gt_accel",
                         "naccel", "gt_naccel", "naccel_to_gt_ratio",
                         "predicted_normalized_acceleration", "gt_normalized_acceleration",
                         "predicted_to_gt_acceleration_ratio"))
        for joint, row in summary["per_joint"].items():
            writer.writerow((joint, *row.values()))
