"""Extended accuracy checks derived from canonical NTU prediction matrices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def compute_extended_pose_metrics(
    prediction: np.ndarray, target: np.ndarray, visibility: np.ndarray,
    head_length: np.ndarray,
) -> dict:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    visibility = np.asarray(visibility)
    head_length = np.asarray(head_length, dtype=np.float32)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [N, J, 2]")
    if visibility.shape != prediction.shape[:2]:
        raise ValueError("visibility must have shape [N, J]")
    if head_length.shape != (prediction.shape[0],):
        raise ValueError("head_length must have shape [N]")
    pixel_error = np.linalg.norm(prediction - target, axis=2)
    scale = head_length[:, None]
    valid = (visibility > 0) & np.isfinite(scale) & (scale > 1e-6)
    normalized_error = pixel_error / np.maximum(scale, 1e-6)
    values = normalized_error[valid]
    pixels = pixel_error[valid]
    if not len(values):
        raise ValueError("No valid joints for extended pose metrics")
    thresholds = np.linspace(0.0, 0.5, 101, dtype=np.float64)
    curve = np.asarray([(values <= threshold).mean() for threshold in thresholds])
    selected = {
        f"{threshold:.1f}": float((values <= threshold).mean())
        for threshold in (0.1, 0.2, 0.3, 0.4, 0.5)
    }
    return {
        "coordinate_space": "complete_frame_pixels",
        "normalization": "ground_truth_head_to_neck_distance",
        "aggregation": "micro_average_over_valid_joints",
        "samples": int(prediction.shape[0]),
        "valid_joints": int(valid.sum()),
        "pckhn": selected,
        "auc_0_5": float(np.sum(
            0.5 * (curve[:-1] + curve[1:]) * np.diff(thresholds)
        ) / 0.5),
        "nme": float(values.mean()),
        "median_nme": float(np.median(values)),
        "p90_normalized_error": float(np.percentile(values, 90)),
        "p95_normalized_error": float(np.percentile(values, 95)),
        "mean_2d_pixel_error": float(pixels.mean()),
        "median_2d_pixel_error": float(np.median(pixels)),
        "p90_2d_pixel_error": float(np.percentile(pixels, 90)),
        "curve": {
            "thresholds": thresholds.tolist(),
            "pckhn": curve.tolist(),
        },
    }


def save_extended_pose_metrics(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "extended_pose_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
