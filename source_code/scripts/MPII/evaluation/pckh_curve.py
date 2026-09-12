"""PCKh threshold curves and normalized AUC for saved MPII predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def compute_pckh_curve(normalized_distance: np.ndarray,
                       valid: np.ndarray) -> dict:
    distance = np.asarray(normalized_distance, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if distance.shape != valid.shape or distance.ndim != 2:
        raise ValueError("distance and valid must have matching [N, J] shapes")
    # Match the official MPII mean: pelvis (6) and thorax (7) are excluded.
    report_joints = np.ones(distance.shape[1], dtype=bool)
    report_joints[6:8] = False
    selected_distance = distance[:, report_joints]
    selected_valid = valid[:, report_joints] & np.isfinite(selected_distance)
    count = int(selected_valid.sum())
    if not count:
        raise ValueError("No valid MPII reporting joints")
    thresholds = np.linspace(0.0, 0.5, 101, dtype=np.float64)
    scores = np.asarray([
        np.count_nonzero(selected_valid & (selected_distance <= threshold)) / count
        for threshold in thresholds
    ], dtype=np.float64)
    # Normalized trapezoidal area, kept NumPy-version independent.
    area = np.sum((scores[:-1] + scores[1:]) * np.diff(thresholds) * 0.5)
    return {
        "normalization": "normalized_area_over_threshold_range_0_to_0.5",
        "reported_mean_excludes_joint_ids": [6, 7],
        "valid_joints": count,
        "pckh": {
            f"{threshold:.1f}": float(scores[int(round(threshold / 0.005))])
            for threshold in (0.1, 0.2, 0.3, 0.4, 0.5)
        },
        "auc_0_5": float(area / 0.5),
        "curve": {
            "thresholds": thresholds.tolist(),
            "scores": scores.tolist(),
        },
    }


def backfill_prediction_archive(archive: Path, output: Path) -> dict:
    with np.load(archive) as arrays:
        result = {
            label: compute_pckh_curve(
                arrays[f"{label}_normalized_distance_matrix"],
                arrays[f"{label}_valid_matrix"],
            )
            for label in ("official", "custom")
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
