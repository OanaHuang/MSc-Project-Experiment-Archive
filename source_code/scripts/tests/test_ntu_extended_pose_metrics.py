from __future__ import annotations

import numpy as np

from scripts.NTU_RGBD.evaluation.extended_pose_metrics import (
    compute_extended_pose_metrics,
)


def test_extended_pose_metrics_use_head_normalized_visible_errors():
    target = np.zeros((1, 2, 2), np.float32)
    prediction = np.asarray([[[1.0, 0.0], [6.0, 0.0]]], np.float32)
    visibility = np.ones((1, 2), np.float32)
    summary = compute_extended_pose_metrics(
        prediction, target, visibility, np.asarray([10.0], np.float32),
    )
    assert summary["pckhn"]["0.1"] == 0.5
    assert summary["pckhn"]["0.5"] == 0.5
    assert np.isclose(summary["nme"], 0.35)
    assert np.isclose(summary["mean_2d_pixel_error"], 3.5)
    assert 0.0 <= summary["auc_0_5"] <= 1.0
