import numpy as np
import pytest

from scripts.NTU_RGBD.evaluation.temporal_metrics import compute_nacce
from scripts.NTU_RGBD.run_nacce_series import build_command


def test_nacce_is_zero_when_predicted_and_gt_acceleration_match():
    target = np.zeros((4, 1, 2), dtype=np.float32)
    target[:, 0, 0] = [0.0, 1.0, 4.0, 9.0]
    result = compute_nacce(target.copy(), target, np.ones((4, 1)), np.full(4, 2.0),
                           np.asarray(["a"] * 4), np.arange(4), ["joint"])
    assert result["consecutive_frame_triplets"] == 2
    assert result["valid_joint_triplets"] == 2
    assert result["nacce"] == pytest.approx(0.0)
    assert result["predicted_to_gt_acceleration_ratio"] == pytest.approx(1.0)


def test_nacce_rejects_gaps_and_video_boundaries():
    points = np.zeros((6, 1, 2), dtype=np.float32)
    result = compute_nacce(points, points, np.ones((6, 1)), np.ones(6),
                           np.asarray(["a", "a", "a", "b", "b", "b"]),
                           np.asarray([0, 1, 3, 4, 5, 6]), ["joint"])
    assert result["consecutive_frame_triplets"] == 1


def test_nacce_t0_command_targets_complete_t0_run():
    command = build_command("nacce_t0", python_executable="python")
    assert command[-1].endswith("Outputs_New/ntu_rgbd/t_ssnn/clip4_256_20ep/t0/seed_42")
