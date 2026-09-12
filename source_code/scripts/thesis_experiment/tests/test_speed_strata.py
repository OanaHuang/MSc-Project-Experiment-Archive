from __future__ import annotations

import numpy as np
import pytest

from spikepose_thesis.evaluation.speed_strata import (
    assign_speed_tertiles,
    speed_stratum_metrics,
    validate_shared_ground_truth,
    video_speed_scores,
)


def _values(speeds: tuple[float, ...] = (0, 1, 2, 3, 4, 5)) -> dict[str, np.ndarray]:
    predictions = []
    targets = []
    visibility = []
    sample_ids = []
    person_ids = []
    frame_indices = []
    video_ids = []
    clip_ids = []
    positions = []
    scales = []
    for video, speed in enumerate(speeds):
        video_id = f"V{video:02d}"
        for frame in range(4):
            # Translate a fixed two-joint pose so its GT pose-box diagonal is 10.
            pose = np.asarray([
                [speed * frame, 0.0],
                [speed * frame, 10.0],
            ], dtype=np.float32)
            targets.append(pose)
            predictions.append(pose.copy())
            visibility.append(np.ones(2, dtype=np.float32))
            sample_ids.append(video_id)
            person_ids.append("primary")
            frame_indices.append(frame)
            video_ids.append(video_id)
            clip_ids.append("full")
            positions.append(frame)
            scales.append(10.0)
    return {
        "prediction": np.asarray(predictions),
        "target": np.asarray(targets),
        "visibility": np.asarray(visibility),
        "scale": np.asarray(scales),
        "scale_hb": np.asarray(scales),
        "sample_id": np.asarray(sample_ids),
        "person_id": np.asarray(person_ids),
        "frame_index": np.asarray(frame_indices),
        "video_id": np.asarray(video_ids),
        "clip_id": np.asarray(clip_ids),
        "frame_position_in_clip": np.asarray(positions),
    }


def test_video_speed_scores_use_gt_normalized_pose_motion() -> None:
    rows = video_speed_scores(_values(), normalization="pose_bbox_diagonal", fps=30)
    np.testing.assert_allclose(
        [row["gt_movement_per_frame"] for row in rows],
        np.arange(6, dtype=np.float64) / 10.0,
    )
    np.testing.assert_allclose(
        [row["gt_movement_per_second"] for row in rows],
        np.arange(6, dtype=np.float64) * 3.0,
    )
    assert all(row["valid_joint_transitions"] == 6 for row in rows)


def test_equal_count_speed_tertiles_are_deterministic() -> None:
    rows = video_speed_scores(_values())
    assignments, protocol = assign_speed_tertiles(reversed(rows))
    assert [row["stratum"] for row in assignments] == [
        "low", "low", "medium", "medium", "high", "high",
    ]
    assert protocol["counts"] == {"low": 2, "medium": 2, "high": 2}
    assert protocol["boundaries"]["low_max"] == pytest.approx(0.1)
    assert protocol["boundaries"]["medium_min"] == pytest.approx(0.2)


def test_speed_stratum_metrics_reuse_fixed_video_assignments() -> None:
    values = _values()
    assignments, _ = assign_speed_tertiles(video_speed_scores(values))
    metrics = speed_stratum_metrics(values, assignments)
    assert set(metrics) == {"low", "medium", "high"}
    for row in metrics.values():
        assert row["videos"] == 2
        assert row["frames"] == 8
        assert row["pck_hb_0_5_all_frames"] == pytest.approx(1.0)
        assert row["vel_e_px_video_macro"] == pytest.approx(0.0)
        assert row["acc_e_px_video_macro"] == pytest.approx(0.0)
        assert row["amr_video_macro"] == pytest.approx(1.0)


def test_candidate_must_share_exact_reference_ground_truth() -> None:
    reference = _values()
    candidate = {key: value.copy() for key, value in reference.items()}
    validate_shared_ground_truth(reference, candidate)
    candidate["target"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="ground truth differ at target"):
        validate_shared_ground_truth(reference, candidate)
