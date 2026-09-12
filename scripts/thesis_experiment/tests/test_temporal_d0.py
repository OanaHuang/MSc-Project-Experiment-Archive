from __future__ import annotations

import numpy as np
import pytest

from spikepose_thesis.evaluation.temporal import temporal_summary


def _sequence(
    target_x: np.ndarray,
    prediction_x: np.ndarray | None = None,
    *,
    clip_id: str = "clip_01",
    frame_offset: int = 0,
) -> dict[str, np.ndarray]:
    steps = len(target_x)
    prediction_x = target_x if prediction_x is None else prediction_x
    target = np.zeros((steps, 16, 2), dtype=np.float32)
    prediction = np.zeros_like(target)
    target[..., 0] = target_x[:, None]
    prediction[..., 0] = prediction_x[:, None]
    return {
        "prediction": prediction,
        "target": target,
        "visibility": np.ones((steps, 16), dtype=np.float32),
        "scale": np.ones(steps, dtype=np.float32),
        "sample_id": np.asarray([f"video::{clip_id}"] * steps),
        "video_id": np.asarray(["video"] * steps),
        "person_id": np.asarray(["person"] * steps),
        "clip_id": np.asarray([clip_id] * steps),
        "frame_index": np.arange(frame_offset, frame_offset + steps),
        "frame_position_in_clip": np.arange(steps),
    }


def _strict(values: dict[str, np.ndarray]) -> dict[str, object]:
    return temporal_summary(values, strict=True, expected_length=16)


@pytest.mark.parametrize(
    "trajectory",
    [
        np.zeros(16, dtype=np.float32),
        np.arange(16, dtype=np.float32),
        np.arange(16, dtype=np.float32) ** 2,
    ],
    ids=["static", "constant_velocity", "acceleration"],
)
def test_d0_exact_prediction_has_zero_motion_error(trajectory) -> None:
    summary = _strict(_sequence(trajectory))
    assert summary["vel_e"] == pytest.approx(0.0)
    assert summary["acc_e"] == pytest.approx(0.0)
    assert summary["racc_e"] == pytest.approx(0.0)
    assert summary["vmr"] == pytest.approx(1.0)
    assert summary["amr"] == pytest.approx(1.0)
    assert summary["lag_frames"] == pytest.approx(0.0)
    assert summary["num_velocity_intervals"] == 15
    assert summary["num_acceleration_intervals"] == 14
    assert summary["num_valid_velocity_pairs"] == 15 * 16
    assert summary["num_valid_acceleration_triplets"] == 14 * 16


def test_d0_fixed_prediction_exposes_over_smoothing() -> None:
    target = np.arange(16, dtype=np.float32) ** 2
    summary = _strict(_sequence(target, np.zeros(16, dtype=np.float32)))
    assert summary["vel_e"] > 0.0
    assert summary["acc_e"] > 0.0
    assert summary["vmr"] == pytest.approx(0.0)
    assert summary["amr"] == pytest.approx(0.0)


def test_d0_never_connects_adjacent_clips() -> None:
    left = _sequence(np.zeros(16, dtype=np.float32), clip_id="clip_01")
    right = _sequence(
        np.full(16, 1000.0, dtype=np.float32),
        clip_id="clip_02",
        frame_offset=16,
    )
    values = {key: np.concatenate((left[key], right[key])) for key in left}
    summary = _strict(values)
    assert summary["num_valid_sequences"] == 2
    assert summary["num_velocity_intervals"] == 30
    assert summary["num_acceleration_intervals"] == 28
    assert summary["vel_e"] == pytest.approx(0.0)
    assert summary["acc_e"] == pytest.approx(0.0)


def test_d0_strict_mode_rejects_invalid_or_missing_groups() -> None:
    values = _sequence(np.zeros(15, dtype=np.float32))
    with pytest.raises(ValueError, match="zero valid sequences"):
        _strict(values)
    del values["clip_id"]
    with pytest.raises(ValueError, match="explicit clip identity"):
        _strict(values)
