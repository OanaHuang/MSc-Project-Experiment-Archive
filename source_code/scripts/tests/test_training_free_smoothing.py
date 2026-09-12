import numpy as np

from scripts.NTU_RGBD.run_sn_series import (
    metrics, one_euro_filter, savgol_filter, sequence_groups,
)


def test_training_free_filters_preserve_constant_sequences_and_boundaries():
    sample_ids = np.array(["a"] * 8 + ["b"] * 8)
    frames = np.array(list(range(8)) * 2)
    groups = sequence_groups(sample_ids, frames)
    values = np.concatenate((np.ones((8, 1, 2)), np.full((8, 1, 2), 9.0)))
    np.testing.assert_allclose(savgol_filter(values, groups), values, atol=1e-6)
    np.testing.assert_allclose(one_euro_filter(values, groups), values, atol=1e-6)


def test_savgol_reduces_impulse_acceleration():
    values = np.zeros((9, 1, 2), dtype=np.float32)
    values[4, 0, 0] = 10.0
    groups = (np.arange(9),)
    filtered = savgol_filter(values, groups)
    assert np.abs(np.diff(filtered[:, 0, 0], n=2)).mean() < np.abs(
        np.diff(values[:, 0, 0], n=2)
    ).mean()


def test_combined_per_joint_metrics_include_pckhn_and_acceleration():
    prediction = np.zeros((3, 25, 2), dtype=np.float32)
    prediction[2, 0, 0] = 2.0
    arrays = {"gt": np.zeros_like(prediction), "visibility": np.ones((3, 25)),
              "head_length": np.full(3, 2.0)}
    result = metrics(prediction, arrays, np.array(["a"] * 3), np.arange(3))
    joint = result["per_joint"]["spine_base"]
    assert joint["pckhn"] == 2 / 3
    assert joint["accel"] == 2.0
    assert joint["naccel"] == 1.0
