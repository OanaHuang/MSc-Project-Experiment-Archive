import numpy as np

from scripts.thesis_experiment.tools.create_ntu_test_visualizations import (
    PCKHB_PROXY_BOX_WIDTH_TO_LENGTH_RATIO,
    PCKHN_CENTER_TO_HEAD_AXIS_RATIO,
    _pckhb_proxy_headbox,
)


def test_proxy_headbox_uses_legacy_pckhn_center_ratio():
    ground_truth = np.zeros((16, 2), dtype=np.float32)
    ground_truth[8] = [100.0, 160.0]
    ground_truth[9] = [100.0, 100.0]
    visibility = np.ones(16, dtype=np.float32)

    corners = _pckhb_proxy_headbox(ground_truth, visibility)
    diagonal = np.linalg.norm(corners[2] - corners[0])
    width = np.linalg.norm(corners[1] - corners[0])
    length = np.linalg.norm(corners[2] - corners[1])
    top_center = 0.5 * (corners[0] + corners[1])
    bottom_center = 0.5 * (corners[2] + corners[3])

    np.testing.assert_allclose(length, 60.0 / 0.75, rtol=1e-6)
    np.testing.assert_allclose(
        width / length, PCKHB_PROXY_BOX_WIDTH_TO_LENGTH_RATIO, rtol=1e-6,
    )
    np.testing.assert_allclose(corners.mean(axis=0), [100.0, 100.0])
    np.testing.assert_allclose(top_center, [100.0, 60.0])
    np.testing.assert_allclose(bottom_center, [100.0, 140.0])
    np.testing.assert_allclose(
        np.linalg.norm(ground_truth[8] - bottom_center), 0.25 * length,
    )
    expected_scale_ratio = (
        0.6 / PCKHN_CENTER_TO_HEAD_AXIS_RATIO
        * np.sqrt(1.0 + PCKHB_PROXY_BOX_WIDTH_TO_LENGTH_RATIO ** 2)
    )
    np.testing.assert_allclose(
        0.6 * diagonal / 60.0, expected_scale_ratio, rtol=1e-6,
    )


def test_proxy_headbox_long_axis_follows_head_to_neck_direction():
    ground_truth = np.zeros((16, 2), dtype=np.float32)
    ground_truth[8] = [140.0, 140.0]
    ground_truth[9] = [100.0, 100.0]
    visibility = np.ones(16, dtype=np.float32)

    corners = _pckhb_proxy_headbox(ground_truth, visibility)
    box_long_axis = corners[2] - corners[1]
    box_long_axis /= np.linalg.norm(box_long_axis)
    box_width_axis = corners[1] - corners[0]
    box_width_axis /= np.linalg.norm(box_width_axis)
    head_to_neck = ground_truth[8] - ground_truth[9]
    head_to_neck /= np.linalg.norm(head_to_neck)

    np.testing.assert_allclose(box_long_axis, head_to_neck, rtol=1e-6)
    np.testing.assert_allclose(
        np.dot(box_width_axis, head_to_neck), 0.0, atol=1e-6,
    )
    assert not np.isclose(corners[0, 0], corners[1, 0])
    assert not np.isclose(corners[0, 1], corners[1, 1])


def test_proxy_headbox_is_absent_when_head_pair_is_not_visible():
    ground_truth = np.zeros((16, 2), dtype=np.float32)
    ground_truth[8] = [100.0, 160.0]
    ground_truth[9] = [100.0, 100.0]
    visibility = np.ones(16, dtype=np.float32)
    visibility[8] = 0

    assert _pckhb_proxy_headbox(ground_truth, visibility) is None
