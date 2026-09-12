from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.NTU_RGBD.core.config import NTU_FLIP_PAIRS, NTU_NUM_JOINTS
from scripts.NTU_RGBD.datasets.ntu_frame_dataset import (
    compute_head_length, generate_gaussian_heatmaps,
)
from scripts.NTU_RGBD.evaluation.metrics import compute_pckhn
from scripts.NTU_RGBD.migrate_single_pckhn import canonicalize
from scripts.NTU_RGBD.datasets.transforms import PoseTransform
from scripts.spikepose.training.losses import VisibleHeatmapMSE


def test_ntu_flip_pairs_cover_each_lateral_joint_once():
    flattened = [joint for pair in NTU_FLIP_PAIRS for joint in pair]
    assert len(flattened) == len(set(flattened))
    assert all(0 <= joint < NTU_NUM_JOINTS for joint in flattened)


def test_forced_horizontal_flip_swaps_joint_identity_and_visibility():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    keypoints = np.full((NTU_NUM_JOINTS, 2), 50.0, dtype=np.float32)
    visibility = np.ones(NTU_NUM_JOINTS, dtype=np.float32)
    keypoints[4] = (10.0, 30.0)
    keypoints[8] = (80.0, 40.0)
    visibility[4] = 0.0
    transform = PoseTransform(
        image_size=100, training=True, flip_probability=1.0,
    )

    output = transform(image, keypoints, visibility)

    np.testing.assert_allclose(output["keypoints"][4].numpy(), (19.0, 40.0))
    np.testing.assert_allclose(output["keypoints"][8].numpy(), (89.0, 30.0))
    assert output["visibility"][4].item() == 1.0
    assert output["visibility"][8].item() == 0.0


def test_transform_rejects_invalid_augmentation_configuration():
    with pytest.raises(ValueError, match="scale_range"):
        PoseTransform(scale_range=(1.2, 0.8))
    with pytest.raises(ValueError, match="flip_probability"):
        PoseTransform(flip_probability=1.1)


def test_head_length_and_heatmaps_share_input_coordinate_system():
    keypoints = np.zeros((NTU_NUM_JOINTS, 2), dtype=np.float32)
    visibility = np.zeros(NTU_NUM_JOINTS, dtype=np.float32)
    keypoints[2] = (100.0, 80.0)
    keypoints[3] = (100.0, 100.0)
    visibility[[2, 3]] = 1.0

    assert compute_head_length(keypoints, visibility) == pytest.approx(20.0)
    heatmaps = generate_gaussian_heatmaps(
        keypoints, visibility, image_size=200, heatmap_size=50, sigma=2.0,
    )
    assert heatmaps.shape == (NTU_NUM_JOINTS, 50, 50)
    assert np.unravel_index(np.argmax(heatmaps[2]), heatmaps[2].shape) == (20, 25)
    assert np.unravel_index(np.argmax(heatmaps[3]), heatmaps[3].shape) == (25, 25)


def test_invisible_joints_match_mpii_masking_contract():
    keypoints = np.full((NTU_NUM_JOINTS, 2), 25.0, dtype=np.float32)
    visibility = np.ones(NTU_NUM_JOINTS, dtype=np.float32)
    visibility[6] = 0.0
    heatmaps = generate_gaussian_heatmaps(
        keypoints, visibility, image_size=100, heatmap_size=25, sigma=2.0,
    )
    assert np.count_nonzero(heatmaps[6]) == 0
    assert np.count_nonzero(heatmaps[5]) > 0

    target = torch.from_numpy(heatmaps[None])
    prediction = target.clone()
    prediction[:, 6] = 1000.0
    loss = VisibleHeatmapMSE()(prediction, target, torch.from_numpy(visibility[None]))
    assert loss.item() == pytest.approx(0.0)


def test_out_of_bounds_joint_becomes_invisible_like_mpii():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    keypoints = np.full((NTU_NUM_JOINTS, 2), 50.0, dtype=np.float32)
    visibility = np.ones(NTU_NUM_JOINTS, dtype=np.float32)
    keypoints[6] = (150.0, 50.0)
    output = PoseTransform(image_size=100)(image, keypoints, visibility)
    assert output["visibility"][6].item() == 0.0


def test_clip_augmentation_parameters_are_shared_across_frames():
    transform = PoseTransform(
        image_size=100, training=True, scale_range=(0.8, 1.2),
        rotation_degrees=30.0, flip_probability=0.5,
    )
    parameters = transform.sample_parameters()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    keypoints = np.full((NTU_NUM_JOINTS, 2), 50.0, dtype=np.float32)
    visibility = np.ones(NTU_NUM_JOINTS, dtype=np.float32)
    first = transform(image, keypoints, visibility, parameters=parameters)
    second = transform(image, keypoints, visibility, parameters=parameters)
    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["keypoints"], second["keypoints"])
    assert torch.equal(first["visibility"], second["visibility"])


def test_pckhn_uses_head_neck_scale_boundary_and_visibility():
    target = np.zeros((2, 3, 2), dtype=np.float32)
    prediction = target.copy()
    prediction[0, 0, 0] = 5.0
    prediction[0, 1, 0] = 5.01
    prediction[1, :, 0] = 1.0
    visibility = np.ones((2, 3), dtype=np.float32)
    visibility[0, 2] = 0.0
    result = compute_pckhn(
        prediction, target, visibility,
        np.asarray([10.0, np.nan], dtype=np.float32),
        threshold=0.5,
    )
    np.testing.assert_array_equal(
        result["valid_mask"],
        np.asarray([[True, True, False], [False, False, False]]),
    )
    np.testing.assert_array_equal(
        result["correct_matrix"],
        np.asarray([[True, False, False], [False, False, False]]),
    )


def test_single_pckhn_migration_keeps_only_complete_frame_metric():
    migrated = canonicalize({
        "pckhn": 0.8,
        "pckh": 0.8,
        "per_joint_pckhn": {"head": 0.9},
        "per_joint_pckh": {"head": 0.9},
        "coordinate_space": "original_image_pixels",
    })
    assert migrated["pckhn"] == 0.8
    assert migrated["per_joint_pckhn"] == {"head": 0.9}
    assert migrated["coordinate_space"] == "complete_frame_pixels"
    assert migrated["normalization"] == "head_to_neck_distance_in_original_image_pixels"
    assert "pckh" not in migrated
    assert "per_joint_pckh" not in migrated
