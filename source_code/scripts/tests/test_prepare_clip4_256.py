from pathlib import Path
import tempfile

import cv2
import numpy as np

from scripts.NTU_RGBD.prepare_clip4_256 import clip_frame_groups
from scripts.NTU_RGBD.datasets.person_crop import (
    crop_and_resize_person_with_bbox,
)


def test_clip_groups_keep_four_consecutive_frames_every_sixteen():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for frame in range(36):
            (root / f"frame_{frame:06d}.jpg").touch()
        groups = clip_frame_groups(root, 36)
        assert [[int(path.stem[-6:]) for path in group] for group in groups] == [
            [0, 1, 2, 3], [16, 17, 18, 19], [32, 33, 34, 35],
        ]


def test_clip_groups_support_gap_five_with_shared_six_frame_targets():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for frame in range(36):
            (root / f"frame_{frame:06d}.jpg").touch()
        groups = clip_frame_groups(root, 36, clip_length=6, clip_start_gap=16)
        assert [[int(path.stem[-6:]) for path in group] for group in groups] == [
            list(range(0, 6)), list(range(16, 22)),
        ]


def test_shared_bbox_produces_256_square_clip_coordinates():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    keypoints = np.asarray([[800.0, 300.0], [1000.0, 800.0]], dtype=np.float32)
    visibility = np.ones(2, dtype=np.float32)
    bbox = np.asarray([700.0, 200.0, 1100.0, 900.0], dtype=np.float32)
    result = crop_and_resize_person_with_bbox(
        image, keypoints, visibility, bbox, 256,
    )
    assert result.image.shape == (256, 256, 3)
    np.testing.assert_array_equal(result.bbox_xyxy, bbox)
    assert np.all(result.keypoints >= 0)
    assert np.all(result.keypoints < 256)
    assert cv2.countNonZero(cv2.cvtColor(result.image, cv2.COLOR_BGR2GRAY)) == 0
