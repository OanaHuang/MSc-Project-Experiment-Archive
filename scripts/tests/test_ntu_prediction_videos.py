from types import SimpleNamespace

import pytest

from scripts.NTU_RGBD.render_prediction_videos import select_sample_ids
from scripts.NTU_RGBD.evaluate_original_videos import original_video_config
from scripts.NTU_RGBD.evaluation.metrics import crop_to_original
import numpy as np


def test_video_selection_is_unique_and_reproducible():
    dataset = SimpleNamespace(samples=[
        {"sample_id": "c"}, {"sample_id": "a"}, {"sample_id": "b"},
    ])
    first = select_sample_ids(dataset, 2, 42)
    second = select_sample_ids(dataset, 2, 42)
    assert first == second
    assert len(first) == len(set(first)) == 2


def test_video_selection_rejects_invalid_counts():
    dataset = SimpleNamespace(samples=[{"sample_id": "a"}])
    with pytest.raises(ValueError):
        select_sample_ids(dataset, 0, 42)
    with pytest.raises(ValueError):
        select_sample_ids(dataset, 2, 42)


def test_original_video_evaluation_uses_complete_contiguous_frames():
    config = {"data": {"extracted_frames_dir": "processed", "frame_stride": 16}}
    resolved = original_video_config(config, "full", "validation.csv")
    assert resolved["data"] == {
        "extracted_frames_dir": "full",
        "validation_metadata": "validation.csv",
        "frame_stride": 1,
        "temporal_frame_gap": 1,
        "minimum_temporal_history": 3,
        "preprocessed_pose_cache": False,
    }
    assert config["data"]["extracted_frames_dir"] == "processed"


def test_original_video_evaluation_preserves_trained_temporal_gap():
    config = {"data": {
        "extracted_frames_dir": "processed", "frame_stride": 16,
        "temporal_frame_gap": 3, "minimum_temporal_history": 5,
    }}
    resolved = original_video_config(config, "full", "validation.csv")
    assert resolved["data"]["frame_stride"] == 1
    assert resolved["data"]["temporal_frame_gap"] == 3
    assert resolved["data"]["minimum_temporal_history"] == 5


def test_crop_coordinates_are_mapped_to_original_image_pixels():
    points = np.array([[0.0, 0.0], [128.0, 64.0], [256.0, 256.0]])
    bbox = np.array([100.0, 50.0, 500.0, 250.0])
    restored = crop_to_original(points, bbox, image_size=256)
    np.testing.assert_allclose(
        restored,
        np.array([[100.0, 50.0], [300.0, 100.0], [500.0, 250.0]]),
    )
