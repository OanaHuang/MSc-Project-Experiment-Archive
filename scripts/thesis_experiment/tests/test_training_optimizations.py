import sqlite3

import cv2
import numpy as np
import torch

from spikepose_thesis.data.heatmaps import generate_gaussian_heatmaps_torch
from spikepose_thesis.data.ntu.datasets.ntu_frame_dataset import (
    NTUFrameDataset,
    generate_gaussian_heatmaps,
)
from spikepose_thesis.data.ntu.datasets.person_crop import crop_and_resize_person
from spikepose_thesis.data.ntu.datasets.person_crop import (
    compute_person_bbox,
    crop_and_resize_person_with_bbox,
)
from spikepose_thesis.data.ntu.runtime_cache import (
    RuntimeCacheReader,
    transform_keypoints_to_bbox,
)
from spikepose_thesis.evaluation.runner import collect_predictions


def test_gpu_target_definition_matches_legacy_numpy():
    points = np.asarray([
        [[20.25, 30.75], [255.0, 255.0], [np.nan, 1.0]],
        [[128.5, 100.5], [-5.0, 10.0], [40.0, 50.0]],
    ], dtype=np.float32)
    visible = np.asarray([[1, 1, 1], [1, 1, 0]], dtype=np.float32)
    expected = np.stack([
        generate_gaussian_heatmaps(item, mask, 256, 64, 2.0)
        for item, mask in zip(points, visible)
    ])
    actual = generate_gaussian_heatmaps_torch(
        torch.from_numpy(points), torch.from_numpy(visible), 256, 64, 2.0,
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_lossless_spatial_cache_preserves_crop_and_keypoints(tmp_path):
    sample_id = "S001C001P001R001A001"
    image = np.random.default_rng(7).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    points = np.asarray([[20.0, 20.0], [100.0, 70.0], [64.0, 45.0]], dtype=np.float32)
    visible = np.ones(3, dtype=np.float32)
    crop = crop_and_resize_person(image, points, visible, 64, 0.25, True)
    ok, encoded = cv2.imencode(".webp", crop.image, [cv2.IMWRITE_WEBP_QUALITY, 101])
    assert ok

    path = tmp_path / "S001.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE pose (sample_id TEXT PRIMARY KEY, body_id TEXT, "
        "num_frames INTEGER, num_joints INTEGER, color_xy BLOB, tracking_state BLOB);"
        "CREATE TABLE spatial_frame (sample_id TEXT, clip_name TEXT, frame_number INTEGER, "
        "image_webp BLOB, bbox BLOB, PRIMARY KEY(sample_id, clip_name, frame_number));"
    )
    sequence = points[None].astype("<f4")
    tracking = np.ones((1, 3), dtype="i1") * 2
    connection.execute(
        "INSERT INTO pose VALUES (?, ?, ?, ?, ?, ?)",
        (sample_id, "body_1", 1, 3, sequence.tobytes(), tracking.tobytes()),
    )
    connection.execute(
        "INSERT INTO spatial_frame VALUES (?, ?, ?, ?, ?)",
        (sample_id, "clip_01", 0, encoded.tobytes(), crop.bbox_xyxy.astype("<f4").tobytes()),
    )
    connection.commit()
    connection.close()

    reader = RuntimeCacheReader(tmp_path)
    cached_pose = reader.load_pose(sample_id)
    cached_image, cached_bbox = reader.load_spatial_frame(sample_id, "clip_01", 0)
    cached_points, cached_visibility = transform_keypoints_to_bbox(
        cached_pose["color_xy"][0], visible, cached_bbox, 64,
    )
    np.testing.assert_array_equal(cached_image, crop.image)
    np.testing.assert_allclose(cached_points, crop.keypoints)
    np.testing.assert_array_equal(cached_visibility, crop.visibility)


def test_lossless_tube_cache_preserves_shared_geometry(tmp_path):
    sample_id = "S010C001P001R001A001"
    generator = np.random.default_rng(13)
    images = [
        generator.integers(0, 256, (96, 128, 3), dtype=np.uint8)
        for _ in range(2)
    ]
    points = [
        np.asarray([[20.0, 20.0], [80.0, 60.0], [55.0, 42.0]], dtype=np.float32),
        np.asarray([[26.0, 18.0], [105.0, 72.0], [62.0, 45.0]], dtype=np.float32),
    ]
    visible = [np.ones(3, dtype=np.float32) for _ in points]
    tube_bbox = compute_person_bbox(
        np.concatenate(points), np.concatenate(visible), 128, 96, 0.25, True,
    )
    crops = [
        crop_and_resize_person_with_bbox(image, pose, mask, tube_bbox, 64)
        for image, pose, mask in zip(images, points, visible)
    ]

    path = tmp_path / "S010.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE cache_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE pose (sample_id TEXT PRIMARY KEY, body_id TEXT, "
        "num_frames INTEGER, num_joints INTEGER, color_xy BLOB, tracking_state BLOB);"
        "CREATE TABLE spatial_frame (sample_id TEXT, clip_name TEXT, frame_number INTEGER, "
        "image_webp BLOB, bbox BLOB, PRIMARY KEY(sample_id, clip_name, frame_number));"
    )
    connection.execute("INSERT INTO cache_metadata VALUES ('crop_mode', 'clip_tube')")
    sequence = np.stack(points).astype("<f4")
    tracking = np.ones((2, 3), dtype="i1") * 2
    connection.execute(
        "INSERT INTO pose VALUES (?, ?, ?, ?, ?, ?)",
        (sample_id, "body_1", 2, 3, sequence.tobytes(), tracking.tobytes()),
    )
    for frame, crop in enumerate(crops):
        ok, encoded = cv2.imencode(
            ".webp", crop.image, [cv2.IMWRITE_WEBP_QUALITY, 101],
        )
        assert ok
        connection.execute(
            "INSERT INTO spatial_frame VALUES (?, ?, ?, ?, ?)",
            (
                sample_id, "clip_01", frame, encoded.tobytes(),
                crop.bbox_xyxy.astype("<f4").tobytes(),
            ),
        )
    connection.commit()
    connection.close()

    reader = RuntimeCacheReader(tmp_path)
    assert reader.spatial_crop_mode(sample_id) == "clip_tube"
    for frame, expected in enumerate(crops):
        np.testing.assert_array_equal(
            reader.load_spatial_bbox(sample_id, "clip_01", frame),
            expected.bbox_xyxy,
        )
        cached_image, cached_bbox = reader.load_spatial_frame(
            sample_id, "clip_01", frame,
        )
        cached_points, cached_visibility = transform_keypoints_to_bbox(
            points[frame], visible[frame], cached_bbox, 64,
        )
        np.testing.assert_array_equal(cached_image, expected.image)
        np.testing.assert_array_equal(cached_bbox, expected.bbox_xyxy)
        np.testing.assert_allclose(cached_points, expected.keypoints)
        np.testing.assert_array_equal(cached_visibility, expected.visibility)


def test_short_temporal_window_uses_complete_clip_cached_bbox():
    cached_bbox = np.asarray([10.0, 12.0, 110.0, 92.0], dtype=np.float32)

    class Cache:
        @staticmethod
        def spatial_crop_mode(sample_id):
            assert sample_id == "S001C001P001R001A001"
            return "clip_tube"

        @staticmethod
        def load_spatial_bbox(sample_id, clip_name, frame_number):
            assert (sample_id, clip_name, frame_number) == (
                "S001C001P001R001A001", "clip_00", 18,
            )
            return cached_bbox.copy()

    dataset = NTUFrameDataset.__new__(NTUFrameDataset)
    dataset.person_crop = True
    dataset.tube_crop = True
    dataset.preprocessed_pose_cache = False
    dataset.runtime_cache = Cache()
    dataset.runtime_spatial_crops = True
    dataset.samples = [{"sample_id": "S001C001P001R001A001"}]
    dataset._tube_bbox = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("window-local geometry must not be recomputed")
    )

    actual = dataset._shared_temporal_bbox(0, [18, 19], "clip_00")
    np.testing.assert_array_equal(actual, cached_bbox)


class _CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        return torch.zeros(len(image), 16, 64, 64)


def test_validation_predictions_and_loss_share_one_forward_pass():
    model = _CountingModel()
    batch = {
        "image": torch.zeros(2, 3, 64, 64),
        "keypoints_original": torch.zeros(2, 16, 2),
        "visibility": torch.ones(2, 16),
        "original_visibility": torch.ones(2, 16),
        "person_bbox": torch.tensor([[0.0, 0.0, 256.0, 256.0]] * 2),
        "sample_id": ["S001C001P001R001A001"] * 2,
        "person_id": ["body_1"] * 2,
        "frame_index": torch.arange(2),
    }
    config = {
        "dataset": "ntu60_cs",
        "data": {"image_size": 256},
        "evaluation": {"main_decoder": "argmax"},
    }
    values, loss = collect_predictions(
        model, [batch], config, torch.device("cpu"), decoder="argmax",
        loss_callback=lambda prediction, _: prediction.square().mean(),
    )
    assert model.calls == 1
    assert loss == 0.0
    assert len(values["prediction"]) == 2
