from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from spikepose_thesis.data.ntu.datasets.ntu_frame_dataset import NTUFrameDataset
from spikepose_thesis.evaluation.mam_reset_policy import (
    SameVideoChunkBatchSampler,
    apply_mam_reset_policy,
    trained_window_starts,
    trained_window_stitch_positions,
)


class _TimeSinceResetMAM:
    def forward_sequence(self, heatmaps: torch.Tensor):
        output = torch.empty_like(heatmaps)
        for step in range(len(heatmaps)):
            output[step].fill_(float(step))
        return SimpleNamespace(heatmap=output)


@pytest.mark.parametrize(
    ("frames", "steps", "expected"),
    [
        (16, 16, (0,)),
        (17, 16, (0, 1)),
        (31, 16, (0, 15)),
        (32, 16, (0, 16)),
        (33, 16, (0, 16, 17)),
    ],
)
def test_trained_window_starts_matches_overlapping_tail_policy(
    frames: int, steps: int, expected: tuple[int, ...],
) -> None:
    assert trained_window_starts(frames, steps) == expected


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        (16, (0,)),
        (17, (0, 16)),
        (31, (0, 16)),
        (32, (0, 16)),
        (33, (0, 16, 32)),
        (68, (0, 16, 32, 48, 64)),
    ],
)
def test_stitch_positions_describe_retained_output_boundaries(
    frames: int, expected: tuple[int, ...],
) -> None:
    assert trained_window_stitch_positions(frames, 16) == expected


def test_video_reset_carries_state_across_trained_window_boundaries() -> None:
    heatmaps = torch.zeros(33, 1, 1, 1, 1)
    output = apply_mam_reset_policy(
        _TimeSinceResetMAM(), heatmaps,
        policy="video_boundary", trained_steps=16,
    )
    torch.testing.assert_close(
        output.heatmap[:, 0, 0, 0, 0], torch.arange(33, dtype=torch.float32),
    )
    assert output.model_invocations == 1
    assert output.reset_positions == (0,)


def test_window_reset_reconstructs_every_frame_once_with_warmed_tail() -> None:
    heatmaps = torch.zeros(33, 1, 1, 1, 1)
    output = apply_mam_reset_policy(
        _TimeSinceResetMAM(), heatmaps,
        policy="trained_window", trained_steps=16,
    )
    expected = torch.cat((
        torch.arange(16), torch.arange(16), torch.tensor([15]),
    )).to(torch.float32)
    torch.testing.assert_close(output.heatmap[:, 0, 0, 0, 0], expected)
    assert output.model_invocations == 3
    assert output.reset_positions == (0, 16, 17)


def test_same_video_batch_sampler_never_mixes_samples() -> None:
    frame_index = [
        *((0, frame, None, frame) for frame in range(5)),
        *((1, frame, None, frame) for frame in range(3)),
    ]
    sampler = SameVideoChunkBatchSampler(frame_index, batch_size=2)
    batches = list(sampler)
    assert batches == [[0, 1], [2, 3], [4], [5, 6], [7]]
    assert all(
        len({frame_index[index][0] for index in batch}) == 1
        for batch in batches
    )


def _dataset_stub(scope: str) -> tuple[NTUFrameDataset, list[list[int]]]:
    dataset = NTUFrameDataset.__new__(NTUFrameDataset)
    dataset.person_crop = True
    dataset.tube_crop = True
    dataset.tube_crop_scope = scope
    dataset.preprocessed_pose_cache = False
    dataset.runtime_cache = None
    dataset.runtime_spatial_crops = False
    dataset.samples = [{"sample_id": "S001C001P001R001A001"}]
    dataset._available_frames_by_sample_clip = {(0, "full"): (0, 1, 2, 3, 4)}
    dataset._video_tube_bbox_cache = {}
    calls: list[list[int]] = []

    def tube_bbox(self, sample_index, frame_numbers, clip_name=None):
        calls.append(list(frame_numbers))
        return np.asarray([0.0, 0.0, 10.0, 10.0], dtype=np.float32)

    dataset._tube_bbox = MethodType(tube_bbox, dataset)
    return dataset, calls


def test_video_tube_crop_uses_all_frames_and_is_cached() -> None:
    dataset, calls = _dataset_stub("video")
    first = dataset._shared_temporal_bbox(0, [0, 1], None)
    second = dataset._shared_temporal_bbox(0, [3, 4], None)
    np.testing.assert_array_equal(first, second)
    assert calls == [[0, 1, 2, 3, 4]]


def test_default_window_tube_crop_keeps_window_geometry() -> None:
    dataset, calls = _dataset_stub("window")
    dataset._shared_temporal_bbox(0, [0, 1], None)
    dataset._shared_temporal_bbox(0, [3, 4], None)
    assert calls == [[0, 1], [3, 4]]
