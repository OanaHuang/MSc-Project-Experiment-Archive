from __future__ import annotations

import json

import numpy as np
import torch

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.evaluation.mam_v2_diagnostics import (
    default_d1_variants,
    evaluate_mam_v2_diagnostic_grid,
)


class _CountingSpatialModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward_spatial_per_step(self, image: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.zeros(
            image.shape[1], image.shape[0], 16, 64, 64,
            device=image.device,
        )


def _batch() -> dict:
    steps = 16
    keypoints = torch.zeros(1, steps, 16, 2)
    keypoints[:, :, 9, 1] = 4.0
    return {
        "image": torch.zeros(1, steps, 3, 32, 32),
        "temporal_keypoints": keypoints,
        "temporal_visibility": torch.ones(1, steps, 16),
        "temporal_frame_indices": torch.arange(steps).unsqueeze(0),
        "temporal_frame_positions": torch.arange(steps).unsqueeze(0),
        "person_bbox": torch.tensor([[0.0, 0.0, 256.0, 256.0]]),
        "sample_id": ["S001C001P001R001A001::clip_01"],
        "video_id": ["S001C001P001R001A001"],
        "clip_id": ["clip_01"],
        "person_id": ["body_1"],
    }


def test_d1_grid_reuses_one_spatial_forward_and_writes_isolated_artifacts(tmp_path) -> None:
    model = _CountingSpatialModel()
    variants = default_d1_variants()
    reports = evaluate_mam_v2_diagnostic_grid(
        model,
        [_batch()],
        load_experiment("mamv2_fullcs20"),
        torch.device("cpu"),
        variants,
        output_dir=tmp_path,
        decoder="argmax",
    )
    assert model.calls == 1
    assert reports.keys() == variants.keys()
    for name, report in reports.items():
        assert report["terminal"]["samples"] == 1
        assert report["all_frames"]["samples"] == 16
        assert report["temporal"]["num_valid_sequences"] == 1
        archive = tmp_path / name / "predictions_per_frame.npz"
        summary = tmp_path / name / "summary.json"
        assert archive.is_file()
        assert summary.is_file()
        with np.load(archive) as values:
            assert len(values["prediction"]) == 16
        assert json.loads(summary.read_text())["variant"] == name
