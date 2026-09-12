from pathlib import Path

import numpy as np
import torch

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.evaluation.statistics import paired_sequence_bootstrap
from spikepose_thesis.evaluation.runner import _to_original_ntu, evaluate_checkpoint
from spikepose_thesis.evaluation.temporal import (
    contiguous_sequence_groups, temporal_summary,
)
from spikepose_thesis.experiments.readiness import experiment_readiness


def _values(offset: float = 0.0) -> dict[str, np.ndarray]:
    frames = np.asarray([0, 1, 2, 3, 20, 21, 22, 23])
    target = np.zeros((len(frames), 16, 2), dtype=np.float32)
    prediction = target.copy()
    prediction[..., 0] = offset
    return {
        "prediction": prediction,
        "target": target,
        "visibility": np.ones((len(frames), 16), dtype=np.float32),
        "scale": np.ones(len(frames), dtype=np.float32),
        "sample_id": np.asarray(["S001C001P001R001A001"] * len(frames)),
        "person_id": np.asarray(["body_1"] * len(frames)),
        "frame_index": frames,
    }


def test_temporal_metrics_split_nonconsecutive_clips():
    values = _values()
    groups = contiguous_sequence_groups(values)
    assert [item.tolist() for item in groups] == [list(range(4)), list(range(4, 8))]
    assert paired_sequence_bootstrap(values, values, repeats=1000)["sequences"] == 1
    assert temporal_summary(values)["contiguous_segments"] == 2


def test_temporal_metrics_split_adjacent_frame_numbers_by_clip_identity():
    values = _values()
    values["frame_index"] = np.arange(8)
    values["sample_id"] = np.asarray(
        ["S001C001P001R001A001::clip_01"] * 4
        + ["S001C001P001R001A001::clip_02"] * 4
    )
    groups = contiguous_sequence_groups(values)
    assert [item.tolist() for item in groups] == [list(range(4)), list(range(4, 8))]


def test_paired_bootstrap_has_required_ci():
    report = paired_sequence_bootstrap(_values(0.2), _values(0.1), repeats=1000)
    assert report["repeats"] == 1000
    assert report["metrics"]["pck_0.5"]["ci95_low"] == 0.0
    assert report["metrics"]["nme_hb"]["ci95_high"] < 0.0


def test_ntu_crop_coordinates_restore_to_complete_frame():
    points = np.asarray([[0.0, 0.0], [256.0, 256.0], [128.0, 64.0]])
    restored = _to_original_ntu(
        points, np.asarray([100.0, 50.0, 300.0, 150.0]), 256,
    )
    np.testing.assert_allclose(
        restored,
        np.asarray([[100.0, 50.0], [300.0, 150.0], [200.0, 75.0]]),
    )


def test_readiness_accepts_16_frame_refinement():
    report = {
        "checks": {
            "ntu_skeletons_complete": True,
            "ntu_frames_complete": True,
                "ntu_clip_layout_valid": True,
                "ntu_clips_nonoverlapping": True,
                "ntu_quality_exclusions_complete": True,
                "ntu_pose_preflight_ready": True,
                "ntu_runtime_cache_ready": True,
                "ntu_cross_subject_metadata": True,
                "ntu_cross_subject_disjoint": True,
                "ntu_cross_subject_performance_groups_disjoint": True,
        },
        "contiguous_clip_length": 16,
    }
    readiness = experiment_readiness(
        load_experiment("pilot20_eval_ema_shared"), report=report,
    )
    assert readiness["status"] == "READY"
    assert not readiness["blockers"]


def test_s010_readiness_uses_its_scoped_cache(monkeypatch):
    report = {
        "checks": {
            "ntu_skeletons_complete": True,
            "ntu_frames_complete": True,
            "ntu_clip_layout_valid": True,
            "ntu_clips_nonoverlapping": True,
            "ntu_quality_exclusions_complete": True,
            "ntu_pose_preflight_ready": True,
            "ntu_runtime_cache_ready": False,
            "ntu_cross_subject_metadata": True,
            "ntu_cross_subject_disjoint": True,
            "ntu_cross_subject_performance_groups_disjoint": True,
        },
        "contiguous_clip_length": 16,
    }
    monkeypatch.setattr(
        "spikepose_thesis.data.ntu.runtime_cache.runtime_cache_status",
        lambda data: {"ready": True, "reason": ""},
    )
    readiness = experiment_readiness(
        load_experiment("mamv2_p06_noalign", profile="pilot20"), report=report,
    )
    assert readiness == {"status": "READY", "blockers": []}


class _ZeroHeatmaps(torch.nn.Module):
    def forward(self, image):
        return torch.zeros(len(image), 16, 64, 64, device=image.device)


class _ZeroPerFrameHeatmaps(torch.nn.Module):
    def forward_per_step(self, image):
        return torch.zeros(
            image.shape[1], image.shape[0], 16, 64, 64,
            device=image.device,
        )


def test_unified_ntu_evaluation_emits_temporal_metrics(tmp_path):
    original = torch.zeros(4, 16, 2)
    original[:, 9, 1] = 1.0
    batch = {
        "image": torch.zeros(4, 3, 64, 64),
        "keypoints": torch.zeros(4, 16, 2),
        "keypoints_original": original,
        "visibility": torch.ones(4, 16),
        "original_visibility": torch.ones(4, 16),
        "head_length": torch.ones(4),
        "person_bbox": torch.tensor([[0.0, 0.0, 256.0, 256.0]] * 4),
        "sample_id": ["S001C001P001R001A001"] * 4,
        "person_id": ["body_1"] * 4,
        "frame_index": torch.arange(4),
    }
    config = load_experiment("pilot20_ntu_simplebaseline_r50")
    summary = evaluate_checkpoint(
        _ZeroHeatmaps(), [batch], config, torch.device("cpu"), tmp_path,
    )
    assert "temporal" in summary
    assert summary["temporal"]["contiguous_segments"] == 1
    assert (tmp_path / "predictions.npz").is_file()


def test_mam_evaluation_exports_strict_per_frame_rows(tmp_path):
    steps = 16
    temporal_keypoints = torch.zeros(1, steps, 16, 2)
    temporal_keypoints[:, :, 9, 1] = 1.0
    original = temporal_keypoints[:, -1].clone()
    batch = {
        "image": torch.zeros(1, steps, 3, 64, 64),
        "keypoints": temporal_keypoints[:, -1],
        "keypoints_original": original,
        "visibility": torch.ones(1, 16),
        "original_visibility": torch.ones(1, 16),
        "person_bbox": torch.tensor([[0.0, 0.0, 256.0, 256.0]]),
        "sample_id": ["S001C001P001R001A001::clip_01"],
        "person_id": ["body_1"],
        "frame_index": torch.tensor([3]),
        "temporal_keypoints": temporal_keypoints,
        "temporal_visibility": torch.ones(1, steps, 16),
        "temporal_frame_indices": torch.arange(steps).unsqueeze(0),
        "temporal_frame_positions": torch.arange(steps).unsqueeze(0),
        "video_id": ["S001C001P001R001A001"],
        "clip_id": ["clip_01"],
        "frame_position_in_clip": torch.tensor([steps - 1]),
    }
    config = load_experiment("mamv2_fullcs20")
    summary = evaluate_checkpoint(
        _ZeroPerFrameHeatmaps(), [batch], config, torch.device("cpu"), tmp_path,
    )
    assert summary["pck_hb"]["samples"] == 1
    assert summary["temporal"]["contiguous_segments"] == 1
    with np.load(tmp_path / "predictions.npz") as saved:
        assert len(saved["prediction"]) == 1
    assert (tmp_path / "predictions_per_frame.npz").exists()
    with np.load(tmp_path / "predictions_per_frame.npz") as saved:
        assert len(saved["prediction"]) == steps
        assert saved["video_id"].tolist() == ["S001C001P001R001A001"] * steps
        assert saved["clip_id"].tolist() == ["clip_01"] * steps
        assert saved["frame_position_in_clip"].tolist() == list(range(steps))
    assert summary["temporal"]["num_valid_sequences"] == 1
    assert summary["temporal"]["num_velocity_intervals"] == 15
    assert summary["temporal"]["num_acceleration_intervals"] == 14


def test_unified_mpii_evaluation_emits_flip_dark(tmp_path):
    batch = {
        "image": torch.zeros(2, 3, 64, 64),
        "keypoints_original": torch.zeros(2, 16, 2),
        "visibility": torch.ones(2, 16),
        "head_length": torch.ones(2),
        "inverse": torch.tensor([[1.0, 1.0, 0.0, 0.0]] * 2),
        "image_name": ["a.jpg", "b.jpg"],
        "person_index": torch.tensor([0, 0]),
    }
    summary = evaluate_checkpoint(
        _ZeroHeatmaps(), [batch], load_experiment("pilot20_m_s3_u1"),
        torch.device("cpu"), tmp_path,
    )
    assert "flip_dark" in summary
    assert (tmp_path / "predictions_flip_dark.npz").is_file()
