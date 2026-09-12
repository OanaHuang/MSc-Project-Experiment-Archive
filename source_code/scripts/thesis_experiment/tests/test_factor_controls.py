from __future__ import annotations

from pathlib import Path

import torch

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.data.ntu.datasets.ntu_frame_dataset import NTUFrameDataset
from spikepose_thesis.models import build_model


CONDITIONS = {
    "v1u1": (1, 1, (0,)),
    "v1u2": (1, 2, (0, 0)),
    "v1u4": (1, 4, (0, 0, 0, 0)),
    "v2u2": (2, 2, (0, 1)),
    "v2u4": (2, 4, (0, 0, 1, 1)),
    "v4u4": (4, 4, (0, 1, 2, 3)),
}


def test_factor_configs_are_matched_and_use_exact_schedules() -> None:
    for prefix, epochs, source in (
        ("pilot20", 20, "pilot20_m_s3_u2"),
        ("confirm140", 140, "confirm140_m_s3_u2"),
    ):
        for condition, (frames, updates, schedule) in CONDITIONS.items():
            config = load_experiment(f"{prefix}_factor_{condition}")
            temporal = config["model"]["temporal"]
            assert config["dataset"] == "ntu60_cs"
            assert config["data"].get("setup_filter") is None
            assert config["data"]["minimum_temporal_history"] == 3
            assert config["training"]["epochs"] == epochs
            assert config["training"]["seeds"] == [42]
            assert config["initialization"]["source"] == source
            assert temporal["video_frames"] == frames
            assert config["model"]["num_steps"] == updates
            assert temporal["input_strategy"] == "scheduled_frames"
            assert tuple(temporal["update_schedule"]) == schedule
            assert temporal["aggregation"] == "last"
            assert temporal["state_mode"] == "continuous"
            assert temporal["decouple_video_time"] is False
            assert config["model"]["neck"]["kind"] == "spike_fpn_temporal"
            assert (
                config["training"]["batch_size"]
                * config["training"]["gradient_accumulation"]
                == 128
            )


def test_v2u4_expands_to_one_continuous_ordered_trajectory() -> None:
    model = build_model(load_experiment("pilot20_factor_v2u4")).eval()
    physical = torch.tensor([10.0, 20.0]).reshape(1, 2, 1, 1, 1)
    sequence = model._scheduled_sequence(physical)
    assert sequence.shape == (4, 1, 1, 1, 1)
    assert sequence[:, 0, 0, 0, 0].tolist() == [10.0, 10.0, 20.0, 20.0]


def _factor_index(root: Path, temporal_steps: int) -> list[tuple]:
    sample_id = "S001C001P001R001A001"
    dataset = NTUFrameDataset.__new__(NTUFrameDataset)
    dataset.samples = [{
        "sample_id": sample_id, "rgb_frames": 16, "skeleton_frames": 16,
    }]
    dataset.skipped_samples = []
    dataset.extracted_frames_dir = root
    dataset.frame_layout = "setup_contiguous_clips"
    dataset.frame_clip_subdir = "contiguous_2x16"
    dataset.frame_stride = 1
    dataset.temporal_steps = temporal_steps
    dataset.temporal_frame_gap = 1
    dataset.minimum_temporal_history = 3
    dataset.exclude_overlapping_clips = True
    clip = root / "S001" / "contiguous_2x16" / sample_id / "clip_01"
    clip.mkdir(parents=True, exist_ok=True)
    for frame in range(16):
        (clip / f"frame_{frame:06d}.jpg").touch()
    return dataset._build_frame_index()


def test_all_factor_conditions_share_the_same_target_population(tmp_path: Path) -> None:
    indices = [_factor_index(tmp_path, steps) for steps in (1, 2, 4)]
    targets = [[(item[1], item[3]) for item in values] for values in indices]
    assert targets[0] == targets[1] == targets[2]
    assert targets[0] == [(frame, frame) for frame in range(3, 16)]


def test_factor_forward_paths_return_current_pose_heatmaps() -> None:
    for condition, (frames, _updates, _schedule) in CONDITIONS.items():
        model = build_model(load_experiment(f"pilot20_factor_{condition}")).eval()
        image = torch.zeros(1, frames, 3, 64, 64)
        with torch.no_grad():
            prediction = model(image)
            per_frame = model.forward_per_step(image)
        assert prediction.shape == (1, 16, 64, 64)
        assert per_frame.shape == (frames, 1, 16, 64, 64)
        torch.testing.assert_close(prediction, per_frame[-1])
