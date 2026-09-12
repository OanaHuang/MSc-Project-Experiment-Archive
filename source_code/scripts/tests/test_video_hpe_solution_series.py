from __future__ import annotations

from pathlib import Path

import torch

from scripts.NTU_RGBD.train_video_hpe_queue import RUNS, build_command
from scripts.spikepose.experiments import experiment_roots, resolve_config, validate_config
from scripts.spikepose.models import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve(experiment: str) -> dict:
    config = resolve_config(
        experiment_roots(PROJECT_ROOT, "ntu_rgbd"), experiment,
        PROJECT_ROOT / "scripts/NTU_RGBD/configs/task.yaml",
        PROJECT_ROOT / "scripts/NTU_RGBD/configs/training.yaml",
    )
    validate_config(config)
    return config


def tiny(experiment: str) -> dict:
    config = resolve(experiment)
    config["model"]["backbone"] = {
        "channels": [4, 8, 8, 8], "depths": [1, 1, 1, 1],
        "blocks": ["conv", "conv", "conv", "conv"],
        "output_stages": [1, 2, 3, 4], "attention_stages": [],
    }
    config["model"]["neck"]["out_channels"] = 8
    config["model"]["head"]["hidden_channels"] = 8
    validate_config(config)
    return config


def test_v_series_maps_four_literature_variants_to_four_gpus():
    assert [run.experiment for run in RUNS] == ["v1", "v2", "v3", "v4"]
    assert [run.gpu for run in RUNS] == [0, 1, 2, 3]
    for run in RUNS:
        command = build_command(run)
        assert command[command.index("--epochs") + 1] == "20"
        assert command[command.index("--category") + 1] == "video_hpe_solution"
        assert command[command.index("--minimum-temporal-history") + 1] == "3"


def test_v_series_resolves_to_distinct_temporal_solutions():
    expected = {
        "v1": "joint_residual_correction",
        "v2": "joint_difference_temporal",
        "v3": "joint_decoupled_space_time",
        "v4": "joint_aligned_temporal",
    }
    for experiment, aggregation in expected.items():
        config = resolve(experiment)
        assert config["model"]["neck"]["kind"] == "spike_fpn_temporal"
        assert config["model"]["temporal"]["aggregation"] == aggregation
    assert resolve("v4")["model"]["temporal"]["kinematic_loss_weight"] == 0.1
    assert len(resolve("v3")["model"]["temporal"]["feedback_edges"]) == 24


def test_v_series_forward_and_intermediate_sequences_are_differentiable():
    for experiment in ("v1", "v2", "v3", "v4"):
        model = build_model(tiny(experiment))
        prediction, intermediates = model(
            torch.randn(2, 4, 3, 32, 32), return_intermediates=True,
        )
        assert tuple(prediction.shape) == (2, 25, 64, 64)
        assert len(intermediates) == 4
        (prediction.square().mean() + sum(
            item.square().mean() for item in intermediates
        )).backward()


def test_temporal_dataset_exposes_targets_for_acceleration_supervision():
    from scripts.NTU_RGBD.datasets.ntu_frame_dataset import NTUFrameDataset

    dataset = object.__new__(NTUFrameDataset)
    dataset.frame_index = [(0, 3)]
    dataset.temporal_steps = 4
    dataset.temporal_frame_gap = 1
    dataset.return_temporal_sequence = True
    dataset.return_temporal_targets = True
    dataset.transform = None

    def get_item(sample_index, frame, parameters, include_target=True):
        item = {"image": torch.full((3, 2, 2), float(frame))}
        if include_target:
            item.update({
                "heatmaps": torch.full((1, 2, 2), float(frame)),
                "visibility": torch.ones(1),
            })
        return item

    dataset._get_frame_item = get_item
    output = dataset[0]
    assert tuple(output["temporal_heatmaps"].shape) == (4, 1, 2, 2)
    assert output["temporal_heatmaps"][:, 0, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
