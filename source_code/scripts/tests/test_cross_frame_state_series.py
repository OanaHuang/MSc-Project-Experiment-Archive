from __future__ import annotations

from pathlib import Path

import torch

from scripts.NTU_RGBD.train_cross_frame_series import (
    EXPERIMENTS, build_command,
)
from scripts.spikepose.experiments import (
    experiment_roots, resolve_config, validate_config,
)
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


def test_manifest_contains_baselines_and_dsta_inspired_models():
    assert [item.experiment for item in EXPERIMENTS] == [f"t{i}" for i in range(6)]
    assert max(item.frames for item in EXPERIMENTS) == 4
    assert resolve("t0")["model"]["backbone"]["output_stages"] == [1, 2, 3, 4]
    assert all(resolve(item.experiment)["model"]["head"]["kind"] == "linear_heatmap"
               for item in EXPERIMENTS)


def test_launcher_enforces_one_seed_and_shared_current_frame_population():
    commands = [build_command(item.experiment) for item in EXPERIMENTS]
    for command in commands:
        assert command[command.index("--seed") + 1] == "42"
        assert command[command.index("--extracted-frames-dir") + 1].endswith(
            "frames/S010/clip4_256"
        )
        assert command[command.index("--minimum-temporal-history") + 1] == "3"
        assert command[command.index("--category") + 1] == "t_series"
        assert "--preprocessed-pose-cache" in command
        assert "--disable-augmentation" in command
    assert len({command[command.index("--batch-name") + 1] for command in commands}) == 1


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


def test_dsta_inspired_models_have_distinct_jointwise_readouts():
    expected = {
        "t2": "joint_temporal",
        "t3": "joint_temporal_spatial",
        "t4": "joint_confidence_temporal",
        "t5": "joint_motion_temporal",
    }
    for experiment, aggregation in expected.items():
        model = build_model(tiny(experiment)).eval()
        assert model.head.aggregation == aggregation
        assert tuple(model.head.temporal_logits.shape) == (4, 25)
        with torch.no_grad():
            output = model(torch.randn(2, 4, 3, 64, 64))
        assert tuple(output.shape) == (2, 25, 64, 64)


def test_decoupled_model_uses_normalized_ntu_skeleton_graph():
    model = build_model(tiny("t3"))
    assert tuple(model.head.joint_adjacency.shape) == (25, 25)
    assert torch.allclose(model.head.joint_adjacency.sum(1), torch.ones(25))
