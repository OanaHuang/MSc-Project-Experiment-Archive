from pathlib import Path

import pytest
import torch

from scripts.NTU_RGBD.train_jts_series import EXPERIMENTS, build_command
from scripts.NTU_RGBD.probe_jts_stages import hand_group_summary
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


def test_jts_manifest_and_shared_t0_population():
    assert [item.experiment for item in EXPERIMENTS] == [f"jts{i}" for i in range(6)]
    for item in EXPERIMENTS:
        command = build_command(item.experiment)
        assert command[command.index("--category") + 1] == "joint_timestep"
        assert command[command.index("--minimum-temporal-history") + 1] == "3"
        assert "--disable-augmentation" in command
    assert "--resume" in build_command("jts4", resume=True)
    with pytest.raises(ValueError, match="Unknown JTS"):
        build_command("jts99")


def test_jts_configs_isolate_repeated_current_snn_time():
    assert resolve("jts0")["model"]["num_steps"] == 1
    expected = {
        "jts1": None,
        "jts2": [4, 3, 2, 1],
        "jts3": [4, 4, 3, 1],
        "jts4": [4, 4, 4, 1],
        "jts5": None,
    }
    for experiment, stage_steps in expected.items():
        config = resolve(experiment)
        assert config["model"]["temporal"]["input_strategy"] == "repeat"
        assert config["model"]["temporal"].get("stage_steps") == stage_steps


def test_jts_shrinkage_models_forward_one_current_frame():
    for experiment in ("jts2", "jts3", "jts4"):
        model = build_model(tiny(experiment)).eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 64, 64))
        assert tuple(output.shape) == (2, 25, 64, 64)


def test_jts_stage_probe_hand_group_summary():
    values = {
        "wrist_left": 0.6, "wrist_right": 0.8,
        "hand_left": 0.5, "hand_right": 0.7,
        "hand_tip_left": 0.4, "hand_tip_right": 0.6,
        "thumb_left": 0.3, "thumb_right": 0.5,
    }
    summary = hand_group_summary(values)
    assert summary == {
        "wrist": pytest.approx(0.7), "hand": pytest.approx(0.6),
        "hand_tip": pytest.approx(0.5), "thumb": pytest.approx(0.4),
        "hand_group": pytest.approx(0.55),
    }
