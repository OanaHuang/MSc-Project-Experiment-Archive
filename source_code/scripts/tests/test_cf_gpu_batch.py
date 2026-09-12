from __future__ import annotations

from scripts.NTU_RGBD.train_cf_gpu_batch import RUNS, build_command
from scripts.spikepose.experiments import (
    experiment_roots, resolve_config, validate_config,
)
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve(experiment: str) -> dict:
    config = resolve_config(
        experiment_roots(PROJECT_ROOT, "ntu_rgbd"), experiment,
        PROJECT_ROOT / "scripts/NTU_RGBD/configs/task.yaml",
        PROJECT_ROOT / "scripts/NTU_RGBD/configs/training.yaml",
    )
    validate_config(config)
    return config


def test_cf_batch_maps_exactly_four_runs_to_four_gpus():
    assert len(RUNS) == 4
    assert [run.gpu for run in RUNS] == [0, 1, 2, 3]
    assert [run.experiment for run in RUNS] == ["t5", "cf6", "cf7", "cf8"]


def test_cf_controls_only_change_the_history_mode():
    expected = {
        "cf6": "repeat_current",
        "cf7": "reverse",
        "cf8": "batch_shuffle_history",
    }
    for experiment, history_mode in expected.items():
        config = resolve(experiment)
        assert config["model"]["num_steps"] == 4
        assert config["model"]["temporal"]["input_strategy"] == "frames"
        assert config["model"]["temporal"]["aggregation"] == "joint_temporal"
        assert config["model"]["temporal"]["history_mode"] == history_mode


def test_cf_batch_commands_are_fixed_to_twenty_epochs_and_shared_data():
    for run in RUNS:
        command = build_command(run)
        assert command[command.index("--epochs") + 1] == "20"
        assert command[command.index("--category") + 1] == "cross_frame"
        assert command[command.index("--physical-gpu") + 1] == str(run.gpu)
        assert command[command.index("--device") + 1] == f"cuda:{run.gpu}"
        assert command[command.index("--minimum-temporal-history") + 1] == "3"
        assert "--preprocessed-pose-cache" in command
        assert "--disable-augmentation" in command
        assert "--skip-visualization" in command
