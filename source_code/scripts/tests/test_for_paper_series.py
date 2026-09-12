from pathlib import Path

import torch

from scripts.NTU_RGBD.train_for_paper_queue import PAPER_SEEDS, jobs_for_phase
from scripts.NTU_RGBD.train_for_paper_series import EXPERIMENTS, build_command
from scripts.spikepose.experiments import experiment_roots, resolve_config, validate_config
from scripts.spikepose.models import build_model


ROOT = Path(__file__).resolve().parents[2]


def resolve(experiment: str) -> dict:
    value = resolve_config(
        experiment_roots(ROOT, "ntu_rgbd"), experiment,
        ROOT / "scripts/NTU_RGBD/configs/task.yaml",
        ROOT / "scripts/NTU_RGBD/configs/training.yaml",
    )
    validate_config(value)
    return value


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


def test_fp_ids_and_method_names_are_stable():
    assert [item.experiment for item in EXPERIMENTS] == [f"fp{i}" for i in range(9)]
    expected = [
        "CurrentFrame", "RepeatedCurrent", "FourFrameMean", "JointTemporal",
        "JointAlignedTemporal", "AlignedConfidenceGate", "CausalPoseResidualCorrection",
        "DecoupledTimeMean", "DecoupledTimeResidualCorrection",
    ]
    assert [resolve(f"fp{i}")["name"].split(f"FP{i}-", 1)[1] for i in range(9)] == expected


def test_fp_configs_form_the_declared_controlled_ladder():
    expected = {
        "fp0": (1, "mean", "real"),
        "fp1": (4, "joint_temporal", "repeat_current"),
        "fp2": (4, "mean", "real"),
        "fp3": (4, "joint_temporal", "real"),
        "fp4": (4, "joint_aligned_temporal", "real"),
        "fp5": (4, "joint_aligned_confidence_temporal", "real"),
        "fp6": (4, "joint_residual_correction", "real"),
        "fp7": (4, "mean", "real"),
        "fp8": (4, "joint_residual_correction", "real"),
    }
    for experiment, signature in expected.items():
        config = resolve(experiment)
        assert (
            config["model"]["num_steps"],
            config["model"]["temporal"]["aggregation"],
            config["model"]["temporal"]["history_mode"],
        ) == signature


def test_command_preserves_shared_population_and_output_protocol():
    command = build_command("fp5", seed=2071461405, epochs=120, physical_gpu=2)
    assert command[command.index("--category") + 1] == "for_paper"
    assert command[command.index("--batch-name") + 1] == "clip4_256_120ep"
    assert command[command.index("--minimum-temporal-history") + 1] == "3"
    assert command[command.index("--seed") + 1] == "2071461405"
    assert "--disable-augmentation" in command
    assert "--preprocessed-pose-cache" in command
    fp8 = build_command("fp8")
    assert fp8[fp8.index("--batch-size") + 1] == "4"


def test_decoupled_time_models_forward_and_backpropagate():
    for experiment in ("fp7", "fp8"):
        config = tiny(experiment)
        assert config["model"]["temporal"]["snn_steps_per_frame"] == 4
        assert config["model"]["temporal"]["state_mode"] == "reset"
        model = build_model(config)
        prediction = model(torch.randn(2, 4, 3, 32, 32))
        assert tuple(prediction.shape) == (2, 25, 64, 64)
        prediction.square().mean().backward()


def test_queue_encodes_screen_multiseed_main_and_extension():
    assert len(jobs_for_phase("screen")) == 9
    main = jobs_for_phase("main")
    assert len(main) == 9 * len(PAPER_SEEDS)
    assert {job.seed for job in main} == set(PAPER_SEEDS)
    assert {job.epochs for job in main} == {120}
    extension = jobs_for_phase("extension")
    assert len(extension) == 9
    assert {job.seed for job in extension} == {42}
    assert {job.epochs for job in extension} == {140}
