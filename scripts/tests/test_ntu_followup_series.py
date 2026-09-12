from __future__ import annotations

from pathlib import Path

import torch

from scripts.NTU_RGBD.train_ntu_followup_queue import QUEUE, build_command
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


def test_followup_queue_uses_four_gpus_and_shared_g_targets():
    assert [len(batch) for batch in QUEUE] == [4, 4, 1]
    assert [run.experiment for run in QUEUE[0]] == ["g1", "g2", "g3", "g5"]
    assert [run.gpu for run in QUEUE[0]] == [0, 1, 2, 3]
    assert [run.experiment for batch in QUEUE[1:] for run in batch] == [
        "cf9", "cf10", "cf11", "cf12", "cf13",
    ]
    for run in QUEUE[0]:
        command = build_command(run)
        assert command[command.index("--extracted-frames-dir") + 1].endswith("clip6_256")
        assert command[command.index("--minimum-temporal-history") + 1] == "5"
        assert command[command.index("--temporal-frame-gap") + 1] == str(run.gap)


def test_history_only_removes_current_frame_pixels():
    model = build_model(tiny("cf9"))
    image = torch.arange(2 * 4 * 3 * 2 * 2).reshape(2, 4, 3, 2, 2).float()
    controlled = model._apply_history_control(image)
    assert torch.equal(controlled[:, :3], image[:, :3])
    assert torch.equal(controlled[:, 3], image[:, 2])
    assert not torch.equal(controlled[:, 3], image[:, 3])


def test_aligned_models_have_distinct_valid_readouts():
    expected = {
        "cf10": "joint_aligned_temporal",
        "cf11": "joint_aligned_temporal",
        "cf12": "joint_aligned_motion_temporal",
        "cf13": "joint_aligned_confidence_temporal",
    }
    for experiment, aggregation in expected.items():
        model = build_model(tiny(experiment)).eval()
        assert model.head.aggregation == aggregation
        with torch.no_grad():
            output = model(torch.randn(2, 4, 3, 32, 32))
        assert tuple(output.shape) == (2, 25, 64, 64)
    assert resolve("cf11")["model"]["temporal"]["intermediate_loss_weight"] == 0.2


def test_aligned_forecast_returns_history_predictions_for_auxiliary_loss():
    model = build_model(tiny("cf11")).eval()
    with torch.no_grad():
        prediction, intermediates = model(
            torch.randn(2, 4, 3, 32, 32), return_intermediates=True,
        )
    assert tuple(prediction.shape) == (2, 25, 64, 64)
    assert len(intermediates[:-1]) == 3
