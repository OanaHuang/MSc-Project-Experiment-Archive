from pathlib import Path

import torch

from scripts.NTU_RGBD.train_t_ssnn_series import EXPERIMENTS, build_command
from scripts.spikepose.experiments import experiment_roots, resolve_config, validate_config
from scripts.spikepose.models import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def tiny(experiment: str) -> dict:
    config = resolve_config(
        experiment_roots(PROJECT_ROOT, "ntu_rgbd"), experiment,
        PROJECT_ROOT / "scripts/NTU_RGBD/configs/task.yaml",
        PROJECT_ROOT / "scripts/NTU_RGBD/configs/training.yaml",
    )
    config["model"]["backbone"] = {
        "channels": [4, 8, 8, 8], "depths": [1, 1, 1, 1],
        "blocks": ["conv", "conv", "conv", "conv"],
        "output_stages": [1, 2, 3, 4], "attention_stages": [],
    }
    config["model"]["neck"]["out_channels"] = 8
    config["model"]["head"]["hidden_channels"] = 8
    validate_config(config)
    return config


def test_t_ssnn_manifest_and_commands():
    assert [item.experiment for item in EXPERIMENTS] == [
        "t_ssnn1", "t_ssnn2", "t_ssnn3",
    ]
    for item in EXPERIMENTS:
        command = build_command(item.experiment)
        assert command[command.index("--epochs") + 1] == "20"
        assert command[command.index("--category") + 1] == "t_ssnn"

    assert "--resume" in build_command("t_ssnn3", resume=True)


def test_t_ssnn_models_and_early_classifiers_forward():
    for experiment in ("t_ssnn1", "t_ssnn2"):
        model = build_model(tiny(experiment)).eval()
        with torch.no_grad():
            output = model(torch.randn(2, 4, 3, 64, 64))
        assert tuple(output.shape) == (2, 25, 64, 64)

    model = build_model(tiny("t_ssnn3")).eval()
    assert max(float(head.weight.std().detach())
               for head in model.early_classifiers.values()) < 0.01
    with torch.no_grad():
        output, auxiliaries = model(
            torch.randn(2, 4, 3, 64, 64), return_early_classifiers=True,
        )
    assert tuple(output.shape) == (2, 25, 64, 64)
    assert [tuple(item.shape) for item in auxiliaries] == [
        (2, 25, 64, 64), (2, 25, 64, 64),
    ]
