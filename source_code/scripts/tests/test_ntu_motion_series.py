from __future__ import annotations

from pathlib import Path

import torch

from scripts.NTU_RGBD.train_motion_queue import jobs_for_phase
from scripts.NTU_RGBD.train_motion_series import build_command
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


def test_motion_configs_separate_video_and_ilif_time():
    assert resolve("m0")["model"]["num_steps"] == 1
    for experiment in ("m1", "m2", "m3"):
        temporal = resolve(experiment)["model"]["temporal"]
        assert temporal["decouple_video_time"] is True
        assert temporal["snn_steps_per_frame"] == 1
        assert temporal["state_mode"] == "reset"


def test_motion_forecast_cannot_see_current_frame():
    model = build_model(tiny("m2")).eval()
    image = torch.randn(2, 4, 3, 32, 32)
    changed = image.clone()
    changed[:, -1] = torch.randn_like(changed[:, -1]) * 10
    with torch.no_grad():
        _, original = model(image, return_intermediates=True)
        _, modified = model(changed, return_intermediates=True)
    assert torch.allclose(original[0], modified[0])
    assert not torch.allclose(original[1], modified[1])


def test_m2_and_m3_return_forecast_current_and_final_predictions():
    for experiment in ("m2", "m3"):
        model = build_model(tiny(experiment)).eval()
        with torch.no_grad():
            prediction, intermediates = model(
                torch.randn(2, 4, 3, 32, 32), return_intermediates=True,
            )
        assert len(intermediates) == 3
        assert torch.equal(prediction, intermediates[-1])
        assert tuple(prediction.shape) == (2, 25, 64, 64)


def test_motion_queue_and_commands_use_shared_population():
    assert [job.experiment for job in jobs_for_phase("screen")] == [
        "m0", "m1", "m2", "m3",
    ]
    assert len(jobs_for_phase("main")) == 12
    command = build_command("m3", epochs=20, physical_gpu=2)
    assert command[command.index("--minimum-temporal-history") + 1] == "3"
    assert command[command.index("--physical-gpu") + 1] == "2"
