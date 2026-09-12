from __future__ import annotations

from pathlib import Path

import torch

from scripts.NTU_RGBD.external_baselines import initialize_external_baseline
from scripts.NTU_RGBD.train_nhb_series import build_command
from scripts.spikepose.experiments import experiment_roots, resolve_config, validate_config
from scripts.spikepose.models import build_model
from scripts.spikepose.models.baselines import build_pose_resnet


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve(experiment: str) -> dict:
    config = resolve_config(
        experiment_roots(PROJECT_ROOT, "ntu_rgbd"), experiment,
        PROJECT_ROOT / "scripts/NTU_RGBD/configs/task.yaml",
        PROJECT_ROOT / "scripts/NTU_RGBD/configs/training.yaml", {},
    )
    validate_config(config)
    return config


def test_nhb_models_emit_ntu_heatmaps():
    image = torch.zeros(1, 3, 256, 256)
    for experiment in ("nhb1", "nhb3", "nhb5"):
        model = build_model(resolve(experiment)).eval()
        with torch.no_grad():
            output = model(image)
        assert output.shape == (1, 25, 64, 64)


def test_mpii_initialization_skips_only_changed_pose_resnet_head(tmp_path):
    source = build_pose_resnet(50, 16)
    checkpoint = tmp_path / "mpii.pth"
    torch.save(source.state_dict(), checkpoint)
    target = build_pose_resnet(50, 25)
    report = initialize_external_baseline(target, checkpoint)
    assert set(report["skipped_shape"]) == {
        "final_layer.weight", "final_layer.bias",
    }
    assert set(report["randomly_initialized_tensors"]) == {
        "final_layer.weight", "final_layer.bias",
    }
    assert report["loaded_tensor_count"] == report["source_tensor_count"] - 2


def test_nhb_launcher_matches_fp_population():
    command = build_command("nhb3", epochs=20)
    joined = " ".join(command)
    assert "--minimum-temporal-history 3" in joined
    assert "--disable-augmentation" in command
    assert "--preprocessed-pose-cache" in command
    assert "clip4_256_20ep" in command
