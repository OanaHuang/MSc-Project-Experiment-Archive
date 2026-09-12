from pathlib import Path

import numpy as np
import torch

from scripts.MPII.evaluation.metrics import (
    _flip_back_prediction, prediction_to_keypoints,
)
from scripts.spikepose.experiments import (
    ablation_roots, experiment_roots, load_ablation, resolve_config, validate_config,
)
from scripts.spikepose.models import build_model
from scripts.spikepose.training import build_pose_loss


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "scripts/MPII/configs/task.yaml"
TRAINING = ROOT / "scripts/MPII/configs/training.yaml"


def resolve(name: str) -> dict:
    return resolve_config(experiment_roots(ROOT, "mpii"), name, TASK, TRAINING)


def test_coordinate_series_are_e9_head_only_children():
    baseline = resolve("resformer_fpn")["model"]
    cc = load_ablation(ablation_roots(ROOT, "mpii"), "coordinate_classification")
    rg = load_ablation(ablation_roots(ROOT, "mpii"), "coordinate_regression")
    assert cc["baseline"]["experiment"] == "resformer_fpn"
    assert rg["baseline"]["experiment"] == "resformer_fpn"
    assert cc["experiments"] == ["cc1", "cc2", "cc3"]
    assert rg["experiments"] == ["rg1", "rg2", "rg3"]
    for experiment in (*cc["experiments"], *rg["experiments"]):
        config = resolve(experiment)
        validate_config(config)
        model = config["model"]
        assert model["backbone"] == baseline["backbone"]
        assert model["neck"] == baseline["neck"]
        assert model["num_steps"] == baseline["num_steps"]


def test_coordinate_heads_forward_loss_decode_and_flip():
    for experiment in ("cc1", "cc2", "cc3", "rg1", "rg2", "rg3"):
        config = resolve(experiment)
        model = build_model(config)
        images = torch.randn(2, 3, 256, 256)
        target = torch.rand(2, 16, 2) * 255.0
        visibility = torch.ones(2, 16)
        prediction = model(images)
        loss = build_pose_loss(config)(prediction, target, visibility)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.head.parameters())
        coordinates, confidence = prediction_to_keypoints(prediction, 256)
        assert coordinates.shape == (2, 16, 2)
        assert confidence.shape == (2, 16)
        assert np.isfinite(coordinates).all()
        restored = _flip_back_prediction(prediction)
        restored_coordinates, _ = prediction_to_keypoints(restored, 256)
        assert restored_coordinates.shape == coordinates.shape
