import pytest
import torch

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.models import build_model
from spikepose_thesis.training.losses import VisibleHeatmapMSE


@pytest.mark.parametrize("name", [
    "pilot20_m_s3_u1", "pilot20_m_s3_u2", "pilot20_m_ann_s3",
    "pilot20_factor_v1u1", "pilot20_factor_v1u2", "pilot20_factor_v2u2",
    "mamv2_fullcs20",
])
def test_required_model_forward(name):
    config = load_experiment(name)
    model = build_model(config).eval()
    temporal = config["model"]["temporal"]
    if temporal["input_strategy"] in {"frames", "scheduled_frames"}:
        value = torch.zeros(1, temporal["video_frames"], 3, 64, 64)
    else:
        value = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        prediction = model(value)
    assert prediction.shape == (1, 16, 64, 64)


def test_training_backward_for_final_spatial_model():
    config = load_experiment("pilot20_m_s3_u2")
    model = build_model(config)
    image = torch.randn(2, 3, 64, 64)
    target = torch.zeros(2, 16, 64, 64)
    visibility = torch.ones(2, 16)
    loss = VisibleHeatmapMSE()(model(image), target, visibility)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_mam_has_per_frame_training_output():
    config = load_experiment("mamv2_fullcs20")
    model = build_model(config).eval()
    image = torch.zeros(1, 8, 3, 64, 64)
    with torch.no_grad():
        output = model.forward_per_step(image)
    assert output.shape == (8, 1, 16, 64, 64)
