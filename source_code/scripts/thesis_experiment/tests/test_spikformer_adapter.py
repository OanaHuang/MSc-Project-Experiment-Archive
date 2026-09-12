from copy import deepcopy

import pytest
import torch

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.models import build_model
from spikepose_thesis.models.heads.linear import LinearHeatmapHead


def test_spikformer_t1_forward_backward_and_parameter_scale():
    torch.set_num_threads(2)
    model = build_model(load_experiment("mpii_official_spikformer"))
    assert isinstance(model.head, LinearHeatmapHead)
    assert 15_000_000 < sum(p.numel() for p in model.parameters()) < 18_000_000
    image = torch.randn(1, 3, 64, 64)
    features = model.forward_features(image)
    assert features.shape == (1, 1, 384, 4, 4)
    output = model(image)
    assert output.shape == (1, 16, 64, 64)
    output.mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_spikformer_t1_frame_reset_and_video_readout():
    torch.set_num_threads(2)
    model = build_model(load_experiment("mpii_official_spikformer")).eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        original = model(image)
        model(torch.randn_like(image))
        torch.testing.assert_close(model(image), original, rtol=0, atol=0)
        frames = torch.stack([image, image], dim=1)
        per_frame = model.forward_per_step(frames)
        assert per_frame.shape == (2, 1, 16, 64, 64)
        torch.testing.assert_close(per_frame[0], original, rtol=0, atol=0)
        torch.testing.assert_close(model(frames), per_frame[-1], rtol=0, atol=0)


def test_spikformer_mpii_to_ntu_configuration_is_strictly_compatible():
    source_config = load_experiment("mpii_official_spikformer")
    target_config = load_experiment("pilot20_ntu_spikformer")
    assert source_config["training"]["seeds"] == [42]
    assert target_config["training"]["seeds"] == [42]
    assert source_config["model"]["temporal"]["snn_steps_per_frame"] == 1
    assert target_config["model"]["temporal"]["snn_steps_per_frame"] == 1
    assert target_config["model"]["temporal"]["video_frames"] == 16
    assert target_config["model"]["temporal"]["state_mode"] == "reset"
    assert target_config["initialization"] == {
        "mode": "checkpoint",
        "source": "mpii_official_spikformer",
        "source_output_root": "Outputs_Thesis",
        "source_run_type": "formal",
        "match_seed": True,
        "load": "spatial_weights_only",
    }
    source = build_model(source_config)
    target = build_model(target_config)
    assert {
        name: tuple(value.shape) for name, value in source.state_dict().items()
    } == {
        name: tuple(value.shape) for name, value in target.state_dict().items()
    }


def test_spikformer_rejects_non_t1_variant():
    config = deepcopy(load_experiment("mpii_official_spikformer"))
    config["model"]["backbone"]["snn_steps"] = 4
    with pytest.raises(ValueError, match="T=1"):
        build_model(config)
