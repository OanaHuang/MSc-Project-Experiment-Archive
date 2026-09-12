from copy import deepcopy
import json

import pytest
import torch

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.models import build_model
from spikepose_thesis.models.heads.linear import LinearHeatmapHead
from spikepose_thesis.training.checkpoint import load_model
from spikepose_thesis.artifacts import run_dir
from spikepose_thesis.experiments import experiment_readiness


def test_spikeyolo_forward_backward_and_checkpoint(tmp_path):
    torch.set_num_threads(2)
    config = load_experiment("mpii_official_spikeyolo")
    model = build_model(config)
    assert isinstance(model.head, LinearHeatmapHead)
    image = torch.randn(2, 3, 64, 64)
    output = model(image)
    assert output.shape == (2, 16, 64, 64)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(image)
    path = tmp_path / "weights.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    restored = build_model(config).eval()
    load_model(path, restored, "cpu")
    with torch.no_grad():
        torch.testing.assert_close(restored(image), expected, rtol=0, atol=0)


def test_spikeyolo_frame_reset_and_video_readout():
    torch.set_num_threads(2)
    model = build_model(load_experiment("mpii_official_spikeyolo")).eval()
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
        assert model.forward_features(torch.randn(1, 3, 256, 256)).shape == (1, 1, 128, 32, 32)


def test_spikeyolo_rejects_unimplemented_variants():
    config = deepcopy(load_experiment("mpii_official_spikeyolo"))
    config["model"]["num_steps"] = 2
    with pytest.raises(ValueError, match="T=1"):
        build_model(config)


@pytest.mark.parametrize("experiment,source", [
    ("pilot20_ntu_spikepose_ann", "mpii_official_spikepose_ann"),
    ("pilot20_ntu_spikeyolo", "mpii_official_spikeyolo"),
])
def test_ntu_baseline_protocol_and_source(experiment, source):
    config = load_experiment(experiment)
    assert config["training"]["seeds"] == [42]
    assert config["training"]["epochs"] == 20
    assert config["training"]["loss_all_frames"]
    assert config["initialization"]["source"] == source
    assert config["initialization"]["source_output_root"] == "Outputs_Thesis"
    assert config["model"]["temporal"]["video_frames"] == 16
    assert config["model"]["temporal"]["state_mode"] == "reset"
    model = build_model(config).eval()
    with torch.no_grad():
        result = model.forward_per_step(torch.randn(1, 2, 3, 64, 64))
    assert result.shape == (2, 1, 16, 64, 64)


def test_ann_initialization_architecture_unchanged():
    source = build_model(load_experiment("mpii_official_spikepose_ann"))
    target = build_model(load_experiment("pilot20_ntu_spikepose_ann"))
    target.load_state_dict(source.state_dict(), strict=True)


def test_explicit_cross_root_formal_source(tmp_path):
    formal_root = tmp_path / "formal"
    source_config = load_experiment("mpii_official_spikepose_ann")
    source = run_dir(source_config, 42, formal_root)
    (source / "checkpoints").mkdir(parents=True)
    (source / "checkpoints/best.pt").touch()
    (source / "status.json").write_text(json.dumps({"state": "completed", "run_type": "formal"}))
    target = {
        "id": "transfer", "paper_id": "Transfer", "dataset": "mpii",
        "stage": "transfer", "run_type": "pilot", "training": {"epochs": 20},
        "initialization": {
            "mode": "checkpoint", "source": source_config["id"],
            "source_output_root": str(formal_root), "source_run_type": "formal",
        },
    }
    report = {"checks": {"mpii_images": True, "mpii_train_metadata": True,
                          "mpii_validation_metadata": True}}
    assert experiment_readiness(target, 42, tmp_path / "pilot", report)["status"] == "READY"
    del target["initialization"]["source_run_type"]
    assert experiment_readiness(target, 42, tmp_path / "pilot", report)["status"] == "WAITING"
