import torch

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.experiments import plan_study
from spikepose_thesis.models import build_model
from spikepose_thesis.training.runner import _configure_trainable
from tools import run_icassp2027_queue as queue


EXPERIMENT = "pilot20_ntu_spikepose_ann_mam"


def test_ann_mam_training_chain_is_dependency_ordered():
    configs = plan_study("spikepose_ann_mam_pilot20")
    assert [config["id"] for config in configs] == [
        "mpii_official_spikepose_ann",
        "pilot20_ntu_spikepose_ann",
        EXPERIMENT,
    ]
    assert configs[-1]["initialization"] == {
        "mode": "checkpoint",
        "source": "pilot20_ntu_spikepose_ann",
        "match_seed": True,
        "load": "spatial_weights_only",
    }


def test_ann_mam_is_ann_backbone_with_core_mam():
    config = load_experiment(EXPERIMENT)
    assert config["model_config"] == "s5_ann_a4"
    assert config["model"]["neuron"]["kind"] == "relu"
    assert config["model"]["num_steps"] == 16
    temporal = config["model"]["temporal"]
    assert temporal["memory_kind"] == "mam_v2"
    assert temporal["memory_use_kpa"] is False
    assert temporal["memory_use_tpa"] is False
    assert temporal["state_mode"] == "continuous"

    model = build_model(config)
    assert model.mam is not None
    torch.testing.assert_close(
        model.mam.gamma_parameter,
        torch.zeros_like(model.mam.gamma_parameter),
        rtol=0.0,
        atol=0.0,
    )


def test_ann_frame_checkpoint_is_spatially_shape_compatible_with_ann_mam():
    source = build_model(load_experiment("pilot20_ntu_spikepose_ann"))
    target = build_model(load_experiment(EXPERIMENT))
    source_state = source.state_dict()
    for key, value in target.state_dict().items():
        if key.startswith("mam."):
            continue
        assert key in source_state
        assert source_state[key].shape == value.shape


def test_ann_mam_partial_finetune_uses_only_final_spatial_stage():
    config = load_experiment(EXPERIMENT)
    model = build_model(config)
    phase = config["training"]["phases"][-1]
    optimizer = _configure_trainable(model, config["training"], phase)
    assert optimizer.param_groups
    assert any(parameter.requires_grad for parameter in model.backbone.stages[3].parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone.stages[2].parameters())
    assert model.mam.gamma_parameter.requires_grad


def test_ann_mam_queue_has_three_serial_dependency_waves():
    workflow = queue.WORKFLOWS["spikepose-ann-mam-pilot20"]
    assert [wave for wave, _ in workflow] == ["mpii_ann", "ntu_ann", "ann_mam"]
    assert [jobs[0]["experiment"] for _, jobs in workflow] == [
        "mpii_official_spikepose_ann",
        "pilot20_ntu_spikepose_ann",
        EXPERIMENT,
    ]
    assert workflow[0][1][0]["seeds"] == (42,)
