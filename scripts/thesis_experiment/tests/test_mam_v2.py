from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.models import build_model
from spikepose_thesis.models.mam_v2 import (
    MAMV2Config,
    MotionAlignedMembraneV2,
)
from spikepose_thesis.training.checkpoint import load_spatial_model
from spikepose_thesis.training.runner import (
    _configure_trainable,
    _scheduled_mam_weights,
)


def _cell(mode: str = "mam_v2", **overrides) -> MotionAlignedMembraneV2:
    return MotionAlignedMembraneV2(
        2,
        MAMV2Config(mode=mode, **overrides),
    )


def test_mam_v2_is_exact_spatial_baseline_at_initialization() -> None:
    cell = _cell()
    heatmaps = torch.randn(4, 2, 2, 8, 8)
    result = cell.forward_sequence(heatmaps)
    torch.testing.assert_close(result.heatmap, heatmaps, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        result.gamma, torch.zeros_like(result.gamma), rtol=0.0, atol=0.0,
    )
    torch.testing.assert_close(
        result.residual_offset,
        torch.zeros_like(result.residual_offset),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("align_corners", [False, True])
def test_mam_v2_zero_offset_warp_is_identity(align_corners: bool) -> None:
    cell = _cell(align_corners=align_corners)
    value = torch.randn(1, 2, 7, 9)
    offset = torch.zeros(1, 2, 2)
    torch.testing.assert_close(cell._warp(value, offset), value, atol=1e-6, rtol=1e-6)


def test_mam_v2_positive_x_offset_moves_peak_right() -> None:
    cell = _cell(align_corners=False, padding_mode="zeros")
    value = torch.zeros(1, 2, 7, 7)
    value[:, :, 3, 2] = 1.0
    offset = torch.zeros(1, 2, 2)
    offset[..., 0] = 1.0
    warped = cell._warp(value, offset)
    peak = warped[0, 0].argmax()
    assert (int(peak) // 7, int(peak) % 7) == (3, 3)


def test_mam_v2_is_causal() -> None:
    cell = _cell(residual_gamma_init=0.25)
    original = torch.randn(5, 1, 2, 8, 8)
    changed = original.clone()
    changed[3:] = torch.randn_like(changed[3:])
    left = cell.forward_sequence(original).heatmap
    right = cell.forward_sequence(changed).heatmap
    torch.testing.assert_close(left[:3], right[:3])


def test_mam_v2_bptt_modes_preserve_forward_values() -> None:
    full = _cell(residual_gamma_init=0.2, bptt_mode="full16")
    detach = _cell(residual_gamma_init=0.2, bptt_mode="detach1")
    detach.load_state_dict(full.state_dict())
    heatmaps = torch.randn(5, 1, 2, 8, 8)
    torch.testing.assert_close(
        full.forward_sequence(heatmaps).heatmap,
        detach.forward_sequence(heatmaps).heatmap,
    )


def test_mam_v2_noalign_never_warps_history() -> None:
    cell = _cell(
        "mam_v2_noalign", residual_gamma_init=0.2,
        use_residual_offset=False, use_dynamic_gate=True,
    )
    result = cell.forward_sequence(torch.randn(4, 2, 2, 8, 8))
    torch.testing.assert_close(
        result.final_offset, torch.zeros_like(result.final_offset),
        rtol=0.0, atol=0.0,
    )
    torch.testing.assert_close(result.aligned_state[1:], result.state[:-1])


def test_no_motion_token_replaces_learned_features_with_neutral_input() -> None:
    cell = _cell(use_motion_token=False, use_residual_offset=False)
    _, _, coarse, features = cell._sequence_motion_features(
        torch.randn(4, 2, 2, 8, 8),
    )
    assert torch.count_nonzero(coarse[1:]) > 0
    torch.testing.assert_close(features, torch.zeros_like(features))


def test_no_memory_update_removes_dependencies_beyond_one_frame() -> None:
    cell = _cell(
        residual_gamma_init=0.5,
        use_residual_offset=False,
        use_memory_update=False,
    )
    original = torch.randn(3, 1, 2, 8, 8)
    changed = original.clone()
    changed[0] = torch.randn_like(changed[0])
    left = cell.forward_sequence(original).heatmap
    right = cell.forward_sequence(changed).heatmap
    torch.testing.assert_close(left[2], right[2])


def test_mam_v2_shared_dynamic_decay_is_identical_across_joints() -> None:
    cell = _cell(
        "mam_v2", residual_gamma_init=0.2,
        use_dynamic_gate=True, shared_decay=True,
    )
    with torch.no_grad():
        cell.gate_head[-1].weight.normal_()
    result = cell.forward_sequence(torch.randn(4, 2, 2, 8, 8))
    torch.testing.assert_close(
        result.decay[1:, :, :1], result.decay[1:, :, 1:],
    )


def test_mam_v2_oracle_is_diagnostic_eval_only() -> None:
    cell = _cell("gt_align_oracle", residual_gamma_init=0.1)
    heatmaps = torch.randn(3, 1, 2, 8, 8)
    keypoints = torch.randn(3, 1, 2, 2)
    with pytest.raises(RuntimeError, match="eval-only"):
        cell.forward_sequence(
            heatmaps, gt_keypoints_heatmap=keypoints, diagnostic=True,
        )
    cell.eval()
    with pytest.raises(RuntimeError, match="diagnostic=True"):
        cell.forward_sequence(heatmaps, gt_keypoints_heatmap=keypoints)
    output = cell.forward_sequence(
        heatmaps, gt_keypoints_heatmap=keypoints, diagnostic=True,
    )
    assert output.heatmap.shape == heatmaps.shape


def test_mam_v2_auxiliary_contains_legacy_loss_aliases() -> None:
    cell = _cell("mam_v2_predictive")
    output = cell.forward_sequence(torch.randn(3, 1, 2, 8, 8))
    auxiliary = output.auxiliary()
    assert auxiliary["offset"] is auxiliary["final_offset"]
    assert auxiliary["prediction_next"].shape == output.heatmap.shape
    cell.eval()
    eval_output = cell.forward_sequence(torch.randn(3, 1, 2, 8, 8))
    assert eval_output.predictive_heatmap is None


def test_zero_gamma_is_baseline_preserving_but_trainable() -> None:
    cell = _cell()
    heatmaps = torch.randn(4, 1, 2, 8, 8, requires_grad=True)
    output = cell.forward_sequence(heatmaps).heatmap
    output.square().mean().backward()
    assert cell.gamma_parameter.grad is not None
    assert torch.isfinite(cell.gamma_parameter.grad).all()


def test_paper_mam_v2_config_resolves_and_preserves_zero_init() -> None:
    config = load_experiment("mamv2_fullcs20")
    temporal = config["model"]["temporal"]
    assert temporal["memory_kind"] == "mam_v2"
    assert temporal["memory_decay"] == pytest.approx(0.08)
    assert config["initialization"]["load"] == "spatial_weights_only"
    assert config["initialization"]["source"] == "mamv2_fullcs_p00_source"
    assert config["training"]["phases"][0]["train_modules"] == [
        "mam.offset_head", "mam.joint_embedding",
    ]
    model = build_model(config)
    torch.testing.assert_close(
        model.mam.gamma_parameter,
        torch.zeros_like(model.mam.gamma_parameter),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        model.mam.offset_head[-1].weight,
        torch.zeros_like(model.mam.offset_head[-1].weight),
        rtol=0.0,
        atol=0.0,
    )


def test_spatial_checkpoint_load_is_strict_and_leaves_mam_fresh(tmp_path) -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Linear(2, 2)
            self.head = nn.Linear(2, 1)
            self.mam = nn.Linear(2, 2)

    source = TinyModel()
    target = TinyModel()
    with torch.no_grad():
        source.backbone.weight.fill_(3.0)
        source.head.weight.fill_(4.0)
        source.mam.weight.fill_(5.0)
        target.mam.weight.fill_(7.0)
    checkpoint = tmp_path / "source.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint)
    _, report = load_spatial_model(checkpoint, target, "cpu")
    torch.testing.assert_close(target.backbone.weight, source.backbone.weight)
    torch.testing.assert_close(target.head.weight, source.head.weight)
    torch.testing.assert_close(
        target.mam.weight, torch.full_like(target.mam.weight, 7.0),
    )
    assert report["ignored_source_temporal_keys"] == 2


def test_mam_v2_phase_selectors_unfreeze_only_requested_stage() -> None:
    config = load_experiment("mamv2_fullcs20")
    model = build_model(config)
    phase = config["training"]["phases"][-1]
    optimizer = _configure_trainable(model, config["training"], phase)
    assert optimizer.param_groups
    assert any(parameter.requires_grad for parameter in model.backbone.stages[3].parameters())
    assert any(
        parameter.requires_grad for parameter in model.backbone.downsamples[3].parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.backbone.stages[2].parameters()
    )
    assert model.mam.gamma_parameter.requires_grad
    assert not any(parameter.requires_grad for parameter in model.mam.predictor.parameters())


def test_mam_v2_loss_schedule_is_epoch_aware() -> None:
    training = {
        "loss_schedule": {
            "offset": {"start_epoch": 1, "weight": 0.01},
            "prediction": {"start_epoch": 11, "weight": 0.005},
            "velocity": {"start_epoch": 16, "weight": 0.005},
            "acceleration": {"start_epoch": 21, "weight": 0.001},
        },
    }
    assert _scheduled_mam_weights(training, 10) == {
        "offset": 0.01, "prediction": 0.0, "velocity": 0.0,
        "acceleration": 0.0, "magnitude": 0.0,
    }
    weights = _scheduled_mam_weights(training, 21)
    assert weights["prediction"] == pytest.approx(0.005)
    assert weights["velocity"] == pytest.approx(0.005)
    assert weights["acceleration"] == pytest.approx(0.001)


def test_mam_v2_eval_vectorizes_spatial_frames_without_changing_values() -> None:
    model = build_model(load_experiment("mamv2_fullcs20")).eval()

    def cheap_spatial(sequence: torch.Tensor) -> torch.Tensor:
        value = sequence[0].mean(dim=1, keepdim=True)
        return value.repeat(1, 16, 1, 1)

    model._forward_sequence = cheap_spatial
    image = torch.randn(2, 5, 3, 8, 8)
    vectorized = model.forward_spatial_per_step(image)
    reference = torch.stack([
        cheap_spatial(frame.unsqueeze(0)) for frame in image.unbind(dim=1)
    ])
    torch.testing.assert_close(vectorized, reference)


def test_mam_v2_pilot_source_is_seed_matched_reset_transfer() -> None:
    config = load_experiment("mamv2_fullcs_p00_source")
    assert config["run_type"] == "pilot"
    assert config["training"]["seeds"] == [42]
    assert config["initialization"] == {
        "mode": "checkpoint",
        "source": "pilot20_m_s3_u2",
        "match_seed": True,
        "load": "spatial_weights_only",
    }
    assert config["model"]["temporal"]["memory_kind"] == "reset"
