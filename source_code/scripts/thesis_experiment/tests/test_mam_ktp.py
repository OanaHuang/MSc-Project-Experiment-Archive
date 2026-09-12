from __future__ import annotations

import pytest
import torch

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.experiments import plan_study
from spikepose_thesis.models import build_model
from spikepose_thesis.models.mam_v2 import (
    KTPPriorEncoder,
    MAMV2Config,
    MotionAlignedMembraneV2,
)
from spikepose_thesis.training.runner import _configure_trainable
from tools.run_icassp2027_queue import WORKFLOWS


KTP_MATRIX = {
    "mamv2_fullcs20": (False, False),
    "pilot20_mamv2_kpa": (True, False),
    "pilot20_mamv2_tpa": (False, True),
    "pilot20_mamv2_ktp": (True, True),
}


def test_mam_ktp_study_is_a_strict_two_by_two_ablation() -> None:
    configs = plan_study("mam_ktp_pilot20")
    assert [config["id"] for config in configs] == list(KTP_MATRIX)
    assert all(config["run_type"] == "pilot" for config in configs)
    assert all(config["training"]["epochs"] == 20 for config in configs)
    for config in configs:
        temporal = config["model"]["temporal"]
        assert (
            bool(temporal.get("memory_use_kpa", False)),
            bool(temporal.get("memory_use_tpa", False)),
        ) == KTP_MATRIX[config["id"]]
        assert temporal.get("memory_ktp_causal", True) is True
        assert temporal["memory_kind"] == "mam_v2"
        assert temporal["memory_bptt_mode"] == "full16"
        assert config["temporal"]["video_frames"] == 16
        assert config["training"]["loss_schedule"]["offset"]["weight"] == pytest.approx(0.01)
        for loss in ("prediction", "velocity", "acceleration", "magnitude"):
            assert config["training"]["loss_schedule"][loss]["weight"] == 0.0


def test_mam_ktp_queue_exposes_the_same_four_conditions() -> None:
    waves = WORKFLOWS["mam-ktp-pilot20"]
    assert len(waves) == 1
    assert [job["experiment"] for job in waves[0][1]] == list(KTP_MATRIX)


@pytest.mark.parametrize(
    "name", ("pilot20_mamv2_kpa", "pilot20_mamv2_tpa", "pilot20_mamv2_ktp"),
)
def test_mam_ktp_configs_build_and_all_phase_selectors_resolve(name: str) -> None:
    config = load_experiment(name, profile="pilot20")
    assert config["initialization"] == {
        "mode": "checkpoint",
        "source": "mamv2_fullcs_p00_source",
        "match_seed": True,
        "load": "spatial_weights_only",
    }
    model = build_model(config)
    assert model.mam.motion_feature_dim == 17
    assert model.mam.ktp_prior is not None
    for phase in config["training"]["phases"]:
        optimizer = _configure_trainable(model, config["training"], phase)
        assert "mam.ktp_prior" in {
            group["name"] for group in optimizer.param_groups
        }


def test_core_mam_path_does_not_construct_a_prior_adapter() -> None:
    model = build_model(load_experiment("mamv2_fullcs20"))
    assert model.mam.motion_feature_dim == 17
    assert model.mam.ktp_prior is None


def test_ktp_prior_uses_mpii16_skeleton_and_causal_temporal_topology() -> None:
    encoder = KTPPriorEncoder(
        16, 17, 16, use_kpa=True, use_tpa=True, causal=True,
    )
    assert encoder.kpa.local_affinity[0, 1] > 0
    assert encoder.kpa.local_affinity[0, 15] == 0
    for layer in encoder.tpa:
        affinity = layer.affinity()
        torch.testing.assert_close(
            affinity.triu(diagonal=1), torch.zeros_like(affinity),
            rtol=0.0, atol=0.0,
        )


def test_ktp_prior_never_changes_a_past_token_from_future_input() -> None:
    torch.manual_seed(7)
    encoder = KTPPriorEncoder(
        4, 6, 8, use_kpa=True, use_tpa=True, causal=True,
    ).eval()
    original = torch.randn(8, 2, 4, 6)
    changed = original.clone()
    changed[5:] = torch.randn_like(changed[5:])
    left = encoder(original)
    right = encoder(changed)
    torch.testing.assert_close(left[:5], right[:5], rtol=0.0, atol=0.0)
    assert not torch.equal(left[5:], right[5:])


def test_ktp_global_affinities_receive_gradients() -> None:
    encoder = KTPPriorEncoder(
        4, 6, 8, use_kpa=True, use_tpa=True, causal=True,
    )
    value = torch.randn(8, 2, 4, 6, requires_grad=True)
    encoder(value).square().mean().backward()
    assert encoder.kpa.global_affinity.grad is not None
    assert all(layer.global_affinity.grad is not None for layer in encoder.tpa)


@pytest.mark.parametrize(
    ("use_kpa", "use_tpa"), ((True, False), (False, True), (True, True)),
)
def test_mam_with_ktp_priors_remains_exactly_baseline_preserving_at_init(
    use_kpa: bool, use_tpa: bool,
) -> None:
    cell = MotionAlignedMembraneV2(
        2,
        MAMV2Config(
            use_kpa=use_kpa, use_tpa=use_tpa, ktp_frames=5,
        ),
    )
    heatmaps = torch.randn(5, 2, 2, 8, 8)
    result = cell.forward_sequence(heatmaps)
    torch.testing.assert_close(result.heatmap, heatmaps, rtol=0.0, atol=0.0)


def test_causal_tpa_preserves_end_to_end_mam_causality_after_unfreezing_heads() -> None:
    torch.manual_seed(11)
    cell = MotionAlignedMembraneV2(
        2,
        MAMV2Config(
            residual_gamma_init=0.25,
            use_kpa=True,
            use_tpa=True,
            ktp_frames=6,
            ktp_causal=True,
        ),
    ).eval()
    with torch.no_grad():
        cell.offset_head[-1].weight.normal_()
        cell.gate_head[-1].weight.normal_()
    original = torch.randn(6, 1, 2, 8, 8)
    changed = original.clone()
    changed[4:] = torch.randn_like(changed[4:])
    left = cell.forward_sequence(original).heatmap
    right = cell.forward_sequence(changed).heatmap
    torch.testing.assert_close(left[:4], right[:4])
