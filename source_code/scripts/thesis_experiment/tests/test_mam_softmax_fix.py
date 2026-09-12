from __future__ import annotations

import numpy as np
import pytest
import torch

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.evaluation.softmax_fix import MotionMetrics, choose_candidates, synthetic_checks
from spikepose_thesis.models import build_model
from spikepose_thesis.models.mam_v2 import MAMV2Config, MotionAlignedMembraneV2
from spikepose_thesis.models.mam_v2.coordinate_decoding import CoordinateDecodedMAM, DecoderSpec, decode_coordinates
from spikepose_thesis.training.runner import _configure_trainable, _mam_auxiliary_losses


def gaussian(cx=16.25, cy=24.25):
    y, x = torch.meshgrid(torch.arange(64, dtype=torch.float64), torch.arange(64, dtype=torch.float64), indexing="ij")
    return torch.exp(-((x-cx).square() + (y-cy).square()) / 8)[None, None]


def cell(method="relu", **kwargs):
    result = CoordinateDecodedMAM(1, MAMV2Config(**kwargs))
    result.decoder_spec = DecoderSpec(method)
    return result


def test_document_synthetic_table_and_warp_sign():
    table = synthetic_checks()["document_table"]
    assert table["softmax"]["dx"] == pytest.approx(.0353, abs=5e-5)
    assert table["scaled100"]["dx"] == pytest.approx(4.9707, abs=5e-5)
    for method in ("relu", "power2", "expm1"):
        assert table[method]["dx"] == pytest.approx(4.4, abs=1e-5)
    mam = cell()
    before = gaussian(16, 24).float()
    warped = mam._warp(before, torch.tensor([[[4., 0.]]]))
    assert int(warped[0, 0].argmax()) % 64 == 20


@pytest.mark.parametrize("method,alpha", [("relu", 1), ("power", 2), ("expm1", 1)])
def test_normalizations_match_literal_formula_and_have_gradients(method, alpha):
    h = (torch.rand(1, 1, 6, 7, dtype=torch.float64) + .1).requires_grad_()
    spec = DecoderSpec(method, alpha=alpha)
    xy, _, _ = decode_coordinates(h, spec)
    w = torch.expm1(h) if method == "expm1" else h.pow(alpha)
    p = w / w.sum((-2, -1), keepdim=True)
    y, x = torch.meshgrid(torch.arange(6), torch.arange(7), indexing="ij")
    expected = torch.stack(((p*x).sum((-2,-1)), (p*y).sum((-2,-1))), -1)
    torch.testing.assert_close(xy, expected)
    assert torch.autograd.gradcheck(lambda value: decode_coordinates(value, spec)[0], (h,))


def test_power_is_scale_invariant_but_expm1_is_not_on_asymmetric_map():
    h = gaussian() + .3 * gaussian(24.25)
    for spec in (DecoderSpec("relu"), DecoderSpec("power", alpha=2)):
        torch.testing.assert_close(decode_coordinates(h, spec)[0], decode_coordinates(h * 2, spec)[0])
    assert not torch.allclose(decode_coordinates(h, DecoderSpec("expm1"))[0], decode_coordinates(h*2, DecoderSpec("expm1"))[0])


def test_expm1_stable_for_large_peaks_and_amp_outputs_fp32():
    h = gaussian().float() * 1000
    xy, _, valid = decode_coordinates(h, DecoderSpec("expm1"))
    assert torch.isfinite(xy).all() and valid.all()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = decode_coordinates(h.to(torch.bfloat16), DecoderSpec("expm1"))[0]
    assert output.dtype == torch.float32 and torch.isfinite(output).all()


def test_dark_matches_existing_decoder_and_detaches():
    from spikepose_thesis.data.mpii.core.geometry import heatmaps_to_keypoints
    h = gaussian().float().requires_grad_()
    xy, _, _ = decode_coordinates(h, DecoderSpec("dark"))
    expected, _ = heatmaps_to_keypoints(h.detach()[0].numpy(), image_size=64, method="dark")
    np.testing.assert_array_equal(xy[0].numpy(), expected)
    assert not xy.requires_grad


@pytest.mark.parametrize("method", ["softmax", "relu", "power", "expm1", "dark"])
def test_zero_negative_invalid_maps_bypass_history(method):
    mam = cell(method, residual_gamma_init=.8)
    heatmaps = torch.cat((gaussian().float(), torch.zeros(1, 1, 64, 64), gaussian(20).float()))[:, None]
    result = mam.forward_sequence(heatmaps)
    torch.testing.assert_close(result.heatmap, heatmaps)
    assert torch.count_nonzero(result.decay) == 0
    xy, q, valid = decode_coordinates(-torch.ones(1, 1, 64, 64), DecoderSpec(method))
    assert not valid.any() and not q.any()
    torch.testing.assert_close(xy, torch.tensor([[[31.5, 31.5]]]))


def test_nonfinite_heatmaps_fail_instead_of_polluting_fusion():
    with pytest.raises(FloatingPointError):
        decode_coordinates(torch.full((1, 1, 64, 64), float("nan")), DecoderSpec())


def test_adapter_preserves_parameters_raw_maps_and_causality():
    base = MotionAlignedMembraneV2(1, MAMV2Config(residual_gamma_init=.3))
    adapted = cell(residual_gamma_init=.3)
    adapted.load_state_dict(base.state_dict(), strict=True)
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in adapted.parameters())
    heatmaps = torch.cat((gaussian(), gaussian(17), gaussian(18), gaussian(19)))[:, None].float()
    original = heatmaps.clone()
    left = adapted.forward_sequence(heatmaps)
    heatmaps[2:] = gaussian(30).float()
    right = adapted.forward_sequence(heatmaps)
    torch.testing.assert_close(left.heatmap[:2], right.heatmap[:2])
    torch.testing.assert_close(left.raw_heatmap[:2], original[:2])


def test_all_configs_freeze_spatial_and_legacy_builder_is_unchanged():
    legacy = build_model(load_experiment("mamv2_fullcs20"))
    assert type(legacy.mam) is MotionAlignedMembraneV2
    for label in ("softmax", "relu", "power2", "expm1", "dark"):
        for suffix in ("", "_coarse", "_noalign"):
            cfg = load_experiment("mam_softmax_fix_" + label + suffix)
            model = build_model(cfg)
            assert isinstance(model.mam, CoordinateDecodedMAM)
            assert cfg["training"]["epochs"] == 20
            for phase in cfg["training"]["phases"]:
                _configure_trainable(model, cfg["training"], phase)
                assert all(not p.requires_grad for n,p in model.named_parameters() if not n.startswith("mam."))
                assert phase["freeze_batch_norm"]


def test_offset_loss_supervises_final_not_residual():
    class Model:
        pass
    model = Model()
    final = torch.tensor([[[[0., 0.]]], [[[4., 0.]]]])
    model.last_mam_aux = {"offset": final, "residual_offset": torch.zeros_like(final)}
    prediction = torch.zeros(2, 1, 1, 64, 64)
    batch = {"temporal_keypoints": torch.tensor([[[[32., 32.]], [[48., 32.]]]])}
    losses = _mam_auxiliary_losses(model, prediction, batch, torch.device("cpu"),
        {"heatmap_size": 64, "image_size": 256}, {"mam_loss_weights": {"offset": .01}},
        target_hm=prediction, visibility=torch.ones(2, 1, 1))
    assert losses["loss_offset"] == 0


def test_motion_metrics_do_not_connect_clips_and_keep_invalid_cases():
    metric = MotionMetrics()
    target = torch.tensor([[[[1., 0.]], [[50., 0.]]], [[[2., 0.]], [[50., 0.]]]])
    valid = torch.ones(2, 2, 1, dtype=torch.bool)
    valid[1, 1] = False
    metric.update(target, valid, target, torch.ones_like(valid))
    summary = metric.summary()
    assert summary["all/zero_epe"]["count"] == 2
    assert summary["all/zero_epe"]["mean"] == .5
    assert summary["all/invalid_pair"]["mean"] == .5


def test_selection_does_not_require_beating_zero():
    summaries = {}
    for i, name in enumerate(("relu", "power2", "expm1")):
        row = {"coordinate_epe": {"mean": 1 + i}, "invalid": {"mean": 0}}
        for label in ("0_0.5", "0.5_2", "2_4", "4_plus"):
            row[label + "/coarse_epe"] = {"mean": .6 + i}
            row[label + "/reachable_epe"] = {"mean": 0}
        row["all/zero_epe"] = {"mean": .3}
        summaries[name] = row
    assert "relu" in choose_candidates(summaries)["selected"]
