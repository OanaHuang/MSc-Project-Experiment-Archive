import numpy as np
import torch

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.data.ntu.core.joint_mapping import (
    MPII16_FLIP_PAIRS, NTU25_TO_MPII16, map_ntu25_to_mpii16,
)
from spikepose_thesis.refinement.jtr import JointwiseTemporalRefinement
from spikepose_thesis.refinement.jtr import jtr_loss
from spikepose_thesis.refinement.runner import _apply_module, _clips


def test_fixed_ntu25_to_mpii16_mapping():
    joints = np.arange(25 * 2).reshape(25, 2)
    mapped = map_ntu25_to_mpii16(joints)
    assert mapped.shape == (16, 2)
    assert np.array_equal(mapped, joints[NTU25_TO_MPII16])
    assert MPII16_FLIP_PAIRS[0] == (0, 5)


def test_paper_factor_controls_are_explicit():
    repeated = load_experiment("pilot20_factor_v1u2")
    history = load_experiment("pilot20_factor_v2u2")
    assert repeated["temporal"]["video_frames"] == 1
    assert history["temporal"]["video_frames"] == 2
    assert repeated["temporal"]["snn_updates_per_frame"] == 2
    assert history["temporal"]["snn_updates_per_frame"] == 1
    assert history["model"]["temporal"]["aggregation"] == "last"


def test_jtr_is_causal_and_alpha_is_bounded():
    module = JointwiseTemporalRefinement(16)
    first = torch.randn(1, 8, 16, 2)
    second = first.clone()
    second[:, 6:] += 100
    left = module(first)
    right = module(second)
    assert torch.allclose(left[:, :6], right[:, :6])
    assert torch.all((module.alpha >= 0.05) & (module.alpha <= 0.95))


def _refinement_values() -> dict[str, np.ndarray]:
    prediction = np.arange(4 * 16 * 2, dtype=np.float32).reshape(4, 16, 2)
    target = prediction + 1.0
    target[2, 3] = np.nan
    visibility = np.ones((4, 16), dtype=np.float32)
    visibility[2, 3] = 0.0
    return {
        "prediction": prediction,
        "target": target,
        "visibility": visibility,
        # Metric scales are deliberately invalid: the refiner must not use
        # target-derived evaluation scales as model inputs.
        "scale": np.full(4, np.nan, dtype=np.float32),
        "scale_hb": np.full(4, np.nan, dtype=np.float32),
        "sample_id": np.asarray(["sample"] * 4),
        "person_id": np.asarray(["person"] * 4),
        "frame_index": np.arange(4),
    }


def test_jtr_clips_mask_invalid_targets_without_using_metric_scale():
    prediction, target, visibility = _clips(
        _refinement_values(), length=4, stride=4, coordinate_scale=1920.0,
    )
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(target).all()
    assert visibility[0, 2, 3] == 0


def test_jtr_application_is_finite_when_metric_scale_is_nan():
    values = _refinement_values()
    module = JointwiseTemporalRefinement(16)
    refined = _apply_module(module, values, torch.device("cpu"), 1920.0)
    assert np.isfinite(refined["prediction"]).all()
    np.testing.assert_allclose(
        refined["prediction"][1],
        0.5 * values["prediction"][0] + 0.5 * values["prediction"][1],
        rtol=1e-6, atol=1e-5,
    )


def test_jtr_loss_ignores_nonfinite_invisible_targets():
    prediction = torch.zeros(1, 4, 16, 2, requires_grad=True)
    target = torch.ones_like(prediction)
    visibility = torch.ones(1, 4, 16)
    target.data[0, 2, 3] = float("nan")
    visibility[0, 2, 3] = 0
    loss = jtr_loss(prediction, target, visibility)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
