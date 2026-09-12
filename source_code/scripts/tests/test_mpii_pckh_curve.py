import numpy as np

from scripts.MPII.evaluation.pckh_curve import compute_pckh_curve


def test_curve_excludes_pelvis_and_thorax_and_normalizes_auc():
    distance = np.zeros((1, 16), dtype=np.float32)
    distance[:, :] = 0.25
    distance[:, 6:8] = 10.0
    valid = np.ones_like(distance, dtype=bool)
    result = compute_pckh_curve(distance, valid)
    assert result["valid_joints"] == 14
    assert result["pckh"]["0.2"] == 0.0
    assert result["pckh"]["0.3"] == 1.0
    assert np.isclose(result["auc_0_5"], 0.5, atol=0.01)
