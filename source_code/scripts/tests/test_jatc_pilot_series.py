from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts.NTU_RGBD.jatc_pilot import (
    JointAdaptiveCoordinateLIF, causal_ema, dynamic_decays, load_pilot_sequences,
)
from scripts.NTU_RGBD.run_jatc_series import EXPERIMENTS, build_command


def matrices(path: Path, confidence: bool = True) -> None:
    values = {
        "pred": np.arange(48, dtype=np.float32).reshape(6, 4, 2),
        "gt": np.arange(48, dtype=np.float32).reshape(6, 4, 2),
        "visibility": np.ones((6, 4), dtype=np.float32),
        "head_length": np.full(6, 10, dtype=np.float32),
        "sample_ids": np.asarray(["a"] * 3 + ["b"] * 3),
        "frame_indices": np.asarray([0, 1, 2, 0, 1, 2]),
    }
    if confidence:
        values["confidence"] = np.full((6, 4), 0.75, dtype=np.float32)
    np.savez(path, **values)


def test_manifest_has_exactly_seven_gated_runs():
    assert [item.run_id for item in EXPERIMENTS] == [
        "jp0", "jp1", "jp2", "jp3", "jp4", "jp4_s2", "jp4_s3",
    ]
    assert [item.seed for item in EXPERIMENTS[-3:]] == [42, 3407, 2026]
    command = build_command("jp4_s2", python_executable="python", device="cuda:1")
    assert command[command.index("--seed") + 1] == "3407"
    assert command[command.index("--device") + 1] == "cuda:1"
    assert "--jp2-beta" in command


def test_sequence_loader_and_causal_ema_reset_between_videos(tmp_path: Path):
    path = tmp_path / "sequences.npz"
    matrices(path)
    sequences = load_pilot_sequences(path, require_confidence=True)
    filtered = causal_ema(sequences.pred, sequences.groups, 0.5)
    assert np.array_equal(filtered[0], sequences.pred[0])
    assert np.array_equal(filtered[3], sequences.pred[3])
    assert np.allclose(filtered[1], 0.5 * sequences.pred[0] + 0.5 * sequences.pred[1])


def test_dynamic_decay_and_learned_refiner_shapes(tmp_path: Path):
    path = tmp_path / "sequences.npz"
    matrices(path)
    sequences = load_pilot_sequences(path, require_confidence=True)
    decay = dynamic_decays(sequences, np.full(4, 0.8, dtype=np.float32), 0.1, 0.1)
    assert decay.shape == (6, 4)
    assert np.all((decay >= 0) & (decay <= 0.98))
    model = JointAdaptiveCoordinateLIF(4, np.full(4, 0.8, dtype=np.float32))
    prediction, beta = model(
        torch.randn(2, 5, 4, 2), torch.rand(2, 5, 4), torch.full((2, 5), 10.0),
    )
    assert prediction.shape == (2, 5, 4, 2)
    assert beta.shape == (2, 4, 4)
    assert torch.all((beta >= 0.5) & (beta <= 0.98))


def test_dynamic_decay_does_not_propagate_invalid_head_scale(tmp_path: Path):
    path = tmp_path / "invalid_head.npz"
    matrices(path)
    loaded = dict(np.load(path))
    loaded["head_length"] = np.asarray([10.0, np.nan, 0.0, np.nan, np.nan, np.nan],
                                       dtype=np.float32)
    np.savez(path, **loaded)
    sequences = load_pilot_sequences(path, require_confidence=True)
    base = np.full(4, 0.6, dtype=np.float32)
    decay = dynamic_decays(sequences, base, 0.1, 0.1)
    prediction = causal_ema(sequences.pred, sequences.groups, base, decay)
    assert np.isfinite(decay).all()
    assert np.isfinite(prediction).all()


def test_missing_confidence_is_only_allowed_for_legacy_stages(tmp_path: Path):
    path = tmp_path / "legacy.npz"
    matrices(path, confidence=False)
    loaded = load_pilot_sequences(path)
    assert np.all(loaded.confidence == 1)
    try:
        load_pilot_sequences(path, require_confidence=True)
    except ValueError as error:
        assert "re-export" in str(error)
    else:
        raise AssertionError("dynamic stages must reject missing confidence")
