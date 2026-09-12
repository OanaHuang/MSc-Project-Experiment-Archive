from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.NTU_RGBD.evaluation.temporal_metrics import compute_nacce
from scripts.NTU_RGBD.smoothnet import SmoothNet, load_sequences
from scripts.NTU_RGBD.train_smoothnet import masked_l1


def test_masked_l1_excludes_invisible_nan_coordinates():
    prediction = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    target = torch.tensor([[[[2.0, 1.0], [float("nan"), float("nan")]]]])
    visibility = torch.tensor([[[1.0, 0.0]]])
    loss = masked_l1(prediction, target, visibility)
    assert torch.isfinite(loss)
    assert loss.item() == 1.5


def test_smoothnet_preserves_pose_shape_and_gradients():
    model = SmoothNet(window_size=8, hidden_size=16, num_blocks=2, dropout=0.0)
    poses = torch.randn(2, 8, 25, 2, requires_grad=True)
    output = model(poses)
    assert output.shape == poses.shape
    output.mean().backward()
    assert poses.grad is not None


def test_sequence_loader_rejects_frame_gaps(tmp_path: Path):
    path = tmp_path / "sequences.npz"
    np.savez(path, pred=np.zeros((3, 1, 2)), gt=np.zeros((3, 1, 2)),
             visibility=np.ones((3, 1)), head_length=np.ones(3),
             sample_ids=np.array(["a", "a", "a"]), frame_indices=np.array([0, 1, 3]))
    with pytest.raises(ValueError, match="non-contiguous"):
        load_sequences(path)


def test_accel_and_naccel_are_reported():
    pred = np.array([[[0., 0.]], [[0., 0.]], [[2., 0.]]])
    gt = np.zeros_like(pred)
    result = compute_nacce(pred, gt, np.ones((3, 1)), np.full(3, 2.0),
                           np.array(["a"] * 3), np.arange(3), ["joint"])
    assert result["accel"] == pytest.approx(2.0)
    assert result["naccel"] == pytest.approx(1.0)
    assert result["nacce"] == pytest.approx(1.0)
