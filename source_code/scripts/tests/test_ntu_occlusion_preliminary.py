from pathlib import Path

import numpy as np
import torch

from scripts.NTU_RGBD.occlusion import (
    MaskConfig, OcclusionWindowDataset, build_refiner, evaluate_occlusion,
    generate_occlusion_mask, load_sequences,
)
from scripts.NTU_RGBD.train_occlusion_preliminary import (
    evaluate as evaluate_run, load_experiment, masked_l1,
)


def test_occlusion_loss_excludes_masked_nan_coordinates():
    prediction = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    target = torch.tensor([[[[2.0, 1.0], [float("nan"), float("nan")]]]])
    mask = torch.tensor([[[1.0, 0.0]]])
    loss = masked_l1(prediction, target, mask)
    assert torch.isfinite(loss)
    assert loss.item() == 1.5


def test_run_evaluation_maps_batch_names_to_metric_arguments():
    class Identity(torch.nn.Module):
        def forward(self, pose, confidence, observed):
            return pose

    batch = {
        "pred": torch.zeros(1, 3, 2, 2),
        "gt": torch.zeros(1, 3, 2, 2),
        "confidence": torch.ones(1, 3, 2),
        "visibility": torch.ones(1, 3, 2),
        "observed": torch.ones(1, 3, 2),
        "head_length": torch.ones(1, 3),
    }
    metrics = evaluate_run(Identity(), [batch], torch.device("cpu"), 1.0)
    assert metrics["overall"]["pckhn"] == 1.0


def test_preliminary_manifest_is_complete():
    assert [load_experiment(f"pv{i}")["method"] for i in range(8)] == [
        "identity", "identity", "last_observation", "kalman", "gru", "snn",
        "confidence_snn", "confidence_graph_snn",
    ]


def test_mask_is_deterministic_and_only_removes_visibility():
    visibility = torch.ones(16, 25)
    first = generate_occlusion_mask(visibility, MaskConfig(probability=1.0), torch.Generator().manual_seed(7))
    second = generate_occlusion_mask(visibility, MaskConfig(probability=1.0), torch.Generator().manual_seed(7))
    assert torch.equal(first, second)
    assert torch.all(first <= visibility)
    assert torch.any(first == 0)


def test_all_refiners_are_causal_shape_preserving():
    pose = torch.randn(2, 16, 25, 2)
    confidence = torch.rand(2, 16, 25)
    observed = torch.ones(2, 16, 25)
    observed[:, 5:9, 6] = 0
    pose = pose * observed[..., None]
    for kind in ("identity", "last_observation", "kalman", "gru", "snn", "confidence_snn", "confidence_graph_snn"):
        output = build_refiner(kind, hidden_size=16)(pose, confidence, observed)
        assert output.shape == pose.shape
        assert torch.isfinite(output).all()


def test_sequence_dataset_retains_masked_ground_truth(tmp_path: Path):
    frames, joints = 20, 25
    path = tmp_path / "sequence.npz"
    np.savez_compressed(
        path, pred=np.ones((frames, joints, 2), np.float32),
        confidence=np.ones((frames, joints), np.float32),
        gt=np.full((frames, joints, 2), 2.0, np.float32),
        visibility=np.ones((frames, joints), np.float32),
        head_length=np.ones(frames, np.float32),
        sample_ids=np.asarray(["video"] * frames), frame_indices=np.arange(frames),
    )
    dataset = OcclusionWindowDataset(
        load_sequences(path), 16, mask_config=MaskConfig(probability=1.0),
    )
    item = dataset[0]
    masked = item["observed"] == 0
    assert torch.all(item["pred"][masked] == 0)
    assert torch.all(item["gt"][masked] == 2)


def test_occlusion_metrics_separate_visible_and_hidden():
    target = np.zeros((1, 2, 2), np.float32)
    prediction = target.copy(); prediction[:, 1, 0] = 2
    result = evaluate_occlusion(
        prediction, target, np.ones((1, 2)), np.asarray([[1, 0]]), np.ones(1),
    )
    assert result["visible"]["pckhn"] == 1.0
    assert result["occluded"]["pckhn"] == 0.0
