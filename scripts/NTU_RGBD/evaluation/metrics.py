from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.NTU_RGBD.core.config import NTU_FLIP_PAIRS, NTU_JOINT_NAMES
from scripts.MPII.core.geometry import heatmaps_to_keypoints


def _flip_back_heatmaps(heatmaps: torch.Tensor, shift: bool = True) -> torch.Tensor:
    restored = torch.flip(heatmaps, dims=(-1,)).clone()
    for left, right in NTU_FLIP_PAIRS:
        restored[:, [left, right]] = restored[:, [right, left]].clone()
    if shift:
        shifted = restored.clone()
        shifted[..., 1:] = restored[..., :-1]
        restored = shifted
    return restored


def compute_pckhn(prediction: np.ndarray, target: np.ndarray,
                  visibility: np.ndarray, head_length: np.ndarray,
                  threshold: float = 0.5) -> dict:
    """Compute NTU PCKhn using the 2D Head-to-Neck distance."""
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    visibility = np.asarray(visibility)
    head_length = np.asarray(head_length, dtype=np.float32)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [N, J, 2]")
    if visibility.shape != prediction.shape[:2]:
        raise ValueError("visibility must have shape [N, J]")
    if head_length.shape != (prediction.shape[0],):
        raise ValueError("head_length must have shape [N]")
    scale = head_length[:, None]
    valid_mask = ((visibility > 0) & np.isfinite(scale) & (scale > 1e-6))
    distance = np.linalg.norm(prediction - target, axis=2)
    normalized_distance = distance / np.maximum(scale, 1e-6)
    correct_matrix = (normalized_distance <= float(threshold)) & valid_mask
    return {
        "normalized_distance": normalized_distance,
        "valid_mask": valid_mask,
        "correct_matrix": correct_matrix,
    }


def crop_to_original(points: np.ndarray, bbox: np.ndarray,
                     image_size: int) -> np.ndarray:
    """Map model-input crop coordinates back to original image pixels."""
    points = np.asarray(points, dtype=np.float32)
    bbox = np.asarray(bbox, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [J, 2]")
    if bbox.shape != (4,):
        raise ValueError("bbox must have shape [4]")
    x1, y1, x2, y2 = bbox
    restored = points.copy()
    restored[:, 0] = x1 + restored[:, 0] * (x2 - x1) / float(image_size)
    restored[:, 1] = y1 + restored[:, 1] * (y2 - y1) / float(image_size)
    return restored


@torch.no_grad()
def evaluate(model, loader, dataset, device, output_dir: Path,
             flip_test: bool = False, flip_shift: bool = True) -> dict:
    """Evaluate the canonical NTU metric in complete-frame pixel coordinates.

    Model-input/crop-coordinate PCKhn is intentionally not emitted. Predictions
    are always restored through each person's bounding box before scoring.
    """
    collected = {key: [] for key in (
        "pred", "confidence", "gt", "visibility", "head_length", "sample_ids",
        "frame_indices",
    )}
    model.eval()
    for batch in loader:
        images = batch["image"].to(device)
        heatmaps = model(images)
        if flip_test:
            flipped = model(torch.flip(images, dims=(-1,)))
            heatmaps = 0.5 * (
                heatmaps + _flip_back_heatmaps(flipped, shift=flip_shift)
            )
        heatmaps = heatmaps.cpu().numpy()
        for index, heatmap in enumerate(heatmaps):
            prediction, _ = heatmaps_to_keypoints(heatmap, dataset.image_size)
            peaks = np.partition(heatmap.reshape(heatmap.shape[0], -1), -2, axis=1)[:, -2:]
            # A bounded peak-margin confidence is stable across heatmap logit scales.
            confidence = 1.0 / (1.0 + np.exp(-(peaks[:, 1] - peaks[:, 0])))
            prediction = crop_to_original(
                prediction, batch["person_bbox"][index].numpy(), dataset.image_size,
            )
            target = batch["original_keypoints"][index].numpy()
            visibility = batch["original_visibility"][index].numpy()
            head_length = (
                float(np.linalg.norm(target[3] - target[2]))
                if visibility[3] > 0 and visibility[2] > 0
                else float("nan")
            )
            collected["pred"].append(prediction)
            collected["confidence"].append(confidence.astype(np.float32))
            collected["gt"].append(target)
            collected["visibility"].append(visibility)
            collected["head_length"].append(head_length)
            collected["sample_ids"].append(str(batch["sample_id"][index]))
            frame = batch["frame_index"][index]
            collected["frame_indices"].append(int(frame.item() if hasattr(frame, "item") else frame))
    arrays = {key: np.asarray(value) for key, value in collected.items()}
    matrices = compute_pckhn(
        arrays["pred"], arrays["gt"], arrays["visibility"], arrays["head_length"], 0.5,
    )
    valid_mask = matrices["valid_mask"]
    correct = matrices["correct_matrix"]
    per_joint_valid = valid_mask.sum(axis=0)
    per_joint_correct = correct.sum(axis=0)
    per_joint_pckhn = {
        name: (float(per_joint_correct[index] / per_joint_valid[index])
               if per_joint_valid[index] else None)
        for index, name in enumerate(NTU_JOINT_NAMES)
    }
    valid = int(valid_mask.sum())
    summary = {
        "experiment_id": getattr(model, "experiment_id", "spikepose"),
        "model_name": getattr(model, "model_name", "SpikePose"),
        "dataset": "ntu_rgbd",
        "num_joints": len(NTU_JOINT_NAMES),
        "metric": "pckhn",
        "normalization": "head_to_neck_distance_in_original_image_pixels",
        "coordinate_space": "complete_frame_pixels",
        "threshold": 0.5,
        "threshold_label": "PCKhn@0.5",
        "pckhn": float(correct.sum() / max(valid, 1)),
        "valid_joints": valid,
        "samples": int(len(arrays["pred"])),
        "per_joint_pckhn": per_joint_pckhn,
        "flip_test": bool(flip_test),
        "flip_shift": bool(flip_test and flip_shift),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(summary, indent=2)
    (output_dir / "best.json").write_text(serialized, encoding="utf-8")
    (output_dir / "pckhn_summary.json").write_text(serialized, encoding="utf-8")
    np.savez_compressed(
        output_dir / "prediction_matrices.npz", **arrays,
        **matrices,
    )
    return summary
