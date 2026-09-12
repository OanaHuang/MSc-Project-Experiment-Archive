from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.MPII.core import MPII_FLIP_PAIRS
from scripts.MPII.core.geometry import heatmaps_to_keypoints
from scripts.MPII.evaluation.dual_pckh import compute_dual_pckh, save_dual_pckh_reports


def _flip_back_heatmaps(heatmaps: torch.Tensor, shift: bool = True) -> torch.Tensor:
    restored = torch.flip(heatmaps, dims=(-1,)).clone()
    for left, right in MPII_FLIP_PAIRS:
        restored[:, [left, right]] = restored[:, [right, left]].clone()
    if shift:
        shifted = restored.clone()
        shifted[..., 1:] = restored[..., :-1]
        restored = shifted
    return restored


def _swap_flip_pairs(value: torch.Tensor) -> torch.Tensor:
    restored = value.clone()
    for left, right in MPII_FLIP_PAIRS:
        restored[:, [left, right]] = restored[:, [right, left]].clone()
    return restored


def _flip_back_prediction(prediction, shift: bool = True):
    if torch.is_tensor(prediction):
        return _flip_back_heatmaps(prediction, shift=shift)
    output_type = prediction["output_type"]
    if output_type == "coordinate_classification":
        return {
            **prediction,
            "x_logits": _swap_flip_pairs(torch.flip(prediction["x_logits"], (-1,))),
            "y_logits": _swap_flip_pairs(prediction["y_logits"]),
        }
    if output_type == "coordinate_regression":
        coordinates = prediction["coordinates"].clone()
        coordinates[..., 0] = 1.0 - coordinates[..., 0]
        return {**prediction, "coordinates": _swap_flip_pairs(coordinates)}
    raise ValueError(f"Unsupported output type: {output_type}")


def _average_predictions(left, right):
    if torch.is_tensor(left):
        return 0.5 * (left + right)
    if left["output_type"] == "coordinate_classification":
        return {
            **left,
            "x_logits": 0.5 * (left["x_logits"] + right["x_logits"]),
            "y_logits": 0.5 * (left["y_logits"] + right["y_logits"]),
        }
    return {**left, "coordinates": 0.5 * (
        left["coordinates"] + right["coordinates"]
    )}


def prediction_to_keypoints(prediction, image_size: int, decoder: str = "argmax",
                            udp: bool = False, dark_kernel: int = 11):
    """Decode every supported head into crop-space coordinates and confidence."""
    if torch.is_tensor(prediction):
        arrays = prediction.detach().cpu().numpy()
        decoded = [heatmaps_to_keypoints(
            item, image_size, method=decoder, udp=udp, dark_kernel=dark_kernel,
        ) for item in arrays]
        return (
            np.stack([item[0] for item in decoded]),
            np.stack([item[1] for item in decoded]),
        )
    if prediction["output_type"] == "coordinate_classification":
        x_probability = prediction["x_logits"].softmax(-1)
        y_probability = prediction["y_logits"].softmax(-1)
        ratio = float(prediction["split_ratio"])
        x_confidence, x_index = x_probability.max(-1)
        y_confidence, y_index = y_probability.max(-1)
        coordinates = torch.stack((x_index, y_index), -1).to(torch.float32) / ratio
        confidence = torch.sqrt(x_confidence * y_confidence)
    elif prediction["output_type"] == "coordinate_regression":
        coordinates = prediction["coordinates"] * float(image_size - 1)
        confidence = torch.ones_like(coordinates[..., 0])
    else:
        raise ValueError(f"Unsupported output type: {prediction['output_type']}")
    return coordinates.detach().cpu().numpy(), confidence.detach().cpu().numpy()


@torch.no_grad()
def evaluate(model, loader, dataset, device, output_dir: Path,
             flip_test: bool = False, flip_shift: bool = True,
             decoder: str = "argmax", udp: bool = False,
             dark_kernel: int = 11) -> dict:
    result = {key: [] for key in (
        "pred", "gt", "visibility", "head_length",
    )}
    output_type = None
    model.eval()
    for batch in loader:
        images = batch["image"].to(device)
        predictions = model(images)
        current_type = "heatmap" if torch.is_tensor(predictions) else predictions["output_type"]
        if output_type is None:
            output_type = current_type
        elif output_type != current_type:
            raise RuntimeError("Model output type changed between validation batches")
        if flip_test:
            flipped = model(torch.flip(images, dims=(-1,)))
            predictions = _average_predictions(
                predictions, _flip_back_prediction(flipped, shift=flip_shift),
            )
        decoded, _ = prediction_to_keypoints(
            predictions, dataset.image_size, decoder, udp, dark_kernel,
        )
        for index, prediction in enumerate(decoded):
            inverse = batch["inverse"][index].numpy()
            prediction[:, 0] = prediction[:, 0] * inverse[0] + inverse[2]
            prediction[:, 1] = prediction[:, 1] * inverse[1] + inverse[3]
            result["pred"].append(prediction)
            result["gt"].append(batch["keypoints_original"][index].numpy())
            result["visibility"].append(batch["visibility"][index].numpy())
            result["head_length"].append(float(batch["head_length"][index]))
    arrays = {key: np.asarray(value) for key, value in result.items()}
    pckhn_scale = 0.75 * np.linalg.norm(
        arrays["gt"][:, 9] - arrays["gt"][:, 8], axis=1,
    )
    head_pair_valid = ((arrays["visibility"][:, 8] > 0)
                       & (arrays["visibility"][:, 9] > 0))
    pckhn_scale[~head_pair_valid] = np.nan
    matrices = compute_dual_pckh(
        arrays["pred"], arrays["gt"], arrays["visibility"],
        arrays["head_length"], pckhn_scale, 0.5,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = save_dual_pckh_reports(
        output_dir, matrices, getattr(model, "model_name", "SpikePose"),
        len(arrays["pred"]),
    )
    summary["experiment_id"] = getattr(model, "experiment_id", "spikepose")
    summary["model_name"] = getattr(model, "model_name", "SpikePose")
    summary["flip_test"] = bool(flip_test)
    summary["output_type"] = output_type
    summary["flip_shift"] = bool(flip_test and flip_shift and output_type == "heatmap")
    summary["decoder"] = (
        decoder if output_type == "heatmap"
        else "axis_argmax" if output_type == "coordinate_classification"
        else "direct"
    )
    summary["udp"] = bool(udp and output_type == "heatmap")
    summary["dark_kernel"] = int(dark_kernel) if output_type == "heatmap" else None
    summary["custom_definition"] = (
        "0.75 * distance(head_top[9], upper_neck[8]); equivalent to "
        "distance(HeadCenter, NeckCenter) at r=0.25"
    )
    serialized_summary = json.dumps(summary, indent=2)
    (output_dir / "best.json").write_text(serialized_summary, encoding="utf-8")
    (output_dir / "dual_pckh_summary.json").write_text(
        serialized_summary, encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "prediction_matrices.npz", **arrays,
        pckhn_head_scale=pckhn_scale, **matrices,
    )
    return summary
