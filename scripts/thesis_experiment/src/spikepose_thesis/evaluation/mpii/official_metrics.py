"""Evaluation adapter for official HRNet affine crops."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from spikepose_thesis.data.mpii.core import MPII_FLIP_PAIRS
from spikepose_thesis.data.mpii.core.geometry import heatmaps_to_keypoints
from spikepose_thesis.data.mpii.core.official_transforms import transform_points
from spikepose_thesis.evaluation.mpii.dual_pckh import compute_dual_pckh, save_dual_pckh_reports


def _flip_back(heatmaps, shift=True):
    result = torch.flip(heatmaps, (-1,)).clone()
    for left, right in MPII_FLIP_PAIRS:
        result[:, [left, right]] = result[:, [right, left]].clone()
    if shift:
        shifted = result.clone()
        shifted[..., 1:] = result[..., :-1]
        result = shifted
    return result


@torch.no_grad()
def evaluate_official(model, loader, dataset, device, output_dir: Path,
                      flip_test=True, flip_shift=True):
    collected = {key: [] for key in (
        "pred", "gt", "visibility", "head_length", "official_index",
    )}
    model.eval()
    for batch in loader:
        images = batch["image"].to(device)
        heatmaps = model(images)
        if flip_test:
            flipped = _flip_back(model(torch.flip(images, (-1,))), flip_shift)
            heatmaps = 0.5 * (heatmaps + flipped)
        for index, heatmap in enumerate(heatmaps.cpu().numpy()):
            prediction, _ = heatmaps_to_keypoints(
                heatmap, dataset.image_size, method="quarter",
            )
            prediction = transform_points(
                prediction, batch["inverse_affine"][index].numpy(),
            )
            collected["pred"].append(prediction)
            collected["gt"].append(batch["keypoints_original"][index].numpy())
            collected["visibility"].append(batch["visibility"][index].numpy())
            collected["head_length"].append(float(batch["head_length"][index]))
            collected["official_index"].append(int(batch["official_index"][index]))
    arrays = {key: np.asarray(value) for key, value in collected.items()}
    custom_scale = 0.75 * np.linalg.norm(
        arrays["gt"][:, 9] - arrays["gt"][:, 8], axis=1,
    )
    custom_scale[
        (arrays["visibility"][:, 8] <= 0) | (arrays["visibility"][:, 9] <= 0)
    ] = np.nan
    matrices = compute_dual_pckh(
        arrays["pred"], arrays["gt"], arrays["visibility"],
        arrays["head_length"], custom_scale, 0.5,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = save_dual_pckh_reports(
        output_dir, matrices, model.model_name, len(arrays["pred"]),
    )
    summary.update({
        "experiment_id": model.experiment_id,
        "model_name": model.model_name,
        "split_protocol": "official_hrnet_mpii",
        "flip_test": bool(flip_test),
        "flip_shift": bool(flip_test and flip_shift),
        "decoder": "official_quarter_pixel",
        "samples": len(arrays["pred"]),
    })
    serialized = json.dumps(summary, indent=2)
    (output_dir / "best.json").write_text(serialized, encoding="utf-8")
    (output_dir / "summary.json").write_text(serialized, encoding="utf-8")
    (output_dir / "dual_pckh_summary.json").write_text(serialized, encoding="utf-8")
    np.savez_compressed(
        output_dir / "prediction_matrices.npz", **arrays,
        pckhn_head_scale=custom_scale, **matrices,
    )
    return summary
