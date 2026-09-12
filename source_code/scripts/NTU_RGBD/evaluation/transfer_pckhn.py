from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.MPII.core import MPII_FLIP_PAIRS, MPII_JOINT_NAMES
from scripts.MPII.core.geometry import heatmaps_to_keypoints


# MPII's 16 output channels mapped onto the closest Kinect V2 joints.
MPII_TO_NTU_JOINTS = np.asarray(
    [18, 17, 16, 12, 13, 14, 0, 20, 2, 3, 10, 9, 8, 4, 5, 6],
    dtype=np.int64,
)

TRANSFER_JOINT_NAMES = list(MPII_JOINT_NAMES)
TRANSFER_JOINT_NAMES[8] = "neck_center"
TRANSFER_JOINT_NAMES[9] = "head_center"
TRANSFER_JOINT_NAMES = tuple(TRANSFER_JOINT_NAMES)


def predictions_to_ntu_semantics(prediction: np.ndarray) -> np.ndarray:
    """Calibrate E0 output channels and align MPII head semantics to NTU."""
    converted = np.asarray(prediction, dtype=np.float32).copy()
    for right, left in MPII_FLIP_PAIRS:
        converted[..., [right, left], :] = converted[..., [left, right], :]
    head_top = converted[..., 9, :].copy()
    upper_neck = converted[..., 8, :].copy()
    converted[..., 9, :] = 0.5 * head_top + 0.5 * upper_neck
    converted[..., 8, :] = upper_neck
    return converted


@torch.no_grad()
def evaluate_transfer_pckhn(model, loader, dataset, device, output_dir: Path,
                            threshold: float = 0.5) -> dict:
    collected = {key: [] for key in (
        "pred", "gt", "visibility", "head_length", "sample_id", "frame_index",
    )}
    model.eval()
    for batch in loader:
        heatmaps = model(batch["image"].to(device)).cpu().numpy()
        targets = batch["keypoints"].numpy()[:, MPII_TO_NTU_JOINTS]
        visibility = batch["visibility"].numpy()[:, MPII_TO_NTU_JOINTS]
        for index, heatmap in enumerate(heatmaps):
            prediction, _ = heatmaps_to_keypoints(
                heatmap, dataset.image_size, method="argmax",
            )
            prediction = predictions_to_ntu_semantics(prediction)
            collected["pred"].append(prediction)
            collected["gt"].append(targets[index])
            collected["visibility"].append(visibility[index])
            collected["head_length"].append(float(batch["head_length"][index]))
            collected["sample_id"].append(batch["sample_id"][index])
            collected["frame_index"].append(int(batch["frame_index"][index]))

    arrays = {
        key: np.asarray(value) for key, value in collected.items()
        if key not in {"sample_id"}
    }
    scale = arrays["head_length"][:, None]
    valid = ((arrays["visibility"] > 0) & np.isfinite(scale) & (scale > 1e-6))
    distance = np.linalg.norm(arrays["pred"] - arrays["gt"], axis=2)
    normalized_distance = distance / np.maximum(scale, 1e-6)
    correct = (normalized_distance <= threshold) & valid
    valid_per_joint = valid.sum(axis=0)
    correct_per_joint = correct.sum(axis=0)
    pckhn_per_joint = np.divide(
        correct_per_joint, valid_per_joint,
        out=np.full(len(TRANSFER_JOINT_NAMES), np.nan, dtype=np.float64),
        where=valid_per_joint > 0,
    )
    valid_total = int(valid.sum())
    summary = {
        "experiment_id": getattr(model, "experiment_id", "e0"),
        "model_name": getattr(model, "model_name", "SpikePose-E0"),
        "source_dataset": "mpii",
        "target_dataset": "ntu_rgbd",
        "metric": "pckhn",
        "normalization": "NTU head-to-neck distance in crop coordinates",
        "threshold": float(threshold),
        "samples": int(len(arrays["pred"])),
        "valid_joints": valid_total,
        "pckhn": float(correct.sum() / max(valid_total, 1)),
        "per_joint_pckhn": {
            name: (None if np.isnan(pckhn_per_joint[index])
                   else float(pckhn_per_joint[index]))
            for index, name in enumerate(TRANSFER_JOINT_NAMES)
        },
        "mpii_to_ntu_joint_indices": MPII_TO_NTU_JOINTS.tolist(),
        "left_right_output_correction": [list(pair) for pair in MPII_FLIP_PAIRS],
        "head_center_conversion": "0.5 * MPII head_top + 0.5 * MPII upper_neck",
        "neck_center_conversion": "MPII upper_neck",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(summary, indent=2)
    (output_dir / "pckhn_summary.json").write_text(serialized, encoding="utf-8")
    (output_dir / "best.json").write_text(serialized, encoding="utf-8")
    with (output_dir / "pckhn_per_joint.csv").open(
        "w", newline="", encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("joint_id", "joint_name", "ntu_joint_id", "valid", "pckhn"))
        for index, name in enumerate(TRANSFER_JOINT_NAMES):
            writer.writerow((
                index, name, int(MPII_TO_NTU_JOINTS[index]),
                int(valid_per_joint[index]),
                "" if np.isnan(pckhn_per_joint[index]) else float(pckhn_per_joint[index]),
            ))
    np.savez_compressed(
        output_dir / "prediction_matrices.npz",
        pred=arrays["pred"], gt=arrays["gt"], visibility=arrays["visibility"],
        head_length=arrays["head_length"], frame_index=arrays["frame_index"],
        sample_id=np.asarray(collected["sample_id"]),
        normalized_distance=normalized_distance, valid=valid,
    )
    return summary
