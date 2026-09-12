from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from spikepose_thesis.data.ntu.core.joint_mapping import (
    joint_layout,
    joint_names_for_count,
)
from spikepose_thesis.core.paths import resolve_project_path
from spikepose_thesis.evaluation.calibration import load_head_bone_scale
from spikepose_thesis.evaluation.mpii.metrics import (
    _average_predictions, _flip_back_prediction,
)
from spikepose_thesis.evaluation.mpii.metrics import prediction_to_keypoints
from spikepose_thesis.evaluation.temporal import sequence_groups, temporal_summary


def _to_original_mpii(prediction: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    result = prediction.copy()
    if inverse.shape[-1] == 4:
        result[:, 0] = prediction[:, 0] * inverse[0] + inverse[2]
        result[:, 1] = prediction[:, 1] * inverse[1] + inverse[3]
    elif inverse.shape == (2, 3):
        result = np.concatenate((prediction, np.ones((len(prediction), 1))), 1) @ inverse.T
    else:
        raise ValueError(f"Unsupported MPII inverse transform: {inverse.shape}")
    return result


def _to_original_ntu(prediction: np.ndarray, bbox: np.ndarray,
                     image_size: int) -> np.ndarray:
    """Restore tube-crop coordinates to complete-frame pixel coordinates."""
    if prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("NTU prediction must have shape [J, 2]")
    if bbox.shape != (4,):
        raise ValueError("NTU person bbox must have shape [4]")
    x1, y1, x2, y2 = bbox.astype(np.float32)
    result = prediction.copy()
    result[:, 0] = x1 + result[:, 0] * (x2 - x1) / float(image_size)
    result[:, 1] = y1 + result[:, 1] * (y2 - y1) / float(image_size)
    return result


def _pck(pred: np.ndarray, target: np.ndarray, visibility: np.ndarray,
         scale: np.ndarray, threshold: float) -> float:
    distance = np.linalg.norm(pred - target, axis=-1)
    valid = (visibility > 0) & np.isfinite(distance) & np.isfinite(scale[:, None])
    correct = distance <= threshold * scale[:, None]
    return float(correct[valid].mean()) if valid.any() else float("nan")


def _nanmean(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def _empty_prediction_result(*, temporal_identity: bool = False) -> dict[str, list]:
    result = {
        "prediction": [], "target": [], "visibility": [], "scale": [],
        "scale_hb": [], "sample_id": [], "frame_index": [], "person_id": [],
    }
    if temporal_identity:
        result.update({
            "video_id": [], "clip_id": [], "frame_position_in_clip": [],
        })
    return result


def _to_original_ntu_temporal(
    coordinates: np.ndarray, bboxes: np.ndarray, image_size: int,
) -> np.ndarray:
    """Vectorized B x T x J crop-to-original coordinate transform."""
    if coordinates.ndim != 4 or coordinates.shape[-1] != 2:
        raise ValueError("Temporal NTU coordinates must have shape B x T x J x 2")
    if bboxes.shape != (coordinates.shape[0], 4):
        raise ValueError("Temporal NTU bboxes must have shape B x 4")
    origin = bboxes[:, None, None, :2].astype(np.float32, copy=False)
    extent = (
        bboxes[:, None, None, 2:4] - bboxes[:, None, None, :2]
    ).astype(np.float32, copy=False)
    return origin + coordinates * (extent / float(image_size))


def _append_ntu_temporal_predictions(
    result: dict[str, list], prediction: torch.Tensor, batch: dict,
    image_size: int, decoder: str, head_index: int, neck_index: int,
) -> None:
    """Append one decoded row per physical NTU frame without writing files."""
    if prediction.ndim != 5:
        raise ValueError("Per-frame heatmaps must have shape T x B x J x H x W")
    required = {
        "temporal_keypoints", "temporal_visibility", "temporal_frame_indices",
        "temporal_frame_positions", "person_bbox", "video_id", "clip_id",
    }
    missing = required - batch.keys()
    if missing:
        raise ValueError(
            "Per-frame NTU evaluation requires temporal targets; missing "
            f"{sorted(missing)}"
        )
    steps, batch_size, joints = prediction.shape[:3]
    coordinates, _ = prediction_to_keypoints(
        prediction.flatten(0, 1), image_size, decoder=decoder,
    )
    coordinates = coordinates.reshape(steps, batch_size, joints, 2).transpose(1, 0, 2, 3)
    targets = batch["temporal_keypoints"].numpy()
    visibility = batch["temporal_visibility"].numpy()
    frame_indices = batch["temporal_frame_indices"].numpy()
    frame_positions = batch["temporal_frame_positions"].numpy()
    bboxes = batch["person_bbox"].numpy()
    people = batch.get("person_id", ["primary"] * batch_size)
    if targets.shape[:3] != (batch_size, steps, joints):
        raise ValueError(
            "Temporal prediction/target shape mismatch: "
            f"prediction={tuple(prediction.shape)} target={tuple(targets.shape)}"
        )
    if frame_indices.shape != (batch_size, steps):
        raise ValueError("temporal_frame_indices must have shape B x T")
    if frame_positions.shape != (batch_size, steps):
        raise ValueError("temporal_frame_positions must have shape B x T")

    pred_original = _to_original_ntu_temporal(coordinates, bboxes, image_size)
    target_original = _to_original_ntu_temporal(targets, bboxes, image_size)
    scale = np.linalg.norm(
        target_original[:, :, head_index] - target_original[:, :, neck_index],
        axis=-1,
    ).astype(np.float32)
    head_pair_valid = (
        (visibility[:, :, neck_index] > 0)
        & (visibility[:, :, head_index] > 0)
    )
    scale[~head_pair_valid] = np.nan
    rows = batch_size * steps
    result["prediction"].extend(pred_original.reshape(rows, joints, 2))
    result["target"].extend(target_original.reshape(rows, joints, 2))
    result["visibility"].extend(visibility.reshape(rows, joints))
    result["scale"].extend(scale.reshape(rows))
    result["scale_hb"].extend(scale.reshape(rows))
    result["frame_index"].extend(frame_indices.reshape(rows))
    result["frame_position_in_clip"].extend(frame_positions.reshape(rows))
    for item in range(batch_size):
        result["sample_id"].extend([batch["sample_id"][item]] * steps)
        result["video_id"].extend([str(batch["video_id"][item])] * steps)
        result["clip_id"].extend([str(batch["clip_id"][item])] * steps)
        result["person_id"].extend([str(people[item])] * steps)


@torch.inference_mode()
def collect_predictions(model, loader, config: dict, device: torch.device,
                        decoder: str | None = None,
                        flip_test: bool = False,
                        loss_callback=None,
                        collect_temporal_frames: bool = False):
    model.eval()
    if collect_temporal_frames and config["dataset"] == "mpii":
        raise ValueError("Per-frame temporal evaluation is only defined for NTU")
    if collect_temporal_frames and flip_test:
        raise ValueError("Per-frame temporal evaluation does not support flip test")
    result = _empty_prediction_result()
    temporal_result = (
        _empty_prediction_result(temporal_identity=True)
        if collect_temporal_frames else None
    )
    decoder = decoder or config.get("evaluation", {}).get("main_decoder", "dark")
    hb_factor = None
    ntu_layout = (
        joint_layout(config["data"].get("joint_mapping", "ntu25_to_mpii16_v1"))
        if config["dataset"] != "mpii" else None
    )
    loss_total = 0.0
    loss_count = 0
    if config["dataset"] == "mpii" and config["data"].get("head_bone_calibration"):
        hb_factor = load_head_bone_scale(
            resolve_project_path(config["data"]["head_bone_calibration"])
        )
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if collect_temporal_frames:
            per_frame_prediction = model.forward_per_step(images)
            prediction = per_frame_prediction[-1]
        else:
            per_frame_prediction = None
            prediction = model(images)
        if loss_callback is not None:
            batch_loss = loss_callback(prediction, batch)
            loss_total += float(batch_loss.detach()) * len(images)
            loss_count += len(images)
        if flip_test:
            flipped = model(torch.flip(images, dims=(-1,)))
            prediction = _average_predictions(
                prediction, _flip_back_prediction(flipped, shift=True),
            )
        coordinates, _ = prediction_to_keypoints(
            prediction, int(config["data"]["image_size"]), decoder=decoder,
        )
        size = len(coordinates)
        if config["dataset"] == "mpii":
            targets = batch["keypoints_original"].numpy()
            for index in range(size):
                coordinates[index] = _to_original_mpii(
                    coordinates[index], batch["inverse"][index].numpy(),
                )
            scales = batch["head_length"].numpy()
            head_bone = np.linalg.norm(targets[:, 9] - targets[:, 8], axis=1)
            hb_scales = (
                head_bone * hb_factor
                if hb_factor is not None else np.full(size, np.nan)
            )
            head_pair_valid = (
                (batch["visibility"][:, 8].numpy() > 0)
                & (batch["visibility"][:, 9].numpy() > 0)
            )
            hb_scales[~head_pair_valid] = np.nan
            sample_ids = [
                f"{name}::person_{int(person)}"
                for name, person in zip(batch["image_name"], batch["person_index"])
            ]
            frames = np.full(size, -1, dtype=np.int64)
            people = batch["person_index"].numpy()
        else:
            targets = batch["keypoints_original"].numpy()
            coordinates = _to_original_ntu_temporal(
                coordinates[:, None], batch["person_bbox"].numpy(),
                int(config["data"]["image_size"]),
            )[:, 0]
            head_index = int(ntu_layout["head_index"])
            neck_index = int(ntu_layout["neck_index"])
            head_bone = np.linalg.norm(
                targets[:, head_index] - targets[:, neck_index], axis=1,
            )
            head_pair_valid = (
                (batch["original_visibility"][:, neck_index].numpy() > 0)
                & (batch["original_visibility"][:, head_index].numpy() > 0)
            )
            scales = head_bone.astype(np.float32)
            scales[~head_pair_valid] = np.nan
            hb_scales = scales.copy()
            sample_ids = list(batch["sample_id"])
            frames = batch["frame_index"].numpy()
            people = np.asarray([
                str(item) for item in batch.get("person_id", ["primary"] * size)
            ])
        result["prediction"].extend(coordinates)
        result["target"].extend(targets)
        visibility = (
            batch["visibility"] if config["dataset"] == "mpii"
            else batch["original_visibility"]
        )
        result["visibility"].extend(visibility.numpy())
        result["scale"].extend(scales)
        result["scale_hb"].extend(hb_scales)
        result["sample_id"].extend(sample_ids)
        result["frame_index"].extend(frames)
        result["person_id"].extend(people)
        if temporal_result is not None:
            _append_ntu_temporal_predictions(
                temporal_result, per_frame_prediction, batch,
                int(config["data"]["image_size"]), decoder,
                int(ntu_layout["head_index"]), int(ntu_layout["neck_index"]),
            )
    values = {key: np.asarray(value) for key, value in result.items()}
    temporal_values = (
        {key: np.asarray(value) for key, value in temporal_result.items()}
        if temporal_result is not None else None
    )
    if loss_callback is not None:
        if temporal_values is not None:
            return values, temporal_values, loss_total / max(loss_count, 1)
        return values, loss_total / max(loss_count, 1)
    if temporal_values is not None:
        return values, temporal_values
    return values


def _basic_summary(pred: np.ndarray, target: np.ndarray,
                   visible: np.ndarray, scale: np.ndarray) -> dict:
    summary = {
        f"pck_{threshold:.1f}": _pck(pred, target, visible, scale, threshold)
        for threshold in (0.1, 0.2, 0.5)
    }
    normalized = np.linalg.norm(pred - target, axis=-1) / scale[:, None]
    valid = (visible > 0) & np.isfinite(normalized)
    summary["nme_hb"] = float(normalized[valid].mean()) if valid.any() else float("nan")
    thresholds = np.linspace(0.0, 0.5, 101)
    curve = np.asarray([_pck(pred, target, visible, scale, item) for item in thresholds])
    summary["pck_auc_0_0_5"] = float(np.trapezoid(curve, thresholds) / 0.5)
    summary["samples"] = int(len(pred))
    return summary


def summarize_predictions(values: dict[str, np.ndarray]) -> dict:
    pred, target = values["prediction"], values["target"]
    visible, scale = values["visibility"], values["scale"]
    summary = _basic_summary(pred, target, visible, scale)
    distance = np.linalg.norm(pred - target, axis=-1)
    joint_names = joint_names_for_count(pred.shape[1])
    summary["per_joint"] = {
        name: {
            f"pck_{threshold:.1f}": float(
                (distance[:, joint][mask] <= threshold * scale[mask]).mean()
            ) if mask.any() else float("nan")
            for threshold in (0.1, 0.2, 0.5)
        }
        for joint, name in enumerate(joint_names)
        for mask in [
            (visible[:, joint] > 0)
            & np.isfinite(distance[:, joint])
            & np.isfinite(scale)
        ]
    }
    groups = sequence_groups(values)
    if groups:
        sequence_rows = [
            _basic_summary(
                pred[indices], target[indices], visible[indices], scale[indices],
            )
            for indices in groups
        ]
        summary["sequence_equal"] = {
            key: _nanmean([row[key] for row in sequence_rows])
            for key in ("pck_0.1", "pck_0.2", "pck_0.5", "nme_hb", "pck_auc_0_0_5")
        }
        summary["sequences"] = len(groups)
    return summary


def _ntu_group_summaries(values: dict[str, np.ndarray]) -> dict:
    sample_ids = np.asarray([str(item).split("::", 1)[0] for item in values["sample_id"]])
    definitions = {
        "action": np.asarray([
            sample[sample.rfind("A") + 1:] if "A" in sample else "unknown"
            for sample in sample_ids
        ]),
        "camera": np.asarray([
            sample[5:8] if len(sample) >= 8 else "unknown"
            for sample in sample_ids
        ]),
    }
    result = {}
    for label, groups in definitions.items():
        result[label] = {}
        for group in sorted(set(groups.tolist())):
            indices = np.flatnonzero(groups == group)
            result[label][group] = _basic_summary(
                values["prediction"][indices], values["target"][indices],
                values["visibility"][indices], values["scale"][indices],
            )
    return result


def validation_score(model, loader, config: dict, device: torch.device) -> float:
    values = collect_predictions(model, loader, config, device, decoder="argmax")
    return summarize_predictions(values)["pck_0.5"]


def evaluate_checkpoint(model, loader, config: dict, device: torch.device,
                        output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    collect_temporal_frames = bool(
        config["dataset"] != "mpii"
        and config.get("training", {}).get("loss_all_frames", False)
    )
    collected = collect_predictions(
        model, loader, config, device,
        collect_temporal_frames=collect_temporal_frames,
    )
    if collect_temporal_frames:
        values, temporal_values = collected
    else:
        values = collected
        temporal_values = None
    np.savez_compressed(output_dir / "predictions.npz", **values)
    if temporal_values is not None:
        np.savez_compressed(
            output_dir / "predictions_per_frame.npz", **temporal_values,
        )
    summary = summarize_predictions(values)
    hb_values = {**values, "scale": values["scale_hb"]}
    summary["pck_hb"] = summarize_predictions(hb_values)
    summary["head_bone_calibration_ready"] = bool(
        config["dataset"] != "mpii" or np.isfinite(values["scale_hb"]).any()
    )
    if config["dataset"] != "mpii":
        summary["temporal"] = temporal_summary(
            temporal_values if temporal_values is not None else values,
            strict=temporal_values is not None,
            expected_length=(
                int(config["temporal"]["video_frames"])
                if temporal_values is not None else None
            ),
        )
        summary["groups"] = _ntu_group_summaries(values)
    supplementary = config.get("evaluation", {}).get("supplementary_decoder")
    if supplementary == "flip_dark":
        flip_values = collect_predictions(
            model, loader, config, device, decoder="dark", flip_test=True,
        )
        np.savez_compressed(output_dir / "predictions_flip_dark.npz", **flip_values)
        flip_summary = summarize_predictions(flip_values)
        flip_summary["pck_hb"] = summarize_predictions({
            **flip_values, "scale": flip_values["scale_hb"],
        })
        summary["flip_dark"] = flip_summary
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    return summary
