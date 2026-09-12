#!/usr/bin/env python3
"""Compare trained-window and video-boundary MAM resets on complete videos.

Each physical frame receives one retained spatial prediction with a single tube
crop per source video.  The resulting raw heatmaps are then passed through the
frozen MAM checkpoint under two policies: independent trained-length windows,
and one continuous invocation per complete video.  This isolates reset policy
from crop geometry and spatial-model variation.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.core.paths import PROJECT_ROOT, resolve_project_path
from spikepose_thesis.data import build_dataset
from spikepose_thesis.data.ntu.core.joint_mapping import (
    MPII16_JOINT_NAMES,
    joint_layout,
)
from spikepose_thesis.evaluation.mam_reset_policy import (
    SameVideoChunkBatchSampler,
    apply_mam_reset_policy,
    trained_window_starts,
    trained_window_stitch_positions,
)
from spikepose_thesis.evaluation.mpii.metrics import prediction_to_keypoints
from spikepose_thesis.evaluation.runner import (
    _ntu_group_summaries,
    summarize_predictions,
)
from spikepose_thesis.evaluation.temporal import sequence_groups, temporal_summary
from spikepose_thesis.models import build_model
from spikepose_thesis.models.mam_v2 import MotionAlignedMembraneV2
from spikepose_thesis.training.checkpoint import load_model


DEFAULT_METADATA = "Datasets/NTU_RGBD/metadata/fullvideo_xsub35/test_split.csv"
DEFAULT_FRAMES = "Datasets/NTU_RGBD/extracted_frames_full"
METHODS = ("raw", "window_reset", "video_reset")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--frames-dir", default=DEFAULT_FRAMES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(command: list[str]) -> str:
    return subprocess.run(
        ["git", *command], cwd=PROJECT_ROOT, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()


def _load_config(args) -> tuple[dict, str]:
    if args.resolved_config is None:
        return load_experiment(args.experiment), "experiment_registry"
    source = args.resolved_config.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if config.get("id") != args.experiment:
        raise ValueError(
            f"Resolved config id {config.get('id')!r} does not match "
            f"--experiment {args.experiment!r}"
        )
    return config, str(source)


def _verify_preflight_subset(config: dict, metadata: Path) -> dict:
    manifest_value = config["data"].get("pose_validation_manifest")
    if not manifest_value:
        raise ValueError("Training config does not define pose_validation_manifest")
    manifest_path = resolve_project_path(manifest_value).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("ready") or manifest.get("invalid_samples"):
        raise RuntimeError(f"Pose validation manifest is not clean: {manifest_path}")
    approved = frozenset(str(item) for item in manifest["valid_sample_ids"])
    with metadata.open("r", encoding="utf-8", newline="") as handle:
        subset_ids = [str(row["sample_id"]) for row in csv.DictReader(handle)]
    missing = sorted(set(subset_ids) - approved)
    if missing:
        raise RuntimeError(
            f"Full-video subset contains {len(missing)} samples absent from the "
            f"frozen pose-valid whitelist; examples={missing[:10]}"
        )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_valid_samples": len(approved),
        "subset_rows": len(subset_ids),
        "subset_unique_samples": len(set(subset_ids)),
        "subset_missing_from_valid_whitelist": 0,
    }


def _select_exhaustive_chunks(dataset) -> tuple[int, int]:
    steps = int(dataset.temporal_steps)
    if steps <= 1 or int(dataset.temporal_frame_gap) != 1:
        raise ValueError("Reset-policy validation requires consecutive video frames")
    last_position: dict[int, int] = {}
    for sample_index, _frame, _clip, position in dataset.frame_index:
        last_position[int(sample_index)] = max(
            int(position), last_position.get(int(sample_index), -1),
        )
    if len(last_position) != len(dataset.samples):
        raise RuntimeError("Some dataset videos have no complete temporal window")
    dataset.frame_index = [
        item for item in dataset.frame_index
        if int(item[3]) % steps == steps - 1
        or int(item[3]) == last_position[int(item[0])]
    ]
    return len(dataset.frame_index), sum(value + 1 for value in last_position.values())


class _ProgressLoader:
    def __init__(self, loader: DataLoader, interval: int) -> None:
        self.loader = loader
        self.interval = max(int(interval), 1)

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        started = time.monotonic()
        total = len(self.loader)
        for index, batch in enumerate(self.loader, start=1):
            if index == 1 or index % self.interval == 0 or index == total:
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = index / elapsed
                eta = (total - index) / rate if rate > 0 else float("nan")
                print(
                    f"progress batches={index}/{total} "
                    f"elapsed_min={elapsed / 60:.1f} eta_min={eta / 60:.1f}",
                    flush=True,
                )
            yield batch


def _empty_values() -> dict[str, list]:
    return {
        "prediction": [], "target": [], "visibility": [], "scale": [],
        "scale_hb": [], "sample_id": [], "frame_index": [], "person_id": [],
        "video_id": [], "clip_id": [], "frame_position_in_clip": [],
    }


def _to_original(coordinates: np.ndarray, bboxes: np.ndarray, image_size: int) -> np.ndarray:
    origin = bboxes[:, None, :2].astype(np.float32, copy=False)
    extent = (bboxes[:, 2:4] - bboxes[:, :2])[:, None].astype(
        np.float32, copy=False,
    )
    return origin + coordinates * (extent / float(image_size))


def _append_decoded(
    destination: dict[str, list], heatmaps: torch.Tensor, records: list[dict],
    *, image_size: int, decoder: str, head_index: int, neck_index: int,
) -> None:
    coordinates, _ = prediction_to_keypoints(heatmaps[:, 0], image_size, decoder)
    bboxes = np.stack([record["bbox"] for record in records])
    targets = np.stack([record["target"] for record in records])
    visibility = np.stack([record["visibility"] for record in records])
    prediction_original = _to_original(coordinates, bboxes, image_size)
    target_original = _to_original(targets, bboxes, image_size)
    scale = np.linalg.norm(
        target_original[:, head_index] - target_original[:, neck_index], axis=-1,
    ).astype(np.float32)
    valid_head = (
        (visibility[:, head_index] > 0) & (visibility[:, neck_index] > 0)
    )
    scale[~valid_head] = np.nan
    destination["prediction"].extend(prediction_original)
    destination["target"].extend(target_original)
    destination["visibility"].extend(visibility)
    destination["scale"].extend(scale)
    destination["scale_hb"].extend(scale)
    for record in records:
        destination["sample_id"].append(record["sample_id"])
        destination["frame_index"].append(record["frame_index"])
        destination["person_id"].append(record["person_id"])
        destination["video_id"].append(record["video_id"])
        destination["clip_id"].append("full")
        destination["frame_position_in_clip"].append(record["position"])


def _add_batch_records(
    records: dict[int, dict], batch: dict, model, device: torch.device,
) -> None:
    positions = batch["temporal_frame_positions"].numpy()
    frame_indices = batch["temporal_frame_indices"].numpy()
    targets = batch["temporal_keypoints"].numpy()
    visibility = batch["temporal_visibility"].numpy()
    bboxes = batch["person_bbox"].numpy()
    pending: dict[int, dict] = {}
    for lane in range(len(positions)):
        for step, position in enumerate(positions[lane]):
            position = int(position)
            candidate = {
                "image": batch["image"][lane, step],
                "target": targets[lane, step],
                "visibility": visibility[lane, step],
                "bbox": bboxes[lane].copy(),
                "sample_id": str(batch["sample_id"][lane]).split("::", 1)[0],
                "video_id": str(batch["video_id"][lane]),
                "person_id": str(batch["person_id"][lane]),
                "frame_index": int(frame_indices[lane, step]),
                "position": position,
            }
            existing = records.get(position, pending.get(position))
            if existing is not None:
                if (
                    existing["frame_index"] != candidate["frame_index"]
                    or existing["video_id"] != candidate["video_id"]
                    or existing["person_id"] != candidate["person_id"]
                    or not np.array_equal(
                        existing["target"], candidate["target"], equal_nan=True,
                    )
                    or not np.array_equal(
                        existing["visibility"], candidate["visibility"],
                    )
                    or not np.array_equal(existing["bbox"], candidate["bbox"])
                ):
                    raise RuntimeError(
                        f"Overlapping spatial-window metadata disagree at "
                        f"{batch['video_id'][lane]} position={position}"
                    )
                continue
            pending[position] = candidate

    if not pending:
        return
    # The final trained-length extraction window overlaps the preceding fixed
    # window.  Encode only positions not already retained.  Besides avoiding
    # redundant work, this prevents harmless CUDA batch-shape round-off from
    # making the same frame appear to have two different spatial predictions.
    new_records = list(pending.values())
    images = torch.stack([record.pop("image") for record in new_records])
    spatial = model.forward_spatial_per_step(
        images.unsqueeze(0).to(device, non_blocking=True),
    )
    if spatial.shape[:2] != (len(new_records), 1):
        raise RuntimeError(
            f"Unexpected unique-frame spatial shape {tuple(spatial.shape)}"
        )
    spatial_cpu = spatial[:, 0].detach().cpu()
    for record, heatmap in zip(new_records, spatial_cpu, strict=True):
        record["heatmap"] = heatmap
        records[record["position"]] = record


@torch.inference_mode()
def _collect_predictions(
    model, loader, config: dict, device: torch.device, trained_steps: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    model.eval()
    if not isinstance(getattr(model, "mam", None), MotionAlignedMembraneV2):
        raise TypeError("Reset-policy validation requires a MAM V2 checkpoint")
    image_size = int(config["data"]["image_size"])
    decoder = config.get("evaluation", {}).get("main_decoder", "dark")
    layout = joint_layout(config["data"].get("joint_mapping", "ntu25_to_mpii16_v1"))
    destinations = {method: _empty_values() for method in METHODS}
    current_video: str | None = None
    records: dict[int, dict] = {}
    invocation_counts = {"window_reset": 0, "video_reset": 0}

    def flush() -> None:
        nonlocal records
        if not records:
            return
        ordered_positions = sorted(records)
        if ordered_positions != list(range(len(ordered_positions))):
            raise RuntimeError(
                f"Non-exhaustive video positions for {current_video}: "
                f"first={ordered_positions[:5]} last={ordered_positions[-5:]}"
            )
        ordered = [records[position] for position in ordered_positions]
        frame_indices = [record["frame_index"] for record in ordered]
        if frame_indices != list(range(len(frame_indices))):
            raise RuntimeError(f"Non-contiguous source frames for {current_video}")
        reference_bbox = ordered[0]["bbox"]
        if not all(np.allclose(record["bbox"], reference_bbox) for record in ordered):
            raise RuntimeError(f"Video-level tube crop changed within {current_video}")
        spatial = torch.stack([record["heatmap"] for record in ordered])[
            :, None
        ].to(device, non_blocking=True)
        _append_decoded(
            destinations["raw"], spatial, ordered, image_size=image_size,
            decoder=decoder, head_index=int(layout["head_index"]),
            neck_index=int(layout["neck_index"]),
        )
        window = apply_mam_reset_policy(
            model.mam, spatial, policy="trained_window", trained_steps=trained_steps,
        )
        invocation_counts["window_reset"] += window.model_invocations
        _append_decoded(
            destinations["window_reset"], window.heatmap, ordered,
            image_size=image_size, decoder=decoder,
            head_index=int(layout["head_index"]),
            neck_index=int(layout["neck_index"]),
        )
        del window
        video = apply_mam_reset_policy(
            model.mam, spatial, policy="video_boundary", trained_steps=trained_steps,
        )
        invocation_counts["video_reset"] += video.model_invocations
        _append_decoded(
            destinations["video_reset"], video.heatmap, ordered,
            image_size=image_size, decoder=decoder,
            head_index=int(layout["head_index"]),
            neck_index=int(layout["neck_index"]),
        )
        del video, spatial
        records = {}

    for batch in loader:
        batch_videos = {str(item) for item in batch["video_id"]}
        if len(batch_videos) != 1:
            raise RuntimeError(f"A spatial batch mixed videos: {sorted(batch_videos)}")
        batch_video = next(iter(batch_videos))
        if current_video is not None and batch_video != current_video:
            flush()
        current_video = batch_video
        _add_batch_records(records, batch, model, device)
    flush()
    values = {
        method: {key: np.asarray(value) for key, value in destination.items()}
        for method, destination in destinations.items()
    }
    return values, invocation_counts


def _validate_values(values: dict[str, np.ndarray], frames: int, videos: int) -> None:
    if len(values["prediction"]) != frames:
        raise RuntimeError(f"Expected {frames} frames, got {len(values['prediction'])}")
    groups = sequence_groups(values)
    if len(groups) != videos:
        raise RuntimeError(f"Expected {videos} videos, got {len(groups)}")
    for indices in groups:
        positions = values["frame_position_in_clip"][indices].astype(np.int64)
        frame_indices = values["frame_index"][indices].astype(np.int64)
        expected = np.arange(len(indices), dtype=np.int64)
        if not np.array_equal(positions, expected) or not np.array_equal(
            frame_indices, expected,
        ):
            raise RuntimeError(
                f"Non-exhaustive output video: {values['video_id'][indices[0]]}"
            )


def _full_frame_pck(values: dict[str, np.ndarray]) -> dict:
    distance = np.linalg.norm(values["prediction"] - values["target"], axis=-1)
    valid = (
        (values["visibility"] > 0) & np.isfinite(distance)
        & np.isfinite(values["scale_hb"][:, None])
    )
    correct = distance <= 0.5 * values["scale_hb"][:, None]
    return {
        "pck_0.5": float(correct[valid].mean()),
        "samples": int(len(distance)),
        "per_joint": {
            name: {"pck_0.5": float(correct[:, joint][valid[:, joint]].mean())}
            for joint, name in enumerate(MPII16_JOINT_NAMES)
        },
    }


def _summarize(values: dict[str, np.ndarray], protocol: dict) -> dict:
    result = summarize_predictions(values)
    result["pck_hb"] = summarize_predictions({**values, "scale": values["scale_hb"]})
    result["pck_hb_all_frames"] = _full_frame_pck(values)
    result["pck_hb_video_macro"] = {
        "pck_0.5": float(result["pck_hb"]["sequence_equal"]["pck_0.5"]),
        "videos": int(result["pck_hb"]["sequences"]),
    }
    result["temporal"] = temporal_summary(values, strict=False)
    result["groups"] = _ntu_group_summaries(values)
    result["head_bone_calibration_ready"] = True
    result["physical_frames"] = int(len(values["prediction"]))
    result["videos"] = int(result["pck_hb"]["sequences"])
    result["protocol"] = protocol
    return result


def _per_video_metrics(values: dict[str, np.ndarray]) -> np.ndarray:
    rows = []
    for indices in sequence_groups(values):
        prediction = values["prediction"][indices]
        target = values["target"][indices]
        visible = values["visibility"][indices] > 0
        scale = values["scale_hb"][indices]
        distance = np.linalg.norm(prediction - target, axis=-1)
        valid = visible & np.isfinite(scale[:, None]) & np.isfinite(distance)
        pck = (distance <= 0.5 * scale[:, None])[valid].mean()
        velocity_mask = visible[1:] & visible[:-1]
        velocity_error = np.linalg.norm(
            np.diff(prediction, axis=0) - np.diff(target, axis=0), axis=-1,
        )
        acceleration_mask = visible[2:] & visible[1:-1] & visible[:-2]
        acceleration_error = np.linalg.norm(
            np.diff(prediction, n=2, axis=0) - np.diff(target, n=2, axis=0),
            axis=-1,
        )
        rows.append((
            float(pck), float(velocity_error[velocity_mask].mean()),
            float(acceleration_error[acceleration_mask].mean()),
        ))
    return np.asarray(rows, dtype=np.float64)


def _paired_comparison(
    window: dict[str, np.ndarray], video: dict[str, np.ndarray],
    bootstrap_samples: int,
) -> dict:
    left = _per_video_metrics(window)
    right = _per_video_metrics(video)
    if left.shape != right.shape:
        raise RuntimeError("Reset-policy per-video metric shapes differ")
    differences = right - left
    names = ("pck_hb_0_5", "vel_e_px", "acc_e_px")
    rng = np.random.default_rng(42)
    result = {}
    for column, name in enumerate(names):
        values = differences[:, column]
        if bootstrap_samples > 0:
            indices = rng.integers(
                0, len(values), size=(bootstrap_samples, len(values)),
            )
            distribution = values[indices].mean(axis=1)
            low, high = np.percentile(distribution, (2.5, 97.5))
        else:
            low = high = float("nan")
        result[name] = {
            "video_reset_minus_window_reset": float(values.mean()),
            "paired_video_bootstrap_95_ci": [float(low), float(high)],
        }
    return {
        "videos": int(len(differences)),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": 42,
        "metrics": result,
    }


def _stitch_transition_summary(
    values: dict[str, np.ndarray], trained_steps: int,
) -> dict[str, float | int]:
    stitch_errors = []
    other_errors = []
    stitches = 0
    for indices in sequence_groups(values):
        prediction = values["prediction"][indices]
        target = values["target"][indices]
        visible = values["visibility"][indices] > 0
        velocity_error = np.linalg.norm(
            np.diff(prediction, axis=0) - np.diff(target, axis=0), axis=-1,
        )
        valid = visible[1:] & visible[:-1]
        stitch_positions = set(
            trained_window_stitch_positions(len(indices), trained_steps)[1:]
        )
        for position in range(1, len(indices)):
            selected = velocity_error[position - 1][valid[position - 1]]
            if not len(selected):
                continue
            if position in stitch_positions:
                stitch_errors.extend(selected.tolist())
                stitches += 1
            else:
                other_errors.extend(selected.tolist())
    stitch_mean = float(np.mean(stitch_errors)) if stitch_errors else float("nan")
    other_mean = float(np.mean(other_errors)) if other_errors else float("nan")
    return {
        "output_stitch_transitions": stitches,
        "stitch_velocity_error_px": stitch_mean,
        "non_stitch_velocity_error_px": other_mean,
        "stitch_to_non_stitch_ratio": (
            stitch_mean / other_mean
            if np.isfinite(stitch_mean) and np.isfinite(other_mean) and other_mean > 0
            else float("nan")
        ),
    }


def _write_method(
    output: Path, method: str, values: dict[str, np.ndarray], protocol: dict,
) -> dict:
    directory = output / method
    directory.mkdir(parents=True, exist_ok=True)
    summary = _summarize(values, protocol)
    np.savez_compressed(directory / "predictions.npz", **values)
    np.savez_compressed(directory / "predictions_per_frame.npz", **values)
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    completion = {
        "method": method,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "videos": int(summary["videos"]),
        "physical_frames": int(summary["physical_frames"]),
        "pck_hb_all_frames": float(summary["pck_hb_all_frames"]["pck_0.5"]),
        "pck_hb_video_macro": float(summary["pck_hb_video_macro"]["pck_0.5"]),
        "summary": str(directory / "summary.json"),
        "predictions_per_frame": str(directory / "predictions_per_frame.npz"),
    }
    (directory / "evaluation_complete.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8",
    )
    return completion


def main() -> None:
    args = _parser().parse_args()
    source_config, config_source = _load_config(args)
    original_config = deepcopy(source_config)
    config = deepcopy(source_config)
    checkpoint = args.checkpoint.resolve()
    metadata = resolve_project_path(args.metadata).resolve()
    frames_dir = resolve_project_path(args.frames_dir).resolve()
    output = args.output.resolve()
    for path in (checkpoint, metadata):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not frames_dir.is_dir():
        raise FileNotFoundError(frames_dir)
    output.mkdir(parents=True, exist_ok=True)
    preflight = _verify_preflight_subset(config, metadata)

    temporal = config["model"].get("temporal", {})
    trained_steps = int(temporal.get("video_frames", 1))
    if temporal.get("input_strategy") not in {"frames", "scheduled_frames"}:
        raise ValueError("Reset-policy validation requires temporal frame input")
    if trained_steps <= 1:
        raise ValueError("Reset-policy validation requires trained_steps > 1")
    config["data"].update({
        "frames_dir": str(frames_dir.relative_to(PROJECT_ROOT)),
        "frame_layout": "flat",
        "test_metadata": str(metadata.relative_to(PROJECT_ROOT)),
        "exclude_overlapping_clips": False,
        "frame_stride": 1,
        "temporal_frame_gap": 1,
        "minimum_temporal_history": trained_steps - 1,
        "tube_crop_scope": "video",
        "runtime_spatial_crops": False,
        # A complete-video bbox already requires one pose pass per video.
        # Reading the source skeleton keeps this validation independent of
        # optional clip-oriented runtime-cache databases.
        "runtime_cache_mode": "disabled",
        "pose_validation_mode": "skip",
    })
    config.setdefault("training", {})["loss_all_frames"] = True

    device = torch.device(args.device)
    model = build_model(config).to(device)
    checkpoint_metadata = load_model(checkpoint, model, device)
    if not isinstance(getattr(model, "mam", None), MotionAlignedMembraneV2):
        raise TypeError("The selected experiment is not a MAM V2 model")
    dataset = build_dataset(config, "test", args.max_videos)
    spatial_chunks, expected_frames = _select_exhaustive_chunks(dataset)
    expected_videos = len(dataset.samples)
    batch_size = args.batch_size or int(
        config["training"].get("clip_batch_size", 8)
    )
    batch_sampler = SameVideoChunkBatchSampler(dataset.frame_index, batch_size)
    loader = DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=args.workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0,
    )

    common_protocol = {
        "name": "mam_reset_policy_fullvideo_v1",
        "physical_frames_scored_once": True,
        "trained_sequence_length": trained_steps,
        "tube_crop_scope": "video",
        "spatial_heatmaps_shared_across_methods": True,
        "spatial_batching_does_not_reset_mam": True,
        "tail_policy": "overlap final trained-length window only for window_reset",
    }
    provenance = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(checkpoint_metadata["epoch"]),
        "checkpoint_sha256": _sha256(checkpoint),
        "split": "fullvideo_xsub35_test",
        "split_metadata": str(metadata),
        "split_metadata_sha256": _sha256(metadata),
        "frames_dir": str(frames_dir),
        "max_videos": args.max_videos,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]),
        "config_source": config_source,
        "resolved_training_config": original_config,
        "effective_evaluation_config": config,
        "pose_preflight_subset_check": preflight,
        "protocol": common_protocol,
        "videos": expected_videos,
        "expected_physical_frames": expected_frames,
        "spatial_chunks": spatial_chunks,
        "spatial_batches": len(batch_sampler),
    }
    (output / "evaluation_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8",
    )
    print(
        f"reset_policy_start videos={expected_videos} frames={expected_frames} "
        f"spatial_chunks={spatial_chunks} spatial_batches={len(batch_sampler)}",
        flush=True,
    )
    values, invocation_counts = _collect_predictions(
        model, _ProgressLoader(loader, args.progress_interval), config, device,
        trained_steps,
    )
    for method in METHODS:
        _validate_values(values[method], expected_frames, expected_videos)
    identity_keys = (
        "target", "visibility", "scale", "scale_hb", "sample_id", "frame_index",
        "person_id", "video_id", "clip_id", "frame_position_in_clip",
    )
    for key in identity_keys:
        left = values["window_reset"][key]
        right = values["video_reset"][key]
        equal = (
            np.array_equal(left, right, equal_nan=True)
            if np.issubdtype(left.dtype, np.number)
            else np.array_equal(left, right)
        )
        if not equal:
            raise RuntimeError(f"Reset-policy outputs are misaligned at {key}")
    expected_window_invocations = sum(
        len(trained_window_starts(len(indices), trained_steps))
        for indices in sequence_groups(values["window_reset"])
    )
    if invocation_counts != {
        "window_reset": expected_window_invocations,
        "video_reset": expected_videos,
    }:
        raise RuntimeError(
            f"Unexpected MAM invocation counts: {invocation_counts}; "
            f"expected window={expected_window_invocations} video={expected_videos}"
        )

    protocols = {
        "raw": {**common_protocol, "state_reset": "not_applicable"},
        "window_reset": {
            **common_protocol,
            "state_reset": "every_trained_length_invocation",
            "mam_invocations": invocation_counts["window_reset"],
        },
        "video_reset": {
            **common_protocol,
            "state_reset": "video_boundary_only",
            "mam_invocations": invocation_counts["video_reset"],
        },
    }
    completions = {
        method: _write_method(output, method, values[method], protocols[method])
        for method in METHODS
    }
    comparison = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "invocation_counts": invocation_counts,
        "paired_video_comparison": _paired_comparison(
            values["window_reset"], values["video_reset"], args.bootstrap_samples,
        ),
        "trained_window_stitch_transitions": {
            method: _stitch_transition_summary(values[method], trained_steps)
            for method in ("window_reset", "video_reset")
        },
        "methods": completions,
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
