#!/usr/bin/env python3
"""Evaluate one frozen NTU checkpoint on complete test videos.

Frame-scheduled models use a causal sliding window with stride one.  The
prediction for a physical frame is always the final readout of its window.
At the beginning of a video the context grows from one frame to the configured
window length, so every physical frame is retained without treating
intermediate readouts as independently supervised predictions.
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
from spikepose_thesis.data.ntu.core.joint_mapping import MPII16_JOINT_NAMES
from spikepose_thesis.evaluation.runner import (
    _ntu_group_summaries,
    collect_predictions,
    summarize_predictions,
)
from spikepose_thesis.evaluation.temporal import sequence_groups, temporal_summary
from spikepose_thesis.models import build_model
from spikepose_thesis.training.checkpoint import load_model


DEFAULT_METADATA = "Datasets/NTU_RGBD/metadata/fullvideo_xsub35/test_split.csv"
DEFAULT_FRAMES = "Datasets/NTU_RGBD/extracted_frames_full"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen NTU model on the exhaustive full-video subset.",
    )
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
    parser.add_argument("--progress-interval", type=int, default=250)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", *command], cwd=PROJECT_ROOT, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return result.stdout.strip()


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
                remaining = (total - index) / rate if rate > 0 else float("nan")
                print(
                    f"progress batches={index}/{total} "
                    f"elapsed_min={elapsed / 60:.1f} eta_min={remaining / 60:.1f}",
                    flush=True,
                )
            yield batch


def _prefix_config(config: dict, prefix_frames: int) -> dict:
    """Return a compatible shorter-context config for video warm-up."""
    result = deepcopy(config)
    temporal = result["model"]["temporal"]
    full_frames = int(temporal.get("video_frames", 1))
    if not 1 <= prefix_frames <= full_frames:
        raise ValueError(
            f"prefix_frames must lie in [1, {full_frames}], got {prefix_frames}"
        )
    schedule = tuple(int(item) for item in temporal.get("update_schedule", ()))
    prefix_schedule = tuple(item for item in schedule if item < prefix_frames)
    if sorted(set(prefix_schedule)) != list(range(prefix_frames)):
        raise ValueError(
            "Each physical prefix frame must occur in the truncated update schedule"
        )
    temporal["video_frames"] = prefix_frames
    temporal["update_schedule"] = list(prefix_schedule)
    result["model"]["num_steps"] = len(prefix_schedule)
    result["data"]["minimum_temporal_history"] = 0
    if isinstance(result.get("temporal"), dict):
        result["temporal"]["video_frames"] = prefix_frames
        result["temporal"]["update_schedule"] = list(prefix_schedule)
    return result


def _keep_prefix_target(dataset, prefix_frames: int) -> int:
    """Retain only the first target that has ``prefix_frames`` observations."""
    target_position = prefix_frames - 1
    dataset.frame_index = [
        item for item in dataset.frame_index if int(item[3]) == target_position
    ]
    if len(dataset.frame_index) != len(dataset.samples):
        raise RuntimeError(
            f"Expected one position-{target_position} prefix per video, got "
            f"{len(dataset.frame_index)} for {len(dataset.samples)} videos"
        )
    return len(dataset.frame_index)


def _complete_frame_count(dataset) -> int:
    last_position: dict[int, int] = {}
    for sample_index, _frame, _clip, position in dataset.frame_index:
        last_position[int(sample_index)] = max(
            int(position), last_position.get(int(sample_index), -1),
        )
    if len(last_position) != len(dataset.samples):
        raise RuntimeError("Some dataset videos have no complete temporal window")
    return sum(position + 1 for position in last_position.values())


def _merge_prediction_values(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Merge causal prefix and full-window predictions in video/frame order."""
    if not parts:
        raise ValueError("No prediction parts were provided")
    keys = tuple(parts[0])
    if any(tuple(part) != keys for part in parts[1:]):
        raise ValueError("Prediction parts do not share the same fields")
    merged = {
        key: np.concatenate([np.asarray(part[key]) for part in parts], axis=0)
        for key in keys
    }
    order = np.lexsort((
        merged["frame_index"].astype(np.int64),
        merged["person_id"].astype(str),
        merged["video_id"].astype(str),
    ))
    return {key: value[order] for key, value in merged.items()}


def _attach_frame_identity(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {key: np.asarray(value) for key, value in values.items()}
    sample_ids = np.asarray([str(item).split("::", 1)[0] for item in result["sample_id"]])
    result.update({
        "video_id": sample_ids,
        "clip_id": np.asarray(["full"] * len(sample_ids)),
        "frame_position_in_clip": result["frame_index"].astype(np.int64),
    })
    return result


def _validate_exhaustive_values(
    values: dict[str, np.ndarray], expected_frames: int, expected_videos: int,
) -> None:
    if len(values["prediction"]) != expected_frames:
        raise RuntimeError(
            f"Expected {expected_frames} unique frames, got "
            f"{len(values['prediction'])}"
        )
    groups = sequence_groups(values)
    if len(groups) != expected_videos:
        raise RuntimeError(f"Expected {expected_videos} videos, got {len(groups)}")
    for indices in groups:
        positions = values["frame_position_in_clip"][indices].astype(np.int64)
        frames = values["frame_index"][indices].astype(np.int64)
        expected = np.arange(len(indices), dtype=np.int64)
        if not np.array_equal(positions, expected) or not np.array_equal(frames, expected):
            video = str(values["video_id"][indices[0]])
            raise RuntimeError(f"Non-exhaustive or unordered frame coverage: {video}")


def _full_frame_pck(values: dict[str, np.ndarray]) -> dict:
    prediction = values["prediction"]
    target = values["target"]
    visibility = values["visibility"]
    scale = values["scale_hb"]
    distance = np.linalg.norm(prediction - target, axis=-1)
    valid = (
        (visibility > 0)
        & np.isfinite(distance)
        & np.isfinite(scale[:, None])
    )
    correct = distance <= 0.5 * scale[:, None]
    return {
        "pck_0.5": float(correct[valid].mean()),
        "samples": int(len(scale)),
        "per_joint": {
            name: {
                "pck_0.5": float(correct[:, joint][valid[:, joint]].mean()),
            }
            for joint, name in enumerate(MPII16_JOINT_NAMES)
        },
    }


def _load_config(args) -> tuple[dict, str]:
    if args.resolved_config is None:
        config = load_experiment(args.experiment)
        return config, "experiment_registry"
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
    """Prove that every subset row belongs to the frozen pose-valid whitelist."""
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


def main() -> None:
    args = _parser().parse_args()
    source_config, config_source = _load_config(args)
    original_config = deepcopy(source_config)
    config = deepcopy(source_config)
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    metadata = resolve_project_path(args.metadata).resolve()
    frames_dir = resolve_project_path(args.frames_dir).resolve()
    for path in (checkpoint, metadata):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not frames_dir.is_dir():
        raise FileNotFoundError(frames_dir)
    output.mkdir(parents=True, exist_ok=True)

    preflight_subset_check = _verify_preflight_subset(config, metadata)

    config["data"].update({
        "frames_dir": str(frames_dir.relative_to(PROJECT_ROOT)),
        "frame_layout": "flat",
        "test_metadata": str(metadata.relative_to(PROJECT_ROOT)),
        "exclude_overlapping_clips": False,
        "frame_stride": 1,
        "temporal_frame_gap": 1,
        "tube_crop_scope": "video",
        "runtime_spatial_crops": False,
        # A complete-video tube requires inspecting the complete source
        # sequence.  Do not mix it with the clip-oriented runtime cache.
        "runtime_cache_mode": "disabled",
        # The frozen manifest hashes the original metadata CSV.  Membership in
        # its valid-sample whitelist is checked explicitly above for this
        # derived subset, so re-running the same skeleton audit is unnecessary.
        "pose_validation_mode": "skip",
    })
    temporal_steps = int(config["model"].get("temporal", {}).get("video_frames", 1))
    temporal_input = config["model"].get("temporal", {}).get("input_strategy") in {
        "frames", "scheduled_frames",
    }
    if temporal_input:
        config["data"]["minimum_temporal_history"] = temporal_steps - 1
    else:
        config["data"]["minimum_temporal_history"] = 0

    device = torch.device(args.device)
    model = build_model(config).to(device)
    checkpoint_metadata = load_model(checkpoint, model, device)
    dataset = build_dataset(config, "test", args.max_videos)
    expected_frames = _complete_frame_count(dataset)
    expected_videos = len(dataset.samples)
    batch_size = args.batch_size or (
        int(config["training"].get("clip_batch_size", 8))
        if temporal_input else int(config["training"].get("batch_size", 32))
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0,
    )
    progress_loader = _ProgressLoader(loader, args.progress_interval)

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
        "pose_preflight_subset_check": preflight_subset_check,
        "evaluation_protocol": {
            "name": "causal_sliding_last_readout_video_crop_v3",
            "physical_frames_scored_once": True,
            "tube_crop_scope": "video",
            "temporal_context_frames": (
                f"causal prefix 1..{temporal_steps}, then fixed {temporal_steps}"
                if temporal_input else 1
            ),
            "chunk_stride": 1,
            "tail_policy": (
                "one final-readout prediction for every physical frame"
                if temporal_input else "not applicable"
            ),
            "video_start_policy": (
                "grow causal context one frame at a time; retain every prefix readout"
                if temporal_input else "not applicable"
            ),
            "state_reset": (
                "recompute each causal window; retain only its final readout"
                if temporal_input
                else "not applicable (frame-wise model)"
            ),
            "spatial_pooling": "micro over all physical frames",
            "video_pooling": "macro over complete selected videos",
        },
        "videos": expected_videos,
        "model_invocations": expected_frames,
        "expected_physical_frames": expected_frames,
    }
    (output / "evaluation_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8",
    )
    print(
        f"evaluation_start experiment={args.experiment} videos={expected_videos} "
        f"physical_frames={expected_frames} invocations={expected_frames} "
        f"batch_size={batch_size} workers={args.workers}",
        flush=True,
    )

    if temporal_input:
        parts = [_attach_frame_identity(
            collect_predictions(model, progress_loader, config, device),
        )]
        del model
        del progress_loader
        del loader
        del dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for prefix_frames in range(1, temporal_steps):
            warmup_config = _prefix_config(config, prefix_frames)
            warmup_model = build_model(warmup_config).to(device)
            load_model(checkpoint, warmup_model, device)
            warmup_dataset = build_dataset(
                warmup_config, "test", args.max_videos,
            )
            _keep_prefix_target(warmup_dataset, prefix_frames)
            warmup_loader = DataLoader(
                warmup_dataset, batch_size=batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                persistent_workers=args.workers > 0,
            )
            print(
                f"warmup_prefix frames={prefix_frames} "
                f"samples={len(warmup_dataset)}",
                flush=True,
            )
            parts.append(_attach_frame_identity(collect_predictions(
                warmup_model,
                _ProgressLoader(warmup_loader, args.progress_interval),
                warmup_config,
                device,
            )))
            del warmup_model
            del warmup_loader
            del warmup_dataset
            if device.type == "cuda":
                torch.cuda.empty_cache()
        values = _merge_prediction_values(parts)
    else:
        values = _attach_frame_identity(
            collect_predictions(model, progress_loader, config, device),
        )
    _validate_exhaustive_values(values, expected_frames, expected_videos)

    np.savez_compressed(output / "predictions.npz", **values)
    np.savez_compressed(output / "predictions_per_frame.npz", **values)
    summary = summarize_predictions(values)
    summary["pck_hb"] = summarize_predictions({**values, "scale": values["scale_hb"]})
    summary["pck_hb_all_frames"] = _full_frame_pck(values)
    summary["pck_hb_video_macro"] = {
        "pck_0.5": float(summary["pck_hb"]["sequence_equal"]["pck_0.5"]),
        "videos": int(summary["pck_hb"]["sequences"]),
    }
    summary["head_bone_calibration_ready"] = True
    summary["temporal"] = temporal_summary(values, strict=False)
    summary["groups"] = _ntu_group_summaries(values)
    summary["protocol"] = provenance["evaluation_protocol"]
    summary["videos"] = expected_videos
    summary["physical_frames"] = expected_frames
    summary["model_invocations"] = expected_frames
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    completion = {
        **{key: provenance[key] for key in (
            "experiment", "seed", "checkpoint", "checkpoint_epoch",
            "checkpoint_sha256", "split", "split_metadata_sha256",
        )},
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "videos": expected_videos,
        "physical_frames": expected_frames,
        "model_invocations": expected_frames,
        "pck_hb_all_frames": summary["pck_hb_all_frames"]["pck_0.5"],
        "pck_hb_video_macro": summary["pck_hb_video_macro"]["pck_0.5"],
        "summary": str(output / "summary.json"),
        "predictions": str(output / "predictions.npz"),
        "predictions_per_frame": str(output / "predictions_per_frame.npz"),
    }
    (output / "evaluation_complete.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8",
    )
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
