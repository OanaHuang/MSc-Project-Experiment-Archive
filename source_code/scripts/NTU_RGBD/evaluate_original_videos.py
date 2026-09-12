from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.NTU_RGBD.datasets import build_dataset
from scripts.NTU_RGBD.core.config import NTU_JOINT_NAMES
from scripts.NTU_RGBD.evaluation import (
    compute_extended_pose_metrics, compute_nacce, evaluate,
    save_extended_pose_metrics, save_nacce,
)
from scripts.NTU_RGBD.render_prediction_videos import (
    render_video, select_sample_ids,
)
from scripts.spikepose.models import build_model
from scripts.spikepose.training import load_model


def original_video_config(config: dict, extracted_frames_dir: str,
                          validation_metadata: str) -> dict:
    resolved = deepcopy(config)
    data = resolved["data"]
    resolved["data"].update({
        "extracted_frames_dir": extracted_frames_dir,
        "validation_metadata": validation_metadata,
        "frame_stride": 1,
        # Preserve the trained temporal sampling. In particular, G-series
        # checkpoints must not silently fall back from gaps 2/3/5 to gap 1.
        "temporal_frame_gap": int(data.get("temporal_frame_gap", 1)),
        "minimum_temporal_history": int(data.get("minimum_temporal_history", 3)),
        "preprocessed_pose_cache": False,
    })
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an NTU checkpoint on complete original-video frames",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--extracted-frames-dir", default="Datasets/NTU_RGBD/extracted_frames_full",
    )
    parser.add_argument(
        "--validation-metadata", default="Datasets/NTU_RGBD/metadata/s010/val_split.csv",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-validation-videos", type=int)
    parser.add_argument("--video-count", type=int, default=5)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument(
        "--videos-only", action="store_true",
        help="Render full-frame videos without recomputing or replacing metrics",
    )
    parser.add_argument("--selection-seed", type=int, default=20260811)
    parser.add_argument("--max-video-frames", type=int, default=120)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument(
        "--line-thickness", type=int, default=1,
        help="Prediction and ground-truth pose line thickness in pixels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    if args.line_thickness not in range(1, 9):
        raise ValueError("line-thickness must be between 1 and 8")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = original_video_config(
        checkpoint["config"], args.extracted_frames_dir, args.validation_metadata,
    )
    device = torch.device(args.device)
    model = build_model(config).to(device)
    loaded = load_model(args.checkpoint, model, device)
    dataset = build_dataset(
        PROJECT_ROOT, config, "validation", args.max_validation_videos,
    )
    metrics = temporal = extended = None
    if not args.videos_only:
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        metrics = evaluate(
            model, loader, dataset, device, args.output_dir / "metrics",
        )
        matrices_path = args.output_dir / "metrics/prediction_matrices.npz"
        arrays = np.load(matrices_path)
        extended = compute_extended_pose_metrics(
            arrays["pred"], arrays["gt"], arrays["visibility"],
            arrays["head_length"],
        )
        save_extended_pose_metrics(extended, args.output_dir / "metrics")
        temporal = compute_nacce(
            arrays["pred"], arrays["gt"], arrays["visibility"], arrays["head_length"],
            arrays["sample_ids"], arrays["frame_indices"], NTU_JOINT_NAMES,
        )
        temporal.update({
            "experiment_id": config["id"], "model_name": config["name"],
            "source_predictions": str(matrices_path),
        })
        save_nacce(temporal, args.output_dir / "metrics")
    sample_ids = ([] if args.skip_videos else
                  select_sample_ids(dataset, args.video_count, args.selection_seed))
    videos = ([] if args.skip_videos else [
        render_video(
            model, dataset, device, sample_id,
            args.output_dir / "videos_original_video"
            / f"{config['id']}_prediction_{sample_id}.mp4",
            args.max_video_frames, args.video_fps, args.line_thickness,
        )
        for sample_id in sample_ids
    ])
    summary = {
        "experiment_id": config["id"], "model_name": config["name"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(loaded.get("epoch", -1)),
        "evaluation_source": "complete original-video frames",
        "canonical_metric_directory": "metrics",
        "frame_stride": 1,
        "temporal_frame_gap": config["data"]["temporal_frame_gap"],
        "minimum_temporal_history": config["data"]["minimum_temporal_history"],
        "overlay_style": {
            "prediction": "green", "ground_truth": "blue",
            "line_thickness_px": args.line_thickness,
        },
        "validation_videos": len(dataset.samples),
        "metrics": metrics, "extended_pose_metrics": extended,
        "temporal_metrics": temporal,
        "selection_seed": args.selection_seed,
        "sample_ids": sample_ids, "videos": videos,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_name = (
        "original_video_manifest.json" if args.videos_only else
        "original_video_validation.json"
    )
    (args.output_dir / summary_name).write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
