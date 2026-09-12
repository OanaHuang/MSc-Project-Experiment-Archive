from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.MPII.core import MPII_JOINT_NAMES, MPII_SKELETON_EDGES
from scripts.MPII.core.geometry import heatmaps_to_keypoints
from scripts.NTU_RGBD.datasets import build_dataset
from scripts.NTU_RGBD.evaluation.transfer_pckhn import (
    MPII_TO_NTU_JOINTS, TRANSFER_JOINT_NAMES, evaluate_transfer_pckhn,
    predictions_to_ntu_semantics,
)
from scripts.spikepose.experiments import (
    experiment_roots, resolve_config, validate_config,
)
from scripts.spikepose.models import build_model
from scripts.spikepose.training import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an MPII E0 checkpoint on NTU RGB+D and render a prediction video",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-validation-videos", type=int, default=32)
    parser.add_argument("--validation-metadata", type=Path)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--video-count", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--video-sample-id")
    parser.add_argument("--max-video-frames", type=int, default=120)
    parser.add_argument("--video-fps", type=float, default=15.0)
    return parser.parse_args()


def resolve_configs(frame_stride: int) -> tuple[dict, dict]:
    mpii = resolve_config(
        experiment_roots(PROJECT_ROOT, "mpii"), "e0",
        PROJECT_ROOT / "scripts" / "MPII" / "configs" / "task.yaml",
        PROJECT_ROOT / "scripts" / "MPII" / "configs" / "training.yaml", {},
    )
    ntu = resolve_config(
        experiment_roots(PROJECT_ROOT, "ntu_rgbd"), "baseline",
        PROJECT_ROOT / "scripts" / "NTU_RGBD" / "configs" / "task.yaml",
        PROJECT_ROOT / "scripts" / "NTU_RGBD" / "configs" / "training.yaml", {},
    )
    ntu = deepcopy(ntu)
    ntu["data"]["image_size"] = int(mpii["data"]["image_size"])
    ntu["data"]["heatmap_size"] = int(mpii["data"]["heatmap_size"])
    ntu["data"]["frame_stride"] = int(frame_stride)
    validate_config(mpii)
    return mpii, ntu


def select_random_metadata(source: Path, output: Path, count: int,
                           seed: int) -> Path:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not fieldnames:
        raise ValueError(f"Metadata has no header: {source}")
    if len(rows) < count:
        raise ValueError(f"Requested {count} videos, but {source} has {len(rows)}")
    selected = random.Random(seed).sample(rows, count)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    return output


def crop_to_original(points: np.ndarray, bbox: np.ndarray,
                     image_size: int) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    restored = np.asarray(points, dtype=np.float32).copy()
    restored[:, 0] = x1 + restored[:, 0] * (x2 - x1) / image_size
    restored[:, 1] = y1 + restored[:, 1] * (y2 - y1) / image_size
    return restored


def render_prediction_video(model, dataset, device, output_path: Path,
                            requested_sample_id: str | None,
                            max_frames: int, fps: float) -> dict:
    sample_id = requested_sample_id
    if sample_id is None:
        sample_id = str(dataset.samples[0]["sample_id"])
    matching = [
        index for index, (sample_index, _) in enumerate(dataset.frame_index)
        if str(dataset.samples[sample_index]["sample_id"]) == sample_id
    ][:max_frames]
    if not matching:
        raise ValueError(f"No extracted frames found for sample {sample_id}")
    first = dataset[matching[0]]
    first_frame = cv2.imread(first["rgb_path"], cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Could not read {first['rgb_path']}")
    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    model.eval()
    with torch.no_grad():
        for dataset_index in matching:
            item = dataset[dataset_index]
            frame = cv2.imread(item["rgb_path"], cv2.IMREAD_COLOR)
            heatmap = model(item["image"].unsqueeze(0).to(device))[0].cpu().numpy()
            prediction, confidence = heatmaps_to_keypoints(
                heatmap, dataset.image_size, method="argmax",
            )
            prediction = predictions_to_ntu_semantics(prediction)
            prediction = crop_to_original(
                prediction, item["person_bbox"].numpy(), dataset.image_size,
            )
            gt = item["original_keypoints"].numpy()[MPII_TO_NTU_JOINTS]
            visible = item["original_visibility"].numpy()[MPII_TO_NTU_JOINTS] > 0
            bbox = item["person_bbox"].numpy().astype(int)
            cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), (0, 220, 255), 2)
            for left, right in MPII_SKELETON_EDGES:
                if confidence[left] > 0 and confidence[right] > 0:
                    cv2.line(frame, tuple(np.rint(prediction[left]).astype(int)),
                             tuple(np.rint(prediction[right]).astype(int)), (0, 255, 0), 3)
                if visible[left] and visible[right]:
                    cv2.line(frame, tuple(np.rint(gt[left]).astype(int)),
                             tuple(np.rint(gt[right]).astype(int)), (255, 100, 0), 2)
            for joint, point in enumerate(prediction):
                cv2.circle(frame, tuple(np.rint(point).astype(int)), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(joint), tuple(np.rint(point).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(
                frame, f"E0 MPII->NTU | {sample_id} | frame {item['frame_index']}",
                (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
            )
            cv2.putText(frame, "prediction=green  NTU GT=blue  crop=yellow", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            writer.write(frame)
    writer.release()
    return {
        "sample_id": sample_id, "frames": len(matching), "fps": fps,
        "joint_names": list(TRANSFER_JOINT_NAMES), "path": str(output_path),
    }


def main() -> None:
    args = parse_args()
    model_config, ntu_config = resolve_configs(args.frame_stride)
    if args.validation_metadata is not None:
        ntu_config["data"]["validation_metadata"] = str(
            args.validation_metadata.resolve()
        )
    if args.random_seed is not None:
        source = PROJECT_ROOT / ntu_config["data"]["validation_metadata"]
        selected = select_random_metadata(
            source, args.output_dir / "selected_metadata.csv",
            args.max_validation_videos, args.random_seed,
        )
        ntu_config["data"]["validation_metadata"] = str(selected.resolve())
    device = torch.device(args.device)
    model = build_model(model_config).to(device)
    checkpoint = load_model(args.checkpoint, model, device)
    dataset = build_dataset(
        PROJECT_ROOT, ntu_config, "validation", args.max_validation_videos,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    metrics_dir = args.output_dir / "metrics"
    summary = evaluate_transfer_pckhn(model, loader, dataset, device, metrics_dir)
    if args.video_sample_id:
        video_sample_ids = [args.video_sample_id]
    else:
        video_sample_ids = [
            str(sample["sample_id"]) for sample in dataset.samples[:args.video_count]
        ]
    videos = [
        render_prediction_video(
            model, dataset, device,
            args.output_dir / "videos" / f"e0_ntu_prediction_{sample_id}.mp4",
            sample_id, args.max_video_frames, args.video_fps,
        )
        for sample_id in video_sample_ids
    ]
    run_summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "validation_videos": int(args.max_validation_videos),
        "frame_stride": int(args.frame_stride),
        "metrics": summary,
        "videos": videos,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
