from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.MPII.core.geometry import heatmaps_to_keypoints
from scripts.NTU_RGBD.datasets import build_dataset
from scripts.NTU_RGBD.visualization.skeleton import NTU_EDGES
from scripts.spikepose.models import build_model
from scripts.spikepose.training import load_model


PREDICTION_COLOR = (0, 255, 0)
GROUND_TRUTH_COLOR = (255, 100, 0)


def select_sample_ids(dataset, count: int, seed: int) -> list[str]:
    available = sorted({str(sample["sample_id"]) for sample in dataset.samples})
    if count < 1:
        raise ValueError("video count must be positive")
    if count > len(available):
        raise ValueError(f"Requested {count} videos, but only {len(available)} exist")
    return random.Random(seed).sample(available, count)


def draw_pose(frame: np.ndarray, points: np.ndarray, visible: np.ndarray,
              color: tuple[int, int, int], thickness: int) -> None:
    for left, right in NTU_EDGES:
        if visible[left] and visible[right]:
            start = tuple(np.rint(points[left]).astype(int))
            end = tuple(np.rint(points[right]).astype(int))
            cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)
    for point, is_visible in zip(points, visible):
        if is_visible:
            cv2.circle(
                frame, tuple(np.rint(point).astype(int)),
                max(2, thickness + 1), color, -1, cv2.LINE_AA,
            )


def crop_to_original(points: np.ndarray, bbox: np.ndarray,
                     image_size: int) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    restored = np.asarray(points, dtype=np.float32).copy()
    restored[:, 0] = x1 + restored[:, 0] * (x2 - x1) / image_size
    restored[:, 1] = y1 + restored[:, 1] * (y2 - y1) / image_size
    return restored


@torch.no_grad()
def render_video(model, dataset, device: torch.device, sample_id: str,
                 output_path: Path, max_frames: int, fps: float,
                 line_thickness: int = 1) -> dict:
    indices = [
        index for index, (sample_index, _) in enumerate(dataset.frame_index)
        if str(dataset.samples[sample_index]["sample_id"]) == sample_id
    ][:max_frames]
    if not indices:
        raise ValueError(f"No frames found for sample {sample_id}")
    first = dataset[indices[0]]
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
    try:
        for dataset_index in indices:
            item = dataset[dataset_index]
            frame = cv2.imread(item["rgb_path"], cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Could not read {item['rgb_path']}")
            heatmap = model(item["image"].unsqueeze(0).to(device))[0].cpu().numpy()
            prediction, confidence = heatmaps_to_keypoints(
                heatmap, dataset.image_size, method="argmax",
            )
            if frame.shape[:2] != (dataset.image_size, dataset.image_size):
                person_bbox = item["person_bbox"].numpy()
                bbox = np.rint(person_bbox).astype(int)
                cv2.rectangle(
                    frame, tuple(bbox[:2]), tuple(bbox[2:]),
                    (0, 220, 255), 2, cv2.LINE_AA,
                )
                prediction = crop_to_original(
                    prediction, person_bbox, dataset.image_size,
                )
                ground_truth = item["original_keypoints"].numpy()
                visibility = item["original_visibility"].numpy() > 0
            else:
                ground_truth = item["keypoints"].numpy()
                visibility = item["visibility"].numpy() > 0
            draw_pose(
                frame, ground_truth, visibility,
                GROUND_TRUTH_COLOR, line_thickness,
            )
            draw_pose(
                frame, prediction, confidence > 0,
                PREDICTION_COLOR, line_thickness,
            )
            cv2.putText(
                frame,
                f"{model.model_name} | {sample_id} | frame {item['frame_index']}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame, "prediction=green  ground truth=blue  crop=yellow", (8, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()
    return {
        "sample_id": sample_id, "frames": len(indices), "fps": fps,
        "path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render native NTU pose prediction videos from a checkpoint",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--video-count", type=int, default=5)
    parser.add_argument("--selection-seed", type=int, default=20260811)
    parser.add_argument("--max-video-frames", type=int, default=120)
    parser.add_argument("--video-fps", type=float, default=5.0)
    parser.add_argument(
        "--line-thickness", type=int, default=1,
        help="Prediction and ground-truth pose line thickness in pixels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_video_frames < 1 or args.video_fps <= 0:
        raise ValueError("max-video-frames and video-fps must be positive")
    if args.line_thickness not in range(1, 9):
        raise ValueError("line-thickness must be between 1 and 8")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = build_model(config).to(device)
    loaded = load_model(args.checkpoint, model, device)
    dataset = build_dataset(PROJECT_ROOT, config, "validation")
    sample_ids = select_sample_ids(dataset, args.video_count, args.selection_seed)
    videos = [
        render_video(
            model, dataset, device, sample_id,
            args.output_dir / "videos" / f"{config['id']}_prediction_{sample_id}.mp4",
            args.max_video_frames, args.video_fps, args.line_thickness,
        )
        for sample_id in sample_ids
    ]
    summary = {
        "experiment_id": config["id"], "model_name": config["name"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(loaded.get("epoch", -1)),
        "overlay_style": {
            "prediction": "green", "ground_truth": "blue",
            "line_thickness_px": args.line_thickness,
        },
        "selection_seed": args.selection_seed, "sample_ids": sample_ids,
        "videos": videos,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "video_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
