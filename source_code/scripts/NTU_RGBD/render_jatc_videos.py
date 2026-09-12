from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.NTU_RGBD.datasets import build_dataset
from scripts.NTU_RGBD.evaluate_original_videos import original_video_config
from scripts.NTU_RGBD.render_prediction_videos import (
    GROUND_TRUTH_COLOR,
    PREDICTION_COLOR,
    draw_pose,
)


JOINT_TIMESTEP_SAMPLE_IDS = (
    "S010C001P008R001A023",
    "S010C001P016R001A041",
    "S010C002P013R001A037",
    "S010C001P018R001A025",
    "S010C003P013R002A004",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render NTU JATC predictions on the joint-timestep video set",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--matrices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True, choices=("jp0", "jp1", "jp2", "jp3"))
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--extracted-frames-dir", default="Datasets/NTU_RGBD/extracted_frames_full")
    parser.add_argument("--validation-metadata", default="Datasets/NTU_RGBD/metadata/s010/val_split.csv")
    parser.add_argument("--max-video-frames", type=int, default=120)
    parser.add_argument("--video-fps", type=float, default=15.0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = original_video_config(
        checkpoint["config"], args.extracted_frames_dir, args.validation_metadata,
    )
    dataset = build_dataset(PROJECT_ROOT, config, "validation")
    arrays = np.load(args.matrices)
    prediction = arrays["pred"]
    sample_ids = arrays["sample_ids"].astype(str)
    frame_indices = arrays["frame_indices"].astype(np.int64)
    lookup = {
        (sample_id, int(frame)): row
        for row, (sample_id, frame) in enumerate(zip(sample_ids, frame_indices))
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos = []
    for sample_id in JOINT_TIMESTEP_SAMPLE_IDS:
        indices = [
            index
            for index, (sample_index, _) in enumerate(dataset.frame_index)
            if str(dataset.samples[sample_index]["sample_id"]) == sample_id
        ][: args.max_video_frames]
        if not indices:
            raise ValueError(f"No validation frames found for {sample_id}")
        first = dataset[indices[0]]
        first_frame = cv2.imread(first["rgb_path"], cv2.IMREAD_COLOR)
        if first_frame is None:
            raise RuntimeError(f"Could not read {first['rgb_path']}")
        height, width = first_frame.shape[:2]
        output_path = args.output_dir / f"{args.experiment_id}_prediction_{sample_id}.mp4"
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
            args.video_fps, (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open {output_path}")
        try:
            for dataset_index in indices:
                item = dataset[dataset_index]
                frame = cv2.imread(item["rgb_path"], cv2.IMREAD_COLOR)
                frame_index = int(item["frame_index"])
                row = lookup[(sample_id, frame_index)]
                visible = arrays["visibility"][row] > 0
                bbox = np.rint(item["person_bbox"].numpy()).astype(int)
                cv2.rectangle(
                    frame, tuple(bbox[:2]), tuple(bbox[2:]),
                    (0, 220, 255), 2, cv2.LINE_AA,
                )
                draw_pose(frame, arrays["gt"][row], visible, GROUND_TRUTH_COLOR, 2)
                draw_pose(frame, prediction[row], visible, PREDICTION_COLOR, 3)
                cv2.putText(
                    frame, f"{args.method_label} | {sample_id} | frame {frame_index}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, "prediction=green  ground truth=blue  crop=yellow",
                    (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )
                writer.write(frame)
        finally:
            writer.release()
        videos.append({
            "sample_id": sample_id,
            "frames": len(indices),
            "fps": args.video_fps,
            "path": str(output_path),
        })

    manifest = {
        "experiment_id": args.experiment_id,
        "series": "JATC",
        "source_experiment": "t0",
        "method": args.method_label,
        "evaluation_source": "complete original-video frames",
        "selection_contract": "joint_timestep fixed sample set",
        "selection_seed": 20260811,
        "sample_ids": list(JOINT_TIMESTEP_SAMPLE_IDS),
        "max_video_frames": args.max_video_frames,
        "video_fps": args.video_fps,
        "videos": videos,
    }
    (args.output_dir.parent / "original_video_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
