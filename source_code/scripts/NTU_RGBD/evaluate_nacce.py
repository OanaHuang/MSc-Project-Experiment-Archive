from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.NTU_RGBD.core.config import NTU_JOINT_NAMES
from scripts.NTU_RGBD.datasets import build_dataset
from scripts.NTU_RGBD.evaluate_original_videos import original_video_config
from scripts.NTU_RGBD.evaluation.temporal_metrics import compute_nacce, save_nacce


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute NAccE from complete original-video predictions")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--extracted-frames-dir", default="Datasets/NTU_RGBD/extracted_frames_full")
    parser.add_argument("--validation-metadata", default="Datasets/NTU_RGBD/metadata/s010/val_split.csv")
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else PROJECT_ROOT / args.run
    checkpoint_path = run / "checkpoints/best.pt"
    matrices_path = run / "metrics/prediction_matrices.npz"
    if not checkpoint_path.is_file() or not matrices_path.is_file():
        raise FileNotFoundError("run requires best.pt and canonical complete-frame prediction_matrices.npz")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = original_video_config(checkpoint["config"], args.extracted_frames_dir, args.validation_metadata)
    dataset = build_dataset(PROJECT_ROOT, config, "validation")
    arrays = np.load(matrices_path)
    if len(arrays["pred"]) != len(dataset.frame_index):
        raise ValueError("prediction rows do not match the complete validation dataset")
    sample_ids = np.asarray([str(dataset.samples[sample]["sample_id"]) for sample, _ in dataset.frame_index])
    frame_indices = np.asarray([frame for _, frame in dataset.frame_index])
    summary = compute_nacce(arrays["pred"], arrays["gt"], arrays["visibility"], arrays["head_length"],
                            sample_ids, frame_indices, NTU_JOINT_NAMES)
    summary.update({"experiment_id": config["id"], "model_name": config["name"],
                    "source_predictions": str(matrices_path.relative_to(PROJECT_ROOT))})
    save_nacce(summary, run / "metrics_nacce")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
