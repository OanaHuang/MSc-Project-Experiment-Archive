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
    parser = argparse.ArgumentParser(
        description="Recompute the complete NTU temporal metric family from canonical matrices",
    )
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--extracted-frames-dir", default="Datasets/NTU_RGBD/extracted_frames_full",
    )
    parser.add_argument(
        "--validation-metadata", default="Datasets/NTU_RGBD/metadata/s010/val_split.csv",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    reports = []
    for matrices_path in sorted(root.rglob("metrics/prediction_matrices.npz")):
        run_dir = matrices_path.parent.parent
        status_path = run_dir / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") != "completed":
                continue
        arrays = np.load(matrices_path)
        required = {"pred", "gt", "visibility", "head_length"}
        if not required.issubset(arrays.files):
            continue
        if {"sample_ids", "frame_indices"}.issubset(arrays.files):
            sample_ids, frame_indices = arrays["sample_ids"], arrays["frame_indices"]
        else:
            checkpoint_path = run_dir / "checkpoints/best.pt"
            if not checkpoint_path.is_file():
                continue
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            config = original_video_config(
                checkpoint["config"], args.extracted_frames_dir, args.validation_metadata,
            )
            dataset = build_dataset(PROJECT_ROOT, config, "validation")
            if len(arrays["pred"]) != len(dataset.frame_index):
                print(f"SKIP prediction rows do not match dataset for {run_dir}", file=sys.stderr)
                continue
            sample_ids = np.asarray([
                str(dataset.samples[sample]["sample_id"]) for sample, _ in dataset.frame_index
            ])
            frame_indices = np.asarray([frame for _, frame in dataset.frame_index])
        summary = compute_nacce(
            arrays["pred"], arrays["gt"], arrays["visibility"], arrays["head_length"],
            sample_ids, frame_indices, NTU_JOINT_NAMES,
        )
        pckhn_path = run_dir / "metrics/pckhn_summary.json"
        pckhn = (json.loads(pckhn_path.read_text(encoding="utf-8"))
                 if pckhn_path.is_file() else {})
        summary.update({
            "experiment_id": pckhn.get("experiment_id"),
            "model_name": pckhn.get("model_name"),
            "source_predictions": str(matrices_path),
        })
        save_nacce(summary, run_dir / "metrics")
        reports.append({
            "run": str(run_dir.relative_to(root)), "acce": summary["acce"],
            "nacce": summary["nacce"], "valid_joint_triplets": summary["valid_joint_triplets"],
        })
    print(json.dumps({"updated": len(reports), "runs": reports}, indent=2))


if __name__ == "__main__":
    main()
