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
from scripts.NTU_RGBD.evaluation import compute_nacce, evaluate, save_nacce
from scripts.NTU_RGBD.core.config import NTU_JOINT_NAMES
from scripts.spikepose.models import build_model
from scripts.spikepose.training import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export contiguous F-series coordinate sequences")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--extracted-frames-dir", default="Datasets/NTU_RGBD/extracted_frames_full")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = deepcopy(checkpoint["config"])
    config["data"].update({
        "extracted_frames_dir": args.extracted_frames_dir,
        "validation_metadata": args.metadata,
        "frame_stride": 1, "preprocessed_pose_cache": False,
    })
    device = torch.device(args.device)
    model = build_model(config).to(device)
    load_model(args.checkpoint, model, device)
    dataset = build_dataset(PROJECT_ROOT, config, "validation")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    summary = evaluate(model, loader, dataset, device, args.output_dir)
    arrays = np.load(args.output_dir / "prediction_matrices.npz")
    temporal = compute_nacce(
        arrays["pred"], arrays["gt"], arrays["visibility"], arrays["head_length"],
        arrays["sample_ids"], arrays["frame_indices"], NTU_JOINT_NAMES,
    )
    temporal.update({"experiment_id": config["id"], "model_name": config["name"]})
    save_nacce(temporal, args.output_dir / "temporal_metrics")
    print(json.dumps({"pckhn": summary["pckhn"], **temporal}, indent=2))


if __name__ == "__main__":
    main()
