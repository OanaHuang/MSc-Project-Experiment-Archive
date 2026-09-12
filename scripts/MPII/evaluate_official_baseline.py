"""Evaluate an official Pose-ResNet/HRNet MPII checkpoint in the local pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.MPII.datasets import OfficialMPIIDataset
from scripts.MPII.evaluation.official_metrics import evaluate_official
from scripts.spikepose.models.baselines import (
    align_official_state_dict, build_hrnet, build_pose_resnet,
    checkpoint_state_dict, tensor_fingerprint,
)


def load_checkpoint(model, path: Path):
    value = torch.load(path, map_location="cpu", weights_only=False)
    raw_state = checkpoint_state_dict(value)
    state = align_official_state_dict(model, raw_state)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )
    return tensor_fingerprint(raw_state)


def build(config):
    model = config["model"]
    if model["family"] == "pose_resnet":
        value = build_pose_resnet(model["depth"], model["num_joints"])
    elif model["family"] == "hrnet":
        value = build_hrnet(model["width"], model["num_joints"])
    else:
        raise ValueError(f"Unknown official baseline family: {model['family']}")
    value.experiment_id = config["id"]
    value.model_name = config["name"]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=("hb0", "hb1", "hb2"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--no-flip-shift", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = PROJECT_ROOT / "scripts/MPII/experiments" / f"{args.experiment}.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or PROJECT_ROOT / config["checkpoint"]
    metadata = PROJECT_ROOT / config["data"]["validation_metadata"]
    images = PROJECT_ROOT / config["data"]["images_dir"]
    if args.dry_run:
        print(json.dumps({
            "config": str(config_path), "checkpoint": str(checkpoint),
            "metadata": str(metadata), "images": str(images),
        }, indent=2))
        return
    dataset = OfficialMPIIDataset(
        metadata, images, config["data"]["image_size"],
        config["data"]["heatmap_size"], config["data"]["sigma"],
        max_samples=args.max_validation_samples,
        color_rgb=config["data"].get("color_rgb", True),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    model = build(config)
    checkpoint_fingerprint = load_checkpoint(model, checkpoint)
    expected_fingerprint = config.get("official_reference", {}).get(
        "checkpoint_tensor_sha256"
    )
    if expected_fingerprint and checkpoint_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "Official checkpoint tensor fingerprint mismatch: "
            f"expected={expected_fingerprint}, actual={checkpoint_fingerprint}"
        )
    device = torch.device(args.device)
    model.to(device)
    output = (PROJECT_ROOT / "Outputs_New/mpii/external_baselines/mpii_hrbase_v1" /
              args.experiment / "official_checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8",
    )
    summary = evaluate_official(
        model, loader, dataset, device, output / "metrics",
        flip_test=not args.no_flip, flip_shift=not args.no_flip_shift,
    )
    (output / "status.json").write_text(json.dumps({
        "status": "completed", "checkpoint": str(checkpoint),
        "checkpoint_tensor_sha256": checkpoint_fingerprint,
        "metrics": summary,
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
