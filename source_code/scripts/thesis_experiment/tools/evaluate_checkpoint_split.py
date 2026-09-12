#!/usr/bin/env python3
"""Evaluate an explicit checkpoint on a requested dataset split.

This utility is intended for documented evaluation of a frozen checkpoint. It
does not impose a one-time test lock; callers can use a new output directory for
each technical rerun. The regular training CLI keeps its stricter Pilot guard.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.core.paths import PROJECT_ROOT, resolve_project_path
from spikepose_thesis.data import build_dataset
from spikepose_thesis.data.ntu.core.joint_mapping import MPII16_JOINT_NAMES
from spikepose_thesis.evaluation.runner import evaluate_checkpoint
from spikepose_thesis.models import build_model
from spikepose_thesis.training.checkpoint import load_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an explicit frozen checkpoint on validation or test.",
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--resolved-config", type=Path,
        help=(
            "Optional resolved_config.yaml saved with the run. Use this when "
            "the checkpoint was trained with an explicit runtime variant."
        ),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int)
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


def _full_frame_pck(path: Path) -> dict:
    with np.load(path) as archive:
        prediction = archive["prediction"]
        target = archive["target"]
        visibility = archive["visibility"]
        scale = archive["scale_hb"]
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


def main() -> None:
    args = _parser().parse_args()
    if args.resolved_config is None:
        config = load_experiment(args.experiment)
        config_source = "experiment_registry"
    else:
        resolved_config = args.resolved_config.resolve()
        if not resolved_config.is_file():
            raise FileNotFoundError(resolved_config)
        config = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
        if config.get("id") != args.experiment:
            raise ValueError(
                f"Resolved config id {config.get('id')!r} does not match "
                f"--experiment {args.experiment!r}"
            )
        config_source = str(resolved_config)
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    metadata_key = (
        "validation_metadata" if args.split == "validation" else "test_metadata"
    )
    split_metadata = resolve_project_path(config["data"][metadata_key]).resolve()
    if not split_metadata.is_file():
        raise FileNotFoundError(split_metadata)
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = build_model(config).to(device)
    checkpoint_metadata = load_model(checkpoint, model, device)
    dataset = build_dataset(config, args.split, args.max_samples)
    workers = int(config["training"]["num_workers"])
    batch_size = int(
        config["training"].get(
            "clip_batch_size", config["training"]["batch_size"],
        )
        if getattr(dataset, "temporal_steps", 1) > 1
        else config["training"]["batch_size"]
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
    )

    provenance = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(checkpoint_metadata["epoch"]),
        "checkpoint_sha256": _sha256(checkpoint),
        "split": args.split,
        "split_metadata": str(split_metadata),
        "split_metadata_sha256": _sha256(split_metadata),
        "max_samples": args.max_samples,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]),
        "config_source": config_source,
        "resolved_config": config,
    }
    (output / "evaluation_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8",
    )

    summary = evaluate_checkpoint(model, loader, config, device, output)
    per_frame_path = output / "predictions_per_frame.npz"
    spatial_archive = (
        per_frame_path if per_frame_path.is_file() else output / "predictions.npz"
    )
    summary["pck_hb_all_frames"] = _full_frame_pck(spatial_archive)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    completion = {
        **{key: provenance[key] for key in (
            "experiment", "seed", "checkpoint", "checkpoint_epoch",
            "checkpoint_sha256", "split", "split_metadata_sha256",
        )},
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "samples": int(summary["samples"]),
        "sampled_frames": int(summary["pck_hb_all_frames"]["samples"]),
        "summary": str(output / "summary.json"),
        "predictions": str(output / "predictions.npz"),
        "predictions_per_frame": str(output / "predictions_per_frame.npz"),
    }
    (output / "evaluation_complete.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8",
    )
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
