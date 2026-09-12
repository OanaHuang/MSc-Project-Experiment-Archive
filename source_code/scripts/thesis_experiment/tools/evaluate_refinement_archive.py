#!/usr/bin/env python3
"""Apply a frozen learned refiner to a documented prediction archive."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import yaml

from spikepose_thesis.core.paths import PROJECT_ROOT
from spikepose_thesis.refinement.jtr import JointwiseTemporalRefinement
from spikepose_thesis.refinement.runner import _apply_module, _load, _summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a frozen learned temporal refiner to saved predictions.",
    )
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command: list[str]) -> str:
    return subprocess.run(
        ["git", *command], cwd=PROJECT_ROOT, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()


def main() -> None:
    args = _parser().parse_args()
    resolved_config = args.resolved_config.resolve()
    checkpoint = args.checkpoint.resolve()
    source = args.source.resolve()
    output = args.output.resolve()
    for path in (resolved_config, checkpoint, source):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    if config.get("refinement", {}).get("method") != "jtr_jointwise":
        raise ValueError("Only jtr_jointwise checkpoints are supported")
    protocol = config["refinement_protocol"]
    refinement = config["refinement"]
    device = torch.device(args.device)
    module = JointwiseTemporalRefinement(
        int(refinement["joints"]), jointwise=True,
        alpha_min=float(protocol["alpha_min"]),
        alpha_max=float(protocol["alpha_max"]),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    module.load_state_dict(payload["model_state_dict"], strict=True)
    module.eval()
    if any(not torch.isfinite(item).all() for item in module.parameters()):
        raise FloatingPointError("refinement checkpoint contains non-finite parameters")

    started_at = datetime.now(timezone.utc).isoformat()
    values = _load(source)
    refined = _apply_module(
        module, values, device, float(protocol["coordinate_scale"]),
    )
    if not np.isfinite(refined["prediction"]).all():
        raise FloatingPointError("refinement output contains non-finite predictions")
    summary = _summary(refined)

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions_per_frame.npz", **refined)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    parameters = {
        "alpha": module.alpha.detach().cpu().tolist(),
        "coordinate_scale": float(protocol["coordinate_scale"]),
    }
    (output / "parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8",
    )
    provenance = {
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "experiment": config["id"],
        "paper_id": config["paper_id"],
        "resolved_config": str(resolved_config),
        "resolved_config_sha256": _sha256(resolved_config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source": str(source),
        "source_sha256": _sha256(source),
        "output": str(output),
        "sampled_frames": int(len(refined["prediction"])),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]),
    }
    (output / "evaluation_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
