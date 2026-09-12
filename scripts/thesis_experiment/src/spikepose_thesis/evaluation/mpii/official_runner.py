"""Reproduce released MPII baselines under one frozen official protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from spikepose_thesis.core.paths import default_output_root, resolve_project_path
from spikepose_thesis.data.mpii.datasets import OfficialMPIIDataset
from spikepose_thesis.models import build_model
from spikepose_thesis.models.baselines import load_official_checkpoint

from .official_metrics import evaluate_official


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_official_mpii_baseline(
    config: dict, device: str | torch.device, *, batch_size: int | None = None,
    num_workers: int | None = None, max_samples: int | None = None,
    output_root: Path | None = None,
) -> dict:
    if config.get("action") != "evaluate_official_checkpoint":
        raise ValueError(
            f"{config['id']} is not an official-checkpoint evaluation experiment"
        )
    initialization = config["initialization"]
    checkpoint = resolve_project_path(initialization["path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Official checkpoint not found: {checkpoint}")
    data = config["data"]
    if data.get("split_protocol") != "official_hrnet_mpii":
        raise ValueError("Official baselines require split_protocol=official_hrnet_mpii")
    dataset = OfficialMPIIDataset(
        resolve_project_path(data["validation_metadata"]),
        resolve_project_path(data["images_dir"]),
        data["image_size"], data["heatmap_size"], data["sigma"],
        training=False, max_samples=max_samples,
        color_rgb=bool(data.get("color_rgb", True)),
    )
    workers = int(num_workers if num_workers is not None else config["training"]["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size or config["training"]["batch_size"]),
        shuffle=False, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
    )
    device_value = torch.device(device)
    model = build_model(config).to(device_value)
    model.experiment_id = config["id"]
    model.model_name = config["paper_id"]
    load_report = load_official_checkpoint(checkpoint, model, device_value)
    expected_tensor_hash = initialization.get("tensor_sha256")
    if expected_tensor_hash and load_report["tensor_sha256"] != expected_tensor_hash:
        raise RuntimeError(
            "Official checkpoint tensor fingerprint mismatch: "
            f"expected={expected_tensor_hash}, actual={load_report['tensor_sha256']}"
        )
    root = Path(output_root) if output_root else default_output_root("formal")
    output = root / "mpii_official_baselines" / config["id"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8",
    )
    summary = evaluate_official(
        model, loader, dataset, device_value, output / "validation",
        flip_test=bool(config["evaluation"].get("flip_test", True)),
        flip_shift=bool(config["evaluation"].get("flip_shift", True)),
    )
    status = {
        "status": "completed",
        "checkpoint": str(checkpoint),
        "checkpoint_file_sha256": _file_sha256(checkpoint),
        "checkpoint_load": load_report,
        "official_reported_pckh": initialization.get("reported_pckh"),
        "metrics": summary,
    }
    (output / "status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8",
    )
    return status
