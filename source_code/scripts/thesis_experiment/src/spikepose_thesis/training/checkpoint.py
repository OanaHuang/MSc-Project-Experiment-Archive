from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch

from spikepose_thesis.data.ntu.core.joint_mapping import NTU25_TO_MPII16


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int,
                    best_metric: float, config: dict, history: list[dict], *,
                    scaler=None, data_loader_generator=None) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric": best_metric,
        "config": config,
        "history": history,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "data_loader_generator_state": (
            data_loader_generator.get_state()
            if data_loader_generator is not None else None
        ),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }, path)


def _load_checkpoint(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument was introduced.
        return torch.load(path, map_location=device)


def load_checkpoint(path: Path, device):
    return _load_checkpoint(path, device)


def load_model(path: Path, model, device):
    checkpoint = _load_checkpoint(path, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def load_spatial_model(path: Path, model, device) -> tuple[dict, dict]:
    """Load backbone/neck/head weights while leaving temporal memory fresh.

    This is deliberately strict for every non-MAM model key. It supports the
    MAM V2 requirement to reuse a spatial source without importing legacy MAM
    parameters or optimizer state.
    """
    checkpoint = _load_checkpoint(path, device)
    source = checkpoint["model_state_dict"]
    target = model.state_dict()
    spatial_keys = {key for key in target if not key.startswith("mam.")}
    missing_source = sorted(key for key in spatial_keys if key not in source)
    shape_mismatch = sorted(
        key for key in spatial_keys
        if key in source and tuple(source[key].shape) != tuple(target[key].shape)
    )
    if missing_source or shape_mismatch:
        raise RuntimeError(
            "Spatial checkpoint is incompatible: "
            f"missing={missing_source[:8]} shape_mismatch={shape_mismatch[:8]}"
        )
    selected = {key: source[key] for key in spatial_keys}
    incompatible = model.load_state_dict(selected, strict=False)
    unexpected_missing = sorted(
        key for key in incompatible.missing_keys if not key.startswith("mam.")
    )
    if incompatible.unexpected_keys or unexpected_missing:
        raise RuntimeError(
            "Spatial-only load violated strict spatial coverage: "
            f"missing={unexpected_missing} unexpected={incompatible.unexpected_keys}"
        )
    report = {
        "loaded_spatial_keys": len(selected),
        "ignored_source_temporal_keys": len([
            key for key in source if key.startswith("mam.")
        ]),
        "fresh_target_temporal_keys": len([
            key for key in target if key.startswith("mam.")
        ]),
    }
    return checkpoint, report


def load_mpii16_to_ntu25_model(path: Path, model, device) -> tuple[dict, dict]:
    """Warm-start a 25-joint model from a compatible 16-joint checkpoint.

    Shared layers are loaded strictly. The 16 learned output channels are
    transplanted to their corresponding Kinect V2 channels; the nine NTU-only
    channels keep the target model's fresh initialization.
    """
    checkpoint = _load_checkpoint(path, device)
    source = checkpoint["model_state_dict"]
    target = model.state_dict()
    output_keys = {
        "head.output.weight", "head.output.bias",
        "final_layer.weight", "final_layer.bias",
    }
    selected = {}
    remapped = []
    missing = []
    mismatched = []
    target_indices = torch.as_tensor(
        NTU25_TO_MPII16, dtype=torch.long, device=device,
    )
    for key, target_value in target.items():
        source_value = source.get(key)
        if source_value is None:
            missing.append(key)
            continue
        if tuple(source_value.shape) == tuple(target_value.shape):
            selected[key] = source_value
            continue
        can_remap = (
            key in output_keys
            and source_value.ndim == target_value.ndim
            and source_value.shape[0] == 16
            and target_value.shape[0] == 25
            and tuple(source_value.shape[1:]) == tuple(target_value.shape[1:])
        )
        if not can_remap:
            mismatched.append(
                f"{key}:source={tuple(source_value.shape)} target={tuple(target_value.shape)}"
            )
            continue
        value = target_value.clone()
        value.index_copy_(0, target_indices, source_value)
        selected[key] = value
        remapped.append(key)

    unexpected = sorted(
        key for key in source
        if key not in target and not key.startswith("mam.")
    )
    if missing or mismatched or unexpected:
        raise RuntimeError(
            "MPII16-to-NTU25 checkpoint is incompatible: "
            f"missing={missing[:8]} mismatched={mismatched[:8]} "
            f"unexpected={unexpected[:8]}"
        )
    incompatible = model.load_state_dict(selected, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Joint-remapped load did not cover the target model: "
            f"missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    report = {
        "source_joints": 16,
        "target_joints": 25,
        "loaded_same_shape_keys": len(selected) - len(remapped),
        "remapped_output_keys": sorted(remapped),
        "source_to_target_joint_indices": NTU25_TO_MPII16.tolist(),
        "fresh_target_joint_indices": sorted(
            set(range(25)) - set(NTU25_TO_MPII16.tolist())
        ),
        "ignored_source_temporal_keys": len([
            key for key in source if key.startswith("mam.") and key not in target
        ]),
    }
    return checkpoint, report


def restore_training_state(path: Path, model, optimizer, scheduler, device, *,
                           scaler=None, data_loader_generator=None) -> dict:
    checkpoint = _load_checkpoint(path, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if (data_loader_generator is not None
            and checkpoint.get("data_loader_generator_state") is not None):
        # The checkpoint is loaded on the training device so optimizer tensors
        # are restored directly to that device.  Generator state is different:
        # torch.Generator.set_state only accepts a CPU ByteTensor.
        data_loader_generator.set_state(
            checkpoint["data_loader_generator_state"].cpu()
        )
    rng = checkpoint.get("rng_state", {})
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    return checkpoint
