"""Initialization helpers for external NTU pose baselines."""

from __future__ import annotations

from pathlib import Path

import torch

from scripts.spikepose.models.baselines import (
    align_official_state_dict, checkpoint_state_dict, tensor_fingerprint,
)


def initialize_external_baseline(
    model: torch.nn.Module, checkpoint_path: Path,
    expected_fingerprint: str | None = None,
) -> dict:
    """Load all shape-compatible MPII tensors and leave the NTU head random."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing MPII pretrained checkpoint: {checkpoint_path}")
    try:
        value = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced.
        value = torch.load(checkpoint_path, map_location="cpu")
    raw_source = checkpoint_state_dict(value)
    fingerprint = tensor_fingerprint(raw_source)
    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise RuntimeError(
            "MPII checkpoint tensor fingerprint mismatch: "
            f"expected={expected_fingerprint}, actual={fingerprint}"
        )
    source = align_official_state_dict(model, raw_source)
    target = model.state_dict()
    loaded = {
        key: tensor for key, tensor in source.items()
        if key in target and target[key].shape == tensor.shape
    }
    skipped_shape = {
        key: {"source": list(tensor.shape), "target": list(target[key].shape)}
        for key, tensor in source.items()
        if key in target and target[key].shape != tensor.shape
    }
    unexpected = sorted(key for key in source if key not in target)
    incompatible = model.load_state_dict(loaded, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError("Unexpected partial checkpoint loading result")
    missing = sorted(incompatible.missing_keys)
    if not loaded:
        raise RuntimeError("No MPII pretrained tensors matched the NTU model")
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_tensor_sha256": fingerprint,
        "loaded_tensor_count": len(loaded),
        "source_tensor_count": len(source),
        "target_tensor_count": len(target),
        "loaded_tensors": sorted(loaded),
        "skipped_shape": skipped_shape,
        "unexpected_source_tensors": unexpected,
        "randomly_initialized_tensors": missing,
    }
