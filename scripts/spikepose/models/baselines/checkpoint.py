"""Checkpoint compatibility for official HRNet-family releases."""

from __future__ import annotations

from collections import OrderedDict
import hashlib

import torch


def checkpoint_state_dict(value: object) -> OrderedDict[str, torch.Tensor]:
    if isinstance(value, dict) and "state_dict" in value:
        value = value["state_dict"]
    if not isinstance(value, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    return OrderedDict(
        (str(key).removeprefix("module."), tensor)
        for key, tensor in value.items()
        if isinstance(tensor, torch.Tensor)
    )


def tensor_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def align_official_state_dict(
    model: torch.nn.Module, source: dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Align official HRNet/MMPose nesting names to the adapted implementation.

    The official HRNet implementation wraps some downsample blocks in an extra
    Sequential. MMPose additionally prefixes the backbone and names its output
    head differently. State traversal order is unchanged. Alignment is accepted
    only when tensor counts match and every non-head tensor shape agrees.
    """
    normalized = OrderedDict()
    for key, tensor in source.items():
        key = key.removeprefix("backbone.")
        key = key.replace("keypoint_head.final_layer", "final_layer")
        normalized[key] = tensor
    target = model.state_dict()
    if set(normalized).issubset(target):
        return normalized
    if len(normalized) != len(target):
        raise RuntimeError(
            f"Checkpoint tensor count mismatch: source={len(normalized)}, "
            f"target={len(target)}"
        )
    aligned = OrderedDict()
    for (source_key, tensor), (target_key, target_tensor) in zip(
        normalized.items(), target.items()
    ):
        if tensor.shape != target_tensor.shape and target_key not in {
            "final_layer.weight", "final_layer.bias",
        }:
            raise RuntimeError(
                "Checkpoint order/shape mismatch at "
                f"source={source_key}, target={target_key}"
            )
        aligned[target_key] = tensor
    return aligned
