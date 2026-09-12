import torch


def merge_time_batch(value: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if value.ndim != 5:
        raise ValueError("Expected T x B x C x H x W input")
    steps, batch = value.shape[:2]
    return value.flatten(0, 1), steps, batch


def restore_time_batch(value: torch.Tensor, steps: int, batch: int) -> torch.Tensor:
    return value.reshape(steps, batch, *value.shape[1:])
