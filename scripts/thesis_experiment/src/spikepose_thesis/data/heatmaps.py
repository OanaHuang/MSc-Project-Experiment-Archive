from __future__ import annotations

import torch


def generate_gaussian_heatmaps_torch(
    keypoints: torch.Tensor,
    visibility: torch.Tensor,
    image_size: int,
    heatmap_size: int,
    sigma: float,
) -> torch.Tensor:
    """Generate a batch of truncated Gaussian targets on the input device.

    The definition intentionally matches the legacy NumPy implementation: joint
    centres are rounded to pixels and each Gaussian is truncated at 3 sigma.
    ``keypoints`` may have any leading dimensions followed by ``[J, 2]``.
    """
    if keypoints.ndim < 2 or keypoints.shape[-1] != 2:
        raise ValueError("keypoints must end with [J, 2]")
    if visibility.shape != keypoints.shape[:-1]:
        raise ValueError("visibility must match keypoints without the xy axis")
    if image_size <= 0 or heatmap_size <= 0 or sigma <= 0:
        raise ValueError("image_size, heatmap_size and sigma must be positive")

    original_shape = keypoints.shape[:-2]
    joints = keypoints.shape[-2]
    points = keypoints.reshape(-1, joints, 2).float()
    visible = visibility.reshape(-1, joints).bool()
    scaled = points * (float(heatmap_size) / float(image_size))
    finite = torch.isfinite(scaled).all(dim=-1)
    safe_scaled = torch.nan_to_num(scaled)
    centres = torch.round(safe_scaled).to(torch.long)
    valid = (
        visible
        & finite
        & (centres[..., 0] >= 0)
        & (centres[..., 0] < heatmap_size)
        & (centres[..., 1] >= 0)
        & (centres[..., 1] < heatmap_size)
    )

    axis = torch.arange(heatmap_size, device=points.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    x = safe_scaled[..., 0, None, None]
    y = safe_scaled[..., 1, None, None]
    gaussian = torch.exp(
        -((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2.0 * float(sigma) ** 2)
    )
    radius = int(3 * sigma)
    centre_x = centres[..., 0, None, None]
    centre_y = centres[..., 1, None, None]
    support = (
        (grid_x >= centre_x - radius)
        & (grid_x <= centre_x + radius)
        & (grid_y >= centre_y - radius)
        & (grid_y <= centre_y + radius)
    )
    heatmaps = gaussian * support * valid[..., None, None]
    return heatmaps.reshape(*original_shape, joints, heatmap_size, heatmap_size)


__all__ = ["generate_gaussian_heatmaps_torch"]
