from __future__ import annotations

import torch
import torch.nn as nn

from .concat import resize_time


class MaskedAdditiveFusion(nn.Module):
    """Fixed-capacity multi-stage fusion controlled only by a binary mask."""

    def __init__(self, stages: tuple[int, ...], channels: dict[int, int],
                 out_channels: int, stage_mask: tuple[int, ...],
                 interpolation: str = "nearest",
                 align_corners: bool = False) -> None:
        super().__init__()
        if len(stages) != len(stage_mask):
            raise ValueError("stage_mask must contain one value per fusion stage")
        if any(value not in (0, 1) for value in stage_mask):
            raise ValueError("stage_mask values must be binary")
        if not any(stage_mask):
            raise ValueError("stage_mask must enable at least one stage")
        self.selected_stages = stages
        self.stage_mask = stage_mask
        self.projections = nn.ModuleDict({
            str(index): nn.Conv2d(channels[index], out_channels, 1, bias=False)
            for index in stages
        })
        self.out_channels = out_channels
        self.interpolation = interpolation
        self.align_corners = align_corners

    def forward(self, features: dict[int, torch.Tensor]) -> torch.Tensor:
        # The target and projections are independent of the active mask so all
        # S-series models retain the same modules, tensor shapes and capacity.
        selected = [features[index] for index in self.selected_stages]
        target = max((item.shape[-2:] for item in selected), key=lambda size: size[0])
        fused = None
        for index, mask, item in zip(
                self.selected_stages, self.stage_mask, selected):
            if item.shape[-2:] != target:
                item = resize_time(item, target, self.interpolation, self.align_corners)
            steps, batch = item.shape[:2]
            value = self.projections[str(index)](item.flatten(0, 1))
            value = value.reshape(steps, batch, *value.shape[1:]) * mask
            fused = value if fused is None else fused + value
        assert fused is not None
        return fused
