from __future__ import annotations

import torch
import torch.nn as nn

from .concat import resize_time


class AdditiveFusion(nn.Module):
    def __init__(self, stages: tuple[int, ...], channels: dict[int, int],
                 out_channels: int, interpolation: str = "nearest",
                 align_corners: bool = False) -> None:
        super().__init__()
        self.selected_stages = stages
        self.projections = nn.ModuleDict({
            str(index): nn.Conv2d(channels[index], out_channels, 1, bias=False)
            for index in stages
        })
        self.out_channels = out_channels
        self.interpolation = interpolation
        self.align_corners = align_corners

    def forward(self, features: dict[int, torch.Tensor]) -> torch.Tensor:
        selected = [features[index] for index in self.selected_stages]
        target = max((item.shape[-2:] for item in selected), key=lambda size: size[0])
        projected = []
        for index, item in zip(self.selected_stages, selected):
            if item.shape[-2:] != target:
                item = resize_time(item, target, self.interpolation, self.align_corners)
            steps, batch = item.shape[:2]
            value = self.projections[str(index)](item.flatten(0, 1))
            projected.append(value.reshape(steps, batch, *value.shape[1:]))
        return torch.stack(projected).sum(0)
