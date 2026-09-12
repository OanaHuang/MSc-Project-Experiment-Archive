from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_time(value: torch.Tensor, size: tuple[int, int], mode: str = "nearest",
                align_corners: bool = False) -> torch.Tensor:
    steps, batch = value.shape[:2]
    options = {"align_corners": align_corners} if mode in {"bilinear", "bicubic"} else {}
    flat = F.interpolate(value.flatten(0, 1), size=size, mode=mode, **options)
    return flat.reshape(steps, batch, *flat.shape[1:])


class ConcatFusion(nn.Module):
    def __init__(self, stages: tuple[int, ...], channels: dict[int, int],
                 out_channels: int, interpolation: str = "nearest",
                 align_corners: bool = False) -> None:
        super().__init__()
        self.selected_stages = stages
        self.out_channels = sum(channels[index] for index in stages)
        self.interpolation = interpolation
        self.align_corners = align_corners

    def forward(self, features: dict[int, torch.Tensor]) -> torch.Tensor:
        selected = [features[index] for index in self.selected_stages]
        target = max((item.shape[-2:] for item in selected), key=lambda size: size[0])
        selected = [resize_time(item, target, self.interpolation, self.align_corners)
                    if item.shape[-2:] != target else item
                    for item in selected]
        return torch.cat(selected, dim=2)
