from __future__ import annotations

import torch
from torch import nn


class SmoothNetBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width), nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout), nn.Linear(width, width),
            nn.LeakyReLU(0.1, inplace=True), nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.layers(value)


class SmoothNet(nn.Module):
    """Coordinate-wise temporal refinement network inspired by SmoothNet.

    Input and output have shape [batch, window, joints, 2]. The same temporal
    MLP is shared by every joint coordinate, keeping the post-processor
    independent of the upstream F-series architecture.
    """

    def __init__(self, window_size: int = 32, hidden_size: int = 512,
                 num_blocks: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        if window_size < 3 or hidden_size < 1 or num_blocks < 1:
            raise ValueError("window_size >= 3, hidden_size >= 1 and num_blocks >= 1 required")
        self.window_size = int(window_size)
        self.input = nn.Linear(window_size, hidden_size)
        self.blocks = nn.Sequential(*[
            SmoothNetBlock(hidden_size, dropout) for _ in range(num_blocks)
        ])
        self.output = nn.Linear(hidden_size, window_size)

    def forward(self, poses: torch.Tensor) -> torch.Tensor:
        if poses.ndim != 4 or poses.shape[1] != self.window_size or poses.shape[-1] != 2:
            raise ValueError("poses must have shape [B, window_size, J, 2]")
        coordinates = poses.permute(0, 2, 3, 1)
        residual = self.output(self.blocks(self.input(coordinates)))
        return (coordinates + residual).permute(0, 3, 1, 2)
