from __future__ import annotations

import torch
import torch.nn as nn

from ..config import NeuronConfig
from ..layers import SpikeConv
from .separable import SpikePoseSeparableBlock


class SpikePoseAllConvBlock(nn.Module):
    def __init__(self, channels: int, neuron: NeuronConfig,
                 expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.separable = SpikePoseSeparableBlock(channels, neuron)
        self.expand = SpikeConv(channels, hidden, 3, neuron, padding=1)
        self.project = SpikeConv(hidden, channels, 3, neuron, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.separable(value)
        return value + self.project(self.expand(value))
