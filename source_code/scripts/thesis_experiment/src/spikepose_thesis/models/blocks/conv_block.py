from __future__ import annotations

import torch
import torch.nn as nn

from ..config import NeuronConfig
from ..layers import SpikeConv
from .separable import SpikePoseSeparableBlock


class SpikePoseConvBlock(nn.Module):
    def __init__(self, channels: int, neuron: NeuronConfig,
                 expansion: int = 3) -> None:
        super().__init__()
        hidden = channels * expansion
        self.separable = SpikePoseSeparableBlock(channels, neuron)
        self.expand = SpikeConv(channels, hidden, 1, neuron)
        self.depthwise = SpikeConv(hidden, hidden, 3, neuron, padding=1, groups=hidden)
        self.project = SpikeConv(hidden, channels, 1, neuron)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.separable(value)
        return value + self.project(self.depthwise(self.expand(value)))
