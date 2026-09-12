from __future__ import annotations

import torch
import torch.nn as nn

from ..config import NeuronConfig
from ..layers import SpikeConv


class SpikePoseSeparableBlock(nn.Module):
    def __init__(self, channels: int, neuron: NeuronConfig,
                 expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.expand = SpikeConv(channels, hidden, 1, neuron)
        self.depthwise = SpikeConv(hidden, hidden, 7, neuron, padding=3, groups=hidden)
        self.project = SpikeConv(hidden, channels, 1, neuron)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.project(self.depthwise(self.expand(value)))
