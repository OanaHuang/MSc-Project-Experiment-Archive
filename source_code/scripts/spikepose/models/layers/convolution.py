from __future__ import annotations

import torch
import torch.nn as nn

from ..config import NeuronConfig
from ..neurons import build_neuron
from .temporal import merge_time_batch, restore_time_batch


class TimeConvBN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        flat, steps, batch = merge_time_batch(value)
        return restore_time_batch(self.bn(self.conv(flat)), steps, batch)


class SpikeConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 neuron: NeuronConfig, stride: int = 1, padding: int = 0,
                 groups: int = 1) -> None:
        super().__init__()
        parameters = dict(
            decay=neuron.decay, threshold=neuron.threshold,
            max_spikes=neuron.max_spikes,
            membrane_readout=neuron.membrane_readout,
            membrane_readout_init=neuron.membrane_readout_init,
            learnable_decay=neuron.learnable_decay,
            learnable_initial_membrane=neuron.learnable_initial_membrane,
            initial_membrane_scale=neuron.initial_membrane_scale,
        )
        self.neuron = build_neuron(neuron.kind, **parameters)
        self.conv = TimeConvBN(in_channels, out_channels, kernel_size, stride,
                               padding, groups)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(self.neuron(value))
