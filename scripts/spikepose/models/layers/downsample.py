from __future__ import annotations

import torch
import torch.nn as nn

from ..config import NeuronConfig
from ..neurons import build_neuron
from .convolution import TimeConvBN


class SpikePoseDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 neuron: NeuronConfig, first: bool = False) -> None:
        super().__init__()
        self.neuron = None if first else build_neuron(
            neuron.kind, decay=neuron.decay, threshold=neuron.threshold,
            max_spikes=neuron.max_spikes,
            membrane_readout=neuron.membrane_readout,
            membrane_readout_init=neuron.membrane_readout_init,
            learnable_decay=neuron.learnable_decay,
            learnable_initial_membrane=neuron.learnable_initial_membrane,
            initial_membrane_scale=neuron.initial_membrane_scale,
        )
        kernel, padding = (7, 2) if first else (3, 1)
        self.conv = TimeConvBN(in_channels, out_channels, kernel, stride, padding)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.neuron is not None:
            value = self.neuron(value)
        return self.conv(value)
