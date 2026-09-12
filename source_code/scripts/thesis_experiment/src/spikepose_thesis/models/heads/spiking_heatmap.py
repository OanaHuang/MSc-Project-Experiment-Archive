from __future__ import annotations

import torch
import torch.nn as nn

from ..config import NeuronConfig
from ..layers import TimeConvBN
from ..neurons import build_neuron


class SpikingHeatmapHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_joints: int,
                 output_size: tuple[int, int] | None, neuron: NeuronConfig,
                 num_steps: int, upsample_factor: int = 1,
                 aggregation: str = "mean") -> None:
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
        self.spiking = nn.Sequential(
            build_neuron(neuron.kind, **parameters),
            TimeConvBN(in_channels, hidden_channels, 1),
            build_neuron(neuron.kind, **parameters),
            TimeConvBN(hidden_channels, hidden_channels, 3, padding=1),
        )
        self.output_size = output_size
        self.aggregation = aggregation
        self.upsample = (nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True),
        ) if upsample_factor == 2 else nn.Identity())
        self.final = nn.Conv2d(hidden_channels, num_joints, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError("SpikingHeatmapHead requires T x B x C x H x W input")
        value = self.spiking(value)
        value = value.mean(0) if self.aggregation == "mean" else value[-1]
        value = self.final(self.upsample(value))
        if self.output_size is not None and value.shape[-2:] != self.output_size:
            value = torch.nn.functional.interpolate(
                value, self.output_size, mode="bilinear", align_corners=False)
        return value
