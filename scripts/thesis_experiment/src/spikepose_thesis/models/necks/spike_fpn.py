from __future__ import annotations

import torch
import torch.nn as nn

from ..config import NeuronConfig
from ..layers import SpikeConv, TimeConvBN
from ..neurons import build_neuron
from .concat import resize_time


class SpikeFPNFusion(nn.Module):
    def __init__(self, stages: tuple[int, ...], channels: dict[int, int],
                 out_channels: int, neuron: NeuronConfig,
                 interpolation: str = "bilinear", align_corners: bool = False,
                 preserve_time: bool = False) -> None:
        super().__init__()
        self.selected_stages = stages
        self.out_channels = out_channels
        self.interpolation = interpolation
        self.align_corners = align_corners
        self.preserve_time = preserve_time
        self.projections = nn.ModuleDict({
            str(index): TimeConvBN(channels[index], out_channels, 1)
            for index in stages
        })
        self.refine = nn.Sequential(
            SpikeConv(out_channels, out_channels, 3, neuron, padding=1),
            SpikeConv(out_channels, out_channels, 3, neuron, padding=1),
        )
        self.readout = build_neuron(
            neuron.kind, decay=neuron.decay, threshold=neuron.threshold,
            max_spikes=neuron.max_spikes,
            membrane_readout=neuron.membrane_readout,
            membrane_readout_init=neuron.membrane_readout_init,
            learnable_decay=neuron.learnable_decay,
            learnable_initial_membrane=neuron.learnable_initial_membrane,
            initial_membrane_scale=neuron.initial_membrane_scale,
        )

    def forward(self, features: dict[int, torch.Tensor]) -> torch.Tensor:
        selected = [features[index] for index in self.selected_stages]
        target = max((item.shape[-2:] for item in selected), key=lambda size: size[0])
        fused = None
        for index, item in zip(self.selected_stages, selected):
            projected = self.projections[str(index)](item)
            if projected.shape[-2:] != target:
                projected = resize_time(
                    projected, target, self.interpolation, self.align_corners)
            fused = projected if fused is None else fused + projected
        value = self.readout(self.refine(fused / len(selected)))
        return value if self.preserve_time else value.mean(0)
