from __future__ import annotations

import torch
import torch.nn as nn

from ..config import NeuronConfig
from ..layers import SpikeConv, TimeConvBN
from ..neurons import build_neuron


class MembraneResidualBlock(nn.Module):
    def __init__(self, channels: int, neuron: NeuronConfig) -> None:
        super().__init__()
        self.conv1 = SpikeConv(channels, channels, 3, neuron, padding=1)
        self.conv2 = SpikeConv(channels, channels, 3, neuron, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.conv2(self.conv1(value))


class MultiConvBlock(nn.Module):
    def __init__(self, channels: int, neuron: NeuronConfig) -> None:
        super().__init__()
        self.conv1 = SpikeConv(channels, channels, 3, neuron, padding=1)
        self.conv2 = SpikeConv(channels, channels, 3, neuron, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(value))


class SEWAddBlock(nn.Module):
    def __init__(self, channels: int, neuron: NeuronConfig) -> None:
        super().__init__()
        self.identity_spike = build_neuron(
            neuron.kind, decay=neuron.decay, threshold=neuron.threshold,
            max_spikes=neuron.max_spikes,
            membrane_readout=neuron.membrane_readout,
            membrane_readout_init=neuron.membrane_readout_init,
            learnable_decay=neuron.learnable_decay,
            learnable_initial_membrane=neuron.learnable_initial_membrane,
            initial_membrane_scale=neuron.initial_membrane_scale,
        )
        self.conv1 = SpikeConv(channels, channels, 3, neuron, padding=1)
        self.conv2 = SpikeConv(channels, channels, 3, neuron, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.identity_spike(value) + self.conv2(self.conv1(value))


class SpikeAttentionBlock(nn.Module):
    def __init__(self, channels: int, neuron: NeuronConfig,
                 heads: int = 4) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by attention heads")
        self.heads = heads
        self.head_dim = channels // heads
        self.qkv = SpikeConv(channels, channels * 3, 1, neuron)
        self.projection = TimeConvBN(channels, channels, 1)
        self.ffn = nn.Sequential(
            SpikeConv(channels, channels * 2, 1, neuron),
            SpikeConv(channels * 2, channels, 1, neuron),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        steps, batch, channels, height, width = value.shape
        qkv = self.qkv(value).reshape(
            steps, batch, 3, self.heads, self.head_dim, height * width)
        query, key, content = qkv.unbind(2)
        attention = torch.matmul(query.transpose(-2, -1), key)
        attention = attention * (self.head_dim ** -0.5)
        mixed = torch.matmul(attention.softmax(-1), content.transpose(-2, -1))
        mixed = mixed.transpose(-2, -1).reshape(
            steps, batch, channels, height, width)
        value = value + self.projection(mixed)
        return value + self.ffn(value)
