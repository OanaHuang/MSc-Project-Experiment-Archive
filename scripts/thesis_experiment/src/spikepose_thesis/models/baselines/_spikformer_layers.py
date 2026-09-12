"""PyTorch-only Spikformer layers adapted from ZK-Zhou/spikformer.

The upstream ImageNet implementation is MIT licensed (see
``SPIKFORMER_LICENSE``).  This copy preserves the SPS, SSA and MLP computation
while removing the timm registry, the classification head, the CuPy backend
requirement and two LayerNorm modules that upstream constructs but never uses
in ``Block.forward``.  The local LIF implementation matches SpikingJelly's
tau=2, hard-reset, detached-reset neuron and sigmoid surrogate, and keeps this
thesis pipeline dependent on PyTorch only.
"""
from __future__ import annotations

import torch
from torch import nn


class _SigmoidSpike(torch.autograd.Function):
    """Binary spike with SpikingJelly's sigmoid(alpha=4) surrogate gradient."""

    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)
        return (value >= 0).to(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        (value,) = ctx.saved_tensors
        sigmoid = torch.sigmoid(4.0 * value)
        return gradient * 4.0 * sigmoid * (1.0 - sigmoid)


class MultiStepLIFNode(nn.Module):
    """Stateless multi-step LIF used by the original Spikformer blocks."""

    def __init__(self, tau: float = 2.0, v_threshold: float = 1.0,
                 detach_reset: bool = True) -> None:
        super().__init__()
        self.tau = float(tau)
        self.v_threshold = float(v_threshold)
        self.detach_reset = bool(detach_reset)
        self.max_spikes = 1

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim not in {4, 5}:
            raise ValueError("Spikformer LIF expects T-first 4D or 5D input")
        membrane = torch.zeros_like(value[0])
        spikes = []
        for current in value.unbind(0):
            membrane = membrane + (current - membrane) / self.tau
            spike = _SigmoidSpike.apply(membrane - self.v_threshold)
            reset = spike.detach() if self.detach_reset else spike
            membrane = membrane * (1.0 - reset)
            spikes.append(spike)
        return torch.stack(spikes)


class MLP(nn.Module):
    def __init__(self, features: int, ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(features * ratio)
        self.hidden = hidden
        self.features = features
        self.fc1_conv = nn.Conv2d(features, hidden, 1)
        self.fc1_bn = nn.BatchNorm2d(hidden)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.fc2_conv = nn.Conv2d(hidden, features, 1)
        self.fc2_bn = nn.BatchNorm2d(features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        steps, batch, channels, height, width = value.shape
        value = self.fc1_conv(value.flatten(0, 1))
        value = self.fc1_bn(value).reshape(
            steps, batch, self.hidden, height, width,
        ).contiguous()
        value = self.fc1_lif(value)
        value = self.fc2_conv(value.flatten(0, 1))
        value = self.fc2_bn(value).reshape(
            steps, batch, channels, height, width,
        ).contiguous()
        return self.fc2_lif(value)


class SSA(nn.Module):
    def __init__(self, features: int, heads: int = 8) -> None:
        super().__init__()
        if features % heads:
            raise ValueError("Spikformer feature width must be divisible by heads")
        self.features = features
        self.heads = heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(features, features, 1, bias=False)
        self.q_bn = nn.BatchNorm1d(features)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.k_conv = nn.Conv1d(features, features, 1, bias=False)
        self.k_bn = nn.BatchNorm1d(features)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.v_conv = nn.Conv1d(features, features, 1, bias=False)
        self.v_bn = nn.BatchNorm1d(features)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.attn_lif = MultiStepLIFNode(
            tau=2.0, v_threshold=0.5, detach_reset=True,
        )
        self.proj_conv = nn.Conv1d(features, features, 1)
        self.proj_bn = nn.BatchNorm1d(features)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

    def _project(self, value: torch.Tensor, conv: nn.Conv1d,
                 bn: nn.BatchNorm1d, lif: MultiStepLIFNode) -> torch.Tensor:
        steps, batch, channels, tokens = value.shape
        projected = bn(conv(value.flatten(0, 1)))
        return lif(projected.reshape(
            steps, batch, channels, tokens,
        ).contiguous())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        steps, batch, channels, height, width = value.shape
        tokens = height * width
        flat = value.flatten(3)
        head_width = channels // self.heads
        q = self._project(flat, self.q_conv, self.q_bn, self.q_lif)
        k = self._project(flat, self.k_conv, self.k_bn, self.k_lif)
        v = self._project(flat, self.v_conv, self.v_bn, self.v_lif)
        q = q.transpose(-1, -2).reshape(
            steps, batch, tokens, self.heads, head_width,
        ).permute(0, 1, 3, 2, 4).contiguous()
        k = k.transpose(-1, -2).reshape(
            steps, batch, tokens, self.heads, head_width,
        ).permute(0, 1, 3, 2, 4).contiguous()
        v = v.transpose(-1, -2).reshape(
            steps, batch, tokens, self.heads, head_width,
        ).permute(0, 1, 3, 2, 4).contiguous()
        attended = (q @ (k.transpose(-2, -1) @ v)) * self.scale
        attended = attended.transpose(3, 4).reshape(
            steps, batch, channels, tokens,
        ).contiguous()
        attended = self.attn_lif(attended)
        attended = self.proj_conv(attended.flatten(0, 1))
        attended = self.proj_bn(attended).reshape(
            steps, batch, channels, height, width,
        ).contiguous()
        return self.proj_lif(attended)


class Block(nn.Module):
    def __init__(self, features: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.attn = SSA(features, heads)
        self.mlp = MLP(features, mlp_ratio)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.attn(value)
        return value + self.mlp(value)


class SPS(nn.Module):
    """Four-stage spiking patch splitting stem with output stride 16."""

    def __init__(self, in_channels: int, embed_dims: int) -> None:
        super().__init__()
        widths = (embed_dims // 8, embed_dims // 4, embed_dims // 2, embed_dims)
        channels = (in_channels, *widths[:-1])
        self.convs = nn.ModuleList([
            nn.Conv2d(source, target, 3, padding=1, bias=False)
            for source, target in zip(channels, widths)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for width in widths])
        self.lifs = nn.ModuleList([
            MultiStepLIFNode(tau=2.0, detach_reset=True) for _ in widths
        ])
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, 3, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        steps, batch = value.shape[:2]
        for conv, bn, lif in zip(self.convs, self.bns, self.lifs):
            value = conv(value.flatten(0, 1))
            channels, height, width = value.shape[-3:]
            value = bn(value).reshape(
                steps, batch, channels, height, width,
            ).contiguous()
            value = lif(value).flatten(0, 1).contiguous()
            value = self.pool(value).reshape(
                steps, batch, channels, height // 2, width // 2,
            ).contiguous()
        residual = value
        value = self.rpe_conv(value.flatten(0, 1))
        channels, height, width = value.shape[-3:]
        value = self.rpe_bn(value).reshape(
            steps, batch, channels, height, width,
        ).contiguous()
        return self.rpe_lif(value) + residual


class SpikformerBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dims: int = 384,
                 depth: int = 8, heads: int = 8,
                 mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.patch_embed = SPS(in_channels, embed_dims)
        self.blocks = nn.ModuleList([
            Block(embed_dims, heads, mlp_ratio) for _ in range(depth)
        ])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.patch_embed(value)
        for block in self.blocks:
            value = block(value)
        return value
