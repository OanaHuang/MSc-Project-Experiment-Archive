"""Spikformer-8-384 T1 features with the shared linear pose heatmap head."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from ..heads.linear import LinearHeatmapHead
from ._spikformer_layers import SpikformerBackbone


class SpikformerPose(nn.Module):
    """Low-latency Spikformer pose baseline with one SNN update per frame."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        backbone = config["backbone"]
        temporal = config.get("temporal", {})
        expected = {
            "variant": "spikformer_8_384_t1",
            "embed_dims": 384,
            "depth": 8,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "feature_stride": 16,
            "snn_steps": 1,
        }
        actual = {"variant": config.get("variant"), **{
            key: backbone.get(key) for key in expected if key != "variant"
        }}
        if actual != expected:
            raise ValueError(
                "Spikformer pose adapter supports Spikformer-8-384 T=1 only"
            )
        if (int(temporal.get("snn_steps_per_frame", 1)) != 1
                or temporal.get("state_mode", "reset") != "reset"):
            raise ValueError("Spikformer-T1 requires one reset SNN update per frame")
        if config["head"]["kind"] != "linear_heatmap":
            raise ValueError("Spikformer requires the shared linear heatmap head")
        self.config = SimpleNamespace(temporal=SimpleNamespace(
            video_frames=int(temporal.get("video_frames", 1)),
            snn_steps_per_frame=1,
        ))
        self.backbone = SpikformerBackbone(
            in_channels=int(config.get("in_channels", 3)),
            embed_dims=384, depth=8, heads=8, mlp_ratio=4.0,
        )
        self.head = LinearHeatmapHead(
            384, int(config["head"].get("hidden_channels", 384)),
            int(config["num_joints"]), tuple(config["heatmap_size"]),
            aggregation="mean", num_steps=1,
        )
        nn.init.normal_(
            self.head.output.weight,
            std=float(config["head"].get("output_init_std", 0.001)),
        )
        nn.init.zeros_(self.head.output.bias)

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Spikformer expects B x 3 x H x W images")
        if any(size % 16 for size in image.shape[-2:]):
            raise ValueError("Spikformer image dimensions must be divisible by 16")
        return self.backbone(image.unsqueeze(0))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 5:
            return self.forward_per_step(image)[-1]
        return self.head(self.forward_features(image))

    def forward_per_step(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError("Expected B x V x 3 x H x W video input")
        # Every physical frame starts a fresh one-step SNN trajectory.
        return torch.stack([self(frame) for frame in images.unbind(1)], dim=0)


def build_spikformer(config: dict) -> SpikformerPose:
    return SpikformerPose(config)
