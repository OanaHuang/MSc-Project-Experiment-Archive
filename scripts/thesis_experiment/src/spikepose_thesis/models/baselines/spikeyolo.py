"""SpikeYOLO-s P3 features with the common linear pose heatmap head.

The graph follows BICLab/SpikeYOLO dc51f35, snn_yolov8.yaml, scale s,
through feature layer 19. Detection-only descendants 20--26 are omitted:
they cannot affect the selected P3 feature and would have no pose gradients.
No upstream detector checkpoint or detection-training stack is required.
"""
from __future__ import annotations

import torch
from torch import nn

from ..heads.linear import LinearHeatmapHead
from . import _spikeyolo_layers as upstream


class _Concat(nn.Module):
    def forward(self, values):
        return torch.cat(values, dim=2)


class SpikeYOLOPose(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        backbone = config["backbone"]
        temporal = config.get("temporal", {})
        if (backbone.get("scale") != "s" or backbone.get("feature_layer") != 19
                or backbone.get("keep_layers") != [0, 19]
                or backbone.get("feature_channels") != 128):
            raise ValueError("SpikeYOLO pose adapter supports scale s, layers 0--19, P3/128 only")
        expected_steps = (int(temporal.get("video_frames", 1))
                          if temporal.get("input_strategy") == "frames" else 1)
        if (int(config.get("num_steps", 1)) != expected_steps
                or int(temporal.get("snn_steps_per_frame", 1)) != 1
                or temporal.get("state_mode", "reset") != "reset"):
            raise ValueError("SpikeYOLO pose adapter requires T=1 and frame-reset state")
        if config["head"]["kind"] != "linear_heatmap":
            raise ValueError("SpikeYOLO requires the shared linear heatmap head")
        # Width multiplier .5 and depth multiplier .33, identical to upstream s.
        self.backbone = nn.ModuleList([
            upstream.MS_GetT(3, 3, 1),
            upstream.MS_DownSampling(3, 64, 7, 4, 2, True),
            upstream.MS_AllConvBlock(64, 4, 7),
            upstream.MS_DownSampling(64, 128, 3, 2, 1, False),
            nn.Sequential(*(upstream.MS_AllConvBlock(128, 4, 7) for _ in range(2))),
            upstream.MS_DownSampling(128, 256, 3, 2, 1, False),
            nn.Sequential(*(upstream.MS_ConvBlock(256, 3, 7) for _ in range(3))),
            upstream.MS_DownSampling(256, 512, 3, 2, 1, False),
            upstream.MS_ConvBlock(512, 2, 7),
            upstream.SpikeSPPF(512, 512, 5),
            upstream.MS_StandardConv(512, 256, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            upstream.MS_ConvBlock(256, 3, 7),
            _Concat(),
            upstream.MS_StandardConv(512, 128, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            upstream.MS_AllConvBlock(128, 4, 7),
            _Concat(),
            upstream.MS_StandardConv(256, 128, 1, 1),
            upstream.MS_AllConvBlock(128, 4, 7),
        ])
        self.head = LinearHeatmapHead(
            128, int(config["head"].get("hidden_channels", 128)),
            int(config["num_joints"]), tuple(config["heatmap_size"]),
            aggregation="mean", num_steps=1,
        )
        nn.init.normal_(self.head.output.weight, std=float(config["head"].get("output_init_std", .001)))
        nn.init.zeros_(self.head.output.bias)

    def forward_features(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("SpikeYOLO expects B x 3 x H x W images")
        if any(size % 32 for size in image.shape[-2:]):
            raise ValueError("SpikeYOLO image dimensions must be divisible by 32")
        saved = {}
        value = image
        for index, layer in enumerate(self.backbone):
            if index == 13:
                value = [value, saved[6]]
            elif index == 17:
                value = [value, saved[4]]
            value = layer(value)
            if index in {4, 6}:
                saved[index] = value
        return value

    def forward(self, image):
        if image.ndim == 5:
            return self.forward_per_step(image)[-1]
        return self.head(self.forward_features(image))

    def forward_per_step(self, images):
        if images.ndim != 5:
            raise ValueError("Expected B x V x 3 x H x W video input")
        # Each call reinitializes every upstream mem_update's local state.
        return torch.stack([self(frame) for frame in images.unbind(1)], dim=0)


def build_spikeyolo(config: dict) -> SpikeYOLOPose:
    return SpikeYOLOPose(config)
