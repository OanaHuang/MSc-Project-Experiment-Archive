from __future__ import annotations

import torch
import torch.nn as nn

from ..blocks import build_block
from ..blocks.residual import SpikeAttentionBlock
from ..config import BackboneConfig, NeuronConfig, TemporalConfig
from ..layers import SpikePoseDownsample
from ..temporal_transformer import TemporalScaleTransformer
from .base import BackboneOutput


class SpikePoseMSBackbone(nn.Module):
    def __init__(self, in_channels: int, config: BackboneConfig,
                 neuron: NeuronConfig, temporal: TemporalConfig | None = None,
                 num_steps: int | None = None) -> None:
        super().__init__()
        if not (len(config.channels) == len(config.depths) == len(config.blocks)):
            raise ValueError("channels, depths and blocks must have equal length")
        self.config = config
        stages = []
        downsamples = []
        previous = in_channels
        stride = 4
        self.strides = {}
        for index, (channels, depth, block) in enumerate(
                zip(config.channels, config.depths, config.blocks), 1):
            downsamples.append(SpikePoseDownsample(
                previous, channels, 4 if index == 1 else 2, neuron, first=index == 1,
            ))
            stages.append(nn.Sequential(*[
                build_block(block, channels, neuron) for _ in range(depth)
            ], *([SpikeAttentionBlock(channels, neuron)]
                  if index in config.attention_stages else [])))
            self.strides[index] = stride
            stride *= 2
            previous = channels
        self.downsamples = nn.ModuleList(downsamples)
        self.stages = nn.ModuleList(stages)
        stage_steps = tuple(temporal.stage_steps) if temporal is not None else ()
        if stage_steps:
            if len(stage_steps) != len(stages):
                raise ValueError("stage_steps must contain one value per backbone stage")
            if num_steps is None or stage_steps[0] != num_steps:
                raise ValueError("stage_steps must start at model.num_steps")
            output_steps = temporal.output_steps or stage_steps[-1]
            transform_kind = temporal.transform_kind
            self.transition_adapters = nn.ModuleDict({
                str(index): TemporalScaleTransformer(
                    stage_steps[index - 1], stage_steps[index], transform_kind,
                )
                for index in range(1, len(stage_steps))
                if stage_steps[index - 1] != stage_steps[index]
            })
            self.output_adapters = nn.ModuleDict({
                str(index): TemporalScaleTransformer(
                    stage_steps[index - 1], output_steps, transform_kind,
                )
                for index in config.output_stages
                if stage_steps[index - 1] != output_steps
            })
        else:
            self.transition_adapters = nn.ModuleDict()
            self.output_adapters = nn.ModuleDict()

    def forward(self, value: torch.Tensor) -> BackboneOutput:
        features = {}
        for index, (downsample, stage) in enumerate(
                zip(self.downsamples, self.stages), 1):
            value = stage(downsample(value))
            features[index] = (
                self.output_adapters[str(index)](value)
                if str(index) in self.output_adapters else value
            )
            if str(index) in self.transition_adapters:
                value = self.transition_adapters[str(index)](value)
        channels = {index: channels for index, channels in
                    enumerate(self.config.channels, 1)}
        return BackboneOutput(features, channels, self.strides)
