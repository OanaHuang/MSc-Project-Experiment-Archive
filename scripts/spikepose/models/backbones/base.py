from dataclasses import dataclass

import torch


@dataclass
class BackboneOutput:
    features: dict[int, torch.Tensor]
    channels: dict[int, int]
    strides: dict[int, int]
