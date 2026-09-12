import torch
import torch.nn as nn


class TimeDistributedReLU(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(value)
