from __future__ import annotations

import math

import torch
import torch.nn as nn


class IntegerSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, max_spikes: int) -> torch.Tensor:
        ctx.save_for_backward(value)
        ctx.max_spikes = int(max_spikes)
        return torch.round(torch.clamp(value, 0.0, float(max_spikes)))

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        (value,) = ctx.saved_tensors
        active = (value >= 0.0) & (value <= float(ctx.max_spikes))
        return gradient * active.to(gradient.dtype), None


class MultiStepILIF(nn.Module):
    """Integer-valued LIF neuron for T x B x C x H x W tensors."""

    def __init__(self, decay: float = 0.90, threshold: float = 1.0,
                 max_spikes: int = 4, detach_reset: bool = True,
                 membrane_readout: bool = False,
                 membrane_readout_init: float = 0.01,
                 learnable_decay: bool = False,
                 learnable_initial_membrane: bool = False,
                 initial_membrane_scale: float = 0.25) -> None:
        super().__init__()
        if not 0.5 < decay < 0.99:
            raise ValueError("decay must lie strictly between 0.5 and 0.99")
        self.learnable_decay = bool(learnable_decay)
        if self.learnable_decay:
            fraction = (float(decay) - 0.5) / 0.49
            self.decay_logit = nn.Parameter(torch.tensor(math.log(fraction / (1.0 - fraction))))
        else:
            self.decay = float(decay)
        self.threshold = float(threshold)
        self.max_spikes = int(max_spikes)
        self.detach_reset = bool(detach_reset)
        self.membrane_readout = bool(membrane_readout)
        if not 0.0 < membrane_readout_init < 1.0:
            raise ValueError("membrane_readout_init must lie strictly between zero and one")
        if self.membrane_readout:
            initial = math.log(membrane_readout_init / (1.0 - membrane_readout_init))
            self.membrane_readout_logit = nn.Parameter(torch.tensor(initial))
        self.learnable_initial_membrane = bool(learnable_initial_membrane)
        self.initial_membrane_scale = float(initial_membrane_scale)
        if self.learnable_initial_membrane:
            self.initial_membrane_logit = nn.Parameter(torch.zeros(()))

    def current_decay(self) -> torch.Tensor | float:
        if not self.learnable_decay:
            return self.decay
        return 0.5 + 0.49 * self.decay_logit.sigmoid()

    def initial_membrane(self, reference: torch.Tensor) -> torch.Tensor:
        if not self.learnable_initial_membrane:
            return torch.zeros_like(reference)
        value = self.initial_membrane_scale * self.initial_membrane_logit.tanh()
        return torch.zeros_like(reference) + value

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError("Expected T x B x C x H x W input")
        membrane = self.initial_membrane(value[0])
        previous = torch.zeros_like(value[0])
        spikes = []
        decay = self.current_decay()
        for step in range(value.shape[0]):
            if step == 0:
                membrane = membrane + value[step]
            else:
                reset = previous.detach() if self.detach_reset else previous
                membrane = (membrane - reset * self.threshold) * decay + value[step]
            previous = IntegerSpike.apply(membrane / self.threshold, self.max_spikes)
            output = previous
            if self.membrane_readout:
                residual = membrane / self.threshold - previous
                output = previous + self.membrane_readout_logit.sigmoid() * residual
            spikes.append(output)
        return torch.stack(spikes)
