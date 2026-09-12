from __future__ import annotations

import torch
import torch.nn as nn


class TemporalScaleTransformer(nn.Module):
    """Reassign a temporal feature sequence to a shorter causal scale.

    The learned mode follows SSNN: spatial/channel descriptors predict a
    per-sample distribution over output steps, while the summed feature state
    is redistributed without discarding its total magnitude.  ``truncate`` is
    the parameter-free control and keeps the most recent causal states.
    """

    def __init__(self, input_steps: int, output_steps: int, kind: str) -> None:
        super().__init__()
        if input_steps < output_steps or output_steps < 1:
            raise ValueError("Temporal scales require input_steps >= output_steps >= 1")
        if kind not in {"learned", "truncate"}:
            raise ValueError("Temporal transformer kind must be learned or truncate")
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.kind = kind
        self.score = (
            nn.Linear(self.input_steps, self.output_steps, bias=False)
            if kind == "learned" and input_steps != output_steps
            else None
        )
        if self.score is not None:
            nn.init.zeros_(self.score.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError("TemporalScaleTransformer requires T x B x C x H x W")
        if value.shape[0] != self.input_steps:
            raise ValueError(
                f"Expected {self.input_steps} temporal states, got {value.shape[0]}"
            )
        if self.input_steps == self.output_steps:
            return value
        if self.kind == "truncate":
            # Inputs are chronological, so retain the states closest to the
            # supervised current frame rather than future or earliest states.
            return value[-self.output_steps:]
        descriptor = value.mean(dim=(2, 3, 4)).transpose(0, 1)
        weights = self.score(descriptor).softmax(dim=1).transpose(0, 1)
        total = value.sum(dim=0)
        return weights[:, :, None, None, None] * total[None]
