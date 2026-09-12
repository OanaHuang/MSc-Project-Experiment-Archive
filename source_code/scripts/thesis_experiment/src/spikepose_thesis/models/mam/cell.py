from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionAlignedMembrane(nn.Module):
    """Causal heatmap-level Motion-Aligned Membrane memory.

    State is local to one ``forward_sequence`` call.  The caller therefore
    owns clip boundaries and cannot accidentally leak state between samples.
    """

    def __init__(self, joints: int, kind: str, *, decay: float = 0.90,
                 decay_min: float = 0.50, decay_max: float = 0.99,
                 shared_decay: bool = False, use_current_residual: bool = True) -> None:
        super().__init__()
        valid = {"reset", "carry", "plif", "align", "mam", "mamp", "mamtl"}
        kind = kind.lower()
        if kind not in valid:
            raise ValueError(f"Unknown membrane memory kind: {kind}")
        if not 0.0 < decay_min < decay_max < 1.0:
            raise ValueError("MAM decay bounds must satisfy 0 < min < max < 1")
        self.joints = int(joints)
        self.kind = kind
        self.decay_min = float(decay_min)
        self.decay_max = float(decay_max)
        self.use_current_residual = bool(use_current_residual)
        channels = 1 if shared_decay else joints
        probability = (float(decay) - decay_min) / (decay_max - decay_min)
        probability = min(max(probability, 1e-4), 1.0 - 1e-4)
        initial = math.log(probability / (1.0 - probability))
        value = torch.full((1, channels, 1, 1), initial)
        self.decay_logit = nn.Parameter(value, requires_grad=kind not in {"reset", "carry"})
        self.offset = nn.Sequential(
            nn.Conv2d(joints * 2, joints, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(joints, joints * 2, 1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(joints * 3, joints, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(joints, joints, 1),
        )
        self.current_residual = nn.Conv2d(joints, joints, 1)
        self.predictor = nn.Conv2d(joints, joints, 3, padding=1)

    @property
    def aligned(self) -> bool:
        return self.kind in {"align", "mam", "mamp", "mamtl"}

    @property
    def adaptive(self) -> bool:
        return self.kind in {"mam", "mamp", "mamtl"}

    @property
    def predictive(self) -> bool:
        return self.kind in {"mamp", "mamtl"}

    def _decay(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        static = self.decay_min + (self.decay_max - self.decay_min) * self.decay_logit.sigmoid()
        if not self.adaptive:
            return static.expand(current.shape[0], self.joints, 1, 1)
        confidence = current.flatten(2).amax(-1).unsqueeze(-1).unsqueeze(-1)
        motion = (current - previous).abs()
        dynamic = self.gate(torch.cat((current, previous, motion), dim=1)).sigmoid()
        # High confidence favours current evidence; motion lets the gate shorten
        # memory when the pose changes quickly.
        dynamic = (dynamic + confidence.sigmoid()) * 0.5
        return self.decay_min + (self.decay_max - self.decay_min) * dynamic

    @staticmethod
    def _soft_coordinates(heatmap: torch.Tensor) -> torch.Tensor:
        batch, joints, height, width = heatmap.shape
        probability = heatmap.flatten(2).softmax(-1)
        y, x = torch.meshgrid(
            torch.arange(height, device=heatmap.device, dtype=heatmap.dtype),
            torch.arange(width, device=heatmap.device, dtype=heatmap.dtype),
            indexing="ij",
        )
        coordinates = torch.stack((x.flatten(), y.flatten()), dim=-1)
        return probability @ coordinates

    @staticmethod
    def _warp(value: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        batch, joints, height, width = value.shape
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=value.device, dtype=value.dtype),
            torch.linspace(-1, 1, width, device=value.device, dtype=value.dtype),
            indexing="ij",
        )
        grid = torch.stack((x, y), dim=-1).reshape(1, 1, height, width, 2)
        shift = displacement.clone()
        shift[..., 0] *= 2.0 / max(width - 1, 1)
        shift[..., 1] *= 2.0 / max(height - 1, 1)
        # grid_sample maps output positions back to the source.
        grid = grid - shift.reshape(batch, joints, 1, 1, 2)
        warped = F.grid_sample(
            value.reshape(batch * joints, 1, height, width),
            grid.reshape(batch * joints, height, width, 2),
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        return warped.reshape(batch, joints, height, width)

    def forward_sequence(self, heatmaps: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if heatmaps.ndim != 5:
            raise ValueError("MAM expects T x B x J x H x W heatmaps")
        state = previous_current = None
        outputs, decays, offsets, predictions = [], [], [], []
        for current in heatmaps.unbind(0):
            if state is None or self.kind == "reset":
                aligned = current.new_zeros(current.shape)
                offset = current.new_zeros((current.shape[0], self.joints, 2))
                decay = current.new_zeros((current.shape[0], self.joints, 1, 1))
                state = current
            else:
                offset = current.new_zeros((current.shape[0], self.joints, 2))
                aligned = state
                if self.aligned:
                    coarse = self._soft_coordinates(current) - self._soft_coordinates(previous_current)
                    residual = self.offset(torch.cat((current, previous_current), dim=1))
                    residual = residual.reshape(current.shape[0], self.joints, 2).tanh()
                    offset = coarse + (residual if self.adaptive else 0.0)
                    aligned = self._warp(state, offset)
                decay = self._decay(current, aligned)
                candidate = current
                if self.adaptive and self.use_current_residual:
                    candidate = candidate + self.current_residual(current)
                state = decay * aligned + (1.0 - decay) * candidate
            outputs.append(state)
            decays.append(decay)
            offsets.append(offset)
            if self.predictive:
                predictions.append(self.predictor(state))
            previous_current = current
        auxiliary = {
            "decay": torch.stack(decays),
            "offset": torch.stack(offsets),
        }
        if predictions:
            auxiliary["prediction_next"] = torch.stack(predictions)
        return torch.stack(outputs), auxiliary
