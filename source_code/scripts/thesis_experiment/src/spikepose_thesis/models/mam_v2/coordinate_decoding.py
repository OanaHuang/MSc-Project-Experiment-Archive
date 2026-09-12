"""User-proposed response-heatmap decoding; no change to stored/fused heatmaps.

The expm1 formula is a BCIR-inspired discrete adaptation, not full BCIR.
All coordinates and offsets here are in heatmap pixels, ordered (x, y).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .cell import MotionAlignedMembraneV2


@dataclass(frozen=True)
class DecoderSpec:
    method: str = "softmax"
    beta: float = 1.0
    alpha: float = 1.0
    min_mass: float = 1e-8
    dark_kernel: int = 11

    def __post_init__(self):
        if self.method not in {"softmax", "relu", "power", "expm1", "dark"}:
            raise ValueError(f"Unknown coordinate decoder: {self.method}")
        if not all(math.isfinite(x) and x > 0 for x in (self.beta, self.alpha, self.min_mass)):
            raise ValueError("beta, alpha and min_mass must be finite and positive")
        if self.dark_kernel < 3 or self.dark_kernel % 2 == 0:
            raise ValueError("DARK kernel must be odd and >= 3")


def decode_coordinates(heatmap: torch.Tensor, spec: DecoderSpec):
    """Return xy, uncalibrated raw peak, validity. Supports [..., J, H, W].

Use FP32 inside AMP (FP64 is retained for numerical tests). Invalid zero/negative
maps return the grid centre and q=0 as a sentinel, never as motion evidence.
Nonfinite raw heatmaps fail fast because fusion could not safely use them either.
"""
    if heatmap.ndim < 3:
        raise ValueError("Expected [..., J, H, W] heatmaps")
    if not bool(torch.isfinite(heatmap).all()):
        raise FloatingPointError("Nonfinite raw heatmap in mam_softmax_fix")
    dtype = torch.float64 if heatmap.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=heatmap.device.type, enabled=False):
        h = heatmap.to(dtype)
        height, width = h.shape[-2:]
        flat = h.flatten(-2)
        response = flat.clamp_min(0)
        mass = response.sum(-1)
        valid = mass > spec.min_mass
        peak = response.amax(-1, keepdim=True)
        ys, xs = torch.meshgrid(
            torch.arange(height, device=h.device, dtype=dtype),
            torch.arange(width, device=h.device, dtype=dtype), indexing="ij",
        )
        grid = torch.stack((xs.flatten(), ys.flatten()), -1)
        if spec.method == "dark":
            from spikepose_thesis.data.mpii.core.geometry import _dark_refine
            arrays = h.detach().float().cpu().numpy().reshape(-1, height, width)
            indices = arrays.reshape(len(arrays), -1).argmax(-1)
            initial = np.stack((indices % width, indices // width), -1).astype(np.float32)
            refined = _dark_refine(arrays, initial, kernel=spec.dark_kernel)
            xy = torch.as_tensor(refined, device=h.device, dtype=dtype).reshape(*flat.shape[:-1], 2)
        elif spec.method == "softmax":
            xy = (flat * spec.beta).softmax(-1) @ grid
        else:
            if spec.method in {"relu", "power"}:
                # Dividing by the common peak before exponentiation preserves
                # the specified probability exactly and avoids overflow.
                scaled = response / torch.where(peak > 0, peak, torch.ones_like(peak))
                weights = scaled.pow(spec.alpha if spec.method == "power" else 1.0)
            else:
                u = response * spec.beta
                maximum = u.amax(-1, keepdim=True)
                # exp(-maximum)*expm1(u), without overflow or subtracting two
                # almost equal large numbers. Do NOT use expm1(u-maximum).
                weights = torch.exp(u - maximum) * (-torch.expm1(-u))
            denominator = weights.sum(-1, keepdim=True)
            probability = weights / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
            xy = probability @ grid
        centre = h.new_tensor([(width - 1) / 2, (height - 1) / 2])
        xy = torch.where(valid.unsqueeze(-1), xy, centre)
        q = torch.where(valid.unsqueeze(-1), peak, torch.zeros_like(peak))
        return xy, q, valid


class CoordinateDecodedMAM(MotionAlignedMembraneV2):
    """Same parameters and equations as MAM V2; explicit invalid-pair fallback."""

    def _soft_coordinates_and_confidence(self, heatmap):
        xy, q, _ = decode_coordinates(heatmap, self.decoder_spec)
        return xy, q

    def _motion_features(self, current_xy, previous_xy, current_confidence, previous_confidence):
        coarse, features = super()._motion_features(
            current_xy, previous_xy, current_confidence, previous_confidence,
        )
        pair_valid = (current_confidence > 0) & (previous_confidence > 0)
        coarse = torch.where(pair_valid, coarse, torch.zeros_like(coarse))
        features = torch.cat((features[..., :6], coarse, coarse.norm(dim=-1, keepdim=True), features[..., 9:]), -1)
        return coarse, features

    def _decay(self, features, reference):
        decay = super()._decay(features, reference)
        pair_valid = (features[..., 4] > 0) & (features[..., 5] > 0)
        # If either coordinate is undefined, output/write the current raw map.
        # Identical rule for every decoder, including the retrained softmax.
        return decay * pair_valid[..., None, None]


def install_coordinate_decoder(model, options):
    if not isinstance(model.mam, MotionAlignedMembraneV2):
        raise TypeError("mam_softmax_fix requires MAM V2")
    spec = DecoderSpec(**options)
    if not model.mam.config.use_motion_token or model.mam.config.use_kpa or model.mam.config.use_tpa:
        raise ValueError("Initial softmax-fix study requires the original motion tokens without priors")
    replacement = CoordinateDecodedMAM(model.mam.joints, model.mam.config)
    replacement.load_state_dict(model.mam.state_dict(), strict=True)
    replacement.decoder_spec = spec
    model.mam = replacement
