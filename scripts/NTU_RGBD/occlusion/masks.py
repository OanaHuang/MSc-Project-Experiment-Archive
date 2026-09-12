from __future__ import annotations

from dataclasses import dataclass

import torch


NTU_LIMBS = (
    (4, 5, 6, 7, 21, 22), (8, 9, 10, 11, 23, 24),
    (12, 13, 14, 15), (16, 17, 18, 19),
)


@dataclass(frozen=True)
class MaskConfig:
    probability: float = 0.65
    short_probability: float = 0.45
    limb_probability: float = 0.25
    min_length: int = 1
    max_length: int = 8


def generate_occlusion_mask(
    visibility: torch.Tensor, config: MaskConfig, generator: torch.Generator,
) -> torch.Tensor:
    """Return an observation mask while retaining visible joints as supervision."""
    if visibility.ndim != 2:
        raise ValueError("visibility must have shape [T, J]")
    steps, joints = visibility.shape
    observed = (visibility > 0).clone()
    if torch.rand((), generator=generator).item() >= config.probability:
        return observed.to(torch.float32)
    use_limb = torch.rand((), generator=generator).item() < config.limb_probability
    if use_limb:
        candidates = [limb for limb in NTU_LIMBS if max(limb) < joints]
        selected = candidates[int(torch.randint(len(candidates), (), generator=generator))]
    else:
        selected = (int(torch.randint(joints, (), generator=generator)),)
    short = torch.rand((), generator=generator).item() < config.short_probability
    upper = min(3 if short else config.max_length, steps)
    lower = min(config.min_length, upper)
    length = int(torch.randint(lower, upper + 1, (), generator=generator))
    start = int(torch.randint(0, steps - length + 1, (), generator=generator))
    observed[start:start + length, list(selected)] = False
    return observed.to(torch.float32)
