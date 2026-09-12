from __future__ import annotations

from .ilif import MultiStepILIF


class MultiStepLIF(MultiStepILIF):
    """Binary LIF variant."""

    def __init__(self, decay: float = 0.90, threshold: float = 1.0,
                 max_spikes: int = 1, detach_reset: bool = True,
                 **kwargs) -> None:
        super().__init__(decay, threshold, 1, detach_reset, **kwargs)
