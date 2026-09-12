from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import torch
from torch.utils.data import Sampler


class _MAMSequence(Protocol):
    def forward_sequence(self, heatmaps: torch.Tensor): ...


@dataclass(frozen=True)
class ResetPolicyOutput:
    heatmap: torch.Tensor
    model_invocations: int
    reset_positions: tuple[int, ...]


def trained_window_starts(frame_count: int, trained_steps: int) -> tuple[int, ...]:
    """Return fixed-context starts with one overlapping full-length tail."""
    frame_count = int(frame_count)
    trained_steps = int(trained_steps)
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if trained_steps <= 0:
        raise ValueError("trained_steps must be positive")
    if frame_count <= trained_steps:
        return (0,)
    starts = list(range(0, frame_count - trained_steps + 1, trained_steps))
    final_start = frame_count - trained_steps
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def trained_window_stitch_positions(
    frame_count: int, trained_steps: int,
) -> tuple[int, ...]:
    """Return output positions where reconstruction switches invocations."""
    starts = trained_window_starts(frame_count, trained_steps)
    filled_until = 0
    stitches = []
    for start in starts:
        first_retained = max(start, filled_until)
        if first_retained < frame_count:
            stitches.append(first_retained)
        filled_until = max(filled_until, min(start + trained_steps, frame_count))
    return tuple(stitches)


def _prediction_heatmap(output) -> torch.Tensor:
    if hasattr(output, "heatmap"):
        return output.heatmap
    if isinstance(output, tuple):
        return output[0]
    return output


def apply_mam_reset_policy(
    mam: _MAMSequence,
    spatial_heatmaps: torch.Tensor,
    *,
    policy: str,
    trained_steps: int,
) -> ResetPolicyOutput:
    """Apply MAM to one complete video under a declared reset policy.

    ``spatial_heatmaps`` must be T x 1 x J x H x W.  The video-boundary
    policy invokes MAM once.  The trained-window policy batches independent
    fixed-length invocations and reconstructs exhaustive, non-duplicated
    frame predictions using the same overlapping-tail rule as the frozen
    full-video evaluator.
    """
    if spatial_heatmaps.ndim != 5 or spatial_heatmaps.shape[1] != 1:
        raise ValueError(
            "Reset-policy evaluation expects T x 1 x J x H x W heatmaps"
        )
    if len(spatial_heatmaps) == 0:
        raise ValueError("A video must contain at least one heatmap")
    if policy == "video_boundary":
        prediction = _prediction_heatmap(mam.forward_sequence(spatial_heatmaps))
        return ResetPolicyOutput(prediction, 1, (0,))
    if policy != "trained_window":
        raise ValueError(f"Unknown MAM reset policy: {policy!r}")

    starts = trained_window_starts(len(spatial_heatmaps), trained_steps)
    windows = torch.cat([
        spatial_heatmaps[start:min(start + trained_steps, len(spatial_heatmaps))]
        for start in starts
    ], dim=1)
    window_prediction = _prediction_heatmap(mam.forward_sequence(windows))
    prediction = torch.empty_like(spatial_heatmaps)
    filled_until = 0
    for lane, start in enumerate(starts):
        end = min(start + window_prediction.shape[0], len(spatial_heatmaps))
        keep_from = max(filled_until - start, 0)
        if start + keep_from < end:
            prediction[start + keep_from:end, 0] = window_prediction[
                keep_from:keep_from + end - (start + keep_from), lane
            ]
        filled_until = max(filled_until, end)
    if filled_until != len(spatial_heatmaps):
        raise RuntimeError(
            f"Reset-policy reconstruction covered {filled_until}/"
            f"{len(spatial_heatmaps)} frames"
        )
    return ResetPolicyOutput(prediction, len(starts), starts)


class SameVideoChunkBatchSampler(Sampler[list[int]]):
    """Batch consecutive temporal windows without mixing source videos."""

    def __init__(self, frame_index: list[tuple], batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._batches: list[list[int]] = []
        current_sample = None
        current_indices: list[int] = []
        for index, item in enumerate(frame_index):
            sample_index = int(item[0])
            if current_sample is not None and sample_index != current_sample:
                self._append_group(current_indices, batch_size)
                current_indices = []
            current_sample = sample_index
            current_indices.append(index)
        if current_indices:
            self._append_group(current_indices, batch_size)

    def _append_group(self, indices: list[int], batch_size: int) -> None:
        self._batches.extend([
            indices[start:start + batch_size]
            for start in range(0, len(indices), batch_size)
        ])

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


__all__ = [
    "ResetPolicyOutput",
    "SameVideoChunkBatchSampler",
    "apply_mam_reset_policy",
    "trained_window_starts",
    "trained_window_stitch_positions",
]
