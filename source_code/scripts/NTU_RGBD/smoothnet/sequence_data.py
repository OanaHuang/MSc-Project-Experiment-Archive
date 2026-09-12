from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PoseSequences:
    pred: np.ndarray
    gt: np.ndarray
    visibility: np.ndarray
    head_length: np.ndarray
    sample_ids: np.ndarray
    frame_indices: np.ndarray
    groups: tuple[np.ndarray, ...]


def load_sequences(path: Path) -> PoseSequences:
    arrays = np.load(path)
    required = ("pred", "gt", "visibility", "head_length", "sample_ids", "frame_indices")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"{path} is missing sequence fields: {', '.join(missing)}")
    sample_ids = arrays["sample_ids"].astype(str)
    frame_indices = arrays["frame_indices"].astype(np.int64)
    groups = []
    for sample_id in dict.fromkeys(sample_ids.tolist()):
        indices = np.flatnonzero(sample_ids == sample_id)
        indices = indices[np.argsort(frame_indices[indices], kind="stable")]
        if len(indices) and np.any(np.diff(frame_indices[indices]) != 1):
            raise ValueError(f"non-contiguous frames in {sample_id}")
        groups.append(indices)
    return PoseSequences(
        arrays["pred"].astype(np.float32), arrays["gt"].astype(np.float32),
        arrays["visibility"].astype(np.float32), arrays["head_length"].astype(np.float32),
        sample_ids, frame_indices, tuple(groups),
    )


class PoseWindowDataset(Dataset):
    def __init__(self, sequences: PoseSequences, window_size: int,
                 stride: int = 1) -> None:
        self.sequences = sequences
        self.window_size = int(window_size)
        self.windows = tuple(
            group[start:start + window_size]
            for group in sequences.groups
            for start in range(0, len(group) - window_size + 1, stride)
        )
        if not self.windows:
            raise ValueError("no complete pose windows are available")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rows = self.windows[index]
        return {
            "pred": torch.from_numpy(self.sequences.pred[rows]),
            "gt": torch.from_numpy(self.sequences.gt[rows]),
            "visibility": torch.from_numpy(self.sequences.visibility[rows]),
        }


@torch.no_grad()
def refine_sequences(model: torch.nn.Module, sequences: PoseSequences,
                     device: torch.device, coordinate_scale: float = 1.0) -> np.ndarray:
    model.eval()
    output = sequences.pred.copy()
    window = int(model.window_size)
    for group in sequences.groups:
        if len(group) < window:
            continue
        accumulated = np.zeros_like(sequences.pred[group])
        counts = np.zeros((len(group), 1, 1), dtype=np.float32)
        starts = list(range(len(group) - window + 1))
        for start in starts:
            value = torch.from_numpy(sequences.pred[group[start:start + window]])[None].to(device)
            refined = (model(value / coordinate_scale) * coordinate_scale)[0].cpu().numpy()
            accumulated[start:start + window] += refined
            counts[start:start + window] += 1
        output[group] = accumulated / np.maximum(counts, 1)
    return output
