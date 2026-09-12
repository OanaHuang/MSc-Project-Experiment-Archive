from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .masks import MaskConfig, generate_occlusion_mask


@dataclass(frozen=True)
class PoseSequences:
    pred: np.ndarray
    confidence: np.ndarray
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
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    pred = arrays["pred"].astype(np.float32)
    confidence = (
        arrays["confidence"].astype(np.float32)
        if "confidence" in arrays else np.ones(pred.shape[:2], dtype=np.float32)
    )
    sample_ids = arrays["sample_ids"].astype(str)
    frame_indices = arrays["frame_indices"].astype(np.int64)
    groups = []
    for sample_id in dict.fromkeys(sample_ids.tolist()):
        rows = np.flatnonzero(sample_ids == sample_id)
        rows = rows[np.argsort(frame_indices[rows], kind="stable")]
        breaks = np.flatnonzero(np.diff(frame_indices[rows]) != 1) + 1
        groups.extend(part for part in np.split(rows, breaks) if len(part))
    return PoseSequences(
        pred, confidence, arrays["gt"].astype(np.float32),
        arrays["visibility"].astype(np.float32), arrays["head_length"].astype(np.float32),
        sample_ids, frame_indices, tuple(groups),
    )


class OcclusionWindowDataset(Dataset):
    def __init__(self, sequences: PoseSequences, window_size: int = 16,
                 stride: int = 1, seed: int = 42,
                 mask_config: MaskConfig = MaskConfig()) -> None:
        self.sequences = sequences
        self.window_size = int(window_size)
        self.seed = int(seed)
        self.epoch = 0
        self.mask_config = mask_config
        self.windows = tuple(
            group[start:start + self.window_size]
            for group in sequences.groups
            for start in range(0, len(group) - self.window_size + 1, stride)
        )
        if not self.windows:
            raise ValueError("no complete contiguous pose windows are available")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rows = self.windows[index]
        visibility = torch.from_numpy(self.sequences.visibility[rows])
        generator = torch.Generator().manual_seed(self.seed + self.epoch * len(self) + index)
        observed = generate_occlusion_mask(visibility, self.mask_config, generator)
        prediction = torch.from_numpy(self.sequences.pred[rows])
        return {
            "pred": prediction * observed[..., None],
            "clean_pred": prediction,
            "confidence": torch.from_numpy(self.sequences.confidence[rows]),
            "gt": torch.from_numpy(self.sequences.gt[rows]),
            "visibility": visibility,
            "observed": observed,
            "head_length": torch.from_numpy(self.sequences.head_length[rows]),
        }
