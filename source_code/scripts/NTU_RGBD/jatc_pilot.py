from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


BETA_GRID = np.asarray((0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95), dtype=np.float32)


@dataclass(frozen=True)
class PilotSequences:
    pred: np.ndarray
    confidence: np.ndarray
    gt: np.ndarray
    visibility: np.ndarray
    head_length: np.ndarray
    sample_ids: np.ndarray
    frame_indices: np.ndarray
    groups: tuple[np.ndarray, ...]


def load_pilot_sequences(path: Path, *, require_confidence: bool = False) -> PilotSequences:
    arrays = np.load(path)
    required = ("pred", "gt", "visibility", "head_length", "sample_ids", "frame_indices")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
    pred = arrays["pred"].astype(np.float32)
    if "confidence" in arrays:
        confidence = arrays["confidence"].astype(np.float32)
    elif require_confidence:
        raise ValueError(f"{path} has no heatmap confidence; re-export the T0 sequences")
    else:
        confidence = np.ones(pred.shape[:2], dtype=np.float32)
    sample_ids = arrays["sample_ids"].astype(str)
    frame_indices = arrays["frame_indices"].astype(np.int64)
    groups = []
    for sample_id in dict.fromkeys(sample_ids.tolist()):
        indices = np.flatnonzero(sample_ids == sample_id)
        indices = indices[np.argsort(frame_indices[indices], kind="stable")]
        if len(indices) and np.any(np.diff(frame_indices[indices]) != 1):
            raise ValueError(f"non-contiguous frames in {sample_id}")
        groups.append(indices)
    return PilotSequences(
        pred, confidence, arrays["gt"].astype(np.float32),
        arrays["visibility"].astype(np.float32), arrays["head_length"].astype(np.float32),
        sample_ids, frame_indices, tuple(groups),
    )


def causal_ema(values: np.ndarray, groups: tuple[np.ndarray, ...], beta: float | np.ndarray,
               dynamic_beta: np.ndarray | None = None) -> np.ndarray:
    output = values.copy()
    base = np.asarray(beta, dtype=np.float32)
    for group in groups:
        if not len(group):
            continue
        state = values[group[0]].copy()
        output[group[0]] = state
        for row in group[1:]:
            decay = dynamic_beta[row] if dynamic_beta is not None else base
            # A recurrent state must never be poisoned by one invalid gate.
            decay = np.where(np.isfinite(decay), decay, base)
            decay = np.clip(decay, 0.0, 0.98)
            state = decay[..., None] * state + (1.0 - decay[..., None]) * values[row]
            output[row] = state
    return output


def dynamic_decays(sequences: PilotSequences, base_beta: np.ndarray,
                   confidence_scale: float, velocity_scale: float) -> np.ndarray:
    decay = np.broadcast_to(base_beta[None], sequences.pred.shape[:2]).copy()
    for group in sequences.groups:
        if len(group) < 2:
            continue
        head = sequences.head_length[group].astype(np.float32, copy=True)
        valid_head = np.isfinite(head) & (head > 1e-6)
        # Head/neck visibility occasionally makes the normalization unavailable.
        # Use a video-local robust scale; if the whole video is invalid, retain
        # the static per-joint decay by using a neutral finite scale.
        replacement = float(np.median(head[valid_head])) if np.any(valid_head) else 1.0
        head[~valid_head] = replacement
        scale = head[:, None]
        velocity = np.linalg.norm(np.diff(sequences.pred[group], axis=0), axis=2) / scale[1:]
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
        confidence = np.nan_to_num(
            sequences.confidence[group[1:]], nan=0.0, posinf=1.0, neginf=0.0,
        )
        confidence = np.clip(confidence, 0.0, 1.0)
        decay[group[1:]] += confidence_scale * (1.0 - confidence) - velocity_scale * velocity
    decay = np.where(np.isfinite(decay), decay, np.broadcast_to(base_beta[None], decay.shape))
    return np.clip(decay, 0.0, 0.98)


class PilotWindowDataset(Dataset):
    def __init__(self, sequences: PilotSequences, window_size: int) -> None:
        self.sequences = sequences
        self.windows = tuple(
            group[start:start + window_size]
            for group in sequences.groups
            for start in range(len(group) - window_size + 1)
        )
        if not self.windows:
            raise ValueError("no complete JATC windows are available")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rows = self.windows[index]
        return {
            "pred": torch.from_numpy(self.sequences.pred[rows]),
            "confidence": torch.from_numpy(self.sequences.confidence[rows]),
            "gt": torch.from_numpy(self.sequences.gt[rows]),
            "visibility": torch.from_numpy(self.sequences.visibility[rows]),
            "head_length": torch.from_numpy(self.sequences.head_length[rows]),
        }


class JointAdaptiveCoordinateLIF(nn.Module):
    def __init__(self, num_joints: int, base_beta: np.ndarray, hidden_channels: int = 16,
                 beta_min: float = 0.5, beta_max: float = 0.98,
                 residual_gate_init: float = 0.1) -> None:
        super().__init__()
        if base_beta.shape != (num_joints,):
            raise ValueError("base_beta must contain one value per joint")
        fraction = np.clip((base_beta - beta_min) / (beta_max - beta_min), 1e-4, 1 - 1e-4)
        self.base_logit = nn.Parameter(torch.from_numpy(np.log(fraction / (1 - fraction))).float())
        self.conditioner = nn.Sequential(
            nn.Linear(3, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, 1),
        )
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)
        gate = np.clip(residual_gate_init, 1e-4, 1 - 1e-4)
        self.gate_logit = nn.Parameter(torch.full((num_joints,), float(np.log(gate / (1 - gate)))))
        self.beta_min, self.beta_max = float(beta_min), float(beta_max)

    def forward(self, coordinates: torch.Tensor, confidence: torch.Tensor,
                head_length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # coordinates [B,T,J,2], confidence [B,T,J], head_length [B,T]
        state = coordinates[:, 0]
        previous = coordinates[:, 0]
        previous_velocity = torch.zeros_like(confidence[:, 0])
        outputs, decays = [state], []
        base = self.base_logit[None, :]
        gate = self.gate_logit.sigmoid()[None, :, None]
        for step in range(1, coordinates.shape[1]):
            scale = head_length[:, step, None].clamp_min(1e-6)
            velocity = (coordinates[:, step] - previous).norm(dim=-1) / scale
            acceleration = (velocity - previous_velocity).abs()
            features = torch.stack((confidence[:, step], velocity, acceleration), dim=-1)
            beta = self.beta_min + (self.beta_max - self.beta_min) * (
                base + self.conditioner(features).squeeze(-1)
            ).sigmoid()
            state = beta[..., None] * state + (1.0 - beta[..., None]) * coordinates[:, step]
            outputs.append(coordinates[:, step] + gate * (state - coordinates[:, step]))
            decays.append(beta)
            previous, previous_velocity = coordinates[:, step], velocity
        if not decays:
            decays = [coordinates.new_zeros(coordinates.shape[0], coordinates.shape[2])]
        return torch.stack(outputs, dim=1), torch.stack(decays, dim=1)


@torch.no_grad()
def refine_causally(model: JointAdaptiveCoordinateLIF, sequences: PilotSequences,
                    device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    prediction = sequences.pred.copy()
    beta_rows = np.zeros(sequences.pred.shape[:2], dtype=np.float32)
    for group in sequences.groups:
        coordinates = torch.from_numpy(sequences.pred[group])[None].to(device)
        confidence = torch.from_numpy(sequences.confidence[group])[None].to(device)
        head = torch.from_numpy(sequences.head_length[group])[None].to(device)
        refined, beta = model(coordinates, confidence, head)
        prediction[group] = refined[0].cpu().numpy()
        if len(group) > 1:
            beta_rows[group[1:]] = beta[0].cpu().numpy()
    return prediction, beta_rows
