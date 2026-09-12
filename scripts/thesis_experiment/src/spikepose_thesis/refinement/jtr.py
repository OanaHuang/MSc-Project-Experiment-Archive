from __future__ import annotations

import torch
import torch.nn as nn


class JointwiseTemporalRefinement(nn.Module):
    def __init__(self, joints: int = 16, jointwise: bool = True,
                 alpha_min: float = 0.05, alpha_max: float = 0.95):
        super().__init__()
        self.joints = int(joints)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.logits = nn.Parameter(torch.zeros(joints if jointwise else 1))

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_min + (self.alpha_max - self.alpha_min) * self.logits.sigmoid()

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim != 4 or coordinates.shape[2:] != (self.joints, 2):
            raise ValueError("JTR expects batch x time x joints x 2")
        alpha = self.alpha
        if alpha.numel() == 1:
            alpha = alpha.expand(self.joints)
        alpha = alpha.reshape(1, self.joints, 1)
        states = [coordinates[:, 0]]
        for step in range(1, coordinates.shape[1]):
            states.append(alpha * states[-1] + (1.0 - alpha) * coordinates[:, step])
        return torch.stack(states, dim=1)


class DynamicTemporalRefinement(nn.Module):
    """Causal joint-wise EMA whose decay responds to observed motion."""
    def __init__(self, joints: int = 16, alpha_min: float = 0.05,
                 alpha_max: float = 0.95):
        super().__init__()
        self.joints = joints
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.bias = nn.Parameter(torch.zeros(joints))
        self.motion_scale = nn.Parameter(torch.zeros(joints))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        states = [coordinates[:, 0]]
        for step in range(1, coordinates.shape[1]):
            motion = (coordinates[:, step] - coordinates[:, step - 1]).norm(dim=-1)
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(
                self.bias[None] + self.motion_scale[None] * motion
            )
            states.append(
                alpha[..., None] * states[-1]
                + (1.0 - alpha[..., None]) * coordinates[:, step]
            )
        return torch.stack(states, dim=1)


class CausalTCNLite(nn.Module):
    def __init__(self, joints: int = 16, hidden_channels: int = 64,
                 layers: int = 2, kernel_size: int = 5):
        super().__init__()
        self.joints = joints
        channels = joints * 2
        self.input = nn.Conv1d(channels, hidden_channels, 1)
        self.depthwise = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                      groups=hidden_channels)
            for _ in range(layers)
        ])
        self.pointwise = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, 1) for _ in range(layers)
        ])
        self.output = nn.Conv1d(hidden_channels, channels, 1)
        self.kernel_size = kernel_size

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        batch, steps, joints, axes = coordinates.shape
        value = coordinates.reshape(batch, steps, joints * axes).transpose(1, 2)
        hidden = torch.relu(self.input(value))
        for depthwise, pointwise in zip(self.depthwise, self.pointwise):
            padded = torch.nn.functional.pad(hidden, (self.kernel_size - 1, 0))
            hidden = hidden + torch.relu(pointwise(depthwise(padded)))
        residual = self.output(hidden).transpose(1, 2).reshape(batch, steps, joints, axes)
        return coordinates + residual


def jtr_loss(prediction: torch.Tensor, target: torch.Tensor,
             visibility: torch.Tensor) -> torch.Tensor:
    def masked_smooth_l1(left, right, mask):
        if not torch.isfinite(left).all():
            raise FloatingPointError("non-finite JTR values cannot be scored")
        finite_target = torch.isfinite(right).all(dim=-1)
        valid = (mask > 0) & torch.isfinite(mask) & finite_target
        # torch loss functions propagate NaN before masking (NaN * 0 is NaN).
        # Substitute only invalid, unsupervised targets before computing loss.
        safe_right = torch.where(finite_target[..., None], right, left.detach())
        loss = torch.nn.functional.smooth_l1_loss(
            left, safe_right, reduction="none",
        ).mean(-1)
        weights = valid.to(loss.dtype)
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)

    def masked_absolute(left, right, mask):
        if not torch.isfinite(left).all():
            raise FloatingPointError("non-finite JTR values cannot be scored")
        finite_target = torch.isfinite(right)
        valid = (mask > 0) & torch.isfinite(mask) & finite_target
        safe_right = torch.where(finite_target, right, left.detach())
        loss = (left - safe_right).abs()
        weights = valid.to(loss.dtype)
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)

    position = masked_smooth_l1(prediction, target, visibility)
    pred_v, target_v = prediction.diff(dim=1), target.diff(dim=1)
    vel_mask = visibility[:, 1:] * visibility[:, :-1]
    velocity = masked_smooth_l1(pred_v, target_v, vel_mask)
    pred_a, target_a = prediction.diff(n=2, dim=1), target.diff(n=2, dim=1)
    acc_mask = visibility[:, 2:] * visibility[:, 1:-1] * visibility[:, :-2]
    acceleration = masked_smooth_l1(pred_a, target_a, acc_mask)
    magnitude = masked_absolute(
        pred_v.norm(dim=-1), target_v.norm(dim=-1), vel_mask,
    )
    return position + 0.25 * velocity + 0.05 * acceleration + 0.05 * magnitude
