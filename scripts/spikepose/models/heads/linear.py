from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHeatmapHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_joints: int,
                 output_size: tuple[int, int] | None,
                 aggregation: str = "mean", num_steps: int = 1,
                 feedback_edges: tuple[tuple[int, int], ...] = ()) -> None:
        super().__init__()
        self.output_size = output_size
        self.aggregation = aggregation
        self.num_joints = num_joints
        self.num_steps = num_steps
        self.output = nn.Conv2d(in_channels, num_joints, 1)
        self.joint_temporal_modes = {
            "joint_temporal", "joint_temporal_spatial",
            "joint_confidence_temporal", "joint_motion_temporal",
            "joint_aligned_temporal", "joint_aligned_motion_temporal",
            "joint_aligned_confidence_temporal",
            "joint_residual_correction", "joint_difference_temporal",
            "joint_decoupled_space_time",
            "motion_trend_fixed", "motion_trend_gated",
        }
        if aggregation in self.joint_temporal_modes:
            # One causal temporal distribution per joint. Uniform initialization
            # makes T2-T5 start from the same readout as an ordinary mean.
            self.temporal_logits = nn.Parameter(torch.zeros(num_steps, num_joints))
        if aggregation == "joint_temporal_spatial":
            adjacency = torch.eye(num_joints)
            for left, right in feedback_edges:
                adjacency[left, right] = 1.0
                adjacency[right, left] = 1.0
            adjacency = adjacency / adjacency.sum(1, keepdim=True).clamp_min(1.0)
            self.register_buffer("joint_adjacency", adjacency)
            self.spatial_scale = nn.Parameter(torch.full((num_joints,), 0.1))
        if aggregation in {"joint_motion_temporal", "joint_aligned_motion_temporal"}:
            self.motion_decay = nn.Parameter(torch.zeros(num_joints))
        if aggregation == "joint_residual_correction":
            self.residual_scale = nn.Parameter(torch.full((num_joints,), -2.0))
        if aggregation == "joint_difference_temporal":
            self.difference_scale = nn.Parameter(torch.zeros(num_joints))
        if aggregation == "joint_decoupled_space_time":
            adjacency = torch.eye(num_joints)
            for left, right in feedback_edges:
                adjacency[left, right] = 1.0
                adjacency[right, left] = 1.0
            adjacency = adjacency / adjacency.sum(1, keepdim=True).clamp_min(1.0)
            self.register_buffer("joint_adjacency", adjacency)
            self.spatial_scale = nn.Parameter(torch.full((num_joints,), 0.1))
        if aggregation == "motion_trend_fixed":
            self.motion_residual_scale = 0.25
        if aggregation == "motion_trend_gated":
            # Predict trust in the current frame and initialize conservatively.
            self.motion_gate = nn.Linear(3, 1)
            nn.init.zeros_(self.motion_gate.weight)
            nn.init.constant_(self.motion_gate.bias, math.log(0.8 / 0.2))

    def _resize(self, heatmaps: torch.Tensor) -> torch.Tensor:
        if self.output_size is None or heatmaps.shape[-2:] == self.output_size:
            return heatmaps
        leading = heatmaps.shape[:-3]
        resized = F.interpolate(
            heatmaps.reshape(-1, *heatmaps.shape[-3:]), self.output_size,
            mode="bilinear", align_corners=False,
        )
        return resized.reshape(*leading, *resized.shape[-3:])

    def _base_weights(self, steps: int) -> torch.Tensor:
        if steps > self.num_steps:
            raise ValueError(f"Expected at most {self.num_steps} temporal states, got {steps}")
        return self.temporal_logits[-steps:].softmax(0)

    def _confidence_weights(self, heatmaps: torch.Tensor) -> torch.Tensor:
        steps, batch, joints, height, width = heatmaps.shape
        confidence = heatmaps.flatten(-2).softmax(-1).amax(-1)
        scores = confidence + self.temporal_logits[-steps:, None, :]
        return scores.softmax(0).reshape(steps, batch, joints, 1, 1)

    def _motion_weights(self, heatmaps: torch.Tensor) -> torch.Tensor:
        steps, batch, joints, height, width = heatmaps.shape
        probabilities = heatmaps.flatten(-2).softmax(-1)
        y, x = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=heatmaps.device,
                           dtype=heatmaps.dtype),
            torch.linspace(0.0, 1.0, width, device=heatmaps.device,
                           dtype=heatmaps.dtype), indexing="ij",
        )
        coordinates = torch.stack((x.flatten(), y.flatten()), -1)
        coordinates = torch.einsum("tbjn,nc->tbjc", probabilities, coordinates)
        # The current frame has exactly zero distance from itself.  sqrt has an
        # infinite derivative at zero, which turns the motion-gate gradients
        # into NaNs even though the forward values are finite.
        squared_distance = (coordinates - coordinates[-1:]).square().sum(-1)
        distance = squared_distance.clamp_min(1e-12).sqrt()
        decay = F.softplus(self.motion_decay)[None, None, :]
        scores = self.temporal_logits[-steps:, None, :] - decay * distance
        return scores.softmax(0).reshape(steps, batch, joints, 1, 1)

    def _difference_weights(self, heatmaps: torch.Tensor) -> torch.Tensor:
        difference = heatmaps.new_zeros(heatmaps.shape[:3])
        difference[1:] = (heatmaps[1:] - heatmaps[:-1]).abs().mean((-2, -1))
        scores = (
            self.temporal_logits[:, None, :]
            + self.difference_scale[None, None, :] * difference
        )
        return scores.softmax(0)[..., None, None]

    def _residual_correction(self, aligned: torch.Tensor) -> torch.Tensor:
        history = aligned[:-1]
        weights = self.temporal_logits[:-1].softmax(0)[:, None, :, None, None]
        proposal = (history * weights).sum(0)
        current = aligned[-1]
        scale = self.residual_scale.sigmoid()[None, :, None, None]
        return current + scale * (proposal - current)

    @staticmethod
    def _heatmap_confidence(heatmaps: torch.Tensor) -> torch.Tensor:
        return heatmaps.flatten(-2).softmax(-1).amax(-1)

    def _motion_trend(
        self, heatmaps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forecast frame four from the first three without current-frame leakage."""
        if heatmaps.shape[0] != 4:
            raise ValueError("motion trend aggregation requires exactly four video frames")
        first, second, third, current = heatmaps.unbind(0)
        velocity = third - second
        acceleration = velocity - (second - first)
        history_forecast = third + velocity + 0.5 * acceleration
        if self.aggregation == "motion_trend_fixed":
            final = current + self.motion_residual_scale * (history_forecast - current)
            return final, history_forecast, current
        disagreement = (current - history_forecast).abs().mean((-2, -1))
        gate_features = torch.stack((
            self._heatmap_confidence(current),
            self._heatmap_confidence(history_forecast),
            disagreement,
        ), dim=-1)
        current_gate = self.motion_gate(gate_features).sigmoid().squeeze(-1)[..., None, None]
        final = current_gate * current + (1.0 - current_gate) * history_forecast
        return final, history_forecast, current

    def _align_to_current(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Translate each historical joint heatmap to the current soft coordinate."""
        steps, batch, joints, height, width = heatmaps.shape
        probabilities = heatmaps.flatten(-2).softmax(-1)
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=heatmaps.device,
                           dtype=heatmaps.dtype),
            torch.linspace(-1.0, 1.0, width, device=heatmaps.device,
                           dtype=heatmaps.dtype), indexing="ij",
        )
        coordinates = torch.stack((x.flatten(), y.flatten()), -1)
        coordinates = torch.einsum("tbjn,nc->tbjc", probabilities, coordinates)
        shifts = coordinates[-1:] - coordinates
        theta = heatmaps.new_zeros((steps * batch * joints, 2, 3))
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        # affine_grid defines sampling rather than visible-content offsets.
        theta[:, :, 2] = -shifts.reshape(-1, 2)
        flat = heatmaps.reshape(steps * batch * joints, 1, height, width)
        grid = F.affine_grid(theta, flat.shape, align_corners=True)
        aligned = F.grid_sample(
            flat, grid, mode="bilinear", padding_mode="border", align_corners=True,
        )
        # Do not numerically perturb the current-frame reference heatmaps.
        aligned = aligned.reshape(steps, batch, joints, height, width)
        return torch.cat((aligned[:-1], heatmaps[-1:]), dim=0)

    def _aggregate_joint_heatmaps(
        self, heatmaps: torch.Tensor, weighting_heatmaps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        steps = heatmaps.shape[0]
        weighting_heatmaps = heatmaps if weighting_heatmaps is None else weighting_heatmaps
        if self.aggregation in {
            "joint_confidence_temporal", "joint_aligned_confidence_temporal",
        }:
            weights = self._confidence_weights(weighting_heatmaps)
        elif self.aggregation in {
            "joint_motion_temporal", "joint_aligned_motion_temporal",
        }:
            weights = self._motion_weights(weighting_heatmaps)
        elif self.aggregation == "joint_difference_temporal":
            weights = self._difference_weights(weighting_heatmaps)
        else:
            weights = self._base_weights(steps)[:, None, :, None, None]
        result = (heatmaps * weights).sum(0)
        if self.aggregation in {"joint_temporal_spatial", "joint_decoupled_space_time"}:
            neighbours = torch.einsum("ij,bjhw->bihw", self.joint_adjacency, result)
            result = result + self.spatial_scale[None, :, None, None] * neighbours
        return result

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            return self._resize(self.output(value))
        if self.aggregation not in self.joint_temporal_modes:
            value = value.mean(0) if self.aggregation == "mean" else value[-1]
            return self._resize(self.output(value))
        steps, batch, channels, height, width = value.shape
        heatmaps = self.output(value.reshape(steps * batch, channels, height, width))
        heatmaps = heatmaps.reshape(steps, batch, self.num_joints, height, width)
        heatmaps = self._resize(heatmaps)
        if self.aggregation in {"motion_trend_fixed", "motion_trend_gated"}:
            return self._motion_trend(heatmaps)[0]
        if self.aggregation == "joint_residual_correction":
            return self._residual_correction(self._align_to_current(heatmaps))
        if self.aggregation in {
            "joint_difference_temporal", "joint_decoupled_space_time",
        }:
            return self._aggregate_joint_heatmaps(
                self._align_to_current(heatmaps), weighting_heatmaps=heatmaps,
            )
        if self.aggregation.startswith("joint_aligned_"):
            return self._aggregate_joint_heatmaps(
                self._align_to_current(heatmaps), weighting_heatmaps=heatmaps,
            )
        return self._aggregate_joint_heatmaps(heatmaps)

    def forward_with_intermediates(
        self, value: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        intermediate_modes = {
            "joint_aligned_temporal", "joint_residual_correction",
            "joint_difference_temporal", "joint_decoupled_space_time",
            "motion_trend_fixed", "motion_trend_gated",
        }
        if value.ndim != 5 or self.aggregation not in intermediate_modes:
            return self.forward(value), []
        steps, batch, channels, height, width = value.shape
        heatmaps = self.output(value.reshape(steps * batch, channels, height, width))
        heatmaps = self._resize(
            heatmaps.reshape(steps, batch, self.num_joints, height, width)
        )
        if self.aggregation in {"motion_trend_fixed", "motion_trend_gated"}:
            prediction, forecast, current = self._motion_trend(heatmaps)
            return prediction, [forecast, current, prediction]
        aligned = self._align_to_current(heatmaps)
        if self.aggregation == "joint_residual_correction":
            prediction = self._residual_correction(aligned)
        else:
            prediction = self._aggregate_joint_heatmaps(
                aligned, weighting_heatmaps=heatmaps,
            )
        return prediction, list(aligned)
