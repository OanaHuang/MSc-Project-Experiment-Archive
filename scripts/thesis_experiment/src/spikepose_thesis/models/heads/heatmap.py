from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 num_joints: int, output_size: tuple[int, int] | None,
                 upsample_factor: int = 2, aggregation: str = "mean",
                 feedback_edges: tuple[tuple[int, int], ...] = ()) -> None:
        super().__init__()
        if upsample_factor not in (1, 2):
            raise ValueError("HeatmapHead upsample_factor must be 1 or 2")
        self.output_size = output_size
        self.aggregation = aggregation
        self.num_joints = num_joints
        self.refinement_modes = {
            "residual_refinement", "feature_control_refinement",
            "heatmap_feedback", "heatmap_feedback_no_residual",
            "skeleton_heatmap_feedback",
        }
        if aggregation == "learned_heatmap":
            self.step_logits = nn.Parameter(torch.zeros(2))
        if aggregation in self.refinement_modes - {"residual_refinement"}:
            self.feedback_projection = nn.Sequential(
                nn.Conv2d(num_joints, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )
        if aggregation == "skeleton_heatmap_feedback":
            adjacency = torch.eye(num_joints)
            for left, right in feedback_edges:
                adjacency[left, right] = 1.0
                adjacency[right, left] = 1.0
            adjacency = adjacency / adjacency.sum(1, keepdim=True).clamp_min(1.0)
            self.register_buffer("feedback_adjacency", adjacency)
        self.input_projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.upsample = (nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True),
        ) if upsample_factor == 2 else nn.Identity())
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(hidden_channels, num_joints, 1)

    def _predict(self, value: torch.Tensor) -> torch.Tensor:
        value = self.input_projection(value)
        value = self.refine(self.upsample(value))
        value = self.final(value)
        if self.output_size is not None and value.shape[-2:] != self.output_size:
            value = F.interpolate(value, size=self.output_size, mode="bilinear",
                                  align_corners=False)
        return value

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            return self._predict(value)
        if self.aggregation in self.refinement_modes:
            return self.forward_with_intermediates(value)[0]
        if self.aggregation in ("mean", "last"):
            reduced = value.mean(0) if self.aggregation == "mean" else value[-1]
            return self._predict(reduced)
        if self.aggregation not in ("heatmap_mean", "learned_heatmap"):
            raise ValueError(f"Unknown temporal aggregation: {self.aggregation}")

        steps, batch = value.shape[:2]
        flat = value.flatten(0, 1)
        predicted = self._predict(flat)
        heatmaps = predicted.reshape(steps, batch, *predicted.shape[1:])
        if self.aggregation == "heatmap_mean":
            return heatmaps.mean(0)
        if steps > len(self.step_logits):
            raise ValueError("learned_heatmap supports at most two time steps")
        weights = self.step_logits[:steps].softmax(0)
        return (heatmaps * weights[:, None, None, None, None]).sum(0)

    def _feedback(self, heatmap: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        if self.aggregation == "skeleton_heatmap_feedback":
            heatmap = torch.einsum("ij,bjhw->bihw", self.feedback_adjacency, heatmap)
        heatmap = F.interpolate(
            heatmap, size=feature.shape[-2:], mode="bilinear", align_corners=False,
        )
        return self.feedback_projection(heatmap)

    def forward_with_intermediates(
            self, value: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if value.ndim != 5:
            raise ValueError("Temporal refinement requires T x B x C x H x W input")
        if self.aggregation not in self.refinement_modes:
            return self.forward(value), []

        heatmaps: list[torch.Tensor] = []
        current = self._predict(value[0])
        heatmaps.append(current)
        for step in range(1, value.shape[0]):
            feature = value[step]
            if self.aggregation == "feature_control_refinement":
                control = feature[:, :self.num_joints]
                feature = feature + self.feedback_projection(control)
            elif self.aggregation in {
                    "heatmap_feedback", "heatmap_feedback_no_residual",
                    "skeleton_heatmap_feedback"}:
                feature = feature + self._feedback(current, feature)
            update = self._predict(feature)
            current = (update if self.aggregation == "heatmap_feedback_no_residual"
                       else current + update)
            heatmaps.append(current)
        return current, heatmaps
