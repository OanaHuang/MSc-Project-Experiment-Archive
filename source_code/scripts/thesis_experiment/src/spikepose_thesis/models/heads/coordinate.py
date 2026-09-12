from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _aggregate_steps(value: torch.Tensor) -> torch.Tensor:
    return value.mean(0) if value.ndim == 5 else value


class CoordinateClassificationHead(nn.Module):
    """SimCC-style independent horizontal and vertical classifiers."""

    def __init__(self, in_channels: int, num_joints: int,
                 feature_size: tuple[int, int], coordinate_size: int,
                 split_ratio: int) -> None:
        super().__init__()
        if split_ratio < 1:
            raise ValueError("split_ratio must be at least one")
        self.split_ratio = int(split_ratio)
        self.num_bins = int(coordinate_size) * self.split_ratio
        self.feature_size = tuple(feature_size)
        self.projection = nn.Conv2d(in_channels, num_joints, 1)
        flattened_features = self.feature_size[0] * self.feature_size[1]
        # The same classifier is applied independently to each joint channel,
        # matching the simple SimCC head while retaining the full 2-D layout.
        self.x_classifier = nn.Linear(flattened_features, self.num_bins)
        self.y_classifier = nn.Linear(flattened_features, self.num_bins)
        self.output_modules = (self.x_classifier, self.y_classifier)

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor | str | int]:
        value = _aggregate_steps(value)
        if value.shape[-2:] != self.feature_size:
            value = F.interpolate(
                value, self.feature_size, mode="bilinear", align_corners=False,
            )
        spatial = self.projection(value).flatten(-2)
        x_logits = self.x_classifier(spatial)
        y_logits = self.y_classifier(spatial)
        return {
            "output_type": "coordinate_classification",
            "x_logits": x_logits,
            "y_logits": y_logits,
            "split_ratio": self.split_ratio,
        }


class CoordinateRegressionHead(nn.Module):
    """Direct normalized-coordinate regression with controlled spatial readouts."""

    def __init__(self, in_channels: int, hidden_channels: int, num_joints: int,
                 variant: str, regression_channels: int,
                 spatial_pool_size: tuple[int, int]) -> None:
        super().__init__()
        self.variant = variant
        self.num_joints = num_joints
        if variant == "gap":
            self.reducer = nn.Identity()
            self.pool = nn.AdaptiveAvgPool2d(1)
            input_features = in_channels
        elif variant == "spatial":
            self.reducer = nn.Conv2d(in_channels, regression_channels, 1)
            self.pool = nn.AdaptiveAvgPool2d(tuple(spatial_pool_size))
            input_features = regression_channels * spatial_pool_size[0] * spatial_pool_size[1]
        elif variant == "integral":
            # Keep the complete two-dimensional evidence map and regress the
            # expectation of each joint distribution.  Unlike the flattened
            # spatial MLP, this preserves translation-aware localization and
            # avoids collapsing ambiguous joints to a dataset-average pose.
            self.reducer = nn.Conv2d(in_channels, num_joints, 1)
            self.pool = nn.Identity()
            self.mlp = nn.Identity()
            self.final = nn.Identity()
            self.output_modules = (self.reducer,)
            return
        else:
            raise ValueError(f"Unknown coordinate regression variant: {variant}")
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Linear(hidden_channels, num_joints * 2)
        self.output_modules = (self.final,)

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor | str]:
        value = _aggregate_steps(value)
        if self.variant == "integral":
            logits = self.reducer(value)
            height, width = logits.shape[-2:]
            probabilities = logits.flatten(-2).softmax(-1)
            x = torch.linspace(0.0, 1.0, width, device=logits.device, dtype=logits.dtype)
            y = torch.linspace(0.0, 1.0, height, device=logits.device, dtype=logits.dtype)
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            grid = torch.stack((grid_x.flatten(), grid_y.flatten()), -1)
            coordinates = torch.einsum("bjn,nc->bjc", probabilities, grid)
            return {
                "output_type": "coordinate_regression",
                "coordinates": coordinates,
            }
        coordinates = self.final(self.mlp(self.pool(self.reducer(value))))
        return {
            "output_type": "coordinate_regression",
            "coordinates": coordinates.reshape(-1, self.num_joints, 2).sigmoid(),
        }
