import torch
import torch.nn as nn
import torch.nn.functional as F


class VisibleHeatmapMSE(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                visibility: torch.Tensor) -> torch.Tensor:
        weights = visibility[:, :, None, None].to(prediction.dtype)
        error = (prediction - target).square() * weights
        pixels = prediction.shape[-1] * prediction.shape[-2]
        return error.sum() / (weights.sum().clamp_min(1.0) * pixels)


class VisibleCoordinateKL(nn.Module):
    """Visibility-aware SimCC loss with Gaussian-smoothed one-dimensional labels."""

    target_key = "keypoints"

    def __init__(self, coordinate_size: int, split_ratio: int,
                 label_sigma: float) -> None:
        super().__init__()
        self.coordinate_size = int(coordinate_size)
        self.split_ratio = int(split_ratio)
        self.label_sigma = float(label_sigma)

    def _labels(self, coordinate: torch.Tensor, bins: int) -> torch.Tensor:
        positions = torch.arange(
            bins, device=coordinate.device, dtype=coordinate.dtype,
        )
        centers = coordinate[..., None] * self.split_ratio
        labels = torch.exp(-0.5 * ((positions - centers) / self.label_sigma).square())
        return labels / labels.sum(-1, keepdim=True).clamp_min(1e-12)

    def forward(self, prediction: dict, target: torch.Tensor,
                visibility: torch.Tensor) -> torch.Tensor:
        x_logits = prediction["x_logits"]
        y_logits = prediction["y_logits"]
        x_labels = self._labels(target[..., 0], x_logits.shape[-1])
        y_labels = self._labels(target[..., 1], y_logits.shape[-1])
        x_loss = F.kl_div(x_logits.log_softmax(-1), x_labels, reduction="none").sum(-1)
        y_loss = F.kl_div(y_logits.log_softmax(-1), y_labels, reduction="none").sum(-1)
        weights = visibility.to(x_loss.dtype)
        return ((x_loss + y_loss) * weights).sum() / (2.0 * weights.sum().clamp_min(1.0))


class VisibleCoordinateSmoothL1(nn.Module):
    """Direct coordinate loss in normalized crop space."""

    target_key = "keypoints"

    def __init__(self, coordinate_size: int) -> None:
        super().__init__()
        self.coordinate_scale = float(coordinate_size - 1)

    def forward(self, prediction: dict, target: torch.Tensor,
                visibility: torch.Tensor) -> torch.Tensor:
        normalized = target / self.coordinate_scale
        error = F.smooth_l1_loss(
            prediction["coordinates"], normalized, reduction="none",
        ).sum(-1)
        weights = visibility.to(error.dtype)
        return (error * weights).sum() / weights.sum().clamp_min(1.0)


def build_pose_loss(config: dict) -> nn.Module:
    head = config["model"]["head"]
    kind = head["kind"]
    if kind == "coordinate_classification":
        return VisibleCoordinateKL(
            head.get("coordinate_size", config["data"]["image_size"]),
            head.get("split_ratio", 1), head.get("label_sigma", 2.0),
        )
    if kind == "coordinate_regression":
        return VisibleCoordinateSmoothL1(
            head.get("coordinate_size", config["data"]["image_size"]),
        )
    return VisibleHeatmapMSE()
