from .heatmap import HeatmapHead
from .linear import LinearHeatmapHead
from .spiking_heatmap import SpikingHeatmapHead
from .coordinate import CoordinateClassificationHead, CoordinateRegressionHead


def build_head(kind, in_channels, hidden_channels, num_joints, output_size,
               neuron, num_steps, upsample_factor=2, aggregation="mean",
               feedback_edges=(), head_config=None):
    if kind == "ann_heatmap":
        return HeatmapHead(in_channels, hidden_channels, num_joints, output_size,
                           upsample_factor, aggregation, feedback_edges)
    if kind == "linear_heatmap":
        return LinearHeatmapHead(in_channels, hidden_channels, num_joints, output_size,
                                 aggregation, num_steps, feedback_edges)
    if kind == "spiking_heatmap":
        return SpikingHeatmapHead(in_channels, hidden_channels, num_joints,
                                  output_size, neuron, num_steps, upsample_factor,
                                  aggregation)
    if kind == "coordinate_classification":
        return CoordinateClassificationHead(
            in_channels, num_joints, output_size, head_config.coordinate_size,
            head_config.split_ratio,
        )
    if kind == "coordinate_regression":
        return CoordinateRegressionHead(
            in_channels, hidden_channels, num_joints,
            head_config.regression_variant, head_config.regression_channels,
            head_config.spatial_pool_size,
        )
    raise KeyError(f"Unknown head: {kind}")


__all__ = [
    "CoordinateClassificationHead", "CoordinateRegressionHead", "HeatmapHead",
    "LinearHeatmapHead", "SpikingHeatmapHead", "build_head",
]
