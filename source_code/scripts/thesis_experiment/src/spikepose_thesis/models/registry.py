"""Public component registries used by experiment validation."""

NEURONS = ("ilif", "lif", "relu")
BLOCKS = ("all_conv", "conv", "membrane", "multi_conv", "sew_add")
NECKS = ("concat", "add", "masked_add", "spike_fpn", "spike_fpn_temporal")
HEADS = (
    "ann_heatmap", "spiking_heatmap", "linear_heatmap",
    "coordinate_classification", "coordinate_regression",
)
