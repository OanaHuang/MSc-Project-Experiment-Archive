from .all_conv import SpikePoseAllConvBlock
from .conv_block import SpikePoseConvBlock
from .separable import SpikePoseSeparableBlock
from .residual import MembraneResidualBlock, MultiConvBlock, SEWAddBlock

BLOCKS = {
    "all_conv": SpikePoseAllConvBlock,
    "conv": SpikePoseConvBlock,
    "membrane": MembraneResidualBlock,
    "multi_conv": MultiConvBlock,
    "sew_add": SEWAddBlock,
}


def build_block(kind: str, channels: int, neuron):
    if kind not in BLOCKS:
        raise KeyError(f"Unknown block: {kind}")
    return BLOCKS[kind](channels, neuron)


__all__ = ["MembraneResidualBlock", "MultiConvBlock", "SEWAddBlock",
           "SpikePoseAllConvBlock", "SpikePoseConvBlock",
           "SpikePoseSeparableBlock", "build_block"]
