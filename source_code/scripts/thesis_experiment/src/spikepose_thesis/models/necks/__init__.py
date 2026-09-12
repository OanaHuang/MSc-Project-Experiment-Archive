from .additive import AdditiveFusion
from .concat import ConcatFusion
from .masked_additive import MaskedAdditiveFusion
from .spike_fpn import SpikeFPNFusion


def build_neck(kind: str, stages, channels, out_channels, neuron=None,
               interpolation="nearest", align_corners=False, stage_mask=None):
    options = {"concat": ConcatFusion, "add": AdditiveFusion,
               "masked_add": MaskedAdditiveFusion,
               "spike_fpn": SpikeFPNFusion,
               "spike_fpn_temporal": SpikeFPNFusion}
    if kind not in options:
        raise KeyError(f"Unknown neck: {kind}")
    if kind in {"spike_fpn", "spike_fpn_temporal"}:
        return options[kind](stages, channels, out_channels, neuron,
                             interpolation, align_corners,
                             preserve_time=(kind == "spike_fpn_temporal"))
    if kind == "masked_add":
        if stage_mask is None:
            raise ValueError("masked_add requires neck.stage_mask")
        return options[kind](stages, channels, out_channels, stage_mask,
                             interpolation, align_corners)
    return options[kind](stages, channels, out_channels,
                         interpolation, align_corners)


__all__ = [
    "AdditiveFusion", "ConcatFusion", "MaskedAdditiveFusion",
    "SpikeFPNFusion", "build_neck",
]
