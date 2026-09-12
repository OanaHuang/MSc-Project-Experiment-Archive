from .builder import SpikePose, build_model as _build_model
from .config import (BackboneConfig, HeadConfig, NeckConfig, NeuronConfig,
                     SpikePoseConfig)


def build_model(config):
    """Build the historical model, with opt-in experiment adapters only."""
    model = _build_model(config)
    if isinstance(config, dict) and config.get("mam_softmax_fix"):
        from .mam_v2.coordinate_decoding import install_coordinate_decoder
        install_coordinate_decoder(model, config["mam_softmax_fix"])
    return model

__all__ = [
    "BackboneConfig", "HeadConfig", "NeckConfig", "NeuronConfig",
    "SpikePose", "SpikePoseConfig", "build_model",
]
