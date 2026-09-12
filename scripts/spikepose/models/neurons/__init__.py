from .activation import TimeDistributedReLU
from .ilif import MultiStepILIF
from .lif import MultiStepLIF


def build_neuron(kind: str, **kwargs):
    options = {
        "ilif": MultiStepILIF,
        "lif": MultiStepLIF,
        "relu": TimeDistributedReLU,
    }
    if kind not in options:
        raise KeyError(f"Unknown neuron: {kind}")
    return options[kind](**kwargs) if kind != "relu" else options[kind]()


__all__ = ["MultiStepILIF", "MultiStepLIF", "TimeDistributedReLU", "build_neuron"]
