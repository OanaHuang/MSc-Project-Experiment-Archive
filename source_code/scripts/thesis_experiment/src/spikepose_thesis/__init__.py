"""Canonical, independent SpikePose implementation for the ICASSP study."""

from .models import SpikePose, SpikePoseConfig, build_model

__version__ = "0.1.0"

__all__ = ["SpikePose", "SpikePoseConfig", "build_model", "__version__"]
