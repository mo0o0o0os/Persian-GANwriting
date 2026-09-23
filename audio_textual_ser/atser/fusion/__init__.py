"""Fusion methods. Importing this package registers every built-in method in ``REGISTRY``."""

from .base import REGISTRY, FusionInput, FusionMethod, available, create, register, softmax
from . import classic, late, neural  # noqa: F401  (registration side effects)
from .neural import TorchFusion

__all__ = ["REGISTRY", "FusionInput", "FusionMethod", "TorchFusion", "available", "create", "register", "softmax"]
