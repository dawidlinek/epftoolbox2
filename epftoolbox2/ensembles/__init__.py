"""Forecasts made of other forecasts: see `AverageEnsemble`, `FallbackEnsemble` and the docs pages on ensembles and fallbacks."""

from .average import AverageEnsemble
from .base import Choice, Ensemble
from .fallback import FallbackEnsemble

__all__ = ["AverageEnsemble", "Choice", "Ensemble", "FallbackEnsemble"]
