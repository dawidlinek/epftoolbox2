"""Forecasts made of other forecasts: see `AverageEnsemble` and the docs page on ensembles."""

from .average import AverageEnsemble
from .base import Choice, Ensemble

__all__ = ["AverageEnsemble", "Choice", "Ensemble"]
