"""Hyper-parameter search on the validation period: Bayesian proposals with successive halving (`BayesSearch`, extra `[tune]`).

The documented default search is a plain random loop of `add_model` calls averaged by `AverageEnsemble(top=k,
by="validation")` (see the docs page on hyper-parameter search). `BayesSearch` is the option for settings without in-fit
selection and with several dimensions: Optuna proposes settings from the trials so far, every trial is first scored on
every k-th validation issue day (`ModelPipeline.run(validation_every=k)`), and only the best fraction goes on to the dearer
rungs. It never reads the test period, and returns ordinary models. Importing this module imports no extra; Optuna is
imported when a search runs.
"""

from .bayes import BayesSearch, SearchResult, Subset

__all__ = ["BayesSearch", "SearchResult", "Subset"]
