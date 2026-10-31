from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Union

import pandas as pd

# Columns that identify one forecast. `EvaluationReport` hands every evaluator frames that carry them (next to
# `prediction` and `actual`), so a metric that compares models can align them on the same forecasts.
FORECAST_KEYS = ("run_date", "target_date", "hour", "horizon")

# What an evaluator gives one model: a value, or several named values (several columns of the table).
EvaluatorResult = Union[float, Dict[str, Any]]


class Evaluator(ABC):
    """A metric column (or several) of the report's tables.

    `compute` scores one model. An evaluator whose value for a model depends on the set of models it is compared with
    (a test, a confidence set) overrides `compute_group`, which the report calls once per table group. A dict result
    becomes several columns, named by `column_name`.
    """

    name: str

    @abstractmethod
    def compute(self, df: pd.DataFrame, **kwargs) -> EvaluatorResult:
        pass

    def compute_group(self, model_dfs: Mapping[str, pd.DataFrame]) -> Dict[str, EvaluatorResult]:
        """`{model: result}` for one group: every model's frame of that group, keyed as in `compute`."""
        return {name: self.compute(df, model_dfs=model_dfs) for name, df in model_dfs.items()}

    def column_name(self, key: str) -> str:
        """The table column of one entry of a dict result: `"{name} {key}"`, or just the name for the key `""`."""
        return f"{self.name} {key}" if key else self.name


class SetEvaluator(Evaluator):
    """Base of evaluators defined on a set of models: they implement `compute_group`, and `compute` reads from it."""

    @abstractmethod
    def compute_group(self, model_dfs: Mapping[str, pd.DataFrame]) -> Dict[str, EvaluatorResult]:
        pass

    def clones(self, model_dfs: Mapping[str, pd.DataFrame]) -> list:
        """The groups of models this evaluator counts as one on this group (see `epftoolbox2.evaluators._compare`)."""
        return []

    def compute(self, df: pd.DataFrame, **kwargs) -> EvaluatorResult:
        model_dfs = kwargs.get("model_dfs")
        if not model_dfs:
            raise ValueError(f"{type(self).__name__} compares models: it needs the 'model_dfs' keyword.")
        for name, frame in model_dfs.items():
            if frame is df:
                return self.compute_group(model_dfs)[name]
        raise ValueError(f"{type(self).__name__}: df must be one of the frames in model_dfs.")
