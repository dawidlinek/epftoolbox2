from abc import ABC, abstractmethod
import pandas as pd

# Columns that identify one forecast. `EvaluationReport` hands every evaluator frames that carry them (next to
# `prediction` and `actual`), so a metric that compares models can align them on the same forecasts.
FORECAST_KEYS = ("run_date", "target_date", "hour", "horizon")


class Evaluator(ABC):
    name: str

    @abstractmethod
    def compute(self, df: pd.DataFrame, **kwargs) -> float:
        pass
