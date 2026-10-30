import pandas as pd

from .base import Evaluator


class BiasEvaluator(Evaluator):
    """Mean error, `mean(prediction - actual)`: positive when the model forecasts too high on average."""

    name = "Bias"

    def compute(self, df: pd.DataFrame, **kwargs) -> float:
        return (df["prediction"] - df["actual"]).mean()
