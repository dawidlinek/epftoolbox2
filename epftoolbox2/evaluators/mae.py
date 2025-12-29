import pandas as pd
from .base import Evaluator


class MAEEvaluator(Evaluator):
    name = "MAE"

    def compute(self, df: pd.DataFrame) -> float:
        return (df["prediction"] - df["actual"]).abs().mean()
