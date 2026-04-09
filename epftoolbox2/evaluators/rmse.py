import numpy as np
import pandas as pd
from .base import Evaluator


class RMSEEvaluator(Evaluator):
    name = "RMSE"

    def compute(self, df: pd.DataFrame, **kwargs) -> float:
        return float(np.sqrt(((df["prediction"] - df["actual"]) ** 2).mean()))
