import math
from typing import Dict

import pandas as pd
from .base import Evaluator


class rMAEEvaluator(Evaluator):
    name = "rMAE"

    def __init__(self, base_model: str):
        self.base_model = base_model

    def compute(self, df: pd.DataFrame, **kwargs) -> float:
        model_dfs: Dict[str, pd.DataFrame] = kwargs.get("model_dfs", {})
        if not model_dfs:
            raise ValueError(
                f"rMAE base model '{self.base_model}' not found in pipeline models."
            )
        if self.base_model not in model_dfs:
            return math.nan
        base_df = model_dfs[self.base_model]
        if base_df.empty:
            return math.nan
        base_mae = (base_df["prediction"] - base_df["actual"]).abs().mean()
        if base_mae == 0:
            return float("inf")
        model_mae = (df["prediction"] - df["actual"]).abs().mean()
        return model_mae / base_mae
