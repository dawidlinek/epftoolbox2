import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .base import FORECAST_KEYS, Evaluator


class rMAEEvaluator(Evaluator):
    """MAE of a model divided by the MAE of `base_model`, both taken on the forecasts the two share.

    A forecast is identified by `FORECAST_KEYS` (run date, target date, hour, horizon). Models whose
    predictors differ can produce different forecast sets, so only the keys present in both models, with a
    finite prediction and actual on both sides, enter either MAE. NaN if the two share no forecast.
    """

    name = "rMAE"

    def __init__(self, base_model: str):
        self.base_model = base_model

    def compute(self, df: pd.DataFrame, **kwargs) -> float:
        model_dfs: Dict[str, pd.DataFrame] = kwargs.get("model_dfs", {})
        if not model_dfs:
            raise ValueError("rMAEEvaluator requires 'model_dfs' kwarg containing all model DataFrames.")
        if self.base_model not in model_dfs:
            return math.nan
        base_df = model_dfs[self.base_model]
        if base_df.empty or df.empty:
            return math.nan

        self._require_keys(df, "the model's frame")
        self._require_keys(base_df, f"the frame of the base model {self.base_model!r}")
        in_model, in_base = self._shared_rows(df, base_df)

        prediction = np.asarray(df["prediction"], dtype=float)[in_model]
        actual = np.asarray(df["actual"], dtype=float)[in_model]
        base_prediction = np.asarray(base_df["prediction"], dtype=float)[in_base]
        base_actual = np.asarray(base_df["actual"], dtype=float)[in_base]
        finite = np.isfinite(prediction) & np.isfinite(actual) & np.isfinite(base_prediction) & np.isfinite(base_actual)
        if not finite.any():
            return math.nan

        base_mae = np.abs(base_prediction - base_actual)[finite].mean()
        if base_mae == 0:
            return float("inf")
        model_mae = np.abs(prediction - actual)[finite].mean()
        return float(model_mae / base_mae)

    @staticmethod
    def _require_keys(frame: pd.DataFrame, what: str) -> None:
        missing = [k for k in FORECAST_KEYS if k not in frame.columns]
        if missing:
            raise ValueError(
                f"rMAEEvaluator compares models on shared forecast keys {list(FORECAST_KEYS)}, but {what} lacks "
                f"the column(s) {missing}. Result records need run_date, target_date, hour and horizon."
            )

    @staticmethod
    def _shared_rows(df: pd.DataFrame, base_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Row positions in `df` and in `base_df` of the forecasts that both hold, in `df`'s order.

        The four key columns of both frames are turned into one integer code per row (exact, not a hash), so
        the join costs a few vector operations even on the many small groups of a grouped table.
        """
        n = len(df)
        codes = np.zeros(n + len(base_df), dtype=np.int64)
        for column in FORECAST_KEYS:
            labels, uniques = pd.factorize(np.concatenate([df[column].to_numpy(), base_df[column].to_numpy()]))
            if (labels < 0).any():
                raise ValueError(f"rMAEEvaluator: a frame has rows with a missing forecast key ({column}).")
            # `codes` is dense (< number of rows) after every step, so this cannot overflow.
            codes, _ = pd.factorize(codes * len(uniques) + labels)
        size = int(codes.max()) + 1
        model_codes, base_codes = codes[:n], codes[n:]
        for who, part in (("the model", model_codes), ("the base model", base_codes)):
            if np.bincount(part, minlength=size).max() > 1:
                raise ValueError(
                    f"rMAEEvaluator needs each forecast key {list(FORECAST_KEYS)} once per model, but {who} has "
                    "duplicated keys (that forecast would count twice)."
                )
        position_in_base = np.full(size, -1, dtype=np.int64)
        position_in_base[base_codes] = np.arange(len(base_codes))
        matched = position_in_base[model_codes]
        shared = matched >= 0
        return np.flatnonzero(shared), matched[shared]
