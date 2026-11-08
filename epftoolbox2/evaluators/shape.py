import math
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from ._days import QUARTERS, day_codes, periods_per_hour, whole_hours
from .base import Evaluator, EvaluatorResult

# The results a within-hour shape exists in: quarter-hours.
_SHAPE_FREQ = "15min"


class ShapeMAEEvaluator(Evaluator):
    """The error of the shape within each hour: the mean of |(p - p̄) - (y - ȳ)| over the quarter-hours of whole hours.

    p̄ and ȳ are the means of the prediction and the actual over the four quarter-hours of their hour, so a forecast that
    gets every hour's level wrong but the movement inside it right scores 0. Only hours with all four quarter-hours finite
    count, as for `report.aggregate("1h")` (`evaluators._days.whole_hours`): the repeated autumn hour, which has no actual,
    never does. This is the resolution study's shape MAE. A report of hourly or daily results has no shape within the hour,
    so the column is NaN there (and in `by_hour()`, whose groups hold one quarter-hour of each hour).

    Args:
        period: The period the shape is measured within: `"1h"`, the only one so far.
    """

    name = "ShapeMAE"

    def __init__(self, period: str = "1h"):
        if period != "1h":
            raise ValueError(
                f"ShapeMAEEvaluator measures the shape within the hour: period must be '1h', got {period!r}"
            )
        self.period = period

    def compute_group(self, model_dfs: Mapping[str, pd.DataFrame]) -> Dict[str, EvaluatorResult]:
        freqs = getattr(model_dfs, "freqs", {})
        return {name: shape_mae(df, freqs.get(name)) for name, df in model_dfs.items()}

    def compute(self, df: pd.DataFrame, **kwargs) -> float:
        model_dfs = kwargs.get("model_dfs")
        for name, frame in (model_dfs or {}).items():
            if frame is df:
                return shape_mae(df, getattr(model_dfs, "freqs", {}).get(name))
        return shape_mae(df, None)


def shape_mae(df: pd.DataFrame, freq: Optional[str]) -> float:
    """The shape MAE of one model's frame; `freq` is its results' frequency, or None to read it from the period ids (an id
    above 24 means quarter-hours)."""
    if df.empty:
        return math.nan
    quarter_hours = freq == _SHAPE_FREQ if freq is not None else periods_per_hour(df["hour"].to_numpy()) == QUARTERS
    if not quarter_hours:
        return math.nan
    day = day_codes(df, "ShapeMAEEvaluator")
    prediction = df["prediction"].to_numpy(dtype=float)
    actual = df["actual"].to_numpy(dtype=float)
    rows, _ = whole_hours(day, df["hour"].to_numpy(), np.isfinite(prediction) & np.isfinite(actual))
    if not len(rows):
        return math.nan
    p = prediction[rows].reshape(-1, QUARTERS)
    y = actual[rows].reshape(-1, QUARTERS)
    deviation = (p - p.mean(axis=1, keepdims=True)) - (y - y.mean(axis=1, keepdims=True))
    return float(np.abs(deviation).mean())
