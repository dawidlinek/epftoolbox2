import math
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from .base import Evaluator


class NMAEEvaluator(Evaluator):
    """MAE divided by the mean actual, both on the forecasts with a finite prediction and actual: `mean|e| / mean(y)`.

    The normalised MAE of generation and load forecasts, comparable across zones and years of different size. It is read on
    the same keys as the MAE, never on every actual of the data, so a model that skips the night hours of solar is divided
    by its own mean, not by one that the night's zeros lower. NaN without a forecast, and NaN when the mean actual is 0 (the
    night hours of solar in `by_hour()`, even forecast perfectly): there is nothing to normalise by. Meant for quantities
    whose mean is positive.
    """

    name = "nMAE"

    def compute(self, df: pd.DataFrame, **kwargs) -> float:
        prediction = df["prediction"].to_numpy(dtype=float)
        actual = df["actual"].to_numpy(dtype=float)
        keys = np.isfinite(prediction) & np.isfinite(actual)
        if not keys.any():
            return math.nan
        mean_actual = actual[keys].mean()
        mae = np.abs(prediction[keys] - actual[keys]).mean()
        return float(mae / mean_actual) if mean_actual != 0 else math.nan


class ScaledMAEEvaluator(Evaluator):
    """The mean of |e| / c, with c the value of `column` in the report's source data at each forecast's target row.

    With an installed-capacity column this is the MAE as a share of capacity (the renewables study's `MAE_cap_pct`, here
    a fraction). c is joined by the target date and period of the forecast, as `CsvExporter(extra_columns=)` joins (the
    repeated autumn hour takes its second row), so the source must be at the results' frequency. A forecast without a
    finite positive c is left out and counted: the table has two columns, `MAE/{column}` and `MAE/{column} left out`.
    The report needs `source_data` (a report of `ModelPipeline` has it); `aggregate()` leaves this evaluator out.

    Args:
        column: The source-data column to divide by.
    """

    def __init__(self, column: str):
        if not isinstance(column, str) or not column:
            raise ValueError(f"ScaledMAEEvaluator needs the name of a source-data column, got {column!r}")
        self.column = column

    @property
    def name(self) -> str:  # one per column, so two scales are two columns
        return f"MAE/{self.column}"

    @property
    def source_columns(self) -> Tuple[str, ...]:
        """The source-data columns the report joins to the frames this evaluator gets."""
        return (self.column,)

    def compute(self, df: pd.DataFrame, **kwargs) -> Dict[str, Union[float, int]]:
        if self.column not in df.columns:
            raise ValueError(
                f"ScaledMAEEvaluator({self.column!r}) reads {self.column!r} from the report's source data, but the frame "
                "has no such column: build the report with source_data that has it."
            )
        prediction = df["prediction"].to_numpy(dtype=float)
        actual = df["actual"].to_numpy(dtype=float)
        scale = pd.to_numeric(df[self.column], errors="coerce").to_numpy(dtype=float)
        keys = np.isfinite(prediction) & np.isfinite(actual)
        with np.errstate(invalid="ignore"):
            scaled = keys & np.isfinite(scale) & (scale > 0)
        value = (
            float(np.mean(np.abs(prediction[scaled] - actual[scaled]) / scale[scaled])) if scaled.any() else math.nan
        )
        return {"": value, "left out": int((keys & ~scaled).sum())}
