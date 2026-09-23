from typing import List, Optional, Union

import pandas as pd

from ._utils import as_utc, column_values, display_time, parse_timestamp, select_columns, to_duration
from .base import Validator
from .result import ValidationResult


class FreshnessValidator(Validator):
    """Checks that every column has values up to an expected end, e.g. before issuing a live forecast.

    Args:
        columns: Columns to check. None checks every column.
        expected_end: Timestamp of the last value you need. None uses the last row of the frame, which finds a
            source that stopped updating while the others went on. Accepts the "today" / "today_d+1" keywords of
            `DataPipeline.run`; naive values are UTC, like pipeline dates.
        max_delay: How far before `expected_end` a column's last value may be ("0h" = at `expected_end` or later).

    A column is stale, an error, when its last value is earlier than `expected_end - max_delay`. `stats` lists the
    last value of each column and how far it is behind `expected_end`.
    """

    def __init__(
        self,
        columns: Optional[Union[str, List[str]]] = None,
        expected_end: Optional[str] = None,
        max_delay: str = "0h",
    ):
        self.columns = [columns] if isinstance(columns, str) else columns
        self.expected_end = expected_end
        self.max_delay = max_delay
        self._max_delay = to_duration(max_delay, "max_delay")

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = self._result()

        if df.empty:
            result.warnings.append("DataFrame is empty")
            return result

        if not isinstance(df.index, pd.DatetimeIndex):
            result.add_error("Index is not a DatetimeIndex")
            return result

        columns, missing = select_columns(df, self.columns)
        if missing:
            result.add_error(f"Missing required columns: {sorted(missing)}")
            result.info["missing_columns"] = sorted(missing)

        index = df.index
        times = as_utc(index)
        expected_end = parse_timestamp(self.expected_end)
        expected_end = times.max() if expected_end is None else expected_end.tz_convert("UTC")
        deadline = expected_end - self._max_delay
        shown_end = display_time(expected_end, index)
        result.info["expected_end"] = shown_end

        rows, stale = [], []
        for column in columns:
            valid = times[column_values(df, column).notna().to_numpy()]
            if len(valid) == 0:
                rows.append({"column": column, "last_valid": None, "behind": None})
                stale.append(column)
                result.add_error(f"Column '{column}' has no values")
                continue
            last = valid.max()
            rows.append({"column": column, "last_valid": display_time(last, index), "behind": expected_end - last})
            if last < deadline:
                stale.append(column)
                delay = f" (max delay {self.max_delay})" if self._max_delay > pd.Timedelta(0) else ""
                result.add_error(
                    f"Column '{column}' ends at {display_time(last, index)}, {expected_end - last} before the "
                    f"expected end {shown_end}{delay}"
                )

        result.stats = pd.DataFrame(rows, columns=["column", "last_valid", "behind"])
        result.info["last_valid"] = {row["column"]: row["last_valid"] for row in rows}
        result.info["stale_columns"] = stale
        return result
