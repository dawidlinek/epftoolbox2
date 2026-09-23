import re
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from ._utils import MAX_LISTED, as_utc, column_values
from .base import Validator
from .result import ValidationResult

_DAY_LAG = re.compile(r"^(?P<base>.+)_d(?P<sign>[+-])(?P<days>\d+)$")


class LagConsistencyValidator(Validator):
    """Checks that `x_d-k` / `x_d+k` columns hold `x` from k local calendar days earlier / later, same wall-clock time.

    This is the column naming contract (see `.claude/rules/data.md`): on a local index, "the same hour yesterday" is
    a calendar day back, not 24 elapsed hours, which differ around DST switches. A lag built from elapsed time is
    wrong on the switch days and, for `d-7`, for a week after each switch; this validator reports those rows.

    Args:
        columns: Lag columns to check. None checks every `x_d-k` / `x_d+k` column whose base column `x` is in the
            frame (forecast columns without a base, like Open-Meteo's `temperature_2m_d+1`, are skipped).
        tz: Time zone that defines the local days. None uses the index's own time zone. Set it for a UTC index whose
            lags were built on local days.
        tolerance: Absolute difference still counted as equal, for numeric columns.

    Lookup rule, per row at wall-clock time w: the base value at w ± k days. On the repeated hour of the autumn switch
    the first occurrence is used when the source day has only one; a source time the clock skipped (the spring
    switch hour) has no value, so the lag must be missing there. Rows whose source time lies outside the frame can't
    be verified and are skipped (`unverifiable` in `stats`). A lag value where the base is missing is a mismatch.
    """

    def __init__(
        self,
        columns: Optional[Union[str, List[str]]] = None,
        tz: Optional[str] = None,
        tolerance: float = 1e-9,
    ):
        if tz is not None:
            try:
                pd.Timestamp("2000-01-01", tz=tz)
            except Exception as exc:
                raise ValueError(f"Unknown time zone {tz!r}") from exc
        if tolerance < 0:
            raise ValueError(f"tolerance must not be negative, got {tolerance}")
        self.columns = [columns] if isinstance(columns, str) else columns
        self.tz = tz
        self.tolerance = tolerance

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = self._result()

        if df.empty:
            result.warnings.append("DataFrame is empty")
            return result

        if not isinstance(df.index, pd.DatetimeIndex):
            result.add_error("Index is not a DatetimeIndex")
            return result

        targets = self._targets(df, result)
        if not targets:
            if self.columns is None:
                result.warnings.append("No x_d-k / x_d+k columns whose base column x is in the frame")
            return result

        index = df.index
        if index.has_duplicates:
            result.warnings.append("Index has duplicate timestamps; they are matched like a repeated DST hour")
        local = as_utc(index).tz_convert(self.tz) if self.tz is not None else index
        result.info["tz"] = str(local.tz) if local.tz is not None else None
        wall = local.tz_localize(None) if local.tz is not None else local

        # Occurrence of each wall-clock time in time order: 1 marks the second pass through the repeated autumn hour.
        n = len(wall)
        order = np.argsort(as_utc(index).asi8, kind="stable")
        occurrence = np.empty(n, dtype=np.int64)
        occurrence[order] = pd.Series(np.arange(n)).groupby(wall[order]).cumcount().to_numpy()
        keys = pd.MultiIndex.from_arrays([wall, occurrence])
        first, last = wall.min(), wall.max()

        lookups = {}
        for shift in sorted({t[2] for t in targets}):
            source = wall + pd.Timedelta(days=shift)
            position = keys.get_indexer(pd.MultiIndex.from_arrays([source, occurrence]))
            retry = (position < 0) & (occurrence > 0)
            if retry.any():
                fallback = pd.MultiIndex.from_arrays([source[retry], np.zeros(int(retry.sum()), dtype=np.int64)])
                position[retry] = keys.get_indexer(fallback)
            lookups[shift] = (position, np.asarray((source >= first) & (source <= last)))

        rows, counts, examples = [], {}, {}
        for column, base, shift in targets:
            position, checkable = lookups[shift]
            actual, expected = self._values(df, column, base, position)
            if actual.dtype == float and expected.dtype == float:
                equal = (np.isnan(actual) & np.isnan(expected)) | np.isclose(
                    actual, expected, rtol=0.0, atol=self.tolerance
                )
            else:
                equal = (pd.isna(actual) & pd.isna(expected)) | (actual == expected)
            wrong = np.flatnonzero(checkable & ~equal)
            counts[column] = len(wrong)
            rows.append(
                {
                    "column": column,
                    "base": base,
                    "days": shift,
                    "checked": int(checkable.sum()),
                    "unverifiable": int((~checkable).sum()),
                    "mismatches": len(wrong),
                    "first_mismatch": index[wrong[0]] if len(wrong) else None,
                }
            )
            if len(wrong):
                examples[column] = [
                    {"timestamp": index[i], "actual": actual[i], "expected": expected[i]} for i in wrong[:MAX_LISTED]
                ]
                days = f"{abs(shift)} local calendar day{'' if abs(shift) == 1 else 's'}"
                direction = "earlier" if shift < 0 else "later"
                result.add_error(
                    f"Column '{column}' is not '{base}' {days} {direction} at the same "
                    f"wall-clock time on {len(wrong)} of {int(checkable.sum())} rows (first: {index[wrong[0]]} has "
                    f"{_show(actual[wrong[0]])}, expected {_show(expected[wrong[0]])})"
                )

        result.stats = pd.DataFrame(rows)
        result.info["checked"] = {row["column"]: row["checked"] for row in rows}
        result.info["mismatch_counts"] = counts
        if examples:
            result.info["mismatches"] = examples
        return result

    def _targets(self, df: pd.DataFrame, result: ValidationResult) -> list:
        """(column, base column, shift in days) for each lag column to check."""
        if self.columns is None:
            candidates = [(c, _DAY_LAG.match(c)) for c in dict.fromkeys(df.columns) if isinstance(c, str)]
            return [(c, m["base"], _shift(m)) for c, m in candidates if m and m["base"] in df.columns]

        targets = []
        for column in dict.fromkeys(self.columns):
            match = _DAY_LAG.match(column)
            if match is None:
                result.add_error(f"Column '{column}' is not named like a day lag (x_d-k or x_d+k)")
            elif column not in df.columns:
                result.add_error(f"Column '{column}' is not in the frame")
            elif match["base"] not in df.columns:
                result.add_error(f"Column '{column}': base column '{match['base']}' is not in the frame")
            else:
                targets.append((column, match["base"], _shift(match)))
        return targets

    @staticmethod
    def _values(df: pd.DataFrame, column: str, base: str, position: np.ndarray):
        lagged, source = column_values(df, column), column_values(df, base)
        numeric = pd.api.types.is_numeric_dtype(lagged) and pd.api.types.is_numeric_dtype(source)
        dtype = float if numeric else object
        actual, values = lagged.to_numpy(dtype=dtype), source.to_numpy(dtype=dtype)
        expected = np.full(len(position), np.nan if numeric else None, dtype=dtype)
        found = position >= 0
        expected[found] = values[position[found]]
        return actual, expected


def _shift(match: re.Match) -> int:
    days = int(match["days"])
    return -days if match["sign"] == "-" else days


def _show(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NaN"
    return f"{value:g}" if isinstance(value, float) else repr(value)
