from typing import List, Optional, Union

import pandas as pd

from ._utils import (
    ONE_DAY,
    as_utc,
    column_values,
    display_time,
    format_step,
    infer_step,
    longest_run,
    parse_timestamp,
    select_columns,
    to_duration,
    to_step,
)
from .base import Validator
from .result import ValidationResult


class CoverageValidator(Validator):
    """Checks per column that a value exists at every expected timestamp of a range.

    Unlike `ContinuityValidator` (the index alone), coverage is measured on the full `freq` grid, so it catches
    both an outage of one column on an otherwise complete index and rows missing from the index entirely.

    Args:
        columns: Columns to check. None checks every column.
        start: First expected timestamp, e.g. the `start` given to `DataPipeline.run`. None uses each column's first
            value, so lag columns that are empty at the edges by construction pass.
        end: End of the range, exclusive. None uses each column's last value.
        freq: Grid step ("15min", "1h"). None infers it per column as the most common step between the column's
            values, so hourly prices and 15-minute load in one raw source frame are each measured on their own grid.
        min_coverage: Smallest share of expected timestamps that must have a value (1.0 = all).
        max_gap: Longest run of missing timestamps allowed, as a duration ("6h"). None = no limit.

    `start` and `end` accept dates, timestamps and the "today" / "today_d-1" keywords of `DataPipeline.run`; naive
    values are UTC, like pipeline dates. `stats` has one row per column with its step, expected, present and missing
    counts, coverage, the longest missing run and the first and last value in the range.
    """

    def __init__(
        self,
        columns: Optional[Union[str, List[str]]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        freq: Optional[str] = None,
        min_coverage: float = 1.0,
        max_gap: Optional[str] = None,
    ):
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError(f"min_coverage must be between 0 and 1, got {min_coverage}")
        self.columns = [columns] if isinstance(columns, str) else columns
        self.start = start
        self.end = end
        self.freq = freq
        self.min_coverage = min_coverage
        self.max_gap = max_gap
        self._step = to_step(freq) if freq is not None else None
        self._max_gap = to_duration(max_gap, "max_gap") if max_gap is not None else None

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
        utc = as_utc(index)
        index_step = infer_step(utc)
        if self._step is None and index_step is None:
            result.warnings.append("Fewer than two distinct timestamps: coverage can't be measured")
            return result
        start, end = parse_timestamp(self.start), parse_timestamp(self.end)

        rows, errors, warnings = [], [], []
        for column in columns:
            has_value = column_values(df, column).notna().to_numpy()
            # Each column on its own grid: a raw source frame mixes hourly prices with 15-minute load.
            step = self._step
            if step is None:
                step = infer_step(utc[has_value]) or index_step
            freq = self.freq if self.freq is not None else format_step(step)
            grid = _Grid(index, step)
            valid = grid.times[has_value].unique().sort_values()
            if len(valid) == 0 and (start is None or end is None):
                rows.append({"column": column, "freq": freq, "expected": 0, "present": 0, "missing": 0})
                rows[-1]["coverage"] = 0.0
                errors.append(f"Column '{column}' has no values")
                continue

            lo = grid.to_grid(start) if start is not None else valid[0]
            hi = grid.to_grid(end) if end is not None else valid[-1] + step
            origin = valid[0] if len(valid) else grid.times.min()
            offset = (lo - origin) % step  # put the grid on the same phase as the column's values
            if offset != pd.Timedelta(0):
                lo = lo + (step - offset)
            expected = pd.date_range(lo, hi, freq=step, inclusive="left")
            if len(expected) == 0:
                warnings.append(f"Column '{column}': the range [{grid.shown(lo)}, {grid.shown(hi)}) is empty")
                continue

            present = expected.isin(valid)
            count = int(present.sum())
            run, run_start = longest_run(~present)
            row = {
                "column": column,
                "freq": freq,
                "start": grid.shown(lo),
                "end": grid.shown(hi),
                "expected": len(expected),
                "present": count,
                "missing": len(expected) - count,
                "coverage": count / len(expected),
                "longest_missing_run": run,
                "longest_missing_start": grid.shown(expected[run_start]) if run else None,
                "longest_missing_end": grid.shown(expected[run_start + run - 1]) if run else None,
                "first_valid": grid.shown(expected[present][0]) if count else None,
                "last_valid": grid.shown(expected[present][-1]) if count else None,
            }
            rows.append(row)
            if count == 0:
                errors.append(f"Column '{column}' has no values in [{row['start']}, {row['end']})")
                continue
            if run == 0:
                continue

            too_sparse = row["coverage"] < self.min_coverage
            too_long = self._max_gap is not None and run * step > self._max_gap
            message = (
                f"Column '{column}' has values at {count} of {len(expected)} expected {freq} timestamps in "
                f"[{row['start']}, {row['end']}) ({row['coverage']:.2%}, min {self.min_coverage:.2%}); longest "
                f"missing run {run} timestamps ({format_step(run * step)}) from {row['longest_missing_start']} "
                f"to {row['longest_missing_end']}"
            )
            if too_long:
                message += f", longer than max_gap {self.max_gap}"
            (errors if too_sparse or too_long else warnings).append(message)

        result.stats = pd.DataFrame(rows)
        if rows:
            result.info["freq"] = {r["column"]: r["freq"] for r in rows}
            result.info["coverage"] = {r["column"]: r["coverage"] for r in rows}
            result.info["longest_missing_run"] = {r["column"]: r.get("longest_missing_run", 0) for r in rows}
            result.info["first_valid"] = {r["column"]: r.get("first_valid") for r in rows}
            result.info["last_valid"] = {r["column"]: r.get("last_valid") for r in rows}
        for message in errors:
            result.add_error(message)
        result.warnings.extend(warnings)
        return result


class _Grid:
    """Time coordinate for a column's grid: elapsed (UTC) time, or the local wall clock for day-or-longer steps on a
    tz-aware index, where a local day lasts 23 or 25 hours on DST switches."""

    def __init__(self, index: pd.DatetimeIndex, step: pd.Timedelta):
        self._index = index
        self._on_calendar = index.tz is not None and step % ONE_DAY == pd.Timedelta(0)
        self.times = index.tz_localize(None) if self._on_calendar else as_utc(index)

    def to_grid(self, ts: pd.Timestamp) -> pd.Timestamp:
        return ts.tz_convert(self._index.tz).tz_localize(None) if self._on_calendar else ts.tz_convert("UTC")

    def shown(self, ts: pd.Timestamp) -> pd.Timestamp:
        if self._on_calendar:
            return ts.tz_localize(self._index.tz, ambiguous=True, nonexistent="shift_forward")
        return display_time(ts, self._index)
