"""A statistic of a column's day statistics over a window of local days ending some days before the row's."""

import numbers

import numpy as np
import pandas as pd

from ._daily import check_index, check_new_columns, check_spacing, local_days, parse_step, slot_of_day, slots_in_day
from .base import Transformer

STATS = ("min", "max", "mean")


def _whole(value, name: str, least: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < least:
        raise ValueError(f"{name} must be an integer of at least {least}, got {value!r}")
    return int(value)


def rolling_name(column: str, window_days: int, stat: str, lag_days: int) -> str:
    """`{column}_roll{window_days}{stat}_d-{lag_days}`, without the shift for `lag_days=0`."""
    name = f"{column}_roll{window_days}{stat}"
    return f"{name}_d-{lag_days}" if lag_days else name


class RollingStatsTransformer(Transformer):
    """A statistic of a column's day statistics over the `window_days` local days ending `lag_days` before the row's.

    Adds `{column}_roll{W}{stat}_d-{lag}`: on every row of local day D, the `stat` over days D - lag - W + 1 to D - lag of
    the `stat` of `column` over each day. A day counts only when it is complete, every slot it has (23, 24 or 25 hours on a
    time-zone-aware index) holding a value, as in `DailyStatsTransformer`; an incomplete or missing day is left out of the
    windows it falls in. The value is NaN until the frame reaches back over the whole window, and where no day of it counts.

    The renewables study's capacity proxy, the largest hourly generation over the 365 days ending D-1, is
    `RollingStatsTransformer("generation_solar", window_days=365, stat="max", lag_days=1)`, which adds
    `generation_solar_roll365max_d-1`. The window ends `lag_days` before the row's day and never reads it with `lag_days`
    of 1 or more. The information set judges the column through the one it was built from, at the last slot of the
    window's newest day (D - lag): a realised series needs `lag_days=1`, while a rolling statistic of a day-ahead price may end
    on D (`lag_days=0`, named without the shift).

    Args:
        column: The column to summarise.
        window_days: Local days in the window, at least 1. Default 365.
        stat: "min", "max" (default) or "mean", taken over each day and then over the window's days.
        lag_days: Days from the row's day back to the window's newest day, 0 or more. Default 1.
        freq: Slot length of the data, which decides when a day is complete. Default "1h".

    Example:
        >>> df = RollingStatsTransformer("generation_solar", window_days=365, stat="max", lag_days=1).transform(df)
    """

    def __init__(self, column: str, window_days: int = 365, stat: str = "max", lag_days: int = 1, freq: str = "1h"):
        if not isinstance(column, str) or not column:
            raise ValueError(f"column must be a column name, got {column!r}")
        self.column = column
        self.window_days = _whole(window_days, "window_days", 1)
        if stat not in STATS:
            raise ValueError(f"stat must be one of {list(STATS)}, got {stat!r}")
        self.stat = stat
        self.lag_days = _whole(lag_days, "lag_days", 0)
        self.freq = freq
        self._step = parse_step(freq)

    @property
    def output_column(self) -> str:
        """The name of the column it adds."""
        return rolling_name(self.column, self.window_days, self.stat, self.lag_days)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        index = check_index(df, "RollingStatsTransformer")
        if self.column not in df.columns:
            raise ValueError(f"RollingStatsTransformer: column {self.column!r} is not in the frame")
        slot_of_day(index, self._step, self.freq)  # rejects data off the freq grid
        check_spacing(index, self._step, self.freq, type(self).__name__)  # and data further apart than freq
        check_new_columns(df, [self.output_column], type(self).__name__)

        days = local_days(index)
        out = np.full(len(index), np.nan)
        if len(index):
            order = np.argsort(index.asi8, kind="stable")
            groups = df[self.column].astype(float).iloc[order].groupby(days[order])
            counts = groups.count()
            complete = counts.to_numpy() == slots_in_day(counts.index, index.tz, self._step)
            per_day = getattr(groups, self.stat)().where(complete)
            calendar = pd.date_range(per_day.index.min(), per_day.index.max(), freq="D")
            daily = per_day.reindex(calendar)
            window = getattr(daily.rolling(self.window_days, min_periods=1), self.stat)()
            window.iloc[: self.window_days - 1] = np.nan  # the window reaches back before the frame's first day
            out = window.shift(self.lag_days).reindex(days).to_numpy()
        return pd.concat([df, pd.DataFrame({self.output_column: out}, index=df.index)], axis=1)
