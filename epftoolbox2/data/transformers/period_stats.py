"""Statistics of a column over each hour of finer data (`PeriodStatsTransformer`): the hour's mean and each row's deviation
from it, the within-hour shape of quarter-hourly prices or generation."""

import numpy as np
import pandas as pd

from ._daily import as_column_list, check_new_columns, parse_step, slot_of_day
from .base import Transformer

_STATS = ("mean", "dev")
# The periods a statistic can be taken over, and the word of their column names.
_PERIODS = {pd.Timedelta(hours=1): "hour"}


def hour_starts(index: pd.DatetimeIndex) -> np.ndarray:
    """The instant each row's hour starts at (int64, the index's unit): one key per hour of the local clock and occurrence.

    The hour the clocks repeat in autumn is two hours, its first occurrence's rows and its second's; the hour they skip in
    spring has no rows. On a naive index (`DSTNormalizeTransformer`'s wall clock) an hour is its wall-clock hour."""
    wall = index.tz_localize(None) if index.tz is not None else index
    return np.asarray((index - (wall - wall.floor("h"))).asi8)


class PeriodStatsTransformer(Transformer):
    """The mean of a column over each hour's slots, on every row of the hour, and each row's deviation from it.

    Adds `{column}_hour_mean`, the mean over the slots of the row's hour (the four quarter-hours of 15-minute data), and
    `{column}_hour_dev`, the row's value minus that mean. An hour gets statistics only when it is complete: every one of
    its slots has a row and holds a value; otherwise both are NaN on its rows. Hours are those of the local clock, each
    occurrence on its own: the hour the clocks repeat in autumn has eight rows, two hours of four, and the one they skip in
    spring has none.

    A row of an hour only gets statistics of that hour. The information set judges both columns at the last slot of the
    hour (10:45 for 10:00 to 11:00 on 15-minute data), by the rule of the column they were built from: the deviation at
    10:15 of a realised series is known when its 10:45 value is, that of a day-ahead forecast with the forecast.

    Args:
        columns: Column name(s) to summarise.
        period: The period of the statistics, "1h" (the names say `_hour_`).
        stats: Which statistics to add, any of "mean" and "dev". Default both.
        freq: Slot length of the data; it must divide the period. Default "15min".

    Example:
        >>> df = PeriodStatsTransformer(columns="price").transform(df)  # price_hour_mean, price_hour_dev
    """

    def __init__(
        self,
        columns: str | list[str],
        period: str = "1h",
        stats: list[str] | tuple = _STATS,
        freq: str = "15min",
    ):
        self.columns = as_column_list(columns)
        self.period = period
        self.stats = [stats] if isinstance(stats, str) else list(stats)
        self.freq = freq
        self._step = parse_step(freq)
        try:
            length = pd.Timedelta(period)
        except (TypeError, ValueError):
            length = None
        if length not in _PERIODS:
            raise ValueError(f"period must be '1h', the one period with names of its own (_hour_), got {period!r}")
        if self._step > length or length % self._step != pd.Timedelta(0):
            raise ValueError(f"freq must divide the period {period!r} into whole slots, got {freq!r}")
        self._word = _PERIODS[length]
        self._slots = int(length // self._step)
        unknown = [s for s in self.stats if s not in _STATS]
        if unknown or not self.stats:
            raise ValueError(f"stats must be a non-empty subset of {list(_STATS)}, got {unknown or self.stats}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        index = df.index
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("PeriodStatsTransformer needs a DatetimeIndex")
        slot_of_day(index, self._step, self.freq)  # rejects data off the freq grid
        names = [f"{column}_{self._word}_{stat}" for column in self.columns for stat in self.stats]
        check_new_columns(df, names, type(self).__name__)

        keys = hour_starts(index)
        values = df[self.columns].astype(float).reset_index(drop=True)
        groups = values.groupby(keys)
        complete = groups.count().eq(self._slots)
        means = groups.mean().where(complete).reindex(keys).to_numpy()
        per_stat = {"mean": means, "dev": values.to_numpy() - means}

        out = {
            f"{column}_{self._word}_{stat}": per_stat[stat][:, j]
            for j, column in enumerate(self.columns)
            for stat in self.stats
        }
        return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)
