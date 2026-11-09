import numpy as np
import pandas as pd

from ._daily import (
    PEAK_HOURS,
    as_column_list,
    check_index,
    check_new_columns,
    check_unique_index,
    local_days,
    parse_step,
    slot_of_day,
    slots_in_day,
    wall_hours,
)
from .base import Transformer

_STATS = ("min", "max", "mean", "last")
# What `stats` may name, `{column}_day_{stat}`: the default four, and the mean over the peak hours. The information set reads
# these names (`information_set._DAY_AGGREGATE` is built from them).
STATS = (*_STATS, "peak")


class DailyStatsTransformer(Transformer):
    """Per local day minimum, maximum, mean and last value of a column, and its mean over the peak hours, on every row of
    that day.

    Adds `{column}_day_min`, `{column}_day_max`, `{column}_day_mean` and `{column}_day_last` (the value in the
    last slot of the day, 23:00 for hourly data), and with "peak" in `stats` `{column}_day_peak`, the mean over the
    slots whose local hour is in `peak_hours` (08:00 to 19:59 by default). A day gets statistics only when it is complete:
    every slot it has (23, 24 or 25 hours on a time-zone-aware index, 24 after `DSTNormalizeTransformer`) holds a value.
    Incomplete days, including partial days at the ends of the data, get NaN, the peak too, whichever slot is missing.

    A row of day D only gets statistics of day D, never a rolling window that reaches into D+1. Whether they are
    known when a forecast is issued on D depends on the column: day-ahead prices for D are, realised values are
    not, so use those only lagged (`load_actual_day_max_d-1`). The information set judges every statistic, the peak
    included, at the last slot of its day.

    Args:
        columns: Column name(s) to summarise.
        stats: Which statistics to add, any of "min", "max", "mean", "last", "peak". Default the first four.
        freq: Slot length of the data, used to decide when a day is complete. Default "1h".
        peak_hours: `(first, end)` local hours of the peak block, the end left out: `(8, 20)` is 08:00 to 19:59. Only
            for "peak".

    Example:
        >>> df = DailyStatsTransformer(columns="price", stats=["min", "max", "peak"]).transform(df)
    """

    # Parameters a saved pipeline writes only when not at these values, so files saved now load in versions from before them.
    _WRITTEN_WHEN_SET = {"peak_hours": PEAK_HOURS}

    def __init__(
        self,
        columns: str | list[str],
        stats: list[str] | tuple = _STATS,
        freq: str = "1h",
        peak_hours: tuple[int, int] | list[int] = PEAK_HOURS,
    ):
        from ...results.breakdowns import check_peak_hours  # results import the models, which import the data layer

        self.columns = as_column_list(columns)
        self.stats = [stats] if isinstance(stats, str) else list(stats)
        self.freq = freq
        self.peak_hours = check_peak_hours(peak_hours)
        self._step = parse_step(freq)
        unknown = [s for s in self.stats if s not in STATS]
        if unknown or not self.stats:
            raise ValueError(f"stats must be a non-empty subset of {list(STATS)}, got {unknown or self.stats}")
        if "peak" not in self.stats and self.peak_hours != PEAK_HOURS:
            raise ValueError(f"peak_hours sets the block of stats=['peak']; it means nothing for stats={self.stats}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        index = check_index(df, "DailyStatsTransformer")
        check_unique_index(index, "DailyStatsTransformer")
        slot_of_day(index, self._step, self.freq)  # rejects data off the freq grid
        names = [f"{column}_day_{stat}" for column in self.columns for stat in self.stats]
        check_new_columns(df, names, type(self).__name__)

        order = np.argsort(index.asi8, kind="stable")
        days = local_days(index)
        values = df[self.columns].astype(float).iloc[order]
        groups = values.groupby(days[order])
        counts = groups.count()
        expected = slots_in_day(counts.index, index.tz, self._step)
        complete = counts.eq(pd.Series(expected, index=counts.index), axis=0)
        # On a complete day no value is missing, so last() (last non-missing) is the value of the last slot, and the
        # mean of the values left in the peak block is the mean over all of its slots.
        per_day = {stat: getattr(groups, stat)().where(complete) for stat in self.stats if stat != "peak"}
        if "peak" in self.stats:
            hours = wall_hours(index)[order]
            block = (hours >= self.peak_hours[0]) & (hours < self.peak_hours[1])
            peak = pd.DataFrame(np.where(block[:, None], values.to_numpy(), np.nan), columns=values.columns)
            per_day["peak"] = peak.groupby(days[order]).mean().where(complete)

        out = {
            f"{column}_day_{stat}": per_day[stat][column].reindex(days).to_numpy()
            for column in self.columns
            for stat in self.stats
        }
        return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)
