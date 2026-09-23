import numpy as np
import pandas as pd

from ._daily import as_column_list, check_index, check_new_columns, local_days, parse_step, slot_of_day, slots_in_day
from .base import Transformer

_STATS = ("min", "max", "mean", "last")


class DailyStatsTransformer(Transformer):
    """Per local day minimum, maximum, mean and last value of a column, on every row of that day.

    Adds `{column}_day_min`, `{column}_day_max`, `{column}_day_mean` and `{column}_day_last` (the value in the
    last slot of the day, 23:00 for hourly data). A day gets statistics only when it is complete: every slot it
    has (23, 24 or 25 hours on a time-zone-aware index, 24 after `DSTNormalizeTransformer`) holds a value.
    Incomplete days, including partial days at the ends of the data, get NaN.

    A row of day D only gets statistics of day D, never a rolling window that reaches into D+1. Whether they are
    known when a forecast is issued on D depends on the column: day-ahead prices for D are, realised values are
    not, so use those only lagged (`load_actual_day_max_d-1`).

    Args:
        columns: Column name(s) to summarise.
        stats: Which statistics to add, any of "min", "max", "mean", "last". Default all four.
        freq: Slot length of the data, used to decide when a day is complete. Default "1h".

    Example:
        >>> df = DailyStatsTransformer(columns="price", stats=["min", "max"]).transform(df)
    """

    def __init__(self, columns: str | list[str], stats: list[str] | tuple = _STATS, freq: str = "1h"):
        self.columns = as_column_list(columns)
        self.stats = [stats] if isinstance(stats, str) else list(stats)
        self.freq = freq
        self._step = parse_step(freq)
        unknown = [s for s in self.stats if s not in _STATS]
        if unknown or not self.stats:
            raise ValueError(f"stats must be a non-empty subset of {list(_STATS)}, got {unknown or self.stats}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        index = check_index(df, "DailyStatsTransformer")
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
        # On a complete day no value is missing, so last() (last non-missing) is the value of the last slot.
        per_day = {stat: getattr(groups, stat)().where(complete) for stat in self.stats}

        out = {
            f"{column}_day_{stat}": per_day[stat][column].reindex(days).to_numpy()
            for column in self.columns
            for stat in self.stats
        }
        return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)
