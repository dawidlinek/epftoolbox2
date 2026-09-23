import warnings

import numpy as np
import pandas as pd

from ._daily import (
    as_column_list,
    check_index,
    check_new_columns,
    local_days,
    parse_step,
    slot_names,
    slot_of_day,
    slots_in_day,
)
from .base import Transformer


class DailyProfileTransformer(Transformer):
    """Copy each local day's values of a column onto every row of that day.

    With hourly data, `price` becomes `price_h00`..`price_h23`: on every row of day D, `price_h05` is the price
    at 05:00 on D. 15-minute data gives `price_s00`..`price_s95`. Lagging these columns by days then expresses
    cross-hour inputs such as "all 24 prices of the previous day" (LEAR) in the one-model-per-hour setup.

    A row of day D only gets values of day D. Whether they are known when a forecast is issued on D depends on
    the column: day-ahead prices and day-ahead forecasts for D are, realised values (actual load) are not, so
    use those only lagged (`load_actual_h05_d-1`).

    On a time-zone-aware index the skipped spring slot is NaN and the repeated autumn slot takes its first
    occurrence, and a warning suggests running `DSTNormalizeTransformer` first, which gives complete profiles.

    Args:
        columns: Column name(s) to spread.
        freq: Slot length of the data, e.g. "1h" (24 columns per input column) or "15min" (96). Default "1h".

    Example:
        >>> df = DailyProfileTransformer(columns=["price"]).transform(df)  # price_h00 .. price_h23
    """

    def __init__(self, columns: str | list[str], freq: str = "1h"):
        self.columns = as_column_list(columns)
        self.freq = freq
        self._step = parse_step(freq)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        index = check_index(df, "DailyProfileTransformer")
        n_slots = pd.Timedelta(days=1) // self._step
        days = local_days(index)
        slots = slot_of_day(index, self._step, self.freq)
        names = {column: slot_names(column, self._step) for column in self.columns}
        check_new_columns(df, [name for column_names in names.values() for name in column_names], type(self).__name__)

        unique_days = days.unique()
        if index.tz is not None and (slots_in_day(unique_days, index.tz, self._step) != n_slots).any():
            warnings.warn(
                "DailyProfileTransformer: some local days have 23 or 25 hours; their skipped slot is NaN and the "
                "repeated slot takes its first occurrence. Run DSTNormalizeTransformer first for full profiles.",
                UserWarning,
                stacklevel=2,
            )

        first = ~pd.MultiIndex.from_arrays([days, slots]).duplicated(keep="first")
        day_pos = unique_days.get_indexer(days)
        blocks = []
        for column in self.columns:
            table = np.full((len(unique_days), n_slots), np.nan)
            table[day_pos[first], slots[first]] = df[column].to_numpy(dtype=float)[first]
            blocks.append(pd.DataFrame(table[day_pos], index=df.index, columns=names[column]))
        return pd.concat([df, *blocks], axis=1)
