import warnings
from typing import Optional

import numpy as np
import pandas as pd

from ._daily import as_column_list, check_index, check_new_columns, local_days
from .base import Transformer

_MAD_TO_STD = 1.4826  # MAD * 1.4826 estimates the standard deviation of normal data
_SIDES = ("both", "upper", "lower")


class SpikeFlagTransformer(Transformer):
    """Flag values far from the same slot's recent history: 1 for a spike, 0 otherwise.

    For a row at slot s (e.g. 19:00) on local day D, the reference band is median ± k·MAD of the column's values
    at slot s on the `window` calendar days before D (D itself excluded, like `closed="left"`). MAD is scaled by
    1.4826 so it estimates the standard deviation for normal data. The flag is NaN when the value is missing or
    fewer than `min_periods` past values exist.

    The band uses only days before D, so a row's flag depends on its own value and on earlier days, never on
    later data. Like the column itself, a flag on D is only known when a forecast is issued on D if the value
    is (day-ahead prices are); otherwise use it lagged (`price_spike_d-1`).

    Args:
        columns: Column name(s) to flag. Adds `{column}_spike` for each.
        window: Number of previous calendar days in the reference band. Default 28.
        k: Band half-width in (scaled) MADs. Default 3.0.
        min_periods: Past values needed for a flag. Default None = half the window.
        side: "both", "upper" (only values above the band) or "lower". Default "both".

    Example:
        >>> df = SpikeFlagTransformer(columns="price", window=28, k=3.0).transform(df)  # adds price_spike
    """

    def __init__(
        self,
        columns: str | list[str],
        window: int = 28,
        k: float = 3.0,
        min_periods: Optional[int] = None,
        side: str = "both",
    ):
        self.columns = as_column_list(columns)
        self.window = int(window)
        self.k = float(k)
        self.min_periods = None if min_periods is None else int(min_periods)
        self.side = side
        if self.window < 1:
            raise ValueError(f"window must be at least 1 day, got {window}")
        if self.min_periods is not None and not 1 <= self.min_periods <= self.window:
            raise ValueError(f"min_periods must be between 1 and window ({self.window}), got {min_periods}")
        if side not in _SIDES:
            raise ValueError(f"side must be one of {_SIDES}, got {side!r}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        index = check_index(df, "SpikeFlagTransformer")
        names = [f"{column}_spike" for column in self.columns]
        check_new_columns(df, names, type(self).__name__)

        days = local_days(index)
        wall = index.tz_localize(None) if index.tz is not None else index
        time_of_day = wall - days
        calendar = pd.date_range(days.min(), days.max(), freq="D")
        slots = pd.Index(time_of_day.unique()).sort_values()
        day_pos, slot_pos = calendar.get_indexer(days), slots.get_indexer(time_of_day)
        first = ~pd.MultiIndex.from_arrays([days, time_of_day]).duplicated(keep="first")
        min_periods = self.min_periods if self.min_periods is not None else max(1, self.window // 2)

        out = {}
        for column, name in zip(self.columns, names):
            values = df[column].to_numpy(dtype=float)
            # history[d, s]: the value at slot s on calendar day d (first occurrence on the autumn switch day).
            history = np.full((len(calendar), len(slots)), np.nan)
            history[day_pos[first], slot_pos[first]] = values[first]
            padded = np.vstack([np.full((self.window, len(slots)), np.nan), history])
            # windows[d] holds days d-window .. d-1 of history, never day d itself.
            windows = np.lib.stride_tricks.sliding_window_view(padded, self.window, axis=0)[: len(calendar)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
                median = np.nanmedian(windows, axis=-1)
                mad = np.nanmedian(np.abs(windows - median[..., None]), axis=-1) * _MAD_TO_STD
            enough = (~np.isnan(windows)).sum(axis=-1) >= min_periods

            centre, spread = median[day_pos, slot_pos], self.k * mad[day_pos, slot_pos]
            above, below = values > centre + spread, values < centre - spread
            flag = {"both": above | below, "upper": above, "lower": below}[self.side].astype(float)
            flag[~enough[day_pos, slot_pos] | np.isnan(values)] = np.nan
            out[name] = flag
        return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)
