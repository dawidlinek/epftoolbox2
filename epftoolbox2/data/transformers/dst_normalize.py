from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..._date_utils import is_utc_index
from .base import Transformer

FLAG_COLUMN = "is_dst_adjusted"


class DSTNormalizeTransformer(Transformer):
    """Give every local day exactly 24 (or 96) wall-clock slots, as in Lago et al. (2021).

    The index becomes naive local wall-clock time. On the day clocks go back, the two occurrences of the
    repeated hour are averaged into one slot; on the day clocks go forward, the skipped hour is filled with
    the average of the slot before and the slot after it (the same value for every skipped slot of a 15-min
    day). Non-numeric columns take the first occurrence and the slot before, respectively. A missing value
    in either input gives a missing result. Filled and averaged slots get `is_dst_adjusted = 1`, every other
    row 0; the other values are left as they are. Gaps that are not caused by DST, and partial days at the
    ends of the data, are left alone.

    The values on flagged slots are synthetic, including prices that end up as model targets and
    "actuals". Exclude them from error metrics, or report them separately.

    Every filled value comes from the same local day, so no row uses data from a later day. Run it after
    `TimezoneTransformer` (and any resampling) and before `LagTransformer`: on the naive index day lags
    are plain 24-slot shifts, and the model sees one row per hour and day. Don't run `TimezoneTransformer`
    afterwards; it would read the naive wall-clock index as UTC.

    Args:
        freq: Slot length of the data, e.g. "1h" (24 slots a day) or "15min" (96). Default "1h".
        tz: Time zone to convert the index to first. If None, the index must already be in local time.

    Example:
        >>> df = DSTNormalizeTransformer(freq="1h").transform(TimezoneTransformer("Europe/Warsaw").transform(df))
    """

    def __init__(self, freq: str = "1h", tz: Optional[str] = None):
        self.freq = freq
        self.tz = tz
        self._step = pd.Timedelta(freq)
        if self._step <= pd.Timedelta(0) or pd.Timedelta(days=1) % self._step != pd.Timedelta(0):
            raise ValueError(f"freq must divide a day into whole slots, got {freq!r}")
        if tz is not None:
            try:
                ZoneInfo(tz)
            except (KeyError, ValueError):
                raise ValueError(f"Invalid timezone: {tz!r}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        index = df.index
        if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
            raise ValueError("DSTNormalizeTransformer needs a time-zone-aware DatetimeIndex (local time)")
        if self.tz is not None:
            index = index.tz_convert(self.tz)
        if is_utc_index(index):
            raise ValueError(
                "DSTNormalizeTransformer got a UTC index, which has no DST. Pass tz= or run TimezoneTransformer first."
            )
        if FLAG_COLUMN in df.columns:
            raise ValueError(f"Column '{FLAG_COLUMN}' already exists; the data looks normalised already")
        if not (index.is_unique and index.is_monotonic_increasing):
            raise ValueError("DSTNormalizeTransformer needs a sorted index without duplicate timestamps")

        wall = index.tz_localize(None)
        offsets = wall - wall.normalize()
        if (offsets % self._step != pd.Timedelta(0)).any():
            raise ValueError(f"Timestamps are not on the {self.freq} grid; resample to {self.freq} first")
        gaps = wall[1:] - wall[:-1]
        positive_gaps = gaps[gaps > pd.Timedelta(0)]
        if len(positive_gaps) and positive_gaps.min() != self._step:
            # Coarser data (e.g. hourly) still lands on a finer freq's grid, so the check above alone misses it.
            raise ValueError(
                f"Data is spaced {positive_gaps.min()} apart, not the configured freq {self.freq!r}; "
                f"resample to {self.freq} first"
            )

        data = df.set_axis(wall)
        numeric = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
        data = data.astype({c: "float64" for c in numeric})

        repeated = wall.duplicated(keep=False)
        if repeated.any():
            # A wall-clock time occurs at most twice; average the pair (a missing value stays missing).
            pairs = data[repeated]
            first = pairs[~pairs.index.duplicated(keep="first")]
            second = pairs[~pairs.index.duplicated(keep="last")]
            merged = first.copy()
            merged[numeric] = (first[numeric] + second[numeric]) / 2
            data = pd.concat([data[~repeated], merged]).sort_index()

        filled = self._fill_skipped_slots(data, numeric, index.tz)
        flag = pd.Series(0.0, index=data.index)
        flag[wall[repeated]] = 1.0
        if filled is not None:
            data = pd.concat([data, filled]).sort_index()
            flag = pd.concat([flag, pd.Series(1.0, index=filled.index)]).sort_index()
        data[FLAG_COLUMN] = flag.astype(int)
        return data

    def _fill_skipped_slots(self, data: pd.DataFrame, numeric: list, tz) -> Optional[pd.DataFrame]:
        """Rows for the wall-clock slots the spring switch skips, within the span of the data."""
        start, end = data.index[0], data.index[-1]
        grid = pd.date_range(
            start.normalize(), end.normalize() + pd.Timedelta(days=1), freq=self._step, inclusive="left"
        )
        exists = grid.tz_localize(tz, ambiguous=np.ones(len(grid), dtype=bool), nonexistent="NaT")
        skipped = grid[exists.isna() & (grid >= start) & (grid <= end)]
        if skipped.empty:
            return None

        # Consecutive skipped slots form one run per switch; each run takes the mean of its two neighbours.
        run_ids = np.concatenate([[0], np.cumsum((skipped[1:] - skipped[:-1]) != self._step)])
        rows = []
        for run_id in np.unique(run_ids):
            run = skipped[run_ids == run_id]
            before = data.reindex([run[0] - self._step]).iloc[0]
            after = data.reindex([run[-1] + self._step]).iloc[0]
            values = before.copy()
            if (run[-1] + self._step).date() == run[0].date():
                # Plain arithmetic, not DataFrame.mean: a missing neighbour must give a missing slot.
                values[numeric] = (before[numeric].astype(float) + after[numeric].astype(float)) / 2
            rows.append(pd.DataFrame([values.to_numpy()] * len(run), index=run, columns=data.columns))
        filled = pd.concat(rows)
        return filled.astype({c: "float64" for c in numeric})
