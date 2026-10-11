import math
from dataclasses import dataclass
from functools import reduce

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pandas.tseries.frequencies import to_offset

from .base import Transformer

AGGREGATIONS = ("mean", "sum", "first", "last", "min", "max", "median")
FILLS = ("ffill", None)
DEFAULT_KEY = "default"

_NUMERIC_ONLY = {"mean", "sum", "median"}
# Number of spacings on each side of an observation used to find the period it covers (see _periods).
_PERIOD_WINDOW = 4
_HOUR_NS = 3_600_000_000_000
_DAY_NS = 24 * _HOUR_NS
# Spacings of at least this length are whole local calendar days (23, 24 or 25 hours).
_CALENDAR_DAY_NS = 23 * _HOUR_NS


@dataclass
class _Observations:
    times: np.ndarray  # int64 ns (UTC for tz-aware indexes)
    values: np.ndarray  # float64 for numeric columns, object otherwise
    ends: np.ndarray  # int64 ns: end of the period each value covers (exclusive)
    numeric: bool


class ResampleTransformer(Transformer):
    """Bring every column of a frame to one regular frequency without inventing data.

    Each value is treated as the value of a *period* that starts at its timestamp and lasts as long as the
    column's own spacing (15 min, 1 h, one local calendar day, ...). The spacing is found per column from the
    timestamps of its non-null values, so a frame that mixes hourly prices, 15-minute load and daily A31
    values is handled column by column. Then:

    * **Upsampling** (the column is coarser than ``freq``): the finer slots inside a value's own period get
      that value (``fill="ffill"``, a step, e.g. every quarter-hour of 10:00-11:00 gets the 10:00 price). For
      ``sum`` columns the value is split evenly so totals are kept. ``fill=None`` leaves those slots empty.
      The last period is completed, so an hourly day becomes 96 quarter-hours.
    * **Downsampling** (the column is finer than ``freq``): the values inside each target period are combined
      with ``agg``. A period that is not completely covered (a missing quarter-hour, the partial first and
      last local day, ...) gives NaN, never a partial sum or mean.

    Nothing else is filled: a value missing in the source stays missing, and nothing is extrapolated past a
    column's last value, so no slot ever takes a value from a later timestamp. Fill real gaps explicitly and
    leak-free afterwards if you need to.

    Run it after ``TimezoneTransformer`` so that daily bins and daily columns follow local calendar days
    (23 or 25 hours on DST switch days).

    Args:
        freq: Target pandas frequency, e.g. ``"15min"``, ``"1h"``, ``"1D"``.
        agg: Aggregation used when downsampling: one of ``mean``, ``sum``, ``first``, ``last``, ``min``,
            ``max``, ``median``, or a dict ``{column: aggregation}``. Columns missing from the dict use the
            dict's ``"default"`` entry, else ``"mean"``. Non-numeric columns fall back to ``"first"`` unless
            given an aggregation that works for them.
        fill: How slots created by upsampling are filled: ``"ffill"`` (step fill inside the value's own period)
            or ``None`` (leave them empty); a dict ``{column: fill}`` works like ``agg``.

    Example:
        >>> ResampleTransformer(freq="1h", agg={"default": "mean", "energy_mwh": "sum"}).transform(df)
    """

    def __init__(
        self,
        freq: str = "1h",
        agg: str | dict[str, str] = "mean",
        fill: str | dict[str, str | None] | None = "ffill",
    ):
        self.freq = freq
        self.agg = agg
        self.fill = _normalise_fill(fill)
        self._offset = _parse_freq(freq)
        _validate_agg(agg)
        self._agg_default, self._agg_by_column = _split(self.agg, "mean")
        self._fill_default, self._fill_by_column = _split(self.fill, "ffill")

    def _check_columns(self, df: pd.DataFrame) -> None:
        named = set(self._agg_by_column) | set(self._fill_by_column)
        missing = named - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {sorted(missing)}")

    def _column_agg(self, series: pd.Series, numeric: bool) -> str:
        agg = self._agg_by_column.get(series.name, self._agg_default)
        if numeric or agg not in _NUMERIC_ONLY:
            return agg
        if series.name in self._agg_by_column:
            raise ValueError(f"ResampleTransformer: agg '{agg}' needs a numeric column, '{series.name}' is not")
        return "first"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        self._check_columns(df)
        if df.index.has_duplicates:
            duplicated = df.index[df.index.duplicated()].unique()
            raise ValueError(
                f"ResampleTransformer: the index has {len(duplicated)} duplicate timestamp(s), e.g. {duplicated[0]}"
            )
        if len(df) == 0:
            return df.copy()

        unit = df.index.unit
        df = df.set_axis(df.index.as_unit("ns"))
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        fallback = int(np.median(np.diff(df.index.asi8))) if len(df) > 1 else _nominal_ns(self._offset)
        observations = {col: _observe(df[col], fallback) for col in df.columns}
        labels, end = self._target_bins(df.index, observations)
        step = self._grid_step(observations)
        grid = pd.date_range(labels[0], end, freq=pd.Timedelta(step), inclusive="left").as_unit("ns")
        if len(grid) > 64 * (len(df) + len(labels)):
            raise ValueError(
                f"ResampleTransformer: timestamps are too irregular to resample (common step {pd.Timedelta(step)})"
            )

        grid_ns, label_ns = grid.asi8, labels.asi8
        bin_of_slot = np.searchsorted(label_ns, grid_ns, side="right") - 1
        one_to_one = len(grid) == len(labels)
        slots_per_bin = np.bincount(bin_of_slot, minlength=len(labels))

        columns = {}
        for col in df.columns:
            obs = observations[col]
            if obs is None:
                columns[col] = np.full(len(labels), np.nan)
                continue
            agg = self._column_agg(df[col], obs.numeric)
            fine = _spread(obs, grid_ns, split=agg == "sum")
            if one_to_one:
                values = fine
            else:
                grouped = pd.Series(fine).groupby(bin_of_slot)
                complete = grouped.count().to_numpy() == slots_per_bin
                values = grouped.agg(agg).to_numpy()
                values = np.where(complete, values, np.nan if obs.numeric else None)
            if self._fill_by_column.get(col, self._fill_default) is None:
                starts = np.bincount(np.searchsorted(label_ns, obs.times, side="right") - 1, minlength=len(labels))
                values = np.where(starts > 0, values, np.nan if obs.numeric else None)
            columns[col] = values

        result = pd.DataFrame(columns, index=labels.as_unit(unit), columns=df.columns)
        result.index.name = df.index.name
        return result

    def _target_bins(self, index: pd.DatetimeIndex, observations: dict) -> tuple[pd.DatetimeIndex, pd.Timestamp]:
        """Bin labels from the first row to the bin holding the end of the last period, plus that bin's end."""
        last_needed = index[-1]
        ends = [obs.ends.max() for obs in observations.values() if obs is not None]
        if ends:
            last_end = _to_timestamp(max(ends) - 1, index.tz)
            last_needed = max(last_needed, last_end)
        probe = pd.DatetimeIndex([index[0], last_needed, last_needed + 2 * self._offset])
        extended = pd.Series(0, index=probe).resample(self.freq).size().index.as_unit("ns")
        count = extended.searchsorted(last_needed, side="right")
        return extended[:count], extended[count]

    def _grid_step(self, observations: dict) -> int:
        """Finest step (ns) that divides the target bins and every column's period."""
        steps = {_HOUR_NS}
        if isinstance(self._offset, pd.offsets.Tick):
            steps.add(int(self._offset.nanos))
        for obs in observations.values():
            if obs is not None:
                spans = obs.ends - obs.times
                steps.update(int(s) for s in np.unique(spans[spans < _CALENDAR_DAY_NS]))
        return reduce(math.gcd, steps)


def _observe(series: pd.Series, fallback: int) -> _Observations | None:
    present = series.notna().to_numpy()
    if not present.any():
        return None
    numeric = pd.api.types.is_numeric_dtype(series.dtype)
    values = series.to_numpy(dtype="float64", na_value=np.nan) if numeric else series.to_numpy(dtype=object)
    index = series.index[present]
    times = index.asi8
    periods = _periods(times, fallback)
    ends = times + periods
    calendar = periods >= _CALENDAR_DAY_NS
    if calendar.any():
        days = np.rint(periods[calendar] / _DAY_NS).astype(np.int64)
        ends[calendar] = _add_local_days(index[calendar], days)
    return _Observations(times=times, values=values[present], ends=ends, numeric=numeric)


def _periods(times: np.ndarray, fallback: int) -> np.ndarray:
    """Length of the period each observation covers, from the spacing of its column's observations.

    The smallest of the few spacings before an observation, and the smallest of the few after it, are both
    ignorant of a single gap (a missing hour does not make its neighbours two hours long); taking the larger of
    the two follows a change of resolution inside a column (hourly prices, then 15-minute prices) in either
    direction. Only timestamps are used, never values.
    """
    count = len(times)
    if count == 1:
        return np.array([fallback], dtype=np.int64)
    never = np.iinfo(np.int64).max
    padding = np.full(_PERIOD_WINDOW, never)
    window_min = sliding_window_view(np.concatenate([padding, np.diff(times), padding]), _PERIOD_WINDOW).min(axis=1)
    before, after = window_min[:count], window_min[_PERIOD_WINDOW : _PERIOD_WINDOW + count]
    before = np.where(before == never, after, before)
    after = np.where(after == never, before, after)
    return np.maximum(before, after)


def _add_local_days(index: pd.DatetimeIndex, days: np.ndarray) -> np.ndarray:
    """`index + days` on the local wall clock (so a day is 23, 24 or 25 hours long), as int64 ns."""
    wall = index.tz_localize(None) if index.tz is not None else index
    shifted = wall + pd.to_timedelta(days, unit="D")
    if index.tz is not None:
        shifted = shifted.tz_localize(
            index.tz, ambiguous=np.zeros(len(shifted), dtype=bool), nonexistent="shift_forward"
        )
    return shifted.as_unit("ns").asi8


def _spread(obs: _Observations, grid_ns: np.ndarray, split: bool) -> np.ndarray:
    """Value of each grid slot: the observation whose period contains it, or missing."""
    position = np.searchsorted(obs.times, grid_ns, side="right") - 1
    covered = position >= 0
    covered[covered] = grid_ns[covered] < obs.ends[position[covered]]
    owner = position[covered]
    if obs.numeric:
        fine = np.full(len(grid_ns), np.nan)
        fine[covered] = obs.values[owner]
        if split:
            fine[covered] /= np.bincount(owner, minlength=len(obs.times))[owner]
    else:
        fine = np.full(len(grid_ns), None, dtype=object)
        fine[covered] = obs.values[owner]
    return fine


def _to_timestamp(ns: int, tz) -> pd.Timestamp:
    return pd.Timestamp(ns, tz="UTC").tz_convert(tz) if tz is not None else pd.Timestamp(ns)


def _nominal_ns(offset) -> int:
    return int(offset.nanos) if isinstance(offset, pd.offsets.Tick) else _DAY_NS


def _parse_freq(freq: str):
    try:
        return to_offset(freq)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid freq: {freq!r} is not a pandas frequency string ({exc})") from None


def _split(value, fallback) -> tuple:
    if isinstance(value, dict):
        return value.get(DEFAULT_KEY, fallback), {k: v for k, v in value.items() if k != DEFAULT_KEY}
    return value, {}


def _validate_agg(agg) -> None:
    values = agg.values() if isinstance(agg, dict) else [agg]
    bad = [v for v in values if v not in AGGREGATIONS]
    if bad or (isinstance(agg, dict) and not all(isinstance(k, str) for k in agg)):
        raise ValueError(
            f"Invalid agg: {agg!r}. Use one of {list(AGGREGATIONS)} or a dict {{column: aggregation}}; "
            "interpolation is not supported because it uses later values"
        )


def _normalise_fill(fill):
    def one(value):
        if value is None or (isinstance(value, str) and value.lower() == "none"):
            return None
        if value in FILLS:
            return value
        raise ValueError(
            f"Invalid fill: {value!r}. Use 'ffill' (step fill inside each value's own period) or None; "
            "interpolation and backward fill are not supported because they use later values"
        )

    if isinstance(fill, dict):
        return {key: one(value) for key, value in fill.items()}
    return one(fill)
