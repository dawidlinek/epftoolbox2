import numbers
import warnings
from collections.abc import Iterable

import pandas as pd

from .base import Transformer
from ..._date_utils import is_utc_index as _is_utc
from ..._date_utils import shift_calendar_days

_ELAPSED_UNITS = [(3600, "h"), (60, "min"), (1, "s")]


def _as_int_list(lags) -> list[int]:
    items = list(lags) if isinstance(lags, Iterable) and not isinstance(lags, str) else [lags]
    for lag in items:
        if isinstance(lag, bool) or not isinstance(lag, numbers.Integral):
            raise TypeError(f"Lags must be integers, got {lag!r}")
    return [int(lag) for lag in items]


class LagTransformer(Transformer):
    """Transform columns by shifting them to create lagged features.

    Day units shift by local calendar days: on a local-time index, `price_d-1` at 10:00 is the price at 10:00
    on the previous local date, also across DST switches (23- and 25-hour days). The hour skipped in spring
    has no value (NaN); both occurrences of the hour repeated in autumn look back to that hour's only
    occurrence, and looking onto the repeated hour from another day takes its first occurrence. Other units
    (hours, minutes, seconds, or any frequency that isn't a whole number of days such as "24h" or "15min")
    shift by elapsed time and are named in h/min/s, so an elapsed 24-hour lag is `price_h-24`, never
    `price_d-1`.

    Args:
        columns: Column name(s) to create lags for. If None, uses all columns of the input frame (backward
              lags only: forward lags of every column would include realised series and the target).
        lags: Positive values look back, negative values look forward. Lag 0 is skipped, since it is
              the column itself.
        freq: Frequency string. Accepts intuitive names ("day", "hour", "minute", "second")
              or pandas frequency strings (e.g., "1h", "15min", "1D", "2D"). Default is "1h".

    Day lags need a local-time index, so run them after `TimezoneTransformer`; on a UTC index they
    shift by UTC days and a warning is raised.

    Example:
        >>> transformer = LagTransformer(columns=["price"], lags=[1, 24], freq="hour")
        >>> result = transformer.transform(df)
    """

    _FREQ_MAPPING: dict[str, tuple[str, str]] = {}
    for _aliases, _td, _unit in [
        (("day", "days", "d", "1d"), "1D", "d"),
        (("hour", "hours", "h", "1h"), "1h", "h"),
        (("minute", "minutes", "min", "1min"), "1min", "min"),
        (("second", "seconds", "s", "1s"), "1s", "s"),
    ]:
        for _alias in _aliases:
            _FREQ_MAPPING[_alias] = (_td, _unit)

    def __init__(
        self,
        columns: str | list[str] | None = None,
        lags: int | list[int] = 1,
        freq: str = "1h",
    ):
        self.columns = [columns] if isinstance(columns, str) else (None if columns is None else list(columns))
        self.lags = _as_int_list(lags)
        self.freq = freq

        if not self.lags:
            raise ValueError("At least one lag value must be provided")
        if all(lag == 0 for lag in self.lags):
            raise ValueError("Only lag 0 was given; lag 0 is the column itself")
        if self.columns is not None and not self.columns:
            raise ValueError("At least one column must be provided (or None for all columns)")
        if self.columns is None and any(lag < 0 for lag in self.lags):
            raise ValueError(
                "columns=None works with backward lags only: forward lags of every column would look up realised "
                "future values and create columns named like the model's targets. List the columns explicitly."
            )

        mapping = self._FREQ_MAPPING.get(freq.lower())
        if mapping:
            freq_normalized, self._freq_unit = mapping
        else:
            freq_normalized, self._freq_unit = freq, None
        self._freq = pd.Timedelta(freq_normalized)
        # Whole days given in day units ("day", "1D", "2D") shift by calendar days; "24h" stays elapsed time.
        if self._freq_unit == "d":
            self._calendar_days = 1
        else:
            self._calendar_days = self._day_count(freq) if self._freq_unit is None else None

    @staticmethod
    def _day_count(freq: str) -> int | None:
        try:
            offset = pd.tseries.frequencies.to_offset(freq)
        except ValueError:
            return None
        return int(offset.n) if isinstance(offset, pd.offsets.Day) else None

    def _format_lag_name(self, column: str, lag: int) -> str:
        sign = "-" if lag >= 0 else "+"
        if self._calendar_days is not None:
            return f"{column}_d{sign}{self._calendar_days * abs(lag)}"
        if self._freq_unit:
            unit, value = self._freq_unit, abs(lag)
        else:
            total_seconds = int((self._freq * abs(lag)).total_seconds())
            value, unit = next((total_seconds // d, u) for d, u in _ELAPSED_UNITS if total_seconds % d == 0)
        return f"{column}_{unit}{sign}{value}"

    def _source_index(self, index: pd.DatetimeIndex, lag: int) -> pd.DatetimeIndex:
        if self._calendar_days is not None:
            return shift_calendar_days(index, -self._calendar_days * lag)
        return index - self._freq * lag

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = list(df.columns) if self.columns is None else self.columns
        if not df.index.is_unique:
            raise ValueError("LagTransformer needs an index without duplicate timestamps")
        if self._calendar_days is not None and _is_utc(df.index):
            warnings.warn(
                "LagTransformer: day lags on a UTC index shift by UTC days, so local hours drift by one around "
                "DST switches and daily features can point at the wrong local day. Run TimezoneTransformer "
                "before LagTransformer.",
                UserWarning,
                stacklevel=2,
            )

        lags = [lag for lag in dict.fromkeys(self.lags) if lag != 0]
        names = [self._format_lag_name(column, lag) for column in columns for lag in lags]
        clashes = sorted({name for name in names if name in df.columns or names.count(name) > 1})
        if clashes:
            raise ValueError(f"LagTransformer would create columns that already exist: {clashes}")

        blocks = []
        for lag in lags:
            block = df[columns].reindex(self._source_index(df.index, lag))
            block.index = df.index
            block.columns = [self._format_lag_name(column, lag) for column in columns]
            blocks.append(block)
        return pd.concat([df, pd.concat(blocks, axis=1)[names]], axis=1)
