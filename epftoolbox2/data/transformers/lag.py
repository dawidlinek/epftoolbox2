import pandas as pd
from .base import Transformer

_UNIT_THRESHOLDS = [(86400, "d"), (3600, "h"), (60, "min"), (1, "s")]


class LagTransformer(Transformer):
    """Transform columns by shifting them to create lagged features.
    Args:
        columns: Column name(s) to create lags for. If None, uses all columns.
        lags: Positive values look back, negative values look forward.
        freq: Frequency string. Accepts intuitive names ("day", "hour", "minute", "second")
              or pandas frequency strings (e.g., "1h", "15min", "1D"). Default is "1h".

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
        self.columns = [columns] if isinstance(columns, str) else columns
        self.lags = [lags] if isinstance(lags, int) else list(lags)
        self.freq = freq

        if not self.lags:
            raise ValueError("At least one lag value must be provided")
        if not self.columns:
            raise ValueError("At least one column must be provided")

        mapping = self._FREQ_MAPPING.get(freq.lower())
        if mapping:
            freq_normalized, self._freq_unit = mapping
        else:
            freq_normalized, self._freq_unit = freq, None
        self._freq = pd.Timedelta(freq_normalized)

    def _format_lag_name(self, column: str, lag: int) -> str:
        if self._freq_unit:
            unit, value = self._freq_unit, abs(lag)
        else:
            total_seconds = int((self._freq * abs(lag)).total_seconds())
            value, unit = next(
                (total_seconds // d, u)
                for d, u in _UNIT_THRESHOLDS
                if total_seconds % d == 0
            )
        sign = "-" if lag >= 0 else "+"
        return f"{column}_{unit}{sign}{value}"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        lagged_data = {}
        for column in self.columns:
            series = df[column]
            for lag in self.lags:
                name = self._format_lag_name(column, lag)
                shifted = pd.Series(series.values, index=df.index + self._freq * lag)
                lagged_data[name] = shifted.reindex(df.index)

        return pd.concat([df, pd.DataFrame(lagged_data, index=df.index)], axis=1)
