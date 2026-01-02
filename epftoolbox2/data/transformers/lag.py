import pandas as pd
from .base import Transformer


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

    _FREQ_MAPPING = {
        "day": "1D",
        "days": "1D",
        "d": "1D",
        "hour": "1h",
        "hours": "1h",
        "h": "1h",
        "minute": "1min",
        "minutes": "1min",
        "min": "1min",
        "second": "1s",
        "seconds": "1s",
        "s": "1s",
    }

    def __init__(
        self,
        columns: str | list[str] | None = None,
        lags: int | list[int] = 1,
        freq: str = "1h",
    ):
        self.columns = [columns] if isinstance(columns, str) else columns
        self.lags = [lags] if isinstance(lags, int) else lags
        freq_normalized = self._FREQ_MAPPING.get(freq.lower(), freq)
        self.freq = pd.Timedelta(freq_normalized)
        self._validate()

    def _validate(self) -> None:
        if not self.lags:
            raise ValueError("At least one lag value must be provided")

        if not self.columns:
            raise ValueError("At least one column must be provided")

    def _get_timedelta(self, lag: int) -> pd.Timedelta:
        return self.freq * lag

    def _format_lag_name(self, column: str, lag: int) -> str:
        total_td = self.freq * abs(lag)
        total_seconds = int(total_td.total_seconds())

        if total_seconds % 86400 == 0:
            value = total_seconds // 86400
            unit = "d"
        elif total_seconds % 3600 == 0:
            value = total_seconds // 3600
            unit = "h"
        elif total_seconds % 60 == 0:
            value = total_seconds // 60
            unit = "min"
        else:
            value = total_seconds
            unit = "s"

        sign = "-" if lag >= 0 else "+"
        return f"{column}_{unit}{sign}{value}"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        lagged_data = {}
        for column in self.columns:
            series = df[column]
            for lag in self.lags:
                name = self._format_lag_name(column, lag)
                timedelta = self._get_timedelta(lag)
                shifted_index = df.index + timedelta
                shifted_series = pd.Series(series.values, index=shifted_index)
                lagged_data[name] = shifted_series.reindex(df.index)

        return pd.concat([df, pd.DataFrame(lagged_data, index=df.index)], axis=1)
