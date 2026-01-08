import pandas as pd
from .base import Transformer


class ResampleTransformer(Transformer):
    def __init__(self, freq: str = "1h", method: str = "linear", columns: list[str] | str | None = None):
        self.freq = freq
        self.method = method
        self.columns = [columns] if isinstance(columns, str) else columns
        self._validate_method()

    def _validate_method(self) -> None:
        valid_methods = {"linear", "ffill", "bfill"}
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method: '{self.method}'. Must be one of: {valid_methods}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        if self.columns:
            missing_cols = set(self.columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

        result = df.resample(self.freq).asfreq()

        cols_to_interpolate = result.columns if self.columns is None else self.columns
        subset = result[cols_to_interpolate]
        if self.method == "linear":
            subset = subset.interpolate(method="linear")
        elif self.method == "ffill":
            subset = subset.ffill()
        elif self.method == "bfill":
            subset = subset.bfill()

        result[cols_to_interpolate] = subset

        result = result.round(3)

        return result
