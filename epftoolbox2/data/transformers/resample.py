import pandas as pd
from .base import Transformer


class ResampleTransformer(Transformer):
    def __init__(self, freq: str = "1h", method: str = "linear"):
        self.freq = freq
        self.method = method
        self._validate_method()

    def _validate_method(self) -> None:
        valid_methods = {"linear", "ffill", "bfill"}
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method: '{self.method}'. Must be one of: {valid_methods}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        result = df.resample(self.freq).asfreq()

        if self.method == "linear":
            result = result.interpolate(method="linear")
        elif self.method == "ffill":
            result = result.ffill()
        elif self.method == "bfill":
            result = result.bfill()

        result = result.round(3)

        return result
