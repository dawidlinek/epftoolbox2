import pandas as pd
from .base import Transformer


_METHOD_MAP = {
    "ffill": ("first", "ffill"),
    "bfill": ("last", "bfill"),
    "linear": ("mean", "linear"),
    "sum": ("sum", "ffill"),
    "first": ("first", "ffill"),
    "last": ("last", "bfill"),
}


class ResampleTransformer(Transformer):
    """Resample a DataFrame to a new frequency.

    The ``method`` parameter controls behaviour for **both** directions:

    * **Downsampling** (e.g. 15 min → 1 h) – selects the aggregation function.
    * **Upsampling** (e.g. 1 h → 15 min) – selects how NaN gaps are filled.

    +---------+----------------+---------------------+
    | method  | agg (↓)        | fill (↑)            |
    +---------+----------------+---------------------+
    | ffill   | first          | forward-fill        |
    | bfill   | last           | backward-fill       |
    | linear  | mean           | linear interpolation|
    | sum     | sum            | forward-fill        |
    | first   | first          | forward-fill        |
    | last    | last           | backward-fill       |
    +---------+----------------+---------------------+
    """

    def __init__(self, freq: str = "1h", method: str = "linear", columns: list[str] | str | None = None):
        self.freq = freq
        self.method = method
        self.columns = [columns] if isinstance(columns, str) else columns
        self._validate_method()

    def _validate_method(self) -> None:
        if self.method not in _METHOD_MAP:
            raise ValueError(f"Invalid method: '{self.method}'. Must be one of: {set(_METHOD_MAP)}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        if self.columns:
            missing_cols = set(self.columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

        agg_func, fill_method = _METHOD_MAP[self.method]

        result = df.resample(self.freq).agg(agg_func)

        cols = result.columns if self.columns is None else self.columns
        subset = result[cols]
        if fill_method == "linear":
            subset = subset.interpolate(method="linear")
        elif fill_method == "ffill":
            subset = subset.ffill()
        elif fill_method == "bfill":
            subset = subset.bfill()
        result[cols] = subset

        return result.round(3)
