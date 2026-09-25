from typing import Callable, Optional, Union

import pandas as pd

from .base import Transformer

ColumnExpr = Union[str, Callable[[pd.DataFrame], pd.Series]]


class LambdaTransformer(Transformer):
    """Custom columns or a custom transform, for one-off feature arithmetic that doesn't need its own class.

    Exactly one of `func` and `columns` must be given.

    Args:
        func: A callable `df -> df`, for full control over the output frame.
        columns: A dict `{new_or_existing_column: expression}`. Each expression is either a callable
            `df -> Series` or a string evaluated with `df.eval(...)`. A string expression is plain data (it
            can be saved to YAML); a callable, in `func` or as a `columns` value, cannot and raises the same
            `TypeError` as any other non-YAML-safe parameter in this library. Entries run in the order given,
            each against the frame as built so far, so a later expression may refer to a column an earlier
            one just added.

    Column names that `df.eval` can't parse as a plain identifier (spaces, operators, a name starting with a
    digit, ...) need backticks, exactly as in `pandas.DataFrame.eval`: ``"`price (EUR)` - load"``.

    The result must have exactly the same index as the input: the same rows, in the same order. `columns`
    mode can't break this (it only ever assigns aligned columns onto a copy of the input), but a `func` that
    resamples, filters or reorders rows raises `ValueError` instead of silently changing what runs after it.

    Nothing here understands look-ahead. Same-row arithmetic (`"load_forecast - wind_forecast"`) only
    combines values already on that row, so it is safe. `df[c].shift(-1)`, `.bfill()`, a rolling window that
    reaches past the current row, or normalising with a whole-sample statistic (mean/std/min/max of the
    entire column, computed once over the loaded range) all pull in values a forecast issued on that row's
    day would not yet have. Writing a leak-free expression is your responsibility; see the docs page.

    Example:
        >>> df = LambdaTransformer(
        ...     columns={"residual_load": "load_forecast - wind_forecast - solar_forecast"}
        ... ).transform(df)
        >>> df = LambdaTransformer(func=lambda df: df.assign(spread=df["price"] - df["price_d-1"])).transform(df)
    """

    def __init__(
        self,
        func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        columns: Optional[dict] = None,
    ):
        if (func is None) == (columns is None):
            raise ValueError("LambdaTransformer needs exactly one of `func` or `columns`, not both or neither")
        if func is not None and not callable(func):
            raise TypeError(f"func must be callable, got {type(func).__name__}")
        if columns is not None:
            if not isinstance(columns, dict):
                raise TypeError(f"columns must be a dict {{column: expression}}, got {type(columns).__name__}")
            if not columns:
                raise ValueError("columns must not be empty")
            bad = {k: v for k, v in columns.items() if not (isinstance(v, str) or callable(v))}
            if bad:
                raise TypeError(f"columns values must be a string or a callable df -> Series, got {bad}")

        self.func = func
        self.columns = columns

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.func is not None:
            result = self.func(df)
        else:
            result = df.copy()
            for name, expr in self.columns.items():
                result[name] = expr(result) if callable(expr) else result.eval(expr)

        if not isinstance(result, pd.DataFrame) or not result.index.equals(df.index):
            raise ValueError(
                "LambdaTransformer changed the index: the result must have exactly the same rows, in the same "
                "order, as the input. A transform that resamples, drops or reorders rows belongs in a dedicated "
                "transformer, not here."
            )
        return result
