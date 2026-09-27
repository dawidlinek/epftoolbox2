import ast
import inspect
import linecache
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from .base import Transformer

ColumnExpr = Union[str, Callable[[pd.DataFrame], pd.Series]]


class LambdaTransformer(Transformer):
    """Custom columns or a custom transform, for one-off feature arithmetic that doesn't need its own class.

    Exactly one of `func` and `columns` must be given.

    Args:
        func: A `df -> df` lambda for full control over the output frame, either a real lambda or its source
            as a string (``"lambda df: df.assign(spread=df['price'] - df['price_d-1'])"``).
        columns: A dict `{new_or_existing_column: expression}`. Each expression is a `df -> Series` lambda
            (real or as a ``"lambda df: ..."`` string), or any other string, which is evaluated with
            `df.eval(...)`. Entries run in the order given, each against the frame as built so far, so a later
            expression may refer to a column an earlier one just added.

    Lambdas are saved to YAML as their source text. A real lambda is converted when its source can be
    recovered unambiguously and it only uses its argument, `pd`, `np` and builtins; one that closes over
    other variables (``lambda df: df.x * k``) can't be saved and raises `TypeError` on `save()`: write the
    value into the lambda, or pass the lambda as a string. Named functions (`def`) are never saved. Text
    lambdas are evaluated with `pd`, `np` and the builtins in scope, so loading a pipeline YAML runs the code
    in it: only load pipeline files you trust.

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
        self._func = None
        self._exprs = None
        if func is not None:
            if isinstance(func, str):
                self._func = _compile_lambda(func, "func")
            elif callable(func):
                self._func = func
                func = _lambda_source(func) or func
            else:
                raise TypeError(f"func must be a lambda or its source text, got {type(func).__name__}")
        if columns is not None:
            if not isinstance(columns, dict):
                raise TypeError(f"columns must be a dict {{column: expression}}, got {type(columns).__name__}")
            if not columns:
                raise ValueError("columns must not be empty")
            bad = {k: v for k, v in columns.items() if not (isinstance(v, str) or callable(v))}
            if bad:
                raise TypeError(f"columns values must be a string or a callable df -> Series, got {bad}")
            self._exprs = {}
            saved = {}
            for name, expr in columns.items():
                if callable(expr):
                    self._exprs[name] = expr
                    saved[name] = _lambda_source(expr) or expr
                elif _is_lambda_text(expr):
                    self._exprs[name] = _compile_lambda(expr, f"columns[{name!r}]")
                    saved[name] = expr
                else:
                    self._exprs[name] = expr
                    saved[name] = expr
            columns = saved

        self.func = func
        self.columns = columns

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._func is not None:
            result = self._func(df)
        else:
            result = df.copy()
            for name, expr in self._exprs.items():
                result[name] = expr(result) if callable(expr) else result.eval(expr)

        if not isinstance(result, pd.DataFrame) or not result.index.equals(df.index):
            raise ValueError(
                "LambdaTransformer changed the index: the result must have exactly the same rows, in the same "
                "order, as the input. A transform that resamples, drops or reorders rows belongs in a dedicated "
                "transformer, not here."
            )
        return result


_NAMESPACE = {"pd": pd, "np": np}


def _is_lambda_text(text: str) -> bool:
    return text.lstrip().startswith("lambda")


def _compile_lambda(text: str, where: str) -> Callable:
    """Evaluate the source of a lambda with `pd`, `np` and the builtins in scope."""
    if not _is_lambda_text(text):
        raise ValueError(f'{where} must be a lambda, e.g. \'lambda df: df.assign(x=df["a"] - df["b"])\'')
    try:
        fn = eval(text, dict(_NAMESPACE))
    except SyntaxError as e:
        raise ValueError(f"{where} is not valid Python: {e.msg} in {text!r}") from None
    if not callable(fn):
        raise ValueError(f"{where} must evaluate to a callable, got {type(fn).__name__}")
    return fn


def _lambda_source(fn: Callable) -> Optional[str]:
    """Source text of a self-contained lambda, or None if it can't be recovered unambiguously.

    The lambda is found by parsing the file (or notebook cell) it was defined in and keeping the only lambda
    node on its first line whose compiled bytecode matches `fn`. It must not close over local variables and
    may only use `pd`, `np` and builtins as globals, so that the text means the same thing when reloaded.
    """
    code = getattr(fn, "__code__", None)
    if code is None or getattr(fn, "__name__", None) != "<lambda>":
        return None
    try:
        free = inspect.getclosurevars(fn)
    except (TypeError, ValueError):
        return None
    if free.nonlocals or set(free.globals) - set(_NAMESPACE):
        return None
    source = "".join(linecache.getlines(code.co_filename))
    if not source:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda) and node.lineno == code.co_firstlineno:
            text = ast.get_source_segment(source, node)
            if text is None:
                continue
            try:
                candidate = eval(compile(text, "<lambda>", "eval"), dict(_NAMESPACE))
            except Exception:
                continue
            if candidate.__code__.co_code == code.co_code and candidate.__code__.co_names == code.co_names:
                matches.append(text)
    return matches[0] if len(set(matches)) == 1 else None
