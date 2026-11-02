import operator
import warnings
from typing import Dict, Mapping

import pandas as pd

from ._compare import (
    MIN_SET_MODELS,
    MIN_VERDICT_DAYS,
    compare_set,
    confidence_set,
    is_grouped,
    is_per_hour,
    panel_of,
    too_few_distinct_models,
    warn_if_short,
)
from .base import EvaluatorResult, SetEvaluator


class MCSEvaluator(SetEvaluator):
    """Model confidence set (Hansen, Lunde & Nason 2011) of all the models of a group, on their common keys.

    Columns: `MCS p` (the MCS p-value), `in MCS` (whether the p-value is above `alpha`), `MCS n_days`, `MCS n_forecasts`.
    The loss is the daily mean absolute error, with the daily losses of several horizons averaged per run date. Here the
    model set is the comparison, so one model with few forecasts shrinks the sample of all: check `n_days`, or use
    `report.mcs()`, which names the models that limited the keys and takes a subset. Needs at least three models; with
    fewer the columns are NaN, with a warning. Exact clones are one model to the set, and each gets its result, and they count
    once in those three. `in MCS` is a verdict, so it is empty per hour (the p-values are a heatmap
    there) and below 30 days. `summary()` warns below a span of 182 calendar days.

    Args:
        alpha: Size of the set (0.10 gives the 90 % set).
        statistic: `"R"` (T_R, recommended: seed-stable) or `"max"` (T_max).
        block: Mean block length of the stationary bootstrap, in issue days (observations: seven Fridays are seven weeks).
        reps: Bootstrap repetitions (at least 1,000 recommended).
        seed: Bootstrap seed.
    """

    name = "MCS"

    def __init__(self, alpha: float = 0.10, statistic: str = "R", block: int = 7, reps: int = 1000, seed: int = 0):
        check_mcs_settings(alpha, statistic, seed)
        self.alpha = alpha
        self.statistic = statistic
        self.block = block
        self.reps = reps
        self.seed = seed

    def column_name(self, key: str) -> str:
        return "in MCS" if key == "in" else super().column_name(key)

    def compute_group(self, model_dfs: Mapping[str, pd.DataFrame]) -> Dict[str, EvaluatorResult]:
        names = list(model_dfs)
        out: Dict[str, EvaluatorResult] = {
            name: {"p": float("nan"), "in": float("nan"), "n_days": 0, "n_forecasts": 0} for name in names
        }
        if len(names) < MIN_SET_MODELS:
            warnings.warn(
                f"The model confidence set needs at least {MIN_SET_MODELS} models, got {len(names)}: the MCS columns are "
                "NaN here.",
                stacklevel=2,
            )
            return out
        panel = panel_of(model_dfs)
        compared = compare_set(panel, names, panel.rows())
        counts = {"n_days": compared.daily.n_days, "n_forecasts": compared.daily.n_forecasts}
        for name in names:
            out[name].update(counts)
        too_few = too_few_distinct_models(compared)
        if too_few is not None:
            warnings.warn(too_few, stacklevel=2)
            return out
        if compared.daily.n_days < 2:
            warnings.warn(
                f"The model confidence set needs at least 2 common days, got {compared.daily.n_days} (models short of "
                f"forecasts: {compared.limited_by}): the MCS columns are NaN here.",
                stacklevel=2,
            )
            return out
        try:
            result = confidence_set(compared, self.alpha, self.statistic, self.block, self.reps, self.seed)
        except ValueError as exc:
            warnings.warn(f"No model confidence set: {exc}. The MCS columns are NaN here.", stacklevel=2)
            return out
        judge = compared.daily.n_days >= MIN_VERDICT_DAYS and not is_per_hour(model_dfs)
        for name, p, inside in zip(names, result.p_values, result.included()):
            out[name].update({"p": float(p), "in": bool(inside) if judge else float("nan")})
        if not is_grouped(model_dfs):
            warn_if_short(compared.daily.n_days, span_days=compared.daily.span_days)
        return out

    def clones(self, model_dfs: Mapping[str, pd.DataFrame]) -> list:
        panel = panel_of(model_dfs)
        return [list(group) for group in compare_set(panel, list(model_dfs), panel.rows()).clones]


def check_mcs_settings(alpha: float, statistic: str, seed: int) -> None:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if statistic not in ("R", "max"):
        raise ValueError(f"statistic must be 'R' or 'max', got {statistic!r}")
    operator.index(seed)
