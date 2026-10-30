import operator
import warnings
from typing import Dict, Mapping

import pandas as pd

from ._compare import compare_set, confidence_set, panel_of
from .base import EvaluatorResult, SetEvaluator

_MIN_MODELS = 3


class MCSEvaluator(SetEvaluator):
    """Model confidence set (Hansen, Lunde & Nason 2011) of all the models of a group, on their common keys.

    Columns: `MCS p` (the MCS p-value), `in MCS` (whether the p-value is above `alpha`), `MCS n_days`, `MCS n_forecasts`.
    The loss is the daily mean absolute error, with the daily losses of several horizons averaged per run date. Here the
    model set is the comparison, so one model with few forecasts shrinks the sample of all: check `n_days`, or use
    `report.mcs()`, which names the models that limited the keys and takes a subset. Needs at least three models; with
    fewer the columns are NaN, with a warning.

    Args:
        alpha: Size of the set (0.10 gives the 90 % set).
        statistic: `"R"` (T_R, recommended: seed-stable) or `"max"` (T_max).
        block: Mean block length of the stationary bootstrap, in days.
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
        if len(names) < _MIN_MODELS:
            warnings.warn(
                f"The model confidence set needs at least {_MIN_MODELS} models, got {len(names)}: the MCS columns are "
                "NaN here.",
                stacklevel=2,
            )
            return out
        panel = panel_of(model_dfs)
        compared = compare_set(panel, names, panel.rows())
        counts = {"n_days": compared.daily.n_days, "n_forecasts": compared.daily.n_forecasts}
        for name in names:
            out[name].update(counts)
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
        included = result.included()
        for name, p, inside in zip(names, result.p_values, included):
            out[name].update({"p": float(p), "in": bool(inside)})
        return out


def check_mcs_settings(alpha: float, statistic: str, seed: int) -> None:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if statistic not in ("R", "max"):
        raise ValueError(f"statistic must be 'R' or 'max', got {statistic!r}")
    operator.index(seed)
