import operator
import warnings
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from ._compare import adjust_family, compare_pair, is_grouped, is_per_hour, panel_of, verdict, warn_if_short
from .base import EvaluatorResult, SetEvaluator


class DMEvaluator(SetEvaluator):
    """Each model against `benchmark` on the keys the two share: MAE ratio, its interval and a Diebold-Mariano test.

    Columns (`"DM {key}"`): `ratio` (MAE of the model / MAE of the benchmark, so below 1 is better), `ci_low`, `ci_high`,
    `p`, `p_holm` (Holm over the `m` models compared with the benchmark in that group, with exact clones counted once), `m`,
    `verdict`, `n_days`, `n_forecasts`. The test runs on the daily mean absolute error, with the daily losses of several horizons averaged per
    run date. In the per-hour tables (`by_hour`, `by_hour_horizon`) `p_holm` and `verdict` stay empty: a per-hour DM is a
    heatmap, never a verdict. `verdict` is also empty below 30 days, and when the test and the ratio point in opposite
    ways. The interval is empty when the days fit in one bootstrap block. `summary()` warns below 182 days.
    `report.tests()` gives the same comparison as a table of its own.

    Args:
        benchmark: Name of the model the others are compared with.
        variance: `"hac"` (Newey-West) or `"plain"` (epftoolbox's variance).
        lag: Newey-West lag; the default is `stats.default_lag` of the number of days (5 at 365).
        correction: `"holm"`, or `None` for no correction (`p_holm` empty, `verdict` from `p`).
        alpha: Two-sided size behind `verdict`.
        interval: Coverage of the bootstrap interval of the ratio; `None` skips it.
        block: Block length of that bootstrap, in days.
        reps: Bootstrap repetitions.
        seed: Bootstrap seed.
    """

    name = "DM"

    def __init__(
        self,
        benchmark: str,
        variance: str = "hac",
        lag: Optional[int] = None,
        correction: Optional[str] = "holm",
        alpha: float = 0.05,
        interval: Optional[float] = 0.95,
        block: int = 7,
        reps: int = 4000,
        seed: int = 0,
    ):
        check_test_settings(variance, lag, correction, seed, interval)
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.benchmark = benchmark
        self.variance = variance
        self.lag = lag
        self.correction = correction
        self.alpha = alpha
        self.interval = interval
        self.block = block
        self.reps = reps
        self.seed = seed

    def compute_group(self, model_dfs: Mapping[str, pd.DataFrame]) -> Dict[str, EvaluatorResult]:
        empty = {key: float("nan") for key in ("ratio", "ci_low", "ci_high", "p", "p_holm")}
        empty.update(m=0, verdict=None, n_days=0, n_forecasts=0)
        out: Dict[str, Dict[str, Any]] = {name: dict(empty) for name in model_dfs}
        panel = panel_of(model_dfs)
        rows = panel.rows()
        if self.benchmark not in model_dfs or not len(model_dfs[self.benchmark]):
            warnings.warn(
                f"DMEvaluator: the benchmark {self.benchmark!r} has no forecasts here (models: {list(model_dfs)}), so "
                "there is nothing to compare with.",
                stacklevel=2,
            )
            return out

        names = [name for name in model_dfs if name != self.benchmark]
        pairs = [
            compare_pair(
                panel,
                name,
                self.benchmark,
                rows,
                variance=self.variance,
                lag=self.lag,
                interval=self.interval,
                block=self.block,
                reps=self.reps,
                seed=self.seed,
            )
            for name in names
        ]
        per_hour = is_per_hour(model_dfs)
        p_values = [pair.p_value for pair in pairs]
        p_holm, m, _ = adjust_family(
            panel, rows, names, self.benchmark, p_values, None if per_hour else self.correction
        )
        for name, pair, adjusted in zip(names, pairs, p_holm):
            tested_p = float(adjusted) if self.correction == "holm" else pair.p_value
            out[name] = {
                "ratio": pair.ratio,
                "ci_low": pair.ci_low,
                "ci_high": pair.ci_high,
                "p": pair.p_value,
                "p_holm": float(adjusted),
                "m": m,
                "verdict": None if per_hour else verdict(pair, tested_p, self.alpha),
                "n_days": pair.n_days,
                "n_forecasts": pair.n_forecasts,
            }
        own = panel.common(rows, (self.benchmark,))
        out[self.benchmark].update(
            ratio=1.0, m=m, n_days=panel.daily(own, (self.benchmark,)).n_days, n_forecasts=len(own)
        )
        if pairs and not is_grouped(model_dfs):
            warn_if_short(min(pair.n_days for pair in pairs))
        return out


def check_test_settings(
    variance: str, lag: Optional[int], correction: Optional[str], seed: int, interval: Optional[float] = None
) -> None:
    """Refuse what the kernels would only refuse once a pair is tested (or never, on a small group)."""
    if variance not in ("hac", "plain"):
        raise ValueError(f"variance must be 'hac' or 'plain', got {variance!r}")
    if lag is not None:
        lag = operator.index(lag)
        if lag < 0:
            raise ValueError(f"lag must not be negative, got {lag}")
        if variance == "plain" and lag != 0:
            raise ValueError("lag applies only to variance='hac'")
    if correction not in ("holm", "none", None):
        raise ValueError(f"correction must be 'holm' or None, got {correction!r}")
    if interval is not None and not 0 < interval < 1:
        raise ValueError(f"interval must be in (0, 1) or None, got {interval}")
    operator.index(seed)
