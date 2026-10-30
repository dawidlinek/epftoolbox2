"""Put the forecasts of several models on one grid of forecast keys, and compare them on the keys they share.

The kernels in `stats.py` pair their inputs by position. Everything here exists so that what reaches them is aligned:
a comparison only ever sees the forecasts all its models hold (same key, finite prediction and actual), and the loss
of a day is the mean absolute error over the periods of that day that those forecasts cover.

Shared by the set-level evaluators and by the comparison methods of `EvaluationReport`, so that a column of `summary()`
and the matching table give the same number.
"""

import functools
import math
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import stats
from .base import FORECAST_KEYS

# Two models that disagree on the actual price of a forecast by more than this (relative) were not given the same task.
_ACTUAL_TOLERANCE = 1e-9
# How many sets of bootstrap indices stay in memory. One set is `reps x n` integers (11.7 MB at 4000 x 365).
_INDEX_SETS = 4
# Periods of a whole local day; a day with any other number of shared periods is a partial one.
_WHOLE_DAY_PERIODS = (23, 24, 25, 92, 96, 100)
# Below this many days nothing here gives a verdict, whatever its p-value: the normal approximation of the DM test
# rejects a true null 9 % of the time at 30 days, 21 % at 7 and 35 % at 3.
MIN_VERDICT_DAYS = 30
# Test lengths behind the messages of the comparison tables: below the first a comparison says very little, and below
# the second it lacks a whole year of seasons.
SHORT_TEST_DAYS = 182
FULL_TEST_DAYS = 365


class ModelFrames(dict):
    """`{model: frame}` of one group, as the report hands it to `Evaluator.compute_group`.

    A plain dict to whoever reads it. It remembers which grouping it was cut by, and builds the loss panel at most once,
    however many evaluators of the group ask for it.
    """

    def __init__(self, frames: Mapping[str, pd.DataFrame], group_keys: Sequence[str] = ()):
        super().__init__(frames)
        self.group_keys = tuple(group_keys)
        self._panel: Optional["LossPanel"] = None

    @property
    def panel(self) -> "LossPanel":
        if self._panel is None:
            self._panel = LossPanel(self)
        return self._panel


def panel_of(model_dfs: Mapping[str, pd.DataFrame]) -> "LossPanel":
    return model_dfs.panel if isinstance(model_dfs, ModelFrames) else LossPanel(model_dfs)


def is_per_hour(model_dfs: Mapping[str, pd.DataFrame]) -> bool:
    """Whether this group is one period of the day: its comparisons are a heatmap, never a verdict."""
    return "hour" in getattr(model_dfs, "group_keys", ())


def is_grouped(model_dfs: Mapping[str, pd.DataFrame]) -> bool:
    """Whether this is one group of a breakdown, which is short by nature, rather than the whole test."""
    return bool(getattr(model_dfs, "group_keys", ()))


class LossPanel:
    """The absolute errors of every model on one grid of forecast keys.

    `errors[i, j]` is |prediction - actual| of model j on key i, NaN when the model has no forecast there or its
    prediction or actual is not finite. Keys are sorted by (run_date, target_date, horizon, hour), so days are in
    calendar order and every sum is the same however the records were written.

    Raises ValueError when a frame lacks a key column, has a missing key, or holds a key twice (that forecast would count
    twice). A frame without rows is a model without forecasts.

    `whole_day_periods` are the numbers of periods a whole day may have; a day of any other number of shared periods is
    counted as partial. A panel of daily aggregates has one period a day, so it passes `(1,)`.
    """

    def __init__(self, model_dfs: Mapping[str, pd.DataFrame], whole_day_periods: Sequence[int] = _WHOLE_DAY_PERIODS):
        self.models: List[str] = list(model_dfs)
        self.whole_day_periods = tuple(whole_day_periods)
        self._column = {name: j for j, name in enumerate(self.models)}
        frames = {name: df for name, df in model_dfs.items() if len(df)}
        for name, df in frames.items():
            missing = [k for k in FORECAST_KEYS if k not in df.columns]
            if missing:
                raise ValueError(
                    f"Comparing models needs the forecast keys {list(FORECAST_KEYS)}, but the frame of {name!r} lacks "
                    f"{missing}. Result records need run_date, target_date, hour and horizon."
                )

        if not frames:
            self.errors = np.empty((0, len(self.models)))
            self.hour = self.horizon = self._day = self._run_of_day = np.empty(0, dtype=np.int64)
            return

        coded, radix = [], []
        for column in ("run_date", "target_date", "horizon", "hour"):
            values = np.concatenate([df[column].to_numpy() for df in frames.values()])
            codes, uniques = pd.factorize(values, sort=True)
            if (codes < 0).any():
                raise ValueError(f"Comparing models: a frame has rows with a missing forecast key ({column}).")
            coded.append(codes.astype(np.int64))
            radix.append(uniques)
        sizes = [len(u) for u in radix]
        if math.prod(sizes) >= 2**63:
            raise ValueError(f"Comparing models: too many distinct forecast keys ({sizes}).")
        codes = coded[0]
        for part, size in zip(coded[1:], sizes[1:]):
            codes = codes * size + part
        keys, position = np.unique(codes, return_inverse=True)
        position = position.ravel()

        self.errors = np.full((len(keys), len(self.models)), np.nan)
        reference = np.full(len(keys), np.nan)
        disagreeing = np.zeros(len(keys), dtype=bool)
        start = 0
        for name, df in frames.items():
            rows = position[start : start + len(df)]
            start += len(df)
            if np.bincount(rows, minlength=len(keys)).max() > 1:
                raise ValueError(
                    f"Comparing models needs each forecast key {list(FORECAST_KEYS)} once per model, but {name!r} has "
                    "duplicated keys (that forecast would count twice)."
                )
            prediction = np.asarray(df["prediction"], dtype=float)
            actual = np.asarray(df["actual"], dtype=float)
            finite = np.isfinite(prediction) & np.isfinite(actual)
            self.errors[rows[finite], self._column[name]] = np.abs(prediction[finite] - actual[finite])

            seen = reference[rows[finite]]
            disagreeing[rows[finite]] |= np.abs(seen - actual[finite]) > _ACTUAL_TOLERANCE * np.maximum(
                1.0, np.abs(actual[finite])
            )
            reference[rows[finite]] = np.where(np.isnan(seen), actual[finite], seen)
        if disagreeing.any():
            warnings.warn(
                f"{int(disagreeing.sum())} forecasts have different actual prices across the models "
                f"({', '.join(frames)}): these models were not scored against the same data, so comparing them is "
                "not meaningful.",
                stacklevel=2,
            )

        run_code, _, horizon_code, hour_code = _digits(keys, sizes)
        self.horizon = np.asarray(radix[2], dtype=np.int64)[horizon_code]
        self.hour = np.asarray(radix[3], dtype=np.int64)[hour_code]
        # A day of a model's forecasts is (run_date, horizon); codes in calendar order, then by horizon.
        day_key = run_code * sizes[2] + horizon_code
        day_keys, self._day = np.unique(day_key, return_inverse=True)
        self._day = self._day.ravel()
        self._run_of_day = day_keys // sizes[2]

    def __contains__(self, model: str) -> bool:
        return model in self._column

    def column(self, model: str) -> int:
        return self._column[model]

    @property
    def horizons(self) -> List[int]:
        return [int(h) for h in np.unique(self.horizon)]

    def rows(self, horizon: Optional[int] = None) -> np.ndarray:
        """Positions of the keys of one horizon, or of all of them."""
        if horizon is None:
            return np.arange(len(self.horizon))
        return np.flatnonzero(self.horizon == horizon)

    def common(self, rows: np.ndarray, models: Iterable[str]) -> np.ndarray:
        """The positions in `rows` where every one of `models` has a finite forecast."""
        columns = [self._column[m] for m in models]
        return rows[~np.isnan(self.errors[np.ix_(rows, columns)]).any(axis=1)]

    def daily(self, rows: np.ndarray, models: Sequence[str]) -> "DailyLosses":
        """Daily losses of `models` on exactly the keys in `rows`, which all of them must hold.

        The loss of (run_date, horizon) is the mean absolute error over its periods, whatever their number (23, 24, 25,
        92, 96, 100, or fewer when forecasts are missing). When `rows` span several horizons a run_date's loss is the mean
        of its horizons' losses. `sums` holds the error sums behind them, one row per run_date: their ratio is the ratio
        of forecast-weighted MAEs.
        """
        day = self._day[rows]
        count = np.bincount(day, minlength=len(self._run_of_day))
        present = np.flatnonzero(count)
        run_ids, run_position = np.unique(self._run_of_day[present], return_inverse=True)
        horizons_per_run = np.bincount(run_position)
        losses = np.empty((len(run_ids), len(models)))
        sums = np.empty((len(run_ids), len(models)))
        for i, model in enumerate(models):
            day_sum = np.bincount(day, weights=self.errors[rows, self._column[model]], minlength=len(count))[present]
            losses[:, i] = np.bincount(run_position, weights=day_sum / count[present]) / horizons_per_run
            sums[:, i] = np.bincount(run_position, weights=day_sum)
        partial = int(np.isin(count[present], self.whole_day_periods, invert=True).sum())
        return DailyLosses(losses=losses, sums=sums, n_forecasts=len(rows), partial_days=partial)


@dataclass(frozen=True)
class DailyLosses:
    """`losses` and `sums` are days x models. `n_forecasts` is the number of keys behind them, for each model, and
    `partial_days` how many (run_date, horizon) among them hold only part of a day."""

    losses: np.ndarray
    sums: np.ndarray
    n_forecasts: int
    partial_days: int = 0

    @property
    def n_days(self) -> int:
        return self.losses.shape[0]

    def mae(self) -> np.ndarray:
        """Forecast-weighted MAE of every model, NaN without forecasts."""
        if self.n_forecasts == 0:
            return np.full(self.losses.shape[1], np.nan)
        return self.sums.sum(axis=0) / self.n_forecasts


def _digits(codes: np.ndarray, sizes: Sequence[int]) -> Tuple[np.ndarray, ...]:
    """Undo the mixed-radix coding of `codes`, most significant digit first."""
    digits = []
    for size in reversed(sizes):
        digits.append(codes % size)
        codes = codes // size
    return tuple(reversed(digits))


@functools.lru_cache(maxsize=_INDEX_SETS)
def _shared_indices(n: int, block: float, reps: int, seed: int, kind: str) -> np.ndarray:
    indices = stats.block_bootstrap_indices(n, block, reps, seed, kind=kind)
    indices.setflags(write=False)
    return indices


def bootstrap_indices(n: int, block: float, reps: int, seed: int, kind: str = "circular") -> np.ndarray:
    """Bootstrap indices that depend only on these arguments, so a pair's interval does not depend on the other models
    of the table, and all pairs with the same number of days share one draw (drawing is most of the cost of a group)."""
    return _shared_indices(int(n), block, int(reps), int(seed), kind)


@dataclass(frozen=True)
class PairComparison:
    """A model against a benchmark on the keys they share. NaN where the sample is too small to tell."""

    n_days: int
    n_forecasts: int
    mae: float
    mae_benchmark: float
    ratio: float
    ci_low: float
    ci_high: float
    statistic: float
    p_value: float
    mde: float
    partial_days: int = 0


def compare_pair(
    panel: LossPanel,
    model: str,
    benchmark: str,
    rows: np.ndarray,
    variance: str = "hac",
    lag: Optional[int] = None,
    interval: Optional[float] = 0.95,
    block: float = 7,
    reps: int = 4000,
    seed: int = 0,
) -> PairComparison:
    """Compare `model` with `benchmark` on the keys of `rows` that both hold, and on nothing else.

    `mae` and `mae_benchmark` are forecast-weighted and `ratio = mae / mae_benchmark`. The interval resamples daily error
    sums, so it agrees with that ratio on 23 and 25-hour days. The DM test takes daily mean losses. With 24 periods on
    every day the two are the same thing up to a factor. With days of very different lengths they can disagree: see
    `verdict`. There is no interval when the days fit in one block: every resample would hold all of them.
    """
    shared = panel.common(rows, (model, benchmark))
    nan = float("nan")
    if len(shared) == 0:
        return PairComparison(0, 0, nan, nan, nan, nan, nan, nan, nan, nan)

    daily = panel.daily(shared, (model, benchmark))
    mae, mae_benchmark = (float(v) for v in daily.mae())
    if mae_benchmark > 0:
        ratio = mae / mae_benchmark
    else:
        ratio = math.inf if mae > 0 else nan
    n_days = daily.n_days
    loss_model, loss_benchmark = daily.losses[:, 0], daily.losses[:, 1]
    statistic = p_value = ci_low = ci_high = mde = nan

    if n_days >= 2 and (lag is None or lag < n_days):
        result = stats.dm_test(loss_model, loss_benchmark, variance=variance, lag=lag)
        statistic, p_value = result.statistic, result.p_value
        if mae_benchmark > 0:
            mde = stats.minimum_detectable_effect(loss_model, loss_benchmark, lag=lag)
    if n_days > block and interval is not None and mae_benchmark > 0:
        with np.errstate(divide="ignore", invalid="ignore"):  # a sample of days on which the benchmark is exact
            found = stats.ratio_interval(
                daily.sums[:, 0],
                daily.sums[:, 1],
                level=interval,
                indices=bootstrap_indices(n_days, block, reps, seed),
            )
        ci_low, ci_high = found.low, found.high
    return PairComparison(
        n_days=n_days,
        n_forecasts=daily.n_forecasts,
        mae=mae,
        mae_benchmark=mae_benchmark,
        ratio=ratio,
        ci_low=ci_low,
        ci_high=ci_high,
        statistic=statistic,
        p_value=p_value,
        mde=mde,
        partial_days=daily.partial_days,
    )


def adjust(p_values: Sequence[float], correction: Optional[str]) -> np.ndarray:
    """Holm-adjusted p-values (NaN entries were not tested and do not count), or all NaN without a correction."""
    p = np.asarray(p_values, dtype=float)
    return stats.holm(p) if correction == "holm" else np.full(p.shape, np.nan)


def verdict(pair: PairComparison, p_value: float, alpha: float) -> Optional[str]:
    """`better`, `worse` or `no difference` for a model against its benchmark, `p_value` being the one to read at `alpha`.

    None when nothing was tested, on fewer than `MIN_VERDICT_DAYS` days, and when the test and the ratio point in opposite
    directions. The test weighs every day the same and the ratio every forecast, so that takes days of very different
    lengths, and a row must not say `better` next to an interval above 1.
    """
    if math.isnan(p_value) or pair.n_days < MIN_VERDICT_DAYS:
        return None
    if p_value >= alpha:
        return "no difference"
    better = pair.statistic > 0
    if better != (pair.ratio < 1):
        return None
    return "better" if better else "worse"


def short_test_message(n_days: int) -> Optional[str]:
    """What to say about a comparison of `n_days` days; None for at least a year."""
    if n_days >= FULL_TEST_DAYS:
        return None
    return (
        f"The comparison rests on {n_days} days, less than a year: it does not see every season, and even one test year "
        "only resolves differences of about 5 % of the MAE."
    )


def warn_if_short(n_days: int, stacklevel: int = 2) -> None:
    """Warn on a very short comparison, for a table that has no `attrs` to leave a note in."""
    if n_days < SHORT_TEST_DAYS:
        warnings.warn(
            f"{short_test_message(n_days)} Below {SHORT_TEST_DAYS} days read it as indicative only.",
            stacklevel=stacklevel + 1,
        )


@dataclass(frozen=True)
class SetComparison:
    """Models on the keys all of them hold. `limited_by`: how many forecasts each model lacks against the others."""

    models: List[str]
    daily: DailyLosses
    limited_by: Dict[str, int]


def compare_set(panel: LossPanel, models: Sequence[str], rows: np.ndarray) -> SetComparison:
    """The daily losses of `models` on the keys of `rows` that all of them hold (the model set's common keys)."""
    columns = [panel.column(m) for m in models]
    held = ~np.isnan(panel.errors[np.ix_(rows, columns)])
    union = held.any(axis=1).sum()
    short = {m: int(union - held[:, j].sum()) for j, m in enumerate(models)}
    limited_by = dict(sorted(((m, k) for m, k in short.items() if k > 0), key=lambda item: (-item[1], item[0])))
    return SetComparison(list(models), panel.daily(rows[held.all(axis=1)], models), limited_by)


def confidence_set(
    compared: SetComparison, alpha: float, statistic: str, block: float, reps: int, seed: int
) -> stats.MCSResult:
    """The model confidence set of `compared`. Raises ValueError on a degenerate set, naming the models."""
    indices = bootstrap_indices(compared.daily.n_days, block, reps, seed, kind="stationary")
    try:
        return stats.model_confidence_set(compared.daily.losses, alpha=alpha, statistic=statistic, indices=indices)
    except ValueError as exc:
        raise ValueError(f"{exc} (models by column: {', '.join(compared.models)})") from exc
