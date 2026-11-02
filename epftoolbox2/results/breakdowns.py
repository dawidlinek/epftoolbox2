"""The parts of the breakdown tables and of `EvaluationReport.aggregate()` that do not need the report itself.

Price levels are read from the actual price alone. Daily aggregates are made only of whole days, and a whole day is
judged by `evaluators._days`, the same rule that decides which days `EconomicEvaluator` can schedule.
"""

from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ..evaluators._days import whole_days
from ..evaluators.base import FORECAST_KEYS

AGGREGATES = ("max", "min", "mean", "spread")

# A spike is a price at or above the lower of these two: evalproto's definition (its Q4), 88 hours a year per market.
SPIKE_CAP = 300.0
SPIKE_QUANTILE = 0.99

_REDUCE: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "max": lambda a: a.max(axis=1),
    "min": lambda a: a.min(axis=1),
    "mean": lambda a: a.mean(axis=1),
    "spread": lambda a: a.max(axis=1) - a.min(axis=1),
}


def spike_threshold(model_dfs: Mapping[str, pd.DataFrame], spike: Optional[float] = None) -> Tuple[float, str, float]:
    """`(threshold, rule, quantile)`: the price from which a period is a spike, how it was set, and the 99th percentile.

    With `spike=None` the threshold is `min(300, 99th percentile)` of the actual prices of the periods any model forecast,
    each period (target date, hour) once however many horizons forecast it. It comes from the actuals of all the models
    together, so it is the same for every model, and never from a prediction. `quantile` is NaN when `spike` is given.
    The threshold is NaN when no model has a finite actual price.
    """
    if spike is not None:
        value = float(spike)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"spike must be a positive price, got {spike!r}")
        return value, f"given: {value:g}", float("nan")

    pools = []
    for df in model_dfs.values():
        if len(df) and "actual" in df:
            pools.append(df[[c for c in ("target_date", "hour", "actual") if c in df]])
    periods = pd.concat(pools) if pools else pd.DataFrame({"actual": []})
    periods = periods[np.isfinite(periods["actual"].to_numpy(dtype=float))]
    if {"target_date", "hour"} <= set(periods.columns):
        periods = periods.drop_duplicates(["target_date", "hour"])
    actual = periods["actual"].to_numpy(dtype=float)
    if not len(actual):
        return float("nan"), "no finite actual price", float("nan")

    quantile = float(np.quantile(actual, SPIKE_QUANTILE))
    threshold = min(SPIKE_CAP, quantile)
    if threshold <= 0:
        raise ValueError(
            f"The 99th percentile of the actual prices is {quantile:g}, so no price level above 0 can be a spike: "
            "pass spike=... with the price from which a period is one."
        )
    rule = f"min({SPIKE_CAP:g}, 99th percentile of the actual prices = {quantile:g}) = {threshold:g}"
    return threshold, rule, quantile


def price_level(actual: pd.Series, threshold: float) -> pd.Series:
    """`negative` (actual below 0), `spike` (at or above `threshold`) or `normal`, from the actual alone.

    A period belongs to exactly one level; a negative price is never a spike. A period without a finite actual price has
    no level (NaN), so it falls in no group.
    """
    values = actual.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        level = np.where(values < 0, "negative", np.where(values >= threshold, "spike", "normal")).astype(object)
    level[~np.isfinite(values)] = np.nan
    return pd.Series(level, index=actual.index)


def _days_on_the_clock(
    day: np.ndarray, hour: np.ndarray, dates: np.ndarray, clock: Mapping[pd.Timestamp, Tuple[int, ...]]
) -> List[Tuple[int, np.ndarray]]:
    """`whole_days` by the clock: the day codes whose period ids are those its target date has, by number of periods. `dates` has
    the target date of each day code."""
    length = np.bincount(day) if len(day) else np.empty(0, dtype=np.int64)
    stamps = pd.DatetimeIndex(dates)
    found = []
    for periods in np.unique(length).tolist():
        days = np.flatnonzero(length == periods)
        ids = hour[np.isin(day, days)].reshape(-1, periods)
        whole = np.array([clock.get(stamps[d], ()) == tuple(row) for d, row in zip(days.tolist(), ids.tolist())])
        if whole.any():
            found.append((periods, days[whole]))
    return found


def daily_aggregate(
    frame: pd.DataFrame,
    agg: str,
    per_hour: int,
    periods_in_target: Optional[pd.Series] = None,
    clock: Optional[Mapping[pd.Timestamp, Tuple[int, ...]]] = None,
) -> Tuple[pd.DataFrame, int, int]:
    """One model's forecasts as one forecast per (run_date, target_date, horizon): `(days, n_seen, n_dropped)`.

    The prediction and the actual of a day are each reduced by `agg` over the day's periods. Only whole days count, as
    `evaluators._days.whole_days` judges them (23, 24 or 25 hours; 92, 96 or 100 quarter-hours), with a finite prediction
    and actual in every period. `n_seen` is the number of days with any forecast in `frame` and `n_dropped` how many of
    them were not whole. `days` has `FORECAST_KEYS` (with `hour` 0), `prediction` and `actual`.

    `periods_in_target` maps each target date to the number of distinct periods that any model has a record for at any
    horizon. A day whose finite periods are fewer is dropped even when its ids look like a clock change: a model that lacks
    the hour 02:00 on an ordinary Tuesday must not be aggregated over 23 hours while another is aggregated over 24, since
    the two would be scored against different actuals on the same key.

    `clock` maps each target date to the period ids its clock has (`models._calendar.clock_periods`), when the data's time zone
    is known. A day is then whole when its finite periods are exactly those, in place of both rules above: so a zone whose clock
    skips midnight, 22:00 or half an hour has its short day too.
    """
    if agg not in _REDUCE:
        raise ValueError(f"agg must be one of {list(AGGREGATES)}, got {agg!r}")
    empty = pd.DataFrame({c: [] for c in (*FORECAST_KEYS, "prediction", "actual")})
    if frame.empty:
        return empty, 0, 0
    missing = [k for k in FORECAST_KEYS if k not in frame.columns]
    if missing:
        raise ValueError(
            f"Aggregating needs the forecast keys {list(FORECAST_KEYS)}, but the frame lacks {missing}. "
            "Result records need run_date, target_date, hour and horizon."
        )

    run, runs = pd.factorize(frame["run_date"].to_numpy(), sort=True)
    target, targets = pd.factorize(frame["target_date"].to_numpy(), sort=True)
    horizon, horizons = pd.factorize(frame["horizon"].to_numpy(), sort=True)
    if (run < 0).any() or (target < 0).any() or (horizon < 0).any() or frame["hour"].isna().any():
        raise ValueError("Aggregating: a frame has rows with a missing forecast key.")
    day_key = (run.astype(np.int64) * len(targets) + target) * len(horizons) + horizon
    hour = frame["hour"].to_numpy(dtype=np.int64)
    order = np.lexsort((hour, day_key))
    if ((day_key[order][1:] == day_key[order][:-1]) & (hour[order][1:] == hour[order][:-1])).any():
        raise ValueError("Aggregating needs each forecast key once; the frame has duplicated keys.")
    n_seen = len(np.unique(day_key))

    prediction = frame["prediction"].to_numpy(dtype=float)
    actual = frame["actual"].to_numpy(dtype=float)
    keep = order[np.isfinite(prediction[order]) & np.isfinite(actual[order])]
    day_key, hour, prediction, actual = day_key[keep], hour[keep], prediction[keep], actual[keep]
    run, target, horizon = run[keep], target[keep], horizon[keep]
    codes, first = np.unique(day_key, return_index=True)
    day = np.searchsorted(codes, day_key)
    sunday = pd.DatetimeIndex(targets[target[first]]).dayofweek == 6

    reduce = _REDUCE[agg]
    n = len(codes)
    reduced_prediction, reduced_actual = np.full(n, np.nan), np.full(n, np.nan)
    found = (
        whole_days(day, hour, sunday, per_hour)
        if clock is None
        else _days_on_the_clock(day, hour, targets[target[first]], clock)
    )
    for periods, days in found:
        rows = np.isin(day, days)
        reduced_prediction[days] = reduce(prediction[rows].reshape(-1, periods))
        reduced_actual[days] = reduce(actual[rows].reshape(-1, periods))
    whole = np.isfinite(reduced_prediction) & np.isfinite(reduced_actual)
    if periods_in_target is not None and clock is None:
        expected = periods_in_target.reindex(pd.DatetimeIndex(targets[target[first]])).to_numpy()
        whole &= np.bincount(day, minlength=n) == expected

    kept = first[whole]
    days = pd.DataFrame(
        {
            "run_date": runs[run[kept]],
            "target_date": targets[target[kept]],
            "hour": np.zeros(len(kept), dtype=np.int64),
            "horizon": horizons[horizon[kept]],
            "prediction": reduced_prediction[whole],
            "actual": reduced_actual[whole],
        }
    )[[*FORECAST_KEYS, "prediction", "actual"]]
    return days, n_seen, n_seen - int(whole.sum())
