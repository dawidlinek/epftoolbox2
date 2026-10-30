"""Result records with a known structure, and an independent computation of what a pair of models should score."""

from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

KEYS = ["run_date", "target_date", "hour", "horizon"]


def forecast_records(
    seed: int,
    run_dates: Iterable,
    noise: float = 5.0,
    bias: float = 0.0,
    horizons: Iterable[int] = (1,),
    periods: Callable[[pd.Timestamp], int] = lambda day: 24,
    drop: Optional[Callable[[dict], bool]] = None,
) -> list[dict]:
    """One model's records. The actual depends only on the target day and period, so models share it. The error is
    normal(bias, noise). `periods(target_day)` is the number of periods of that day, `drop(record)` removes a forecast."""
    rng = np.random.default_rng(seed)
    records = []
    for i, run_date in enumerate(pd.to_datetime(list(run_dates))):
        for horizon in horizons:
            target = run_date + pd.Timedelta(days=horizon)
            count = periods(target)
            for period in range(count):
                actual = 50 + 20 * np.sin(2 * np.pi * period / count) + 10 * np.sin(target.dayofyear / 9)
                record = {
                    "run_date": run_date.date().isoformat(),
                    "target_date": target.date().isoformat(),
                    "run_weekday": int(run_date.dayofweek),
                    "hour": period,
                    "horizon": horizon,
                    "day_in_test": i,
                    "prediction": float(actual + rng.normal(bias, noise)),
                    "actual": float(actual),
                }
                if drop is None or not drop(record):
                    records.append(record)
    return records


def shared_errors(records_a: list[dict], records_b: list[dict], horizon: Optional[int] = None) -> pd.DataFrame:
    """Absolute errors `a`, `b` of a pair on the keys it shares (finite on both sides), by merging on the keys."""
    a, b = pd.DataFrame(records_a), pd.DataFrame(records_b)
    merged = a.merge(b, on=KEYS, suffixes=("_a", "_b"))
    if horizon is not None:
        merged = merged[merged["horizon"] == horizon]
    merged["a"] = (merged["prediction_a"] - merged["actual_a"]).abs()
    merged["b"] = (merged["prediction_b"] - merged["actual_b"]).abs()
    merged = merged[np.isfinite(merged["a"]) & np.isfinite(merged["b"])]
    return merged.sort_values(["run_date", "horizon", "hour"])[[*KEYS, "a", "b"]]


def daily_losses(errors: pd.DataFrame) -> pd.DataFrame:
    """Mean error per (run_date, horizon), averaged over the horizons of each run_date, in date order."""
    return errors.groupby(["run_date", "horizon"])[["a", "b"]].mean().groupby("run_date").mean()


def daily_sums(errors: pd.DataFrame) -> pd.DataFrame:
    return errors.groupby("run_date")[["a", "b"]].sum()
