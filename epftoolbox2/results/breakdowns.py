"""The parts of the breakdown tables that do not need the report itself.

Price levels are read from the actual price alone.
"""

from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd


# A spike is a price at or above the lower of these two: evalproto's definition (its Q4), 88 hours a year per market.
SPIKE_CAP = 300.0
SPIKE_QUANTILE = 0.99


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
