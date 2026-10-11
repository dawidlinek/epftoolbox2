"""Synthetic market data with a known answer, for tests that check numbers rather than shapes.

Price is an exact linear function of load, wind and a weekend dummy (plus optional noise), so an OLS
model given the right predictors should reproduce it. Future-dated predictor columns (`x_d+h`) are
built with local *calendar-day* shifts, independently of the library's own implementation.
"""

import numpy as np
import pandas as pd

PRICE_INTERCEPT = 20.0
PRICE_COEFS = {"load": 0.08, "wind": -0.15, "is_weekend": 10.0}
# One-hot pair (like the library's weekday dummies) so the no-intercept OLS can represent the level.
PREDICTORS = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]


def calendar_shift(series: pd.Series, days: int) -> pd.Series:
    """Value of `series` at the same local wall-clock time `days` calendar days later (NaN if absent)."""
    index = series.index
    wall_clock = index.tz_localize(None) if index.tz is not None else index
    lookup = pd.Series(series.to_numpy(), index=wall_clock)
    lookup = lookup[~lookup.index.duplicated(keep="first")]
    return pd.Series(lookup.reindex(wall_clock + pd.Timedelta(days=days)).to_numpy(), index=index)


def make_market_data(
    start: str = "2023-01-01",
    days: int = 150,
    tz: str = "Europe/Warsaw",
    freq: str = "1h",
    horizon: int = 2,
    noise: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Hourly (or 15-min) frame in local time with `price`, drivers, and `_d+h` / `_d-1` columns.

    The default range covers the spring DST switch (2023-03-26 in Europe/Warsaw).
    """
    rng = np.random.default_rng(seed)
    first = pd.Timestamp(start, tz=tz)
    index = pd.date_range(first, first + pd.Timedelta(days=days), freq=freq, inclusive="left")

    hour = index.hour + index.minute / 60
    day_of_year = index.dayofyear.to_numpy()
    load = (
        1000
        + 200 * np.sin(2 * np.pi * (hour - 6) / 24)
        + 50 * np.sin(2 * np.pi * day_of_year / 365)
        + rng.normal(0, 30, len(index))
    )
    wind = rng.gamma(2.0, 50.0, len(index))
    is_weekend = (index.dayofweek >= 5).astype(float)

    df = pd.DataFrame(
        {"load": load, "wind": wind, "is_weekend": is_weekend, "is_weekday": 1.0 - is_weekend}, index=index
    )
    df["price"] = (
        PRICE_INTERCEPT
        + PRICE_COEFS["load"] * df["load"]
        + PRICE_COEFS["wind"] * df["wind"]
        + PRICE_COEFS["is_weekend"] * df["is_weekend"]
        + noise * rng.normal(0, 1, len(index))
    )

    columns = {}
    for h in range(1, horizon + 1):
        for col in ("load", "wind", "is_weekend", "is_weekday"):
            columns[f"{col}_d+{h}"] = calendar_shift(df[col], h)
    columns["price_d-1"] = calendar_shift(df["price"], -1)
    return pd.concat([df, pd.DataFrame(columns, index=index)], axis=1)
