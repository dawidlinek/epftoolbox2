"""Predictor lists for benchmark models from the literature, in this library's column naming.

Both presets are day-ahead models. The papers index inputs by the target (delivery) day d; this library indexes
columns by the issue day D of the row, and a forecast issued on D for horizon h targets D+h. For h = 1 (d = D+1):

    paper         column on the row of D              built by
    p(d-1, ·)     price, or price_h00 .. price_h23    (DailyProfileTransformer)
    p(d-2, ·)     price_d-1, price_h05_d-1            LagTransformer(lags=[1], freq="day")
    p(d-3, ·)     price_h05_d-2                       LagTransformer(lags=[2], freq="day")
    p(d-7, ·)     price_d-6, price_h05_d-6            LagTransformer(lags=[6], freq="day")
    x(d, ·)       x_d+{horizon}, x_h05_d+{horizon}    LagTransformer(lags=[-1, ..., -H], freq="day")
    x(d-1, ·)     x, x_h05
    x(d-7, ·)     x_h05_d-6                           LagTransformer(lags=[6], freq="day")
    weekday of d  is_monday_d+{horizon}, ...          CalendarSource(weekday="onehot") + LagTransformer

Using `price` (the day-ahead prices of D) assumes the forecast is issued after D's prices were published, which
happens on D-1 around 13:00. The exogenous series must be known in advance for the day they describe (day-ahead
forecasts such as `load_forecast`), never realised values. For h > 1 the price inputs stay anchored at the issue
day (the latest prices available) and `x_d+{horizon}` must really be known h days ahead, which ENTSO-E's
day-ahead forecasts are not.

The functions return plain strings with a `{horizon}` template, so the lists can be saved to YAML.
"""

from typing import Sequence

from .data.transformers._daily import parse_step, slot_names

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def profile_columns(column: str, freq: str = "1h") -> list[str]:
    """Names `DailyProfileTransformer(columns=column, freq=freq)` creates: price_h00..price_h23 for hourly data."""
    return slot_names(column, parse_step(freq))


def _weekday_columns(weekdays: Sequence[str], calendar_prefix: str) -> list[str]:
    unknown = [d for d in weekdays if d not in WEEKDAYS]
    if unknown:
        raise ValueError(f"Unknown weekday names {unknown}; use {list(WEEKDAYS)}")
    return [f"{calendar_prefix}is_{d}_d+{{horizon}}" for d in weekdays]


def lear_predictors(
    exogenous: Sequence[str],
    price: str = "price",
    freq: str = "1h",
    calendar_prefix: str = "",
) -> list[str]:
    """Inputs of the LEAR model (Lago, Marcjasz, De Schutter & Weron 2021, Applied Energy 293).

    All slot prices of d-1, d-2, d-3 and d-7; all slot values of each exogenous series for d, d-1 and d-7; seven
    weekday dummies of d. With hourly data and two exogenous series that is 96 + 144 + 7 = 247 inputs.

    The data pipeline must produce (hourly shown; see the module docstring for the mapping):
    - `DailyProfileTransformer(columns=[price, *exogenous])` after `DSTNormalizeTransformer`;
    - `LagTransformer(columns=profile_columns(price), lags=[1, 2, 6], freq="day")`;
    - `LagTransformer(columns=<profile columns of each exogenous series>, lags=[-1, ..., -H, 6], freq="day")`;
    - weekday dummies `is_monday` .. `is_sunday` lagged by -1 .. -H days.
    """
    prices = [f"{c}{lag}" for lag in ("", "_d-1", "_d-2", "_d-6") for c in profile_columns(price, freq)]
    exog = [f"{c}{lag}" for x in exogenous for lag in ("_d+{horizon}", "", "_d-6") for c in profile_columns(x, freq)]
    return prices + exog + _weekday_columns(WEEKDAYS, calendar_prefix)


def expert_predictors(
    exogenous: Sequence[str] = ("load_forecast",),
    price: str = "price",
    weekdays: Sequence[str] = WEEKDAYS,
    calendar_prefix: str = "",
) -> list[str]:
    """Inputs of the "expert" model (Ziel & Weron 2018, Energy Economics 70), per slot.

    The same slot's price on d-1, d-2 and d-7; the minimum, maximum and last price of d-1; each exogenous series
    for d; weekday dummies of d.

    Ziel & Weron use an intercept plus Monday, Saturday and Sunday dummies. `OLSModel` fits without an intercept
    and gets the level from a full set of one-hot dummies, so the default is all seven weekdays (which spans the
    intercept and the three dummies). Pass `weekdays=("monday", "saturday", "sunday")` for the original set with
    a model that fits an intercept, such as `LassoCVModel`.

    The data pipeline must produce:
    - `LagTransformer(columns=price, lags=[1, 6], freq="day")`;
    - `DailyStatsTransformer(columns=price, stats=["min", "max", "last"])`;
    - `LagTransformer(columns=exogenous, lags=[-1, ..., -H], freq="day")`;
    - weekday dummies lagged by -1 .. -H days.
    """
    own = [price, f"{price}_d-1", f"{price}_d-6", f"{price}_day_min", f"{price}_day_max", f"{price}_day_last"]
    exog = [f"{x}_d+{{horizon}}" for x in exogenous]
    return own + exog + _weekday_columns(weekdays, calendar_prefix)


__all__ = ["WEEKDAYS", "expert_predictors", "lear_predictors", "profile_columns"]
