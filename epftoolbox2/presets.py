"""Predictor lists for benchmark models from the literature, in this library's column naming.

Both benchmark presets are day-ahead models (`tree_predictors` is LEAR's inputs, the period and the row's own period). The
papers index inputs by the target (delivery) day d; this library indexes columns by the issue day D of the row, and a forecast
issued on D for horizon h targets D+h. For h = 1 (d = D+1):

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

from typing import Any, Optional, Sequence

from .data.transformers._daily import parse_step, slot_names

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def profile_columns(column: str, freq: str = "1h") -> list[str]:
    """Names `DailyProfileTransformer(columns=column, freq=freq)` creates: price_h00..price_h23 for hourly data."""
    return slot_names(column, parse_step(freq))


def exaa_profile(freq: str = "1h", prefix: str = "") -> list[str]:
    """The EXAA auction's profile as predictors of the target day: `price_exaa_h00_d+{horizon}` .. `price_exaa_h23_d+{horizon}`.

    The names `DailyProfileTransformer(columns="price_exaa", freq=freq)` creates (`profile_columns`), shifted to the target day with
    `LagTransformer(columns=profile_columns("price_exaa", freq), lags=[-1], freq="day")`. `prefix` is the `EntsoeSource`
    prefix: `prefix="de"` gives `de_price_exaa_h00_d+{horizon}`. Only AT and DE_LU have the column
    (`EntsoeSource(price_exaa=True)`), and it is out at 11:15 on D (the legal bound), so the list is for horizon 1 and a
    model that is a function of `ctx` should add it only where `ctx.published(exaa_profile())` is true.

    Evidence: the docs' "Early auctions" section of the ENTSO-E page (DE_LU, EXAA added to a LEAR as a function of `ctx`). It
    gives no accuracy gain for adding the column, only the serving lesson: a model always trained with EXAA and served
    without it raised the DE error from 15.23 to 38.44 in earlier projects, so keep the list without it as the fallback.
    """
    head = f"{prefix}_" if prefix and not prefix.endswith("_") else prefix
    return [f"{c}_d+{{horizon}}" for c in profile_columns(f"{head}price_exaa", freq)]


def _weekday_columns(weekdays: Sequence[str], calendar_prefix: str) -> list[str]:
    unknown = [d for d in weekdays if d not in WEEKDAYS]
    if unknown:
        raise ValueError(f"Unknown weekday names {unknown}; use {list(WEEKDAYS)}")
    return [f"{calendar_prefix}is_{d}_d+{{horizon}}" for d in weekdays]


ORDERS = ("library", "lago")


def lear_predictors(
    exogenous: Sequence[str],
    price: str = "price",
    freq: str = "1h",
    calendar_prefix: str = "",
    order: str = "library",
) -> list[str]:
    """Inputs of the LEAR model (Lago, Marcjasz, De Schutter & Weron 2021, Applied Energy 293).

    All slot prices of d-1, d-2, d-3 and d-7; all slot values of each exogenous series for d, d-1 and d-7; seven
    weekday dummies of d. With hourly data and two exogenous series that is 96 + 144 + 7 = 247 inputs.

    `order` is the order of the same names. `"library"` (the default) is lag-major: every slot's price of d-1, then of d-2,
    d-3 and d-7, then per exogenous series every slot for d, d-1 and d-7, then the dummies. `"lago"` is the original
    epftoolbox's (`LEAR._build_and_split_XYs`), slot-major: per slot the prices of d-1, d-2, d-3 and d-7; then per slot
    each exogenous series for d-1, each for d-7 and each for d; then the dummies, Monday to Sunday. The order changes no
    fit that converges, but coordinate descent that stops early (`LEARModel(solver="cd")`, Lago's `Lasso(max_iter=2500)`)
    sweeps the columns in it, so reproducing the original's forecasts needs `"lago"` and the exogenous series in the order
    of the dataset's file.

    The data pipeline must produce (hourly shown; see the module docstring for the mapping):
    - `DailyProfileTransformer(columns=[price, *exogenous])` after `DSTNormalizeTransformer`;
    - `LagTransformer(columns=profile_columns(price), lags=[1, 2, 6], freq="day")`;
    - `LagTransformer(columns=<profile columns of each exogenous series>, lags=[-1, ..., -H, 6], freq="day")`;
    - weekday dummies `is_monday` .. `is_sunday` lagged by -1 .. -H days.
    """
    if order not in ORDERS:
        raise ValueError(f"order must be one of {list(ORDERS)}, got {order!r}")
    dummies = _weekday_columns(WEEKDAYS, calendar_prefix)
    if order == "lago":
        prices = profile_columns(price, freq)
        slots = {x: profile_columns(x, freq) for x in exogenous}
        out = []
        for column in prices:
            out += [f"{column}{lag}" for lag in ("", "_d-1", "_d-2", "_d-6")]
        for slot in range(len(prices)):
            out += [f"{slots[x][slot]}{lag}" for lag in ("", "_d-6") for x in exogenous]
            out += [f"{slots[x][slot]}_d+{{horizon}}" for x in exogenous]
        return out + dummies
    prices = [f"{c}{lag}" for lag in ("", "_d-1", "_d-2", "_d-6") for c in profile_columns(price, freq)]
    exog = [f"{c}{lag}" for x in exogenous for lag in ("_d+{horizon}", "", "_d-6") for c in profile_columns(x, freq)]
    return prices + exog + dummies


def tree_predictors(
    exogenous: Sequence[str],
    price: str = "price",
    freq: str = "1h",
    calendar_prefix: str = "",
) -> list[str]:
    """Inputs of a tree pooled over the periods of a day (`periods="pooled"`: `HistGBModel`, `LightGBMModel`, `XGBoostModel`,
    `CatBoostModel`), the research's "global" rows (`zoo_classic`'s `ztree.py`): every name of `lear_predictors` (the day's
    slot prices, slot values of each exogenous series and weekday dummies, the same on every row of a date), then `"period"`
    (the row's period, 0 to P-1), then the row's own period of the same blocks: the price of d-1, d-2, d-3 and d-7 (`price`,
    `price_d-1`, `price_d-2`, `price_d-6`) and each exogenous series for d, d-1 and d-7 (`x_d+{horizon}`, `x`, `x_d-6`). With
    hourly data and one exogenous series that is 175 + 1 + 7 = 183 inputs, in the research's column order.

    A `ResidualModel` learner adds its base's forecasts after them: `[f"base_fc_h{h:02d}_d+{{horizon}}" for h in range(24)]`
    (the target day's, day-level) and `"base_fc_d+{horizon}"` (the row's own period), as the research's rows have them.

    The data pipeline must produce what `lear_predictors` needs and the plain lags of the own-period names:
    `LagTransformer(columns=[price], lags=[1, 2, 6], freq="day")` and `LagTransformer(columns=exogenous, lags=[-1, ..., -H, 6],
    freq="day")`.
    """
    own = [price, f"{price}_d-1", f"{price}_d-2", f"{price}_d-6"]
    own += [f"{x}{lag}" for x in exogenous for lag in ("_d+{horizon}", "", "_d-6")]
    return [*lear_predictors(exogenous, price, freq, calendar_prefix), "period", *own]


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


WEEK_AHEAD = ("calendar_x", "regional_nwp", "expert")


def week_ahead_predictors(which: str = "regional_nwp", stat: Optional[str] = None) -> list[str]:
    """The predictor lists of the week-ahead daily minimum and maximum load study (20 European markets, Friday 12:00 issues).

    `which="calendar_x"` (25 names): holiday flags of D, D-1..D-7 and D+1..D+9, the seven weekday dummies of the target day
    and its daylight hours. `which="regional_nwp"` (266 names, in the order coordinate descent visits them): `calendar_x`,
    168 hourly lags of the realised load, the TSO's next 35 hourly forecasts and its forecast of the issue row, the last
    seven daily minima and maxima, the TSO's A31 daily minima and maxima for D+3..D+9, seven ECMWF variables averaged over
    the market's regions and their heating and cooling degree days. `which="expert"` (8 names, `stat="min"` or `"max"`
    required): Monday, Saturday and Sunday-or-holiday dummies, the TSO's forecast of D and of the target day, the actual
    statistic of D-1 and D-7 and the capital's temperature.

    Evidence: the S02 project's study, reproduced by the library (docs "Week-ahead Forecasting", "Reproducing the study"): 20
    markets, 104 Friday issues each (2024-01-05 to 2025-12-26), horizons 3..9. `regional_nwp` with a LassoCV (`cv=7`,
    `fit_intercept=False`, standardised, `training_window=1092`) beats the TSO's A31 forecast in all 40 market-statistics, by a
    mean log-gain of +37.44 % (29,120 forecasts of each statistic, within 0.0005 MW of the stored forecasts); for Poland the
    log-gain is 23.6 % (minimum) and 20.3 % (maximum) over 728 forecasts each. `expert` is the benchmark OLS
    (`fit_intercept=True`; Poland MAE 417.24 MW over 1,456 forecasts against the Lasso's 367.1).

    The lists hold names of series that a daily frame stamped at noon builds (see the docs page); `{horizon}` is the
    forecast's horizon. `load_actual_h-1` (11:00's load) is known only at 13:00: drop it for a noon issue, as the page says.
    """
    if which not in WEEK_AHEAD:
        raise ValueError(f"which must be one of {list(WEEK_AHEAD)}, got {which!r}")
    if which == "expert":
        if stat not in ("min", "max"):
            raise ValueError(f'which="expert" needs stat="min" or "max", got {stat!r}')
    elif stat is not None:
        raise ValueError(f"stat is only for which='expert', got stat={stat!r} with which={which!r}")
    if which == "expert":
        return [
            "is_monday_d+{horizon}",
            "is_saturday_d+{horizon}",
            "is_sunday_or_holiday_d+{horizon}",
            f"load_forecast_daily_{stat}_d+0",
            f"load_forecast_daily_{stat}_d+{{horizon}}",
            f"actual_daily_{stat}_d-1",
            f"actual_daily_{stat}_d-6",
            "capital_2t_dmean_d+{horizon}",
        ]
    calendar_x = (
        ["is_holiday"]
        + [f"is_holiday_d-{k}" for k in range(1, 8)]
        + [f"is_holiday_d+{k}" for k in range(1, 10)]
        + [f"is_{d}_d+{{horizon}}" for d in WEEKDAYS]
        + ["daylight_hours_d+{horizon}"]
    )
    if which == "calendar_x":
        return calendar_x
    return (
        calendar_x
        + [f"load_actual_h-{h}" for h in range(1, 169)]
        + [f"load_forecast_h+{h}" for h in range(1, 36)]
        + ["load_forecast"]
        + [f"actual_daily_{s}_d-{k}" for s in ("min", "max") for k in range(1, 8)]
        + [f"load_forecast_daily_{s}_d+{k}" for s in ("min", "max") for k in range(3, 10)]
        + [f"all_avg_{v}_dmean_d+{{horizon}}" for v in ("2t", "wind_speed", "rh", "tcc", "sp", "ssr_inc", "tp")]
        + ["all_avg_hdd_d+{horizon}", "all_avg_cdd_d+{horizon}"]
    )


# zone -> (issue_timezone, name of the day-ahead load forecast column). Every docs example issues at 11:45 local time.
_LIVE_ZONES = {
    "DE_LU": ("Europe/Berlin", "load_forecast"),
    "DE": ("Europe/Berlin", "load_forecast"),
    "AT": ("Europe/Vienna", "load_forecast"),
    "PL": ("Europe/Warsaw", "pse_load_fcst"),
    "BE": ("Europe/Brussels", "load_forecast"),
}
_EXAA_ZONES = ("DE_LU", "AT")


def live_defaults(zone: str, exaa: bool = False) -> dict[str, Any]:
    """Live-run settings of a day-ahead model for `zone`, as a dict for the `forecast(**...)` of step 8e (which does not exist yet).

    Keys: `issue_time` and `issue_timezone` (what `ModelPipeline(issue_time=, issue_timezone=)` takes today), `deadline` (no
    revision is committed after it) and `wait_for` (per column, the local time to stop polling for its issue-row cells). The
    values are those of the docs and the simulated week of the operations spec: issue at 11:45 local time for the next day,
    wait for the day-ahead load forecast (published 10:00, ENTSO-E rule; PSE's `pse_load_fcst` for PL) until 10:45, commit
    nothing after 11:55. `exaa=True` (DE_LU and AT) also waits for `price_exaa` until 11:40, the second call of the simulated
    week (the legal bound for EXAA is 11:15).

    Evidence: the issue times are those of the docs' examples (`InformationSet("11:45", ...)` for Berlin, Warsaw and
    Brussels); `deadline` and `wait_for` are the operations spec's proposal, not measured on a live run yet. Unknown zones raise.
    """
    if zone not in _LIVE_ZONES:
        raise ValueError(f"No live defaults for zone {zone!r}; known: {sorted(_LIVE_ZONES)}")
    if exaa and zone not in _EXAA_ZONES:
        raise ValueError(f"exaa=True needs one of {list(_EXAA_ZONES)}, got {zone!r}")
    timezone, load = _LIVE_ZONES[zone]
    wait_for = {load: "10:45"}
    if exaa:
        wait_for["price_exaa"] = "11:40"
    return {"issue_time": "11:45", "issue_timezone": timezone, "deadline": "11:55", "wait_for": wait_for}


__all__ = [
    "ORDERS",
    "WEEKDAYS",
    "WEEK_AHEAD",
    "exaa_profile",
    "expert_predictors",
    "lear_predictors",
    "live_defaults",
    "profile_columns",
    "tree_predictors",
    "week_ahead_predictors",
]
