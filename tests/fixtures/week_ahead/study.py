"""The published week-ahead study's predictor lists and settings, in this library's terms.

The study forecasts the daily minimum and maximum load of D+3 to D+9 on Fridays at noon, as the TSO's week-ahead (A31) forecast
does, with a Lasso on regional weather (`regional_nwp`) and an expert OLS. Its rows are anchored at local noon. The lists are its
`model_config.py`'s (a verbatim copy is `~/Desktop/min-max-hour-load-forecasting/src/mmh/parent_config.py`), with `{hz}` written
`{horizon}` and in the same order, since the order of a Lasso's columns is the order coordinate descent visits them;
`build.py` checks them against that file whenever it runs.
"""

HORIZONS = range(3, 10)  # D+3 .. D+9: the week the TSO's Friday forecast covers
TRAINING_WINDOW = 1092
STATS = ("min", "max")
ECMWF_VARS = ("2t", "wind_speed", "rh", "tcc", "sp", "ssr_inc")

CALENDAR_CORE = [
    "is_holiday",
    *[f"is_holiday_d-{h}" for h in range(1, 8)],
    *[f"is_holiday_d+{h}" for h in range(1, 10)],
    "is_monday_d+{horizon}",
    "is_tuesday_d+{horizon}",
    "is_wednesday_d+{horizon}",
    "is_thursday_d+{horizon}",
    "is_friday_d+{horizon}",
    "is_saturday_d+{horizon}",
    "is_sunday_d+{horizon}",
    "daylight_hours_d+{horizon}",
]
LAGS_FULL = [
    *[f"load_actual_h-{h}" for h in range(1, 169)],
    *[f"load_forecast_h+{h}" for h in range(1, 36)],
    "load_forecast",
]
# The TSO's A31 forecasts of both statistics at every horizon, whatever the forecast's own horizon.
TSO_WEEK = [f"load_forecast_daily_{stat}_d+{h}" for stat in STATS for h in HORIZONS]
DAILY_EXT = [
    *[f"actual_daily_{stat}_d-{h}" for stat in STATS for h in range(1, 8)],
    *TSO_WEEK,
]
REGIONAL_NWP = (
    CALENDAR_CORE
    + LAGS_FULL
    + DAILY_EXT
    + [f"all_avg_{var}_dmean_d+{{horizon}}" for var in ECMWF_VARS]
    + ["all_avg_tp_dmean_d+{horizon}", "all_avg_hdd_d+{horizon}", "all_avg_cdd_d+{horizon}"]
)


def expert(stat: str) -> list:
    """The expert benchmark of one statistic: weekday structure, the TSO's forecasts of the run day and the target day, two lags
    of the statistic and the capital's temperature. The study fits it by OLS with an intercept on raw values."""
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


# The one name of the study's list that is late at a Friday noon issue: the hour before noon, known 13:00 (`realised("1h")`).
LATE_AT_NOON = "load_actual_h-1"
# Everything the TSO's weekly forecast may not have published yet on a day other than Friday, and the late hour: what the daily
# form drops when `ctx.published` says so.
MAYBE_LATE = frozenset([*TSO_WEEK, LATE_AT_NOON])


def expanded(names: list, horizon: int) -> list:
    return [name.replace("{horizon}", str(horizon)) for name in names]


def columns() -> list:
    """Every column the lists name at some horizon, in first use: the fixture keeps these and the two targets."""
    names = [name for h in HORIZONS for name in expanded(REGIONAL_NWP, h)]
    names += [name for h in HORIZONS for stat in STATS for name in expanded(expert(stat), h)]
    return list(dict.fromkeys(names))


# --- the daily form -----------------------------------------------------------------------------------------------------------
# Issued every day, the lists can't be the Friday ones: on Monday to Thursday the TSO hasn't published next week's forecast at
# noon. Each function returns its list without the names `ctx.published` says aren't out on that issue day.


def published_only(names: list, ctx) -> list:
    names = [name.replace("{horizon}", str(ctx.horizon)) for name in names]
    return [name for name in names if name not in MAYBE_LATE or ctx.published(name)]


def regional_nwp_known(ctx):
    return published_only(REGIONAL_NWP, ctx)


def expert_known_min(ctx):
    return published_only(expert("min"), ctx)


def expert_known_max(ctx):
    return published_only(expert("max"), ctx)
