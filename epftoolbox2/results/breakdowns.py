"""The parts of the breakdown tables and of `EvaluationReport.aggregate()` that do not need the report itself.

Price levels are read from the actual price alone. Daily aggregates are made only of whole days, and a whole day is
judged by `evaluators._days`, the same rule that decides which days `EconomicEvaluator` can schedule. A block of the day
(`peak`, `offpeak`) is whole when every slot of it the day has is there, and an hour of quarter-hours when all four are.
Source-data columns are joined to forecasts by target date and period, as `CsvExporter(extra_columns=)` joins them.
"""

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._date_utils import zone_name
from ..evaluators._days import QUARTERS, day_codes, hour_codes, hour_of_code, whole_days, whole_hours
from ..evaluators.base import FORECAST_KEYS
from ..models._contract import check_index, period_index

# Reductions over the whole day, and means over a block of its hours (`peak_hours` and the rest of the day).
DAY_AGGREGATES = ("max", "min", "mean", "spread")
BLOCK_AGGREGATES = ("peak", "offpeak")
AGGREGATES = (*DAY_AGGREGATES, *BLOCK_AGGREGATES)

# The block of `agg="peak"`: 08:00 to 19:59 local time, the window of the resolution and renewables studies.
PEAK_HOURS = (8, 20)

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
        raise ValueError(f"agg must be one of {list(DAY_AGGREGATES)}, got {agg!r}")
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
    sunday = pd.DatetimeIndex(targets[target[first]]).day_of_week == 6

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


def check_peak_hours(peak_hours: Any) -> Tuple[int, int]:
    """`(first, end)` of the peak block in local hours, checked: whole hours with `0 <= first < end <= 24`."""
    try:
        first, end = peak_hours
    except (TypeError, ValueError):
        raise ValueError(f"peak_hours must be (first hour, end hour), such as (8, 20), got {peak_hours!r}") from None
    whole = all(isinstance(h, (int, np.integer)) and not isinstance(h, (bool, np.bool_)) for h in (first, end))
    if not whole or not 0 <= first < end <= 24:
        raise ValueError(
            f"peak_hours must be two whole local hours with 0 <= first < end <= 24 (the block is first:00 to end:00, end "
            f"left out), got {peak_hours!r}"
        )
    return int(first), int(end)


def in_block(period: np.ndarray, per_hour: int, agg: str, peak_hours: Tuple[int, int]) -> np.ndarray:
    """Whether each period id is in the block of `agg`: the peak hours, or the rest of the day for `"offpeak"`.

    A period id is the clock's (`hour * per_hour + quarter`), as a run writes it, so its local hour is `id // per_hour`.
    """
    hour = np.asarray(period, dtype=np.int64) // per_hour
    peak = (hour >= peak_hours[0]) & (hour < peak_hours[1])
    return peak if agg == "peak" else ~peak


def _keyed_days(frame: pd.DataFrame) -> Tuple[np.ndarray, ...]:
    """`(run, runs, target, targets, horizon, horizons, day_key, hour, order)` of a frame of forecasts, as `daily_aggregate`
    reads them: factorized keys, one code per (run_date, target_date, horizon), rows ordered by (day, period)."""
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
    return run, runs, target, targets, horizon, horizons, day_key, hour, order


def block_aggregate(
    frame: pd.DataFrame,
    agg: str,
    per_hour: int,
    peak_hours: Tuple[int, int],
    day_periods: Mapping[pd.Timestamp, Sequence[int]],
) -> Tuple[pd.DataFrame, int, int]:
    """One model's forecasts as one forecast per (run_date, target_date, horizon): the mean over a block of the day.

    The block is the peak hours (`agg="peak"`) or the rest of the day (`"offpeak"`), by the local hour of each period id.
    `day_periods` maps each target date to the period ids the date has: the clock's (`clock_periods_on`) when the zone is
    known, else every period that any model has a record for on that date. A day counts when the model has every one of
    them that is in the block, each with a finite prediction and actual, so the peak of a switch day counts and its offpeak
    counts only when the day has no repeated hour (whose actual is missing). Returns `(days, n_seen, n_dropped)` as
    `daily_aggregate` does. A day with a record of a slot its date doesn't have in `day_periods` is dropped too: its ids are
    numbered otherwise (see `numbered_by_the_clock`), so they aren't its local hours.
    """
    if agg not in BLOCK_AGGREGATES:
        raise ValueError(f"agg must be one of {list(BLOCK_AGGREGATES)}, got {agg!r}")
    empty = pd.DataFrame({c: [] for c in (*FORECAST_KEYS, "prediction", "actual")})
    if frame.empty:
        return empty, 0, 0
    run, runs, target, targets, horizon, horizons, day_key, hour, order = _keyed_days(frame)
    n_seen = len(np.unique(day_key))

    stamps = pd.DatetimeIndex(targets)
    expected = [np.asarray(day_periods.get(date, ()), dtype=np.int64) for date in stamps]
    size = max([int(hour.max()), *(int(ids.max()) for ids in expected if len(ids))]) + 1
    known = np.zeros((len(targets), size), dtype=bool)  # the slots of each target date
    wanted = np.zeros((len(targets), size), dtype=bool)  # the block's slots of each target date
    for code, ids in enumerate(expected):
        known[code, ids] = True
        wanted[code, ids[in_block(ids, per_hour, agg, peak_hours)]] = True
    # A day with a record of a slot its date doesn't have is numbered otherwise than by the clock: its hours aren't the
    # local hours of the ids, so its block can't be read.
    stray = np.unique(day_key[~known[target, hour]])

    prediction = frame["prediction"].to_numpy(dtype=float)
    actual = frame["actual"].to_numpy(dtype=float)
    rows = order[np.isfinite(prediction[order]) & np.isfinite(actual[order])]
    rows = rows[wanted[target[rows], hour[rows]]]
    codes, first, counts = np.unique(day_key[rows], return_index=True, return_counts=True)
    whole = (counts == wanted.sum(axis=1)[target[rows[first]]]) & ~np.isin(codes, stray)

    reduced_prediction, reduced_actual = np.full(len(codes), np.nan), np.full(len(codes), np.nan)
    for length in np.unique(counts[whole]).tolist():
        days = np.flatnonzero(whole & (counts == length))
        block = rows[np.isin(day_key[rows], codes[days])]
        reduced_prediction[days] = prediction[block].reshape(-1, length).mean(axis=1)
        reduced_actual[days] = actual[block].reshape(-1, length).mean(axis=1)

    kept = rows[first[whole]]
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


def numbered_by_the_clock(
    day_periods: Mapping[pd.Timestamp, Sequence[int]], per_hour: int
) -> Dict[pd.Timestamp, Tuple]:
    """`day_periods` without the dates whose ids are numbered through rather than by the clock, which `whole_days` accepts:
    a day with an id past 23:59 (a 25-hour day numbered 0-24, 0-99 at 15 minutes) and a Sunday numbered exactly 0-22 (0-91),
    a 23-hour day. Their ids aren't local hours after the switch, so a block of hours would be read an hour early or late;
    a date mapped to no periods is dropped and counted by `block_aggregate`."""
    out: Dict[pd.Timestamp, Tuple] = {}
    short = tuple(range(23 * per_hour))
    for date, ids in day_periods.items():
        ids = tuple(ids)
        through = (ids and max(ids) >= 24 * per_hour) or (pd.Timestamp(date).day_of_week == 6 and ids == short)
        out[date] = () if through else ids
    return out


def hourly_aggregate(frame: pd.DataFrame, agg: str) -> Tuple[pd.DataFrame, int, int]:
    """One model's quarter-hour forecasts as one forecast per (run_date, target_date, hour, horizon): `(hours, n_seen,
    n_dropped)`.

    The prediction and the actual of an hour are each reduced by `agg` over its four quarter-hours. An hour counts when all
    four are there with a finite prediction and actual (`evaluators._days.whole_hours`): the repeated autumn hour, whose
    actual is missing, and an hour that lacks a quarter-hour are dropped; the hour the spring switch skips has no
    quarter-hour at all, so it is no hour of that day. `n_seen` is the number of hours with any forecast in `frame` and
    `n_dropped` how many of them were not whole. `hours` has `FORECAST_KEYS` (`hour` the local hour), `prediction` and
    `actual`, ordered by (run_date, target_date, horizon, hour).
    """
    if agg not in _REDUCE:
        raise ValueError(f"agg must be one of {list(DAY_AGGREGATES)}, got {agg!r}")
    empty = pd.DataFrame({c: [] for c in (*FORECAST_KEYS, "prediction", "actual")})
    if frame.empty:
        return empty, 0, 0
    day = day_codes(frame, "Aggregating")
    period = frame["hour"].to_numpy(dtype=np.int64)
    prediction = frame["prediction"].to_numpy(dtype=float)
    actual = frame["actual"].to_numpy(dtype=float)
    rows, codes = whole_hours(day, period, np.isfinite(prediction) & np.isfinite(actual))
    n_seen = len(np.unique(hour_codes(day, period)))

    reduce = _REDUCE[agg]
    first = rows[::QUARTERS]
    hours = pd.DataFrame(
        {
            "run_date": frame["run_date"].to_numpy()[first],
            "target_date": frame["target_date"].to_numpy()[first],
            "hour": hour_of_code(codes)[1].astype(np.int64),
            "horizon": frame["horizon"].to_numpy()[first],
            "prediction": reduce(prediction[rows].reshape(-1, QUARTERS)),
            "actual": reduce(actual[rows].reshape(-1, QUARTERS)),
        }
    )[[*FORECAST_KEYS, "prediction", "actual"]]
    return (hours if len(hours) else empty), n_seen, n_seen - len(hours)


# The quarter-hours of a day that has neither a skipped nor a repeated hour.
_DAY_QUARTERS = 24 * QUARTERS


def error_split(frame: pd.DataFrame, switch_dates: Optional[set] = None) -> Dict[str, Any]:
    """How one model's squared error divides into the level of the day, the profile of its hours and the shape within them.

    On the whole non-switch days of the frame: forecast days (run date, target date, horizon) whose finite quarter-hours are
    exactly 0-95, whose target date is not in `switch_dates` (the dates a known zone's clocks change on; without a zone, a
    switch day lacks quarter-hours or its repeated hour has no actual, so it is not whole). With `e` the errors of a day,
    `level` the day's mean error and `hour_mean` each hour's, the mean squared error is exactly
    `mean(level²) + mean((hour_mean - level)²) + mean((e - hour_mean)²)`, averaged over the days, hours and quarter-hours,
    as the resolution study computes it (`evaluate.py`). Returns `days`, `forecasts` (quarter-hours used), `mse` and the
    three shares `level`, `hour` and `quarter` (NaN without a whole day).
    """
    out: Dict[str, Any] = {"days": 0, "forecasts": 0, "mse": np.nan, "level": np.nan, "hour": np.nan, "quarter": np.nan}
    if frame.empty:
        return out
    day = day_codes(frame, "error_split()")
    period = frame["hour"].to_numpy(dtype=np.int64)
    error = frame["prediction"].to_numpy(dtype=float) - frame["actual"].to_numpy(dtype=float)
    order = np.lexsort((period, day))
    if ((day[order][1:] == day[order][:-1]) & (period[order][1:] == period[order][:-1])).any():
        raise ValueError("error_split() needs each forecast key once; the frame has duplicated keys.")
    kept = order[np.isfinite(error[order])]
    codes, first, counts = np.unique(day[kept], return_index=True, return_counts=True)
    # Each day's finite ids are distinct and in order, so 96 of them ending at 95 are exactly 0-95: a day numbered through
    # with 100 (a 25-hour day) is not whole, whatever its first 96 hold.
    whole = (counts == _DAY_QUARTERS) & (period[kept][first + counts - 1] < _DAY_QUARTERS)
    if switch_dates:
        dates = pd.DatetimeIndex(frame["target_date"].to_numpy()[kept[first]])
        whole &= ~dates.isin(pd.DatetimeIndex(sorted(switch_dates)))
    rows = kept[np.isin(day[kept], codes[whole])]
    if not len(rows):
        return out
    days = error[rows].reshape(-1, _DAY_QUARTERS)
    level = days.mean(1, keepdims=True)
    hour_mean = days.reshape(-1, 24, QUARTERS).mean(2)
    total = (days**2).mean()
    out.update(
        days=len(days),
        forecasts=days.size,
        mse=float(total),
        level=float((level**2).mean() / total),
        hour=float(((hour_mean - level) ** 2).mean() / total),
        quarter=float(((days.reshape(-1, 24, QUARTERS) - hour_mean[:, :, None]) ** 2).mean() / total),
    )
    return out


def switch_dates(dates: Sequence[Any], zone: Any) -> set:
    """The local dates among `dates` that are not 24 hours long in `zone`: the days its clocks change on."""
    days = pd.DatetimeIndex(pd.to_datetime(list(dates))).unique().sort_values()
    if zone is None or not len(days):
        return set()
    starts = days.tz_localize(zone, nonexistent="shift_forward", ambiguous=np.ones(len(days), dtype=bool))
    ends = (days + pd.Timedelta(days=1)).tz_localize(
        zone, nonexistent="shift_forward", ambiguous=np.ones(len(days), dtype=bool)
    )
    lengths = ends.as_unit("ns").asi8 - starts.as_unit("ns").asi8
    return set(days[lengths != pd.Timedelta(days=1).value])


def zone_key(tz: Any) -> Optional[str]:
    """How a time zone is compared: its name, or `str` of a zone without one that reads back (a `pytz.FixedOffset(60)`), as
    a manifest's `data_timezone` is written; None for no zone."""
    return zone_name(tz) or (None if tz is None else str(tz))


def check_source_zone(source_data: pd.DataFrame, zones: Iterable[Any], what: str) -> None:
    """Raise unless the source's index is in the results' time zone, when the results have one (`zones`, the refs').

    A join by target date and period reads the source's wall clock, so a source indexed in another zone, or in none, would
    give each forecast another hour's row without a word: Europe/Berlin results read from a UTC frame were all an hour or
    two off. Results without a zone (plain lists) read the wall clock as given, as before."""
    keys = {zone_key(zone) for zone in zones if zone is not None}
    if not keys:
        return
    source = zone_key(getattr(source_data.index, "tz", None))
    if keys != {source}:
        raise ValueError(
            f"{what}: the source data's index is in the time zone {source}, the results in {sorted(keys)}, and the join "
            "reads its wall clock, which would put each forecast on another hour's row. Convert the source to the results' "
            "zone (source.tz_convert(...)), as the frame of the run is."
        )


def source_table(source_data: pd.DataFrame, columns: Sequence[str], freq: str, days: Any = None) -> pd.DataFrame:
    """The values of `columns` of the source data by (target date, period of the day), the key a forecast is joined by.

    The join of `CsvExporter(extra_columns=)`: the source must have one row per local date and period at the results' `freq`
    (`check_index`, over `days`, the first and last target date), its local wall clock gives the date and the period, and
    the hour that occurs twice when clocks fall back keeps its second occurrence (the later row, the one a run forecasts
    from). The index has two levels, `target_date` (naive midnight) and `hour`.
    """
    missing = [c for c in columns if c not in source_data.columns]
    if missing:
        raise ValueError(f"The source data has no column {missing}; it has {list(source_data.columns)}.")
    try:
        check_index(source_data, freq, days)
    except ValueError as exc:
        raise ValueError(
            f"Joining source columns to {freq} forecasts needs source data at that resolution: {exc} Each forecast takes "
            f"the source row of its target date and period, so resample the source to {freq} first."
        ) from exc
    src = source_data[list(columns)]
    src = src.iloc[np.argsort(src.index.asi8, kind="stable")]  # in time order, so "last" is the second occurrence
    index = src.index
    if index.tz is not None:
        index = index.tz_localize(None)
    keys = pd.MultiIndex.from_arrays(
        [index.normalize().as_unit("ns"), period_index(index, freq).astype(np.int64)], names=["target_date", "hour"]
    )
    table = src.copy()
    table.index = keys
    return table[~keys.duplicated(keep="last")]


def source_values(table: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """The rows of `source_table` at each forecast of `frame` (its target date and period), aligned to `frame`'s index; NaN
    where the source has no row."""
    keys = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(frame["target_date"]).as_unit("ns"), frame["hour"].to_numpy(dtype=np.int64)],
        names=["target_date", "hour"],
    )
    values = table.reindex(keys)
    values.index = frame.index
    return values
