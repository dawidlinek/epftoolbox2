"""The data contract a model run checks before any fit.

Every check here turns a failure mode that would otherwise produce wrong or missing forecasts without an error
into an error: data at a finer resolution than `freq` (a period mixing several times of day) or at a coarser one
in part of the frame (some periods trained on far fewer days), columns that
collide with the model's own target or id columns, predictors that don't exist, and non-numeric predictors.
"""

import datetime
import re
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._date_utils import daily_anchor, fmt_minutes, is_utc_index
from ..information_set import parse_predictor, rule_elements
from ._calendar import DAILY, FREQ_STEPS, clock_periods

DayRange = Tuple[pd.Timestamp, pd.Timestamp]

# Columns the model adds to its working frame; data columns with these names would be shadowed.
RESERVED_COLUMNS = ("hour", "day")

_FREQ_STEPS = FREQ_STEPS
_DAY_LAG_SUFFIX = re.compile(r"_d-\d+$")
# A coarser stretch: at least this many consecutive rows, none with a neighbour one `freq` step away, evenly
# spaced at one multiple of the step and covering at least this span (rows x spacing). Anything less is a gap.
_MIN_COARSE_ROWS = 3
_MIN_COARSE_SPAN = pd.Timedelta(days=1)


def period_index(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Period of the (local) day each row belongs to: the hour for "1h", the quarter-hour for "15min", 0 for "1D"."""
    if freq == "1h":
        return index.hour.to_numpy()
    if freq == "15min":
        return (index.hour * 4 + index.minute // 15).to_numpy()
    if freq == DAILY:
        return np.zeros(len(index), dtype=index.hour.to_numpy().dtype)
    raise ValueError(f"Unsupported freq '{freq}'. Supported: {list(_FREQ_STEPS)}")


def wall_clock(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Local wall-clock times as a naive index in nanoseconds."""
    return (index.tz_localize(None) if index.tz is not None else index).as_unit("ns")


def target_columns(target: str, horizon: int) -> List[str]:
    return [f"{target}_d+{h}" for h in range(1, horizon + 1)]


def read_days(
    test_start, test_end, training_window, horizon, reach: int = 0
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """The first and last local day to check for a run, or None when they can't be told: check the whole frame.

    For a test day D and horizon hz the worker reads the rows of days [D - training_window - hz, D], and a row on
    day D holds its target from day D + hz, so nothing outside [test_start - training_window - horizon,
    test_end + horizon] reaches a fit or an `actual`. The result is one day wider on each side, so an error in
    that reasoning ends in a refusal rather than in unchecked rows being read.

    `reach` is how many days earlier a span may start (`NewestTargets.reach`: a model with `newest_target="published"` whose
    newest known target is older than D-1).
    """
    if not (_is_int(training_window) and _is_int(horizon)):
        return None
    try:
        first = pd.Timestamp(datetime.date.fromisoformat(test_start))
        last = pd.Timestamp(datetime.date.fromisoformat(test_end))
        if last < first:
            return None
        return (
            first - pd.Timedelta(days=int(training_window) + int(horizon) + int(reach) + 1),
            last + pd.Timedelta(days=int(horizon) + 1),
        )
    except (TypeError, ValueError, OverflowError):  # the run reports dates it can't read
        return None


def read_ranges(
    periods: Iterable[Tuple[str, str]], training_window, horizon, reach: int = 0
) -> Optional[List[DayRange]]:
    """`read_days` of several periods of one run, merged where they touch; None if any of them can't be told."""
    ranges = []
    for start, end in sorted(periods):
        days = read_days(start, end, training_window, horizon, reach)
        if days is None:
            return None
        if ranges and days[0] <= ranges[-1][1] + pd.Timedelta(days=1):
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], days[1]))
        else:
            ranges.append(days)
    return ranges


def check_index(data: pd.DataFrame, freq: str, days: Union[DayRange, Sequence[DayRange], None] = None) -> None:
    """Raise unless the index is a DatetimeIndex with one row per (local day, period) at resolution `freq`.

    The only allowed repeat is the hour that occurs twice when clocks fall back (same wall-clock time,
    different UTC offset). Rows must be `freq` apart throughout, except across gaps (see `_check_spacing`):
    a frame that is 15-minute only in part would otherwise train some periods on far fewer days than others.

    `days` (see `read_days`, or several ranges of `read_ranges`) limits the spacing checks to those local days, both
    included, because rows no run can read can't harm it: hourly history before a switch to 15-minute data doesn't block
    a run inside the 15-minute part. Duplicates and slots with several rows are looked for in the whole frame.

    At `"1D"` the frame has at most one row per local date, every row at one wall-clock time that the clock has once on every
    date, gaps allowed (`_check_daily`, on the whole frame); finer data raises and says to take one time of day.
    """
    if days is not None and isinstance(days[0], pd.Timestamp):
        days = [days]
    if freq not in _FREQ_STEPS:
        raise ValueError(f"Unsupported freq '{freq}'. Supported: {list(_FREQ_STEPS)}")
    index = data.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"Data must have a DatetimeIndex, got {type(index).__name__}")

    duplicated = index[index.duplicated()]
    if len(duplicated):
        raise ValueError(
            f"Data index has {len(duplicated)} duplicate timestamp(s), first {duplicated[0]}. "
            "Each timestamp must appear once; drop or aggregate the duplicates before running the model."
        )
    if freq == DAILY:
        _check_daily(index)
        return
    if len(index) < 2:
        return

    step = _FREQ_STEPS[freq]
    ordered = index.sort_values()
    if days is not None:
        wall = wall_clock(ordered)
        readable = np.zeros(len(ordered), dtype=bool)
        for first, last in days:
            readable |= (wall >= first) & (wall < last + pd.Timedelta(days=1))
        ordered = ordered[readable]
    # Elapsed time for a tz-aware index (asi8 is UTC), so DST switches don't make the spacing irregular.
    spacings = np.diff(ordered.as_unit("ns").asi8)
    if len(spacings):
        values, counts = np.unique(spacings, return_counts=True)
        common = pd.Timedelta(int(values[np.argmax(counts)]), unit="ns")
        if common != step:
            raise ValueError(
                f"Data rows are most commonly {_fmt_step(common)} apart, but the model runs with freq='{freq}'. "
                f"Resample the data to {freq} (ResampleTransformer) or pass the matching freq "
                f"({_suggest_freq(common)}).{_checked_days(days)}"
            )

    local = wall_clock(index)
    frame = pd.DataFrame({"date": local.normalize().asi8, "period": period_index(index, freq), "wall": local.asi8})
    distinct = frame.groupby(["date", "period"], sort=True)["wall"].nunique()
    crowded = distinct[distinct > 1]
    if len(crowded):
        examples = ", ".join(f"{pd.Timestamp(d).date()} period {p}" for d, p in crowded.index[:3].tolist())
        raise ValueError(
            f"{len(crowded)} (local day, period) slot(s) have more than one row for freq='{freq}' "
            f"(e.g. {examples}). The data mixes resolutions or is finer than freq, so one period would mix several "
            f"times of day. Resample the data to {freq} first."
        )
    if len(spacings):
        _check_spacing(ordered, spacings, freq, days)


def _check_daily(index: pd.DatetimeIndex) -> None:
    """Raise unless `index` has at most one row per local date, every row at one wall-clock time, which the clock has once on
    every date from the first to the last. Gaps are allowed.

    The elapsed spacing checks can't serve a daily frame: its rows are 23, 24 or 25 hours apart around a switch. What a daily row
    stands for is its time of day (`_date_utils.daily_anchor`), so a time the clocks skip or repeat on some date would make that
    date's row another instant than the others', or two of them. The whole frame is checked, as slots with several rows are at
    finer frequencies.
    """
    if not len(index):
        return
    local = wall_clock(index)
    dates = local.normalize()
    if dates.has_duplicates:
        spacings = np.diff(index.sort_values().as_unit("ns").asi8)
        values, counts = np.unique(spacings, return_counts=True)
        common = pd.Timedelta(int(values[np.argmax(counts)]), unit="ns")
        if common < pd.Timedelta(hours=23):
            raise ValueError(
                f"Data rows are most commonly {_fmt_step(common)} apart, but freq='1D' takes one row per local date. Take "
                "one time of day from finer data, e.g. df.at_time('12:00') after the transformers, so that each row stands "
                f"for the issue time on its date, or run with the data's own freq ({_suggest_freq(common)})."
            )
        crowded = dates[dates.duplicated(keep=False)].unique()
        examples = ", ".join(str(day.date()) for day in crowded[:3])
        raise ValueError(
            f"{len(crowded)} local date(s) have more than one row (e.g. {examples}), but freq='1D' takes one row per local "
            "date. Keep one row per date, at one time of day, e.g. df.at_time('12:00')."
        )
    anchor = daily_anchor(index)
    if index.tz is None or is_utc_index(index):
        return
    first, last = dates.min(), dates.max()
    walls = pd.date_range(first, last, freq="D", unit="ns") + pd.Timedelta(minutes=anchor)
    once = walls.tz_localize(index.tz, ambiguous="NaT", nonexistent="NaT")
    odd = walls[once.isna()]
    if len(odd):
        shown = ", ".join(str(day.date()) for day in odd[:3]) + (" ..." if len(odd) > 3 else "")
        raise ValueError(
            f"The rows are at {fmt_minutes(anchor)}, which the clock of {index.tz} skips or repeats on {len(odd)} date(s) "
            f"from {first.date()} to {last.date()} ({shown}), so a daily row there would be another instant, or two. Stamp "
            "the rows at a time of day that exists once on every date, e.g. df.at_time('12:00')."
        )


def warn_about_a_midnight_stamp(
    index: pd.DatetimeIndex,
    freq: str,
    predictors: Iterable[str],
    information_set: Any = None,
    model_name: str = "Model",
    stacklevel: int = 4,
) -> None:
    """A `UserWarning` when the rows of a daily frame are stamped at midnight, or within the offset of a rule's zone from it, and
    a predictor counts hours from the stamp (an `h`, `min` or `s` shift). Such a frame may hold what is true at another time of
    day, such as noon, under a midnight stamp: then `load_actual_h-1` holds 11:00 of D, read as 23:00 of D-1, known, and neither
    the check nor a probe can tell (decision 11 of the step 3 spec). A frame cut with `at_time("00:00")` is right as it is, so
    this warns and refuses nothing. The rule zones are those of the hour-shifted predictors' rules in `information_set`."""
    if freq != DAILY or not len(index):
        return
    shifted = sorted({name for name in predictors if _counts_hours(name)})
    if not shifted:
        return
    anchor = daily_anchor(index)
    zones = set()
    if information_set is not None:
        for name in shifted:
            rule = information_set._resolve_for(name, 60).rule
            for one in rule_elements(rule) if rule is not None else ():
                if one.timezone is not None:
                    zones.add(one.timezone)
    slack = 0.0
    if index.tz is not None:
        for zone in sorted(zones):
            gap = (wall_clock(index.tz_convert(zone)) - wall_clock(index)) / pd.Timedelta(minutes=1)
            slack = max(slack, float(np.abs(np.asarray(gap)).max()))
    if min(anchor, 1440.0 - anchor) > slack:
        return
    shown = ", ".join(shifted[:3]) + (f" and {len(shifted) - 3} more" if len(shifted) > 3 else "")
    where = "midnight" if anchor == 0 else f"within {fmt_minutes(slack)} of midnight in the zone of a rule"
    warnings.warn(
        f"{model_name}: the daily rows are stamped at {fmt_minutes(anchor)}, {where}, and {len(shifted)} predictor(s) count "
        f"hours from the stamp ({shown}): `x_h-1` is the hour before {fmt_minutes(anchor)} on the row's date. If the rows hold "
        "what is true at another time of day, such as noon, restamp them there on the wall clock first: wall = "
        "df.index.tz_localize(None).normalize() + pd.Timedelta(hours=12); df = df.set_axis(wall.tz_localize(zone)). Else an "
        "hour known after the issue time reads as known, and no check can see it. A frame cut with at_time() at this time is "
        "right as it is. See the docs page Time and Calendars.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _counts_hours(name: Any) -> bool:
    """Whether a column name shifts its row by hours, minutes or seconds (`x_h-1`), counted from a daily row's stamp. An hour
    aggregate (`x_hour_mean`) isn't: on a daily row it is read over its whole date, whatever the stamp."""
    return isinstance(name, str) and any(shift.unit != "d" for shift in parse_predictor(name)[-1][1])


def _checked_days(days: Optional[Sequence[DayRange]]) -> str:
    if days is None:
        return ""
    ranges = " and ".join(f"{first.date()} to {last.date()}" for first, last in days)
    return f" Only the days this run can read are checked, {ranges}."


def _check_spacing(
    ordered: pd.DatetimeIndex,
    spacings: np.ndarray,
    freq: str,
    days: Optional[Sequence[DayRange]] = None,
) -> None:
    """Raise unless the rows are `freq` apart throughout, with gaps of whole steps in between.

    `spacings` are the elapsed nanoseconds between consecutive rows of the sorted index `ordered`. Gaps (missing
    days, periods or scattered rows, e.g. after `dropna()`) are allowed: they leave rows whose neighbours are
    missing, but irregularly. A coarser stretch is rows evenly spaced at one multiple of the step, none with a
    neighbour one step away, for at least a day (e.g. hourly rows before 15-minute ones): the periods with no
    row there would train on fewer days than the others.
    """
    step = _FREQ_STEPS[freq].value
    off_grid = np.flatnonzero(spacings % step)
    if len(off_grid):
        i = off_grid[0]
        raise ValueError(
            f"Rows {ordered[i]} and {ordered[i + 1]} are {_fmt_step(pd.Timedelta(int(spacings[i]), unit='ns'))} "
            f"apart, not a whole number of {freq} steps ({len(off_grid)} such spacing(s)). Some rows are off the "
            f"{freq} grid (e.g. a :30 row in hourly data), so a period would mix different times of day. Bring the "
            f"data to one {freq} grid first, e.g. ResampleTransformer(freq='{freq}').{_checked_days(days)}"
        )

    on_step = spacings == step
    lone = np.ones(len(ordered), dtype=bool)  # no neighbour one step away
    lone[:-1] &= ~on_step
    lone[1:] &= ~on_step
    # Spacing between two consecutive lone rows (0 elsewhere); a coarser stretch is a run of one repeated value.
    between = np.where(lone[:-1] & lone[1:], spacings, 0)
    starts = np.flatnonzero(np.concatenate(([True], between[1:] != between[:-1])))
    lengths = np.diff(np.append(starts, len(between)))
    stretches = [
        (int(s), int(s + n))  # first and last row
        for s, n in zip(starts, lengths)
        if between[s] and n + 1 >= _MIN_COARSE_ROWS and int(n + 1) * int(between[s]) >= _MIN_COARSE_SPAN.value
    ]
    if not stretches:
        return
    first, last = stretches[0]
    spacing = _fmt_step(pd.Timedelta(int(between[first]), unit="ns"))
    more = sum(stop - start + 1 for start, stop in stretches[1:])
    raise ValueError(
        f"{last - first + 1} rows from {ordered[first]} to {ordered[last]} are {spacing} apart with no row {freq} "
        f"away: the data is at a coarser resolution there than freq='{freq}'"
        + (f", and so are {more} rows in {len(stretches) - 1} later stretch(es)" if more else "")
        + ". The periods without rows there would train on fewer days than the others. Bring the whole frame to "
        f"{freq} first, e.g. ResampleTransformer(freq='{freq}') step-fills the coarser rows, or start the data "
        f"where it is at {freq}.{_checked_days(days)}"
        + (" A training window and test period that stay clear of these rows also work." if days else "")
    )


def check_timezone_with_day_lags(data: pd.DataFrame) -> None:
    """Raise if the data has a UTC index and already carries day-lag (`_d-<k>`) columns.

    UTC days are not local market days. `BaseModel._shift_index_by_calendar_days` only warns for a plain UTC
    index, because its own `{target}_d+{h}` shift is then at least internally consistent (every day is 24h
    UTC). A day-lag column built before the index lost its local time zone (e.g. `epf data` writing local
    time, `epf model` reading it back as UTC) is not: its values were computed against a different set of
    calendar-day boundaries than the ones the model now derives from this index, so they would silently
    disagree with the targets around DST. That is worse than the plain UTC case, so it raises instead of warns.
    """
    if not is_utc_index(data.index):
        return
    day_lag_columns = [c for c in data.columns if isinstance(c, str) and _DAY_LAG_SUFFIX.search(c)]
    if day_lag_columns:
        shown = ", ".join(day_lag_columns[:5]) + (", ..." if len(day_lag_columns) > 5 else "")
        raise ValueError(
            f"Data has a UTC index and day-lag column(s) {shown} (built from '_d-<k>' suffixes). UTC days are "
            "not local market days, so a day lag computed before the index lost its time zone would not agree "
            "with the model's own calendar-day target shift. Keep the market's local time zone all the way "
            "through (see epftoolbox2._frame_csv for the epf data -> epf model round trip) instead of "
            "converting to UTC."
        )


def check_names(data: pd.DataFrame, target: str, horizon: int) -> None:
    """Raise if the data uses a name the model reserves for its own columns."""
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' is not in the data")
    if target in RESERVED_COLUMNS:
        raise ValueError(f"Target column '{target}' clashes with the model's internal '{target}' id; rename it")

    reserved = [c for c in RESERVED_COLUMNS if c in data.columns]
    if reserved:
        names = ", ".join(f"'{c}'" for c in reserved)
        raise ValueError(
            f"Data column(s) {names} clash with the model's internal period/day ids. Rename them "
            "(for example CalendarSource(prefix='cal_') gives 'cal_hour') or drop them before running the model."
        )

    pattern = re.compile(rf"^{re.escape(target)}_d\+(\d+)$")
    collisions = []
    for column in data.columns:
        match = pattern.match(column) if isinstance(column, str) else None
        if match and 1 <= int(match[1]) <= horizon:
            collisions.append(column)
    if collisions:
        raise ValueError(
            f"Data already has column(s) {collisions}. Names '{target}_d+<h>' are the model's targets "
            f"(built from '{target}' by a calendar-day shift), so existing columns would silently replace or shift "
            "them. Drop or rename them; a LagTransformer with negative lags on the target creates them."
        )

    duplicated = data.columns[data.columns.duplicated()]
    if len(duplicated):
        raise ValueError(f"Data has duplicate column names: {sorted(set(duplicated))}")


def check_predictors(
    data: pd.DataFrame,
    expanded: Dict[Tuple[int, int], List[str]],
    target: str,
    horizon: int,
) -> List[str]:
    """Raise if any expanded predictor is a model target, reserved, missing or non-numeric.

    Returns the sorted list of distinct predictor columns.
    """
    targets = set(target_columns(target, horizon))
    used = sorted({p for preds in expanded.values() for p in preds}, key=str)

    as_target = [p for p in used if p in targets]
    if as_target:
        raise ValueError(
            f"Predictor(s) {as_target} name the model's own targets (the future value of '{target}'); "
            "using them as inputs is look-ahead."
        )
    reserved = [p for p in used if p in RESERVED_COLUMNS]
    if reserved:
        raise ValueError(f"Predictor(s) {reserved} are reserved names for the model's internal ids")

    missing = [p for p in used if p not in data.columns]
    if missing:
        where = _where(expanded, missing)
        raise ValueError(
            f"Predictor column(s) not in the data: {missing} (needed for {where}). "
            f"Available columns: {_preview(data.columns)}"
        )

    non_numeric = [p for p in used if not _is_numeric(data[p])]
    if not _is_numeric(data[target]):
        non_numeric.append(target)
    if non_numeric:
        dtypes = ", ".join(f"'{c}' ({data[c].dtype})" for c in non_numeric)
        raise ValueError(f"Predictors and the target must be numeric: {dtypes}")
    return used


def missing_test_rows(index: pd.DatetimeIndex, freq: str, days: Iterable) -> List[Tuple[int, pd.Timestamp]]:
    """(period, local day) slots on the given days that have no row although that wall-clock time exists.

    A slot the clock doesn't have, because clocks spring forward, is not listed: it is no gap in the data, and
    `IssueCalendar` doesn't plan a forecast from it.
    """
    local = wall_clock(index)
    present = set(zip(local.normalize().asi8.tolist(), period_index(index, freq).tolist()))
    step = _FREQ_STEPS[freq]
    periods = int(pd.Timedelta(days=1) / step)
    absent = []
    for day in days:
        day = pd.Timestamp(day).normalize()
        absent.extend((p, day) for p in range(periods) if (day.value, p) not in present)
    if not absent or index.tz is None:
        return absent
    on_the_clock = {day: clock_periods(day, freq, index.tz) for day in {day for _, day in absent}}
    return [(period, day) for period, day in absent if period in on_the_clock[day]]


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)


def _where(expanded: Dict[Tuple[int, int], List[str]], columns: Sequence[str]) -> str:
    keys = sorted(k for k, preds in expanded.items() if any(c in preds for c in columns))
    horizons = sorted({hz for _, hz in keys})
    periods = sorted({p for p, _ in keys})
    return f"horizon(s) {horizons}, period(s) {_ranges(periods)}"


def _ranges(values: Sequence[int]) -> str:
    if not values:
        return "[]"
    parts, start, prev = [], values[0], values[0]
    for v in list(values[1:]) + [None]:
        if v is not None and v == prev + 1:
            prev = v
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        if v is not None:
            start = prev = v
    return ", ".join(parts)


def _preview(columns: Sequence[str], limit: int = 20) -> str:
    cols = [str(c) for c in columns]
    return ", ".join(cols[:limit]) + (f", ... ({len(cols)} in total)" if len(cols) > limit else "")


def _fmt_step(step: pd.Timedelta) -> str:
    seconds = int(step.total_seconds())
    for unit, size in (("day", 86400), ("h", 3600), ("min", 60)):
        if seconds % size == 0 and seconds >= size:
            n = seconds // size
            return f"{n} {unit}{'s' if unit == 'day' and n > 1 else ''}"
    return str(step)


def _suggest_freq(step: pd.Timedelta) -> str:
    for name, value in _FREQ_STEPS.items():
        if value == step:
            return f"freq='{name}'"
    return f"supported: {list(_FREQ_STEPS)}"
