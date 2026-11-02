"""One place for dates: which rows a forecast trains on, and which forecasts a run plans.

A forecast is a key (period, horizon h, issue day D). Everything here is about local calendar dates, because a period of a day
isn't always one row: the hour clocks repeat in autumn has two rows on its date, and the hour they skip in spring has none. So
rows are chosen by their date, never by counting them, and a key is planned only if the clock has its period on D and on D + h.

A period is on the clock of a date when any minute of it is: Lord Howe skips 02:00 to 02:29 in spring, so at 1 h its period 2 stays,
with its row at 02:30, and at 15 min periods 8 and 9 go.

Day ids are integers, one per local date, as `BaseModel._preprocess` numbers them. `zone` is what a frame's `index.tz` holds, or
the name of a zone, or None for a frame with no clock of its own (a naive index): such a frame, like a UTC one, has every period
on every date.
"""

import collections.abc
import datetime
import functools
from typing import Any, Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .._date_utils import zone_name

FREQ_STEPS = {"1h": pd.Timedelta(hours=1), "15min": pd.Timedelta(minutes=15)}


def periods_per_day(freq: str) -> int:
    """How many periods an ordinary day has at `freq`."""
    if freq not in FREQ_STEPS:
        raise ValueError(f"Unsupported freq '{freq}'. Supported: {list(FREQ_STEPS)}")
    return int(pd.Timedelta(days=1) / FREQ_STEPS[freq])


def training_span(anchor: Any, window: int, hz: int, newest: Any = -1) -> Tuple[Any, Any]:
    """The first and last row date a forecast from day id `anchor` at horizon `hz` may train on, both included.

    `newest` is the day of the newest training target relative to `anchor`, so with the default of -1 the last row is
    `anchor - 1 - hz`, whose target is dated `anchor - 1`, and the span holds `window` dates. No row is dated `anchor` or later,
    whatever `newest` says. `anchor` and `newest` may be arrays of day ids.
    """
    last = np.minimum(anchor + newest - hz, anchor - 1)
    return last - window + 1, last


def resolve_horizons(horizon: Any, horizons: Any, default: int) -> Tuple[Any, Tuple[int, ...]]:
    """(the largest horizon, every horizon planned, in order) of a run told `horizon=H`, which means 1 to H, or `horizons=`.

    `horizons` is any sequence of whole numbers from 1, such as `[1, 3, 7]` or `range(3, 10)`: it is sorted, and an empty one or
    one with a repeat raises. Giving both raises; giving neither is `horizon=default`. `range(1, H + 1)` is then exactly
    `horizon=H`. Whatever the list, targets are built for 1 to its largest, so `{target}_d+k` up to it stays refused as an input.
    """
    if horizons is None:
        horizon = default if horizon is None else horizon
        if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError(f"horizon must be a whole number of days from 1, got {horizon!r}")
        return horizon, tuple(range(1, horizon + 1))
    if horizon is not None:
        raise ValueError(
            f"Pass horizon or horizons, not both: horizon={horizon!r} means every horizon from 1 to it, horizons={horizons!r} "
            "the ones it lists."
        )
    if isinstance(horizons, (str, bytes)) or not isinstance(horizons, collections.abc.Iterable):
        raise ValueError(
            f"horizons must be a sequence of whole numbers of days, such as [1, 3, 7] or range(3, 10), got {horizons!r}; "
            "horizon=H is every horizon from 1 to H"
        )
    listed = list(horizons)
    bad = [h for h in listed if isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer)) or h < 1]
    if bad:
        raise ValueError(f"horizons are whole numbers of days from 1, got {bad[:3]} in {listed}")
    if not listed:
        raise ValueError("horizons is empty: a run needs at least one horizon")
    found = [int(h) for h in listed]
    repeated = sorted({h for h in found if found.count(h) > 1})
    if repeated:
        raise ValueError(f"horizons lists {repeated} more than once: {found}")
    found.sort()
    return found[-1], tuple(found)


def every_horizon(horizon: Any, horizons: Sequence[int]) -> bool:
    """Whether `horizons` is every horizon from 1 to `horizon`, the run `horizon=H` plans."""
    return tuple(horizons) == tuple(range(1, horizon + 1))


def horizon_arguments(horizon: Any, horizons: Sequence[int]) -> Dict[str, Any]:
    """What to pass on to a `run(...)`: `horizon=H` for every horizon from 1 to H, as before `horizons` existed, else `horizons`."""
    return {"horizon": horizon} if every_horizon(horizon, horizons) else {"horizons": tuple(horizons)}


NEWEST_TARGETS = ("d-1", "published")


def check_newest_target(value: Any) -> str:
    """`value` if it is one of `NEWEST_TARGETS`, else a ValueError."""
    if not isinstance(value, str) or value not in NEWEST_TARGETS:
        raise ValueError(
            f"newest_target must be one of {list(NEWEST_TARGETS)}, got {value!r}: 'd-1' trains on targets up to the day "
            "before the issue day, 'published' on every target its rule says is known by the issue time"
        )
    return value


class NewestTargets:
    """The newest training target of each forecast of a model with `newest_target="published"`.

    `k[period, i]` is the day of that target relative to the issue day `days[i]`, as `InformationSet.newest_targets` reads it
    from the target's rule, and the forecast (period, h, D) trains on the rows of `training_span(D, W, h, k)`. Days are local
    dates, looked up by their ordinal: day ids depend on where a frame starts, dates don't.
    """

    def __init__(self, days: Sequence[Any], k: np.ndarray):
        self.days = [_as_date(day) for day in days]
        self.k = np.asarray(k, dtype=np.int64)
        ordinals = np.array([day.toordinal() for day in self.days], dtype=np.int64)
        self._order = np.argsort(ordinals, kind="stable")
        self._ordinals = ordinals[self._order]

    @property
    def reach(self) -> int:
        """How many days before the first row of `training_span(D, W, h)` a span may start: more than none only where a target
        older than D-1 is the newest known."""
        return max(0, -1 - int(self.k.min())) if self.k.size else 0

    def at(self, period: int, ordinals: Any) -> Any:
        """k of `period` on these issue days, given by their date ordinals. A day that isn't one of `days` raises KeyError."""
        wanted = np.asarray(ordinals, dtype=np.int64)
        place = np.clip(np.searchsorted(self._ordinals, wanted), 0, max(len(self._ordinals) - 1, 0))
        if not len(self._ordinals) or (self._ordinals[place] != wanted).any():
            raise KeyError(f"no newest target was read for issue day ordinal(s) {wanted}")
        found = self.k[period, self._order[place]]
        return int(found) if np.ndim(found) == 0 else found

    def by_position(self) -> Dict[int, np.ndarray]:
        """period -> k on each of `days`, in their order: what a worker reads by a task's place among the issue days."""
        return {period: self.k[period].copy() for period in range(self.k.shape[0])}


def _as_date(day: Any) -> datetime.date:
    return pd.Timestamp(day).date()


_DAY = pd.Timedelta(days=1).value
_MINUTE = pd.Timedelta(minutes=1).value


def _clock_gaps(
    first: datetime.date, last: datetime.date, freq: str, zone: Any
) -> Dict[datetime.date, Tuple[int, ...]]:
    """The periods the clock doesn't have on each date from `first` to `last`, for the dates that lack any: those with no minute
    on the clock."""
    if zone is None or last < first:
        return {}
    days = (last - first).days + 1
    periods = periods_per_day(freq)
    minutes_per_period = 1440 // periods
    midnights = pd.date_range(first, periods=days + 1, freq="D", unit="ns")
    # Only a date whose clock changes lasts other than 24 hours from its midnight to the next, so only those are read minute by
    # minute. A midnight the clock skips is read as the first minute after it, a repeated one as its first occurrence.
    starts = midnights.tz_localize(zone, nonexistent="shift_forward", ambiguous=np.ones(days + 1, dtype=bool))
    changing = np.flatnonzero(np.diff(starts.as_unit("ns").asi8) != _DAY)
    gaps: Dict[datetime.date, Tuple[int, ...]] = {}
    for day in changing.tolist():
        minutes = midnights.asi8[day] + np.arange(1440, dtype=np.int64) * _MINUTE
        wall = pd.DatetimeIndex(minutes.astype("datetime64[ns]"))
        exists = ~wall.tz_localize(zone, nonexistent="NaT", ambiguous=np.ones(1440, dtype=bool)).isna()
        skipped = np.flatnonzero(~exists.reshape(periods, minutes_per_period).any(axis=1))
        if len(skipped):
            gaps[first + datetime.timedelta(days=day)] = tuple(skipped.tolist())
    return gaps


def clock_periods(day: Any, freq: str, zone: Any) -> Tuple[int, ...]:
    """The period ids that exist on a local date, in order: the hour skipped in spring has none, and one repeated in autumn has one."""
    date = _as_date(day)
    skipped = _clock_gaps(date, date, freq, zone).get(date, ())
    return tuple(period for period in range(periods_per_day(freq)) if period not in skipped)


def skipped_periods(day: Any, freq: str, zone: Any) -> FrozenSet[int]:
    """The period ids a local date doesn't have on the clock: what `clock_periods` leaves out, for a look-up per record."""
    try:
        hash(zone)
    except TypeError:  # dateutil's zones, which are read by their name
        zone = zone_name(zone) or zone
        try:
            hash(zone)
        except TypeError:
            return _skipped_periods.__wrapped__(day, freq, zone)
    return _skipped_periods(day, freq, zone)


@functools.lru_cache(maxsize=4096)
def _skipped_periods(day: Any, freq: str, zone: Any) -> FrozenSet[int]:
    date = _as_date(day)
    return frozenset(_clock_gaps(date, date, freq, zone).get(date, ()))


def clock_periods_on(days: Iterable[Any], freq: str, zone: Any) -> Dict[pd.Timestamp, Tuple[int, ...]]:
    """`clock_periods` of each of these local dates, keyed by the date as a timestamp."""
    dates = pd.DatetimeIndex(pd.to_datetime(list(days))).unique().sort_values()
    if not len(dates):
        return {}
    gaps = _clock_gaps(dates[0].date(), dates[-1].date(), freq, zone)
    every = range(periods_per_day(freq))
    return {date: tuple(p for p in every if p not in gaps.get(date.date(), ())) for date in dates}


class IssueCalendar:
    """Which forecasts a run plans: every key whose period the clock has on its issue day and on its target day.

    A slot the clock doesn't have is no missing forecast: nothing could be issued from it, or scored against it. A real gap in the
    data is another matter: the slot is planned, and the run counts it as `missing_test_row` or the record has no `actual`.

    This is the one count of planned forecasts, so that `BaseModel.run`, `ModelPipeline.report` and the ensembles agree per window.
    `report` builds it from a manifest's `data_timezone`, with no data at hand. `zone` is as in the module docstring.
    """

    def __init__(self, freq: str, zone: Any, horizons: Iterable[int]):
        self.freq = freq
        self.zone = zone
        self.horizons = tuple(horizons)
        self._periods = periods_per_day(freq)

    def issue_days(self, start: Any, end: Any) -> List[datetime.date]:
        """The issue days from `start` to `end`, both included."""
        first, last = _as_date(start), _as_date(end)
        return [first + datetime.timedelta(days=day) for day in range((last - first).days + 1)]

    def _gaps(self, first: datetime.date, last: datetime.date) -> Dict[datetime.date, Tuple[int, ...]]:
        """The clock's gaps on these issue days and the days their forecasts are for."""
        return _clock_gaps(first, last + datetime.timedelta(days=max(self.horizons, default=0)), self.freq, self.zone)

    @staticmethod
    def _gone(gaps: Dict[datetime.date, Tuple[int, ...]], day: datetime.date, horizon: int) -> Set[int]:
        return {*gaps.get(day, ()), *gaps.get(day + datetime.timedelta(days=horizon), ())}

    def tasks(self, first_day: datetime.date, days: Sequence[int]) -> List[Tuple[int, int, int]]:
        """The planned `(period, horizon, day)` of these days, each a count of days from `first_day`, by day, then horizon, then
        period."""
        if not len(days):
            return []
        gaps = self._gaps(
            first_day + datetime.timedelta(days=min(days)), first_day + datetime.timedelta(days=max(days))
        )
        tasks = []
        for day in days:
            date = first_day + datetime.timedelta(days=day)
            for horizon in self.horizons:
                gone = self._gone(gaps, date, horizon)
                tasks.extend((period, horizon, day) for period in range(self._periods) if period not in gone)
        return tasks

    def count(self, start: Any, end: Any) -> int:
        """How many forecasts are planned on the issue days from `start` to `end`."""
        days = self.issue_days(start, end)
        if not days:
            return 0
        whole = len(days) * len(self.horizons) * self._periods
        gaps = self._gaps(days[0], days[-1])
        if not gaps:
            return whole
        return whole - sum(len(self._gone(gaps, day, horizon)) for day in days for horizon in self.horizons)

    def planned_cells(self, days: Sequence[datetime.date]) -> Dict[int, np.ndarray]:
        """horizon -> whether the key of each period and issue day is planned, as [period, position in days]."""
        grids = {horizon: np.ones((self._periods, len(days)), dtype=bool) for horizon in self.horizons}
        for period, horizon, position in self.unplanned(days):
            grids[horizon][period, position] = False
        return grids

    def unplanned(self, days: Sequence[datetime.date]) -> List[Tuple[int, int, int]]:
        """`(period, horizon, position in days)` of the keys on these issue days that the clock doesn't have."""
        if not len(days):
            return []
        gaps = self._gaps(min(days), max(days))
        if not gaps:
            return []
        return [
            (period, horizon, position)
            for position, day in enumerate(days)
            for horizon in self.horizons
            for period in sorted(self._gone(gaps, day, horizon))
        ]
