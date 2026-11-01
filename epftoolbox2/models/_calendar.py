"""One place for dates: which rows a forecast trains on, and which forecasts a run plans.

A forecast is a key (period, horizon h, issue day D). Everything here is about local calendar dates, because a period of a day
isn't always one row: the hour clocks repeat in autumn has two rows on its date, and the hour they skip in spring has none. So
rows are chosen by their date, never by counting them, and a key is planned only if the clock has its period on D and on D + h.

Day ids are integers, one per local date, as `BaseModel._preprocess` numbers them. `zone` is what a frame's `index.tz` holds, or
the name of a zone, or None for a frame with no clock of its own (a naive index): such a frame, like a UTC one, has every period
on every date.
"""

import datetime
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

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


def _as_date(day: Any) -> datetime.date:
    return pd.Timestamp(day).date()


def _clock_gaps(
    first: datetime.date, last: datetime.date, freq: str, zone: Any
) -> Dict[datetime.date, Tuple[int, ...]]:
    """The periods the clock skips on each date from `first` to `last`, for the dates that skip any."""
    if zone is None or last < first:
        return {}
    days = (last - first).days + 1
    periods = periods_per_day(freq)
    midnights = pd.date_range(first, periods=days, freq="D", unit="ns").asi8
    offsets = np.arange(periods, dtype=np.int64) * FREQ_STEPS[freq].value
    wall = pd.DatetimeIndex((midnights[:, np.newaxis] + offsets).ravel().astype("datetime64[ns]"))
    exists = ~wall.tz_localize(zone, nonexistent="NaT", ambiguous=np.ones(len(wall), dtype=bool)).isna()
    gaps: Dict[datetime.date, List[int]] = {}
    for slot in np.flatnonzero(~exists).tolist():
        gaps.setdefault(first + datetime.timedelta(days=slot // periods), []).append(slot % periods)
    return {day: tuple(skipped) for day, skipped in gaps.items()}


def clock_periods(day: Any, freq: str, zone: Any) -> Tuple[int, ...]:
    """The period ids that exist on a local date, in order: the hour skipped in spring has none, and one repeated in autumn has one."""
    date = _as_date(day)
    skipped = _clock_gaps(date, date, freq, zone).get(date, ())
    return tuple(period for period in range(periods_per_day(freq)) if period not in skipped)


def periods_on_dates(days: Iterable[Any], freq: str, zone: Any) -> pd.Series:
    """How many period ids exist on each of these local dates (`clock_periods`), indexed by the date as a timestamp."""
    dates = pd.DatetimeIndex(pd.to_datetime(list(days))).unique().sort_values()
    if not len(dates):
        return pd.Series(dtype=np.int64, index=dates)
    gaps = _clock_gaps(dates[0].date(), dates[-1].date(), freq, zone)
    whole = periods_per_day(freq)
    return pd.Series([whole - len(gaps.get(date.date(), ())) for date in dates], index=dates, dtype=np.int64)


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
