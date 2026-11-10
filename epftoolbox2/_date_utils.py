import re
from datetime import date, datetime, timedelta
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from ._clock import today

_DATE_PATTERN = re.compile(r"^(today|now)(?:_d([+-]\d+))?$", re.IGNORECASE)


def resolve_date(s: Any, tz: Optional[str] = None) -> Any:
    """Resolve relative date keywords to ISO date strings.

    Supported patterns:
        'today', 'now'          → today's date
        'today_d+7', 'now_d+7'  → today + 7 days
        'today_d-365'           → today - 365 days
        '2024-01-01'            → passed through unchanged
        date(2024, 1, 1)        → '2024-01-01' (what YAML makes of an unquoted 2024-01-01)

    The offset N in _d±N accepts any integer, so training windows of any
    length can be expressed (e.g. 'now_d-800' for an 800-day history).

    "Today" is a date of `tz` at the current clock's now (`epftoolbox2.clock`), or the host's local date when `tz` is None:
    at 23:30 UTC on 2026-10-21, "today" is 2026-10-22 in "Europe/Warsaw" and 2026-10-21 in "UTC".
    """
    return resolve_dates([s], tz)[0]


def resolve_dates(values: Any, tz: Optional[str] = None, at: Optional[pd.Timestamp] = None) -> List[Any]:
    """Each of `values` as `resolve_date` resolves it, "today" read from the clock once for all of them (or at the instant
    `at`): a period whose ends are both keywords ("today", "today") is one day even when the clock passes midnight between
    them."""
    day: Optional[date] = None
    out: List[Any] = []
    for s in values:
        if isinstance(s, date) and not isinstance(s, datetime):
            out.append(s.isoformat())
            continue
        m = _DATE_PATTERN.match(s.strip()) if isinstance(s, str) else None
        if m is None:
            out.append(s)
            continue
        if day is None:
            day = today(tz, at)
        delta = int(m.group(2)) if m.group(2) else 0
        out.append((day + timedelta(days=delta)).isoformat())
    return out


def resolve_span(
    start: Any, end: Any, tz: Optional[str] = None, at: Optional[pd.Timestamp] = None, where: str = ""
) -> List[Any]:
    """`forecast_only`'s first and last issue day: `start` and `end` ("today" when None) as `resolve_dates` resolves them in
    `tz`, and a `ValueError` when they are dates and the first is after the last, as `to_period` refuses a reversed period
    (`where` names the zone in the message). Without the check such a run plans no forecast and says nothing."""
    first, last = resolve_dates([start or "today", end or "today"], tz, at)
    try:
        reversed_ = date.fromisoformat(str(first)[:10]) > date.fromisoformat(str(last)[:10])
    except ValueError:
        return [first, last]  # not dates: what reads them says so
    if reversed_:
        zone = f" in {tz}" if tz else ""
        raise ValueError(f"{where}The test period is reversed{zone}: it starts on {first} and ends on {last}")
    return [first, last]


def epoch_ns(index: pd.DatetimeIndex) -> np.ndarray:
    """The instants of `index` as int64 nanoseconds since the epoch (`NaT` is the smallest int64).

    `DatetimeIndex.asi8` counts in the index's own resolution. That is nanoseconds up to pandas 2, but
    microseconds for what pandas 3 creates by default (`date_range`, `to_datetime`, `Timestamp` strings), so
    arithmetic against nanosecond constants or `pd.Timestamp(int)` (always nanoseconds) must go through this.
    """
    return index.as_unit("ns").asi8


_UTC_NAMES = {"UTC", "UCT", "GMT", "UNIVERSAL", "ZULU", "GREENWICH", "TZUTC()"}


def is_utc_index(index: pd.DatetimeIndex) -> bool:
    """True for a UTC (or GMT) index, where days are UTC days rather than local market days."""
    return index.tz is not None and str(index.tz).upper().removeprefix("ETC/") in _UTC_NAMES


def zone_name(tz: Any) -> Optional[str]:
    """A name that reads back as this time zone, as a manifest's `data_timezone` is read, or None if it has none.

    The IANA key of a zoneinfo, pytz or dateutil zone (dateutil's from the path of its file after `zoneinfo/`), "UTC" for any UTC,
    and `str(tz)` when that reads back, such as a fixed offset's "UTC+02:00". `str` of a dateutil zone is
    "tzfile('/usr/share/zoneinfo/Europe/Warsaw')", which doesn't.
    """
    if tz is None:
        return None
    candidates = [getattr(tz, "key", None), getattr(tz, "zone", None)]  # zoneinfo, pytz
    filename = getattr(tz, "_filename", None)  # dateutil
    if isinstance(filename, str):
        candidates += [filename.split("zoneinfo/", 1)[1] if "zoneinfo/" in filename else None, filename]
    text = str(tz)
    candidates.append("UTC" if text.upper().removeprefix("ETC/") in _UTC_NAMES else text)
    for name in candidates:
        if isinstance(name, str) and name and _reads_back(name):
            return name
    return None


def wall_minutes(index: pd.DatetimeIndex) -> np.ndarray:
    """Each row's local wall-clock time of day, in minutes after its midnight (seconds as fractions of a minute)."""
    local = index.tz_localize(None) if index.tz is not None else index
    return np.asarray(local.hour * 60 + local.minute + local.second / 60.0, dtype=float)


def fmt_minutes(minutes: float) -> str:
    """Minutes after midnight as `HH:MM`, or `HH:MM:SS` when there are seconds."""
    seconds = int(round(float(minutes) * 60))
    hours, rest = divmod(seconds, 3600)
    text = f"{hours:02d}:{rest // 60:02d}"
    return text + (f":{rest % 60:02d}" if rest % 60 else "")


def daily_anchor(index: pd.DatetimeIndex) -> float:
    """The wall-clock time of day every row of a daily frame (`freq="1D"`) is stamped at, in minutes after local midnight.

    A daily row stands for its issue instant on its date: noon in every example, so that `load_actual_h-1` is 11:00 of that
    date. Its time is read from the index, and a `ValueError` says so if the rows are at several times or there is none.
    """
    if not len(index):
        raise ValueError(
            "freq='1D' reads the time of day a daily row stands for from its index, and the index is empty"
        )
    times = np.unique(wall_minutes(index))
    if len(times) > 1:
        shown = ", ".join(fmt_minutes(t) for t in times[:3]) + (", ..." if len(times) > 3 else "")
        raise ValueError(
            f"freq='1D' needs every row at one wall-clock time of day, the time its row stands for, but the rows are at "
            f"{len(times)} times ({shown}). Take one time of day from data at a finer resolution, e.g. "
            "df.at_time('12:00') after the transformers, or stamp every row at the same time."
        )
    return float(times[0])


def _reads_back(name: str) -> bool:
    try:
        pd.Timestamp("2000-01-01").tz_localize(name)
    except Exception:
        return False
    return True


def _occurrences(wall: pd.DatetimeIndex, tz: Any) -> tuple:
    """Earlier and later UTC instant of each local wall-clock time.

    Both are equal where the time occurs once and NaT where it doesn't exist (the hour skipped in spring).
    They differ only on the hour repeated in autumn. Ordering by instant rather than by pandas' DST flag
    keeps "first" and "second" independent of how a zone's tz data defines DST.
    """
    n = len(wall)
    a = wall.tz_localize(tz, ambiguous=np.ones(n, dtype=bool), nonexistent="NaT")
    b = wall.tz_localize(tz, ambiguous=np.zeros(n, dtype=bool), nonexistent="NaT")
    b_first = b.asi8 < a.asi8
    return a.where(~b_first, b), b.where(~b_first, a)


def shift_calendar_days(index: pd.DatetimeIndex, days: int, ambiguous: str = "first") -> pd.DatetimeIndex:
    """Each timestamp moved to the same local wall-clock time `days` calendar days later (earlier if negative).

    `index + pd.Timedelta(days=days)` moves by elapsed time, so on a local-time index it drifts by an hour
    across a DST switch. This keeps the wall-clock time instead, which is what "the same hour yesterday"
    means when a local day has 23 or 25 hours:

    - a time that doesn't exist on the target date (the hour skipped in spring) gives NaT;
    - on the hour repeated in autumn, the second occurrence maps to the second occurrence when the target
      date repeats that hour too, and otherwise to the target's only occurrence;
    - a time that occurs once but lands on a repeated hour maps to its first occurrence, or to NaT with
      `ambiguous="NaT"`.

    A naive index is taken as wall-clock time without DST and a UTC index as UTC days; both shift by whole
    24-hour days. Look values up with `series.reindex(shift_calendar_days(series.index, -k))` to get the
    value k calendar days earlier (NaN where the target is NaT or absent).
    """
    if ambiguous not in ("first", "NaT"):
        raise ValueError(f"ambiguous must be 'first' or 'NaT', got {ambiguous!r}")
    offset = pd.Timedelta(days=int(days))
    if index.tz is None:
        return index + offset

    wall = index.tz_localize(None)
    src_first, src_second = _occurrences(wall, index.tz)
    src_repeated = src_first.asi8 != src_second.asi8
    src_is_second = src_repeated & (index.asi8 == src_second.asi8)

    tgt_first, tgt_second = _occurrences(wall + offset, index.tz)
    tgt_repeated = tgt_first.asi8 != tgt_second.asi8

    shifted = tgt_first.where(~(src_is_second & tgt_repeated), tgt_second)
    if ambiguous == "NaT":
        shifted = shifted.where(~(tgt_repeated & ~src_repeated))
    return shifted
