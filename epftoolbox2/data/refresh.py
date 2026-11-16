"""What a `DataPipeline.run` did, source by source: `DataPipeline.last_refresh` (step 8b).

`run(start, end, cache, *, budget=, deadline=, timeout=, retries=, on_failure=)` refreshes the sources one after another,
each within what is left of the time (`budget` seconds from the start, `deadline` an instant, the earlier of the two), and
records each source's part in a `SourceRefresh`: its status, the time it took on the clock (`epftoolbox2.clock`), the HTTP
requests it sent and how many asked again, the error it raised (every secret of the pipeline's sources replaced by
`***`), the rows it gave the frame and the last timestamp with a value of each of its columns.

A source that isn't ok gets, with `on_failure="cache"`, only what its per-source cache holds: settled rows, never the last
two days before now (`UNSETTLED_PERIOD`) or a tail the source declared unsettled, so its recent cells are missing and the
models decide what that means. Nothing is filled with an older value.
"""

import math
import numbers
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, quote_plus

import pandas as pd

from epftoolbox2._clock import Clock
from epftoolbox2.data.sources._http import SourceTimeout

__all__ = ["RefreshError", "RefreshRecord", "SourceRefresh", "SourceTimeout", "STATUSES", "ON_FAILURE"]

# What happened to a source: fetched (or served by its cache without a request), its fetch raised, its fetch ran out of the
# refresh's time (`SourceTimeout`) after sending requests, or before it could send one (a deadline already past, a cache
# lock another process held for longer than the time left).
STATUSES = ("ok", "failed", "timeout", "skipped")
# What a source that isn't ok gives: raise its error (today's behaviour), or its cached settled rows.
ON_FAILURE = ("raise", "cache")

REDACTED = "***"

# A deadline given as a time of day alone: "11:55" or "11:55:30".
_TIME_OF_DAY = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$")
# A string with a full date: a four-digit year next to a month and a day.
_FULL_DATE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[./]\d{1,2}[./]\d{4}|\d{4}\d{2}\d{2}")


class RefreshError(RuntimeError):
    """`DataPipeline.run(..., on_failure="cache")` had no row to give: every source that was asked failed or ran out of
    time and no cache held a row. The message names each source and what it raised."""


def _iso(instant: Optional[pd.Timestamp]) -> Optional[str]:
    return None if instant is None else instant.isoformat()


@dataclass(frozen=True)
class SourceRefresh:
    """One source's part of a refresh.

    Attributes:
        name: `"<class> #<position>"`, the pipeline's name for it in its warnings (`"EntsoeSource #1"`).
        status: `"ok"` (fetched, or served by its cache without a request), `"failed"` (its fetch raised), `"timeout"` (it
            ran out of the refresh's time after sending requests: `SourceTimeout`) or `"skipped"` (it ran out of time
            before it could send one: the deadline had passed, or another process held its cache lock for longer).
        seconds: The time its turn took, on the clock.
        requests: HTTP requests it sent; a source that reads files sends none.
        retries: How many of those asked again after a failed answer.
        error: `"<class>: <message>"` of what it raised, with every secret of the pipeline's sources replaced by `***`.
        rows: The rows it gave the frame in `[start, end)`; for a source that isn't ok (with `on_failure="cache"`) its
            cached settled rows. None when unknown: a hit of the single-file cache reads no source.
        last_valid: For each column it gave, the last timestamp (UTC) holding a value, None for a column without one. None
            when unknown, as `rows`.
    """

    name: str
    status: str
    seconds: float = 0.0
    requests: int = 0
    retries: int = 0
    error: Optional[str] = None
    rows: Optional[int] = None
    last_valid: Optional[Dict[str, Optional[pd.Timestamp]]] = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe: timestamps as ISO 8601 strings."""
        return {
            "name": self.name,
            "status": self.status,
            "seconds": self.seconds,
            "requests": self.requests,
            "retries": self.retries,
            "error": self.error,
            "rows": self.rows,
            "last_valid": None
            if self.last_valid is None
            else {str(column): _iso(instant) for column, instant in self.last_valid.items()},
        }


@dataclass(frozen=True)
class RefreshRecord:
    """What a `DataPipeline.run` did, source by source (`DataPipeline.last_refresh`).

    Attributes:
        start, end: The run's range `[start, end)`, UTC.
        started: The clock's now when the run began, UTC.
        deadline: The refresh's deadline, the earlier of `deadline` and `started + budget`, UTC; None without either.
        budget, timeout, retries, on_failure: The run's arguments (None: not given).
        cache: `"none"`, `"sources"` (`cache=True`) or `"file"` (a path).
        seconds: The whole refresh's time on the clock.
        sources: One `SourceRefresh` per source, in the pipeline's order. With `on_failure="raise"` it ends at the source
            that raised.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    started: pd.Timestamp
    deadline: Optional[pd.Timestamp]
    budget: Optional[float]
    timeout: Optional[float]
    retries: Optional[int]
    on_failure: str
    cache: str
    seconds: float
    sources: Tuple[SourceRefresh, ...]

    @property
    def ok(self) -> bool:
        """Whether every source of the record is ok."""
        return all(source.ok for source in self.sources)

    @property
    def failed(self) -> Tuple[str, ...]:
        """The names of the sources that aren't ok."""
        return tuple(source.name for source in self.sources if not source.ok)

    def __getitem__(self, name: str) -> SourceRefresh:
        for source in self.sources:
            if source.name == name:
                return source
        raise KeyError(f"no source {name!r} in this refresh; it has {[source.name for source in self.sources]}")

    @property
    def last_valid(self) -> Dict[str, Optional[pd.Timestamp]]:
        """The last timestamp with a value of every column the sources gave (those whose columns are known)."""
        out: Dict[str, Optional[pd.Timestamp]] = {}
        for source in self.sources:
            out.update(source.last_valid or {})
        return out

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe: timestamps as ISO 8601 strings."""
        return {
            "start": _iso(self.start),
            "end": _iso(self.end),
            "started": _iso(self.started),
            "deadline": _iso(self.deadline),
            "budget": self.budget,
            "timeout": self.timeout,
            "retries": self.retries,
            "on_failure": self.on_failure,
            "cache": self.cache,
            "seconds": self.seconds,
            "ok": self.ok,
            "sources": [source.to_dict() for source in self.sources],
        }


def redact(text: str, secrets: Iterable[str]) -> str:
    """`text` with every secret, and its URL-encoded forms, replaced by `***`."""
    for secret in secrets:
        for form in {secret, quote(secret, safe=""), quote_plus(secret)}:
            text = text.replace(form, REDACTED)
    return text


def last_valid(df: Optional[pd.DataFrame]) -> Dict[str, Optional[pd.Timestamp]]:
    """Per column, the latest timestamp holding a value (None for a column without one)."""
    if df is None:
        return {}
    out: Dict[str, Optional[pd.Timestamp]] = {}
    for position, column in enumerate(df.columns):
        stamps = df.index[df.iloc[:, position].notna().to_numpy()]
        out[column] = stamps.max() if len(stamps) else None
    return out


def _number(value: Any, name: str, *, allow_zero: bool) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number of seconds, got {value!r}")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}, got {value!r}")
    return float(value)


class Refresh:
    """The bookkeeping of one `DataPipeline.run`: its limits on the clock and each source's entry."""

    def __init__(
        self,
        *,
        clock: Clock,
        budget: Any = None,
        deadline: Any = None,
        timeout: Any = None,
        retries: Any = None,
        on_failure: str = "raise",
        cache: str = "none",
        timezone: Optional[str] = None,
    ):
        if on_failure not in ON_FAILURE:
            raise ValueError(f"on_failure must be one of {ON_FAILURE}, got {on_failure!r}")
        if retries is not None and (
            isinstance(retries, bool) or not isinstance(retries, numbers.Integral) or retries < 0
        ):
            raise ValueError(f"retries must be a whole number of retries, 0 or more, got {retries!r}")
        self.budget = _number(budget, "budget", allow_zero=True)
        self.timeout = _number(timeout, "timeout", allow_zero=False)
        self.retries = None if retries is None else int(retries)
        self.on_failure = on_failure
        self.cache = cache
        self.clock = clock
        self.started = clock.now()
        self._began = clock.monotonic()
        self.deadline = self._deadline(deadline, timezone)
        self.sources: List[SourceRefresh] = []

    def _deadline(self, deadline: Any, timezone: Optional[str]) -> Optional[pd.Timestamp]:
        candidates = []
        if deadline is not None:
            instant = self._instant(deadline, timezone)
            if instant is pd.NaT:
                raise ValueError("deadline must be an instant, got NaT")
            # A naive deadline is a time of the pipeline's zone, as a naive `start` or `end` is.
            instant = instant.tz_localize(timezone or "UTC") if instant.tzinfo is None else instant
            candidates.append(instant.tz_convert("UTC"))
        if self.budget is not None:
            candidates.append(self.started + pd.Timedelta(seconds=self.budget))
        return min(candidates) if candidates else None

    def _instant(self, deadline: Any, timezone: Optional[str]) -> pd.Timestamp:
        """The deadline as a timestamp. A time of day alone ("11:55") is that time on today's date in ``timezone``
        (UTC without one) on the clock, however late it is: past, it leaves no time. Any other string needs a full date,
        so that no parser fills one in from the machine's clock."""
        if isinstance(deadline, str):
            time_of_day = _TIME_OF_DAY.match(deadline)
            if time_of_day:
                hours, minutes, seconds = (int(part or 0) for part in time_of_day.groups())
                if hours > 23 or minutes > 59 or seconds > 59:
                    raise ValueError(f"deadline must be a time of day 'HH:MM' or a timestamp, got {deadline!r}")
                today = self.started.tz_convert(timezone or "UTC").tz_localize(None).normalize()
                local = today + pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)
                return local.tz_localize(timezone or "UTC", ambiguous=True, nonexistent="shift_forward")
            if not _FULL_DATE.search(deadline):
                raise ValueError(
                    f"deadline must be a time of day 'HH:MM' or a timestamp with a full date, got {deadline!r}"
                )
        try:
            return pd.Timestamp(deadline)
        except (TypeError, ValueError) as e:
            raise ValueError(f"deadline must be an instant (a timestamp), got {deadline!r}") from e

    def began(self) -> float:
        return self.clock.monotonic()

    def add(self, entry: SourceRefresh) -> None:
        self.sources.append(entry)

    def record(self, start: pd.Timestamp, end: pd.Timestamp) -> RefreshRecord:
        return RefreshRecord(
            start=start,
            end=end,
            started=self.started,
            deadline=self.deadline,
            budget=self.budget,
            timeout=self.timeout,
            retries=self.retries,
            on_failure=self.on_failure,
            cache=self.cache,
            seconds=self.clock.monotonic() - self._began,
            sources=tuple(self.sources),
        )
