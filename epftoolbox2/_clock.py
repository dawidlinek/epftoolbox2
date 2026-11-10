"""One clock for every clock read and every sleep of the package.

`current_clock()` is the clock in use: `SYSTEM`, the machine's, unless code runs inside `use_clock(clock)`, a context
variable, so a block of code (a test, a simulated week) sees another clock and the rest of the process doesn't. Every time
of day the package reads (a source's Retry-After, manifest timestamps, `"today"`), every interval it times and every wait it
makes for itself (a retry's backoff, a Retry-After) goes through it. A `FakeClock` stands still until it is advanced: its
`sleep` advances it and returns at once, so code that waits never waits.

What settles data reads `settled_now()`, the earlier of the clock's now and the machine's: the per-source cache's unsettled
boundary and tail cap, the single-file cache's `settled_at`, Energy-Charts' response cache and Open-Meteo's cut between final
and provisional rows. A clock behind real time (a simulated week in the past) settles on its own now; one ahead of it can't
call days settled that the world hasn't settled yet, which a real cache would then keep.

What doesn't read the current clock, on purpose:

- Pool workers. They never read the clock (a forked worker would see the parent's clock, a spawned one `SYSTEM`).
- `time.perf_counter`, which times compute (a plan's `seconds`, a torch fit's), never a time of day or a wait.
- The results store's writer thread, whose batching ages and polls are real time by nature: they use `SYSTEM` itself, so
  that a buffer stamped in one thread and aged in another is on one clock (a context variable isn't a thread's).

`tests/test_clock.py` scans the package's source for other clock reads.
"""

import contextlib
import contextvars
import datetime
import threading
import time
from typing import Any, Iterator, List, Optional

import pandas as pd

__all__ = ["Clock", "SYSTEM", "FakeClock", "use_clock", "current_clock"]


class Clock:
    """The machine's clock, `SYSTEM`. Subclass it for another clock: `now()` is a UTC `pd.Timestamp`, `sleep(seconds)`
    waits, and `monotonic()` is seconds for timing intervals, which only go forward."""

    def now(self) -> pd.Timestamp:
        """The current instant, a UTC `pd.Timestamp`."""
        return pd.Timestamp.now(tz="UTC")

    def sleep(self, seconds: float) -> None:
        """Wait `seconds`."""
        time.sleep(seconds)

    def monotonic(self) -> float:
        """Seconds of a clock that never goes back, for intervals."""
        return time.monotonic()

    def __repr__(self) -> str:
        return "SYSTEM" if self is SYSTEM else f"{type(self).__name__}()"


SYSTEM = Clock()

_MICROSECOND = pd.Timedelta(microseconds=1)


def _microseconds(seconds: float, what: str) -> int:
    if seconds < 0:
        raise ValueError(f"{what} must be non-negative, got {seconds}")
    return round(float(seconds) * 1_000_000)


class FakeClock(Clock):
    """A clock that stands still until it is advanced, for tests and simulations.

    Args:
        start: The first instant: anything `pd.Timestamp` reads; a naive one is UTC.
        tick: Seconds every read (`now()`, `monotonic()`) advances it after it answers, so a loop that waits on the clock
            without sleeping still ends. 0 (the default): only `sleep` and `advance` move it.

    `sleep(seconds)` returns at once, advances it by `seconds` and appends them to `sleeps`; `advance(seconds)` moves it
    without counting a sleep. `monotonic()` is the seconds it advanced since `start`. Time moves in whole microseconds, the
    resolution of the system clock. Safe to share between threads.

    Example:
        >>> clock = FakeClock("2026-10-21 23:30")
        >>> with use_clock(clock):
        ...     clock.sleep(3600)  # returns at once
        >>> clock.now()
        Timestamp('2026-10-22 00:30:00+0000', tz='UTC')
    """

    def __init__(self, start: Any, *, tick: float = 0.0):
        instant = pd.Timestamp(start)
        if instant is pd.NaT:
            raise ValueError("FakeClock needs a start instant, got NaT")
        self.start = instant.tz_localize("UTC") if instant.tzinfo is None else instant.tz_convert("UTC")
        self.tick = float(tick)
        self._tick = _microseconds(self.tick, "tick")
        self._elapsed = 0  # microseconds since `start`
        self.sleeps: List[float] = []
        self._lock = threading.Lock()

    def _read(self) -> int:
        with self._lock:
            elapsed = self._elapsed
            self._elapsed += self._tick
        return elapsed

    def now(self) -> pd.Timestamp:
        return self.start + self._read() * _MICROSECOND

    def monotonic(self) -> float:
        return self._read() / 1_000_000

    def sleep(self, seconds: float) -> None:
        step = _microseconds(seconds, "sleep length")
        with self._lock:
            self.sleeps.append(float(seconds))
            self._elapsed += step

    def advance(self, seconds: float) -> None:
        """Move the clock `seconds` forward (never back)."""
        step = _microseconds(seconds, "advance")
        with self._lock:
            self._elapsed += step

    def __repr__(self) -> str:
        with self._lock:
            elapsed = self._elapsed
        return f"FakeClock({str(self.start + elapsed * _MICROSECOND)!r}, tick={self.tick})"


_CURRENT: contextvars.ContextVar[Clock] = contextvars.ContextVar("epftoolbox2_clock", default=SYSTEM)


def current_clock() -> Clock:
    """The clock in use: the innermost `use_clock` block's, else `SYSTEM`."""
    return _CURRENT.get()


@contextlib.contextmanager
def use_clock(clock: Clock) -> Iterator[Clock]:
    """Read every clock of the package from `clock` inside the block (a context variable: this thread and the tasks it
    starts in this context, not threads started without it, nor pool workers, which never read it). Yields `clock`."""
    missing = [name for name in ("now", "sleep", "monotonic") if not callable(getattr(clock, name, None))]
    if missing:
        raise TypeError(f"use_clock needs a clock with now(), sleep() and monotonic(); {clock!r} has no {missing}")
    token = _CURRENT.set(clock)
    try:
        yield clock
    finally:
        _CURRENT.reset(token)


def settled_now() -> pd.Timestamp:
    """The instant data is settled against: the current clock's now, but never later than the machine's (see the module)."""
    return min(current_clock().now(), SYSTEM.now())


def wall_time(tz: Optional[str] = None, at: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Now on the current clock (or the instant `at`): in `tz` (tz-aware), or with `tz` None the host's local wall-clock
    time, naive, as `pd.Timestamp.now()` gives it."""
    instant = current_clock().now() if at is None else at
    if tz is not None:
        return instant.tz_convert(tz)
    # The host's offset at that instant (`astimezone()` with no zone is the host's local zone, DST included).
    offset = instant.floor("us").to_pydatetime().astimezone().utcoffset()
    return instant.tz_convert(None) + pd.Timedelta(offset)


def today(tz: Optional[str] = None, at: Optional[pd.Timestamp] = None) -> datetime.date:
    """Today's date on the current clock (or at the instant `at`): in `tz`, or the host's local date when `tz` is None
    (`date.today()`'s)."""
    return wall_time(tz, at).date()
