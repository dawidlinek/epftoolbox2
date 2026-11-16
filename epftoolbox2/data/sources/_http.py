"""One HTTP request with retries for every source, on the clock and within a refresh's time (step 8b).

`request(session, url, ...)` sends a GET, asks again after a network error or an answer the source's `read` calls
transient (it returns a `Retry`), waits between attempts (the server's Retry-After, else `backoff * 2**attempt`, at most
`max_wait`) on the current clock (`epftoolbox2.clock`), and raises the source's own error when the retries run out. Each
source keeps its own counts, backoffs, timeouts, messages and error classes: this module runs the loop, nothing more.

Inside `DataPipeline.run(..., budget=, deadline=, timeout=, retries=)` the pipeline sets a refresh's limits around each
source's fetch (`limits`, a context variable, so no source parameter, YAML or cache key moves): `timeout` and `retries`
replace every request's own, and with a deadline a request is sent only while time is left, with its timeout cut to what
is left, and a wait that would end at or after the deadline is never started. `SourceTimeout` is raised instead, with the
last error. Outside such a run (a source's own `fetch`, or `run` without the new arguments) every source asks and waits as
it did before.

A server's Retry-After given as an HTTP date is an instant of real time: it is measured against the machine's clock
(`SYSTEM`), whatever clock is in use, so a replay on a `FakeClock` against a real server doesn't read a real server's date
against a fake now. A wait it gives is still a wait on the current clock, like every other wait here.
"""

import contextlib
import contextvars
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterator, Optional, Tuple, Type

import pandas as pd
import requests

from epftoolbox2._clock import SYSTEM, Clock, current_clock

__all__ = [
    "NETWORK_ERRORS",
    "Limits",
    "Retry",
    "SourceTimeout",
    "is_unavailable",
    "unavailable",
    "limits",
    "request",
    "retry_after_seconds",
    "time_left",
]

# Errors of `session.get` that a source asks again after (the rest of `requests.RequestException` is a mistake in the
# request itself, see `request`'s `refuse`).
NETWORK_ERRORS: Tuple[Type[BaseException], ...] = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


class SourceTimeout(RuntimeError):
    """A source ran out of a refresh's time (`DataPipeline.run(..., budget=, deadline=)`): its next request, or a wait before
    one, would end at or after the deadline. The message names the request and its last error (redacted by the source);
    `deadline` is the instant, UTC."""

    def __init__(self, message: str, deadline: Optional[pd.Timestamp] = None):
        super().__init__(message)
        self.deadline = deadline


class Retry:
    """What a source's `read` returns for an answer worth asking again.

    Args:
        error: What went wrong, for the log and the error raised when the retries run out (no secret in it).
        wait: Seconds the server asked to wait (Retry-After), else None for the backoff.
        exception: The exception behind it, for a source that raises that one when its retries run out.
        status: The answer's HTTP status, for one the exception doesn't carry (a body that isn't JSON).
    """

    __slots__ = ("error", "wait", "exception", "status")

    def __init__(
        self,
        error: str,
        wait: Optional[float] = None,
        exception: Optional[BaseException] = None,
        status: Optional[int] = None,
    ):
        self.error = error
        self.wait = wait
        self.exception = exception
        self.status = status  # the HTTP status behind it, when the exception doesn't carry its response

    def __repr__(self) -> str:
        return f"Retry({self.error!r}, wait={self.wait!r})"


@dataclass
class Limits:
    """A refresh's limits for one source's fetch (None: the source's own), and how many requests the fetch sent
    (`requests`) and how many of them asked again (`retries`)."""

    deadline: Optional[pd.Timestamp] = None
    timeout: Optional[float] = None
    retries: Optional[int] = None
    requests: int = 0
    retried: int = 0


_LIMITS: contextvars.ContextVar[Optional[Limits]] = contextvars.ContextVar("epftoolbox2_refresh_limits", default=None)


@contextlib.contextmanager
def limits(
    deadline: Optional[pd.Timestamp] = None, timeout: Optional[float] = None, retries: Optional[int] = None
) -> Iterator[Limits]:
    """Every `request` inside the block keeps to these limits and is counted in the `Limits` yielded. `DataPipeline.run`
    opens one around each source's fetch."""
    current = Limits(deadline=deadline, timeout=timeout, retries=retries)
    token = _LIMITS.set(current)
    try:
        yield current
    finally:
        _LIMITS.reset(token)


def time_left() -> Optional[float]:
    """Seconds left of the refresh's deadline on the current clock (never negative), or None outside a refresh with one."""
    active = _LIMITS.get()
    if active is None or active.deadline is None:
        return None
    return max(0.0, (active.deadline - current_clock().now()).total_seconds())


# Set on the error `request` raises when a source's retries ran out on answers worth asking again (network errors, 429,
# 5xx): the source is unavailable for now, not refusing the request.
_UNAVAILABLE = "_epftoolbox2_source_unavailable"


def is_unavailable(error: BaseException) -> bool:
    """Whether `error` says a source is out of reach for now: it ran out of time (`SourceTimeout`), out of retries on
    network errors or server-side answers (`request`), or hit a network error itself. An authentication or request error,
    a missing file or a bug is not: `DataPipeline.run(on_failure="cache")` raises those."""
    return isinstance(error, (SourceTimeout, *NETWORK_ERRORS)) or getattr(error, _UNAVAILABLE, False)


def unavailable(error: BaseException) -> BaseException:
    """`error`, marked as a source out of reach for now (`is_unavailable`): a rate limit a source raises at once rather
    than wait out (Energy-Charts' Retry-After beyond its cap, Open-Meteo's hourly or daily limit)."""
    try:
        setattr(error, _UNAVAILABLE, True)
    except AttributeError:  # an exception class with __slots__
        pass
    return error


def _client_error(retry: Retry) -> bool:
    """A 4xx answer other than 429 behind `retry` (Open-Meteo retries every HTTP error, also a 403 page that isn't JSON)."""
    status = retry.status
    if status is None:
        status = getattr(getattr(retry.exception, "response", None), "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status != 429


def retry_after_seconds(value: Optional[str]) -> Optional[float]:
    """The seconds a Retry-After header asks to wait (never negative), or None without one or for a value that is
    neither. A number is seconds; an HTTP date is an instant of real time, measured against the machine's clock
    (`SYSTEM`) whatever clock is in use."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, (parsedate_to_datetime(value) - SYSTEM.now()).total_seconds())
    except (TypeError, ValueError):
        return None


def _unchanged(text: str) -> str:
    return text


def request(
    session: Any,
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float,
    retries: int,
    backoff: float,
    read: Callable[[Any], Any],
    give_up: Callable[[Retry, int], BaseException],
    max_wait: Optional[float] = None,
    refuse: Optional[Callable[[requests.RequestException], BaseException]] = None,
    retry_on: Tuple[Type[BaseException], ...] = NETWORK_ERRORS,
    log: Optional[Callable[[int, int, Retry, Optional[float]], None]] = None,
    before: Optional[Callable[[Callable[..., None]], None]] = None,
    label: str = "",
    redact: Callable[[str], str] = _unchanged,
    deadline: Optional[pd.Timestamp] = None,
    clock: Optional[Clock] = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> Any:
    """GET `url` with retries; returns what `read` returns for the first answer that isn't a `Retry`.

    Args:
        session: Has `get(url, params=..., headers=..., timeout=...)`, a `requests.Session` (`headers` is passed only
            when given).
        timeout, retries, backoff: The request's timeout in seconds, how many times it is asked again (so `retries + 1`
            attempts) and the base of the backoff `backoff * 2**attempt`. Inside a refresh with limits (`limits`) its
            `timeout` and `retries` replace these.
        read: Turns an answer into the result, a `Retry`, or raises (not retried; it runs outside any `except`, so
            the error carries no context).
        give_up: The exception to raise when the retries run out, from the last `Retry` and the number of attempts.
        max_wait: The longest wait between two attempts (a longer Retry-After is cut to it).
        refuse: The exception to raise, outside the `except`, for a `requests.RequestException` of `session.get` that
            is not in `retry_on`; None re-raises it.
        retry_on: Errors of `session.get` asked again (`NETWORK_ERRORS` by default).
        log: Called before each wait as `log(attempt, retries, retry, wait)`, and with `wait=None` after the last attempt.
        before: Called before each attempt with a `pause(seconds, sleep=None)` that keeps to the deadline and waits with
            `sleep` (the request's by default): a rate limit's wait.
        label, redact: Name the request in a `SourceTimeout`, and take any secret out of its message.
        deadline: An instant (UTC) the request must not run past; a refresh's deadline is added to it.
        clock, sleep: The clock the deadline is read on (the current one by default) and what waits (its `sleep`).

    Raises:
        SourceTimeout: When a request or a wait before one would end at or after the deadline.
    """
    clock = current_clock() if clock is None else clock
    sleep = clock.sleep if sleep is None else sleep
    active = _LIMITS.get()
    if active is not None:
        timeout = timeout if active.timeout is None else active.timeout
        retries = retries if active.retries is None else active.retries
        if active.deadline is not None:
            deadline = active.deadline if deadline is None else min(deadline, active.deadline)
    last: Optional[Retry] = None

    def seconds_left() -> float:
        return (deadline - clock.now()).total_seconds()

    def timed_out(what: str) -> SourceTimeout:
        message = f"{label}: no time is left for {what} before the refresh's deadline {deadline.isoformat()}"
        if last is not None:
            message += f"; last error: {last.error}"
        return SourceTimeout(redact(message), deadline)

    def check(seconds: float, what: str) -> None:
        if deadline is not None and seconds >= seconds_left():
            raise timed_out(f"{what} of {seconds:.0f} s")

    def pause(seconds: float, sleeper: Optional[Callable[[float], None]] = None) -> None:
        check(seconds, "a wait")
        (sleep if sleeper is None else sleeper)(seconds)

    attempts = retries + 1
    for attempt in range(attempts):
        if before is not None:
            before(pause)
        arguments: dict = {"params": params}
        if headers is not None:
            arguments["headers"] = headers
        arguments["timeout"] = timeout
        if deadline is not None:
            left = seconds_left()
            if left <= 0:
                raise timed_out("a request")
            arguments["timeout"] = min(timeout, left)
        if active is not None:
            active.requests += 1
            if attempt:
                active.retried += 1
        refused = None
        try:
            response = session.get(url, **arguments)
        except retry_on as exc:
            outcome = Retry(f"{type(exc).__name__}: {exc}", exception=exc)
        except requests.RequestException as exc:
            if refuse is None:
                raise
            refused = refuse(exc)
        else:
            outcome = read(response)
        if refused is not None:
            raise refused  # outside the `except`: no context that could carry a secret
        if not isinstance(outcome, Retry):
            return outcome
        last = outcome
        if attempt == retries:
            if log is not None:
                log(attempt, retries, outcome, None)
            break
        wait = outcome.wait if outcome.wait is not None else backoff * 2**attempt
        if max_wait is not None:
            wait = min(wait, max_wait)
        check(wait, "the wait before the next attempt")
        if log is not None:
            log(attempt, retries, outcome, wait)
        sleep(wait)
    if deadline is not None and seconds_left() <= 0:
        raise timed_out("another attempt")  # the last attempt ran into the deadline: out of time, not failed
    failure = give_up(last, attempts)
    raise failure if _client_error(last) else unavailable(failure)
