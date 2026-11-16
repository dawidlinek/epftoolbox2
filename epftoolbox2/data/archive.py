"""Archives of raw downloads (`SnapshotArchive`, step 8j): what each source returned and when, kept as evidence and for replay.

Whoever runs an operations pipeline calls `SnapshotArchive(path).record(data_pipeline)` on a schedule of their own (cron, a systemd
timer, a SLURM job: the library schedules nothing and never edits a crontab). Each call fetches every source of the pipeline over
the whole local days from `days_back` days before today to `days_ahead` days after it, bypassing every cache, and stores each frame
as the source returned it, with one catalog line: when it was fetched on the clock, the range it asked for, its rows and columns,
the sha256 of its file and the source's cache config, never a secret. What the archive then tells:

- `first_seen(column)`: each cell's first fetch that had a value, a publication time measured instead of assumed;
- `rewrites(column)`: the cells whose value changed between fetches (a TSO forecast revised after the gate, a corrected actual);
- `ArchiveSource` (`data/sources/archive.py`) replays it as an honest vintage: for each delivery day, the values of the newest
  fetch at or before a stated time, under the rule that says so.

Layout:

    <path>/catalog.jsonl                         one line per fetch, appended under <path>/catalog.lock
    <path>/<source key>/<fetched_at>.parquet     the frame as returned, index in UTC, `attrs` dropped

A source's key (`archive_key`) is its class name and a hash of its class and cache config: another configuration, or a parsing
change that bumps the source's `schema_version`, is another key, so the fetches compared with each other were parsed alike.

An observation is a row of a fetch inside the range the fetch asked for, what a `DataPipeline` keeps of it: a fetch that asked for
a cell and has no value there (NaN, or no row) saw nothing of it. `fetched_at` is the instant the fetch returned, on the clock
(`epftoolbox2.clock`): a value counts as seen once it was in hand, never from when the request started.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from urllib.parse import quote, quote_plus

import numpy as np
import pandas as pd

from epftoolbox2._clock import current_clock
from filelock import Timeout

from epftoolbox2.data.cache_manager import make_file_lock, settled_until_of
from epftoolbox2.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "EXIT_WARNING",
    "ArchiveError",
    "ArchiveLockedError",
    "Fetch",
    "RecordResult",
    "SecretInArchiveError",
    "SnapshotArchive",
    "archive_key",
    "record_snapshot",
    "source_secrets",
]

CATALOG = "catalog.jsonl"
CATALOG_LOCK = "catalog.lock"
# The catalog lines' layout; a reader refuses a newer one.
ARCHIVE_FORMAT = 1
OK, FAILED, SKIPPED = "ok", "failed", "skipped"
# `RecordResult.exit_code` when every fetch was written but one asks for a look: no rows, or fewer columns than before.
EXIT_WARNING = 4
NO_ROWS = "no rows: a replay reads every day of its range as missing"
REDACTED = "***"
# Attribute and environment-variable names whose values are secrets: never written, and refused if a frame or line holds them.
# The words of a name (snake_case, kebab-case or camelCase) that make its value a secret, and the pairs of words.
_SECRET_WORDS = frozenset(
    {"auth", "authorization", "token", "tokens", "password", "passwd", "secret", "secrets", "credential", "credentials",
     "apikey", "passphrase"}
)  # fmt: skip
_SECRET_PAIRS = frozenset({("api", "key"), ("access", "key"), ("secret", "key"), ("private", "key")})
_SECRET_ENVIRONMENT = re.compile(r"(API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD)$", re.IGNORECASE)
# Shorter strings aren't searched for: a short word would refuse frames that merely contain it.
_SECRET_MIN_LENGTH = 8
_PROXY_USER = re.compile(r"^[a-z][a-z0-9+.-]*://([^@/]+)@", re.IGNORECASE)
_FILE_STAMP = "%Y%m%dT%H%M%S%fZ"
# Step 8b's refresh budget (`DataPipeline.run(budget=, deadline=, timeout=, retries=)`): `record` takes the same arguments.
_PENDING_8B = ("budget", "deadline", "timeout", "retries")
# `SnapshotArchive.catalog()`'s columns, in order.
_CATALOG_COLUMNS = [
    "source",
    "class",
    "status",
    "started_at",
    "fetched_at",
    "start",
    "end",
    "day_zone",
    "rows",
    "columns",
    "file",
    "sha256",
    "settled_until",
    "config",
    "error",
    "warnings",
]


class ArchiveError(ValueError):
    """An archive that can't be read as asked: no catalog, a column in no source or in several, a file that isn't its own."""


class ArchiveLockedError(ArchiveError):
    """The archive's catalog lock wasn't free within `SnapshotArchive.LOCK_TIMEOUT` seconds."""


class SecretInArchiveError(ValueError):
    """A frame or catalog line about to be written holds a secret value (an API key of the source or the environment)."""


# --- keys, secrets and plain values ----------------------------------------------------------------------------------------


def _qualified_name(obj: Any) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _plain(value: Any) -> Any:
    """A JSON-safe, deterministic stand-in for a cache config value."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, range)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "tolist"):  # numpy scalars and arrays
        return _plain(value.tolist())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return repr(value)


def _words(name: Any) -> List[str]:
    """The words of a name: `api_key`, `api-key` and `apiKey` are ["api", "key"]."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [word for word in re.split(r"[^a-z0-9]+", spaced.lower()) if word]


def is_secret_name(name: Any) -> bool:
    """Whether a name says its value is a secret: a whole word such as token, password, secret, auth or credentials, or a pair
    such as api key or access key. `author`, `sort_key` and `tokenizer` aren't."""
    words = _words(name)
    return any(word in _SECRET_WORDS for word in words) or any(pair in _SECRET_PAIRS for pair in zip(words, words[1:]))


def _collect_secrets(name: Any, value: Any, found: set, named: bool) -> None:
    """The strings of `value` that are secrets: every string under a name like a key, token, password, auth or credentials
    (at any depth of dicts, lists and tuples), and the user info of a URL under a name ending in "url"."""
    named = named or is_secret_name(name)
    if isinstance(value, str):
        if named and value:
            found.add(value)
        user = _PROXY_USER.match(value) if str(name).lower().endswith("url") else None
        if user is not None:
            found.add(user.group(1))
            found.update(part for part in user.group(1).split(":") if part)
    elif isinstance(value, dict):
        for key, item in value.items():
            _collect_secrets(key, item, found, named)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _collect_secrets(name, item, found, named)


def source_secrets(source: Any) -> List[str]:
    """The secret values a fetch of `source` must never write, longest first: its attributes named like a key, token, password,
    auth or credentials (`api_key` first of all, also nested in dicts, lists and tuples), the credentials of a URL, and every
    environment variable whose name ends in API_KEY, TOKEN, SECRET or PASSWORD; each also URL-encoded. Values shorter than 6
    characters are left out."""
    found: set = set()
    for name, value in vars(source).items():
        _collect_secrets(name, value, found, False)
    for name, value in os.environ.items():
        if _SECRET_ENVIRONMENT.search(name):
            found.add(value)
    found = {value for value in found if len(value) >= _SECRET_MIN_LENGTH}
    encoded = {form for value in found for form in (quote(value, safe=""), quote_plus(value, safe=""), quote(value))}
    return sorted(found | encoded, key=lambda v: (-len(v), v))


def _redact(text: str, secrets: Sequence[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, REDACTED)
    return text


def _scrub(value: Any, secrets: Sequence[str]) -> Any:
    """`value` without the entries named like a secret and with every secret string redacted."""
    if isinstance(value, dict):
        return {k: _scrub(v, secrets) for k, v in value.items() if not is_secret_name(k)}
    if isinstance(value, list):
        return [_scrub(v, secrets) for v in value]
    if isinstance(value, str):
        return _redact(value, secrets)
    return value


def _refuse_secrets(data: bytes, secrets: Sequence[str], what: str) -> None:
    for secret in secrets:
        if secret.encode() in data:
            raise SecretInArchiveError(
                f"{what} holds a secret value of the source or the environment; nothing was written"
            )


def _refuse_secrets_in_frame(frame: pd.DataFrame, secrets: Sequence[str], what: str) -> None:
    """The frame's own text (column and index names, every value of a column that isn't numeric) searched for each secret:
    a compressed Parquet page needn't hold a string verbatim, so the byte search of the file alone could miss one."""
    if not secrets:
        return
    texts = [str(column) for column in frame.columns] + [str(frame.index.name or "")]
    for column in frame.columns:
        values = frame[column]
        if not (pd.api.types.is_numeric_dtype(values) or pd.api.types.is_datetime64_any_dtype(values)):
            texts.extend(str(value) for value in pd.unique(values.dropna()))
    for secret in secrets:
        if any(secret in text for text in texts):
            raise SecretInArchiveError(
                f"{what} holds a secret value of the source or the environment; nothing was written"
            )


def archive_key(source: Any) -> Optional[str]:
    """The directory and catalog key of `source` in an archive: `<ClassName>-<16 hex>`, the hash of its class and cache config
    (secrets left out). None of a source without a cache config (`get_cache_config()` None: local files, another pipeline's
    results), which has nothing to archive."""
    config = source.get_cache_config()
    if config is None:
        return None
    return _key(source, _scrub(_plain(config), source_secrets(source)))


def _key(source: Any, config: Dict[str, Any]) -> str:
    payload = json.dumps({"class": _qualified_name(source), "config": config}, sort_keys=True)
    return f"{type(source).__name__}-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


# --- one fetch ---------------------------------------------------------------------------------------------------------------


def _local_days(now: pd.Timestamp, zone: str, days_back: int, days_ahead: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """[midnight of today - days_back, midnight of today + days_ahead + 1) in `zone`, as UTC instants: the spec's
    [now - days_back, now + days_ahead) widened to whole local days, so a source never stamps a daily value mid-day."""
    today = pd.Timestamp(now.tz_convert(zone).date())

    def midnight(day: pd.Timestamp) -> pd.Timestamp:
        return day.tz_localize(zone, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")

    return midnight(today - pd.Timedelta(days=days_back)), midnight(today + pd.Timedelta(days=days_ahead + 1))


def _uncached(source: Any) -> Any:
    """`source`, or a shallow copy that skips its own response cache (`use_cache`, Energy-Charts'): an archive keeps what the
    provider answers now, not an answer kept from before."""
    if getattr(source, "use_cache", None) is True:
        clone = copy.copy(source)
        clone.use_cache = False
        return clone
    return source


def _as_returned(frame: Any) -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    """The frame a fetch returned, ready to store: index in UTC (a naive one is UTC, the source contract), `attrs` dropped
    (pandas writes them into the Parquet file), and the `settled_until` the source declared."""
    if frame is None:
        frame = pd.DataFrame()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"fetch returned {type(frame).__name__}, not a DataFrame")
    settled_until = settled_until_of(frame)
    frame = frame.copy(deep=False)
    frame.attrs = {}
    if not isinstance(frame.index, pd.DatetimeIndex):
        if len(frame):
            raise TypeError(f"fetch returned a frame indexed by {type(frame.index).__name__}, not a DatetimeIndex")
        frame.index = pd.DatetimeIndex([], tz="UTC")
    elif frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    if not all(isinstance(column, str) for column in frame.columns):
        raise TypeError(f"fetch returned column names that aren't strings: {list(frame.columns)[:5]}")
    if frame.index.has_duplicates:
        twice = frame.index[frame.index.duplicated()].unique()
        shown = ", ".join(stamp.isoformat() for stamp in twice[:5]) + (", ..." if len(twice) > 5 else "")
        raise ValueError(
            f"the frame holds {len(twice)} duplicated instants ({shown}): refused, as no one value of such a cell can be "
            "read back"
        )
    return frame, settled_until


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    return buffer.getvalue()


def _write_durably(path: Path, data: bytes) -> None:
    """`data` at `path`, renamed into place after an fsync: a catalog line never names a file that isn't whole."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _iso(stamp: Optional[pd.Timestamp]) -> Optional[str]:
    return None if stamp is None else stamp.tz_convert("UTC").isoformat()


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


@dataclass
class RecordResult:
    """What one `SnapshotArchive.record` call did, per source in the pipeline's order: the catalog line it wrote (status "ok"
    or "failed"), or, for a source with nothing to archive, a line with status "skipped" that isn't written."""

    lines: List[Dict[str, Any]] = field(default_factory=list)

    def _with(self, status: str) -> List[Dict[str, Any]]:
        return [line for line in self.lines if line["status"] == status]

    @property
    def ok(self) -> List[Dict[str, Any]]:
        return self._with(OK)

    @property
    def failed(self) -> List[Dict[str, Any]]:
        return self._with(FAILED)

    @property
    def skipped(self) -> List[Dict[str, Any]]:
        return self._with(SKIPPED)

    @property
    def warned(self) -> List[Dict[str, Any]]:
        """The ok lines with warnings: a fetch with no rows, or with fewer columns than the source's previous one."""
        return [line for line in self.ok if line.get("warnings")]

    @property
    def exit_code(self) -> int:
        """1 when a fetch failed, else `EXIT_WARNING` (4) when one was written with warnings, else 0: what a scheduler should
        alert on."""
        if self.failed:
            return 1
        return EXIT_WARNING if self.warned else 0

    def summary(self) -> str:
        """One line per source: its key or class, the status, the rows or the error."""
        out = []
        for line in self.lines:
            name = line.get("source") or line.get("class")
            if line["status"] == OK:
                detail = f"{line['rows']} rows, {line['start']} to {line['end']}, fetched {line['fetched_at']}"
                detail += "".join(f"; warning: {warning}" for warning in line.get("warnings") or ())
            else:
                detail = line.get("error", "")
            out.append(f"{name}: {line['status']} ({detail})")
        return "\n".join(out)


# --- the archive -------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Fetch:
    """One ok catalog line, as the readers use it: the source key, when it returned, the range it asked for, its file."""

    key: str
    fetched_at: pd.Timestamp
    start: pd.Timestamp
    end: pd.Timestamp
    file: str
    sha256: str
    columns: Tuple[str, ...]
    order: int  # the line's place in the catalog: ties of fetched_at keep the order they were written in


class SnapshotArchive:
    """A directory of raw downloads: every fetch of every source as returned, with a catalog (see the module).

    Args:
        path: The archive's directory; made by the first `record`. `~` is expanded.

    Example:
        >>> archive = SnapshotArchive("archive/de")
        >>> archive.record(data_pipeline)           # run by cron, a systemd timer or a SLURM job, e.g. every 15 minutes
        >>> archive.first_seen("load_forecast")      # each cell's first fetch with a value
        >>> archive.rewrites("load_forecast")        # the cells whose value changed between fetches
    """

    # Seconds `record` waits for the catalog lock (another recording, or a lock file left by a process that died where the file
    # system has no flock) before raising `ArchiveLockedError`.
    LOCK_TIMEOUT = 600.0

    def __init__(self, path: Union[str, os.PathLike]):
        self.path = Path(os.path.expanduser(os.fspath(path)))

    def __repr__(self) -> str:
        return f"SnapshotArchive({str(self.path)!r})"

    @property
    def catalog_path(self) -> Path:
        return self.path / CATALOG

    # -- recording --------------------------------------------------------------------------------------------------------

    def record(
        self,
        data_pipeline: Any,
        days_back: int = 3,
        days_ahead: int = 2,
        *,
        budget: Any = None,
        deadline: Any = None,
        timeout: Any = None,
        retries: Any = None,
    ) -> RecordResult:
        """Fetch every source of `data_pipeline` over the local days from `days_back` before today to `days_ahead` after it,
        bypassing every cache, and store each frame with its catalog line.

        The days are local days of the pipeline's `timezone` (or, without one, of the zone a source stamps its daily values at,
        else UTC), whole: `[now - days_back, now + days_ahead)` widened to midnights. A source without a cache config (a CSV
        file, another pipeline's results) has nothing to download and is skipped; a source whose fetch raises gets a "failed"
        line with the error, secrets redacted, and the others still run. A frame or line holding a secret is refused.

        `budget`, `deadline`, `timeout` and `retries` are step 8b's refresh budget (`DataPipeline.run`'s); until it lands any
        of them set raises `NotImplementedError`.

        Returns:
            A `RecordResult`: each source's line, and `exit_code` 1 when any fetch failed.
        """
        from epftoolbox2.pipelines.data_pipeline import DataPipeline, _cache_day_zone

        pending = [name for name, value in zip(_PENDING_8B, (budget, deadline, timeout, retries)) if value is not None]
        if pending:
            raise NotImplementedError(
                f"record({', '.join(f'{name}=' for name in pending)}) needs step 8b's refresh budget, which isn't in this "
                "version yet: leave them None (each source then uses its own retries)"
            )
        if not isinstance(data_pipeline, DataPipeline):
            raise TypeError(f"record() takes a DataPipeline, got {type(data_pipeline).__name__}")
        for name, value in (("days_back", days_back), ("days_ahead", days_ahead)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a whole number of days, 0 or more, got {value!r}")
        if not data_pipeline.sources:
            raise ValueError("record() needs a DataPipeline with at least one source")
        data_pipeline._validate_settings()  # binds the market zone to the sources that read it, as run() does
        now = current_clock().now()
        result = RecordResult()
        previous = self._last_columns()
        for source in data_pipeline.sources:
            zone = _cache_day_zone(source, data_pipeline.timezone) or "UTC"
            start, end = _local_days(now, zone, int(days_back), int(days_ahead))
            line = self._record_one(source, start, end, zone, previous)
            if line["status"] == OK and line["rows"]:
                previous[line["source"]] = list(line["columns"])
            result.lines.append(line)
        for line in result.failed:
            logger.warning(f"Archive: {line.get('source') or line.get('class')} failed: {line['error']}")
        for line in result.warned:
            logger.warning(f"Archive: {line['source']}: {'; '.join(line['warnings'])}")
        return result

    def _last_columns(self) -> Dict[str, List[str]]:
        """Each key's columns in its newest ok line with rows, for the warning on a fetch with fewer."""
        if not self.catalog_path.exists():
            return {}
        out: Dict[str, List[str]] = {}
        for line in self._lines():
            if line.get("status") == OK and line.get("source") and line.get("rows"):
                out[line["source"]] = list(line.get("columns") or ())
        return out

    def _record_one(
        self, source: Any, start: pd.Timestamp, end: pd.Timestamp, zone: str, previous: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        from epftoolbox2 import __version__

        secrets = source_secrets(source)
        config = source.get_cache_config()
        if config is None:
            return {
                "source": None,
                "class": _qualified_name(source),
                "status": SKIPPED,
                "error": "no cache config: a local or derived source, nothing to download",
            }
        config = _scrub(_plain(config), secrets)
        key = _key(source, config)
        clock = current_clock()
        line: Dict[str, Any] = {
            "format": ARCHIVE_FORMAT,
            "source": key,
            "class": _qualified_name(source),
            "status": OK,
            "started_at": _iso(clock.now()),
            "fetched_at": None,
            "start": _iso(start),
            "end": _iso(end),
            "day_zone": zone,
        }
        data = None
        try:
            frame = _uncached(source).fetch(start, end)
            line["fetched_at"] = _iso(clock.now())  # the instant the answer was in hand
            frame, settled_until = _as_returned(frame)
            # The frame's data (values and labels), not the Parquet bytes, whose metadata holds words like "datetime" and "UTC".
            _refuse_secrets_in_frame(frame, secrets, f"The frame {key} returned")
            data = _parquet_bytes(frame)
            warnings = []
            if not len(frame):
                warnings.append(NO_ROWS)
            else:
                missing = [column for column in previous.get(key, ()) if column not in frame.columns]
                if missing:
                    warnings.append(f"fewer columns than the previous fetch with rows: missing {missing}")
            line.update(
                rows=int(len(frame)),
                columns=list(frame.columns),
                file=None,
                sha256=hashlib.sha256(data).hexdigest(),
                settled_until=_iso(settled_until),
                warnings=warnings,
            )
        except Exception as exc:  # noqa: BLE001 - one source failing must not stop the others; the line says why
            if line["fetched_at"] is None:
                line["fetched_at"] = _iso(clock.now())
            line["status"] = FAILED
            line["error"] = _redact(f"{type(exc).__name__}: {exc}", secrets)
            data = None
        line["config"] = config
        line["epftoolbox2"] = __version__
        self._append(line, data, secrets)
        return line

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """The archive's catalog lock, waited for at most `LOCK_TIMEOUT` seconds of real time (another process holds it)."""
        path = self.path / CATALOG_LOCK
        lock = make_file_lock(path)
        try:
            lock.acquire(timeout=self.LOCK_TIMEOUT)
        except Timeout:
            raise ArchiveLockedError(
                f"{path} wasn't free within {self.LOCK_TIMEOUT:g} s: another recording into {self.path} is running, or a "
                "process died holding it where the file system has no flock (then the file stays behind: remove it when "
                "no recording runs)"
            ) from None
        try:
            yield
        finally:
            lock.release()

    def _append(self, line: Dict[str, Any], data: Optional[bytes], secrets: Sequence[str]) -> None:
        """Write the frame (when there is one) under a name of its own, then its catalog line, under the archive's lock. A line
        that can't be written (not JSON, a secret, an OSError) becomes a failed line, so the later sources still run; one that
        reached the catalog whole before the error (an fsync that failed) stays the only line, with the error as a warning."""
        self.path.mkdir(parents=True, exist_ok=True)
        with self._locked():
            self._end_last_line()
            try:
                self._write(line, data, secrets)
                return
            except Exception as exc:  # noqa: BLE001 - recorded as a failed line, as a failed fetch is
                error = exc
            self._end_last_line()  # a write that failed half-way left a torn line
            if self._is_last_line(line):
                line.setdefault("warnings", []).append(
                    _redact(f"the catalog line was written, but {type(error).__name__}: {error}", secrets)
                )
                return
            failed = self._as_failed(line, error, secrets)
            try:
                self._write(failed, None, secrets)
            except Exception as exc:  # noqa: BLE001 - nothing more can be written for this source; the result says so
                self._end_last_line()
                failed["error"] += f"; the line couldn't be written either ({type(exc).__name__})"
                failed["written"] = False
            line.clear()
            line.update(failed)

    def _write(self, line: Dict[str, Any], data: Optional[bytes], secrets: Sequence[str]) -> None:
        if data is not None:
            folder = self.path / line["source"]
            folder.mkdir(exist_ok=True)
            stem = _utc(line["fetched_at"]).strftime(_FILE_STAMP)
            name, n = f"{stem}.parquet", 2
            while (folder / name).exists():  # two fetches of one source at one instant (a clock standing still)
                name, n = f"{stem}-{n}.parquet", n + 1
            line["file"] = f"{line['source']}/{name}"
        try:
            text = json.dumps(line, sort_keys=False, allow_nan=False) + "\n"
            _refuse_secrets(text.encode(), secrets, f"The catalog line of {line['source']}")
        except Exception:
            line.pop("file", None)
            raise
        if data is not None:
            _write_durably(self.path / line["file"], data)
        with open(self.catalog_path, "a", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _as_failed(line: Dict[str, Any], exc: BaseException, secrets: Sequence[str]) -> Dict[str, Any]:
        """`line` as a failed line: its identity and range, the error (redacted), and its config only when that can be written."""
        keep = ("format", "source", "class", "started_at", "fetched_at", "start", "end", "day_zone")
        failed = {name: line[name] for name in keep if name in line}
        failed["status"] = FAILED
        failed["error"] = _redact(f"{type(exc).__name__}: {exc}", secrets)
        config = line.get("config")
        try:
            _refuse_secrets(json.dumps(config, allow_nan=False).encode(), secrets, "config")
        except (ValueError, TypeError):
            config = None
        failed["config"] = config
        failed["epftoolbox2"] = line.get("epftoolbox2")
        return failed

    def _last_line(self, handle) -> Tuple[int, bytes, bool]:
        """(offset, bytes, ended) of the catalog's last line: where it starts, its bytes without the newline, and whether it
        ends in one. (0, b"", True) for an empty file."""
        size = handle.seek(0, os.SEEK_END)
        if size == 0:
            return 0, b"", True
        handle.seek(size - 1)
        ended = handle.read(1) == b"\n"
        stop = size - 1 if ended else size
        begin = stop
        while begin > 0:  # back to the newline before it, a block at a time
            lo = max(0, begin - 65536)
            handle.seek(lo)
            found = handle.read(begin - lo).rfind(b"\n")
            if found >= 0:
                begin = lo + found + 1
                break
            begin = lo
        handle.seek(begin)
        return begin, handle.read(stop - begin), ended

    def _end_last_line(self) -> None:
        """The catalog ending in a newline, so the next line starts on a line of its own: a whole last line without its newline
        (it reads as JSON, as the readers take it) gets one; a torn one (a crash or a failed write while appending) is cut off."""
        try:
            handle = open(self.catalog_path, "rb+")
        except FileNotFoundError:
            return
        with handle:
            begin, last, ended = self._last_line(handle)
            if ended:
                return
            try:
                whole = isinstance(json.loads(last), dict)
            except ValueError:
                whole = False
            if whole:
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
            else:
                logger.warning(f"Archive: cut a torn last line ({len(last)} bytes) off {self.catalog_path}")
                handle.truncate(begin)
            handle.flush()
            os.fsync(handle.fileno())

    def _is_last_line(self, line: Dict[str, Any]) -> bool:
        """Whether the catalog's last line is `line` (it reached the file whole before an error)."""
        try:
            handle = open(self.catalog_path, "rb")
        except FileNotFoundError:
            return False
        with handle:
            _, last, _ = self._last_line(handle)
        try:
            return bool(last) and json.loads(last) == json.loads(json.dumps(line, allow_nan=False))
        except (ValueError, TypeError):
            return False

    # -- reading ----------------------------------------------------------------------------------------------------------

    def _lines(self) -> List[Dict[str, Any]]:
        """Every catalog line in the order written; a torn last line (a crash while appending) is left out with a warning."""
        try:
            text = self.catalog_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ArchiveError(
                f"{self.catalog_path} doesn't exist: nothing has been recorded into {self.path}"
            ) from None
        raw = text.split("\n")
        out = []
        for number, chunk in enumerate(raw, start=1):
            if not chunk.strip():
                continue
            try:
                line = json.loads(chunk)
            except ValueError:
                if number == len(raw):  # the last chunk, not ended by a newline
                    logger.warning(f"Archive: {self.catalog_path} ends in a torn line; it is left out")
                    continue
                raise ArchiveError(f"{self.catalog_path}: line {number} isn't JSON") from None
            if int(line.get("format", 1)) > ARCHIVE_FORMAT:
                raise ArchiveError(
                    f"{self.catalog_path}: line {number} has format {line['format']}, newer than this version reads "
                    f"({ARCHIVE_FORMAT}); upgrade epftoolbox2"
                )
            out.append(line)
        return out

    def catalog(self, source: Optional[str] = None) -> pd.DataFrame:
        """The catalog as a table, one row per line in the order written: `source` (the key), `class`, `status`, `started_at`,
        `fetched_at`, `start`, `end` (UTC timestamps), `day_zone`, `rows`, `columns`, `file`, `sha256`, `settled_until`,
        `config`, `error`. `source` keeps one key's lines (a key, or a class name that names one key)."""
        lines = self._lines()
        if source is not None:
            key = self._resolve(source, lines)
            lines = [line for line in lines if line.get("source") == key]
        table = pd.DataFrame(
            [{name: line.get(name) for name in _CATALOG_COLUMNS} for line in lines], columns=_CATALOG_COLUMNS
        )
        for name in ("started_at", "fetched_at", "start", "end", "settled_until"):
            table[name] = pd.to_datetime(table[name], utc=True)
        table["warnings"] = [list(value) if isinstance(value, list) else [] for value in table["warnings"]]
        return table

    def keys(self) -> List[str]:
        """The source keys the archive holds, in the order of their first line."""
        return list(dict.fromkeys(line["source"] for line in self._lines() if line.get("source")))

    def resolve(self, source: str) -> str:
        """The key `source` names: a key, or a class name ("EntsoeSource") with exactly one key in the archive."""
        return self._resolve(source, self._lines())

    def _resolve(self, source: str, lines: List[Dict[str, Any]]) -> str:
        keys = list(dict.fromkeys(line["source"] for line in lines if line.get("source")))
        if source in keys:
            return source
        matches = [key for key in keys if key.rsplit("-", 1)[0] == source]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ArchiveError(
                f"{self.path} holds {len(matches)} configurations of {source} ({', '.join(matches)}): name one by its key "
                "(SnapshotArchive.catalog() shows each key's config)"
            )
        raise ArchiveError(f"{self.path} holds no source {source!r}; its keys are {keys}")

    def fetches(self, source: str) -> List[Fetch]:
        """The ok fetches of one source (a key or a class name with one key), ordered by when they returned and then by the
        order written."""
        return self._fetches(source, self._lines())

    def _fetches(self, source: str, lines: List[Dict[str, Any]]) -> List[Fetch]:
        key = self._resolve(source, lines)
        out = []
        for order, line in enumerate(lines):
            if line.get("source") != key or line.get("status") != OK:
                continue
            out.append(
                Fetch(
                    key=key,
                    fetched_at=_utc(line["fetched_at"]),
                    start=_utc(line["start"]),
                    end=_utc(line["end"]),
                    file=line["file"],
                    sha256=line["sha256"],
                    columns=tuple(line.get("columns") or ()),
                    order=order,
                )
            )
        out.sort(key=lambda fetch: (fetch.fetched_at, fetch.order))
        return out

    def read(self, fetch: Fetch, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """The frame of one fetch as stored (only `columns`, when given), its file's bytes checked against the catalog's
        sha256 first."""
        data = (self.path / fetch.file).read_bytes()
        if hashlib.sha256(data).hexdigest() != fetch.sha256:
            raise ArchiveError(f"{self.path / fetch.file} doesn't match the sha256 its catalog line recorded")
        frame = pd.read_parquet(io.BytesIO(data), columns=columns)
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ArchiveError(f"{self.path / fetch.file} isn't indexed by UTC instants")
        return frame

    def _key_of_column(self, column: str, source: Optional[str], lines: List[Dict[str, Any]]) -> str:
        if source is not None:
            return self._resolve(source, lines)
        holders = []
        for line in lines:
            if line.get("status") == OK and column in (line.get("columns") or ()) and line["source"] not in holders:
                holders.append(line["source"])
        if len(holders) == 1:
            return holders[0]
        if not holders:
            raise ArchiveError(f"no fetch in {self.path} holds a column {column!r}")
        raise ArchiveError(f"{column!r} is a column of {len(holders)} sources in {self.path} ({holders}): pass source=")

    def observations(self, column: str, source: Optional[str] = None) -> pd.DataFrame:
        """Every value `column` had in a fetch: one row per (cell, fetch) with a value, inside the range the fetch asked for,
        with `instant` (the cell, UTC), `fetched_at` and `value`, ordered by cell and then by fetch. `source` names the source
        when several hold the column."""
        lines = self._lines()  # once per call
        key = self._key_of_column(column, source, lines)
        instants, fetched, order, values = [], [], [], []
        for fetch in self._fetches(key, lines):
            if column not in fetch.columns:
                continue
            series = self.read(fetch, columns=[column])[column]
            series = series[(series.index >= fetch.start) & (series.index < fetch.end)].dropna()
            if series.index.has_duplicates:
                raise ArchiveError(f"{self.path / fetch.file} holds {column!r} twice for one instant")
            instants.append(series.index.as_unit("ns").asi8)
            fetched.append(np.full(len(series), fetch.fetched_at.as_unit("ns").value, dtype=np.int64))
            order.append(np.full(len(series), fetch.order, dtype=np.int64))
            values.append(series.to_numpy())
        if not instants:
            return pd.DataFrame(
                {
                    "instant": pd.DatetimeIndex([], tz="UTC"),
                    "fetched_at": pd.DatetimeIndex([], tz="UTC"),
                    "value": np.array([], dtype=float),
                }
            )
        frame = pd.DataFrame(
            {
                "instant": pd.to_datetime(np.concatenate(instants), utc=True),
                "fetched_at": pd.to_datetime(np.concatenate(fetched), utc=True),
                "_order": np.concatenate(order),
                "value": np.concatenate(values),
            }
        )
        frame = frame.sort_values(["instant", "fetched_at", "_order"], kind="stable").reset_index(drop=True)
        return frame.drop(columns="_order")

    def first_seen(self, column: str, source: Optional[str] = None) -> pd.DataFrame:
        """Each cell's first fetch that had a value: indexed by `instant` (the cell, UTC), with `fetched_at` (when that fetch
        returned) and `value` (what it held then). A cell no fetch had a value for isn't there."""
        seen = self.observations(column, source)
        first = seen.drop_duplicates("instant", keep="first")
        return first.set_index("instant")[["fetched_at", "value"]]

    def rewrites(self, column: str, source: Optional[str] = None) -> pd.DataFrame:
        """The cells whose value changed between fetches: one row per change, indexed by `instant`, with the fetch before
        (`before_fetched_at`, `before`) and the fetch that held the new value (`after_fetched_at`, `after`), compared exactly.
        Only fetches with a value count: a cell missing from a fetch is no change (a later value equal to the one before isn't
        one either)."""
        seen = self.observations(column, source)
        by_cell = seen.groupby("instant", sort=False)
        before, before_at = by_cell["value"].shift(1), by_cell["fetched_at"].shift(1)
        changed = (by_cell.cumcount() > 0).to_numpy() & (seen["value"] != before).to_numpy()
        out = pd.DataFrame(
            {
                "before_fetched_at": before_at[changed].to_numpy(),
                "before": before[changed].to_numpy(),
                "after_fetched_at": seen["fetched_at"][changed].to_numpy(),
                "after": seen["value"][changed].to_numpy(),
            },
            index=pd.DatetimeIndex(seen["instant"][changed], name="instant"),
        )
        for name in ("before_fetched_at", "after_fetched_at"):
            out[name] = pd.to_datetime(out[name], utc=True)
        return out

    def content_digest(self) -> Optional[str]:
        """The sha256 of the catalog's bytes (None without one): it changes with every fetch recorded."""
        try:
            return hashlib.sha256(self.catalog_path.read_bytes()).hexdigest()
        except FileNotFoundError:
            return None


def record_snapshot(
    config: Any, archive: Union[str, os.PathLike], *, days_back: int = 3, days_ahead: int = 2
) -> RecordResult:
    """`SnapshotArchive(archive).record(...)` of a pipeline file, for a scheduled job (and the `epf snapshot` command):
    `config` is a `DataPipeline` YAML, a `Workflow` YAML (its `data_pipeline`), or a `DataPipeline`. Secrets come from the
    environment as `${ENTSOE_API_KEY}` references in the file say."""
    from epftoolbox2.pipelines.data_pipeline import DataPipeline

    if isinstance(config, DataPipeline):
        pipeline = config
    else:
        import yaml

        path = Path(os.fspath(config))
        raw = yaml.safe_load(path.read_text()) or {}
        if isinstance(raw, dict) and "data_pipeline" in raw:
            from epftoolbox2.pipelines.workflow import Workflow

            pipeline = Workflow.load(path)._build_data_pipeline()
        else:
            pipeline = DataPipeline.load(path)
    return SnapshotArchive(archive).record(pipeline, days_back=days_back, days_ahead=days_ahead)


def _iter_days(start: pd.Timestamp, end: pd.Timestamp, zone: str) -> Iterator[pd.Timestamp]:
    """The naive local dates of `zone` that [start, end) touches."""
    first = pd.Timestamp(start.tz_convert(zone).date())
    last = pd.Timestamp((end - pd.Timedelta(1, "ns")).tz_convert(zone).date())
    return iter(pd.date_range(first, last, freq="D"))
