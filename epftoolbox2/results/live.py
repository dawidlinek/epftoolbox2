"""Live directories (step 8c of the forecasting layer): revisions, the commit log, deliveries and the readers' policies.

A live pipeline writes into a directory of its own, never into a backtest's:

    live/de_price/
      live.json                          what the directory is: mode "live", and what the live calls record in it (a
                                         configuration digest, frozen choices, the backtest it came from, `issue_timezone`,
                                         `deadline`)
      live.lock                          held by every forecast, refit and mark-delivered call (flock)
      <model>.jsonl | <model>.parquet    every revision's records, append-only (the 5a and 5b formats)
      <model>.manifest.json              mode "live" (`results.manifest.check_and_write_live`)
      issues.jsonl                       one line per call: issue, call, started, finished, status, exit_code, revision,
                                         committed, and for a commit issued_at, snapshot and deadline; and before it, a
                                         "begun" line of each revision, which takes its number
      delivered.jsonl                    one line per delivery: issue, revision, delivered_at, receipt_sha256, receipt_path
      issues/<issue>/r<n>/               a revision's own files (snapshots), made before anything else of it

**Revisions.** A call for issue day D writes revision r = 1 + the highest revision any line of D carries, committed or not: a
call line, a revision folder `issues/D/r<n>`, or a record of `run_date` D in any model's results. `LiveStore.begin` appends a
"begun" line, then makes the folder, so a number is taken before any record of it is written and is never taken again, whatever
a crash leaves; a visible record whose `issued_at` or `snapshot` isn't its commit line's raises.
Every record of the revision carries `revision` r and, after D4's keys, `issued_at` (UTC), `mode` "live" and `snapshot` (the
hash of its inputs, `with_live`). Then the snapshot files, then one line in `issues.jsonl` with `committed: true` (appended and
synced, `LiveRevision.commit`): only that makes the revision visible. A crash before it leaves lines no reader sees.

**Readers** (`LiveStore.visible`, `results.store.read_records(committed=...)`, `ModelResultRef(policy=..., cutoff=...)`). Per key
`(hour, horizon, run_date)` the highest visible revision under the readers' rule of 5a. What is visible is decided per issue day,
from the committed revisions whose commit (the line's `finished`) is at or before the reader's cut-off on that day:

- `"latest"`: all of them, so the newest committed one wins.
- `"first"`: the lowest of them alone.
- `"delivered"`: those up to the revision of the issue's newest delivery at or before the cut-off; an issue day without one has
  nothing.

The cut-off is `"deadline"` by default: the deadline `live.json` records (`deadline` "HH:MM" in `issue_timezone`) on each issue
day, and a revision committed after the deadline its own call recorded is out too. None reads every committed revision; an
instant, "HH:MM" (in `issue_timezone`), a mapping of issue day to instant or a function of the issue day give others. A revision
committed after the cut-off is never seen, which for a chained model would be a look-ahead (`ForecastSource`).

**Refusals.** A backtest run into a live directory, a live revision's store outside one, a live directory made where a backtest's
results are, and a reader of a live directory's records that doesn't say which revisions it may see, all raise
`LiveDirectoryError`. A call that finds `live.lock` held raises `LiveBusyError` (status "busy", exit code 75) and writes nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd

from .store import (
    LIVE_FILE,
    LIVE_KEYS,
    LIVE_LOCK_STEM,
    LIVE_LOG_STEMS,
    MODE_LIVE,
    NOTHING_VISIBLE,
    SAVE_FORMATS,
    LiveDirectoryError,
    ResultStore,
    StoreWriter,
    _hold_lock,
    _instant,
    _LockHeld,
    _preview,
    _Store,
    check_no_other_format,
    check_save_format,
    cutoff_of_day,
    is_live_dir,
    issue_day,
    is_parquet_path,
    max_revision,
    outside_stacklevel,
    visible_revisions,
)

LIVE_LOCK = f"{LIVE_LOCK_STEM}.lock"
ISSUES = f"{LIVE_LOG_STEMS[0]}.jsonl"
DELIVERED = f"{LIVE_LOG_STEMS[1]}.jsonl"
REVISIONS_DIR = "issues"  # issues/<issue>/r<n>/
POLICIES = ("latest", "first", "delivered")
DEADLINE = "deadline"
BEGUN = "begun"  # the status of the line `begin` writes before any record of a revision: its number is taken
_TIME = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
# Backtest files a live directory is never made beside: model results and manifests, and a run over zones' index.
_ZONE_INDEX = "zone_run.json"


class LiveBusyError(RuntimeError):
    """Another call holds the live directory's `live.lock`: this one wrote nothing."""

    status = "busy"
    exit_code = 75  # EX_TEMPFAIL


class LiveLateError(RuntimeError):
    """A revision would be committed after the deadline its call was given: it isn't."""

    status = "late"
    exit_code = 30


def now_utc() -> pd.Timestamp:
    """The instant now, UTC (the clock step 8a routes)."""
    return pd.Timestamp.now(tz="UTC")


def local_instant(day: str, time: str, zone: str) -> pd.Timestamp:
    """The instant of wall-clock `time` ("HH:MM") on `day` in `zone`, UTC: a time a spring switch skips moves forward to the
    switch, a repeated autumn one is its first occurrence (as `information_set._localize` reads them)."""
    if not isinstance(time, str) or not _TIME.match(time.strip()):
        raise ValueError(f"a time of day is 'HH:MM', got {time!r}")
    wall = pd.Timestamp(f"{issue_day(day)}T{time.strip()}")
    return wall.tz_localize(zone, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")


def _iso(stamp: pd.Timestamp) -> str:
    return stamp.isoformat()


def with_live(record: Dict[str, Any], revision: int, issued_at: str, snapshot: Optional[str]) -> Dict[str, Any]:
    """A copy of `record` as a live revision writes it: `revision` set where the record has it (D4's keys keep their places),
    then `issued_at`, `mode` "live" and `snapshot` after every other key."""
    out = {key: value for key, value in record.items() if key not in LIVE_KEYS}
    out["revision"] = int(revision)
    out["issued_at"] = issued_at
    out["mode"] = MODE_LIVE
    out["snapshot"] = snapshot
    return out


# --- a live revision's stores -----------------------------------------------------------------------------------------------


class _LiveRecords:
    """What a live revision's store does on top of its format's: it writes into a live directory alone (`_Store._live`), each
    record stamped with the revision's number, issue instant and snapshot (`with_live`), and it reads nothing back at its
    opening: a revision writes each key once at its own number, and nothing on file decides it (a backtest's resume state
    would read the file without knowing what is committed)."""

    _live = True
    _stamp: Tuple[int, str, Optional[str]]

    def _load_existing(self) -> None:
        return None

    def save(self, result: Dict) -> None:
        super().save(with_live(result, *self._stamp))

    def save_skip(self, record: Dict) -> bool:
        return super().save_skip(with_live(record, *self._stamp))


class LiveResultStore(_LiveRecords, ResultStore):
    """A live revision's JSONL store (`LiveRevision.open`)."""

    def __init__(self, path: Any, stamp: Tuple[int, str, Optional[str]], *, lock: bool = True, **kwargs: Any):
        self._stamp = stamp
        super().__init__(path, lock=lock, **kwargs)


_PARQUET_CLASS: List[type] = []


def live_parquet_store_class() -> type:
    """`LiveParquetResultStore`, a live revision's Parquet store (`LiveRevision.open`), made on first use: the Parquet store's
    module (and `pyarrow.parquet`) load only when Parquet results are used."""
    if not _PARQUET_CLASS:
        from .parquet_store import ParquetResultStore

        class LiveParquetResultStore(_LiveRecords, ParquetResultStore):
            """A live revision's Parquet store (`LiveRevision.open`)."""

            def __init__(self, path: Any, stamp: Tuple[int, str, Optional[str]], *, lock: bool = True, **kwargs: Any):
                self._stamp = stamp
                super().__init__(path, lock=lock, **kwargs)

        _PARQUET_CLASS.append(LiveParquetResultStore)
    return _PARQUET_CLASS[0]


def _sync_directory(directory: Path) -> None:
    from .parquet_store import _sync_directory as sync

    sync(directory)


# --- the logs ---------------------------------------------------------------------------------------------------------------


def _read_log(path: Path) -> List[Dict[str, Any]]:
    """The lines of a live directory's log (`issues.jsonl`, `delivered.jsonl`) as dicts. A last line without its newline that
    isn't one is what a crash while appending leaves: skipped with a warning (the next writer cuts it off), so a half-written
    commit commits nothing. A damaged line before it raises ValueError naming its number."""
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.split(b"\n")
    out = []
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        try:
            entry = json.loads(line.decode("utf-8"))
            if not isinstance(entry, dict):
                raise ValueError(f"a JSON {type(entry).__name__}, not an object")
        except ValueError as exc:
            if number == len(lines):  # no newline after it
                warnings.warn(
                    f"{path}: its last line ({len(raw)} bytes) is not a whole line, as a call that stopped while writing it "
                    f"leaves it: {_preview(line)}. It is left out (a revision it would commit stays uncommitted); the next "
                    "call cuts it off.",
                    UserWarning,
                    stacklevel=outside_stacklevel(),
                )
                break
            raise ValueError(f"{path}: line {number} is not a log line ({exc}): {_preview(line)}") from None
        out.append(entry)
    return out


def _cut_torn_tail(path: Path) -> None:
    """Before appending: end a whole last line that lost only its newline with one; cut off one a crash left half-written."""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    start = data.rfind(b"\n") + 1
    tail = data[start:]
    try:
        whole = isinstance(json.loads(tail.strip().decode("utf-8")), dict)
    except ValueError:
        whole = False
    with open(path, "rb+") as f:
        if whole:
            f.seek(len(data))
            f.write(b"\n")
            return
        f.truncate(start)
    warnings.warn(
        f"{path}: cut off its last {len(tail)} bytes (from byte {start}), a line a call that stopped while writing left "
        f"half-written: {_preview(tail.strip())}.",
        UserWarning,
        stacklevel=outside_stacklevel(),
    )


def _append_line(path: Path, entry: Dict[str, Any]) -> None:
    """Append `entry` as one line, synced (and the directory, when this makes the file)."""
    _cut_torn_tail(path)
    created = not path.exists()
    with open(path, "ab") as f:
        f.write((json.dumps(entry) + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    if created:
        _sync_directory(path.parent)


def _as_json(value: Any) -> Any:
    """`value` as `live.json` holds it once written and read back (`default=str` for what JSON has no type for)."""
    return json.loads(json.dumps(value, default=str))


def _fsync_file(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class Commit:
    """A committed revision as its `issues.jsonl` line says: `finished` is when the line was written, the instant the revision
    became visible; `deadline` the one its call was given, if any."""

    issue: str
    revision: int
    finished: pd.Timestamp
    issued_at: Optional[pd.Timestamp]
    deadline: Optional[pd.Timestamp]
    line: Dict[str, Any] = field(repr=False, compare=False)

    @property
    def marks(self) -> Optional[Tuple[Optional[str], Optional[str]]]:
        """(issued_at, snapshot) as the commit line writes them, which every record of the revision carries; None for a line
        without `issued_at` (written by hand), whose records aren't checked (`results.store.read_records`)."""
        if self.line.get("issued_at") is None:
            return None
        return self.line["issued_at"], self.line.get("snapshot")


# --- the directory ----------------------------------------------------------------------------------------------------------


def refuse_backtest_files(save_dir: Path) -> None:
    """Raise `LiveDirectoryError` where `save_dir` holds a backtest's results, manifests or zone index: a live directory is
    never made beside them."""
    if not save_dir.is_dir():
        return
    found = sorted(
        p.name
        for p in save_dir.iterdir()
        if p.name.endswith(".manifest.json")
        or p.name == _ZONE_INDEX
        or (p.suffix == SAVE_FORMATS["jsonl"] and p.is_file() and p.stem not in LIVE_LOG_STEMS)
        or (is_parquet_path(p) and p.is_dir())
    )
    if found:
        raise LiveDirectoryError(
            f"{save_dir} holds a backtest's results ({', '.join(found[:5])}{', ...' if len(found) > 5 else ''}), so it "
            "can't become a live directory: live revisions go into a directory of their own. Pass another save_dir."
        )


def refuse_live_directory(save_dir: Any, what: str = "run()") -> None:
    """Raise `LiveDirectoryError` where `save_dir` is a live directory, which `what` (a backtest's writer or reader) mustn't
    use."""
    if save_dir and is_live_dir(save_dir):
        raise LiveDirectoryError(
            f"{save_dir} is a live directory ({Path(save_dir) / LIVE_FILE}): {what} writes and reads a backtest's results, "
            "and a live directory's are revisions, visible once committed. Run the backtest into another save_dir; read the "
            "live records with ModelResultRef(path, ..., policy='latest' | 'first' | 'delivered')."
        )


class LiveStore:
    """A live directory's revisions, commit log and deliveries (internal: the live calls write through it, the readers read
    their visibility from it). See the module docstring."""

    def __init__(self, save_dir: Any):
        self.save_dir = Path(os.path.expanduser(os.fspath(save_dir)))
        self._release: Optional[Callable[[], None]] = None
        self._begun: List["LiveRevision"] = []  # the revisions begun under the lock held now

    # -- paths and identity --------------------------------------------------------------------------------------------

    @property
    def identity_path(self) -> Path:
        return self.save_dir / LIVE_FILE

    @property
    def lock_path(self) -> Path:
        return self.save_dir / LIVE_LOCK

    @property
    def issues_path(self) -> Path:
        return self.save_dir / ISSUES

    @property
    def delivered_path(self) -> Path:
        return self.save_dir / DELIVERED

    def revision_dir(self, issue: Any, revision: int) -> Path:
        return self.save_dir / REVISIONS_DIR / self.day(issue) / f"r{int(revision)}"

    def zone(self) -> Optional[str]:
        """The issue zone `live.json` records (`issue_timezone`), None without one."""
        return self.identity().get("issue_timezone") if self.is_live() else None

    def day(self, issue: Any) -> str:
        """`issue` as an issue day (`results.store.issue_day`): an aware instant is read in the issue zone, and raises without
        one."""
        return issue_day(issue, self.zone())

    def is_live(self) -> bool:
        return is_live_dir(self.save_dir)

    def identity(self) -> Dict[str, Any]:
        """`live.json` as a dict; `LiveDirectoryError` where there is none."""
        if not self.is_live():
            raise LiveDirectoryError(f"{self.save_dir} is no live directory (no {LIVE_FILE})")
        data = json.loads(self.identity_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("mode") != MODE_LIVE:
            raise LiveDirectoryError(
                f"{self.identity_path} doesn't say mode 'live': {_preview(json.dumps(data).encode())}"
            )
        return data

    def init(self, **identity: Any) -> Dict[str, Any]:
        """Make `save_dir` a live directory with `identity` in `live.json`, or check an existing one: a key it holds with
        another value raises `LiveDirectoryError` (another pipeline's directory), a key it lacks is added. A directory holding a
        backtest's results is refused. Values are compared as `live.json` holds them, after the same JSON round trip (a tuple is
        a list, a Timestamp its text, a dict's int keys text), so what was given yesterday compares equal today. Returns what
        `live.json` holds."""
        identity = {key: _as_json(value) for key, value in identity.items()}
        if self.is_live():
            stored = self.identity()
            differ = {
                key: (stored[key], value) for key, value in identity.items() if key in stored and stored[key] != value
            }
            if differ:
                raise LiveDirectoryError(
                    f"{self.identity_path} was made for another live pipeline: "
                    + "; ".join(f"{key}: {held!r} on file, {given!r} now" for key, (held, given) in differ.items())
                    + ". Use another save_dir for this one."
                )
            added = {key: value for key, value in identity.items() if key not in stored}
            if not added:
                return stored
            stored = {**stored, **added}
        else:
            refuse_backtest_files(self.save_dir)
            if "mode" in identity and identity["mode"] != MODE_LIVE:
                raise ValueError(f"a live directory's mode is 'live', got {identity['mode']!r}")
            stored = {"mode": MODE_LIVE, **{key: value for key, value in identity.items() if key != "mode"}}
        from ..data.cache_manager import atomic_write

        self.save_dir.mkdir(parents=True, exist_ok=True)

        def write(path: Path) -> None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(stored, indent=2, sort_keys=False, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())

        atomic_write(self.identity_path, write)
        _sync_directory(self.save_dir)
        return stored

    def results(self) -> List[Path]:
        """Every model's results in the directory: `<name>.jsonl` files but the logs, `<name>.parquet` directories."""
        if not self.save_dir.is_dir():
            return []
        return sorted(
            p
            for p in self.save_dir.iterdir()
            if (p.suffix == SAVE_FORMATS["jsonl"] and p.is_file() and p.stem not in LIVE_LOG_STEMS)
            or (is_parquet_path(p) and p.is_dir())
        )

    # -- the lock -------------------------------------------------------------------------------------------------------

    @contextmanager
    def hold(self, create: bool = True) -> Iterator["LiveStore"]:
        """Hold `live.lock` (`flock`, taken without waiting) for the block; `LiveBusyError` when another call holds it, in
        another process or through another `LiveStore` in this one. `create=False` (a dry run, which writes nothing) makes
        neither the directory nor the lock file: without one, no call holds it. A directory holding a backtest's results
        raises `LiveDirectoryError`, before anything is made in it."""
        if self._release is not None:
            raise RuntimeError(f"{self.lock_path} is already held by this LiveStore")
        if not create and not self.lock_path.exists():
            yield self
            return
        if not self.is_live():
            refuse_backtest_files(self.save_dir)  # no live.lock beside a backtest's results either
        self.save_dir.mkdir(parents=True, exist_ok=True)
        try:
            release = _hold_lock(self.lock_path)
        except _LockHeld:
            raise LiveBusyError(
                f"{self.lock_path} is held by another call (a forecast, refit or mark-delivered of this live directory): "
                "nothing was written. Try again once it has finished."
            ) from None
        except OSError as exc:
            warnings.warn(
                f"{self.lock_path}: the file system refused a lock ({exc}), so a second call into this live directory at "
                "the same time would not be refused. Run one call at a time here.",
                UserWarning,
                stacklevel=outside_stacklevel(),
            )
            release = None
        self._release = release or (lambda: None)
        try:
            yield self
        finally:
            try:
                self._abandon_unfinished()
            finally:
                done, self._release = self._release, None
                done()

    def _abandon_unfinished(self) -> None:
        """Close the stores of every revision begun under this hold that was neither committed, failed nor abandoned (the block
        raised, or ended without finishing it), before `live.lock` is let go: no store of the directory keeps its lock or a
        writer thread past the call."""
        begun, self._begun = self._begun, []
        error = None
        for revision in begun:
            if not revision._done:
                try:
                    revision.abandon()
                except BaseException as exc:  # every revision's stores, then the first failure
                    error = error or exc
        if error is not None:
            raise error

    def _require_held(self, what: str) -> None:
        if self._release is None:
            raise RuntimeError(f"{what} needs {self.lock_path} held: call it inside `with LiveStore(...).hold():`")

    # -- the commit log ---------------------------------------------------------------------------------------------------

    def calls(self) -> List[Dict[str, Any]]:
        """Every line of `issues.jsonl`, in order (a torn last line left out)."""
        return _read_log(self.issues_path)

    def committed(self, issue: Any = None) -> Dict[str, Dict[int, Commit]]:
        """Issue day -> revision -> its `Commit`, from the lines with `committed: true` (of `issue` alone, if given). A revision
        committed twice, or a commit line without its issue, revision or instant, raises ValueError: the log was damaged."""
        wanted = None if issue is None else self.day(issue)
        out: Dict[str, Dict[int, Commit]] = {}
        for number, line in enumerate(self.calls(), 1):
            if line.get("committed") is not True:
                continue
            day, revision, finished = line.get("issue"), line.get("revision"), line.get("finished")
            if not isinstance(day, str) or isinstance(revision, bool) or not isinstance(revision, int) or not finished:
                raise ValueError(
                    f"{self.issues_path}: commit line {number} lacks its issue, revision or finished: {line}"
                )
            if wanted is not None and day != wanted:
                continue
            held = out.setdefault(day, {})
            if revision in held:
                raise ValueError(f"{self.issues_path}: revision {revision} of {day} is committed twice (line {number})")
            held[revision] = Commit(
                issue=day,
                revision=revision,
                finished=_instant(finished, "finished"),
                issued_at=None if line.get("issued_at") is None else _instant(line["issued_at"], "issued_at"),
                deadline=None if line.get("deadline") is None else _instant(line["deadline"], "deadline"),
                line=line,
            )
        return out

    def next_revision(self, issue: Any) -> int:
        """1 + the highest revision any line of the issue day carries, committed or not, so a crashed revision's number is
        never used again: a call line (`begin` writes one, status "begun", before anything else of a revision), a revision
        folder (made next), and a record of that `run_date` in any model's results (`results.store.max_revision`). The
        records are read only for a day without any folder (records written by hand, or a directory from before the
        folders): reading every record of every model on each call costs about 0.65 s per model-year of hourly JSONL. A number
        can be taken again only if both its folder and its log lines are removed by hand; a crashed revision's records then
        carry another `issued_at` than the new commit's line, and reading them raises (`results.store.check_marks`)."""
        day = self.day(issue)
        highest = 0
        for line in self.calls():
            revision = line.get("revision")
            if line.get("issue") == day and isinstance(revision, int) and not isinstance(revision, bool):
                highest = max(highest, revision)
        folder = self.save_dir / REVISIONS_DIR / day
        reserved = [
            int(child.name[1:])
            for child in (folder.iterdir() if folder.is_dir() else ())
            if child.name[:1] == "r" and child.name[1:].isdigit()
        ]
        if reserved:
            return max(highest, *reserved) + 1
        for path in self.results():
            highest = max(highest, max_revision(path, day))
        return highest + 1

    def begin(
        self,
        issue: Any,
        *,
        issued_at: Any = None,
        snapshot: Optional[str] = None,
        call: Optional[str] = None,
        started: Any = None,
    ) -> "LiveRevision":
        """Take the next revision of `issue` (`next_revision`) by making its folder `issues/<issue>/r<n>/`, before any of its
        records: the number is taken whatever happens next. Under the lock (`hold`). `issued_at` (an instant, now by default) and
        `snapshot` go into every record of it; `started` (the call's start, `issued_at` by default) and `call` (an id, made
        if not given) into its call line."""
        self._require_held("begin()")
        if not self.is_live():
            raise LiveDirectoryError(f"{self.save_dir} is no live directory (no {LIVE_FILE}): init() it first")
        day = self.day(issue)
        issued = _instant(now_utc() if issued_at is None else issued_at, "issued_at")
        revision = self.next_revision(day)
        begun = LiveRevision(
            store=self,
            issue=day,
            revision=revision,
            issued_at=_iso(issued),
            snapshot=snapshot,
            call=call or uuid.uuid4().hex[:12],
            started=_iso(issued if started is None else _instant(started, "started")),
        )
        # The number is taken by a line in the log first, then by the folder: either alone keeps it taken, whatever happens
        # to the other (a failed call's folder removed by hand, say).
        _append_line(
            self.issues_path,
            {
                "issue": day,
                "call": begun.call,
                "started": begun.started,
                "status": BEGUN,
                "revision": revision,
                "committed": False,
            },
        )
        folder = self.revision_dir(day, revision)
        made = [p for p in (folder.parent.parent, folder.parent) if not p.exists()]
        folder.mkdir(parents=True)  # exists: someone took the number meanwhile, which the lock forbids
        for directory in (*made, folder):
            _sync_directory(directory.parent)
        self._begun.append(begun)
        return begun

    def record_call(
        self,
        issue: Any,
        status: str,
        exit_code: int,
        *,
        revision: Optional[int] = None,
        call: Optional[str] = None,
        started: Any = None,
        finished: Any = None,
        **fields: Any,
    ) -> Dict[str, Any]:
        """Append the line of a call that commits nothing (late, unchanged, noop, an error), under the lock: `committed` false.
        Only `LiveRevision.commit` commits."""
        self._require_held("record_call()")
        if "committed" in fields:
            raise ValueError(
                "record_call() writes uncommitted lines only: commit a revision with LiveRevision.commit()"
            )
        end = _instant(now_utc() if finished is None else finished, "finished")
        line = {
            "issue": self.day(issue),
            "call": call or uuid.uuid4().hex[:12],
            "started": _iso(end if started is None else _instant(started, "started")),
            "finished": _iso(end),
            "status": status,
            "exit_code": int(exit_code),
            "revision": None if revision is None else int(revision),
            "committed": False,
            **fields,
        }
        _append_line(self.issues_path, line)
        return line

    # -- deliveries -------------------------------------------------------------------------------------------------------

    def deliveries(self, issue: Any = None) -> List[Dict[str, Any]]:
        """The lines of `delivered.jsonl` in order (of `issue` alone, if given)."""
        lines = _read_log(self.delivered_path)
        return lines if issue is None else [line for line in lines if line.get("issue") == self.day(issue)]

    def mark_delivered(
        self, issue: Any, revision: int, receipt: Any, *, dry_run: bool = False, delivered_at: Any = None
    ) -> Dict[str, Any]:
        """Record that `revision` of `issue` was delivered: a line in `delivered.jsonl` with the receipt file's sha256 and
        absolute path. An unknown or uncommitted revision raises ValueError, a missing receipt FileNotFoundError. A dry run checks
        the same and writes nothing. Under the lock (`hold`), except a dry run. Returns the line."""
        if not dry_run:
            self._require_held("mark_delivered()")
        if not self.is_live():
            raise LiveDirectoryError(f"{self.save_dir} is no live directory (no {LIVE_FILE})")
        day = self.day(issue)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError(f"revision must be a positive integer, got {revision!r}")
        if revision not in self.committed(day).get(day, {}):
            if revision < self.next_revision(day):
                raise ValueError(
                    f"revision {revision} of {day} is not committed (its call didn't finish, or failed): it was never "
                    f"visible, so it can't have been delivered. Committed: {sorted(self.committed(day).get(day, {}))}."
                )
            raise ValueError(f"no revision {revision} of {day} in {self.save_dir}")
        path = Path(os.path.expanduser(os.fspath(receipt)))
        if not path.is_file():
            raise FileNotFoundError(f"the receipt {path} is not a file")
        line = {
            "issue": day,
            "revision": revision,
            "delivered_at": _iso(_instant(now_utc() if delivered_at is None else delivered_at, "delivered_at")),
            "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "receipt_path": str(path.resolve()),
        }
        if not dry_run:
            _append_line(self.delivered_path, line)
        return line

    # -- what a reader sees -----------------------------------------------------------------------------------------------

    def _cutoff(self, cutoff: Any) -> Tuple[Callable[[str], Any], bool]:
        """The cut-off of each issue day a reader's `cutoff` gives (an instant, None for none, `NOTHING_VISIBLE` for a day a
        mapping doesn't name), and whether a commit needs a deadline of its own, made by it ("deadline" with no deadline in
        `live.json`: the cut-off fails closed, never open)."""
        if isinstance(cutoff, str) and (cutoff == DEADLINE or _TIME.match(cutoff.strip())):
            identity = self.identity() if self.is_live() else {}
            time = identity.get("deadline") if cutoff == DEADLINE else cutoff.strip()
            if time is None:
                return (lambda day: None), True
            zone = identity.get("issue_timezone")
            if not zone:
                raise ValueError(
                    f"cutoff={cutoff!r} is a time of each issue day, and {self.identity_path} records no issue_timezone to "
                    "read it in: pass an instant, or a function of the issue day"
                )
            return (lambda day: local_instant(day, time, zone)), False
        return cutoff_of_day(cutoff, self.zone()), False

    def visible(
        self, policy: str = "latest", cutoff: Any = DEADLINE, *, pending: Optional["LiveRevision"] = None
    ) -> Dict[str, Dict[int, "Commit"]]:
        """What a reader of `policy` and `cutoff` sees (module docstring): issue day -> its visible revisions -> each one's
        `Commit` (`finished`, the commit instant; `marks`, what its records must carry), for
        `results.store.read_records(committed=...)`. `pending`, an uncommitted revision of this call, is read by its own writer
        alone, with `policy="latest"`, and alone on its day: a model with no record of it has none that day, so an ensemble of
        the revision must flush every member's store before it reads them."""
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {list(POLICIES)}, got {policy!r}")
        limit_of, own_deadline = self._cutoff(cutoff)
        deliveries: Dict[str, List[Dict[str, Any]]] = {}
        if policy == "delivered":
            for line in self.deliveries():
                deliveries.setdefault(str(line.get("issue")), []).append(line)
        out: Dict[str, Dict[int, pd.Timestamp]] = {}
        for day, revisions in self.committed().items():
            limit = limit_of(day)
            if limit is NOTHING_VISIBLE:
                continue
            seen = {
                revision: commit
                for revision, commit in revisions.items()
                if (limit is None or commit.finished <= limit)
                and (not own_deadline or commit.deadline is not None)
                and (cutoff != DEADLINE or commit.deadline is None or commit.finished <= commit.deadline)
            }
            if not seen:
                continue
            if policy == "first":
                head = min(seen)
                seen = {head: seen[head]}
            elif policy == "delivered":
                head = None
                for line in deliveries.get(day, ()):  # the newest delivery at or before the cut-off, of a revision seen
                    at = _instant(line["delivered_at"], "delivered_at")
                    if line.get("revision") in seen and (limit is None or at <= limit):
                        head = line["revision"]
                if head is None:
                    continue
                seen = {revision: commit for revision, commit in seen.items() if revision <= head}
            out[day] = {revision: seen[revision] for revision in sorted(seen)}
        if pending is not None:
            if policy != "latest":
                raise ValueError("pending= reads a call's own revision, the latest of its day: policy='latest'")
            issued = _instant(pending.issued_at)
            line = {"issued_at": pending.issued_at, "snapshot": pending.snapshot}
            # The pending revision alone on its day: a member it has no record of yet (not flushed) has none, never an older
            # revision's. Flush every member before reading them for an ensemble of the revision.
            out[pending.issue] = {pending.revision: Commit(pending.issue, pending.revision, issued, issued, None, line)}
        return out

    def fingerprint(self) -> Tuple[Any, ...]:
        """What says the visibility may have changed: the logs' and `live.json`'s sizes and modification times."""
        out = []
        for path in (self.identity_path, self.issues_path, self.delivered_path):
            try:
                status = path.stat()
                out.append((status.st_size, status.st_mtime_ns))
            except FileNotFoundError:
                out.append(None)
        return tuple(out)


@dataclass
class LiveRevision:
    """A revision being written (`LiveStore.begin`): `open` a store per model, write its records, then `commit` (the revision
    becomes visible), `fail` (a call line that says it didn't) or `abandon` (nothing more: what a crash leaves).

    Used as a context manager (`with store.begin(...) as rev:`), a revision the block leaves unfinished, raising or not, is
    abandoned; `LiveStore.hold` abandons any left when its block ends, before `live.lock` goes. A `commit` that raised (a store's
    write failed, `LiveLateError`, anything) leaves the revision unusable but for `fail` and `abandon`: a retry would commit it
    without what the failed one didn't write."""

    store: LiveStore
    issue: str
    revision: int
    issued_at: str
    snapshot: Optional[str]
    call: str
    started: str
    _stores: List[_Store] = field(default_factory=list, repr=False)
    _opened: List[str] = field(default_factory=list, repr=False)  # each model's stem, once
    _done: bool = field(default=False, repr=False)
    _poisoned: Optional[BaseException] = field(default=None, repr=False)  # what made a commit fail

    def __enter__(self) -> "LiveRevision":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if not self._done:
            self.abandon()

    @property
    def directory(self) -> Path:
        """`issues/<issue>/r<n>/`: where the revision's own files (snapshot, quality, checks) go."""
        return self.store.revision_dir(self.issue, self.revision)

    def open(
        self,
        model: str,
        save_format: str = "jsonl",
        save_coefficients: bool = True,
        writer: Optional[StoreWriter] = None,
    ) -> _Store:
        """A store for `model`'s records of this revision (`<name>.jsonl` or `<name>.parquet` in the live directory), holding
        the model's own lock beneath `live.lock` (a model named "live" shares `live.lock`, held already). Every record saved is
        stamped (`with_live`). A model opened twice in one revision raises ValueError: its records are written once."""
        self._check_open()
        self._check_not_poisoned("open()")
        stem = model.lower().replace(" ", "_")
        if stem in self._opened:
            raise ValueError(
                f"{model!r} is open in revision {self.revision} of {self.issue} already: one store per model"
            )
        if stem in LIVE_LOG_STEMS:
            raise ValueError(
                f"a model of a live directory can't be named {model!r}: {stem}.jsonl is the directory's log"
            )
        path = self.store.save_dir / f"{stem}{SAVE_FORMATS[check_save_format(save_format)]}"
        lock = stem != LIVE_LOCK_STEM
        if not lock:
            check_no_other_format(path)
        cls = live_parquet_store_class() if save_format == "parquet" else LiveResultStore
        store = cls(
            path,
            (self.revision, self.issued_at, self.snapshot),
            lock=lock,
            save_coefficients=save_coefficients,
            writer=writer,
        )
        self._stores.append(store)
        self._opened.append(stem)
        return store

    def _check_open(self) -> None:
        if self._done:
            raise RuntimeError(f"revision {self.revision} of {self.issue} is finished")

    def _check_not_poisoned(self, what: str) -> None:
        if self._poisoned is not None:
            raise RuntimeError(
                f"{what}: a commit of revision {self.revision} of {self.issue} failed ({self._poisoned!r}), so what it holds "
                "is unknown: fail() or abandon() it, and write a new revision"
            ) from self._poisoned

    def _close_stores(self) -> None:
        stores, self._stores = self._stores, []
        error = None
        for store in stores:
            try:
                store.close()
            except BaseException as exc:  # close every store, then raise the first failure
                error = error or exc
        if error is not None:
            raise error
        for store in stores:  # what was written survives a power cut before the line that commits it
            if not is_parquet_path(store.path):  # parts and catalogs are synced as they are written
                _fsync_file(store.path)

    def commit(
        self,
        status: str = "ok",
        exit_code: int = 0,
        *,
        finished: Any = None,
        deadline: Any = None,
        **fields: Any,
    ) -> Dict[str, Any]:
        """Close the revision's stores (a failed write raises here and nothing is committed), then append its `committed` line to
        `issues.jsonl`, synced: from then on readers see it. `finished` is the commit's instant (now by default), `deadline` the
        call's: a commit after it raises `LiveLateError` and writes nothing. Anything raised leaves the revision to `fail` or
        `abandon` alone. Returns the line."""
        self._check_open()
        self._check_not_poisoned("commit()")
        try:
            return self._commit(status, exit_code, finished, deadline, fields)
        except BaseException as exc:
            self._poisoned = exc
            raise

    def _commit(
        self, status: str, exit_code: int, finished: Any, deadline: Any, fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.store._require_held("commit()")
        self._close_stores()
        end = _instant(now_utc() if finished is None else finished, "finished")
        if end < _instant(self.issued_at):
            raise ValueError(f"a revision can't be committed ({end}) before it was issued ({self.issued_at})")
        limit = None if deadline is None else _instant(deadline, "deadline")
        if limit is not None and end > limit:  # the caller may still `fail(status="late", exit_code=30)` it
            raise LiveLateError(
                f"revision {self.revision} of {self.issue} would be committed at {end}, after its deadline {limit}: it isn't"
            )
        if "committed" in fields:
            raise ValueError("commit() writes the committed line itself")
        line = {
            "issue": self.issue,
            "call": self.call,
            "started": self.started,
            "finished": _iso(end),
            "status": status,
            "exit_code": int(exit_code),
            "revision": self.revision,
            "committed": True,
            "issued_at": self.issued_at,
            "snapshot": self.snapshot,
            **({} if limit is None else {"deadline": _iso(limit)}),
            **fields,
        }
        _append_line(self.store.issues_path, line)
        self._done = True
        return line

    def fail(self, status: str = "error", exit_code: int = 1, *, finished: Any = None, **fields: Any) -> Dict[str, Any]:
        """Close the stores without committing and append the call's line, `committed` false: the revision's records stay on
        file, never visible, and its number taken."""
        self._check_open()
        self._done = True
        try:
            self._close_stores()
        finally:
            line = self.store.record_call(
                self.issue,
                status,
                exit_code,
                revision=self.revision,
                call=self.call,
                started=self.started,
                finished=finished,
                **fields,
            )
        return line

    def abandon(self) -> None:
        """Close the stores and write nothing more, as a crash before the commit leaves the directory."""
        self._done = True
        self._close_stores()


def live_reading(
    path: Any, policy: Optional[str], cutoff: Any, days: Optional[Tuple[str, str]] = None
) -> Optional[Dict[str, Dict[int, "Commit"]]]:
    """What a reader of the results at `path` may see (`LiveStore.visible`, `policy` "latest" by default) when they are a live
    directory's; None for a backtest's, where `policy` raises `LiveDirectoryError` (and for results held in memory, `path`
    None, a ValueError). A cut-off that leaves out committed revisions of the issue days `days` (first, last; every day
    without) warns with how many, so a reader that fails closed never does so silently."""
    if path is None:
        if policy is not None:
            raise ValueError(f"policy={policy!r} reads a live directory's revisions: results held in memory have none")
        return None
    directory = Path(path).parent
    if not is_live_dir(directory):
        if policy is not None:
            raise LiveDirectoryError(
                f"{path}: policy={policy!r} reads a live directory's revisions, and {directory} is a backtest's (no "
                f"{LIVE_FILE}): read it without a policy"
            )
        return None
    store = LiveStore(directory)
    seen = store.visible(policy or "latest", cutoff)
    if cutoff is not None:
        _warn_dropped(path, store, policy or "latest", cutoff, seen, days)
    return seen


def _warn_dropped(
    path: Any,
    store: "LiveStore",
    policy: str,
    cutoff: Any,
    seen: Dict[str, Dict[int, Any]],
    days: Optional[Tuple[str, str]],
) -> None:
    """Warn when the cut-off leaves out committed revisions that the same policy without one reads, of the issue days from
    `days[0]` to `days[1]` (the reader's; every day without): a cut-off that fails closed (no deadline on file, a mapping that
    doesn't name a day) would otherwise show only as fewer forecasts."""
    every = store.visible(policy, None)
    if days is not None:
        every = {day: revisions for day, revisions in every.items() if days[0] <= day <= days[1]}
    dropped = {day: set(revisions) - set(seen.get(day, {})) for day, revisions in every.items()}
    dropped = {day: revisions for day, revisions in dropped.items() if revisions}
    if not dropped:
        return
    empty = sorted(day for day in dropped if not seen.get(day))
    warnings.warn(
        f"{path}: cutoff={cutoff!r} leaves out {sum(map(len, dropped.values()))} committed revisions of "
        f"{len(dropped)} issue days (e.g. {min(dropped)})"
        + (f", and {len(empty)} of those days read nothing" if empty else "")
        + ". "
        + (
            f"{store.identity_path.name} records no deadline, so only commits made by a deadline of their own count. "
            if cutoff == DEADLINE and "deadline" not in store.identity()
            else ""
        )
        + "Pass cutoff=None to read every committed revision.",
        UserWarning,
        stacklevel=outside_stacklevel(),
    )


def live_fingerprint(path: Any, policy: Optional[str], cutoff: Any) -> Tuple[Any, ...]:
    """What a cached read of the results at `path` depends on besides the results themselves: nothing for a backtest's; for a
    live directory's the reader's policy and cut-off and the state of its logs."""
    if path is None or not is_live_dir(Path(path).parent):
        return ()
    return (policy, repr(cutoff), LiveStore(Path(path).parent).fingerprint())


__all__ = [
    "Commit",
    "DEADLINE",
    "LiveBusyError",
    "LiveDirectoryError",
    "LiveLateError",
    "LiveRevision",
    "LiveStore",
    "POLICIES",
    "live_reading",
    "refuse_live_directory",
    "visible_revisions",
    "with_live",
]
