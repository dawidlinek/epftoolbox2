from pathlib import Path
import datetime
import errno
import json
import os
import re
import zlib
import queue
import threading
import warnings
from typing import Any, BinaryIO, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple
from threading import Lock

# What every record written from record schema 2 on ends with (D4 of the forecasting plan), after the keys it had before, whose
# order and values don't change: `status` ("ok" for a forecast, a skip reason's name for a key with none), `list_hash` (the hash
# of the forecast's predictor list, `models.context.list_hash`, for fixed lists too), `refit_anchor` (the date of the fit the
# forecast came from: `run_date` without `refit`; a refit model's record has it where it always had it) and `revision` (1 for
# every backtest record). A record without `status` was written before them and is a forecast; one without `revision` is
# revision 1. An ensemble's records have `status` and `revision` alone: it has no list or fit of its own.
STATUS_OK = "ok"
OK_ONLY = (STATUS_OK,)
BACKTEST_REVISION = 1
RECORD_SCHEMA = 2  # recorded in a manifest as provenance (`record_schema`)
_KEY_FIELDS = ("hour", "horizon", "run_date")


def with_status(
    record: Dict,
    status: str = STATUS_OK,
    list_hash: Optional[str] = None,
    refit_anchor: Optional[str] = None,
    detail: Optional[str] = None,
) -> Dict:
    """`record` with D4's keys appended in place, after every key it has: `status`, `list_hash` (when given), `refit_anchor` (when
    given and the record has none, so a refit model's stays where it is), `revision`, and a skip record's `detail` when there is
    one. Returns the record."""
    record["status"] = status
    if list_hash is not None:
        record["list_hash"] = list_hash
    if refit_anchor is not None and "refit_anchor" not in record:
        record["refit_anchor"] = refit_anchor
    record["revision"] = BACKTEST_REVISION
    if detail:
        record["detail"] = detail
    return record


# What a forecast differs in from its repeat without being another forecast: D4's keys, which an old-format record lacks.
_NOT_THE_FORECAST = frozenset(("status", "list_hash", "refit_anchor", "revision", "detail"))
# The fields the first pass of `read_records` reads of a line, without parsing it: (name, its value, whether to look from the end,
# where the store writes it). JSON escapes every quote inside a string, so `"name"` followed by a colon is a key, and the first
# (last) one is the record's: a result record has no nested object.
_FIELDS = tuple(
    (b'"' + name + b'"', re.compile(rb"\s*:\s*" + value), from_end)
    for name, value, from_end in (
        (b"hour", rb"(-?\d+)[\s,}]", False),
        (b"horizon", rb"(-?\d+)[\s,}]", False),
        (b"run_date", rb'"([^"\\]*)"', False),
        (b"revision", rb"(-?\d+)[\s,}]", True),
        (b"status", rb'"([^"\\]*)"', True),
    )
)


def _parse_lines(path: Path) -> Iterator[Tuple[int, int, bytes, Dict]]:
    """(line number, byte offset, line, record) of each record of a JSONL results file, read as UTF-8.

    A last line without its newline that isn't a record is what a crash leaves while a batch is written: it is skipped with a
    warning (a writer cuts it off before appending, `ResultStore`). A line that isn't a record anywhere else raises ValueError
    naming its number: the file was damaged, and reading on would count what is left as the results."""
    with open(path, "rb") as f:
        offset = 0
        for number, raw in enumerate(f, 1):
            at, offset = offset, offset + len(raw)
            line = raw.strip()
            if line:
                record = _parse(path, number, raw, line)
                if record is None:
                    return
                yield number, at, line, record


def _parse(path: Path, number: int, raw: bytes, line: bytes) -> Optional[Dict]:
    """The record of one line; None (with a warning) for a half-written last line; ValueError for any other that isn't one."""
    try:
        record = json.loads(line.decode("utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"a JSON {type(record).__name__}, not an object")
        return record
    except ValueError as exc:  # UnicodeDecodeError and JSONDecodeError are ValueErrors
        if raw.endswith(b"\n"):
            raise ValueError(f"{path}: line {number} is not a result record ({exc}): {_preview(line)}") from None
        warnings.warn(
            f"{path}: its last line ({len(raw)} bytes, line {number}) is not a whole record, as a run that stopped "
            f"while writing leaves it: {_preview(line)}. It is left out; the next run into this file cuts it off and "
            "forecasts its key again.",
            UserWarning,
            stacklevel=5,
        )
        return None


def _key_fields(path: Path, number: int, raw: bytes, line: bytes) -> Optional[Tuple[Any, int, str]]:
    """(key, revision, status) of a line, read without parsing it where it is written as the store writes; parsed otherwise.
    None for a half-written last line (`_parse`)."""
    found = []
    if line[:1] == b"{" and line[-1:] == b"}":
        for name, value, from_end in _FIELDS:
            at = line.rfind(name) if from_end else line.find(name)
            match = None if at < 0 else value.match(line, at + len(name))
            if match is None and (at >= 0 or not from_end):
                break  # a key field missing, or a value not as the store writes it: parse the line
            found.append(None if match is None else match.group(1))
    if len(found) == len(_FIELDS):
        hour, horizon, run_date, revision, status = found
        try:
            key = (int(hour), int(horizon), run_date.decode("utf-8"))
            status = STATUS_OK if status is None or status == b"ok" else status.decode("utf-8")
            return key, BACKTEST_REVISION if revision is None else int(revision), status
        except ValueError:
            pass
    record = _parse(path, number, raw, line)
    if record is None:
        return None
    key = (record.get("hour"), record.get("horizon"), record.get("run_date"))
    status = record.get("status", STATUS_OK)
    return (number if None in key else key), record.get("revision", BACKTEST_REVISION), status


def _preview(line: bytes, size: int = 80) -> str:
    return repr(line[:size]) + ("..." if len(line) > size else "")


def read_records(
    path: Path,
    statuses: Optional[Iterable[str]] = OK_ONLY,
    transform: Optional[Callable[[Dict], Any]] = None,
    duplicates: str = "raise",
) -> Iterator[Any]:
    """One record per forecast key `(hour, horizon, run_date)` of a JSONL results file, in file order, `transform(record)` of
    each if given (None drops it).

    Per key the highest `revision` wins; within it a record with `status` "ok" wins over skip records (a key a later run forecast)
    and is the first ok one in the file. Of skip records alone the last is the key's. `statuses` keeps the keys whose chosen
    record has one of them, "ok" by default (a record without `status` is ok); None keeps every key. A file without skip records,
    revisions or repeated keys is read as it stands, line for line. A record without the key fields is kept as it is.

    A forecast repeating its key's chosen one with the same forecast (equal but for D4's keys, so an old-format line and a 5a line
    of the same numbers are one) is left out, with one warning per read giving how many. One with another forecast raises
    ValueError naming the file, how many and a key: two configurations' forecasts on one key, which no reader may pick between.
    `duplicates="first"` reads the first anyway, with a warning: for a tool that repairs such a file, never for a run or a report.

    Two passes, so that memory stays what the caller keeps: the first reads each line's key, revision and status alone and keeps
    the offset of each key's chosen line, the second parses only those lines, in file order. Both read one open file, so a file
    replaced meanwhile (`os.replace`) is read as it was; one written in place meanwhile raises ValueError: each chosen line must be
    where the first pass found it, with the same bytes (CRC-32) and so the same key, revision and status. A line no key chooses (a lower revision, a skip record under a
    forecast) isn't parsed, so a damaged one of those isn't reported; a chosen or repeated one raises with its line number."""
    if duplicates not in ("raise", "first"):
        raise ValueError(f"duplicates must be 'raise' or 'first', not {duplicates!r}")
    wanted = None if statuses is None else frozenset(statuses)
    with open(
        path, "rb"
    ) as f:  # once: a file replaced between the passes (an ensemble's, `os.replace`d) stays this one
        chosen, repeats = _choose(path, f)
        if repeats:
            _check_repeats(path, f, chosen, repeats, duplicates)
        # (offset, CRC-32 of the line, key, revision, status) of each line to parse, in file order
        keep = sorted(
            (at, crc, key, revision, status)
            for key, (revision, status, at, crc) in chosen.items()
            if wanted is None or status in wanted
        )
        del chosen, repeats
        if not keep:
            return
        f.seek(0)
        offset = number = 0
        following = iter(keep)
        expected = next(following)
        for number, raw in enumerate(f, 1):
            at, offset = offset, offset + len(raw)
            if at < expected[0]:
                continue
            if at > expected[0]:
                raise _changed(path, number, "a line it chose in its first pass is no longer where it was")
            if zlib.crc32(raw) != expected[1]:
                raise _changed(path, number, "the line's bytes aren't those its first pass read")
            record = _parse(path, number, raw, raw.strip())
            found = None if record is None else _fields_of(record, number)
            if found != expected[2:]:
                raise _changed(path, number, f"the line holds {found}, its first pass read {expected[2:]}")
            out = record if transform is None else transform(record)
            if out is not None:
                yield out
            expected = next(following, None)
            if expected is None:
                return
    raise _changed(path, number, "it ends before the lines its first pass chose")


def _choose(path: Path, f: BinaryIO) -> Tuple[Dict[Any, Tuple[int, str, int, int]], Dict[Any, List[int]]]:
    """The first pass of `read_records`: key -> (revision, status, offset of its chosen line, CRC-32 of that line), and key ->
    offsets of the forecasts repeating its chosen one."""
    chosen: Dict[Any, Tuple[int, str, int, int]] = {}
    repeats: Dict[Any, List[int]] = {}
    offset = 0
    for number, raw in enumerate(f, 1):
        at, offset = offset, offset + len(raw)
        line = raw.strip()
        if not line:
            continue
        fields = _key_fields(path, number, raw, line)
        if fields is None:
            break  # a half-written last line
        key, revision, status = fields
        held = chosen.get(key)
        if held is not None:
            if revision < held[0]:
                continue
            if revision == held[0] and held[1] == STATUS_OK:
                if status == STATUS_OK:
                    repeats.setdefault(key, []).append(at)
                continue
            repeats.pop(key, None)
        chosen[key] = (revision, status, at, zlib.crc32(raw))
    return chosen, repeats


def _fields_of(record: Dict, number: int) -> Tuple[Any, int, str]:
    """(key, revision, status) of a parsed record, as the first pass reads them off its line (`_key_fields`)."""
    key = (record.get("hour"), record.get("horizon"), record.get("run_date"))
    return (number if None in key else key), record.get("revision", BACKTEST_REVISION), record.get("status", STATUS_OK)


def _changed(path: Path, number: int, what: str) -> ValueError:
    return ValueError(
        f"{path} changed while it was read (line {number}: {what}): another process wrote it in place. Read it again once "
        "nothing writes it."
    )


def _check_repeats(
    path: Path, f: BinaryIO, chosen: Dict[Any, tuple], repeats: Dict[Any, List[int]], duplicates: str
) -> None:
    """Compare each repeated forecast with its key's chosen one, without D4's keys: the same warn, another raises (or warns with
    `duplicates="first"`)."""
    same = differ = 0
    example = None

    def forecast(at: int) -> str:
        f.seek(at)
        record = json.loads(f.readline().strip().decode("utf-8"))
        return json.dumps({k: v for k, v in record.items() if k not in _NOT_THE_FORECAST}, sort_keys=True)

    for key, offsets in repeats.items():
        first = forecast(chosen[key][2])
        for at in offsets:
            if forecast(at) == first:
                same += 1
            else:
                differ += 1
                example = example or key
    if differ and duplicates == "raise":
        hour, horizon, run_date = example if isinstance(example, tuple) else (None, None, example)
        raise ValueError(
            f"{path} holds forecast keys twice with different forecasts ({differ} records, e.g. period {hour}, horizon "
            f"{horizon}, run date {run_date}; {same} more repeat theirs with the same one): two configurations' forecasts on "
            "one key, as two runs writing one file at once or results files joined by hand leave them, and no reader can tell "
            "which is the run's. Write each run into its own save_dir and keep the file it wrote."
        )
    warnings.warn(
        f"{path}: {same + differ} records repeat a forecast key that already has a forecast in the file"
        + (f" ({differ} with another forecast, the first read)" if differ else ", with the same forecast")
        + ": each key is read once. Two runs writing one file at once, or files joined by hand, leave such lines.",
        UserWarning,
        stacklevel=4,
    )


def iter_file_records(path: Path) -> Iterator[Dict]:
    """Every record of a JSONL results file in file order, as it stands (no per-key rule), for a reader that stops early, such
    as one that needs what every record of a file shares (`freq`). A half-written last line is skipped with a warning."""
    for _, _, _, record in _parse_lines(path):
        yield record


class ResultsLockedError(RuntimeError):
    """Another run holds the lock of a results file: two writers would put two forecasts on its keys."""


class ResultStore:
    """Resumable JSONL storage for one model's results, keyed by `(hour, horizon, run_date)`.

    `run_date` (not `day_in_test`) identifies the forecast day: `day_in_test` is an offset from
    `test_start`, so the same offset means a different calendar day if `test_start` changes between
    runs into the same directory. Keying on the absolute date means resuming (deciding which tasks are
    already done) is correct on its own across a `test_start` change; a change to anything else that
    defines a forecast (predictors, target, training window, model params, the data itself, ...) is
    caught separately by the manifest (`epftoolbox2.results.manifest`), which `BaseModel.run` checks
    before resuming.

    Old rows are never deleted from the file, so a run with a narrower or shifted test window than an
    earlier run into the same directory leaves out-of-window rows sitting alongside the new ones.
    Reading results back for evaluation must filter to the current run's window (`EvaluationReport`
    does this using each ref's own `test_start`/`test_end`); this store does not do it for you.

    A key is done when its chosen record under the readers' rule (`read_records`: the highest revision, a forecast over skip
    records within it) is a forecast (status "ok", or no status). A key a run skipped has a skip record (`save_skip`), is retried by
    the next run as before, and gets another skip record only when the reason changed, so repeated resumes don't grow the file.
    What is written for a key whose records reach a higher revision than 1 gets that revision, so a reader takes it. `iter_lines`
    gives the forecasts alone unless told other statuses.

    With `lock=True` (what `BaseModel.run` opens) the store holds `<name>.lock` next to the file (an `fcntl.flock`) until
    `close()`, and a second writer raises `ResultsLockedError` naming it; a file system that refuses locks warns and the run goes
    on. A last line a crash
    left half-written is cut off, with a warning naming its bytes, when the store opens with the lock, and otherwise before its
    first write, never under another writer's hands. Files are read and written as UTF-8.
    """

    _SENTINEL = object()

    def __init__(self, path: str, test_start: Optional[str] = None, batch_size: int = 500, lock: bool = False):
        self.path = Path(path)
        self.test_start = test_start
        self._lock = Lock()
        self._completed: Set[tuple] = set()
        self._latest: Dict[tuple, str] = {}  # key without a forecast -> the status of its newest skip record
        self._revisions: Dict[tuple, int] = {}  # key -> its highest revision on file, where that isn't 1
        self._batch_size = batch_size
        self._tail_checked = False

        self._file_lock = self._acquire_file_lock() if lock else None
        try:
            if lock:
                self._cut_tail()
            self._load_existing()
        except BaseException:
            self._release_file_lock()
            raise

        self._write_queue: queue.Queue = queue.Queue()
        self._writer = threading.Thread(target=self._writer_loop, daemon=True, name="store-writer")
        self._writer.start()

    @property
    def lock_path(self) -> Path:
        """`<name>.lock` next to the results file (`ols.jsonl` -> `ols.lock`)."""
        return self.path.with_name(f"{self.path.stem}.lock")

    def save(self, result: Dict) -> None:
        """Queue a record for writing. A key with a record of a higher revision than 1 on file gets that revision, so that what is
        written is what a reader takes (`read_records`: the highest revision wins)."""
        key = (result["hour"], result["horizon"], result["run_date"])
        with self._lock:
            if key in self._revisions:
                result["revision"] = self._revisions[key]
            if result.get("status", STATUS_OK) == STATUS_OK:
                self._completed.add(key)
        self._write_queue.put(result)

    def save_skip(self, record: Dict) -> bool:
        """Write a skip record (`with_status` with a skip reason as `status`) unless its key has a forecast or its newest skip
        record has the same status. Returns whether it was written. The key stays pending: the next run retries it."""
        key = (record["hour"], record["horizon"], record["run_date"])
        status = record["status"]
        with self._lock:
            if key in self._completed or self._latest.get(key) == status:
                return False
            self._latest[key] = status
            if key in self._revisions:
                record["revision"] = self._revisions[key]
        self._write_queue.put(record)
        return True

    def has_results(self) -> bool:
        """Whether the file holds a forecast: what decides whether a run resumes results, as a file of skip records alone has
        none to resume."""
        return bool(self._completed)

    def flush(self) -> None:
        if self._writer.is_alive():
            self._write_queue.put(self._SENTINEL)
            self._writer.join()

    def close(self) -> None:
        """Write what is queued and release the lock."""
        try:
            self.flush()
        finally:
            self._release_file_lock()

    def _run_date(self, day_in_test: int) -> str:
        if self.test_start is None:
            raise ValueError("ResultStore needs test_start to resolve day_in_test to a calendar date")
        base = datetime.date.fromisoformat(self.test_start)
        return (base + datetime.timedelta(days=day_in_test)).isoformat()

    def is_done(self, hour: int, horizon: int, day_in_test: int) -> bool:
        return (hour, horizon, self._run_date(day_in_test)) in self._completed

    def get_missing(self, all_tasks: List[tuple]) -> List[tuple]:
        return [t for t in all_tasks if not self.is_done(t[0], t[1], t[2])]

    def load_all(self) -> List[Dict]:
        return list(self.iter_lines())

    def iter_lines(
        self, cols: Optional[List[str]] = None, statuses: Optional[Iterable[str]] = OK_ONLY
    ) -> Iterator[Dict]:
        """The records of the file (`read_records`): its forecasts by default, every key's chosen record with `statuses=None`;
        only the keys `cols` names if given."""
        if not self.path.exists():
            return iter(())
        transform = None if not cols else (lambda r: {k: r[k] for k in cols if k in r})
        return read_records(self.path, statuses, transform)

    def _load_existing(self) -> None:
        """Each key's chosen record under the readers' rule (`read_records`): done if it is a forecast, else its status, and its
        revision where that isn't 1."""
        for r in self.iter_lines(cols=[*_KEY_FIELDS, "status", "revision"], statuses=None):
            if None in (r.get("hour"), r.get("horizon"), r.get("run_date")):
                continue
            key = (r["hour"], r["horizon"], r["run_date"])
            status = r.get("status", STATUS_OK)
            if status == STATUS_OK:
                self._completed.add(key)
            else:
                self._latest[key] = status
            revision = r.get("revision", BACKTEST_REVISION)
            if revision != BACKTEST_REVISION:
                self._revisions[key] = revision

    def _acquire_file_lock(self) -> Optional[Callable[[], None]]:
        """Hold the lock of this results file and return what releases it; None, with a warning, where the file system refuses
        locks. The errors are read here (`_hold_lock`), so they mean the same whatever filelock's version."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return _hold_lock(self.lock_path)
        except _LockHeld:
            raise ResultsLockedError(
                f"{self.path} is being written by another run, which holds its lock {self.lock_path}. Two runs writing one "
                "model's results would put two forecasts on its keys: wait for it to finish, or write into another save_dir. "
                "If no run is writing it (a process that was killed holds no lock), report this as a bug."
            ) from None
        except OSError as exc:
            warnings.warn(
                f"{self.path}: the file system refused a lock on {self.lock_path} ({exc}), so a second run writing these "
                "results at the same time would not be refused. Run one writer per model's results here.",
                UserWarning,
                stacklevel=5,
            )
            return None

    def _release_file_lock(self) -> None:
        release, self._file_lock = self._file_lock, None
        if release is not None:
            release()

    def _cut_tail(self) -> None:
        """Cut off a last line a crash left half-written, before anything is appended after it, with a warning naming its
        bytes; end a whole last record that lost only its newline with one."""
        self._tail_checked = True
        if not self.path.exists():
            return
        with open(self.path, "rb+") as f:
            size = f.seek(0, 2)
            if size == 0:
                return
            f.seek(size - 1)
            if f.read(1) == b"\n":
                return
            start = size
            while start > 0:  # back to the byte after the last newline
                step = min(start, 1 << 16)
                f.seek(start - step)
                chunk = f.read(step)
                newline = chunk.rfind(b"\n")
                if newline >= 0:
                    start = start - step + newline + 1
                    break
                start -= step
            f.seek(start)
            tail = f.read()
            try:
                whole = isinstance(json.loads(tail.strip().decode("utf-8")), dict)
            except ValueError:
                whole = False
            if whole:
                f.seek(size)
                f.write(b"\n")
                return
            f.truncate(start)
        warnings.warn(
            f"{self.path}: cut off its last {size - start} bytes (from byte {start}), a line a run that stopped while writing "
            f"left half-written: {_preview(tail.strip())}. Its key is forecast again.",
            UserWarning,
            stacklevel=5,
        )

    def _writer_loop(self) -> None:
        buf: List[Dict] = []

        while True:
            try:
                item = self._write_queue.get(timeout=0.1)
            except queue.Empty:
                if buf:
                    self._write_batch(buf)
                    buf = []
                continue

            if item is self._SENTINEL:
                while True:
                    try:
                        extra = self._write_queue.get_nowait()
                        if extra is not self._SENTINEL:
                            buf.append(extra)
                    except queue.Empty:
                        break
                if buf:
                    self._write_batch(buf)
                return

            buf.append(item)

            while len(buf) < self._batch_size:
                try:
                    nxt = self._write_queue.get_nowait()
                    if nxt is self._SENTINEL:
                        self._write_batch(buf)
                        return
                    buf.append(nxt)
                except queue.Empty:
                    break

            if len(buf) >= self._batch_size:
                self._write_batch(buf)
                buf = []

    def _write_batch(self, items: List[Dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._tail_checked:
            self._cut_tail()
        with open(self.path, "a", encoding="utf-8", buffering=1 << 20) as f:
            for item in items:
                f.write(json.dumps(item) + "\n")


class _LockHeld(Exception):
    """Another process (or another open file in this one) holds the lock."""


# What a non-blocking flock says when another holds the lock; any other error is a file system that refuses locks (ENOLCK on an
# NFS mount without a lock daemon, ENOSYS, EOPNOTSUPP), which warns. filelock 3 read most of those as a held lock and filelock 4
# falls back to a soft lock a killed run leaves held, so the store takes the lock itself where it can.
_HELD_ERRNOS = frozenset((errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES))


def _hold_lock(path: Path) -> Callable[[], None]:
    """An exclusive `fcntl.flock` on `path`, held until the returned function is called: the operating system's lock, so a
    process that is killed holds none. Where there is no flock (Windows) filelock's lock. Raises `_LockHeld` when another holds
    it, OSError when the file system refuses locks.

    The lock file is opened read-write (created if missing) or, where that is refused (another user's lock file in a shared
    directory), read-only: flock doesn't need write access, so a writer of a file it can't write is refused as any other. Only a
    lock file that can't be opened at all is a file system that refuses locks. A process forked while the lock is held closes
    its copy of the descriptor (`_close_held_in_child`), so a pool worker outliving a killed run doesn't keep the lock."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover (Windows)
        return _hold_filelock(path)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except PermissionError:
        fd = os.open(path, os.O_RDONLY)  # raises PermissionError if it can't be read either
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in _HELD_ERRNOS:
            raise _LockHeld() from None
        raise
    _HELD_FDS.add(fd)

    def release() -> None:
        _HELD_FDS.discard(fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    return release


# The descriptors of the locks this process holds. A forked child (a pool worker of `BaseModel.run`) shares their open file
# descriptions, and with them the lock: it closes its copies, never unlocking them (that would release the parent's lock), so
# the lock ends with the process that took it.
_HELD_FDS: Set[int] = set()


def _close_held_in_child() -> None:
    for fd in list(_HELD_FDS):
        try:
            os.close(fd)
        except OSError:
            pass
    _HELD_FDS.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_close_held_in_child)


def _hold_filelock(path: Path) -> Callable[[], None]:  # pragma: no cover (Windows)
    from filelock import FileLock, Timeout

    lock = FileLock(str(path))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        raise _LockHeld() from None
    except NotImplementedError as exc:
        raise OSError(errno.ENOSYS, str(exc)) from exc
    return lock.release
