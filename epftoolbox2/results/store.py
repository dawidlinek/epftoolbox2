from pathlib import Path
import datetime
import functools
import inspect
import json
import queue
import threading
import warnings
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple
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


def _parse_lines(path: Path) -> Iterator[Tuple[int, bytes, Dict]]:
    """(line number, line, record) of each record of a JSONL results file, read as UTF-8.

    A last line without its newline that isn't a record is what a crash leaves while a batch is written: it is skipped with a
    warning (a writer cuts it off before appending, `ResultStore`). A line that isn't a record anywhere else raises ValueError
    naming its number: the file was damaged, and reading on would count what is left as the results."""
    with open(path, "rb") as f:
        for number, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line.decode("utf-8"))
                if not isinstance(record, dict):
                    raise ValueError(f"a JSON {type(record).__name__}, not an object")
            except ValueError as exc:  # UnicodeDecodeError and JSONDecodeError are ValueErrors
                if raw.endswith(b"\n"):
                    raise ValueError(
                        f"{path}: line {number} is not a result record ({exc}): {_preview(line)}"
                    ) from None
                warnings.warn(
                    f"{path}: its last line ({len(raw)} bytes, line {number}) is not a whole record, as a run that stopped "
                    f"while writing leaves it: {_preview(line)}. It is left out; the next run into this file cuts it off and "
                    "forecasts its key again.",
                    UserWarning,
                    stacklevel=4,
                )
                return
            yield number, line, record


def _preview(line: bytes, size: int = 80) -> str:
    return repr(line[:size]) + ("..." if len(line) > size else "")


def read_records(
    path: Path,
    statuses: Optional[Iterable[str]] = OK_ONLY,
    transform: Optional[Callable[[Dict], Any]] = None,
) -> Iterator[Any]:
    """One record per forecast key `(hour, horizon, run_date)` of a JSONL results file, in the order the records chosen stand in
    the file, `transform(record)` of each if given (None drops it).

    Per key the highest `revision` wins; within it a record with `status` "ok" wins over skip records (a key a later run forecast)
    and is the first ok one in the file: ok records after it are left out, with one warning per read giving how many and how many
    differ from it. Of skip records alone the last is the key's. `statuses` keeps the keys whose chosen record has one of them,
    "ok" by default (a record without `status` is ok); None keeps every key. A file without skip records, revisions or repeated
    keys is read as it stands, line for line. A record without the key fields is kept as it is."""
    wanted = None if statuses is None else frozenset(statuses)
    chosen: Dict[Any, tuple] = {}  # key -> (revision, status, what is yielded or None)
    repeated: Dict[Any, int] = {}  # key -> how many forecasts of its chosen revision were left out
    for number, _, record in _parse_lines(path):
        key = (record.get("hour"), record.get("horizon"), record.get("run_date"))
        if None in key:
            key = number  # no key to resume by: kept as it is
        revision = record.get("revision", BACKTEST_REVISION)
        status = record.get("status", STATUS_OK)
        held = chosen.get(key)
        if held is not None:
            if revision < held[0]:
                continue
            if revision == held[0] and held[1] == STATUS_OK:
                if status == STATUS_OK:
                    repeated[key] = repeated.get(key, 0) + 1
                continue
            del chosen[key]  # re-inserted at the end: the chosen records keep their order in the file
            repeated.pop(key, None)
        out = None
        if wanted is None or status in wanted:
            out = record if transform is None else transform(record)
        chosen[key] = (revision, status, out)
    if repeated:
        warnings.warn(
            f"{path}: {sum(repeated.values())} records repeat a forecast key that already has a forecast in the file "
            f"({_differing(path, chosen, repeated)} of them differ from it): the first is read and the others are left out. Two "
            "runs writing one file at once, or files joined by hand, leave such lines.",
            UserWarning,
            stacklevel=3,
        )
    for _, _, out in chosen.values():
        if out is not None:
            yield out


def _differing(path: Path, chosen: Dict[Any, tuple], repeated: Dict[Any, int]) -> int:
    """Of the forecasts `read_records` left out, how many differ from the one it read: a second pass, over a file that has them."""
    first: Dict[Any, bytes] = {}
    differ = 0
    for _, line, record in _parse_lines(path):
        key = (record.get("hour"), record.get("horizon"), record.get("run_date"))
        if key not in repeated or record.get("status", STATUS_OK) != STATUS_OK:
            continue
        if record.get("revision", BACKTEST_REVISION) != chosen[key][0]:
            continue
        if key in first:
            differ += line != first[key]
        else:
            first[key] = line
    return differ


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

    A key is done when the file has a forecast for it (`read_records`: status "ok", or no status). A key a run skipped has a skip
    record (`save_skip`), is retried by the next run as before, and gets another skip record only when the reason changed, so
    repeated resumes don't grow the file. `iter_lines` gives the forecasts alone unless told other statuses.

    With `lock=True` (what `BaseModel.run` opens) the store holds `<name>.lock` next to the file until `close()`, and a second
    writer raises `ResultsLockedError` naming it; a file system that refuses locks warns and the run goes on. A last line a crash
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
        key = (result["hour"], result["horizon"], result["run_date"])
        if result.get("status", STATUS_OK) == STATUS_OK:
            with self._lock:
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
        for r in self.iter_lines(cols=[*_KEY_FIELDS, "status"], statuses=None):
            if None in (r.get("hour"), r.get("horizon"), r.get("run_date")):
                continue
            key = (r["hour"], r["horizon"], r["run_date"])
            status = r.get("status", STATUS_OK)
            if status == STATUS_OK:
                self._completed.add(key)
            else:
                self._latest[key] = status

    def _acquire_file_lock(self):
        """The lock of this results file, held; None (with a warning) where the file system refuses locks."""
        from filelock import FileLock, Timeout

        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock = FileLock(str(self.lock_path), **_lock_options())
            lock.acquire(timeout=0)
        except Timeout:
            raise ResultsLockedError(
                f"{self.path} is being written by another run, which holds its lock {self.lock_path}. Two runs writing one "
                "model's results would put two forecasts on its keys: wait for it to finish, or write into another save_dir. "
                "If no run is writing it (a process that was killed holds no lock), report this as a bug."
            ) from None
        except (OSError, NotImplementedError) as exc:
            warnings.warn(
                f"{self.path}: the file system refused a lock on {self.lock_path} ({exc}), so a second run writing these "
                "results at the same time would not be refused. Run one writer per model's results here.",
                UserWarning,
                stacklevel=4,
            )
            return None
        return lock

    def _release_file_lock(self) -> None:
        lock, self._file_lock = self._file_lock, None
        if lock is not None:
            lock.release()

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
            stacklevel=4,
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


@functools.lru_cache(maxsize=None)
def _lock_options() -> Dict[str, Any]:
    """`FileLock` options: no soft lock where flock is missing (filelock 4 falls back to one, which a killed run leaves held, so
    every later run would be refused); such a file system warns instead (`ResultStore`)."""
    try:
        from filelock import FileLock

        parameters = inspect.signature(FileLock.__init__).parameters
    except (ImportError, TypeError, ValueError):  # pragma: no cover
        return {}
    return {"fallback_to_soft": False} if "fallback_to_soft" in parameters else {}
