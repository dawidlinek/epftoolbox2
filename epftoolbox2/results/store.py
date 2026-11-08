from pathlib import Path
import datetime
import errno
import json
import os
import re
import zlib
import queue
import threading
import time
import uuid
import sys
import warnings
from typing import Any, BinaryIO, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
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


_PACKAGE_DIR = str(Path(__file__).resolve().parent.parent)


def outside_stacklevel() -> int:
    """The `stacklevel` that attributes a warning to the first frame outside this package: the line that called `run`, a
    report method or a store, however many library frames (`open_store`, a store's opening, the run's phases) lie between."""
    frame, level = sys._getframe(1), 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame, level = frame.f_back, level + 1
    return level


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
    columns: Optional[Sequence[str]] = None,
) -> Iterator[Any]:
    """One record per forecast key `(hour, horizon, run_date)` of a JSONL results file, in file order, `transform(record)` of
    each if given (None drops it). A Parquet results directory (`<name>.parquet`) is read by the same rules, in the order its
    catalog commits its parts (`results.parquet_store.read_records`); `columns`, if given, are the keys `transform` reads, which
    a Parquet directory reads alone (a JSONL line is parsed whole).

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
    if is_parquet_path(path):
        from .parquet_store import read_records as read_parquet

        yield from read_parquet(Path(path), statuses, transform, duplicates, columns)
        return
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
        return forecast_text(json.loads(f.readline().strip().decode("utf-8")))

    for key, offsets in repeats.items():
        first = forecast(chosen[key][2])
        for at in offsets:
            if forecast(at) == first:
                same += 1
            else:
                differ += 1
                example = example or key
    repeat_verdict(path, same, differ, example, duplicates, stacklevel=5)


def forecast_text(record: Dict) -> str:
    """What two records of one key are compared on: the record without D4's keys, as sorted JSON (so an old-format line and a
    5a line of the same numbers are one forecast)."""
    return json.dumps({k: v for k, v in record.items() if k not in _NOT_THE_FORECAST}, sort_keys=True)


def repeat_verdict(path: Path, same: int, differ: int, example: Any, duplicates: str, stacklevel: int = 4) -> None:
    """Raise for forecasts repeating their key with another forecast (unless `duplicates="first"`), and warn once for the
    repeats read once."""
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
        stacklevel=stacklevel,
    )


def iter_file_records(path: Path) -> Iterator[Dict]:
    """Every record of a JSONL results file in file order, as it stands (no per-key rule), for a reader that stops early, such
    as one that needs what every record of a file shares (`freq`). A half-written last line is skipped with a warning. A
    Parquet results directory gives its committed records in catalog order."""
    if is_parquet_path(path):
        from .parquet_store import iter_file_records as iter_parquet

        yield from iter_parquet(Path(path))
        return
    for _, _, _, record in _parse_lines(path):
        yield record


class ResultsLockedError(RuntimeError):
    """Another run holds the lock of a results file: two writers would put two forecasts on its keys."""


class ResultsFormatError(ValueError):
    """A run would write one format of results (`save_format`) beside the other format's results of the same model."""


# The formats a model's results can be written in (`ModelPipeline.run(save_format=...)`) and the suffix of each: `<name>.jsonl`
# (a file) or `<name>.parquet` (a directory, `results.parquet_store`). JSONL is the default and defines the record.
SAVE_FORMATS = {"jsonl": ".jsonl", "parquet": ".parquet"}
DEFAULT_SAVE_FORMAT = "jsonl"


def is_parquet_path(path: Any) -> bool:
    """Whether `path` names Parquet results (`<name>.parquet`), by its suffix."""
    return Path(path).suffix == SAVE_FORMATS["parquet"]


def format_of(path: Any) -> str:
    """The save format a results path names: "parquet" for `<name>.parquet`, else "jsonl"."""
    return "parquet" if is_parquet_path(path) else "jsonl"


def check_save_format(save_format: str) -> str:
    if save_format not in SAVE_FORMATS:
        raise ValueError(f"save_format must be one of {sorted(SAVE_FORMATS)}, got {save_format!r}")
    return save_format


def other_format_path(path: Any) -> Path:
    """The path the same model's results have in the other format: `ols.jsonl` <-> `ols.parquet`."""
    path = Path(path)
    other = "jsonl" if is_parquet_path(path) else "parquet"
    return path.with_name(path.stem + SAVE_FORMATS[other])


def check_no_other_format(path: Any) -> None:
    """Raise `ResultsFormatError` if the model of `path` has results in the other format beside it: a run writes one format
    per model, and `epf results convert` turns one into the other exactly."""
    path = Path(path)
    other = other_format_path(path)
    if other.exists():
        raise ResultsFormatError(
            f"{other} holds this model's results as {format_of(other)}, and this run writes {format_of(path)} "
            f"({path.name}). Convert them first, `epf results convert {path.parent} --to {format_of(path)}` (exact both "
            f"ways), or run with save_format={format_of(other)!r}."
        )


def fingerprint_path(path: Any) -> Path:
    """What says a model's results changed: the JSONL file, or a Parquet directory's catalog (every committed part and every
    compaction appends a line to it). The directory itself while it has no catalog."""
    path = Path(path)
    if is_parquet_path(path):
        catalog = path / "_catalog.jsonl"
        return catalog if catalog.exists() else path
    return path


class _Store:
    """What a model's results store does whatever its format: resume state (done keys, skip statuses, revisions), `save` and
    `save_skip` under the readers' rule, the lock, and reading through `read_records`. `ResultStore` writes JSONL lines,
    `parquet_store.ParquetResultStore` Parquet parts; either writes on its own thread or through a `StoreWriter` shared by many
    stores.

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

    With `lock=True` (what `BaseModel.run` opens) the store holds `<name>.lock` next to the results (an `fcntl.flock`; one lock for
    a model's JSONL file and its Parquet directory alike) until `close()`, and a second writer raises `ResultsLockedError` naming
    it; a file system that refuses locks warns and the run goes on. Opened with the lock, a store refuses (`ResultsFormatError`)
    results of the model in the other format beside it.

    `save_coefficients=False` writes `coefficients` as `[]` (as a window-list record has them) and every other key as it is.
    """

    save_format = DEFAULT_SAVE_FORMAT

    def __init__(
        self,
        path: Any,
        test_start: Optional[str],
        lock: bool,
        save_coefficients: bool,
        writer: Optional["StoreWriter"],
    ):
        self.path = Path(path)
        self.test_start = test_start
        self._lock = Lock()
        self._completed: Set[tuple] = set()  # keys whose forecast is on file (read at open, or written by this store)
        self._saved: Set[tuple] = set()  # keys of forecasts saved and not yet written
        self._latest: Dict[tuple, str] = {}  # key without a forecast -> the status of its newest skip record
        self._revisions: Dict[tuple, int] = {}  # key -> its highest revision on file, where that isn't 1
        self._save_coefficients = bool(save_coefficients)
        self._shared = writer
        self._error: Optional[BaseException] = None

        if lock:
            check_no_other_format(self.path)
        self._file_lock = self._acquire_file_lock() if lock else None
        try:
            if lock:
                self._repair_tail()
            self._load_existing()
        except BaseException:
            self._release_file_lock()
            raise

    @property
    def lock_path(self) -> Path:
        """`<name>.lock` next to the results (`ols.jsonl` -> `ols.lock`, `ols.parquet` -> `ols.lock`)."""
        return self.path.with_name(f"{self.path.stem}.lock")

    def save(self, result: Dict) -> None:
        """Queue a record for writing. A key with a record of a higher revision than 1 on file gets that revision, so that what is
        written is what a reader takes (`read_records`: the highest revision wins)."""
        self._raise_error()
        if not self._save_coefficients and result.get("coefficients"):
            result = {**result, "coefficients": []}  # the key keeps its place
        key = (result["hour"], result["horizon"], result["run_date"])
        revision = self._revisions.get(key)  # filled before the first save, read only after
        if revision is not None:
            result["revision"] = revision
        layout = self._validate(result)  # before anything of the record is kept
        if result.get("status", STATUS_OK) == STATUS_OK:
            with self._lock:
                self._saved.add(key)  # done once written (`_written`), not before
        self._enqueue(result, layout)

    def save_skip(self, record: Dict) -> bool:
        """Write a skip record (`with_status` with a skip reason as `status`) unless its key has a forecast or its newest skip
        record has the same status. Returns whether it was written. The key stays pending: the next run retries it."""
        self._raise_error()
        key = (record["hour"], record["horizon"], record["run_date"])
        status = record["status"]
        with self._lock:
            if key in self._completed or key in self._saved or self._latest.get(key) == status:
                return False
            if key in self._revisions:
                record["revision"] = self._revisions[key]
        layout = self._validate(record)
        with self._lock:
            self._latest[key] = status
        self._enqueue(record, layout)
        return True

    def _written(self, keys: Iterable[Optional[tuple]]) -> None:
        """The forecasts of these keys (None for a skip record) are on file: they count as done from now on. A record whose
        write failed never gets here, so a key counts as done only once its line or part is written."""
        with self._lock:
            for key in keys:
                if key is not None:
                    self._saved.discard(key)
                    self._completed.add(key)

    def has_results(self) -> bool:
        """Whether the file holds a forecast: what decides whether a run resumes results, as a file of skip records alone has
        none to resume."""
        return bool(self._completed)

    def _run_date(self, day_in_test: int) -> str:
        if self.test_start is None:
            raise ValueError("ResultStore needs test_start to resolve day_in_test to a calendar date")
        base = datetime.date.fromisoformat(self.test_start)
        return (base + datetime.timedelta(days=day_in_test)).isoformat()

    def is_done(self, hour: int, horizon: int, day_in_test: int) -> bool:
        """Whether the key's forecast is on file. A saved record counts once it is written, which the store's thread (or
        `StoreWriter`) does a moment after `save()`: call `flush()` first to ask about what this store saved."""
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
        return read_records(self.path, statuses, transform, columns=cols or None)

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
        """Hold the lock of these results and return what releases it; None, with a warning, where the file system refuses
        locks. The errors are read here (`_hold_lock`), so they mean the same whatever filelock's version. A store of a shared
        `StoreWriter` holds it through the writer's one lock (`_GroupLock`), so many stores keep no descriptor each."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        hold = _hold_lock if self._shared is None or self._shared._private else self._shared._hold
        try:
            return hold(self.lock_path)
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
                stacklevel=outside_stacklevel(),
            )
            return None

    def _release_file_lock(self) -> None:
        release, self._file_lock = self._file_lock, None
        if release is not None:
            release()

    def _raise_error(self) -> None:
        """Raise what failed while this store's records were written in the background, every time: a store stops writing
        after its first failed write (it can have left half a line or half a catalog line, which nothing may be appended
        after), so each later `save`, `flush` and `close` raises it, and what it hasn't written stays pending for the next
        run, whose writer cuts the half-written tail before it appends."""
        if self._error is not None:
            raise self._error

    def _failed(self, exc: BaseException) -> None:
        """Stop writing: keep the first failure for `_raise_error`, and have the tail checked before any later append."""
        with self._lock:
            self._error = self._error or exc
            self._tail_checked = False

    # What a format defines.
    def _repair_tail(self) -> None:  # pragma: no cover (abstract)
        raise NotImplementedError

    def _validate(self, record: Dict) -> Any:
        """Raise ValueError for a record this format can't hold, before anything of it is kept; what `_enqueue` takes with
        it (a Parquet record's keys, so that they are read once)."""
        return None

    def _enqueue(self, record: Dict, layout: Any = None) -> None:  # pragma: no cover (abstract)
        raise NotImplementedError

    # What a shared `StoreWriter` calls, under its condition (`_buffer`, `_take`, `_buffered`, `_due`) or the store's I/O lock.
    def _buffer(self, record: Dict, layout: Any) -> bool:  # pragma: no cover (abstract)
        """Hold a record until it is written; True if it should be written now (a full batch)."""
        raise NotImplementedError

    def _buffered(self) -> int:  # pragma: no cover (abstract)
        raise NotImplementedError

    def _take(self, now: float, force: bool) -> Any:  # pragma: no cover (abstract)
        """What is due to be written (all of it with `force`), removed from the buffers; falsy for nothing."""
        raise NotImplementedError

    def _write(self, taken: Any) -> None:  # pragma: no cover (abstract)
        raise NotImplementedError


class ResultStore(_Store):
    """Resumable JSONL storage for one model's results, keyed by `(hour, horizon, run_date)` (see `_Store`).

    A last line a crash left half-written is cut off, with a warning naming its bytes, when the store opens with the lock, and
    otherwise before its first write, never under another writer's hands. Files are read and written as UTF-8. Records are
    written by the store's own thread, in batches of `batch_size`, or by `writer`, a `StoreWriter` shared with other stores.
    """

    _SENTINEL = object()
    save_format = "jsonl"

    def __init__(
        self,
        path: str,
        test_start: Optional[str] = None,
        batch_size: int = 500,
        lock: bool = False,
        save_coefficients: bool = True,
        writer: Optional["StoreWriter"] = None,
    ):
        self._batch_size = batch_size
        self._tail_checked = False
        super().__init__(path, test_start, lock, save_coefficients, writer)

        if writer is None:
            self._write_queue: queue.Queue = queue.Queue()
            self._writer = threading.Thread(target=self._writer_loop, daemon=True, name="store-writer")
            self._writer.start()
        else:
            self._pending: List[Dict] = []
            self._io_lock = Lock()
            writer._register(self)

    def flush(self) -> None:
        if self._shared is not None:
            self._shared._write_now(self)
            return
        if self._writer.is_alive():
            self._write_queue.put(self._SENTINEL)
            self._writer.join()
        self._raise_error()

    def close(self) -> None:
        """Write what is queued and release the lock."""
        try:
            self.flush()
        finally:
            self._release_file_lock()
            if self._shared is not None:
                self._shared._forget(self)

    def _repair_tail(self) -> None:
        self._cut_tail()

    def _enqueue(self, record: Dict, layout: Any = None) -> None:
        if self._shared is None:
            self._write_queue.put(record)
        else:
            self._shared._submit(self, record, layout)

    def _buffer(self, record: Dict, layout: Any) -> bool:
        self._pending.append(record)
        return len(self._pending) >= self._batch_size

    def _buffered(self) -> int:
        return len(self._pending)

    def _take(self, now: float, force: bool) -> List[Dict]:
        taken, self._pending = self._pending, []
        return taken

    def _write(self, taken: List[Dict]) -> None:
        with self._shared._handles:
            self._write_batch(taken)

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
            stacklevel=outside_stacklevel(),
        )

    def _writer_loop(self) -> None:
        buf: List[Dict] = []

        while True:
            try:
                item = self._write_queue.get(timeout=0.1)
            except queue.Empty:
                if buf:
                    self._write_or_keep_error(buf)
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
                    self._write_or_keep_error(buf)
                return

            buf.append(item)

            while len(buf) < self._batch_size:
                try:
                    nxt = self._write_queue.get_nowait()
                    if nxt is self._SENTINEL:
                        self._write_or_keep_error(buf)
                        return
                    buf.append(nxt)
                except queue.Empty:
                    break

            if len(buf) >= self._batch_size:
                self._write_or_keep_error(buf)
                buf = []

    def _write_batch(self, items: List[Dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._tail_checked:
            self._cut_tail()
        with open(self.path, "a", encoding="utf-8", buffering=1 << 20) as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        self._written(
            (item["hour"], item["horizon"], item["run_date"]) if item.get("status", STATUS_OK) == STATUS_OK else None
            for item in items
        )

    def _write_or_keep_error(self, items: List[Dict]) -> None:
        """`_write_batch` on the store's own thread: what fails is kept for `flush()` to raise, and the records are lost (their
        keys stay pending), so a failed write never passes for a written one. After a failure nothing more is written (a short
        write leaves half a line, which a later batch would turn into a damaged middle line): the records queued meanwhile
        stay pending too."""
        if self._error is not None:
            return
        try:
            self._write_batch(items)
        except BaseException as exc:
            self._failed(exc)


# How many files a `StoreWriter` has open at once, at most: a JSONL batch or a Parquet part is written with its file open and
# closed after it, so 10,000 stores need no 10,000 descriptors (the soft limit is often 1,024).
MAX_OPEN_HANDLES = 64
# How often the writer's thread looks at its stores: a JSONL store's records are written within this of being saved, as the
# store's own thread does.
_TICK_SECONDS = 0.1
# How many stores `StoreWriter.flush()` writes at once.
FLUSH_THREADS = 4


class StoreWriter:
    """One background thread that writes the records of many models' stores, each into its own file or directory: per-store
    buffers, at most `max_open` files open at once, and one lock for all of them (`_GroupLock`), so 10,000 stores need neither
    10,000 threads (as a `ResultStore` of its own would start) nor 10,000 descriptors.

    `open(path, ...)` opens a model's store (`open_store`, by suffix) on this writer. A JSONL store's records are written within
    0.1 s of being saved, or when 500 are waiting; a Parquet store's when a (model, year) buffer reaches
    `PARQUET_ROWS_PER_PART` rows, is `PARQUET_FLUSH_SECONDS` old, or the writer holds more than `PARQUET_MAX_BUFFERED_ROWS` rows
    (the largest buffers first). `store.flush()` writes that store's records now, `store.close()` also releases its lock, and
    `close()` writes everything and stops the thread. What fails in the background is raised by the store's next `save`,
    `flush` or `close`. What a store writes is what it writes on its own thread: the same records, lines and parts.
    """

    def __init__(self, max_open: int = MAX_OPEN_HANDLES, threads: int = FLUSH_THREADS, _private: bool = False):
        if max_open < 1:
            raise ValueError(f"max_open must be at least 1, got {max_open!r}")
        if threads < 1:
            raise ValueError(f"threads must be at least 1, got {threads!r}")
        self.max_open = max_open
        self.threads = threads
        self._private = _private  # one store's own writer: it holds that store's lock itself
        self._handles = threading.BoundedSemaphore(max_open)
        self._cond = threading.Condition()
        self._dirty: Dict[int, _Store] = {}  # id -> store with buffered records, in the order they first had one
        self._open: Dict[str, _Store] = {}  # resolved path -> store
        self._group: Optional[_GroupLock] = None
        self._stopping = False
        self._buffered_rows = 0  # of the Parquet stores, for `PARQUET_MAX_BUFFERED_ROWS`
        from .parquet_store import limits

        self._rows_per_part, self._flush_seconds, self._max_buffered = limits()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="store-writer")
        self._thread.start()

    def open(
        self, path: Any, test_start: Optional[str] = None, lock: bool = False, save_coefficients: bool = True
    ) -> _Store:
        """A model's store on this writer: `ResultStore` for `<name>.jsonl`, `ParquetResultStore` for `<name>.parquet`. One
        writer opens a path once at a time."""
        if self._stopping:
            raise ValueError("this StoreWriter is closed")
        key = str(Path(path).resolve())
        with self._cond:
            if key in self._open:
                raise ValueError(f"{path} is already open on this StoreWriter")
        store = open_store(path, test_start=test_start, lock=lock, save_coefficients=save_coefficients, writer=self)
        return store

    def flush(self) -> None:
        """Write every store's buffered records now, `threads` stores at a time (a part's encoding and compression run outside
        the GIL, so they overlap the next store's conversion). Raises the first store's failure once every store was written.

        Every open store, not only those with records buffered: one whose records the writer's thread took a moment ago is
        still being written, and its I/O lock (`_write_now`) waits for that, so on return every record saved before is on file
        and counts as done (`is_done`, `get_missing`)."""
        with self._cond:
            stores = list({id(store): store for store in (*self._dirty.values(), *self._open.values())}.values())
        if self.threads == 1 or len(stores) < 2:
            for store in stores:
                self._write_now(store)
            return
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(self.threads, len(stores)), thread_name_prefix="store-flush") as pool:
            futures = [pool.submit(self._write_now, store) for store in stores]
        for future in futures:
            future.result()

    def close(self) -> None:
        """Write everything, stop the thread, and release the stores' locks."""
        try:
            self.flush()
        finally:
            self._stop_and_close()

    def _stop_and_close(self) -> None:
        self._stopping = True
        with self._cond:
            self._cond.notify()
        self._thread.join()
        with self._cond:
            stores = list(self._open.values())
        error = None
        for store in stores:
            try:
                store.close()
            except BaseException as exc:  # close every store, then raise the first failure
                error = error or exc
        if self._group is not None:
            self._group.release()
            self._group = None
        if error is not None:
            raise error

    def __enter__(self) -> "StoreWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # --- what the stores call -----------------------------------------------------------------------------------------------

    def _hold(self, lock_path: Path) -> Callable[[], None]:
        with self._cond:
            if self._group is None:
                self._group = _GroupLock()
            group = self._group
        return group.hold(lock_path)

    def _register(self, store: _Store) -> None:
        with self._cond:
            self._open[str(store.path.resolve())] = store

    def _forget(self, store: _Store) -> None:
        with self._cond:
            self._open.pop(str(store.path.resolve()), None)
            self._dirty.pop(id(store), None)

    def _submit(self, store: _Store, record: Dict, layout: Any = None) -> None:
        with self._cond:
            urgent = store._buffer(record, layout)
            self._dirty[id(store)] = store
            if store.save_format == "parquet":
                self._buffered_rows += 1
                urgent = urgent or self._buffered_rows > self._max_buffered
            if urgent:
                self._cond.notify()

    def _write_now(self, store: _Store) -> None:
        """Write all of `store`'s buffered records in this thread, and raise what failed in the background."""
        self._write_due(store, time.monotonic(), force=True)
        store._raise_error()

    def _write_due(self, store: _Store, now: float, force: bool) -> None:
        """Write what of `store` is due. What fails is the store's (`_fail`) before its I/O lock is let go, so a `flush()`
        waiting for the lock finds it and raises."""
        # The store's I/O lock before taking its records, so that what was taken first is written first.
        with store._io_lock:
            with self._cond:
                taken = store._take(now, force)
                if store.save_format == "parquet" and taken:
                    self._buffered_rows -= sum(item[1] for item in taken)
                if not store._buffered():
                    self._dirty.pop(id(store), None)
            if taken and store._error is None:  # a store whose write failed writes nothing more (`_Store._failed`)
                try:
                    store._write(taken)
                except BaseException as exc:
                    self._fail(store, exc, now)

    def _fail(self, store: _Store, exc: BaseException, now: float) -> None:
        """Keep `exc` for every later `save`, `flush` and `close` of the store to raise, and drop what it still buffers: it is
        not written, its keys stay pending, and the store writes nothing more (a half-written line or catalog line stays last,
        for the next writer to cut). Called with the store's I/O lock held."""
        store._failed(exc)
        with self._cond:
            dropped = store._take(now, True)
            if store.save_format == "parquet" and dropped:
                self._buffered_rows -= sum(item[1] for item in dropped)
            self._dirty.pop(id(store), None)

    def _loop(self) -> None:
        while True:
            with self._cond:
                if not self._stopping:
                    self._cond.wait(_TICK_SECONDS)
                stopping = self._stopping
                stores = list(self._dirty.values())
                pressed = self._buffered_rows > self._max_buffered
            if pressed:  # the largest buffers first, until half the cap is left
                stores.sort(key=lambda s: s._buffered(), reverse=True)
            now = time.monotonic()
            for store in stores:
                force = stopping or (pressed and self._buffered_rows > self._max_buffered // 2)
                self._write_due(store, now, force)  # a failure is the store's, raised by its next call
                if not force and store.save_format == "parquet":
                    store._convert()  # what is waiting, to Arrow, so the part is ready when it is due
            if stopping:
                with self._cond:
                    if not self._dirty:
                        return


def open_store(
    path: Any,
    test_start: Optional[str] = None,
    lock: bool = False,
    save_coefficients: bool = True,
    writer: Optional[StoreWriter] = None,
) -> _Store:
    """A model's results store, by the path's suffix: `ParquetResultStore` for `<name>.parquet` (a directory of parts and a
    catalog, `results.parquet_store`), `ResultStore` for anything else (`<name>.jsonl`). `writer` shares one `StoreWriter` among
    many stores; without it each writes on its own thread."""
    if is_parquet_path(path):
        from .parquet_store import ParquetResultStore

        return ParquetResultStore(
            path, test_start=test_start, lock=lock, save_coefficients=save_coefficients, writer=writer
        )
    return ResultStore(path, test_start=test_start, lock=lock, save_coefficients=save_coefficients, writer=writer)


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
        import fcntl  # noqa: F401  (a flock to take)
    except ImportError:  # pragma: no cover (Windows)
        return _hold_filelock(path)
    here = _here(path)
    with _HELD_HERE_LOCK:
        if here in _HELD_HERE:  # this process holds it: refused without touching the file (`_HELD_HERE`)
            raise _LockHeld()
        _HELD_HERE[here] = ""
    try:
        fd = _open_lock_file(path)
    except BaseException:
        _forget_here(here)
        raise
    try:
        return _flock_held(path, fd, here)
    except BaseException:
        _forget_here(here)
        raise


def _open_lock_file(path: Path) -> int:
    try:
        return os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except PermissionError:
        return os.open(path, os.O_RDONLY)  # raises PermissionError if it can't be read either


def _flock_held(path: Path, fd: int, here: str) -> Callable[[], None]:
    """The exclusive flock of `_hold_lock` on the open lock file `fd`, and what releases it."""
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno in _HELD_ERRNOS:
            raise _LockHeld() from None
        raise
    if _held_by_a_group(path, fd):  # a `StoreWriter` holds it through its own lock (`_GroupLock`)
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        raise _LockHeld()
    _HELD_FDS.add(fd)

    def release() -> None:
        _HELD_FDS.discard(fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
            _forget_here(here)

    return release


# The lock files this process holds (resolved path -> "" for a lock of its own, a group's name for a model a `_GroupLock` holds).
# A second writer in this process is refused from here without opening the file: where flock is emulated by POSIX locks (NFS),
# a process's locks don't conflict with each other and closing any descriptor of the file drops all of them, so probing a lock
# file this process holds could release it.
_HELD_HERE: Dict[str, str] = {}
_HELD_HERE_LOCK = Lock()


def _here(path: Any) -> str:
    return os.path.realpath(path)


def _forget_here(here: str) -> None:
    with _HELD_HERE_LOCK:
        _HELD_HERE.pop(here, None)


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
    _HELD_HERE.clear()  # the child holds none of them


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_close_held_in_child)


# The name a `_GroupLock` writes into each lock file it holds: `.writer-<pid>-<random>.lock`, a file next to it.
_GROUP_PREFIX = ".writer-"


def _held_by_a_group(lock_path: Path, fd: int, ours: Optional[str] = None) -> bool:
    """Whether the lock file open as `fd` names a `_GroupLock` (other than `ours`) whose own file another holds the flock of."""
    try:
        name = os.pread(fd, 256, 0).strip().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if not name.startswith(_GROUP_PREFIX) or name == ours or "/" in name or "\\" in name:
        return False
    return _group_file_held(lock_path.parent / name)


def _group_file_held(path: Path) -> bool:
    """Whether a `_GroupLock`'s file is held: by this process (`_HELD_HERE`, never probed), or by another, whose flock refuses a
    shared one. One nobody holds belongs to a writer that is gone (a killed one's flock went with it). It is left on disk: only
    the writer that made it removes it."""
    import fcntl

    with _HELD_HERE_LOCK:
        if _here(path) in _HELD_HERE:
            return True
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as exc:
            return exc.errno in _HELD_ERRNOS
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return False


class _GroupLock:
    """One `StoreWriter`'s lock for every model whose results it writes, so that its stores keep no descriptor each.

    The writer holds an exclusive flock on a file of its own in each directory (`.writer-<pid>-<random>.lock`) and writes that
    name into the lock file of each model it takes (`<name>.lock`, the file a model's own lock is a flock of). It takes a model
    under the model's flock, refused if another holds that flock or the name of another group whose flock is held; anyone who
    then takes the model's flock reads the name and is refused while the group holds its own (`_hold_lock` checks it). So a
    model has one writer whether its lock is held by itself or by a group, in another process or in this one (a flock is per
    open file). A store's close clears the name; a killed writer's flock goes with its process, which leaves a name nobody
    holds: stale, and taken over. Where there is no flock, each store holds its own lock."""

    def __init__(self):
        self.name = f"{_GROUP_PREFIX}{os.getpid()}-{uuid.uuid4().hex[:12]}.lock"
        self._held: Dict[Path, Callable[[], None]] = {}
        self._mutex = Lock()

    def hold(self, lock_path: Path) -> Callable[[], None]:
        try:
            import fcntl
        except ImportError:  # pragma: no cover (Windows)
            return _hold_lock(lock_path)
        directory = lock_path.parent
        with self._mutex:
            if directory not in self._held:
                self._held[directory] = _hold_lock(directory / self.name)
        here = _here(lock_path)
        with _HELD_HERE_LOCK:
            if here in _HELD_HERE:  # held in this process, alone or by a group: refused without touching the file
                raise _LockHeld()
        try:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        except PermissionError:  # can't write the name into it: the model's own lock, a descriptor of its own
            return _hold_lock(lock_path)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in _HELD_ERRNOS:
                    raise _LockHeld() from None
                raise
            try:
                if _held_by_a_group(lock_path, fd, ours=self.name):
                    raise _LockHeld()
                with _HELD_HERE_LOCK:
                    if here in _HELD_HERE:
                        raise _LockHeld()
                    _HELD_HERE[here] = self.name
                os.ftruncate(fd, 0)
                os.pwrite(fd, self.name.encode("utf-8"), 0)
            except BaseException:
                with _HELD_HERE_LOCK:
                    if _HELD_HERE.get(here) == self.name:
                        del _HELD_HERE[here]
                raise
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        return lambda: self._clear(lock_path)

    def _clear(self, lock_path: Path) -> None:
        """Clear this group's name from a model's lock file: its store closed. Another takes the file's flock only for a moment
        (to read the name), so this waits a little for it and otherwise leaves the name, which goes stale with the group."""
        import fcntl

        here = _here(lock_path)
        with _HELD_HERE_LOCK:
            if _HELD_HERE.get(here) == self.name:
                del _HELD_HERE[here]
        try:
            fd = os.open(lock_path, os.O_RDWR)
        except OSError:
            return
        try:
            for _ in range(100):
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in _HELD_ERRNOS:
                        return
                    time.sleep(0.01)
            else:
                return
            try:
                if os.pread(fd, 256, 0).strip() == self.name.encode("utf-8"):
                    os.ftruncate(fd, 0)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def release(self) -> None:
        """Release the group's own flocks and remove their files: every model it held is free."""
        with self._mutex:
            held, self._held = self._held, {}
        for directory, release in held.items():
            release()
            try:
                (directory / self.name).unlink()
            except OSError:
                pass


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
