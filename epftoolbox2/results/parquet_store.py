"""Parquet results (`run(save_format="parquet")`, D4 of the forecasting plan): the records JSONL holds, as parts of a directory.

A model's results are `save_dir/<name>.parquet/`, beside `<name>.manifest.json` (which `manifest_path` derives from the stem) and
`<name>.lock`, the lock its JSONL file has too:

    <name>.parquet/year=YYYY/part-<run id>-<seq>.parquet   zstd, one row group, rows sorted by run_date, hour, horizon
    <name>.parquet/_catalog.jsonl                          one line per committed part

Partitioned by model (the directory) and the year of `run_date`. A part is written to a temporary name, synced and renamed, and
only then given its catalog line (file, rows, bytes, first and last run date, sha256 of the file, and the CRC-32 of the line).
Readers read the catalog and the parts it lists, nothing else, so a crash leaves at most an orphan file nobody reads, which
`verify` reports and `compact` removes (a half-written Parquet file would break a whole dataset read, store report 3.3). A
catalog line a crash left half-written is skipped with a warning when read and cut off when a writer opens with the lock; a
damaged line before it raises.

Records are read by the JSONL rules (`results.store.read_records`): one per key `(hour, horizon, run_date)`, the highest
`revision`, within it a forecast over skip records and the first forecast in catalog order, a repeat with the same forecast
warned once, one with another raising; forecasts alone by default. `compact` merges a (model, year)'s parts into one, keeping
every record in order, catalogues it as superseding them, then deletes them. `convert` turns a model's JSONL file into a
directory and back, record for record (`results.parquet_schema`), and `verify` checks catalogs, parts, orphans and repeated keys.

pyarrow is imported inside the functions only, so `import epftoolbox2` doesn't load `pyarrow.parquet`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from . import parquet_schema as schema
from .store import (
    BACKTEST_REVISION,
    LIVE_LOCK_STEM,
    LIVE_LOG_STEMS,
    MODE_LIVE,
    OK_ONLY,
    STATUS_OK,
    ResultsLockedError,
    StoreWriter,
    _hold_lock,
    _live_mismatch,
    _LockHeld,
    _preview,
    _Store,
    check_reading,
    forecast_text,
    is_live_dir,
    is_parquet_path,
    outside_stacklevel,
    repeat_verdict,
)

CATALOG = "_catalog.jsonl"
# When a writer's buffer of a (model, year) becomes a part: at this many rows, this old, or when the writer holds this many rows
# in all (its largest buffers first). Environment variables of the same names override them.
PARQUET_ROWS_PER_PART = 200_000
PARQUET_FLUSH_SECONDS = 300.0
PARQUET_MAX_BUFFERED_ROWS = 1_000_000
_CRC_FIELD = b',"crc32":"'
# The columns the per-key rule reads, and `mode`, which says whether a record is live (`results.store.check_reading`).
_RULE_COLUMNS = ("hour", "horizon", "run_date", "revision", "status", "mode")
# What a part is written with: zstd, page checksums, dictionaries for everything but the floats, which are split by byte.
_FLOAT_COLUMNS = ("prediction", "actual")
_ZSTD_LEVEL = 3


def limits() -> Tuple[int, float, int]:
    """(rows per part, seconds a buffer may wait, rows a writer may hold), from the environment where set."""

    def read(name: str, default: Any, kind: Callable) -> Any:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        value = kind(raw)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {raw!r}")
        return value

    return (
        read("PARQUET_ROWS_PER_PART", PARQUET_ROWS_PER_PART, int),
        read("PARQUET_FLUSH_SECONDS", PARQUET_FLUSH_SECONDS, float),
        read("PARQUET_MAX_BUFFERED_ROWS", PARQUET_MAX_BUFFERED_ROWS, int),
    )


# --- catalog ----------------------------------------------------------------------------------------------------------------


def _catalog_line(entry: Dict[str, Any]) -> bytes:
    """A catalog entry as its line: compact JSON with the CRC-32 of the bytes before the CRC field appended as its last key."""
    body = json.dumps(entry, separators=(",", ":")).encode("utf-8")
    crc = zlib.crc32(body[:-1] + b"}")
    return body[:-1] + _CRC_FIELD + f"{crc:08x}".encode() + b'"}\n'


def _parse_entry(line: bytes) -> Optional[Dict[str, Any]]:
    """The entry of a catalog line, None if it isn't one with its CRC right."""
    at = line.rfind(_CRC_FIELD)
    if at < 0 or not line.endswith(b'"}'):
        return None
    try:
        crc = int(line[at + len(_CRC_FIELD) : -2], 16)
        entry = json.loads(line.decode("utf-8"))
    except ValueError:
        return None
    if zlib.crc32(line[:at] + b"}") != crc or not isinstance(entry, dict) or "file" not in entry:
        return None
    entry.pop("crc32", None)
    return entry


@dataclass
class Catalog:
    """A directory's catalog as read: every entry in order, and what reading it found."""

    entries: List[Dict[str, Any]]
    torn_tail: Optional[bytes] = None  # a last line without its newline that isn't an entry

    def live(self) -> List[Dict[str, Any]]:
        """The committed parts a reader reads, in catalog order: every entry no later entry supersedes."""
        superseded = {name for entry in self.entries for name in entry.get("supersedes", ())}
        return [entry for entry in self.entries if entry["file"] not in superseded]

    def superseded(self) -> List[str]:
        return sorted({name for entry in self.entries for name in entry.get("supersedes", ())})


def read_catalog(directory: Path, warn: bool = True) -> Catalog:
    """The catalog of a Parquet results directory. A last line without its newline that isn't an entry is what a crash leaves
    while a line is appended: skipped, with a warning when `warn`. A damaged line anywhere else raises ValueError naming its
    number."""
    path = Path(directory) / CATALOG
    if not path.exists():
        return Catalog([])
    data = path.read_bytes()
    lines = data.split(b"\n")
    entries = []
    torn = None
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        entry = _parse_entry(line)
        if entry is None:
            if number == len(lines):  # no newline after it
                torn = raw
                if warn:
                    warnings.warn(
                        f"{path}: its last line ({len(raw)} bytes, line {number}) is not a whole catalog entry, as a run that "
                        f"stopped while committing a part leaves it: {_preview(line)}. Its part is not read; the next run "
                        "cuts the line off and forecasts its keys again, and `epf results compact` removes the part.",
                        UserWarning,
                        stacklevel=4,
                    )
                break
            raise ValueError(
                f"{path}: line {number} is not a catalog entry, or its CRC-32 doesn't match ({_preview(line)}): the "
                "catalog was damaged, so which parts hold the results is unknown. `epf results verify` lists what is there."
            )
        entries.append(entry)
    return Catalog(entries, torn)


def _repair_catalog(directory: Path) -> None:
    """Cut off a last catalog line a crash left half-written, with a warning naming its bytes, before a writer appends after it;
    end a whole last entry that lost only its newline with one."""
    path = Path(directory) / CATALOG
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    start = data.rfind(b"\n") + 1
    tail = data[start:]
    with open(path, "rb+") as f:
        if _parse_entry(tail.strip()) is not None:
            f.seek(len(data))
            f.write(b"\n")
            return
        f.truncate(start)
    warnings.warn(
        f"{path}: cut off its last {len(tail)} bytes (from byte {start}), a catalog line a run that stopped while committing a "
        f"part left half-written: {_preview(tail.strip())}. Its part's keys are forecast again, and `epf results compact` "
        "removes the part.",
        UserWarning,
        stacklevel=outside_stacklevel(),
    )


def _append_entry(directory: Path, entry: Dict[str, Any]) -> None:
    """Commit a part: its line appended to the catalog and synced (and the directory, when this creates the catalog)."""
    path = Path(directory) / CATALOG
    created = not path.exists()
    with open(path, "ab") as f:
        f.write(_catalog_line(entry))
        f.flush()
        os.fsync(f.fileno())
    if created:
        _sync_directory(path.parent)


def _sync_directory(directory: Path) -> None:
    """fsync a directory, so that a name created or renamed in it survives a power cut; nothing where directories can't be
    opened (Windows)."""
    if os.name == "nt":  # pragma: no cover
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class MissingPartError(ValueError):
    """A part the catalog lists isn't on disk: the results can't be read as committed."""


# --- parts ------------------------------------------------------------------------------------------------------------------


def _year_of(record: Dict[str, Any], where: Any) -> str:
    run_date = record.get("run_date")
    if not isinstance(run_date, str) or len(run_date) < 4 or not run_date[:4].isdigit():
        raise ValueError(f"{where}: a Parquet results record needs its run_date as an ISO date, got {run_date!r}")
    return run_date[:4]


def _write_part(directory: Path, year: str, records: Sequence[Dict[str, Any]], name: str) -> Dict[str, Any]:
    """Write `records` (of one year) as the part `year=<year>/<name>` (`_write_table`). Returns its catalog entry."""
    return _write_table(directory, year, schema.to_table(records, where=str(directory)), name)


def _write_table(directory: Path, year: str, table, name: str) -> Dict[str, Any]:
    """Write `table` (rows of one year, in the order written) as the part `year=<year>/<name>`: sorted by run_date, hour,
    horizon (stably, so a key's records keep their order), to a temporary name, synced, then renamed. Returns its catalog
    entry; the part isn't committed until that is appended."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = table.take(schema.sort_indices(table))
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        compression_level=_ZSTD_LEVEL,
        row_group_size=max(1, table.num_rows),
        use_dictionary=[n for n in table.column_names if n not in _FLOAT_COLUMNS and schema.KINDS.get(n) != "floats"],
        use_byte_stream_split=list(_FLOAT_COLUMNS),
        write_page_checksum=True,
        write_statistics=["run_date"],
    )
    data = sink.getvalue().to_pybytes()
    relative = f"year={year}/{name}"
    final = Path(directory) / relative
    if not final.parent.is_dir():
        final.parent.mkdir(parents=True, exist_ok=True)
        _sync_directory(final.parent.parent)
    tmp = final.with_name(f".{final.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
        _sync_directory(final.parent)  # the rename, before the catalog line that commits it
    finally:
        if tmp.exists():
            tmp.unlink()
    dates = table.column("run_date")
    import pyarrow.compute as pc

    bounds = pc.min_max(dates)
    return {
        "file": relative,
        "rows": table.num_rows,
        "bytes": len(data),
        "first": None if bounds["min"].as_py() is None else bounds["min"].as_py().isoformat(),
        "last": None if bounds["max"].as_py() is None else bounds["max"].as_py().isoformat(),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _read_part(directory: Path, entry: Dict[str, Any], columns: Optional[Sequence[str]], conform: bool = True):
    import pyarrow.parquet as pq

    path = Path(directory) / entry["file"]
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        raise MissingPartError(
            f"{path} is listed in {Path(directory) / CATALOG} as a committed part and isn't on disk, so these results can't "
            "be read (or resumed) as they were committed. `epf results verify` lists what is missing; restore the file, or "
            "write the model's results again into a new save_dir."
        ) from None
    if size != entry.get("bytes", size):
        raise ValueError(
            f"{path} has {size} bytes, its catalog line says {entry['bytes']}: the part changed after it was committed. "
            "`epf results verify` checks every part."
        )
    table = pq.ParquetFile(path, page_checksum_verification=True).read(
        columns=None if columns is None else list(columns)
    )
    if table.num_rows != entry["rows"]:
        raise ValueError(f"{path} holds {table.num_rows} rows, its catalog line says {entry['rows']}")
    return _conform(table, columns) if conform else table


def _concat_parts(tables: List[Any]):
    """Parts read as they are, as one table: of their own schema where they share one (a backtest's compaction writes what it
    wrote before the live columns), else each with every column (`_conform`)."""
    import pyarrow as pa

    if any(not table.schema.equals(tables[0].schema) for table in tables[1:]):
        tables = [_conform(table, None) for table in tables]
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


def _conform(table, columns: Optional[Sequence[str]]):
    """`table` with the schema's columns it lacks (`columns`, or all of them) as nulls, in schema order: a part written before a
    column was added to the schema (the live keys of step 8c) reads, and concatenates, as one written after."""
    import pyarrow as pa

    names = [*schema.NAMES, schema.KEYS_COLUMN] if columns is None else list(columns)
    if table.column_names == names:
        return table
    held = set(table.column_names)
    full = schema.arrow_schema()
    arrays, fields = [], []
    for name in names:
        if name in held:
            arrays.append(table.column(name))
            fields.append(table.schema.field(name))
        elif name in full.names:
            arrays.append(pa.nulls(table.num_rows, full.field(name).type))
            fields.append(full.field(name))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _read_columns(columns: Optional[Sequence[str]]) -> Optional[List[str]]:
    """The columns to read for `columns`: those, the rule's and `_keys`, in schema order; None for all."""
    if columns is None:
        return None
    wanted = set(columns) | set(_RULE_COLUMNS)
    return [name for name in schema.NAMES if name in wanted] + [schema.KEYS_COLUMN]


def read_table(directory: Path, columns: Optional[Sequence[str]] = None, entries: Optional[List[Dict]] = None):
    """The committed rows of a directory, its live parts in catalog order, as one table (`columns` and the rule's, or all). A
    part a compaction removed between reading the catalog and the part is read again from the new catalog."""
    import pyarrow as pa

    directory = Path(directory)
    read = _read_columns(columns)
    for attempt in range(3):
        live = read_catalog(directory).live() if entries is None else entries
        try:
            tables = [_read_part(directory, entry, read) for entry in live]
            break
        except MissingPartError:  # a compaction may have replaced it meanwhile: the catalog again, then it is missing
            if entries is not None or attempt == 2:
                raise
    if not tables:
        empty = schema.arrow_schema()
        if read is not None:
            empty = pa.schema([empty.field(name) for name in read])
        return empty.empty_table()
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


# --- the per-key rule -------------------------------------------------------------------------------------------------------


def _rule_arrays(table):
    """numpy arrays of what the rule reads: whether a row has a whole key, its key, revision and whether it is a forecast."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    def ints(name: str, missing: int):
        column = table.column(name)
        if pa.types.is_date32(column.type):
            column = column.cast(pa.int32())
        return pc.fill_null(column, missing).to_numpy().astype(np.int64)

    valid = np.ones(table.num_rows, dtype=bool)
    for name in ("hour", "horizon", "run_date"):
        valid &= ~table.column(name).is_null().to_numpy(zero_copy_only=False)
    status = pc.cast(table.column("status"), pa.string())
    ok = pc.fill_null(pc.equal(status, STATUS_OK), True).to_numpy(zero_copy_only=False)
    return (
        valid,
        ints("run_date", 0),
        ints("hour", 0),
        ints("horizon", 0),
        ints("revision", BACKTEST_REVISION),
        ok,
    )


def _visible_rows(valid, day, revision, visible: frozenset):
    """Whether each row's (run_date, revision) is one of `visible`, for rows with a whole key."""
    import numpy as np

    from datetime import date

    epoch = date(1970, 1, 1)
    codes = np.fromiter(
        ((date.fromisoformat(run_date) - epoch).days * 65536 + revision for run_date, revision in visible),
        dtype=np.int64,
        count=len(visible),
    )
    return valid & np.isin(day * 65536 + revision, codes)


def choose(table, visible: Optional[frozenset] = None) -> Tuple[Any, List[Tuple[int, int]]]:
    """The rows the readers' rule chooses, one per key, in table order, and (chosen, repeat) pairs of the forecasts repeating a
    chosen forecast at its revision. Per key: the highest revision; within it the first forecast, else the last skip record
    (the JSONL reader's rule, `results.store._choose`). A row without a whole key is its own. With `visible` ((run_date,
    revision) pairs of a live directory's reader, `results.store.visible_revisions`) only the rows of those revisions are
    there, as the JSONL reader passes the other lines over."""
    import numpy as np

    n = table.num_rows
    if n == 0:
        return np.zeros(0, dtype=np.int64), []
    valid, day, hour, horizon, revision, ok = _rule_arrays(table)
    order = np.arange(n, dtype=np.int64)
    if visible is not None:
        seen = _visible_rows(valid, day, revision, visible)
        order = order[seen]
        valid, day, hour, horizon, revision, ok = (a[seen] for a in (valid, day, hour, horizon, revision, ok))
        if not len(order):
            return order, []
    alone = np.where(valid, 0, order + 1)
    skip = ~ok
    tie = np.where(skip, -order, order)
    idx = np.lexsort((tie, skip, -revision, horizon, hour, day, alone))
    s_alone, s_day, s_hour, s_horizon = alone[idx], day[idx], hour[idx], horizon[idx]
    start = np.ones(len(order), dtype=bool)
    start[1:] = (
        (s_alone[1:] != s_alone[:-1])
        | (s_day[1:] != s_day[:-1])
        | (s_hour[1:] != s_hour[:-1])
        | (s_horizon[1:] != s_horizon[:-1])
    )
    if start.all():
        chosen, repeats = idx, []
    else:
        chosen = idx[start]
        group = np.cumsum(start) - 1
        chosen_of = chosen[group]
        candidate = ~start & ok[idx] & ok[chosen_of] & (revision[idx] == revision[chosen_of])
        repeats = list(zip(order[chosen_of[candidate]].tolist(), order[idx[candidate]].tolist()))
    if visible is not None:  # each issue day from its highest revision alone (`results.store._day_top_revisions`)
        days, inverse = np.unique(day[chosen], return_inverse=True)
        top = np.full(len(days), np.iinfo(np.int64).min, dtype=np.int64)
        np.maximum.at(top, inverse, revision[chosen])
        dropped = set(order[chosen[revision[chosen] != top[inverse]]].tolist())
        chosen = chosen[revision[chosen] == top[inverse]]
        repeats = [pair for pair in repeats if pair[0] not in dropped]
    if not repeats and len(chosen) == len(order):
        return order, []
    return np.sort(order[chosen]), repeats


def _check_repeats(directory: Path, table, repeats: List[Tuple[int, int]], duplicates: str, stacklevel: int) -> None:
    """Compare each repeated forecast with its key's chosen one, without D4's keys (`results.store.forecast_text`): the same
    warn once, another raises (or is read as the first with `duplicates="first"`)."""
    rows = sorted({row for pair in repeats for row in pair})
    records = dict(zip(rows, schema.iter_dicts(table, rows=rows)))
    same = differ = 0
    example = None
    for chosen, repeat in repeats:
        if forecast_text(records[chosen]) == forecast_text(records[repeat]):
            same += 1
        else:
            differ += 1
            if example is None:
                r = records[chosen]
                example = (r.get("hour"), r.get("horizon"), r.get("run_date"))
    repeat_verdict(directory, same, differ, example, duplicates, stacklevel=stacklevel)


def _status_mask(table, rows, statuses: Optional[Iterable[str]]):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    if statuses is None:
        return np.ones(len(rows), dtype=bool)
    status = pc.fill_null(pc.cast(table.column("status"), pa.string()), STATUS_OK)
    keep = pc.is_in(status, value_set=pa.array(sorted(set(statuses)), pa.string())).to_numpy(zero_copy_only=False)
    return keep[rows]


def chosen_rows(
    directory: Path,
    table,
    statuses: Optional[Iterable[str]],
    duplicates: str = "raise",
    stacklevel: int = 5,
    visible: Optional[frozenset] = None,
):
    """The rows of `table` (read from `directory`, any columns with the rule's) that the readers' rule gives for `statuses`, in
    table order; with `visible`, of those revisions alone (`choose`). Repeats are compared on every column, read again for them
    if `table` holds fewer. A chosen live record read without `visible`, or one that isn't read with it, raises
    `LiveDirectoryError`, as the JSONL reader's second pass does."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    chosen, repeats = choose(table, visible)
    if repeats:
        full = table if set(schema.NAMES) <= set(table.column_names) else read_table(directory)
        _check_repeats(directory, full, repeats, duplicates, stacklevel)
    rows = chosen[_status_mask(table, chosen, statuses)]
    if len(rows):
        mode = table.column("mode").take(pa.array(rows, pa.int64()))
        live = pc.fill_null(pc.equal(pc.cast(mode, pa.string()), MODE_LIVE), False).to_numpy(zero_copy_only=False)
        wrong = np.flatnonzero(live != (visible is not None))
        if len(wrong):
            raise _live_mismatch(directory, f"the record of row {int(rows[wrong[0]])}", visible)
    return rows


def read_records(
    directory: Path,
    statuses: Optional[Iterable[str]] = OK_ONLY,
    transform: Optional[Callable[[Dict], Any]] = None,
    duplicates: str = "raise",
    columns: Optional[Sequence[str]] = None,
    visible: Optional[frozenset] = None,
) -> Iterator[Any]:
    """One record per forecast key of a Parquet results directory, by the JSONL reader's rules (`results.store.read_records`),
    in catalog order, `transform(record)` of each if given (None drops it). With `columns`, the records hold those keys (and the
    rule's) alone. `visible` are the revisions a live directory's reader may see (`results.store.visible_revisions`)."""
    if duplicates not in ("raise", "first"):
        raise ValueError(f"duplicates must be 'raise' or 'first', not {duplicates!r}")
    check_reading(directory, visible)
    table = read_table(directory, columns)
    rows = chosen_rows(directory, table, statuses, duplicates, stacklevel=6, visible=visible)
    for start in range(0, len(rows), 8192):  # dicts a batch at a time
        for record in schema.iter_dicts(table, rows=rows[start : start + 8192].tolist()):
            out = record if transform is None else transform(record)
            if out is not None:
                yield out


def max_revision(directory: Path, run_date: str) -> int:
    """The highest `revision` any committed part's record of `run_date` carries (0 for none), whatever its status
    (`results.store.max_revision`)."""

    table = read_table(Path(directory), ["run_date", "revision"])
    valid, day, _, _, revision, _ = _rule_arrays(table)
    from datetime import date

    mine = valid & (day == (date.fromisoformat(run_date) - date(1970, 1, 1)).days)
    return int(revision[mine].max()) if mine.any() else 0


def iter_file_records(directory: Path) -> Iterator[Dict]:
    """Every committed record of a directory as it stands (no per-key rule), in catalog order, a part at a time and in batches,
    for a reader that stops early (`ModelResultRef.first_record`) and for `convert` and `compact`."""
    import pyarrow as pa

    directory = Path(directory)
    for entry in read_catalog(directory).live():
        table = _read_part(directory, entry, None)
        for batch in table.to_batches(max_chunksize=1024):
            yield from schema.iter_dicts(pa.Table.from_batches([batch], schema=table.schema))


# --- a frame without dicts (`ModelResultRef.frame`) -------------------------------------------------------------------------


_DATE_DTYPE = None


def _date_dtype():
    """The dtype the dict path gives dates (`EvaluationReport._model_frame`: `pd.to_datetime` of ISO text)."""
    global _DATE_DTYPE
    if _DATE_DTYPE is None:
        import pandas as pd

        _DATE_DTYPE = pd.to_datetime(pd.Series(["2000-01-01"], dtype="object"), format="ISO8601").dtype
    return _DATE_DTYPE


def _series(column, name: str):
    """A column as the pandas Series a list of its record values gives (`pd.DataFrame({name: values})`): ints int64 (float64
    with a null among them), floats float64 (object if every value is null), dates the dict path's datetime dtype, text and
    lists objects."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa

    kind = schema.KINDS.get(name)
    n = len(column)
    nulls = column.null_count
    if kind == "date":
        values = column.to_numpy(zero_copy_only=False)
        return pd.Series(np.asarray(values, dtype="datetime64[D]").astype(_date_dtype()), name=name)
    if kind in ("int8", "int16", "int32"):
        values = column.to_numpy(zero_copy_only=False)
        return pd.Series(values.astype(np.int64) if nulls == 0 else values.astype(np.float64), name=name)
    if kind == "float64":
        if n and nulls == n:
            return pd.Series([None] * n, dtype=object, name=name)
        return pd.Series(column.to_numpy(zero_copy_only=False).astype(np.float64), name=name)
    if kind == "category":
        column = column.cast(pa.string())
    return pd.Series(column.to_pylist(), dtype=object, name=name)


def read_frame(
    directory: Path,
    columns: Sequence[str],
    keep: Callable[[Any], Any],
    statuses: Optional[Iterable[str]] = OK_ONLY,
    extra: Sequence[str] = (),
    visible: Optional[frozenset] = None,
):
    """The records `read_records` gives, as a DataFrame of `columns` (`_series` per column), without a dict per record: the
    rule's rows, then those `keep(table, rows)` (a boolean array over `rows`) keeps, in catalog order. `extra` are the columns
    `keep` reads; `visible` the revisions a live directory's reader may see."""
    import pandas as pd
    import pyarrow as pa

    check_reading(directory, visible)
    table = read_table(directory, [*columns, *extra])
    rows = chosen_rows(directory, table, statuses, stacklevel=7, visible=visible)
    rows = rows[keep(table, rows)]
    taken = table.take(pa.array(rows, pa.int64()))
    return pd.DataFrame({name: _series(taken.column(name).combine_chunks(), name) for name in columns})


# --- the store --------------------------------------------------------------------------------------------------------------


# How many records of a buffer the writer's thread converts to Arrow at a time while a run goes on (`ParquetResultStore._convert`),
# so that a part is mostly converted by the time it is written, and a buffer holds compact columns, not dicts.
_CONVERT_ROWS = 4096


class _YearBuffer:
    """A (model, year)'s records waiting for their part: converted chunks (`tables`, Arrow, in order) and then the records not yet
    converted (`records`, with their keys in `layouts`)."""

    __slots__ = ("tables", "converted", "records", "layouts", "keys", "since")

    def __init__(self, since: float):
        self.tables: List[Any] = []
        self.converted = 0  # rows in `tables`
        self.records: List[Dict] = []
        self.layouts: List[Tuple[str, ...]] = []  # each record's keys (`ParquetResultStore._validate`)
        self.keys: List[Optional[tuple]] = []  # each row's forecast key (None for a skip record), converted rows first
        self.since = since

    @property
    def rows(self) -> int:
        return self.converted + len(self.records)

    def take(
        self, rows: Optional[int] = None
    ) -> Tuple[List[Any], List[Dict], List[Tuple[str, ...]], List[Optional[tuple]]]:
        """The first `rows` rows (all of them by default), removed: (tables, records, layouts, keys), in order. A table split
        at the boundary is sliced, which copies nothing."""
        if rows is None or rows >= self.rows:
            taken = (self.tables, self.records, self.layouts, self.keys)
            self.tables, self.converted, self.records, self.layouts, self.keys = [], 0, [], [], []
            return taken
        tables, left = [], rows
        while self.tables and left:
            table = self.tables[0]
            if table.num_rows <= left:
                tables.append(self.tables.pop(0))
                left -= table.num_rows
            else:
                tables.append(table.slice(0, left))
                self.tables[0] = table.slice(left)
                left = 0
        self.converted -= rows - left
        records, self.records = self.records[:left], self.records[left:]
        layouts, self.layouts = self.layouts[:left], self.layouts[left:]
        keys, self.keys = self.keys[:rows], self.keys[rows:]
        return tables, records, layouts, keys


class ParquetResultStore(_Store):
    """Resumable Parquet storage for one model's results (`<name>.parquet/`, see the module docstring), with the resume state,
    lock and rules of `ResultStore` (`results.store._Store`).

    Records are buffered per year of their `run_date` and written as a part when the buffer reaches `PARQUET_ROWS_PER_PART`
    rows, is `PARQUET_FLUSH_SECONDS` old, or the writer holds too many (`StoreWriter`), and on `flush()` and `close()`; a part is
    read once its catalog line is written. Without `writer` the store has a `StoreWriter` of its own. Opened with the lock, a
    store cuts off a catalog line a crash left half-written."""

    save_format = "parquet"

    def __init__(
        self,
        path: Any,
        test_start: Optional[str] = None,
        lock: bool = False,
        save_coefficients: bool = True,
        writer: Optional[StoreWriter] = None,
    ):
        self._private = writer is None
        writer = StoreWriter(_private=True) if writer is None else writer
        self._buffers: Dict[str, _YearBuffer] = {}
        self._layouts: Dict[Tuple[str, ...], Tuple[str, ...]] = {}  # each layout validated, once
        self._tail_checked = False  # a torn last catalog line cut: at a locked open, else before the first part
        self._rows = 0
        self._io_lock = threading.Lock()
        self._run_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"
        self._seq = 0
        self._rows_per_part, self._flush_seconds, _ = limits()
        try:
            super().__init__(path, test_start, lock, save_coefficients, writer)
        except BaseException:
            if self._private:
                writer.close()
            raise
        writer._register(self)

    def flush(self) -> None:
        """Write every buffered record as parts and commit them."""
        self._shared._write_now(self)

    def close(self) -> None:
        """Write what is buffered and release the lock. Saving into the store afterwards raises."""
        self._closed = True
        try:
            self.flush()
        finally:
            self._release_file_lock()
            self._shared._forget(self)
            if self._private:
                self._shared.close()

    def _repair_tail(self) -> None:
        self._tail_checked = True
        _repair_catalog(self.path)

    def _validate(self, record: Dict) -> Tuple[str, ...]:
        """The record's keys, checked against the schema now, not when its part is written; the same tuple for a layout seen
        before, so that `schema.to_table` compares them by identity."""
        keys = tuple(record)
        known = self._layouts.get(keys)
        if known is None:
            schema.check_keys(keys, str(self.path))
            known = self._layouts.setdefault(keys, keys)
        _year_of(record, self.path)
        return known

    def _enqueue(self, record: Dict, layout: Any = None) -> None:
        self._shared._submit(self, record, layout)

    def _buffer(self, record: Dict, layout: Any) -> bool:
        year = record["run_date"][:4]
        buffer = self._buffers.get(year)
        if buffer is None:
            buffer = self._buffers[year] = _YearBuffer(time.monotonic())
        buffer.records.append(record)
        buffer.layouts.append(layout)
        ok = record.get("status", STATUS_OK) == STATUS_OK
        buffer.keys.append((record["hour"], record["horizon"], record["run_date"]) if ok else None)
        self._rows += 1
        return buffer.rows >= self._rows_per_part

    def _buffered(self) -> int:
        return self._rows

    def _take(
        self, now: float, force: bool
    ) -> List[Tuple[str, int, Tuple[List[Any], List[Dict], List[Tuple[str, ...]]]]]:
        """The buffers due, as (year, rows, (tables, records, layouts)): all of them with `force`, a buffer as old as
        `PARQUET_FLUSH_SECONDS` whole, and of a fuller one its whole parts (the rest waits, so a part has
        `PARQUET_ROWS_PER_PART` rows whatever the moment it is taken)."""
        taken = []
        for year in list(self._buffers):
            buffer = self._buffers[year]
            rows = buffer.rows
            if force or now - buffer.since >= self._flush_seconds:
                del self._buffers[year]
                held = buffer.take()
            elif rows >= self._rows_per_part:
                rows -= rows % self._rows_per_part
                held = buffer.take(rows)
                if not buffer.rows:
                    del self._buffers[year]
            else:
                continue
            self._rows -= rows
            taken.append((year, rows, held))
        return taken

    def _convert(self) -> None:
        """Convert each buffer's waiting records to Arrow, `_CONVERT_ROWS` at a time, in place: what the writer's thread does
        between parts, so a part is mostly converted when it is written. Under the store's I/O lock, so nothing takes the buffer
        meanwhile; records saved during the conversion go after it."""
        with self._io_lock:
            with self._shared._cond:
                work = [
                    (buffer, buffer.records[:whole], buffer.layouts[:whole])
                    for buffer in self._buffers.values()
                    for whole in [len(buffer.records) - len(buffer.records) % _CONVERT_ROWS]
                    if whole
                ]
            for buffer, records, layouts in work:
                try:
                    table = schema.to_table(records, str(self.path), layouts)
                except BaseException as exc:  # the store's, raised by its next call, before the I/O lock is let go
                    self._shared._fail(self, exc, time.monotonic())
                    return
                with self._shared._cond:
                    del buffer.records[: len(records)], buffer.layouts[: len(records)]
                    buffer.tables.append(table)
                    buffer.converted += table.num_rows

    def _write(self, taken: List[Tuple[str, int, Tuple[List[Any], List[Dict], List[Tuple[str, ...]]]]]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        if not self._tail_checked:  # as a JSONL store does: never append after a line a crash left half-written
            self._repair_tail()
        for year, rows, (tables, records, layouts, keys) in taken:
            if records:
                tables = [*tables, schema.to_table(records, str(self.path), layouts)]
            table = _concat_parts(tables).unify_dictionaries() if len(tables) > 1 else tables[0]
            for start in range(0, rows, self._rows_per_part):
                self._seq += 1
                part = table.slice(start, self._rows_per_part).combine_chunks()
                with self._shared._handles:
                    entry = _write_table(self.path, year, part, f"part-{self._run_id}-{self._seq:06d}.parquet")
                    _append_entry(self.path, entry)
                self._written(keys[start : start + self._rows_per_part])  # done once committed


# --- convert, compact, verify -----------------------------------------------------------------------------------------------


def _models_in(save_dir: Path, model: Optional[str]) -> List[str]:
    """The stems of the models with results in `save_dir` (or `model`'s, by its name or file stem), ensembles left out: they
    are computed again by every run and report."""
    from .manifest import is_ensemble_manifest, manifest_path

    stems = sorted(
        {p.stem for p in save_dir.glob("*.jsonl") if p.is_file()}
        | {p.stem for p in save_dir.glob("*.parquet") if p.is_dir()}
    )
    if is_live_dir(save_dir):  # a live directory's call and delivery logs are no model's results
        stems = [stem for stem in stems if stem not in LIVE_LOG_STEMS]
    if model is not None:
        wanted = model.lower().replace(" ", "_")
        stems = [stem for stem in stems if stem == wanted]
        if not stems:
            raise FileNotFoundError(f"No results of model {model!r} in {save_dir}")
    return [stem for stem in stems if not is_ensemble_manifest(manifest_path(save_dir / f"{stem}.jsonl"))]


def _refuse_residual_models(save_dir: Path, stems: List[str]) -> None:
    """A ValueError, before anything is converted, where a model of `stems` is a `ResidualModel` (its manifest's
    `model_class`) or the base of one in `save_dir` (`<name>.base`): the residual's records hold `base_prediction`, which the
    Parquet schema has no column for, and it reads its base's results as JSONL."""
    from .manifest import manifest_path

    residual = []
    for stem in _models_in(save_dir, None):
        path = manifest_path(save_dir / f"{stem}.jsonl")
        if path.is_file() and json.loads(path.read_text()).get("model_class") == "ResidualModel":
            residual.append(stem)
    kept = set(residual) | {f"{stem}.base" for stem in residual}
    found = [stem for stem in stems if stem in kept]
    if found:
        raise ValueError(
            f"{save_dir}: {', '.join(found)} are a ResidualModel's results or its base's (<name>.base), which stay JSONL: the "
            "residual model's records hold `base_prediction`, which the Parquet results schema has no column for, and it reads "
            "its base's results as JSONL. Nothing was converted. Convert the other models one by one (model=...), or keep the "
            "directory as JSONL"
        )


def _lock_model(save_dir: Path, stem: str) -> Callable[[], None]:
    if stem == LIVE_LOCK_STEM and is_live_dir(save_dir):
        return lambda: None  # `live.lock`, which `_live_lock` holds around every model of the directory
    lock_path = save_dir / f"{stem}.lock"
    try:
        return _hold_lock(lock_path)
    except _LockHeld:
        raise ResultsLockedError(
            f"{save_dir / stem} is being written by a run, which holds {lock_path}: wait for it to finish"
        ) from None
    except OSError as exc:
        warnings.warn(
            f"{lock_path}: the file system refused a lock ({exc}); make sure no run writes {stem}",
            stacklevel=outside_stacklevel(),
        )
        return lambda: None


def _set_manifest_format(save_dir: Path, stem: str, save_format: str) -> None:
    """Record `save_format` in a model's manifest (no key for JSONL), every other key as it was, in its place."""
    path = save_dir / f"{stem}.manifest.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    data.pop("save_format", None)
    if save_format != "jsonl":
        data["save_format"] = save_format
    path.write_text(json.dumps(data, indent=2, default=str, sort_keys=False) + "\n")


def _lines(records: Iterable[Dict]) -> List[str]:
    return sorted(json.dumps(record) for record in records)


def _write_parquet_dir(target: Path, records: Iterable[Dict]) -> None:
    """Every record into a new directory (`target`, which must not exist), in their order, a part per year and at most
    `PARQUET_ROWS_PER_PART` rows."""
    rows_per_part = limits()[0]
    run_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    target.mkdir(parents=True)
    by_year: Dict[str, List[Dict]] = {}
    seq = 0

    def commit(year: str, chunk: List[Dict]) -> None:
        nonlocal seq
        seq += 1
        _append_entry(target, _write_part(target, year, chunk, f"part-{run_id}-{seq:06d}.parquet"))

    for record in records:
        schema.check_keys(tuple(record), str(target))
        year = _year_of(record, target)
        chunk = by_year.setdefault(year, [])
        chunk.append(record)
        if len(chunk) >= rows_per_part:
            commit(year, by_year.pop(year))
    for year in sorted(by_year):
        commit(year, by_year[year])


def convert(save_dir: Any, to: str, model: Optional[str] = None) -> List[Tuple[str, int]]:
    """Convert every model's results in `save_dir` (or `model`'s) to `to` ("parquet" or "jsonl"): every record, in its order
    and with its keys in theirs, so the other way gives the same lines. Under each model's lock; the new results are written
    beside the old ones, read back and compared record for record before the old ones are removed and the manifest records the
    format. A model already in `to` is left alone. Returns (model stem, records) of each converted model."""
    from ..data.cache_manager import atomic_write

    def iter_any(path: Path, parquet: bool) -> Iterator[Dict]:
        return iter_file_records(path) if parquet else _iter_jsonl(path)

    if to not in ("parquet", "jsonl"):
        raise ValueError(f"to must be 'parquet' or 'jsonl', got {to!r}")
    save_dir = Path(save_dir)
    stems = _models_in(save_dir, model)
    if to == "parquet":
        _refuse_residual_models(save_dir, stems)
    with _live_lock(save_dir):
        return _convert(save_dir, stems, to, iter_any, atomic_write)


def _live_lock(save_dir: Path):
    """A live directory's `live.lock` held (`results.live.LiveStore.hold`, so no live call writes meanwhile), or nothing."""
    from contextlib import nullcontext

    from .live import LiveStore

    return LiveStore(save_dir).hold() if is_live_dir(save_dir) else nullcontext()


def _convert(
    save_dir: Path, stems: List[str], to: str, iter_any: Callable, atomic_write: Callable
) -> List[Tuple[str, int]]:
    done = []
    for stem in stems:
        source = save_dir / f"{stem}.{'jsonl' if to == 'parquet' else 'parquet'}"
        target = save_dir / f"{stem}.{to}"
        if not source.exists():
            continue
        release = _lock_model(save_dir, stem)
        try:
            original = _lines(iter_any(source, to == "jsonl"))
            if target.exists():  # a convert that stopped after writing the new results
                if _lines(iter_any(target, to == "parquet")) != original:
                    raise ValueError(
                        f"{source} and {target} both hold results of {stem}, with different records. Keep the one the runs "
                        "wrote and remove the other."
                    )
            else:
                tmp = save_dir / f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
                try:
                    if to == "parquet":
                        _write_parquet_dir(tmp, iter_any(source, False))
                    else:

                        def write(path: Path) -> None:
                            with open(path, "w", encoding="utf-8") as f:
                                for record in iter_any(source, True):
                                    f.write(json.dumps(record) + "\n")

                        atomic_write(tmp, write)
                    if _lines(iter_any(tmp, to == "parquet")) != original:
                        raise RuntimeError(f"{stem}: the converted records differ from {source}'s; nothing was changed")
                    os.replace(tmp, target)
                finally:
                    if tmp.is_dir():
                        shutil.rmtree(tmp)
                    elif tmp.exists():
                        tmp.unlink()
            _set_manifest_format(save_dir, stem, to)
            if source.is_dir():
                shutil.rmtree(source)
            else:
                source.unlink()
            done.append((stem, len(original)))
        finally:
            release()
    return done


def _files_on_disk(directory: Path) -> List[str]:
    return sorted(
        str(p.relative_to(directory)).replace(os.sep, "/")
        for p in directory.rglob("*")
        if p.is_file() and p.name != CATALOG
    )


def compact(save_dir: Any, model: Optional[str] = None) -> Dict[str, Dict[str, int]]:
    """Merge each (model, year)'s live parts into one, under the model's lock: every record of them in catalog and row order,
    sorted stably as a part is, written and committed as superseding them, then the old parts deleted; then files the catalog
    doesn't list (orphans a crash left) and superseded ones still on disk are deleted. A (model, year) with one part is left as
    it is, so compacting twice changes nothing. Returns model stem -> {"parts_merged", "parts_written", "files_removed"}. In a
    live directory, under its `live.lock` too."""
    save_dir = Path(save_dir)
    with _live_lock(save_dir):
        return _compact(save_dir, _models_in(save_dir, model))


def _compact(save_dir: Path, stems: List[str]) -> Dict[str, Dict[str, int]]:
    summary = {}
    for stem in stems:
        directory = save_dir / f"{stem}.parquet"
        if not directory.is_dir():
            continue
        release = _lock_model(save_dir, stem)
        try:
            _repair_catalog(directory)
            catalog = read_catalog(directory)
            by_year: Dict[str, List[Dict]] = {}
            for entry in catalog.live():
                by_year.setdefault(entry["file"].split("/", 1)[0], []).append(entry)
            merged = written = 0
            run_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"
            for partition, entries in sorted(by_year.items()):
                if len(entries) < 2:
                    continue
                tables = [_read_part(directory, entry, None, conform=False) for entry in entries]
                table = _concat_parts(tables).unify_dictionaries().combine_chunks()
                written += 1
                entry = _write_table(
                    directory, partition.split("=", 1)[1], table, f"part-{run_id}-{written:06d}.parquet"
                )
                entry["supersedes"] = [e["file"] for e in entries]
                _append_entry(directory, entry)
                merged += len(entries)
            listed = {entry["file"] for entry in read_catalog(directory).live()}
            removed = 0
            for name in _files_on_disk(directory):
                if name not in listed:
                    (directory / name).unlink()
                    removed += 1
            for empty in sorted(directory.glob("year=*"), reverse=True):
                if empty.is_dir() and not any(empty.iterdir()):
                    empty.rmdir()
            summary[stem] = {"parts_merged": merged, "parts_written": written, "files_removed": removed}
        finally:
            release()
    return summary


@dataclass
class Finding:
    """What `verify` found: `level` "error" (results that can't be trusted as they stand) or "note" (nothing a reader reads)."""

    level: str
    model: str
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.model}: {self.message}"


def _duplicate_findings(stem: str, where: Path, table) -> List[Finding]:
    """Repeated keys under the readers' rule: forecasts repeating their key's with another forecast (an error) or the same
    (a note), and keys with records the rule leaves unread (lower revisions, skip records under a forecast; a note)."""
    chosen, repeats = choose(table)
    found = []
    unread = len(table) - len(chosen) - len(repeats)
    if repeats:
        rows = sorted({row for pair in repeats for row in pair})
        records = dict(zip(rows, schema.iter_dicts(table, rows=rows)))
        differ = sum(forecast_text(records[a]) != forecast_text(records[b]) for a, b in repeats)
        if differ:
            found.append(
                Finding(
                    "error",
                    stem,
                    f"{where.name}: {differ} forecasts repeat their key with another forecast (reading raises)",
                )
            )
        if len(repeats) - differ:
            found.append(
                Finding(
                    "note",
                    stem,
                    f"{where.name}: {len(repeats) - differ} forecasts repeat their key with the same forecast",
                )
            )
    if unread:
        found.append(
            Finding(
                "note",
                stem,
                f"{where.name}: {unread} records the readers don't read (lower revisions, skip records under a forecast)",
            )
        )
    return found


def verify(save_dir: Any, model: Optional[str] = None) -> List[Finding]:
    """Check the results in `save_dir`: per Parquet directory the CRC-32 of every catalog line, each live part's size, sha256
    and rows against its line, superseded parts still on disk and files the catalog doesn't list (orphans), and per model
    (Parquet or JSONL) repeated keys; a model with results in both formats is an error. Reads, never changes anything."""
    import pyarrow as pa

    save_dir = Path(save_dir)
    found: List[Finding] = []
    committed = None  # a live directory's records are read as its committed revisions are (`results.live`)
    if is_live_dir(save_dir):
        from .live import LiveStore

        committed = LiveStore(save_dir).visible("latest", cutoff=None)
    for stem in _models_in(save_dir, model):
        jsonl, directory = save_dir / f"{stem}.jsonl", save_dir / f"{stem}.parquet"
        if jsonl.exists() and directory.exists():
            found.append(
                Finding("error", stem, f"results in both formats ({jsonl.name} and {directory.name}): keep one")
            )
        if jsonl.exists():
            from .store import read_records as read_any

            try:  # the reader's own rule: a repeat with another forecast raises, the same one warns, as a torn last line does
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    for _ in read_any(jsonl, statuses=None, transform=lambda r: None, committed=committed):
                        pass
                found.extend(Finding("note", stem, str(w.message)) for w in caught)
            except ValueError as exc:
                found.append(Finding("error", stem, str(exc)))
        if not directory.is_dir():
            continue
        try:
            catalog = read_catalog(directory, warn=False)
        except ValueError as exc:
            found.append(Finding("error", stem, str(exc)))
            continue
        if catalog.torn_tail is not None:
            found.append(
                Finding("note", stem, f"{CATALOG}: a half-written last line ({len(catalog.torn_tail)} bytes), not read")
            )
        live = catalog.live()
        tables = []
        for entry in live:
            path = directory / entry["file"]
            if not path.exists():
                found.append(
                    Finding(
                        "error",
                        stem,
                        f"{entry['file']}: a committed part the catalog lists is missing, so every read and resume of "
                        "these results raises",
                    )
                )
                continue
            data = path.read_bytes()
            if len(data) != entry.get("bytes", len(data)) or hashlib.sha256(data).hexdigest() != entry.get("sha256"):
                found.append(Finding("error", stem, f"{entry['file']}: its bytes aren't those its catalog line hashed"))
                continue
            try:
                tables.append(_read_part(directory, entry, None))
            except (ValueError, OSError) as exc:
                found.append(Finding("error", stem, f"{entry['file']}: {exc}"))
        listed = {entry["file"] for entry in live}
        stale = set(catalog.superseded())
        for name in _files_on_disk(directory):
            if name in listed:
                continue
            if name in stale:
                found.append(
                    Finding(
                        "note",
                        stem,
                        f"{name}: superseded by a compaction, still on disk (`epf results compact` removes it)",
                    )
                )
            else:
                found.append(
                    Finding(
                        "note",
                        stem,
                        f"{name}: an orphan the catalog doesn't list, never read (`epf results compact` removes it)",
                    )
                )
        if tables:
            found.extend(_duplicate_findings(stem, directory, pa.concat_tables(tables)))
    return found


def _iter_jsonl(path: Path) -> Iterator[Dict]:
    from .store import _parse_lines

    for _, _, _, record in _parse_lines(path):
        yield record


__all__ = [
    "CATALOG",
    "ParquetResultStore",
    "compact",
    "convert",
    "is_parquet_path",
    "read_catalog",
    "read_records",
    "verify",
]
