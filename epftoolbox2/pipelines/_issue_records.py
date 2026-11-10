"""What a live revision keeps of its issue (step 8f), as functions over a frame and records: what `forecast()` (8e) writes into
`issues/<issue>/r<rev>/` (`results.live.LiveRevision.directory`) before it commits the revision.

- `snapshot(frame, models, plan, issue)`: the input snapshot, every column any model read on the issue rows of D as float64 with
  each key's list hash, and its hash (`snapshot_hash`), the `snapshot` every record of the revision carries;
  `write_snapshot` writes it as `inputs.parquet`, `read_snapshot` reads it back and checks the hash.
- `quality(frame, refresh, rules, issue)`: `quality.json`, per source the refresh status and the last valid timestamp of each
  column, and per input column the missing cells on the issue rows, the cells its `{column}_imputed` flag marks, its last valid
  timestamp and its vintage class.
- `check(records, checks, clock_periods)`: the checks before export over the revision's member records
  (`results.checks.check_records`); `CheckOutcome.to_dict(checks)` is `checks.json`.
- `refresh_record(refresh)` (`refresh.json`) and `waits_record(polls)` (`waits.json`): a refresh record and the polls of a wait
  as plain JSON values.
- `write_json(directory, name, content)`: one of those files, written whole or not at all, and synced.

Nothing here reads a clock, a network or the models' results files: every value comes from the arguments.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..models.context import list_hash
from ..results.checks import CheckOutcome, ForecastChecks, check_records

SNAPSHOT_FILE = "inputs.parquet"
QUALITY_FILE = "quality.json"
CHECKS_FILE = "checks.json"
REFRESH_FILE = "refresh.json"
WAITS_FILE = "waits.json"
# What a snapshot's hash starts with: the version of its layout, so that another layout can never give the same hash.
_SNAPSHOT_TAG = b"epftoolbox2 snapshot 1\n"
# The keys of `inputs.parquet`'s file metadata.
_META_HASH = b"epftoolbox2.snapshot"
_META_ISSUE = b"epftoolbox2.issue"
_META_LISTS = b"epftoolbox2.lists"
_IMPUTED = "_imputed"
_SHIFT = re.compile(r"^(?P<base>.+)(?P<shift>_d[+-]\d+)$")


def _day(issue: Any) -> datetime.date:
    """The issue day D as a date: a date, or its ISO text."""
    if isinstance(issue, datetime.datetime):
        raise TypeError(f"issue is a day, not an instant: got {issue!r}; resolve it in the issue zone first")
    if isinstance(issue, datetime.date):
        return issue
    return datetime.date.fromisoformat(str(issue))


def issue_rows(frame: pd.DataFrame, issue: Any) -> np.ndarray:
    """Which rows of `frame` are dated D, the issue day, by their local date (the index's own zone, the wall clock of a naive
    index): the rows a forecast issued on D is made from, both occurrences of a repeated autumn hour among them."""
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("the frame needs a DatetimeIndex")
    wall = index.tz_localize(None) if index.tz is not None else index
    return np.asarray(wall.normalize() == pd.Timestamp(_day(issue)))


def _names(models: Sequence[Any]) -> List[str]:
    names = [model if isinstance(model, str) else getattr(model, "name", None) for model in models]
    if any(not isinstance(name, str) for name in names):
        raise TypeError("models are model names or models with a name")
    if len(set(names)) != len(names):
        raise ValueError(f"two models of one name among {names}")
    return names


def _lists_of(name: str, held: Any) -> List[Tuple[Optional[int], Optional[int], List[str]]]:
    """(period, horizon, names) of each key of a model's plan: a mapping (period, horizon) -> its list, or one list for every key
    (period and horizon None)."""
    if isinstance(held, Mapping):
        out = []
        for key, names in held.items():
            period, horizon = key
            out.append((int(period), int(horizon), [str(column) for column in names]))
        return sorted(out, key=lambda one: (one[1], one[0]))
    if isinstance(held, (list, tuple)) and all(isinstance(column, str) for column in held):
        return [(None, None, list(held))]
    raise TypeError(
        f"{name}: its plan is a mapping (period, horizon) -> list of columns, or one list, got {type(held).__name__}"
    )


def _float64(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """`columns` of `frame` as float64, each column's NaNs one NaN and its -0.0 a 0.0, so equal values have equal bytes."""
    out = {}
    for column in columns:
        try:
            values = frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"column {column!r} doesn't read as float64: {exc}") from None
        values = np.where(np.isnan(values), np.nan, values) + 0.0
        out[column] = values
    return pd.DataFrame(out, index=frame.index, columns=list(columns))


def snapshot_hash(inputs: pd.DataFrame) -> str:
    """The sha256 of a snapshot's inputs: its rows' instants (UTC nanoseconds; the wall clock of a naive index), then each column
    in sorted name order, its name and its float64 bytes (little-endian), each part preceded by its length. So the same values
    give the same hash whatever the order of the frame's columns, one changed cell another, and it never depends on how a file
    stores them (Parquet bytes differ across pyarrow versions). NaNs hash as one NaN and -0.0 as 0.0."""
    index = inputs.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("a snapshot's inputs need a DatetimeIndex")
    order = np.argsort(index.asi8, kind="stable")
    columns = sorted(str(column) for column in inputs.columns)
    if len(set(columns)) != len(columns):
        raise ValueError("two columns of one name in a snapshot")
    values = _float64(inputs.set_axis([str(column) for column in inputs.columns], axis=1), columns)
    digest = hashlib.sha256(_SNAPSHOT_TAG)
    instants = (index.tz_convert("UTC") if index.tz is not None else index).as_unit("ns").asi8[order]
    digest.update(len(instants).to_bytes(8, "little"))
    digest.update(instants.astype("<i8").tobytes())
    digest.update(len(columns).to_bytes(8, "little"))
    for column in columns:
        name = column.encode("utf-8")
        digest.update(len(name).to_bytes(8, "little") + name)
        digest.update(np.ascontiguousarray(values[column].to_numpy()[order], dtype="<f8").tobytes())
    return digest.hexdigest()


class Snapshot(NamedTuple):
    """An issue's inputs: the issue day, the issue rows of every column a model read (sorted by name, float64, in time order),
    each key's list hash ({model, period, horizon, list_hash}; period and horizon None for a model with one list) and `hash`."""

    issue: str
    inputs: pd.DataFrame
    lists: List[Dict[str, Any]]
    hash: str


def snapshot(frame: pd.DataFrame, models: Sequence[Any], plan: Mapping[str, Any], issue: Any) -> Snapshot:
    """The input snapshot of issue day `issue` (a date or its ISO text).

    `models` are the revision's models (names, or models with a `name`), `plan` each one's lists as the issue planned them:
    model name -> {(period, horizon): its columns}, or one list of columns for every key. The inputs are every column of any of
    those lists on the issue rows of D (`issue_rows`), as float64; the frame's other columns and rows aren't in it. A model
    without a plan, a column the frame lacks or one that doesn't read as float64 raises."""
    names = _names(models)
    lacking = [name for name in names if name not in plan]
    if lacking:
        raise KeyError(f"no plan for {lacking}: each model's lists on the issue are what its snapshot holds")
    lists, read = [], set()
    for name in names:
        for period, horizon, columns in _lists_of(name, plan[name]):
            read.update(columns)
            lists.append({"model": name, "period": period, "horizon": horizon, "list_hash": list_hash(columns)})
    missing = sorted(read - set(map(str, frame.columns)))
    if missing:
        raise KeyError(f"the frame has no column {missing}")
    rows = frame.loc[issue_rows(frame, issue)]
    rows = rows.iloc[np.argsort(rows.index.asi8, kind="stable")]
    inputs = _float64(rows, sorted(read))
    return Snapshot(_day(issue).isoformat(), inputs, lists, snapshot_hash(inputs))


def write_snapshot(snap: Snapshot, directory: Union[str, Path]) -> Path:
    """`inputs.parquet` in `directory`: the inputs (index and columns) with the issue day, the lists and the hash in the file's
    metadata, written to a temporary name, synced and renamed, so the file is whole or absent."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = Path(directory)
    table = pa.Table.from_pandas(snap.inputs, preserve_index=True)
    meta = dict(table.schema.metadata or {})
    meta.update(
        {
            _META_HASH: snap.hash.encode(),
            _META_ISSUE: snap.issue.encode(),
            _META_LISTS: json.dumps(snap.lists).encode(),
        }
    )
    table = table.replace_schema_metadata(meta)
    path = directory / SNAPSHOT_FILE
    temporary = directory / f".{SNAPSHOT_FILE}.tmp"
    pq.write_table(table, temporary, compression="zstd")
    _sync(temporary)
    os.replace(temporary, path)
    _sync_directory(directory)
    return path


def read_snapshot(path: Union[str, Path]) -> Snapshot:
    """A snapshot `write_snapshot` wrote, its hash computed again from the values read: a file whose values don't give the hash
    its metadata records raises `ValueError`."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    if _META_HASH not in meta:
        raise ValueError(f"{path}: no epftoolbox2 snapshot (no {_META_HASH.decode()} in its metadata)")
    inputs = table.to_pandas()
    recorded = meta[_META_HASH].decode()
    found = snapshot_hash(inputs)
    if found != recorded:
        raise ValueError(f"{path}: its values hash to {found}, its metadata records {recorded}")
    return Snapshot(meta[_META_ISSUE].decode(), inputs, json.loads(meta[_META_LISTS]), recorded)


# -- quality ------------------------------------------------------------------------------------------------------------------


def _field(held: Any, name: str) -> Any:
    if isinstance(held, Mapping):
        return held.get(name)
    return getattr(held, name, None)


def _iso(value: Any) -> Any:
    """An instant as ISO text, None and NaT as None; anything else as it is."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date, np.datetime64)):
        stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else stamp.isoformat()
    return value


def _sources(refresh: Any) -> Optional[Dict[str, Dict[str, Any]]]:
    """Per source of a refresh record its status and the last valid timestamp of each column. The record is 8b's
    `RefreshRecord` or a mapping of the same shape, of which only `sources` is read: source key -> an object or mapping with
    `status` ("ok", "failed", "timeout", "skipped") and `last_valid` (column -> the last timestamp with a value, or None)."""
    if refresh is None:
        return None
    sources = _field(refresh, "sources")
    if not isinstance(sources, Mapping):
        raise TypeError(
            "refresh is a RefreshRecord or a mapping with `sources`: source key -> {status, last_valid: {column: timestamp}}"
        )
    out = {}
    for key, source in sources.items():
        last_valid = _field(source, "last_valid") or {}
        out[str(key)] = {
            "status": _field(source, "status"),
            "last_valid": {str(column): _iso(stamp) for column, stamp in dict(last_valid).items()},
        }
    return out


def imputed_flag(frame: pd.DataFrame, column: str) -> Optional[str]:
    """The column flagging `column`'s imputed cells (`ImputeTransformer(flag=True)`): `{column}_imputed`, or for a shifted column
    `x_d+1` the flag shifted with it, `x_imputed_d+1`; None if the frame has neither."""
    if f"{column}{_IMPUTED}" in frame.columns:
        return f"{column}{_IMPUTED}"
    shifted = _SHIFT.match(column)
    if shifted is not None and f"{shifted['base']}{_IMPUTED}{shifted['shift']}" in frame.columns:
        return f"{shifted['base']}{_IMPUTED}{shifted['shift']}"
    return None


def _step_minutes(index: pd.DatetimeIndex) -> int:
    """The spacing of a frame's rows in minutes, from its UTC instants: 60 for hourly and daily frames, as the rules read them."""
    if len(index) < 2:
        return 60
    step = int(np.median(np.diff(index.asi8)) // 60_000_000_000)
    return 15 if step == 15 else 60


def _vintage(rules: Any, column: str, step: int) -> Optional[str]:
    if rules is None:
        return None
    from ..information_set import vintage_of

    return vintage_of(rules._resolve_for(column, step).rule)


def quality(
    frame: pd.DataFrame,
    refresh: Any,
    rules: Any,
    issue: Any,
    *,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """`quality.json` of issue day `issue`: what is known of the inputs' quality, read off the frame and the refresh record.

    - `sources`: per source of `refresh` (8b's `RefreshRecord` or a mapping of its shape; see `_sources` for the two fields read)
      its status and each column's last valid timestamp; None without a refresh record (a frame used as given).
    - `columns`: per input column (`columns`, else every column of the frame but the `_imputed` flags) `issue_rows` (rows dated
      D), `missing` (of those, cells without a value), `imputed` (of those, cells its flag marks; None without a flag column)
      and `flag` (that column), `last_valid` (the frame's last timestamp with a value, None for none) and `vintage` (its rule's
      class in `rules`, an `InformationSet`; None without one).

    A cell the frame holds is never judged: no value is called implausible or turned into a missing one (validators stay
    advisory, step 8's decision 17)."""
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("the frame needs a DatetimeIndex")
    if columns is None:
        columns = [str(column) for column in frame.columns if not str(column).endswith(_IMPUTED)]
    lacking = [column for column in columns if column not in frame.columns]
    if lacking:
        raise KeyError(f"the frame has no column {lacking}")
    rows = issue_rows(frame, issue)
    step = _step_minutes(index)
    out: Dict[str, Dict[str, Any]] = {}
    for column in columns:
        values = frame[column]
        present = values.notna().to_numpy()
        flag = imputed_flag(frame, column)
        flagged = None
        if flag is not None:
            marks = frame[flag].to_numpy(dtype=np.float64, na_value=0.0)[rows]
            flagged = int(np.count_nonzero(marks != 0))
        last = index[present][-1] if present.any() else None
        out[column] = {
            "issue_rows": int(rows.sum()),
            "missing": int(np.count_nonzero(~present[rows])),
            "imputed": flagged,
            "flag": flag,
            "last_valid": _iso(last),
            "vintage": _vintage(rules, column, step),
        }
    return {"issue": _day(issue).isoformat(), "sources": _sources(refresh), "columns": out}


# -- checks, refresh and waits --------------------------------------------------------------------------------------------------


def check(
    records: Mapping[str, Iterable[Mapping[str, Any]]],
    checks: ForecastChecks,
    clock_periods: Callable[[Any, int], Sequence[int]],
    **kwargs: Any,
) -> CheckOutcome:
    """The checks before export over a revision's member records: `results.checks.check_records` (which see), so a failing
    member-day's forecasts are `check_failed` skip records before anything of the revision is written, at its revision."""
    return check_records(records, checks, clock_periods, **kwargs)


def jsonable(value: Any) -> Any:
    """`value` as plain JSON values: mappings, sequences, dataclasses and objects with `to_dict()` in their parts, instants as ISO
    text, numpy scalars as numbers, NaN and infinities as None (JSON has neither)."""
    if hasattr(value, "to_dict") and callable(value.to_dict) and not isinstance(value, (pd.DataFrame, pd.Series)):
        return jsonable(value.to_dict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable({f.name: getattr(value, f.name) for f in dataclasses.fields(value)})
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date, np.datetime64)):
        return _iso(value)
    if isinstance(value, (pd.Timedelta, datetime.timedelta)):
        return pd.Timedelta(value).total_seconds()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        return [jsonable(item) for item in (sorted(value, key=str) if isinstance(value, (set, frozenset)) else value)]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def refresh_record(refresh: Any) -> Optional[Dict[str, Any]]:
    """`refresh.json`: a refresh record (8b's `RefreshRecord`, a dataclass or a mapping) as plain JSON values, None without one.
    Its errors are as the record holds them, redacted where it was made."""
    return None if refresh is None else jsonable(refresh)


def waits_record(polls: Iterable[Any]) -> Dict[str, Any]:
    """`waits.json`: every poll of an issue's wait (`forecast(wait_for=...)`), in the order made, as plain JSON values."""
    return {"polls": [jsonable(poll) for poll in polls]}


def _sync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sync_directory(directory: Path) -> None:
    if os.name == "nt":  # pragma: no cover (a directory can't be opened for fsync there)
        return
    _sync(directory)


def write_json(directory: Union[str, Path], name: str, content: Any) -> Path:
    """`content` as JSON in `directory/name`, written to a temporary name, synced and renamed: whole or absent, and on disk
    before the revision's commit line is."""
    directory = Path(directory)
    path = directory / name
    temporary = directory / f".{name}.tmp"
    text = json.dumps(jsonable(content), indent=2, allow_nan=False) + "\n"
    with open(temporary, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    _sync_directory(directory)
    return path
