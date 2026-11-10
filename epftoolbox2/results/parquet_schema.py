"""The fixed schema of a Parquet results part (`results.parquet_store`, D4 of the forecasting plan): what a JSONL record is as a
row, and back, exactly.

One column per record key a model writes, in the order a forecast's record has them (`COLUMNS`): dates `date32`; `hour`,
`horizon`, `revision` `int16`; `run_weekday` `int8`; `day_in_test`, `training_days` `int32`; `prediction`, `actual` `float64`;
`coefficients`, `member_predictions` `list<float64>`; `freq`, `window`, `status`, `mode` dictionary strings; hashes, `detail`,
`issued_at` and `snapshot` strings. Every column is nullable. A live revision's three keys (step 8c) are columns of a part with a
live record alone, so a backtest's parts are written as before them; a part without them reads with them null. A record's keys and their order are one more column, `_keys` (a dictionary string, so a model's few
layouts cost next to nothing), because a JSONL record says two things a column of values can't: which keys it has (a skip record
has `prediction: null`, a forecast of a model without `refit` has no `newest_target_day`) and in what order (a refit record has
`refit_anchor` after `freq`, any other after `list_hash`). With it a record converts both ways exactly: its keys in its order, a
JSON null where it had one and no key where it had none; floats keep their bits both ways, since JSON writes `repr` and Parquet
stores the double. A key outside `COLUMNS` raises, so nothing is dropped on the way.

pyarrow is imported inside the functions only, so `import epftoolbox2` doesn't load `pyarrow.parquet`.
"""

from operator import itemgetter
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

# (name, kind) of every record key, in the order of a forecast's record (`_worker._fit_one_numpy`, `_fit_unit`, `skip_record`,
# `ensembles._compute`). `refit_anchor` stands where a refit record has it; the `_keys` column holds each record's own order.
COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("run_date", "date"),
    ("target_date", "date"),
    ("run_weekday", "int8"),
    ("hour", "int16"),
    ("horizon", "int16"),
    ("day_in_test", "int32"),
    ("prediction", "float64"),
    ("member_predictions", "floats"),
    ("actual", "float64"),
    ("coefficients", "floats"),
    ("freq", "category"),
    ("refit_anchor", "date"),
    ("newest_target_day", "date"),
    ("predictor_list", "string"),
    ("training_days", "int32"),
    ("matching_days", "string"),
    ("window", "category"),
    ("status", "category"),
    ("list_hash", "string"),
    ("revision", "int16"),
    ("detail", "string"),
)
# A live revision's records (step 8c, `results.live`) add three: when it was issued (ISO UTC), "live", and its inputs' snapshot
# hash. Only a part with such a record has these columns, so a backtest's parts are what they were before them; a part without
# them reads as if they were null (`parquet_store._conform`).
LIVE_COLUMNS: Tuple[Tuple[str, str], ...] = (("issued_at", "string"), ("mode", "category"), ("snapshot", "string"))
BASE_COLUMNS = COLUMNS
COLUMNS = (*BASE_COLUMNS, *LIVE_COLUMNS)
LIVE_NAMES = frozenset(name for name, _ in LIVE_COLUMNS)
NAMES: Tuple[str, ...] = tuple(name for name, _ in COLUMNS)
KINDS: Dict[str, str] = dict(COLUMNS)
KEYS_COLUMN = "_keys"  # each record's keys, in its order, joined by ","
DATE_COLUMNS = tuple(name for name, kind in COLUMNS if kind == "date")
# The order rows are sorted in within a part (stably, so records of one key keep the order they were written in).
SORT_COLUMNS = ("run_date", "hour", "horizon")

_KNOWN = frozenset(NAMES)


def _pa():
    import pyarrow

    return pyarrow


def arrow_type(kind: str):
    pa = _pa()
    return {
        "date": pa.date32(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "float64": pa.float64(),
        "floats": pa.list_(pa.float64()),
        "category": pa.dictionary(pa.int32(), pa.string()),
        "string": pa.string(),
    }[kind]


def arrow_schema(live: bool = True):
    """The schema of a part: `COLUMNS` and `_keys`, all nullable; without `live`, the columns of a part with no live record
    (`BASE_COLUMNS`), every part's before step 8c."""
    pa = _pa()
    return pa.schema(
        [pa.field(name, arrow_type(kind)) for name, kind in (COLUMNS if live else BASE_COLUMNS)]
        + [pa.field(KEYS_COLUMN, pa.dictionary(pa.int32(), pa.string()))]
    )


def check_keys(keys: Sequence[str], where: str) -> None:
    """Raise ValueError for a record key the schema has no column for: it would be lost on the way to Parquet."""
    unknown = [key for key in keys if key not in _KNOWN]
    if unknown:
        raise ValueError(
            f"{where}: a record has keys the Parquet results schema has no column for ({', '.join(map(repr, unknown))}), so "
            "they would be lost. Write these results as JSONL (save_format='jsonl'), or add the keys to "
            "epftoolbox2.results.parquet_schema.COLUMNS."
        )


def to_table(
    records: Sequence[Dict[str, Any]], where: str = "results", keys_of: Optional[Sequence[Tuple[str, ...]]] = None
):
    """The records as one table of the schema, in their order. Raises ValueError for a key the schema lacks, a value its
    column can't hold (such as a string in `hour`), and a date that isn't written as `date.isoformat()` writes it, since such
    a record wouldn't come back as it was. `keys_of` are the records' keys (`tuple(record)` each) where they are known."""
    import numpy as np

    pa = _pa()
    n = len(records)
    layouts: Dict[Tuple[str, ...], List[int]] = {}
    last: Tuple[str, ...] = ()
    rows: List[int] = []
    for i, record in enumerate(records):
        keys = tuple(record) if keys_of is None else keys_of[i]
        if keys is not last and keys != last:  # a model's records share one or two layouts: look one up when it changes
            rows = layouts.get(keys)
            if rows is None:
                check_keys(keys, where)
                rows = layouts[keys] = []
            last = keys
        rows.append(i)
    # Each layout's values, a record's at a time (`itemgetter`), then per column; scattered by row where layouts mix.
    values: Dict[str, Any] = {}
    if len(layouts) == 1:
        (keys,) = layouts
        if n:
            for name, column in zip(
                keys, zip(*map(itemgetter(*keys), records)) if len(keys) > 1 else [[r[keys[0]] for r in records]]
            ):
                values[name] = list(column)  # a list: pyarrow converts a tuple of numbers 100 times slower
    else:
        for keys, at in layouts.items():
            group = [records[i] for i in at]
            columns = zip(*map(itemgetter(*keys), group)) if len(keys) > 1 else [[r[keys[0]] for r in group]]
            index = np.asarray(at, dtype=np.int64)
            for name, column in zip(keys, columns):
                held = values.get(name)
                if held is None:
                    held = values[name] = np.full(n, None, dtype=object)
                held[index] = np.fromiter(column, dtype=object, count=len(at))
    arrays = []
    dates: List[
        Tuple[Any, Any]
    ] = []  # a date column equal to one before it (refit_anchor is run_date without refit) reuses it
    live = any(name in LIVE_NAMES for keys in layouts for name in keys)
    for name, kind in COLUMNS if live else BASE_COLUMNS:
        column = values.get(name)
        if column is None:
            arrays.append(pa.nulls(n, arrow_type(kind)))
            continue
        made = next((array for earlier, array in dates if kind == "date" and _same(earlier, column)), None)
        if made is None:
            made = _array(column, name, kind, where)
            if kind == "date":
                dates.append((column, made))
        arrays.append(made)
    names = [",".join(keys) for keys in layouts]
    layout_of = np.zeros(n, dtype=np.int32)
    for number, at in enumerate(layouts.values()):
        layout_of[np.asarray(at, dtype=np.int64)] = number
    arrays.append(
        pa.DictionaryArray.from_arrays(pa.array(layout_of, pa.int32()), pa.array(names, pa.string()))
        if n
        else pa.array([], pa.dictionary(pa.int32(), pa.string()))
    )
    return pa.Table.from_arrays(arrays, schema=arrow_schema(live))


def _same(a: Any, b: Any) -> bool:
    """Whether two columns of values are equal value for value (lists, or object arrays where layouts mix)."""
    if isinstance(a, list) and isinstance(b, list):
        return a == b
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


# The types a column's values have, for the columns whose one repeated value is built at once (`_constant`).
_ONE_TYPE = {"category": str, "string": str, "int8": int, "int16": int, "int32": int}


def _constant(values: Any, kind: str):
    """The column of `values` built from its one value, where every value is that one and of the column's own type (so an int 1
    and a bool True, which a set takes as one, never are): a model's `freq`, `status`, `list_hash`, `horizon` and `revision`
    mostly are. None otherwise. Empty lists of `floats` too."""
    pa = _pa()
    n = len(values)
    if kind == "floats":
        if n and not any(values) and set(map(type, values)) == {list}:
            import numpy as np

            return pa.ListArray.from_arrays(pa.array(np.zeros(n + 1, dtype=np.int32)), pa.array([], pa.float64()))
        return None
    expected = _ONE_TYPE.get(kind)
    if expected is None or not n or type(values[0]) is not expected:
        return None
    if len(set(values)) != 1 or set(map(type, values)) != {expected}:
        return None
    if kind == "category":
        import numpy as np

        return pa.DictionaryArray.from_arrays(pa.array(np.zeros(n, dtype=np.int32)), pa.array([values[0]], pa.string()))
    return pa.repeat(pa.scalar(values[0], arrow_type(kind)), n)


def _array(values: Any, name: str, kind: str, where: str):
    pa = _pa()
    try:
        made = _constant(values, kind)
        if made is not None:
            return made
        if kind == "date":
            # A results file has a few hundred distinct dates: parse and check each once.
            text = pa.array(values, pa.string()).dictionary_encode()
            dates = text.dictionary.cast(pa.date32())
            back = dates.cast(pa.string())
            if not back.equals(text.dictionary):
                bad = next(v for v, b in zip(text.dictionary.to_pylist(), back.to_pylist()) if v != b)
                raise ValueError(f"{bad!r} is not a date as date.isoformat() writes it")
            return dates.take(text.indices)
        if kind == "category":
            return pa.array(values, pa.string()).dictionary_encode()
        return pa.array(values, arrow_type(kind))
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError, TypeError, OverflowError) as exc:
        raise ValueError(
            f"{where}: the values of {name!r} don't fit its {kind} column of the Parquet schema: {exc}"
        ) from None


def sort_indices(table):
    """The row order of a part: by `SORT_COLUMNS`, stably (a stored record has all three)."""
    import pyarrow.compute as pc

    return pc.sort_indices(table, sort_keys=[(name, "ascending") for name in SORT_COLUMNS])


def _python_columns(table, columns: Optional[Sequence[str]] = None) -> Dict[str, List[Any]]:
    """Each column of `table` (or those of `columns` it has) as Python values: dates as ISO text, dictionary strings as text."""
    pa = _pa()
    out = {}
    for name in table.column_names:
        if name == KEYS_COLUMN or (columns is not None and name not in columns):
            continue
        column = table.column(name)
        kind = KINDS.get(name)
        if kind == "date":
            column = column.cast(pa.string())
        elif kind == "category":
            column = column.cast(pa.string())
        out[name] = column.to_pylist()
    return out


def layouts_of(table) -> List[Tuple[str, ...]]:
    """Each row's keys, in its order (`_keys`)."""
    pa = _pa()
    column = table.column(KEYS_COLUMN).combine_chunks()
    if not isinstance(column, pa.DictionaryArray):
        column = column.dictionary_encode()
    parsed = [tuple(text.split(",")) if text else () for text in column.dictionary.to_pylist()]
    return [parsed[i] for i in column.indices.to_pylist()]


def iter_dicts(table, columns: Optional[Sequence[str]] = None, rows: Optional[Sequence[int]] = None) -> Iterator[Dict]:
    """The records of `table` as the dicts they were written as: each row's keys in its order (`_keys`), a null where the
    record held one. With `columns`, only those of its keys (the table may hold fewer columns than the schema). With `rows`,
    those rows, in that order."""
    if rows is not None:
        import pyarrow as pa

        table = table.take(pa.array(rows, pa.int64()))
    values = _python_columns(table, columns)
    layouts = layouts_of(table)
    cache: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
    for i, keys in enumerate(layouts):
        kept = cache.get(keys)
        if kept is None:
            kept = cache[keys] = tuple(key for key in keys if key in values)
        yield {key: values[key][i] for key in kept}
