"""The data contract a model run checks before any fit.

Every check here replaces a failure mode that used to produce wrong or missing forecasts without an error:
data at a finer resolution than `freq` (look-ahead through the training cut), columns that collide with
the model's own target or id columns, predictors that don't exist, and non-numeric predictors.
"""

import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

# Columns the model adds to its working frame; data columns with these names would be shadowed.
RESERVED_COLUMNS = ("hour", "day")

_FREQ_STEPS = {"1h": pd.Timedelta(hours=1), "15min": pd.Timedelta(minutes=15)}


def period_index(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Period of the (local) day each row belongs to: the hour for "1h", the quarter-hour for "15min"."""
    if freq == "1h":
        return index.hour.to_numpy()
    if freq == "15min":
        return (index.hour * 4 + index.minute // 15).to_numpy()
    raise ValueError(f"Unsupported freq '{freq}'. Supported: {list(_FREQ_STEPS)}")


def wall_clock(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Local wall-clock times as a naive index in nanoseconds."""
    return (index.tz_localize(None) if index.tz is not None else index).as_unit("ns")


def target_columns(target: str, horizon: int) -> List[str]:
    return [f"{target}_d+{h}" for h in range(1, horizon + 1)]


def check_index(data: pd.DataFrame, freq: str) -> None:
    """Raise unless the index is a DatetimeIndex with one row per (local day, period) at resolution `freq`.

    The only allowed repeat is the hour that occurs twice when clocks fall back (same wall-clock time,
    different UTC offset).
    """
    if freq not in _FREQ_STEPS:
        raise ValueError(f"Unsupported freq '{freq}'. Supported: {list(_FREQ_STEPS)}")
    index = data.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"Data must have a DatetimeIndex, got {type(index).__name__}")

    duplicated = index[index.duplicated()]
    if len(duplicated):
        raise ValueError(
            f"Data index has {len(duplicated)} duplicate timestamp(s), first {duplicated[0]}. "
            "Each timestamp must appear once; drop or aggregate the duplicates before running the model."
        )
    if len(index) < 2:
        return

    step = _FREQ_STEPS[freq]
    diffs = np.diff(np.sort(index.as_unit("ns").asi8))
    values, counts = np.unique(diffs, return_counts=True)
    common = pd.Timedelta(int(values[np.argmax(counts)]), unit="ns")
    if common != step:
        raise ValueError(
            f"Data rows are most commonly {_fmt_step(common)} apart, but the model runs with freq='{freq}'. "
            f"Resample the data to {freq} (ResampleTransformer) or pass the matching freq "
            f"({_suggest_freq(common)})."
        )

    local = wall_clock(index)
    frame = pd.DataFrame({"date": local.normalize().asi8, "period": period_index(index, freq), "wall": local.asi8})
    distinct = frame.groupby(["date", "period"], sort=True)["wall"].nunique()
    crowded = distinct[distinct > 1]
    if len(crowded):
        examples = ", ".join(f"{pd.Timestamp(d).date()} period {p}" for d, p in crowded.index[:3].tolist())
        raise ValueError(
            f"{len(crowded)} (local day, period) slot(s) have more than one row for freq='{freq}' "
            f"(e.g. {examples}). The data mixes resolutions or is finer than freq; the training cut assumes "
            "one row per day and period, so extra rows would leak the issue day's targets into training. "
            f"Resample the data to {freq} first."
        )


def check_names(data: pd.DataFrame, target: str, horizon: int) -> None:
    """Raise if the data uses a name the model reserves for its own columns."""
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' is not in the data")
    if target in RESERVED_COLUMNS:
        raise ValueError(f"Target column '{target}' clashes with the model's internal '{target}' id; rename it")

    reserved = [c for c in RESERVED_COLUMNS if c in data.columns]
    if reserved:
        names = ", ".join(f"'{c}'" for c in reserved)
        raise ValueError(
            f"Data column(s) {names} clash with the model's internal period/day ids. Rename them "
            "(for example CalendarSource(prefix='cal_') gives 'cal_hour') or drop them before running the model."
        )

    pattern = re.compile(rf"^{re.escape(target)}_d\+(\d+)$")
    collisions = []
    for column in data.columns:
        match = pattern.match(column) if isinstance(column, str) else None
        if match and 1 <= int(match[1]) <= horizon:
            collisions.append(column)
    if collisions:
        raise ValueError(
            f"Data already has column(s) {collisions}. Names '{target}_d+<h>' are the model's targets "
            f"(built from '{target}' by a calendar-day shift), so existing columns would silently replace or shift "
            "them. Drop or rename them; a LagTransformer with negative lags on the target creates them."
        )

    duplicated = data.columns[data.columns.duplicated()]
    if len(duplicated):
        raise ValueError(f"Data has duplicate column names: {sorted(set(duplicated))}")


def check_predictors(
    data: pd.DataFrame,
    expanded: Dict[Tuple[int, int], List[str]],
    target: str,
    horizon: int,
) -> List[str]:
    """Raise if any expanded predictor is a model target, reserved, missing or non-numeric.

    Returns the sorted list of distinct predictor columns.
    """
    targets = set(target_columns(target, horizon))
    used = sorted({p for preds in expanded.values() for p in preds}, key=str)

    as_target = [p for p in used if p in targets]
    if as_target:
        raise ValueError(
            f"Predictor(s) {as_target} name the model's own targets (the future value of '{target}'); "
            "using them as inputs is look-ahead."
        )
    reserved = [p for p in used if p in RESERVED_COLUMNS]
    if reserved:
        raise ValueError(f"Predictor(s) {reserved} are reserved names for the model's internal ids")

    missing = [p for p in used if p not in data.columns]
    if missing:
        where = _where(expanded, missing)
        raise ValueError(
            f"Predictor column(s) not in the data: {missing} (needed for {where}). "
            f"Available columns: {_preview(data.columns)}"
        )

    non_numeric = [p for p in used if not _is_numeric(data[p])]
    if not _is_numeric(data[target]):
        non_numeric.append(target)
    if non_numeric:
        dtypes = ", ".join(f"'{c}' ({data[c].dtype})" for c in non_numeric)
        raise ValueError(f"Predictors and the target must be numeric: {dtypes}")
    return used


def missing_test_rows(index: pd.DatetimeIndex, freq: str, days: Iterable) -> List[Tuple[int, pd.Timestamp]]:
    """(period, local day) slots on the given days that have no row although that wall-clock time exists.

    Slots that don't exist because clocks spring forward are not listed: their handling is a known
    issue pinned in tests/test_known_issues.py.
    """
    local = wall_clock(index)
    present = set(zip(local.normalize().asi8.tolist(), period_index(index, freq).tolist()))
    step = _FREQ_STEPS[freq]
    periods = int(pd.Timedelta(days=1) / step)
    absent = []
    for day in days:
        day = pd.Timestamp(day).normalize()
        absent.extend((p, day) for p in range(periods) if (day.value, p) not in present)
    if not absent or index.tz is None:
        return absent
    stamps = pd.DatetimeIndex([day + p * step for p, day in absent])
    localized = stamps.tz_localize(index.tz, nonexistent="NaT", ambiguous=np.ones(len(stamps), dtype=bool))
    return [slot for slot, exists in zip(absent, ~localized.isna()) if exists]


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)


def _where(expanded: Dict[Tuple[int, int], List[str]], columns: Sequence[str]) -> str:
    keys = sorted(k for k, preds in expanded.items() if any(c in preds for c in columns))
    horizons = sorted({hz for _, hz in keys})
    periods = sorted({p for p, _ in keys})
    return f"horizon(s) {horizons}, period(s) {_ranges(periods)}"


def _ranges(values: Sequence[int]) -> str:
    if not values:
        return "[]"
    parts, start, prev = [], values[0], values[0]
    for v in list(values[1:]) + [None]:
        if v is not None and v == prev + 1:
            prev = v
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        if v is not None:
            start = prev = v
    return ", ".join(parts)


def _preview(columns: Sequence[str], limit: int = 20) -> str:
    cols = [str(c) for c in columns]
    return ", ".join(cols[:limit]) + (f", ... ({len(cols)} in total)" if len(cols) > limit else "")


def _fmt_step(step: pd.Timedelta) -> str:
    seconds = int(step.total_seconds())
    for unit, size in (("day", 86400), ("h", 3600), ("min", 60)):
        if seconds % size == 0 and seconds >= size:
            n = seconds // size
            return f"{n} {unit}{'s' if unit == 'day' and n > 1 else ''}"
    return str(step)


def _suggest_freq(step: pd.Timedelta) -> str:
    for name, value in _FREQ_STEPS.items():
        if value == step:
            return f"freq='{name}'"
    return f"supported: {list(_FREQ_STEPS)}"
