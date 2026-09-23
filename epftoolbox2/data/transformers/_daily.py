"""Local-day helpers shared by the daily-feature transformers."""

import warnings

import numpy as np
import pandas as pd

from ..._date_utils import is_utc_index as _is_utc


def parse_step(freq: str) -> pd.Timedelta:
    step = pd.Timedelta(freq)
    if step <= pd.Timedelta(0) or pd.Timedelta(days=1) % step != pd.Timedelta(0):
        raise ValueError(f"freq must divide a day into whole slots, got {freq!r}")
    return step


def as_column_list(columns) -> list:
    columns = [columns] if isinstance(columns, str) else list(columns)
    if not columns:
        raise ValueError("At least one column must be provided")
    return columns


def slot_names(column: str, step: pd.Timedelta) -> list:
    """`price_h00`..`price_h23` for hourly slots, `price_s00`..`price_s95` for 15-minute slots."""
    n = pd.Timedelta(days=1) // step
    letter = "h" if step == pd.Timedelta(hours=1) else "s"
    width = max(2, len(str(n - 1)))
    return [f"{column}_{letter}{k:0{width}d}" for k in range(n)]


def check_index(df: pd.DataFrame, who: str) -> pd.DatetimeIndex:
    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{who} needs a DatetimeIndex")
    if _is_utc(index):
        warnings.warn(
            f"{who}: the index is UTC, so days are UTC days, not local market days. Run TimezoneTransformer first.",
            UserWarning,
            stacklevel=3,
        )
    return index


def local_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Naive midnight of each row's local date."""
    wall = index.tz_localize(None) if index.tz is not None else index
    return wall.normalize()


def slot_of_day(index: pd.DatetimeIndex, step: pd.Timedelta, freq: str) -> np.ndarray:
    """Wall-clock slot number of each row (0..23 for hourly data)."""
    wall = index.tz_localize(None) if index.tz is not None else index
    offsets = wall - wall.normalize()
    if (offsets % step != pd.Timedelta(0)).any():
        raise ValueError(f"Timestamps are not on the {freq} grid; resample to {freq} first")
    return np.asarray(offsets // step, dtype=int)


def slots_in_day(days: pd.DatetimeIndex, tz, step: pd.Timedelta) -> np.ndarray:
    """How many wall-clock slots each local date has (23 or 25 hours on DST switch days)."""
    if tz is None:
        return np.full(len(days), pd.Timedelta(days=1) // step)
    first = np.ones(len(days), dtype=bool)
    start = days.tz_localize(tz, ambiguous=first, nonexistent="shift_forward")
    end = (days + pd.Timedelta(days=1)).tz_localize(tz, ambiguous=first, nonexistent="shift_forward")
    return np.asarray((end - start) // step, dtype=int)


def check_new_columns(df: pd.DataFrame, names: list, who: str) -> None:
    clashes = sorted({name for name in names if name in df.columns})
    if clashes:
        raise ValueError(f"{who} would create columns that already exist: {clashes[:5]}")
