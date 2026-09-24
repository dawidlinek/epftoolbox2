"""Helpers shared by the validators."""

from fnmatch import fnmatchcase
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from epftoolbox2._date_utils import resolve_date

MAX_LISTED = 10  # individual messages per check before the rest are summarised in one line
ONE_DAY = pd.Timedelta(days=1)


def to_step(freq: str) -> pd.Timedelta:
    """Fixed step for a frequency string such as "15min", "1h", "h" or "1D"."""
    try:
        step = pd.Timedelta(freq)
    except ValueError:
        try:
            step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"freq must be a fixed step such as '15min', '1h' or '1D', got {freq!r}") from exc
    if step <= pd.Timedelta(0):
        raise ValueError(f"freq must be positive, got {freq!r}")
    return step


def to_duration(value: str, name: str) -> pd.Timedelta:
    """Non-negative duration such as "0h", "3h" or "2D"."""
    try:
        duration = pd.Timedelta(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a duration such as '3h' or '2D', got {value!r}") from exc
    if duration < pd.Timedelta(0):
        raise ValueError(f"{name} must not be negative, got {value!r}")
    return duration


def format_step(step: pd.Timedelta) -> str:
    """Compact frequency string: 15min, 1h, 26h, 1D."""
    seconds = step.total_seconds()
    for unit_seconds, unit in ((86400, "D"), (3600, "h"), (60, "min"), (1, "s")):
        if seconds % unit_seconds == 0:
            return f"{int(seconds // unit_seconds)}{unit}"
    return str(step)


def infer_step(index: pd.DatetimeIndex) -> Optional[pd.Timedelta]:
    """Most common step between consecutive distinct timestamps (the smallest on ties); None below two timestamps.

    Uses elapsed time, so a tz-aware local index has 1h steps across DST switches.
    """
    values = index.unique().sort_values()
    if len(values) < 2:
        return None
    return pd.Series(values[1:] - values[:-1]).mode().min()


def as_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC version of an index; a naive index is taken to be UTC (the source contract)."""
    return index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")


def parse_timestamp(value: Any) -> Optional[pd.Timestamp]:
    """Timestamp from a string or Timestamp, like `DataPipeline.run` parses dates.

    "today" / "today_d-1" keywords resolve to a date; naive values are UTC.
    """
    if value is None:
        return None
    ts = pd.Timestamp(resolve_date(value))
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts


def display_time(ts: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Show a UTC timestamp in the index's own time zone (naive for a naive index)."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(index.tz) if index.tz is not None else ts.tz_convert("UTC").tz_localize(None)


def true_runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Start positions and lengths of the runs of consecutive True values."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    return starts, np.flatnonzero(edges == -1) - starts


def longest_run(mask: np.ndarray) -> Tuple[int, int]:
    """(length, start position) of the longest run of True values; (0, -1) when there is none."""
    starts, lengths = true_runs(mask)
    if len(lengths) == 0:
        return 0, -1
    i = int(np.argmax(lengths))
    return int(lengths[i]), int(starts[i])


def select_columns(df: pd.DataFrame, columns: Optional[Sequence[str]]) -> Tuple[List[str], List[str]]:
    """(present, missing) columns in order without repeats; None selects every column."""
    wanted = list(dict.fromkeys(df.columns if columns is None else columns))
    return [c for c in wanted if c in df.columns], [c for c in wanted if c not in df.columns]


def matching_columns(columns: Sequence[str], patterns: Sequence[str]) -> List[str]:
    """Columns matching any of the glob patterns ("price*"), in frame order."""
    return [c for c in columns if any(fnmatchcase(str(c), p) for p in patterns)]


def column_values(df: pd.DataFrame, column: str) -> pd.Series:
    """The column as one Series; with duplicated column names the first one is used."""
    values = df[column]
    return values.iloc[:, 0] if isinstance(values, pd.DataFrame) else values


def add_messages(target: List[str], messages: List[str], what: str) -> None:
    """Append up to MAX_LISTED messages and one line counting the rest."""
    target.extend(messages[:MAX_LISTED])
    if len(messages) > MAX_LISTED:
        target.append(f"... and {len(messages) - MAX_LISTED} more {what}")
