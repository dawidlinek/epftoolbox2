"""CSV files for pipeline data that keep the local time zone (used by `epf data`, `epf model` and the TUI).

A CSV stores timestamps with their UTC offset (`2024-03-31 03:00:00+02:00`) but not the zone name, and the
model's days and hours are local: reading such a file back as UTC shifts every forecast key. The writer
therefore names the zone in the index header, e.g. `datetime[Europe/Warsaw]`, and the reader converts back
to it. Files written before this (offsets but no zone name) need the zone passed explicitly.
"""

import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd

_HEADER = re.compile(r"^(?P<name>.*)\[(?P<tz>[^\[\]]+)\]$")
_OFFSET = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
_UTC_OFFSET = re.compile(r"(?:Z|[+-]00:?00)$")


def save_frame_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Write `df` to CSV; a time-zone-aware index keeps its zone name in the header."""
    out = df.copy(deep=False)
    name = df.index.name or "datetime"
    tz = getattr(df.index, "tz", None)
    new_name = f"{name}[{tz}]" if tz is not None else name
    # `out.index.name = ...` would mutate the Index object in place, and a shallow copy shares it with `df`
    # (`copy(deep=False)` copies the DataFrame wrapper, not its index), renaming the caller's own index as a
    # side effect. `rename` returns a new Index instead, leaving `df.index` untouched.
    out.index = out.index.rename(new_name)
    out.to_csv(path)


def load_frame_csv(path: Union[str, Path], tz: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV written by `save_frame_csv` (or `DataFrame.to_csv`) with its local time zone restored.

    `tz` overrides the zone named in the file. A file whose timestamps carry non-zero UTC offsets but no
    zone name raises unless `tz` is given, because reading it as UTC would move local days and hours.
    Floats are parsed exactly (`float_precision="round_trip"`).
    """
    df = pd.read_csv(path, index_col=0, float_precision="round_trip")
    header = str(df.index.name) if df.index.name is not None else ""
    match = _HEADER.match(header)
    zone = tz or (match["tz"] if match else None)
    name = match["name"] if match else (df.index.name or None)
    if zone is not None:
        _check_zone(zone)

    raw = df.index.astype(str)
    if len(raw) and raw.str.contains(_OFFSET).all():
        index = pd.to_datetime(raw, utc=True)
        if zone is not None:
            index = index.tz_convert(zone)
        elif not raw.str.contains(_UTC_OFFSET).all():
            raise ValueError(
                f"{path}: timestamps carry UTC offsets (e.g. '{raw[0]}') but the file doesn't name its time zone. "
                "Pass the zone the data was in (e.g. --tz Europe/Warsaw): reading it as UTC would shift the local "
                "days and hours the model forecasts."
            )
    else:
        index = pd.to_datetime(raw)
        if zone is not None:
            index = index.tz_localize(zone, ambiguous="infer")
    df.index = pd.DatetimeIndex(index, name=name)
    return df


def _check_zone(zone: str) -> None:
    try:
        pd.Timestamp("2000-01-01", tz=zone)
    except Exception as exc:
        raise ValueError(f"Unknown time zone '{zone}'") from exc
