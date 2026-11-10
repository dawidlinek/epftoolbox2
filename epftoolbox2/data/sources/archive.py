"""A `SnapshotArchive` replayed as a data source (`ArchiveSource`, step 8j): the values as they were downloaded at a stated time.

A provider's API gives today's values, revisions included, so a backtest that reads it sees forecasts as they were rewritten after
the gate. An archive that a scheduled job filled (`SnapshotArchive.record`) holds every fetch as it came. `ArchiveSource` reads it
back per delivery day: the values of the newest fetch that had returned by `known_at` on the day `days_before` before it, and the
rule `Availability.published(known_at, days_before, timezone, vintage="snapshot")` says exactly that. A backtest over the days the
archive covers then reads no value before the archive had it.
"""

from __future__ import annotations

import bisect
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..archive import ArchiveError, Fetch, SnapshotArchive, _iter_days, _utc
from .base import DataSource


class ArchiveSource(DataSource):
    """One source's fetches in a `SnapshotArchive`, as known at `known_at` on the day `days_before` before each delivery day.

    For a delivery day d, a local date of `timezone`, each cell of d (an instant whose date in `timezone` is d) takes the value
    of the newest fetch that returned at or before `known_at` on d - `days_before` (wall-clock time of `timezone`: a time the
    clocks repeat is read at its second occurrence, one they skip at the switch, as the information set reads a publication
    time) and asked for that cell. Where that fetch has no value the cell is missing, even if an older fetch had one: it is what
    a refresh at that time saw. Days before the archive's first fetch have no values.

    The rule of every column it emits is `Availability.published(known_at, days_before, timezone=timezone, vintage="snapshot")`
    (`availability()`): `InformationSet.with_source_rules` gives it (a `Workflow` applies it), and it comes before every rule,
    assumption and pattern of the information set, as the values were chosen by it; a rule of yours that names one of its
    columns raises. Outside a `Workflow` pass `information_set.with_source_rules(data_pipeline.sources)`. The default `prefix`
    keeps its column names off the built-in rules of the archived source's names (`archive_load_forecast`, not `load_forecast`).

    Rows of the archive's realised series (actual load, say) for day d are mostly not out on d - 1: replay them with
    `days_before=-1` (known the day after delivery) in another `ArchiveSource` with `columns=`.

    Args:
        path: The archive's directory (`SnapshotArchive(path)`).
        source: The archived source's key (`SnapshotArchive.keys()`), or its class name when the archive holds one key of it.
        known_at: The time of day ("HH:MM", `timezone`) the values are read at.
        days_before: Days from the delivery day back to the day they are read on: 1 (the day before), 0 (the day itself), -1
            (the day after).
        timezone: The zone of the delivery days and of `known_at`.
        columns: The archived columns to emit (their names in the archive); None, every column the source's fetches have.
        prefix: Joined to every emitted name with "_" ("" keeps the archived names).
    """

    def __init__(
        self,
        path: Any,
        source: str,
        known_at: str = "11:45",
        days_before: int = 1,
        timezone: str = "Europe/Brussels",
        columns: Optional[List[str]] = None,
        prefix: str = "archive",
    ):
        self.path = os.fspath(path) if isinstance(path, os.PathLike) else path
        self.source = source
        self.known_at = known_at
        self.days_before = days_before
        self.timezone = timezone
        self.columns = columns
        self.prefix = prefix
        self._validate_config()

    def _validate_config(self) -> bool:
        from ...information_set import _check_zone, _parse_time  # the information set imports the sources

        if not isinstance(self.path, str) or not self.path:
            raise ValueError(f"ArchiveSource: path must be the archive's directory, got {self.path!r}")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError(f"ArchiveSource: source must be a source key of the archive, got {self.source!r}")
        _parse_time(self.known_at)
        if isinstance(self.days_before, bool) or not isinstance(self.days_before, (int, np.integer)):
            raise ValueError(f"ArchiveSource: days_before must be a whole number of days, got {self.days_before!r}")
        self.days_before = int(self.days_before)
        if not isinstance(self.timezone, str) or not self.timezone:
            raise ValueError(
                f"ArchiveSource: timezone must name a zone, such as 'Europe/Brussels', got {self.timezone!r}"
            )
        _check_zone(self.timezone)
        if isinstance(self.columns, str):
            self.columns = [self.columns]
        if self.columns is not None:
            self.columns = list(self.columns)
            if not self.columns or not all(isinstance(c, str) and c for c in self.columns):
                raise ValueError(f"ArchiveSource: columns must be None or a list of column names, got {self.columns!r}")
        if not isinstance(self.prefix, str) or self.prefix.endswith("_"):
            raise ValueError(
                f"ArchiveSource: prefix is joined with '_', so it must be a string not ending in '_', got {self.prefix!r}"
            )
        return True

    # -- names and rule -----------------------------------------------------------------------------------------------------

    def _archive(self) -> SnapshotArchive:
        return SnapshotArchive(self.path)

    def _named(self, column: str) -> str:
        return f"{self.prefix}_{column}" if self.prefix else column

    def _archived_columns(self, fetches: Optional[List[Fetch]] = None) -> List[str]:
        """The archived names it emits: `columns`, or every column the source's fetches have, in order of first appearance."""
        if self.columns is not None and fetches is None:
            return list(self.columns)
        fetches = self._archive().fetches(self.source) if fetches is None else fetches
        seen = list(dict.fromkeys(column for fetch in fetches for column in fetch.columns))
        if self.columns is None:
            return seen
        unknown = [column for column in self.columns if column not in seen]
        if unknown:
            raise ArchiveError(
                f"ArchiveSource({self.source!r}): no fetch in {self.path} holds {unknown}; its columns are {seen}"
            )
        return list(self.columns)

    def column_names(self) -> List[str]:
        """The names of the columns `fetch` returns, prefix included. Without `columns` the archive's catalog says them."""
        return [self._named(column) for column in self._archived_columns()]

    def availability(self):
        """The rule of every column it emits: published at `known_at` in `timezone` on the day `days_before` before the
        delivery day, a date of `timezone`; its archive is a snapshot (one value per cell, as fetched by then)."""
        from ...information_set import Availability  # the information set imports the sources

        return Availability.published(
            self.known_at,
            self.days_before,
            timezone=self.timezone,
            vintage="snapshot",
            note=f"archived {self.source} in {self.path}, as fetched by {self.known_at} {self.timezone} on the delivery day "
            f"minus {self.days_before}",
        )

    def known_instant(self, day: Any) -> pd.Timestamp:
        """The instant the values of delivery day `day` (a date of `timezone`) are read at, UTC: `known_at` on the day
        `days_before` before it, a repeated wall time at its second occurrence and a skipped one at the switch."""
        from ...information_set import _parse_time

        wall = pd.Timestamp(pd.Timestamp(day).date()) - pd.Timedelta(days=self.days_before)
        wall = wall + pd.Timedelta(minutes=_parse_time(self.known_at))
        return wall.tz_localize(self.timezone, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")

    def get_cache_config(self) -> Optional[Dict]:
        # The archive grows with every recording: a per-source cache would serve what it held before. The single-file cache
        # sees a new fetch through `content_digest`.
        return None

    def content_digest(self) -> Optional[str]:
        """The sha256 of the archive's catalog: a new fetch changes it."""
        return self._archive().content_digest()

    # -- the replay ---------------------------------------------------------------------------------------------------------

    def _midnight(self, day: pd.Timestamp) -> pd.Timestamp:
        return day.tz_localize(self.timezone, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """The replayed values of every cell in `[start, end)`, UTC: each delivery day's from the newest fetch by its
        `known_instant` that asked for the cell. A sub-range holds the same values, as each day is read on its own."""
        start, end = _utc(start), _utc(end)
        archive = self._archive()
        fetches = archive.fetches(self.source)
        wanted = self._archived_columns(fetches)
        returned = [fetch.fetched_at for fetch in fetches]
        frames: Dict[str, pd.DataFrame] = {}
        pieces = []
        for day in _iter_days(start, end, self.timezone) if end > start else ():
            lo = max(start, self._midnight(day))
            hi = min(end, self._midnight(day + pd.Timedelta(days=1)))
            if lo >= hi:
                continue
            open_ranges = [(lo, hi)]  # the parts of the day no fetch by the cut-off has asked for yet
            for fetch in reversed(fetches[: bisect.bisect_right(returned, self.known_instant(day))]):
                if not open_ranges:
                    break
                if fetch.end <= lo or fetch.start >= hi:
                    continue
                taken, rest = [], []
                for a, b in open_ranges:
                    x, y = max(a, fetch.start), min(b, fetch.end)
                    if x >= y:
                        rest.append((a, b))
                        continue
                    taken.append((x, y))
                    rest.extend(part for part in ((a, x), (y, b)) if part[0] < part[1])
                open_ranges = rest
                if not taken:
                    continue
                if fetch.file not in frames:
                    frames[fetch.file] = archive.read(fetch)
                frame = frames[fetch.file]
                for x, y in taken:
                    pieces.append(frame.loc[(frame.index >= x) & (frame.index < y)].reindex(columns=wanted))
        names = [self._named(column) for column in wanted]
        pieces = [piece for piece in pieces if len(piece)]
        if not pieces:
            return pd.DataFrame(columns=names, index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
        out = pd.concat(pieces).sort_index(kind="stable")
        out.columns = names
        return out
