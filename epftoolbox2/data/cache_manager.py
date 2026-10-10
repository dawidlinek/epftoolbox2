"""Per-source on-disk cache.

Layout: ``<cache_dir>/<key>/metadata.json`` plus one Parquet file per fetched range. The key hashes the package cache
format version, the source class and ``source.get_cache_config()``. ``metadata.json`` lists the half-open UTC ranges
``[start, end)`` that are cached and the file holding each range; a file is only trusted inside its own range.

Only what a source actually returned is recorded, and never the last ``unsettled_period`` before now, because data
close to now may still be published or completed (actual load, next-day prices, weather model runs). Those ranges
are fetched again on the next run. A source can also say that its rows from some instant on are not settled
(``df.attrs["settled_until"]``, see ``DataSource``), for data that stays provisional or missing for longer than
the unsettled period, and the recorded ranges stop there too.

Writes go to a temporary file that is renamed into place, and every metadata read-modify-write happens under a
per-key file lock, so parallel processes (e.g. SLURM array tasks sharing one cache directory) never see a
half-written file.
"""

import hashlib
import json
import os
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd
from filelock import FileLock, SoftFileLock, Timeout

from epftoolbox2.logging import get_logger

# Bump when the on-disk layout or the meaning of cached ranges changes: every existing cache is then ignored.
# The cache is Parquet files plus a record of the ranges they cover, derived in ``covered_ranges`` and
# ``_still_published_tail_cap``.
CACHE_FORMAT_VERSION = 4

# Data newer than this (relative to now, floored to UTC midnight) is returned but never recorded as cached.
UNSETTLED_PERIOD = pd.Timedelta(days=2)

# ``DataFrame.attrs`` key under which a source declares that its rows at or after that UTC instant are not settled.
SETTLED_UNTIL_ATTR = "settled_until"

# A column whose last valid (non-null) value is older than this, relative to now, is treated as discontinued
# (e.g. a PsrType ENTSO-E no longer reports) rather than still being published, and does not cap
# ``covered_ranges``' tail — see ``_still_published_tail_cap``. Without this cutoff, a column that will never
# publish again would force every future fetch to re-request its tail forever.
DISCONTINUED_AFTER = pd.Timedelta(days=14)

# Diffs looked at on either side of a point to estimate the expected step there (see ``covered_ranges``): local,
# not one mode for the whole response, so a resolution change partway through a fetch does not make one side's
# normal step look like a gap relative to the other side's.
_STEP_WINDOW = 5

logger = get_logger(__name__)


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def unsettled_boundary(period: pd.Timedelta = UNSETTLED_PERIOD, timezone: Optional[str] = None) -> pd.Timestamp:
    """UTC instant ``period`` before now, floored to midnight: the per-source cache never records data past this
    point as covered, because it may still be published or completed. The single-file cache (``DataPipeline``'s
    ``cache="path"``) uses the same boundary (UTC, no ``timezone``) to decide whether a cached file might now be
    stale.

    Floored in UTC by default. Pass the pipeline's market ``timezone`` when capping a per-source cache write
    (``CacheManager.write_cache``/``covered_ranges``): flooring in UTC almost never lines up with where a source
    stamps its own daily values (ENTSO-E's A31 sits at the *area's* local midnight), so resuming a fetch there
    can fall strictly inside a one-day period. A source that carries such a period's value forward to cover its
    fetch's own start (``EntsoeSource._clip``) would then re-stamp the same day's value a second time, on top of
    the one already cached at its true local midnight — DST/timezone review finding 2. Flooring in the source's
    own zone instead lands the boundary exactly on that midnight, so nothing needs to be carried across it.
    ``period`` must be a whole number of days for this to floor to a local midnight rather than an arbitrary time.
    """
    now = _now()
    if timezone is None:
        return now.floor("D") - period
    days, remainder = divmod(period, pd.Timedelta(days=1))
    local_midnight = now.tz_convert(timezone).floor("D")
    return (local_midnight - pd.DateOffset(days=int(days)) - remainder).tz_convert("UTC")


def atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """Call ``write(tmp_path)`` and rename the result onto ``path``; readers see the old or the new file, never a
    partial one. The temporary file is removed if ``write`` fails."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


@contextmanager
def hold_lock(lock: FileLock, description: str) -> Iterator[None]:
    """Acquire an inter-process file lock (re-entrant for the same lock object), logging when we have to wait."""
    try:
        lock.acquire(timeout=0)
    except Timeout:
        logger.info(f"Cache: waiting for another process that is writing {description}")
        lock.acquire()
    try:
        yield
    finally:
        lock.release()


def make_file_lock(lock_path: Path) -> FileLock:
    """``FileLock`` (flock/msvcrt, released by the OS if the process dies); falls back to ``SoftFileLock`` on file
    systems without flock support (``FileLock`` raises NotImplementedError there)."""
    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=0)
        lock.release()
    except Timeout:
        pass
    except NotImplementedError:
        logger.warning(f"Cache: file system does not support flock, using a lock file instead ({lock_path})")
        return SoftFileLock(str(lock_path))
    return lock


def clip_to_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Rows with ``start <= index < end``. A tz-naive index is taken to be UTC (the source contract)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Expected a DatetimeIndex, got {type(df.index).__name__}")
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    return df[(df.index >= start) & (df.index < end)]


def settled_until_of(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """The instant a source declared in ``df.attrs["settled_until"]``, as a UTC timestamp: its rows at or after it
    are not settled yet. ``None`` if the source declared nothing (every row is settled) or ``NaT``. The value is a
    timestamp or an ISO 8601 string (``pandas`` writes ``attrs`` as JSON, so a frame that is saved with
    ``to_parquet`` needs the string). A tz-naive value is taken to be UTC, like a tz-naive index (the source
    contract).

    Read it right after ``fetch``: a clip or a concat of several frames may drop ``attrs``, or keep them only when
    every frame has the same.
    """
    value = df.attrs.get(SETTLED_UNTIL_ATTR)
    if value is None:
        return None
    try:
        settled_until = pd.Timestamp(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"df.attrs[{SETTLED_UNTIL_ATTR!r}] must be a timestamp, got {value!r}") from e
    if pd.isna(settled_until):
        return None
    return settled_until.tz_localize("UTC") if settled_until.tzinfo is None else settled_until.tz_convert("UTC")


def without_settled_until(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` without its ``settled_until`` declaration: the frame itself if it has none, else a shallow copy (the
    source's own frame is left alone). The declaration is for the cache only. It must not end up in a Parquet file
    (a timestamp in ``attrs`` cannot be written) or in the pipeline's output."""
    if SETTLED_UNTIL_ATTR not in df.attrs:
        return df
    stripped = df.copy(deep=False)
    stripped.attrs = {key: value for key, value in df.attrs.items() if key != SETTLED_UNTIL_ATTR}
    return stripped


def _local_steps(diffs: List[pd.Timedelta], window: int = _STEP_WINDOW) -> List[pd.Timedelta]:
    """For each diff, the most common diff among up to ``window`` neighbours on either side of it (the diff
    itself is excluded, so a gap can't count as evidence for its own normality). Ties favor the smaller step.
    Used as the expected spacing at that point instead of one step for the whole response, so a resolution change
    partway through a fetch (e.g. hourly data followed by 15-minute data) only makes the diffs nearest the boundary
    look like a gap, not every diff on either side.
    """
    steps = []
    for i in range(len(diffs)):
        neighbours = diffs[max(0, i - window) : i] + diffs[i + 1 : i + 1 + window]
        if not neighbours:
            steps.append(diffs[i])
            continue
        counts = Counter(neighbours)
        steps.append(min(counts, key=lambda d: (-counts[d], d)))
    return steps


def _still_published_tail_cap(df: pd.DataFrame, now: pd.Timestamp) -> Optional[pd.Timestamp]:
    """One step past the last valid (non-null) value of any column in ``df`` that is still being published,
    or ``None`` if no column restricts the tail.

    A multi-column source (``EntsoeSource`` concatenating its document types) joins its columns on an outer
    index, so the merged row index runs as far as the longest column even when a shorter one stopped early —
    for example ``load_actual`` during a multi-day TSO outage while ``price`` keeps publishing day-ahead rows.
    Judging coverage from that merged index alone (as ``covered_ranges`` otherwise does) would record the
    outage's NaN cells as covered, and they would never be refetched once the source backfills them. Capping
    the whole recorded range at one step past each still-published column's own last value fixes that, at the
    cost of re-requesting the other columns' already-correct data over that same short tail too — the cache
    only ever records whole time slices across every column, not per-column cells.

    A column counts as still published when its last valid value is newer than ``DISCONTINUED_AFTER`` before
    ``now``; an older one is discontinued (e.g. a PsrType ENTSO-E no longer reports) and is excluded, so it
    can't force a refetch forever. "One step" is that column's own local resolution — estimated the same way
    as ``covered_ranges``' row-level gaps, from a window of that column's own recent diffs — not the merged
    frame's, so a coarser column (e.g. a weekly value) alongside a finer one is not capped as if it shared the
    finer column's spacing.
    """
    caps = []
    for _, series in df.items():
        valid_index = series.dropna().index
        if len(valid_index) == 0:
            continue
        last_valid = valid_index[-1]
        if now - last_valid > DISCONTINUED_AFTER:
            continue
        if len(valid_index) == 1:
            continue  # no diff to estimate this column's own step from; it can't tighten the cap
        diffs = list(valid_index[1:] - valid_index[:-1])
        step = _local_steps(diffs)[-1]
        caps.append(last_valid + step)
    return min(caps) if caps else None


class CacheManager:
    def __init__(self, cache_dir: str = ".cache/sources", unsettled_period: pd.Timedelta = UNSETTLED_PERIOD):
        self.cache_dir = Path(os.path.expanduser(os.path.expandvars(str(cache_dir))))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.unsettled_period = pd.Timedelta(unsettled_period)
        self.logger = get_logger(__name__)
        self._locks: Dict[str, FileLock] = {}

    def get_cache_key(self, source_config: Dict, source_class: Optional[str] = None) -> str:
        """Hash of the cache format version, the source class and its cache config."""
        payload = {"cache_format": CACHE_FORMAT_VERSION, "source_class": source_class, "source_config": source_config}
        config_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    @contextmanager
    def lock(self, cache_key: str) -> Iterator[None]:
        """Inter-process lock for one cache key. Re-entrant within a thread."""
        if cache_key not in self._locks:
            self._locks[cache_key] = make_file_lock(self._get_cache_path(cache_key) / ".lock")
        with hold_lock(self._locks[cache_key], f"cache {cache_key}"):
            yield

    def _get_cache_path(self, cache_key: str) -> Path:
        path = self.cache_dir / cache_key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_metadata_path(self, cache_key: str) -> Path:
        return self._get_cache_path(cache_key) / "metadata.json"

    def _read_metadata(self, cache_key: str) -> Dict:
        empty = {"cache_format": CACHE_FORMAT_VERSION, "cached_ranges": [], "source_config": {}}
        metadata_path = self._get_metadata_path(cache_key)
        if not metadata_path.exists():
            return empty
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except (OSError, ValueError) as e:
            self.logger.warning(f"Cache: unreadable metadata {metadata_path} ({e}); treating the cache as empty")
            return empty
        if metadata.get("cache_format") != CACHE_FORMAT_VERSION:
            return empty
        return metadata

    def _write_metadata(self, cache_key: str, metadata: Dict):
        def write(tmp: Path):
            with open(tmp, "w") as f:
                json.dump(metadata, f, indent=2)

        atomic_write(self._get_metadata_path(cache_key), write)

    def _get_data_filename(self, start: pd.Timestamp, end: pd.Timestamp) -> str:
        fmt = "%Y%m%dT%H%M%S"
        return f"data_{start.tz_convert('UTC').strftime(fmt)}_{end.tz_convert('UTC').strftime(fmt)}.parquet"

    @staticmethod
    def _range(entry: Dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
        return pd.Timestamp(entry["start"]), pd.Timestamp(entry["end"])

    def _drop_entries(self, cache_key: str, broken: List[Dict]) -> None:
        with self.lock(cache_key):
            metadata = self._read_metadata(cache_key)
            metadata["cached_ranges"] = [r for r in metadata.get("cached_ranges", []) if r not in broken]
            self._write_metadata(cache_key, metadata)

    def find_missing_ranges(
        self, cache_key: str, requested_start: pd.Timestamp, requested_end: pd.Timestamp
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Sub-ranges of ``[requested_start, requested_end)`` not covered by an existing cache file."""
        with self.lock(cache_key):
            metadata = self._read_metadata(cache_key)
            cache_path = self._get_cache_path(cache_key)
            entries = metadata.get("cached_ranges", [])
            gone = [r for r in entries if not (cache_path / r["filename"]).exists()]
            if gone:
                self.logger.warning(f"Cache: {len(gone)} cached file(s) of {cache_key} are missing; fetching again")
                self._drop_entries(cache_key, gone)
            cached_ts_ranges = sorted(self._range(r) for r in entries if r not in gone)

        missing_ranges = []
        current_start = requested_start

        for cache_start, cache_end in cached_ts_ranges:
            if current_start < cache_start:
                gap_end = min(cache_start, requested_end)
                if current_start < gap_end:
                    missing_ranges.append((current_start, gap_end))
            if cache_end > current_start:
                current_start = cache_end
            if current_start >= requested_end:
                break

        if current_start < requested_end:
            missing_ranges.append((current_start, requested_end))

        return missing_ranges

    def read_cached_data(self, cache_key: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Cached rows in ``[start, end)``, each file restricted to its own recorded range.

        A file that is missing or cannot be read is removed from the metadata (so its range is fetched again) instead
        of silently dropping the source's columns.
        """
        with self.lock(cache_key):
            metadata = self._read_metadata(cache_key)
            cache_path = self._get_cache_path(cache_key)
            chunks, broken = [], []

            for entry in metadata.get("cached_ranges", []):
                range_start, range_end = self._range(entry)
                lo, hi = max(range_start, start), min(range_end, end)
                if lo >= hi:
                    continue
                filepath = cache_path / entry["filename"]
                try:
                    df = pd.read_parquet(filepath)
                except Exception as e:  # noqa: BLE001 - any unreadable file is a cache miss
                    self.logger.warning(f"Cache: cannot read {filepath} ({type(e).__name__}: {e}); fetching again")
                    broken.append(entry)
                    continue
                chunk = clip_to_range(df, lo, hi)
                if not chunk.empty:
                    chunks.append(chunk)

            if broken:
                self._drop_entries(cache_key, broken)

        if not chunks:
            return None

        # Recorded ranges are disjoint (a newer write trims older ranges), so the chunks do not overlap.
        return pd.concat(chunks).sort_index(kind="stable")

    def covered_ranges(
        self,
        df: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        timezone: Optional[str] = None,
        settled_until: Optional[pd.Timestamp] = None,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Contiguous, disjoint sub-ranges of ``[start, end)`` that ``df`` (fetched for ``[start, end)``) may be
        recorded as covering, each capped at ``end``, at the start of the unsettled period before now, at the
        instant from which the source declared its rows unsettled (``settled_until``) and at one step past the last
        value of any column that is still being published (``_still_published_tail_cap``).

        ``timezone``, when given, floors the unsettled-period cap to a local midnight in that zone instead of
        UTC (see ``unsettled_boundary``), so a later resumed fetch starts exactly at a source's own local
        midnight rather than strictly inside a one-day period.

        ``settled_until`` is the source's own statement that its rows from that UTC instant on are placeholders or
        provisional (``DataSource``). Given as ``None`` it is read from ``df.attrs``, which is the frame as the
        source returned it; a caller that has clipped or concatenated the frame passes the value it read before.
        The rows after it are still in ``df``, and are returned to the caller, only not recorded.

        A gap between two consecutive returned timestamps ends a run if it is wider than the expected spacing at
        that point in the response, estimated from a small window of neighbouring diffs (``_local_steps``)
        rather than one step for the whole response — a fetch that changes resolution partway through (e.g.
        hourly data followed by 15-minute data) must not make every diff on the finer side look like a gap
        relative to the coarser side's step, or vice versa; at most the handful of diffs nearest the boundary
        may be treated as a gap and refetched next time. Only rows actually present are ever recorded as covered
        — a genuine hole in an otherwise regular response (one missing hour out of a day of hourly data) is
        fetched again next time instead of being silently treated as covered and then lost for good.

        That row-level gap check works on the merged index across every column, so it cannot by itself see a
        column that stopped early while another kept going (the merged index still runs as far as the longer
        column) — the per-column tail cap below covers that case.

        Fewer than three returned timestamps give fewer than two diffs, so there is nothing to compare a lone
        diff against — it could be the real step, or it could itself be hiding a gap — and no step is inferred:
        a single timestamp still covers the whole range up to the cap (e.g. a source that returns exactly one
        row for any requested range, such as a weekly aggregate); exactly two timestamps cover only themselves,
        not the span between or beyond them.
        """
        if settled_until is None:
            settled_until = settled_until_of(df)
        clipped = clip_to_range(df, start, end)
        index = clipped.index.unique().sort_values()
        if len(index) == 0:
            return []
        cap = min(end, unsettled_boundary(self.unsettled_period, timezone))
        if settled_until is not None:
            cap = min(cap, settled_until)
        tail_cap = _still_published_tail_cap(clipped, _now())
        if tail_cap is not None:
            cap = min(cap, tail_cap)

        if len(index) == 1:
            return [(start, cap)] if cap > start else []
        if len(index) == 2:
            return []

        diffs = list(index[1:] - index[:-1])
        steps = _local_steps(diffs)

        ranges = []
        run_start = start
        prev = index[0]
        for cur, diff, step in zip(index[1:], diffs, steps):
            if diff > step:
                run_end = min(prev + step, cap)
                if run_end > run_start:
                    ranges.append((run_start, run_end))
                run_start = cur
            prev = cur
        run_end = min(prev + steps[-1], cap)
        if run_end > run_start:
            ranges.append((run_start, run_end))
        return ranges

    def _trim_overlap(
        self, entries: List[Dict], new_start: pd.Timestamp, new_end: pd.Timestamp, new_filename: str
    ) -> List[Dict]:
        """``entries`` with the part inside ``[new_start, new_end)`` removed (an entry left empty is dropped)."""
        kept = []
        for entry in entries:
            if entry["filename"] == new_filename:
                continue
            range_start, range_end = self._range(entry)
            if range_start < new_start:
                kept.append({**entry, "end": min(range_end, new_start).isoformat()})
            if range_end > new_end:
                kept.append({**entry, "start": max(range_start, new_end).isoformat()})
        return kept

    def write_cache(
        self,
        cache_key: str,
        df: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        source_config: Dict,
        source_class: Optional[str] = None,
        timezone: Optional[str] = None,
        settled_until: Optional[pd.Timestamp] = None,
    ) -> Optional[pd.Timestamp]:
        """Store the contiguous run(s) of ``df`` (fetched for ``[start, end)``) that are actually present and
        settled — see ``covered_ranges``. Returns the end of the last (rightmost) range recorded, or ``None`` if
        nothing was recorded; this does not imply contiguous coverage back to ``start`` when ``df`` had a gap, so
        callers rely on ``find_missing_ranges`` for that, not this return value.

        ``timezone`` (typically the owning ``DataPipeline``'s market zone) is passed through to
        ``covered_ranges`` so the unsettled-period cap lands on that zone's local midnight; see there. So is
        ``settled_until``, the instant from which the source declared its rows unsettled.

        Rows outside a recorded range are not stored. Each new range replaces any overlapping part of older ranges.
        """
        if df is None or df.empty:
            return None
        ranges = self.covered_ranges(df, start, end, timezone, settled_until)
        if not ranges:
            return None

        with self.lock(cache_key):
            cache_path = self._get_cache_path(cache_key)
            metadata = self._read_metadata(cache_key)
            kept = list(metadata.get("cached_ranges", []))
            new_entries = []

            for range_start, range_end in ranges:
                part = without_settled_until(clip_to_range(df, range_start, range_end))
                filename = self._get_data_filename(range_start, range_end)
                atomic_write(cache_path / filename, lambda tmp, part=part: part.to_parquet(tmp))
                kept = self._trim_overlap(kept, range_start, range_end, filename)
                new_entries.append(
                    {
                        "start": range_start.isoformat(),
                        "end": range_end.isoformat(),
                        "filename": filename,
                        "rows": len(part),
                        "retrieved_at": _now().isoformat(),
                    }
                )

            metadata.update(
                {
                    "cache_format": CACHE_FORMAT_VERSION,
                    "source_class": source_class,
                    "source_config": source_config,
                    "cached_ranges": sorted(kept + new_entries, key=lambda r: pd.Timestamp(r["start"])),
                }
            )
            self._write_metadata(cache_key, metadata)

            referenced = {r["filename"] for r in metadata["cached_ranges"]}
            for old in cache_path.glob("data_*.parquet"):
                if old.name not in referenced:
                    old.unlink()

        self.logger.debug(f"Cached data: {[e['filename'] for e in new_entries]}")
        return ranges[-1][1]

    def get_cache_info(self, cache_key: str) -> Dict:
        metadata = self._read_metadata(cache_key)
        cached_ranges = metadata.get("cached_ranges", [])
        if not cached_ranges:
            return {"exists": False, "num_chunks": 0, "date_ranges": []}
        return {
            "exists": True,
            "num_chunks": len(cached_ranges),
            "date_ranges": [
                {k: r.get(k) for k in ("start", "end", "filename", "rows", "retrieved_at")} for r in cached_ranges
            ],
            "source_config": metadata.get("source_config", {}),
        }
