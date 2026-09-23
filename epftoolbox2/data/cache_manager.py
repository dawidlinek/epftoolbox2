"""Per-source on-disk cache.

Layout: ``<cache_dir>/<key>/metadata.json`` plus one Parquet file per fetched range. The key hashes the package cache
format version, the source class and ``source.get_cache_config()``. ``metadata.json`` lists the half-open UTC ranges
``[start, end)`` that are cached and the file holding each range; a file is only trusted inside its own range.

Only what a source actually returned is recorded, and never the last ``unsettled_period`` before now, because data
close to now may still be published or completed (actual load, next-day prices, weather model runs). Those ranges
are fetched again on the next run.

Writes go to a temporary file that is renamed into place, and every metadata read-modify-write happens under a
per-key file lock, so parallel processes (e.g. SLURM array tasks sharing one cache directory) never see a
half-written file.
"""

import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd
from filelock import FileLock, SoftFileLock, Timeout

from epftoolbox2.logging import get_logger

# Bump when the on-disk layout or the meaning of cached ranges changes: every existing cache is then ignored.
# 1: CSV files, whole requested range recorded. 2: Parquet, only returned and settled data recorded.
CACHE_FORMAT_VERSION = 2

# Data newer than this (relative to now, floored to UTC midnight) is returned but never recorded as cached.
UNSETTLED_PERIOD = pd.Timedelta(days=2)

logger = get_logger(__name__)


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


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
    systems without flock support (old filelock versions raise NotImplementedError there)."""
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

    def covered_end(self, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Timestamp]:
        """End of the range ``[start, x)`` that ``df`` (fetched for ``[start, end)``) may be recorded as covering.

        That is the last returned timestamp plus one step, capped at ``end`` and at the start of the unsettled
        period before now. ``None`` if nothing can be recorded.
        """
        index = clip_to_range(df, start, end).index.unique().sort_values()
        if len(index) == 0:
            return None
        steps = index[1:] - index[:-1]
        step = steps.min() if len(steps) else pd.Timedelta(0)
        settled = _now().floor("D") - self.unsettled_period
        covered = min(end, index[-1] + step, settled)
        return covered if covered > start else None

    def write_cache(
        self,
        cache_key: str,
        df: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        source_config: Dict,
        source_class: Optional[str] = None,
    ) -> Optional[pd.Timestamp]:
        """Store what ``df`` (fetched for ``[start, end)``) covers; returns the recorded end or ``None``.

        Rows outside the recorded range are not stored. The new range replaces any overlapping part of older ranges.
        """
        if df is None or df.empty:
            return None
        covered = self.covered_end(df, start, end)
        if covered is None:
            return None
        part = clip_to_range(df, start, covered)
        filename = self._get_data_filename(start, covered)

        with self.lock(cache_key):
            cache_path = self._get_cache_path(cache_key)
            atomic_write(cache_path / filename, lambda tmp: part.to_parquet(tmp))

            metadata = self._read_metadata(cache_key)
            kept = []
            for entry in metadata.get("cached_ranges", []):
                if entry["filename"] == filename:
                    continue
                range_start, range_end = self._range(entry)
                if range_start < start:
                    kept.append({**entry, "end": min(range_end, start).isoformat()})
                if range_end > covered:
                    kept.append({**entry, "start": max(range_start, covered).isoformat()})
            new_entry = {
                "start": start.isoformat(),
                "end": covered.isoformat(),
                "filename": filename,
                "rows": len(part),
                "retrieved_at": _now().isoformat(),
            }
            metadata.update(
                {
                    "cache_format": CACHE_FORMAT_VERSION,
                    "source_class": source_class,
                    "source_config": source_config,
                    "cached_ranges": sorted(kept + [new_entry], key=lambda r: pd.Timestamp(r["start"])),
                }
            )
            self._write_metadata(cache_key, metadata)

            referenced = {r["filename"] for r in metadata["cached_ranges"]}
            for old in cache_path.glob("data_*.parquet"):
                if old.name not in referenced:
                    old.unlink()

        self.logger.debug(f"Cached data: {filename}")
        return covered

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
