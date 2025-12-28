import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
import logging


class CacheManager:
    def __init__(self, cache_dir: str = ".cache/sources"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.addHandler(RichHandler(console=self.console, rich_tracebacks=True))
            self.logger.setLevel(logging.INFO)

    def get_cache_key(self, source_config: Dict) -> str:
        config_str = json.dumps(source_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _get_cache_path(self, cache_key: str) -> Path:
        path = self.cache_dir / cache_key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_metadata_path(self, cache_key: str) -> Path:
        return self._get_cache_path(cache_key) / "metadata.json"

    def _read_metadata(self, cache_key: str) -> Dict:
        metadata_path = self._get_metadata_path(cache_key)
        if not metadata_path.exists():
            return {"cached_ranges": [], "source_config": {}}
        with open(metadata_path, "r") as f:
            return json.load(f)

    def _write_metadata(self, cache_key: str, metadata: Dict):
        with open(self._get_metadata_path(cache_key), "w") as f:
            json.dump(metadata, f, indent=2)

    def _get_data_filename(self, start: pd.Timestamp, end: pd.Timestamp) -> str:
        return f"data_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"

    def find_missing_ranges(self, cache_key: str, requested_start: pd.Timestamp, requested_end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        metadata = self._read_metadata(cache_key)
        cached_ranges = metadata.get("cached_ranges", [])

        if not cached_ranges:
            return [(requested_start, requested_end)]

        cached_ts_ranges = sorted([(pd.Timestamp(r["start"]), pd.Timestamp(r["end"])) for r in cached_ranges], key=lambda x: x[0])

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
        metadata = self._read_metadata(cache_key)
        cached_ranges = metadata.get("cached_ranges", [])

        if not cached_ranges:
            return None

        cache_path = self._get_cache_path(cache_key)
        overlapping_chunks = []

        for cache_range in cached_ranges:
            cache_start, cache_end = pd.Timestamp(cache_range["start"]), pd.Timestamp(cache_range["end"])
            if cache_start < end and cache_end > start:
                filepath = cache_path / cache_range["filename"]
                if filepath.exists():
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    if df.index.tzinfo is None:
                        df.index = df.index.tz_localize("UTC")
                    overlapping_chunks.append(df.loc[start:end])

        if not overlapping_chunks:
            return None

        combined = pd.concat(overlapping_chunks).sort_index()
        return combined[~combined.index.duplicated(keep="first")]

    def write_cache(self, cache_key: str, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, source_config: Dict):
        if df.empty:
            return

        cache_path = self._get_cache_path(cache_key)
        filename = self._get_data_filename(start, end)
        df.to_csv(cache_path / filename)

        metadata = self._read_metadata(cache_key)
        if "source_config" not in metadata:
            metadata["source_config"] = source_config

        new_range = {"start": start.isoformat(), "end": end.isoformat(), "filename": filename}

        existing_idx = next((i for i, r in enumerate(metadata.get("cached_ranges", [])) if r["filename"] == filename), None)
        if existing_idx is not None:
            metadata["cached_ranges"][existing_idx] = new_range
        else:
            metadata.setdefault("cached_ranges", []).append(new_range)

        self._write_metadata(cache_key, metadata)
        self.logger.debug(f"Cached data: {filename}")

    def get_cache_info(self, cache_key: str) -> Dict:
        metadata = self._read_metadata(cache_key)
        cached_ranges = metadata.get("cached_ranges", [])
        if not cached_ranges:
            return {"exists": False, "num_chunks": 0, "date_ranges": []}
        return {"exists": True, "num_chunks": len(cached_ranges), "date_ranges": [{"start": r["start"], "end": r["end"]} for r in cached_ranges], "source_config": metadata.get("source_config", {})}
