from typing import List, Optional, Union
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
import logging

from epftoolbox2.data.sources.base import DataSource
from epftoolbox2.data.transformers.base import Transformer
from epftoolbox2.data.cache_manager import CacheManager


class DataPipeline:
    """Data pipeline that combines multiple sources and applies transformers.

    Example:
        >>> pipeline = DataPipeline(
        ...     sources=[CsvSource(...), EntsoeSource(...)],
        ...     transformers=[TimezoneTransformer(target_tz="Europe/Warsaw")]
        ... )
        >>> df = pipeline.run(start, end)

    Example:
        >>> pipeline = (
        ...     DataPipeline()
        ...     .add_source(EntsoeSource(country_code="PL", api_key="...", type=["load"]))
        ...     .add_source(OpenMeteoSource(latitude=52.23, longitude=21.01))
        ...     .add_transformer(TimezoneTransformer(target_tz="Europe/Warsaw"))
        ... )
        >>> df = pipeline.run(start, end)
    """

    def __init__(
        self,
        sources: Optional[List[DataSource]] = None,
        transformers: Optional[List[Transformer]] = None,
    ):
        self.sources: List[DataSource] = sources or []
        self.transformers: List[Transformer] = transformers or []

        self.console = Console()
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.addHandler(RichHandler(console=self.console, rich_tracebacks=True))
            self.logger.setLevel(logging.INFO)

    def add_source(self, source: DataSource) -> "DataPipeline":
        if not isinstance(source, DataSource):
            raise TypeError("source must be a DataSource instance")
        self.sources.append(source)
        return self

    def add_transformer(self, transformer: Transformer) -> "DataPipeline":
        if not isinstance(transformer, Transformer):
            raise TypeError("transformer must be a Transformer instance")
        self.transformers.append(transformer)
        return self

    def _fetch_with_cache(self, source: DataSource, start: pd.Timestamp, end: pd.Timestamp, cache_manager: CacheManager) -> pd.DataFrame:
        source_config = source.get_cache_config()
        if source_config is None:
            return source.fetch(start, end)

        cache_key = cache_manager.get_cache_key(source_config)
        missing_ranges = cache_manager.find_missing_ranges(cache_key, start, end)
        source_type = source_config.get("source_type", "unknown")

        if not missing_ranges:
            self.logger.info(f"Cache: Full hit for {source_type} source")
            return cache_manager.read_cached_data(cache_key, start, end)

        if len(missing_ranges) == 1 and missing_ranges[0] == (start, end):
            self.logger.info(f"Cache: Miss for {source_type} source")
        else:
            self.logger.info(f"Cache: Partial hit for {source_type} source")

        for missing_start, missing_end in missing_ranges:
            fresh_df = source.fetch(missing_start, missing_end)
            if fresh_df is not None and not fresh_df.empty:
                cache_manager.write_cache(cache_key, fresh_df, missing_start, missing_end, source_config)

        df = cache_manager.read_cached_data(cache_key, start, end)
        return df if df is not None else pd.DataFrame()

    def _parse_timestamp(self, ts: Union[str, pd.Timestamp]) -> pd.Timestamp:
        if ts == "today":
            return pd.Timestamp("today", tz="UTC").normalize()
        if isinstance(ts, str):
            return pd.Timestamp(ts, tz="UTC")
        return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")

    def run(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp], cache: Union[bool, str] = False) -> pd.DataFrame:
        if not self.sources:
            raise ValueError("At least one data source is required")

        start = self._parse_timestamp(start)
        end = self._parse_timestamp(end)

        if end <= start:
            raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")

        if isinstance(cache, str):
            cache_file = Path(cache)
            if cache_file.exists():
                self.logger.info(f"Cache: Loading from {cache}")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if df.index.tzinfo is None:
                    df.index = df.index.tz_localize("UTC")
                return df

        self.logger.info(f"Pipeline: Fetching data from {len(self.sources)} source(s)")

        cache_manager = CacheManager() if cache is True else None
        dataframes = []

        for source in self.sources:
            df = self._fetch_with_cache(source, start, end, cache_manager) if cache is True else source.fetch(start, end)
            if df is not None and not df.empty:
                dataframes.append(df)

        if not dataframes:
            self.logger.warning("Pipeline: No data returned from any source")
            return pd.DataFrame()

        result = dataframes[0]
        for df in dataframes[1:]:
            result = result.join(df, how="outer")

        for transformer in self.transformers:
            result = transformer.transform(result)

        if isinstance(cache, str):
            result.to_csv(cache)
            self.logger.info(f"Cache: Saved to {cache}")

        self.logger.info(f"Pipeline: Completed with {len(result)} rows")
        return result
