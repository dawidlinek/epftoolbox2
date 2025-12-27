from typing import List, Optional
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
import logging

from epftoolbox2.data.sources.base import DataSource
from epftoolbox2.data.transformers.base import Transformer


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

    def run(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if not self.sources:
            raise ValueError("At least one data source is required")

        start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
        end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")

        if end <= start:
            raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")

        self.logger.info(f"Pipeline: Fetching data from {len(self.sources)} source(s)")

        dataframes = []
        for source in self.sources:
            df = source.fetch(start, end)
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

        self.logger.info(f"Pipeline: Completed with {len(result)} rows")
        return result
