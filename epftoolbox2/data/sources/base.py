from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd


class DataSource(ABC):
    """Abstract base class for all data sources.

    **Settled rows.** The per-source cache (``DataPipeline.run(..., cache=True)``) never records the last two days
    before now as covered, because data that recent may still be published or completed. A source whose rows stay
    placeholders or provisional for longer than that, for example NaN for a period the provider has not published
    yet, declares it on the frame ``fetch`` returns::

        df.attrs["settled_until"] = pd.Timestamp("2026-09-01", tz="UTC")

    Rows at or after that UTC instant are not settled: they are returned for the current run, but the cache does
    not record them as covered, so they are fetched again next time while the settled rows before it are served
    from disk. The value is a ``pd.Timestamp`` or an ISO 8601 string (``ts.isoformat()``), and a naive one is
    taken to be UTC. Use the string if the frame may be saved with ``to_parquet``, which writes ``attrs`` as JSON.
    It has no effect when it is not earlier than the ``end`` that was asked for, and a source leaves it out when
    every row is settled. The single-file cache (``cache="path"``)
    does not reuse a file whose sources were settled only up to a time before its ``end``. The pipeline reads and
    removes the attribute right after ``fetch``, so it never reaches the output frame.
    """

    @abstractmethod
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch data for the specified time period

        Args:
            start: Start timestamp (UTC, inclusive)
            end: End timestamp (UTC, exclusive)

        Returns:
            DataFrame with a UTC DatetimeIndex covering at least ``[start, end)``. Rows outside that range are allowed
            but ``DataPipeline`` drops them, so cached and uncached runs return the same frame. A fetch for a range
            must return the same values for its rows as a fetch for any larger range that contains it, otherwise a
            run assembled from cached pieces differs from a fresh run.
        """
        pass

    @abstractmethod
    def _validate_config(self) -> bool:
        """
        Validate the data source configuration

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def get_cache_config(self) -> Optional[Dict]:
        """
        Get configuration parameters for cache key generation.

        Returns:
            Dictionary of configuration parameters that uniquely identify this source's output, or ``None`` to
            disable caching. Include every parameter that changes the output, and a ``"schema_version"`` entry
            that is bumped whenever parsing changes, so caches written by older code are not served. The cache key
            also contains the source class and the package cache format version.
        """
        pass
