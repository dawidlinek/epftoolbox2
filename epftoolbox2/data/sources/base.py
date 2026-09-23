from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd


class DataSource(ABC):
    """Abstract base class for all data sources"""

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
