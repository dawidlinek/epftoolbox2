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
            start: Start timestamp
            end: End timestamp

        Returns:
            Dictionary mapping data type names to pandas DataFrames
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
            Dictionary of configuration parameters that uniquely identify this source.
            Changes to these parameters should invalidate the cache.
        """
        pass
