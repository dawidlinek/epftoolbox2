from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd


class DataSource(ABC):
    """Abstract base class for all data sources"""

    @abstractmethod
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
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
