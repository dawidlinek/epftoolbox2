import pandas as pd
from typing import List, Optional
from pathlib import Path
from .base import DataSource
from epftoolbox2.logging import get_logger


class CsvSource(DataSource):
    """
    CSV data source for loading time-series data from CSV files
    All timestamps are handled in UTC.
    Example:
        >>> source = CsvSource(
        ...     file_path='data/prices.csv',
        ...     datetime_column='timestamp',
        ...     columns=['price', 'load']
        ... )
        >>> df = source.fetch(
        ...     start=pd.Timestamp('2024-01-01', tz='UTC'),
        ...     end=pd.Timestamp('2024-01-07', tz='UTC')
        ... )
    """

    def __init__(
        self,
        file_path: str,
        datetime_column: str = "datetime",
        columns: Optional[List[str]] = None,
        prefix: str = "",
        datetime_format: Optional[str] = None,
        separator: str = ",",
    ):
        """
        Args:
            file_path: Path to the CSV file
            datetime_column: Name of the datetime column (default: "datetime")
            columns: List of columns to include. If None, all columns except datetime are used.
            prefix: Prefix to add to column names (default: "")
            datetime_format: Format string for parsing datetime. If None, pandas will infer.
            separator: CSV separator character (default: ",")
        """
        self.file_path = Path(file_path)
        self.datetime_column = datetime_column
        self.columns = columns
        self.prefix = prefix
        self.datetime_format = datetime_format
        self.separator = separator

        self.logger = get_logger(__name__)

        self._validate_config()

    def _validate_config(self) -> bool:
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        if not self.file_path.suffix.lower() == ".csv":
            raise ValueError(f"File must have .csv extension, got: {self.file_path.suffix}")

        if not self.datetime_column:
            raise ValueError("datetime_column cannot be empty")

        return True

    def _load_data(self) -> pd.DataFrame:
        self.logger.info(f"CSV Source: Loading data from {self.file_path}")

        df = pd.read_csv(
            self.file_path,
            sep=self.separator,
        )

        if self.datetime_column not in df.columns:
            raise ValueError(f"Datetime column '{self.datetime_column}' not found in CSV. Available columns: {list(df.columns)}")

        if self.datetime_format:
            df[self.datetime_column] = pd.to_datetime(
                df[self.datetime_column],
                format=self.datetime_format,
            )
        else:
            df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])

        if df[self.datetime_column].dt.tz is None:
            df[self.datetime_column] = df[self.datetime_column].dt.tz_localize("UTC")
        else:
            df[self.datetime_column] = df[self.datetime_column].dt.tz_convert("UTC")

        df = df.set_index(self.datetime_column)
        df = df.sort_index()

        if self.columns:
            missing_cols = set(self.columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Columns not found in CSV: {missing_cols}. Available columns: {list(df.columns)}")
            df = df[self.columns]

        self.logger.info(f"CSV Source: Loaded {len(df)} rows")

        return df

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch data for the specified time period

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            DataFrame with the requested data for the given time period
        """
        start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
        end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")

        if end <= start:
            raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")

        df = self._load_data()

        result_df = df.loc[start:end].copy()

        if self.prefix:
            result_df.columns = [f"{self.prefix}_{col}" for col in result_df.columns]

        self.logger.info(f"CSV Source: Fetched {len(result_df)} rows from {start} to {end}")

        return result_df

    def get_cache_config(self) -> dict:
        return None
