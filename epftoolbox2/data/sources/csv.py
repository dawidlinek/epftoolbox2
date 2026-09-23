import pandas as pd
from typing import List, Optional
from pathlib import Path
from .base import DataSource
from epftoolbox2.logging import get_logger

# pandas infers the compression from these suffixes.
SUPPORTED_SUFFIXES = (".csv", ".csv.gz", ".csv.bz2", ".csv.xz", ".csv.zip")


class CsvSource(DataSource):
    """
    CSV data source for loading time-series data from CSV files.

    Timestamps with a UTC offset (including mixed offsets across DST, as pandas writes a local-time index)
    are converted to UTC. Naive timestamps are read as local times in `timezone` (default UTC).
    The output has a UTC index and covers [start, end).

    Example:
        >>> source = CsvSource(
        ...     file_path='data/prices.csv',
        ...     datetime_column='timestamp',
        ...     columns=['price', 'load'],
        ...     timezone='Europe/Warsaw',
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
        timezone: str = "UTC",
    ):
        """
        Args:
            file_path: Path to the CSV file (.csv, or compressed .csv.gz/.csv.bz2/.csv.xz/.csv.zip)
            datetime_column: Name of the datetime column (default: "datetime")
            columns: List of columns to include. If None, all columns except datetime are used.
            prefix: Prefix to add to column names (default: "")
            datetime_format: Format string for parsing datetime. If None, pandas will infer.
            separator: CSV separator character (default: ",")
            timezone: Time zone of naive timestamps (default: "UTC"). Ignored for timestamps with an offset.
                Naive local times must be in chronological order so that the repeated hour of the
                fall-back day can be told apart; times that don't exist (spring-forward gap) raise.
        """
        self.file_path = Path(file_path)
        self.datetime_column = datetime_column
        self.columns = columns
        self.prefix = prefix
        self.datetime_format = datetime_format
        self.separator = separator
        self.timezone = timezone

        self.logger = get_logger(__name__)

        self._validate_config()

    def _validate_config(self) -> bool:
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        if not self.file_path.name.lower().endswith(SUPPORTED_SUFFIXES):
            raise ValueError(
                f"File must have .csv extension (optionally compressed: {', '.join(SUPPORTED_SUFFIXES[1:])}), "
                f"got: {self.file_path.name}"
            )

        if not self.datetime_column:
            raise ValueError("datetime_column cannot be empty")

        try:
            pd.Timestamp("2000-01-01").tz_localize(self.timezone)
        except Exception as e:
            raise ValueError(f"Unknown timezone: {self.timezone!r}") from e

        return True

    def _parse_datetimes(self, values: pd.Series) -> pd.DatetimeIndex:
        kwargs = {"format": self.datetime_format} if self.datetime_format else {}

        if values.isna().any():
            rows = list(values.index[values.isna()][:5])
            raise ValueError(
                f"{values.isna().sum()} row(s) have no timestamp in '{self.datetime_column}' (rows {rows})"
            )
        if values.empty:
            return pd.DatetimeIndex([], tz="UTC", name=self.datetime_column)

        # pandas infers one format from the first value, so a file mixing naive and offset timestamps fails to parse.
        naive = pd.to_datetime(values.iloc[:1], **kwargs).dt.tz is None
        if not naive:
            # utc=True also handles mixed offsets (e.g. +01:00/+02:00 across a DST switch).
            return pd.DatetimeIndex(pd.to_datetime(values, utc=True, **kwargs), name=self.datetime_column)

        parsed = pd.DatetimeIndex(pd.to_datetime(values, **kwargs), name=self.datetime_column)
        try:
            # Localize in file order: "infer" tells the two occurrences of the fall-back hour apart by their order.
            localized = parsed.tz_localize(self.timezone, ambiguous="infer", nonexistent="raise")
        except Exception as e:
            nonexistent = "NonExistent" in type(e).__name__ or "nonexistent" in str(e).lower()
            kind = "a local time that does not exist" if nonexistent else "an ambiguous local time"
            raise ValueError(
                f"CSV Source: {kind} in {self.timezone} around a DST switch ({e}). Naive local timestamps must be "
                f"complete and in chronological order there; otherwise export them in UTC or with a UTC offset."
            ) from e
        return localized.tz_convert("UTC")

    def _load_data(self) -> pd.DataFrame:
        self.logger.info(f"CSV Source: Loading data from {self.file_path}")

        df = pd.read_csv(
            self.file_path,
            sep=self.separator,
        )

        if self.datetime_column not in df.columns:
            raise ValueError(
                f"Datetime column '{self.datetime_column}' not found in CSV. Available columns: {list(df.columns)}"
            )

        index = self._parse_datetimes(df[self.datetime_column])
        df = df.drop(columns=[self.datetime_column])
        df.index = index

        duplicated = df.index[df.index.duplicated()]
        if len(duplicated):
            raise ValueError(
                f"CSV Source: {len(duplicated)} duplicated timestamp(s) in {self.file_path} after conversion to UTC, "
                f"first: {list(duplicated[:3].astype(str))}. If the file holds naive local times, set `timezone`."
            )

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
            start: Start timestamp (inclusive)
            end: End timestamp (exclusive)

        Returns:
            DataFrame with the rows in [start, end), indexed in UTC
        """
        start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
        end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")

        if end <= start:
            raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")

        df = self._load_data()

        result_df = df[(df.index >= start) & (df.index < end)].copy()

        if self.prefix:
            result_df.columns = [f"{self.prefix}_{col}" for col in result_df.columns]

        self.logger.info(f"CSV Source: Fetched {len(result_df)} rows from {start} to {end}")

        return result_df

    def get_cache_config(self) -> Optional[dict]:
        return None
