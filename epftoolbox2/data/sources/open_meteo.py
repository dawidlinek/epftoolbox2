import pandas as pd
from pandas.tseries.offsets import DateOffset
from typing import List
import requests
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.logging import RichHandler
import logging
import time
from .base import DataSource


class OpenMeteoSource(DataSource):
    """
    Open-Meteo data source for weather forecast data

    Supports fetching historical weather forecasts for energy price prediction.
    Uses the Open-Meteo Previous Runs API to retrieve past forecast data.

    Example:
        >>> source = OpenMeteoSource(
        ...     latitude=52.23,
        ...     longitude=21.01,
        ...     horizon=7,
        ...     model="jma_seamless"
        ... )
        >>> df = source.fetch(
        ...     start=pd.Timestamp('2024-01-01', tz='UTC'),
        ...     end=pd.Timestamp('2024-01-07', tz='UTC')
        ... )
        >>> # df contains all weather forecast columns
    """

    API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

    DEFAULT_COLUMNS = [
        "temperature_2m",
        "rain",
        "showers",
        "snowfall",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "precipitation",
        "weather_code",
        "surface_pressure",
        "pressure_msl",
        "cloud_cover",
        "wind_speed_10m",
        "wind_direction_10m",
    ]

    def __init__(
        self,
        latitude: float,
        longitude: float,
        horizon: int = 7,
        model: str = "jma_seamless",
        columns: List[str] = None,
        prefix: str = "",
    ):
        """
        Initialize Open-Meteo data source

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            horizon: Number of days ahead to fetch forecasts for (default: 7)
            model: Weather model to use (default: "jma_seamless")
            columns: List of weather variables to fetch (default: DEFAULT_COLUMNS)
            prefix: Prefix to add to column names (default: "")
        """
        self.latitude = latitude
        self.longitude = longitude
        self.horizon = horizon
        self.model = model
        self.columns = columns if columns else self.DEFAULT_COLUMNS
        self.prefix = prefix
        self.session = requests.Session()

        self.console = Console()
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.addHandler(RichHandler(console=self.console, rich_tracebacks=True))
            self.logger.setLevel(logging.INFO)

        self._validate_config()

    def __del__(self):
        """Cleanup session on object destruction"""
        if hasattr(self, "session") and self.session:
            self.session.close()

    def _validate_config(self) -> bool:
        """Validate the configuration parameters"""
        if not isinstance(self.latitude, (int, float)) or not -90 <= self.latitude <= 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {self.latitude}")

        if not isinstance(self.longitude, (int, float)) or not -180 <= self.longitude <= 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {self.longitude}")

        if not isinstance(self.horizon, int) or self.horizon <= 0:
            raise ValueError(f"Horizon must be a positive integer, got {self.horizon}")

        if not self.columns:
            raise ValueError("At least one weather column must be specified")

        return True

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch weather forecast data for the specified time period

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            DataFrame with weather forecasts
        """
        start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
        end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")

        if end <= start:
            raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")

        start_time = time.time()

        extended_end = end + DateOffset(days=self.horizon - 1)

        chunks = self._generate_chunks(start, extended_end, months=3)

        self.logger.info(f"Open-Meteo [{self.latitude}, {self.longitude}]: Start downloading weather data")

        all_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Open-Meteo [{self.latitude}, {self.longitude}]: Downloading...",
                total=len(chunks),
            )

            for chunk_start, chunk_end in chunks:
                date_range = f"{chunk_start.date()} to {chunk_end.date()}"
                progress.update(task, description=f"[cyan]Open-Meteo [{self.latitude}, {self.longitude}]: {date_range}")

                chunk_data = self._fetch_chunk(chunk_start, chunk_end)
                if chunk_data is not None and not chunk_data.empty:
                    all_results.append(chunk_data)

                progress.advance(task)

        if all_results:
            result_df = pd.concat(all_results).sort_index()
            result_df = result_df[~result_df.index.duplicated(keep="first")]

            if self.prefix:
                result_df.columns = [f"{self.prefix}_{col}" for col in result_df.columns]
        else:
            result_df = pd.DataFrame()

        elapsed = time.time() - start_time
        self.logger.info(f"Open-Meteo [{self.latitude}, {self.longitude}]: Download completed successfully in {elapsed:.2f} sec")

        return result_df

    def _generate_chunks(self, start: pd.Timestamp, end: pd.Timestamp, months: int = 3) -> list:
        if months <= 0:
            raise ValueError(f"months parameter must be positive, got {months}")

        chunks = []
        current = start

        while current < end:
            next_chunk = current + pd.DateOffset(months=months)
            chunk_end = min(next_chunk, end)
            chunks.append((current, chunk_end))
            current = next_chunk - DateOffset(days=self.horizon)

        return chunks

    def _fetch_chunk(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": [f"{column}_previous_day{i + 1}" for i in range(self.horizon) for column in self.columns],
            "models": self.model,
            "timezone": "GMT",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
        }

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = self.session.get(self.API_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if "error" in data and data["error"]:
                    reason = data.get("reason", "Unknown error")
                    self.logger.warning(f"API error: {reason}")

                    if "Minutely API request limit exceeded" in reason:
                        self.logger.info("Rate limit exceeded. Waiting 60 seconds...")
                        time.sleep(60)
                        retry_count += 1
                        continue
                    elif "Too many concurrent requests" in reason:
                        self.logger.info("Too many concurrent requests. Waiting 10 seconds...")
                        time.sleep(10)
                        retry_count += 1
                        continue
                    else:
                        raise ValueError(f"Open-Meteo API error: {reason}")

                return self._parse_weather_data(data)

            except requests.RequestException as e:
                retry_count += 1
                self.logger.warning(f"Request failed (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    time.sleep(10)
                else:
                    raise

        raise RuntimeError(f"Failed to fetch data after {max_retries} retries")

    def _parse_weather_data(self, response_data: dict) -> pd.DataFrame:
        if "hourly" not in response_data or "time" not in response_data["hourly"]:
            return pd.DataFrame()

        weather = {}
        hourly_data = response_data["hourly"]
        times = hourly_data["time"]

        for x, timestamp_str in enumerate(times):
            try:
                tmp = {}
                for i in range(1, self.horizon + 1):
                    for column in self.columns:
                        column_key = f"{column}_previous_day{i}"
                        if column_key in hourly_data:
                            forecast_idx = x + (i * 24)
                            if forecast_idx < len(hourly_data[column_key]):
                                value = hourly_data[column_key][forecast_idx]
                                tmp[f"{column}_d+{i}"] = value

                timestamp = pd.Timestamp(timestamp_str, tz="UTC")
                # Filter out forecasts beyond now+13h to prevent data leakage in backtesting
                # This ensures we only use forecasts that were actually available at that time
                if timestamp > (pd.Timestamp.now(tz="UTC").floor("h") + DateOffset(hours=13)):
                    continue

                if tmp:
                    weather[timestamp_str] = tmp

            except (IndexError, KeyError):
                continue

        if not weather:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(weather, orient="index")
        df.index.name = "datetime"
        df.index = pd.to_datetime(df.index).tz_localize("UTC")

        return df
