import pandas as pd
from typing import Optional, Union
from .base import DataSource
import holidays as holidays_lib
from astral import LocationInfo
from astral.sun import sun

COUNTRY_DATA = {
    "PL": {"timezone": "Europe/Warsaw", "lat": 52.2297, "lon": 21.0122},
    "DE": {"timezone": "Europe/Berlin", "lat": 52.5200, "lon": 13.4050},
    "FR": {"timezone": "Europe/Paris", "lat": 48.8566, "lon": 2.3522},
    "ES": {"timezone": "Europe/Madrid", "lat": 40.4168, "lon": -3.7038},
    "IT": {"timezone": "Europe/Rome", "lat": 41.9028, "lon": 12.4964},
    "NL": {"timezone": "Europe/Amsterdam", "lat": 52.3676, "lon": 4.9041},
    "BE": {"timezone": "Europe/Brussels", "lat": 50.8503, "lon": 4.3517},
    "AT": {"timezone": "Europe/Vienna", "lat": 48.2082, "lon": 16.3738},
    "CH": {"timezone": "Europe/Zurich", "lat": 46.9480, "lon": 7.4474},
    "CZ": {"timezone": "Europe/Prague", "lat": 50.0755, "lon": 14.4378},
    "SK": {"timezone": "Europe/Bratislava", "lat": 48.1486, "lon": 17.1077},
    "HU": {"timezone": "Europe/Budapest", "lat": 47.4979, "lon": 19.0402},
    "RO": {"timezone": "Europe/Bucharest", "lat": 44.4268, "lon": 26.1025},
    "BG": {"timezone": "Europe/Sofia", "lat": 42.6977, "lon": 23.3219},
    "GR": {"timezone": "Europe/Athens", "lat": 37.9838, "lon": 23.7275},
    "PT": {"timezone": "Europe/Lisbon", "lat": 38.7223, "lon": -9.1393},
    "DK": {"timezone": "Europe/Copenhagen", "lat": 55.6761, "lon": 12.5683},
    "SE": {"timezone": "Europe/Stockholm", "lat": 59.3293, "lon": 18.0686},
    "NO": {"timezone": "Europe/Oslo", "lat": 59.9139, "lon": 10.7522},
    "FI": {"timezone": "Europe/Helsinki", "lat": 60.1699, "lon": 24.9384},
    "GB": {"timezone": "Europe/London", "lat": 51.5074, "lon": -0.1278},
    "IE": {"timezone": "Europe/Dublin", "lat": 53.3498, "lon": -6.2603},
}

WEEKDAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]


class CalendarSource(DataSource):
    def __init__(
        self,
        country: str,
        timezone: Optional[str] = None,
        holidays: Union[str, bool] = "binary",
        weekday: Union[str, bool] = "number",
        hour: Union[str, bool] = False,
        month: Union[str, bool] = False,
        daylight: bool = False,
        prefix: str = "",
    ):
        self.country = country.upper()
        self._validate_country()

        country_info = COUNTRY_DATA[self.country]
        self.timezone = timezone or country_info["timezone"]
        self.lat = country_info["lat"]
        self.lon = country_info["lon"]

        self.holidays = holidays
        self.weekday = weekday
        self.hour = hour
        self.month = month
        self.daylight = daylight
        self.prefix = prefix

        self._validate_config()

    def _validate_country(self):
        if self.country not in COUNTRY_DATA:
            raise ValueError(f"Unsupported country: {self.country}. Supported: {list(COUNTRY_DATA.keys())}")

    def _validate_config(self):
        valid_holiday = {False, "binary", "onehot", "name"}
        valid_weekday = {False, "number", "onehot", "name"}
        valid_hour = {False, "number", "onehot"}
        valid_month = {False, "number", "onehot", "name"}

        if self.holidays not in valid_holiday:
            raise ValueError(f"Invalid holidays value: {self.holidays}. Valid: {valid_holiday}")
        if self.weekday not in valid_weekday:
            raise ValueError(f"Invalid weekday value: {self.weekday}. Valid: {valid_weekday}")
        if self.hour not in valid_hour:
            raise ValueError(f"Invalid hour value: {self.hour}. Valid: {valid_hour}")
        if self.month not in valid_month:
            raise ValueError(f"Invalid month value: {self.month}. Valid: {valid_month}")

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
        end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")

        index = pd.date_range(start, end, freq="1h", tz="UTC")
        local_index = index.tz_convert(self.timezone)

        df = pd.DataFrame(index=index)

        if self.holidays:
            df = self._add_holidays(df, local_index)
        if self.weekday:
            df = self._add_weekday(df, local_index)
        if self.hour:
            df = self._add_hour(df, local_index)
        if self.month:
            df = self._add_month(df, local_index)
        if self.daylight:
            df = self._add_daylight(df, local_index)

        return df

    def _add_holidays(self, df: pd.DataFrame, local_index: pd.DatetimeIndex) -> pd.DataFrame:
        country_holidays = holidays_lib.country_holidays(self.country)
        holiday_names = pd.Series([country_holidays.get(d.date()) for d in local_index], index=df.index)

        if self.holidays == "binary":
            df[f"{self.prefix}is_holiday"] = holiday_names.notna().astype(int)
        elif self.holidays == "name":
            df[f"{self.prefix}holiday_name"] = holiday_names
        elif self.holidays == "onehot":
            df[f"{self.prefix}is_holiday"] = holiday_names.notna().astype(int)
            dummies = pd.get_dummies(holiday_names, prefix=f"{self.prefix}holiday", dtype=int)
            df = pd.concat([df, dummies], axis=1)
        return df

    def _add_weekday(self, df: pd.DataFrame, local_index: pd.DatetimeIndex) -> pd.DataFrame:
        weekday_series = pd.Series(local_index.weekday, index=df.index)

        if self.weekday == "number":
            df[f"{self.prefix}weekday"] = weekday_series
        elif self.weekday == "name":
            df[f"{self.prefix}weekday_name"] = weekday_series.map(lambda x: WEEKDAY_NAMES[x])
        elif self.weekday == "onehot":
            weekday_names = weekday_series.map(lambda x: WEEKDAY_NAMES[x])
            dummies = pd.get_dummies(weekday_names, prefix=f"{self.prefix}weekday", dtype=int)
            df = pd.concat([df, dummies], axis=1)
        return df

    def _add_hour(self, df: pd.DataFrame, local_index: pd.DatetimeIndex) -> pd.DataFrame:
        hour_series = pd.Series(local_index.hour, index=df.index)

        if self.hour == "number":
            df[f"{self.prefix}hour"] = hour_series
        elif self.hour == "onehot":
            dummies = pd.get_dummies(hour_series, prefix=f"{self.prefix}hour", dtype=int)
            df = pd.concat([df, dummies], axis=1)
        return df

    def _add_month(self, df: pd.DataFrame, local_index: pd.DatetimeIndex) -> pd.DataFrame:
        month_series = pd.Series(local_index.month, index=df.index)

        if self.month == "number":
            df[f"{self.prefix}month"] = month_series
        elif self.month == "name":
            df[f"{self.prefix}month_name"] = month_series.map(lambda x: MONTH_NAMES[x - 1])
        elif self.month == "onehot":
            month_names = month_series.map(lambda x: MONTH_NAMES[x - 1])
            dummies = pd.get_dummies(month_names, prefix=f"{self.prefix}month", dtype=int)
            df = pd.concat([df, dummies], axis=1)
        return df

    def _add_daylight(self, df: pd.DataFrame, local_index: pd.DatetimeIndex) -> pd.DataFrame:
        location = LocationInfo(latitude=self.lat, longitude=self.lon, timezone=self.timezone)

        sunrise_times = []
        sunset_times = []
        daylight_hours = []

        for dt in local_index:
            try:
                s = sun(location.observer, date=dt.date(), tzinfo=location.timezone)
                sunrise = s["sunrise"]
                sunset = s["sunset"]
                sunrise_times.append(sunrise.hour + sunrise.minute / 60)
                sunset_times.append(sunset.hour + sunset.minute / 60)
                daylight_hours.append((sunset - sunrise).total_seconds() / 3600)
            except ValueError:
                sunrise_times.append(None)
                sunset_times.append(None)
                daylight_hours.append(None)

        df[f"{self.prefix}sunrise"] = sunrise_times
        df[f"{self.prefix}sunset"] = sunset_times
        df[f"{self.prefix}daylight_hours"] = daylight_hours
        return df

    def get_cache_config(self) -> Optional[dict]:
        return None
