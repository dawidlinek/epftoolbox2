import re
import unicodedata
from typing import List, Optional, Union

import holidays as holidays_lib
import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import sun
from pandas.tseries.frequencies import to_offset

from epftoolbox2.logging import get_logger
from .base import DataSource

logger = get_logger(__name__)

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
MONTH_NAMES = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]

# Holiday names are requested in this language so that `holiday_name` values and one-hot column names do not
# depend on the process locale (without it, `holidays` falls back to LANG/LC_ALL) or the country's own language.
HOLIDAY_LANGUAGE = "en_US"


def _sanitise_name(name: str) -> str:
    """'New Year's Day' -> 'new_years_day': ASCII, lower case, words joined by underscores."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    ascii_name = ascii_name.replace("'", "")
    return re.sub(r"[^0-9a-z]+", "_", ascii_name.lower()).strip("_") or "unnamed"


class CalendarSource(DataSource):
    """Calendar features on a UTC grid, computed from local dates and local wall-clock times.

    Every feature is known in advance, so its `_d+k` versions are legal predictors.

    Example:
        >>> source = CalendarSource(country="DE", subdiv="BY", holidays="binary", weekday="onehot")
        >>> df = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
    """

    def __init__(
        self,
        country: str,
        timezone: Optional[str] = None,
        holidays: Union[str, bool] = "binary",
        weekday: Union[str, bool] = "number",
        hour: Union[str, bool] = False,
        month: Union[str, bool] = False,
        daylight: bool = False,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        prefix: str = "",
        freq: str = "1h",
        subdiv: Optional[str] = None,
        categories: Optional[Union[str, List[str]]] = None,
    ):
        """
        Args:
            country: ISO country code (see COUNTRY_DATA).
            timezone: Time zone whose local dates and wall-clock times define the features (default: the country's).
            holidays: False | "binary" | "onehot" | "name".
            weekday: False | "number" | "onehot" | "name".
            hour: False | "number" | "onehot". Hourly or coarser `freq`: local hour of day (`hour`, `is_{h}`).
                Sub-hourly `freq`: index of the period within the local day (`period`, `is_period_{p}`),
                e.g. 0-95 at 15 minutes, numbered like the models' own period index.
            month: False | "number" | "onehot" | "name".
            daylight: Add sunrise, sunset (local decimal hours) and daylight_hours.
            lat, lon: Location for daylight (default: the country's capital).
            prefix: Prefix for column names (no separator is added).
            freq: Row spacing. Must divide one hour, or be a whole number of hours that divides one day.
            subdiv: Subdivision for regional holidays, e.g. "BY" or "DE-BY" (Bavaria), "ENG" (England), "ZH".
            categories: Holiday categories of the `holidays` package, e.g. ["public", "bank"]
                (default: the package default, public holidays only).
        """
        self.country = country.upper()
        self._validate_country()

        country_info = COUNTRY_DATA[self.country]
        self.timezone = timezone or country_info["timezone"]
        self.lat = lat if lat is not None else country_info["lat"]
        self.lon = lon if lon is not None else country_info["lon"]

        self.holidays = holidays
        self.weekday = weekday
        self.hour = hour
        self.month = month
        self.daylight = daylight
        self.prefix = prefix
        self.freq = freq
        self.subdiv = self._normalise_subdiv(subdiv)
        self.categories = [categories] if isinstance(categories, str) else categories
        if self.categories is not None:
            self.categories = list(self.categories)

        self._validate_config()

    def _validate_country(self):
        if self.country not in COUNTRY_DATA:
            raise ValueError(f"Unsupported country: {self.country}. Supported: {list(COUNTRY_DATA.keys())}")

    def _normalise_subdiv(self, subdiv: Optional[str]) -> Optional[str]:
        if subdiv is None:
            return None
        code = str(subdiv).strip()
        if code.upper().startswith(f"{self.country}-"):
            code = code[len(self.country) + 1 :]
        known = getattr(type(holidays_lib.country_holidays(self.country)), "subdivisions", ())
        return code.upper() if code.upper() in known else code

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

        try:
            self._step = pd.Timedelta(to_offset(self.freq))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid freq {self.freq!r}: must be a fixed frequency such as '1h' or '15min'") from e
        hour, day = pd.Timedelta(hours=1), pd.Timedelta(days=1)
        divides_hour = self._step <= hour and hour % self._step == pd.Timedelta(0)
        whole_hours = (
            self._step >= hour and self._step % hour == pd.Timedelta(0) and day % self._step == pd.Timedelta(0)
        )
        if self._step <= pd.Timedelta(0) or not (divides_hour or whole_hours):
            raise ValueError(
                f"Invalid freq {self.freq!r}: must divide one hour (e.g. '15min') or be whole hours dividing a day"
            )

        cls = type(holidays_lib.country_holidays(self.country))
        if self.subdiv is not None:
            try:
                holidays_lib.country_holidays(self.country, subdiv=self.subdiv)
            except NotImplementedError as e:
                raise ValueError(
                    f"Invalid subdiv {self.subdiv!r} for {self.country}. Valid: {list(getattr(cls, 'subdivisions', ()))}"
                ) from e
        if self.categories is not None:
            supported = set(getattr(cls, "supported_categories", ()))
            unknown = [c for c in self.categories if c not in supported]
            if unknown or not self.categories:
                raise ValueError(f"Invalid categories {unknown} for {self.country}. Valid: {sorted(supported)}")

    def _holiday_language(self) -> Optional[str]:
        cls = type(holidays_lib.country_holidays(self.country))
        # Countries without translations (GB, IE) only have English names.
        return HOLIDAY_LANGUAGE if HOLIDAY_LANGUAGE in (getattr(cls, "supported_languages", ()) or ()) else None

    def _holiday_calendar(self, years: List[int]) -> holidays_lib.HolidayBase:
        kwargs = {"years": years, "subdiv": self.subdiv, "language": self._holiday_language()}
        if self.categories is not None:
            kwargs["categories"] = tuple(self.categories)
        if self.country == "SE":
            # holidays.Sweden lists every Sunday as a holiday unless told not to.
            return holidays_lib.Sweden(include_sundays=False, **kwargs)
        return holidays_lib.country_holidays(self.country, **kwargs)

    @staticmethod
    def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Rows on the `freq` grid in [start, end), indexed in UTC."""
        start = self._to_utc(start)
        end = self._to_utc(end)
        if end <= start:
            raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")

        index = pd.date_range(start.ceil(self._step), end, freq=self._step, inclusive="left")
        local_index = index.tz_convert(self.timezone)
        # Features that depend only on the local date are computed once per date and broadcast to its rows.
        codes, dates = pd.factorize(local_index.tz_localize(None).normalize())
        local_dates = pd.DatetimeIndex(dates)

        df = pd.DataFrame(index=index)

        if self.holidays:
            df = self._add_holidays(df, local_dates, codes)
        if self.weekday:
            df = self._add_weekday(df, local_index)
        if self.hour:
            df = self._add_hour(df, local_index)
        if self.month:
            df = self._add_month(df, local_index)
        if self.daylight:
            df = self._add_daylight(df, local_dates, codes)

        return df

    def _add_holidays(self, df: pd.DataFrame, local_dates: pd.DatetimeIndex, codes: np.ndarray) -> pd.DataFrame:
        years = list(range(local_dates.min().year, local_dates.max().year + 1)) if len(local_dates) else []
        calendar = self._holiday_calendar(years)
        names_per_date = [calendar.get_list(d) for d in local_dates.date]
        is_holiday = np.array([len(names) > 0 for names in names_per_date], dtype=int)

        if self.holidays in ("binary", "onehot"):
            df[f"{self.prefix}is_holiday"] = is_holiday[codes] if len(codes) else np.array([], dtype=int)
        if self.holidays == "name":
            joined = np.array(["; ".join(names) if names else None for names in names_per_date], dtype=object)
            df[f"{self.prefix}holiday_name"] = joined[codes] if len(codes) else np.array([], dtype=object)
        elif self.holidays == "onehot":
            # One column per holiday of the years covered, so the column set does not depend on the exact range.
            columns = sorted({_sanitise_name(n) for d in calendar for n in calendar.get_list(d)})
            position = {name: i for i, name in enumerate(columns)}
            per_date = np.zeros((len(local_dates), len(columns)), dtype=int)
            for i, names in enumerate(names_per_date):
                for name in names:
                    per_date[i, position[_sanitise_name(name)]] = 1
            dummies = pd.DataFrame(
                per_date[codes], index=df.index, columns=[f"{self.prefix}holiday_{c}" for c in columns]
            )
            df = pd.concat([df, dummies], axis=1)
        return df

    def _add_weekday(self, df: pd.DataFrame, local_index: pd.DatetimeIndex) -> pd.DataFrame:
        weekday_series = pd.Series(local_index.weekday, index=df.index)

        if self.weekday == "number":
            df[f"{self.prefix}weekday"] = weekday_series
        elif self.weekday == "name":
            df[f"{self.prefix}weekday_name"] = weekday_series.map(lambda x: WEEKDAY_NAMES[x])
        elif self.weekday == "onehot":
            for i, name in enumerate(WEEKDAY_NAMES):
                df[f"{self.prefix}is_{name}"] = (weekday_series == i).astype(int)
        return df

    def _add_hour(self, df: pd.DataFrame, local_index: pd.DatetimeIndex) -> pd.DataFrame:
        if self._step < pd.Timedelta(hours=1):
            # Sub-hourly: index of the period within the local day on the wall-clock grid (0-95 at 15 minutes).
            seconds = local_index.hour * 3600 + local_index.minute * 60 + local_index.second
            values = seconds // int(self._step.total_seconds())
            name, onehot_prefix = "period", "is_period_"
            n_values = int(pd.Timedelta(days=1) / self._step)
        else:
            values = local_index.hour
            name, onehot_prefix = "hour", "is_"
            n_values = 24
        series = pd.Series(np.asarray(values), index=df.index)

        if self.hour == "number":
            df[f"{self.prefix}{name}"] = series
        elif self.hour == "onehot":
            dummies = {f"{self.prefix}{onehot_prefix}{v}": (series == v).astype(int) for v in range(n_values)}
            df = pd.concat([df, pd.DataFrame(dummies, index=df.index)], axis=1)
        return df

    def _add_month(self, df: pd.DataFrame, local_index: pd.DatetimeIndex) -> pd.DataFrame:
        month_series = pd.Series(local_index.month, index=df.index)

        if self.month == "number":
            df[f"{self.prefix}month"] = month_series
        elif self.month == "name":
            df[f"{self.prefix}month_name"] = month_series.map(lambda x: MONTH_NAMES[x - 1])
        elif self.month == "onehot":
            for i, name in enumerate(MONTH_NAMES, start=1):
                df[f"{self.prefix}is_{name}"] = (month_series == i).astype(int)
        return df

    def _add_daylight(self, df: pd.DataFrame, local_dates: pd.DatetimeIndex, codes: np.ndarray) -> pd.DataFrame:
        location = LocationInfo(latitude=self.lat, longitude=self.lon, timezone=self.timezone)

        per_date = np.full((len(local_dates), 3), np.nan)
        missing = []
        for i, date in enumerate(local_dates.date):
            try:
                s = sun(location.observer, date=date, tzinfo=location.timezone)
            except ValueError:  # polar day or night: the sun never rises or never sets
                missing.append(date)
                continue
            sunrise, sunset = s["sunrise"], s["sunset"]
            per_date[i] = (
                sunrise.hour + sunrise.minute / 60,
                sunset.hour + sunset.minute / 60,
                (sunset - sunrise).total_seconds() / 3600,
            )
        if missing:
            logger.warning(
                f"CalendarSource: no sunrise or sunset at ({self.lat}, {self.lon}) on {len(missing)} date(s) "
                f"from {missing[0]} to {missing[-1]}; daylight columns are NaN there"
            )

        values = per_date[codes] if len(codes) else np.empty((0, 3))
        df[f"{self.prefix}sunrise"] = values[:, 0]
        df[f"{self.prefix}sunset"] = values[:, 1]
        df[f"{self.prefix}daylight_hours"] = values[:, 2]
        return df

    def get_cache_config(self) -> Optional[dict]:
        return None
