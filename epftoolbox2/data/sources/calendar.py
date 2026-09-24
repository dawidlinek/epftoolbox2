import re
import unicodedata
from typing import Dict, List, Optional, Union

import holidays as holidays_lib
import numpy as np
import pandas as pd
import requests
from astral import LocationInfo
from astral.sun import sun
from pandas.tseries.frequencies import to_offset

from epftoolbox2.logging import get_logger
from . import _openholidays
from .._market_timezone import resolve as _resolve_timezone
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

# Market-structure changes used by `regimes=True`. "date" is the first delivery day (local date); a regime dummy is 1
# from that date on, for the listed countries only.
DEFAULT_REGIMES = {
    "de_at_split": {
        "date": "2018-10-01",
        "countries": ["DE", "AT"],
        "description": "Split of the German-Austrian bidding zone (DE-LU and AT)",
    },
    "brexit_decoupling": {
        "date": "2021-01-01",
        "countries": ["GB", "IE"],
        "description": "GB and the Irish SEM leave implicit day-ahead coupling with SDAC",
    },
    "core_fb": {
        "date": "2022-06-09",
        "countries": ["AT", "BE", "CZ", "DE", "FR", "HU", "NL", "PL", "RO", "SK"],
        "description": "Core flow-based day-ahead market coupling (first auction 2022-06-08)",
    },
    "nordic_fb": {
        "date": "2024-10-30",
        "countries": ["DK", "FI", "NO", "SE"],
        "description": "Nordic flow-based day-ahead market coupling (first auction 2024-10-29)",
    },
    "sdac_15min_mtu": {
        "date": "2025-10-01",
        "countries": "AT BE BG CZ DE DK ES FI FR GR HU IT NL NO PL PT RO SE SK".split(),
        "description": "15-minute market time unit in SDAC (first auction 2025-09-30)",
    },
    "sem_30min_mtu": {
        "date": "2025-10-01",
        "countries": ["IE"],
        "description": "30-minute market time unit in the Irish SEM day-ahead market",
    },
}


def _sanitise_name(name: str) -> str:
    """'New Year's Day' -> 'new_years_day': ASCII, lower case, words joined by underscores."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    ascii_name = ascii_name.replace("'", "")
    return re.sub(r"[^0-9a-z]+", "_", ascii_name.lower()).strip("_") or "unnamed"


def _grid_step(freq: str) -> pd.Timedelta:
    """Row spacing of a calendar grid: a divisor of one hour, or whole hours dividing a day."""
    try:
        step = pd.Timedelta(to_offset(freq))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid freq {freq!r}: must be a fixed frequency such as '1h' or '15min'") from e
    hour, day, zero = pd.Timedelta(hours=1), pd.Timedelta(days=1), pd.Timedelta(0)
    divides_hour = zero < step <= hour and hour % step == zero
    whole_hours = step >= hour and step % hour == zero and day % step == zero
    if not (divides_hour or whole_hours):
        raise ValueError(f"Invalid freq {freq!r}: must divide one hour (e.g. '15min') or be whole hours dividing a day")
    return step


def _local_grid(start: pd.Timestamp, end: pd.Timestamp, step: pd.Timedelta, timezone: str):
    """UTC grid over [start, end), its local-time view, and the local date of each row (codes into dates)."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
    end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
    if end <= start:
        raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")
    index = pd.date_range(start.ceil(step), end, freq=step, inclusive="left")
    local_index = index.tz_convert(timezone)
    codes, dates = pd.factorize(local_index.tz_localize(None).normalize())
    return index, local_index, codes, pd.DatetimeIndex(dates)


class CalendarSource(DataSource):
    """Calendar features on a UTC grid, computed from local dates and local wall-clock times.

    Every feature is known in advance, so its `_d+k` versions are legal predictors.

    With `school_holidays=True`, it also downloads school holiday flags from the OpenHolidays API
    (https://www.openholidaysapi.org, ODbL 1.0 licence: credit it when you publish results or share a dataset
    built from it, e.g. `epftoolbox2.data.sources._openholidays.ATTRIBUTION`). This is the only case where
    CalendarSource makes network calls; without it, it stays offline and deterministic. Values are NaN before
    the API's history starts for the country, or for school years not yet published, and a warning names the
    affected dates; entries marked as half days are ignored (not counted as a holiday).

    Example:
        >>> source = CalendarSource(country="DE", subdiv="BY", holidays="binary", weekday="onehot")
        >>> df = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))

    Example:
        >>> source = CalendarSource(
        ...     country="PL", school_holidays=True, school_weights={"PL-MZ": 5.5, "PL-SL": 4.4, "PL-WP": 3.5}
        ... )
        >>> # columns include: is_school_holiday_pl_mz, is_school_holiday_pl_sl, is_school_holiday_pl_wp,
        >>> #                  school_holiday_share
    """

    # Set by DataPipeline._bind_component_timezones() when `timezone` is left unset; never written by users.
    _pipeline_timezone: Optional[str] = None

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
        bridge_days: bool = False,
        christmas_period: bool = False,
        holiday_distance: int = 0,
        dst_switch: bool = False,
        fourier_weekly: int = 0,
        fourier_annual: int = 0,
        regimes: Union[bool, Dict[str, str]] = False,
        school_holidays: bool = False,
        school_subdivisions: Optional[Union[str, List[str]]] = None,
        school_weights: Optional[Union[Dict[str, float], List[float], str]] = None,
    ):
        """
        Args:
            country: ISO country code (see COUNTRY_DATA).
            timezone: Time zone whose local dates and wall-clock times define the features. None (default)
                inherits `DataPipeline.timezone` when this source is part of a pipeline; standalone, it falls
                back to the country's own zone.
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
            bridge_days: Add `is_bridge_day`: a working day whose previous and next days are both a weekend
                day or a holiday (e.g. the Friday after a Thursday holiday).
            christmas_period: Add `is_christmas_period`: 24 December to 1 January inclusive.
            holiday_distance: If N > 0, add `days_to_holiday` and `days_since_holiday` (0 on a holiday), capped at N.
            dst_switch: Add `is_dst_switch_day`: local days with 23 or 25 hours.
            fourier_weekly: Number K of weekly harmonics: `week_sin_k`, `week_cos_k` for k = 1..K.
            fourier_annual: Number K of annual harmonics: `year_sin_k`, `year_cos_k` for k = 1..K.
            regimes: Step dummies `regime_{name}` (1 from the local date on). True uses DEFAULT_REGIMES for the
                country; a dict {name: "YYYY-MM-DD"} replaces them.
            school_holidays: Download school holiday flags from the OpenHolidays API (network calls happen only
                when this is True). No or one `school_subdivisions` entry: `is_school_holiday`; several: one
                `is_school_holiday_{code}` per entry, e.g. `is_school_holiday_pl_mz`.
            school_subdivisions: OpenHolidays subdivision codes, e.g. ["PL-MZ", "PL-SL"] or "MZ,SL" (the
                country prefix is optional). None: only holidays that apply to the whole country.
            school_weights: Weight per subdivision (e.g. population) for `school_holiday_share`: a dict
                {code: weight}, or numbers in the order of `school_subdivisions`. With a dict,
                `school_subdivisions` defaults to its keys.
        """
        self.country = country.upper()
        self._validate_country()

        country_info = COUNTRY_DATA[self.country]
        self.timezone = timezone
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
        self.bridge_days = bridge_days
        self.christmas_period = christmas_period
        self.holiday_distance = self._validate_count("holiday_distance", holiday_distance, maximum=366)
        self.dst_switch = dst_switch
        self.fourier_weekly = self._validate_count("fourier_weekly", fourier_weekly)
        self.fourier_annual = self._validate_count("fourier_annual", fourier_annual)
        self.regimes = self._normalise_regimes(regimes)

        self.school_holidays = school_holidays
        if school_subdivisions is None and isinstance(school_weights, dict):
            school_subdivisions = list(school_weights)
        if isinstance(school_subdivisions, str):
            school_subdivisions = [s for s in school_subdivisions.split(",") if s.strip()]
        self.school_subdivisions = (
            [self._normalise_school_code(s) for s in school_subdivisions] if school_subdivisions is not None else None
        )
        self.school_weights = self._normalise_school_weights(school_weights)

        self.session = requests.Session()

        self._validate_config()

    def _normalise_school_code(self, code: str) -> str:
        code = str(code).strip().upper()
        return code if code.startswith(f"{self.country}-") else f"{self.country}-{code}"

    def _normalise_school_weights(self, weights) -> Optional[Dict[str, float]]:
        if weights is None:
            return None
        if isinstance(weights, str):
            weights = [w for w in weights.split(",") if w.strip()]
        if isinstance(weights, dict):
            pairs = [(self._normalise_school_code(k), v) for k, v in weights.items()]
        else:
            if self.school_subdivisions is None or len(weights) != len(self.school_subdivisions):
                raise ValueError("school_weights given as a list need one number per entry of `school_subdivisions`")
            pairs = list(zip(self.school_subdivisions, weights))
        try:
            return {code: float(value) for code, value in pairs}
        except (TypeError, ValueError) as e:
            raise ValueError(f"school_weights must be numbers, got {weights!r}") from e

    @property
    def market_timezone(self) -> str:
        """Resolved zone: `timezone` if set, else the one a `DataPipeline` inherited it from, else the
        country's own zone (COUNTRY_DATA) for standalone use."""
        return _resolve_timezone(self.timezone, self) or COUNTRY_DATA[self.country]["timezone"]

    @staticmethod
    def _validate_count(name: str, value, maximum: Optional[int] = None) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        if value < 0 or (maximum is not None and value > maximum):
            raise ValueError(f"{name} must be between 0 and {maximum if maximum is not None else 'inf'}, got {value}")
        return int(value)

    @staticmethod
    def _normalise_regimes(regimes) -> Union[bool, Dict[str, str]]:
        if isinstance(regimes, bool):
            return regimes
        if not isinstance(regimes, dict) or not regimes:
            raise ValueError(f"regimes must be True, False or a non-empty dict {{name: date}}, got {regimes!r}")
        normalised = {}
        for name, date in regimes.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name):
                raise ValueError(f"regimes: name {name!r} must contain only letters, digits and underscores")
            try:
                normalised[name] = pd.Timestamp(date).date().isoformat()
            except (ValueError, TypeError) as e:
                raise ValueError(f"regimes: {name!r} has an invalid date {date!r}") from e
        return normalised

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

        self._step = _grid_step(self.freq)

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

        for name in ("bridge_days", "christmas_period", "dst_switch", "school_holidays"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False, got {getattr(self, name)!r}")

        if self.school_subdivisions is not None:
            if not self.school_subdivisions:
                raise ValueError("school_subdivisions must not be empty; use None for country-wide school holidays")
            if len(set(self.school_subdivisions)) != len(self.school_subdivisions):
                raise ValueError(f"school_subdivisions contains duplicates: {self.school_subdivisions}")
        if self.school_weights is not None:
            if self.school_subdivisions is None or set(self.school_weights) != set(self.school_subdivisions):
                raise ValueError(
                    "school_weights must have exactly one entry per subdivision: "
                    f"{sorted(self.school_weights)} vs {self.school_subdivisions}"
                )
            if not all(np.isfinite(w) and w > 0 for w in self.school_weights.values()):
                raise ValueError(f"school_weights must be positive numbers, got {self.school_weights}")

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

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Rows on the `freq` grid in [start, end), indexed in UTC."""
        # Features that depend only on the local date are computed once per date and broadcast to its rows.
        index, local_index, codes, local_dates = _local_grid(start, end, self._step, self.market_timezone)

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
        if self.bridge_days or self.holiday_distance:
            df = self._add_holiday_context(df, local_dates, codes)
        if self.christmas_period:
            christmas = ((local_dates.month == 12) & (local_dates.day >= 24)) | (
                (local_dates.month == 1) & (local_dates.day == 1)
            )
            df[f"{self.prefix}is_christmas_period"] = np.asarray(christmas, dtype=int)[codes]
        if self.dst_switch:
            df = self._add_dst_switch(df, local_dates, codes)
        if self.fourier_weekly or self.fourier_annual:
            df = self._add_fourier(df, local_index)
        if self.regimes:
            for name, date in self._resolved_regimes().items():
                df[f"{self.prefix}regime_{name}"] = np.asarray(local_dates >= pd.Timestamp(date), dtype=int)[codes]
        if self.school_holidays:
            df = self._add_school_holidays(df, local_dates, codes)

        return df

    def _holiday_flags(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """True where the local date is a holiday of the configured calendar (country, subdiv, categories)."""
        if not len(dates):
            return np.zeros(0, dtype=bool)
        calendar = self._holiday_calendar(list(range(dates.min().year, dates.max().year + 1)))
        return np.array([date in calendar for date in dates.date], dtype=bool)

    def _add_holiday_context(self, df: pd.DataFrame, local_dates: pd.DatetimeIndex, codes: np.ndarray) -> pd.DataFrame:
        # Look far enough beyond the range that neighbours and the nearest holidays of its edge days are known.
        margin = pd.Timedelta(days=max(1, self.holiday_distance))
        if len(local_dates):
            span = pd.date_range(local_dates.min() - margin, local_dates.max() + margin, freq="D")
        else:
            span = pd.DatetimeIndex([])
        holiday = self._holiday_flags(span)
        rows = span.get_indexer(local_dates)[codes]

        if self.bridge_days:
            day_off = holiday | (span.dayofweek >= 5)
            bridge = np.zeros(len(span), dtype=bool)
            bridge[1:-1] = ~day_off[1:-1] & day_off[:-2] & day_off[2:]
            df[f"{self.prefix}is_bridge_day"] = bridge.astype(int)[rows]

        if self.holiday_distance:
            cap = self.holiday_distance
            positions = np.arange(len(span))
            holidays_at = np.flatnonzero(holiday)
            to_next = np.full(len(span), cap)
            since_last = np.full(len(span), cap)
            if len(holidays_at):
                nxt = np.searchsorted(holidays_at, positions, side="left")
                has_next = nxt < len(holidays_at)
                to_next[has_next] = holidays_at[nxt[has_next]] - positions[has_next]
                prv = np.searchsorted(holidays_at, positions, side="right") - 1
                has_prev = prv >= 0
                since_last[has_prev] = positions[has_prev] - holidays_at[prv[has_prev]]
            df[f"{self.prefix}days_to_holiday"] = np.minimum(to_next, cap)[rows]
            df[f"{self.prefix}days_since_holiday"] = np.minimum(since_last, cap)[rows]
        return df

    def _add_dst_switch(self, df: pd.DataFrame, local_dates: pd.DatetimeIndex, codes: np.ndarray) -> pd.DataFrame:
        # A local day has 23 or 25 hours when its UTC offset at noon differs from the previous day's.
        def utc_offset(naive: pd.DatetimeIndex) -> np.ndarray:
            aware = naive.tz_localize(
                self.market_timezone, ambiguous=np.zeros(len(naive), dtype=bool), nonexistent="shift_forward"
            )
            return np.asarray(aware.tz_localize(None) - aware.tz_convert("UTC").tz_localize(None))

        noon = local_dates + pd.Timedelta(hours=12)
        switch = utc_offset(noon) != utc_offset(noon - pd.Timedelta(days=1))
        df[f"{self.prefix}is_dst_switch_day"] = switch.astype(int)[codes]
        return df

    def _add_fourier(self, df: pd.DataFrame, local_index: pd.DatetimeIndex) -> pd.DataFrame:
        # Phases use the local wall clock: the repeated fall-back hour gets the same value twice, and nothing
        # jumps or drifts when the UTC offset changes.
        day_fraction = np.asarray(local_index.hour * 3600 + local_index.minute * 60 + local_index.second) / 86400
        terms = {}
        if self.fourier_weekly:
            phase = (np.asarray(local_index.dayofweek) + day_fraction) / 7
            for k in range(1, self.fourier_weekly + 1):
                terms[f"{self.prefix}week_sin_{k}"] = np.sin(2 * np.pi * k * phase)
                terms[f"{self.prefix}week_cos_{k}"] = np.cos(2 * np.pi * k * phase)
        if self.fourier_annual:
            days_in_year = np.where(np.asarray(local_index.is_leap_year), 366, 365)
            phase = (np.asarray(local_index.dayofyear) - 1 + day_fraction) / days_in_year
            for k in range(1, self.fourier_annual + 1):
                terms[f"{self.prefix}year_sin_{k}"] = np.sin(2 * np.pi * k * phase)
                terms[f"{self.prefix}year_cos_{k}"] = np.cos(2 * np.pi * k * phase)
        return pd.concat([df, pd.DataFrame(terms, index=df.index)], axis=1)

    def _resolved_regimes(self) -> Dict[str, str]:
        if self.regimes is True:
            return {name: spec["date"] for name, spec in DEFAULT_REGIMES.items() if self.country in spec["countries"]}
        return dict(self.regimes) if self.regimes else {}

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
        location = LocationInfo(latitude=self.lat, longitude=self.lon, timezone=self.market_timezone)

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

    def _add_school_holidays(self, df: pd.DataFrame, local_dates: pd.DatetimeIndex, codes: np.ndarray) -> pd.DataFrame:
        """School holiday flags (and, with `school_weights`, their weighted share) from the OpenHolidays API."""
        targets = self.school_subdivisions or [None]
        if len(targets) == 1:
            columns = {targets[0]: f"{self.prefix}is_school_holiday"}
        else:
            columns = {t: f"{self.prefix}is_school_holiday_{t.lower().replace('-', '_')}" for t in targets}
        if not len(local_dates):
            for column in columns.values():
                df[column] = np.array([], dtype=float)
            if self.school_weights is not None:
                df[f"{self.prefix}school_holiday_share"] = np.array([], dtype=float)
            return df

        if self.school_subdivisions is not None:
            known = _openholidays.known_subdivisions(self.session, self.country)
            unknown = [s for s in self.school_subdivisions if s not in known]
            if unknown:
                raise ValueError(f"Unknown subdivisions for {self.country}: {unknown}. Known: {sorted(known)}")

        margin = pd.Timedelta(days=_openholidays.QUERY_MARGIN_DAYS)
        days = pd.date_range(local_dates.min() - margin, local_dates.max() + margin, freq="D")
        entries = _openholidays.fetch_entries(self.session, self.country, days[0], days[-1])
        logger.info(
            f"OpenHolidays [{self.country}]: {len(entries)} school holiday periods. {_openholidays.ATTRIBUTION}"
        )

        positions = days.get_indexer(local_dates)
        for target, column in columns.items():
            per_date = _openholidays.daily_flags(entries, target, days)[positions]
            df[column] = per_date[codes]
            if np.isnan(per_date).any():
                self._warn_missing_school_holidays(target, column, local_dates[np.isnan(per_date)])

        if self.school_weights is not None:
            weights = np.array([self.school_weights[t] for t in targets])
            flags = df[[columns[t] for t in targets]].to_numpy()
            # A NaN flag makes the share NaN: a partial share would look like fewer regions on holiday.
            df[f"{self.prefix}school_holiday_share"] = flags @ weights / weights.sum()
        return df

    def _warn_missing_school_holidays(self, target: Optional[str], column: str, missing: pd.DatetimeIndex) -> None:
        hint = "" if target is not None else "; pass `school_subdivisions` for regional school holidays"
        logger.warning(
            f"OpenHolidays [{target or self.country}]: no school holiday data for {len(missing)} local date(s) "
            f"from {missing.min().date()} to {missing.max().date()}; {column} is NaN there{hint}"
        )

    def get_cache_config(self) -> Optional[dict]:
        return None
