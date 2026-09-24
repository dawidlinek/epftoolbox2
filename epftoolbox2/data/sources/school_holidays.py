"""School holidays from the OpenHolidays API.

Data: OpenHolidays API (https://www.openholidaysapi.org), made available under the Open Database License
(ODbL 1.0, https://opendatacommons.org/licenses/odbl/1-0/). Publications and derived databases must credit it.
"""

import time
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests

from epftoolbox2.logging import get_logger
from .base import DataSource
from .calendar import COUNTRY_DATA, _grid_step, _local_grid

ATTRIBUTION = "School holidays: OpenHolidays API (https://www.openholidaysapi.org), ODbL 1.0"

# The API rejects requests spanning more than 1095 days.
MAX_DAYS_PER_REQUEST = 1000
# Every school year has holidays, so a stretch of more than a year without any is missing data, not a year of
# school. The query reaches further than that beyond the requested range, so each requested date can be judged.
MAX_GAP_DAYS = 366
QUERY_MARGIN_DAYS = 400


class SchoolHolidaysSource(DataSource):
    """
    School holiday flags from the OpenHolidays API, on a UTC grid built from local dates.

    Output (the prefix is used as is):
        - no or one subdivision: `is_school_holiday`
        - several subdivisions: `is_school_holiday_{code}` per subdivision, e.g. `is_school_holiday_pl_mz`
        - with `weights`: `school_holiday_share`, the weighted share of the subdivisions on school holidays

    Values are 1.0/0.0, and NaN on dates the API has no data for (before its history starts, or school years
    not yet published); a warning names those dates.

    Example:
        >>> source = SchoolHolidaysSource(country="PL", weights={"PL-MZ": 5.5, "PL-SL": 4.4, "PL-WP": 3.5})
        >>> df = source.fetch(pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))
    """

    API_URL = "https://openholidaysapi.org"

    def __init__(
        self,
        country: str,
        subdivisions: Optional[Union[str, List[str]]] = None,
        weights: Optional[Union[Dict[str, float], List[float], str]] = None,
        timezone: Optional[str] = None,
        prefix: str = "",
        freq: str = "1h",
    ):
        """
        Args:
            country: ISO 3166-1 country code, e.g. "PL".
            subdivisions: Subdivision codes as used by OpenHolidays, e.g. ["PL-MZ", "PL-SL"] or "MZ,SL"
                (the country prefix is optional). None: only holidays that apply to the whole country.
            weights: Weight per subdivision (e.g. population) for `school_holiday_share`: a dict {code: weight},
                or numbers in the order of `subdivisions`. With a dict, `subdivisions` defaults to its keys.
            timezone: Time zone whose local dates define the flags (default: the country's, see COUNTRY_DATA).
            prefix: Prefix for column names (no separator is added).
            freq: Row spacing, as for CalendarSource.
        """
        self.country = str(country).upper()
        if subdivisions is None and isinstance(weights, dict):
            subdivisions = list(weights)
        if isinstance(subdivisions, str):
            subdivisions = [s for s in subdivisions.split(",") if s.strip()]
        self.subdivisions = [self._normalise_code(s) for s in subdivisions] if subdivisions is not None else None
        self.weights = self._normalise_weights(weights)
        self.timezone = timezone or COUNTRY_DATA.get(self.country, {}).get("timezone")
        self.prefix = prefix
        self.freq = freq

        self.session = requests.Session()
        self.logger = get_logger(__name__)

        self._validate_config()

    def _normalise_code(self, code: str) -> str:
        code = str(code).strip().upper()
        return code if code.startswith(f"{self.country}-") else f"{self.country}-{code}"

    def _normalise_weights(self, weights) -> Optional[Dict[str, float]]:
        if weights is None:
            return None
        if isinstance(weights, str):
            weights = [w for w in weights.split(",") if w.strip()]
        if isinstance(weights, dict):
            pairs = [(self._normalise_code(k), v) for k, v in weights.items()]
        else:
            if self.subdivisions is None or len(weights) != len(self.subdivisions):
                raise ValueError("weights given as a list need one number per entry of `subdivisions`")
            pairs = list(zip(self.subdivisions, weights))
        try:
            return {code: float(value) for code, value in pairs}
        except (TypeError, ValueError) as e:
            raise ValueError(f"weights must be numbers, got {weights!r}") from e

    def _validate_config(self) -> bool:
        if len(self.country) != 2 or not self.country.isalpha():
            raise ValueError(f"country must be an ISO 3166-1 alpha-2 code, got {self.country!r}")
        if self.timezone is None:
            raise ValueError(f"No default time zone for {self.country}; pass `timezone`")
        try:
            pd.Timestamp("2000-01-01").tz_localize(self.timezone)
        except Exception as e:
            raise ValueError(f"Unknown timezone: {self.timezone!r}") from e
        self._step = _grid_step(self.freq)

        if self.subdivisions is not None:
            if not self.subdivisions:
                raise ValueError("subdivisions must not be empty; use None for country-wide holidays")
            if len(set(self.subdivisions)) != len(self.subdivisions):
                raise ValueError(f"subdivisions contains duplicates: {self.subdivisions}")
        if self.weights is not None:
            if self.subdivisions is None or set(self.weights) != set(self.subdivisions):
                raise ValueError(
                    f"weights must have exactly one entry per subdivision: {sorted(self.weights)} vs {self.subdivisions}"
                )
            if not all(np.isfinite(w) and w > 0 for w in self.weights.values()):
                raise ValueError(f"weights must be positive numbers, got {self.weights}")
        return True

    def _get(self, path: str, params: dict) -> list:
        error: Exception = RuntimeError("no attempt made")
        for attempt in range(3):
            try:
                response = self.session.get(
                    f"{self.API_URL}/{path}", params=params, headers={"accept": "text/json"}, timeout=30
                )
            except requests.RequestException as e:
                error = e
            else:
                if response.status_code < 500:
                    response.raise_for_status()
                    return response.json()
                error = requests.HTTPError(f"HTTP {response.status_code}")
            if attempt < 2:
                time.sleep(2**attempt)
        raise RuntimeError(f"OpenHolidays API request to {path} failed after 3 attempts: {error}") from error

    def _known_subdivisions(self) -> set:
        known = set()
        stack = list(self._get("Subdivisions", {"countryIsoCode": self.country}))
        while stack:
            item = stack.pop()
            known.add(item["code"].upper())
            stack.extend(item.get("children") or [])
        return known

    def _fetch_entries(self, first: pd.Timestamp, last: pd.Timestamp) -> list:
        entries = {}
        chunk_start = first
        while chunk_start <= last:
            chunk_end = min(chunk_start + pd.Timedelta(days=MAX_DAYS_PER_REQUEST - 1), last)
            params = {
                "countryIsoCode": self.country,
                "languageIsoCode": "EN",
                "validFrom": chunk_start.date().isoformat(),
                "validTo": chunk_end.date().isoformat(),
            }
            for entry in self._get("SchoolHolidays", params):
                # Holidays that span a chunk boundary are returned twice.
                key = entry.get("id") or (entry["startDate"], entry["endDate"], str(entry.get("subdivisions")))
                entries[key] = entry
            chunk_start = chunk_end + pd.Timedelta(days=1)
        return list(entries.values())

    @staticmethod
    def _applies(entry: dict, target: Optional[str]) -> bool:
        codes = [s["code"].upper() for s in entry.get("subdivisions") or []]
        if entry.get("nationwide") or not codes:
            return True
        if target is None:
            return entry.get("regionalScope") == "National"
        # A holiday of a subdivision also applies to its children (e.g. DE-BY to DE-BY-AU), not the other way round.
        return any(target == code or target.startswith(f"{code}-") for code in codes)

    def _daily_flags(self, entries: list, target: Optional[str], days: pd.DatetimeIndex) -> np.ndarray:
        """1/0 per day in `days` (consecutive local dates), NaN where the data has a gap of more than a year."""
        intervals = sorted(
            (pd.Timestamp(e["startDate"]), pd.Timestamp(e["endDate"]))
            for e in entries
            if self._applies(e, target) and e.get("temporalScope", "FullDay") != "HalfDay"
        )
        flags = np.zeros(len(days))
        covered = np.ones(len(days), dtype=bool)
        previous_end = days[0] - pd.Timedelta(days=1)
        for start, end in intervals + [(days[-1] + pd.Timedelta(days=1), None)]:
            if (start - previous_end).days > MAX_GAP_DAYS:
                covered[(days > previous_end) & (days < start)] = False
            if end is not None:
                flags[days.searchsorted(start, "left") : days.searchsorted(end, "right")] = 1.0
                previous_end = max(previous_end, end)
        flags[~covered] = np.nan
        return flags

    def _warn_missing(self, target: Optional[str], column: str, missing: pd.DatetimeIndex) -> None:
        hint = "" if target is not None else "; pass `subdivisions` for regional school holidays"
        self.logger.warning(
            f"OpenHolidays [{target or self.country}]: no school holiday data for {len(missing)} local date(s) "
            f"from {missing.min().date()} to {missing.max().date()}; {column} is NaN there{hint}"
        )

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Rows on the `freq` grid in [start, end), indexed in UTC."""
        index, _, codes, local_dates = _local_grid(start, end, self._step, self.timezone)
        df = pd.DataFrame(index=index)
        targets = self.subdivisions or [None]
        if len(targets) == 1:
            columns = {targets[0]: f"{self.prefix}is_school_holiday"}
        else:
            columns = {t: f"{self.prefix}is_school_holiday_{t.lower().replace('-', '_')}" for t in targets}
        if not len(local_dates):
            for column in columns.values():
                df[column] = np.array([], dtype=float)
            if self.weights is not None:
                df[f"{self.prefix}school_holiday_share"] = np.array([], dtype=float)
            return df

        if self.subdivisions is not None:
            known = self._known_subdivisions()
            unknown = [s for s in self.subdivisions if s not in known]
            if unknown:
                raise ValueError(f"Unknown subdivisions for {self.country}: {unknown}. Known: {sorted(known)}")

        margin = pd.Timedelta(days=QUERY_MARGIN_DAYS)
        days = pd.date_range(local_dates.min() - margin, local_dates.max() + margin, freq="D")
        entries = self._fetch_entries(days[0], days[-1])
        self.logger.info(f"OpenHolidays [{self.country}]: {len(entries)} school holiday periods. {ATTRIBUTION}")

        positions = days.get_indexer(local_dates)
        for target, column in columns.items():
            per_date = self._daily_flags(entries, target, days)[positions]
            df[column] = per_date[codes]
            if np.isnan(per_date).any():
                self._warn_missing(target, column, local_dates[np.isnan(per_date)])

        if self.weights is not None:
            weights = np.array([self.weights[t] for t in targets])
            flags = df[[columns[t] for t in targets]].to_numpy()
            # A NaN flag makes the share NaN: a partial share would look like fewer regions on holiday.
            df[f"{self.prefix}school_holiday_share"] = flags @ weights / weights.sum()
        return df

    def get_cache_config(self) -> Optional[dict]:
        # Not cached: a run makes a few small requests, and a cache would keep "no data" (NaN) for school years
        # published after it was written.
        return None
