"""Shared pieces of the Open-Meteo sources: issue-time helpers, HTTP retries, coverage checks, derived columns,
several locations combined into one source."""

import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from rich.console import Console

from epftoolbox2.logging import get_logger
from .._market_timezone import resolve as _resolve_timezone
from .base import DataSource

_ISSUE_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_LOCATION_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
# Variables that cannot be averaged arithmetically when combining several locations (see `_weighted_average`).
_CIRCULAR_VARIABLE_RE = re.compile(r"direction")
_CATEGORICAL_VARIABLES = ("weather_code",)

# Attributes every OpenMeteoBase subclass sets at construction time that must not be forwarded when building one
# instance per location (see `OpenMeteoBase._location_kwargs`): runtime objects, and the parameters `locations`
# itself replaces.
_RUNTIME_ATTRIBUTES = {"session", "console", "logger"}
_NON_LOCATION_KWARGS = {"latitude", "longitude", "locations", "prefix", "keep_locations"}

# Derived columns: name -> input variable. All are deterministic transforms of one forecast column, computed per
# lead and per location, so they inherit that column's vintage.
DERIVED_INPUTS = {
    "hdd": "temperature_2m",
    "cdd": "temperature_2m",
    "wind_speed_100m_cubed": "wind_speed_100m",
}


def parse_issue_time(value: str) -> Tuple[int, int]:
    """Parse "HH:MM" (local wall-clock time) into (hour, minute)."""
    match = _ISSUE_TIME_RE.match(value) if isinstance(value, str) else None
    if not match or int(match.group(1)) > 23 or int(match.group(2)) > 59:
        raise ValueError(f'issue_time must be "HH:MM" between "00:00" and "23:59", got {value!r}')
    return int(match.group(1)), int(match.group(2))


def local_issue_times(rows: pd.DatetimeIndex, timezone: str, issue_time: str) -> pd.DatetimeIndex:
    """UTC instant of the issue time on the local calendar day of each row.

    An issue time that falls into a DST gap or overlap resolves to the earlier instant (the conservative choice).
    """
    hour, minute = parse_issue_time(issue_time)
    local_days = rows.tz_convert(timezone).tz_localize(None).normalize()
    issue = local_days + pd.Timedelta(hours=hour, minutes=minute)
    earlier = np.ones(len(issue), dtype=bool)
    return issue.tz_localize(timezone, ambiguous=earlier, nonexistent="shift_backward").tz_convert("UTC")


def shift_local_days(rows: pd.DatetimeIndex, timezone: str, days: int) -> pd.DatetimeIndex:
    """Same local wall-clock time `days` calendar days later, in UTC.

    A time that does not exist on the target day (spring forward) moves to the next valid hour; an ambiguous one
    (fall back) takes its first occurrence.
    """
    shifted = rows.tz_convert(timezone).tz_localize(None) + pd.Timedelta(days=days)
    first = np.ones(len(shifted), dtype=bool)
    return shifted.tz_localize(timezone, ambiguous=first, nonexistent="shift_forward").tz_convert("UTC")


def add_derived_columns(
    df: pd.DataFrame, derived: Sequence[str], horizon: int, hdd_base_c: float, cdd_base_c: float
) -> pd.DataFrame:
    """Append `{name}_d+{i}` for each derived name, computed from the unprefixed `{input}_d+{i}` columns.

    hdd/cdd are hourly heating/cooling degrees (degree-hours per hour) against the given base temperature;
    wind_speed_100m_cubed is (wind speed in m/s)^3, a proxy for wind power below rated speed.
    """
    new = {}
    for name in derived:
        for lead in range(1, horizon + 1):
            source = df[f"{DERIVED_INPUTS[name]}_d+{lead}"]
            if name == "hdd":
                new[f"{name}_d+{lead}"] = (hdd_base_c - source).clip(lower=0)
            elif name == "cdd":
                new[f"{name}_d+{lead}"] = (source - cdd_base_c).clip(lower=0)
            else:
                new[f"{name}_d+{lead}"] = (source / 3.6) ** 3
    if not new:
        return df
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def parse_hourly(response_data: Dict) -> pd.DataFrame:
    """Open-Meteo JSON → DataFrame indexed by UTC time with one float column per hourly key (nulls → NaN)."""
    hourly = response_data.get("hourly") if isinstance(response_data, dict) else None
    if not hourly or not hourly.get("time"):
        return pd.DataFrame()

    index = pd.DatetimeIndex(pd.to_datetime(hourly["time"])).tz_localize("UTC")
    n = len(index)
    columns = {}
    for key, values in hourly.items():
        if key == "time":
            continue
        array = np.full(n, np.nan)
        values = [np.nan if v is None else v for v in values[:n]]
        array[: len(values)] = np.asarray(values, dtype=float)
        columns[key] = array
    table = pd.DataFrame(columns, index=index)
    table.index.name = "datetime"
    return table


# ---------------------------------------------------------------------------
# Locations: several coordinates combined into one source (see OpenMeteoBase.locations)
# ---------------------------------------------------------------------------


def parse_locations(locations: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalise `locations` to [{"name", "latitude", "longitude", "weight"}, ...] (weight is None if not given).

    Accepts a dict ``{name: {"latitude", "longitude", "weight"?}}``, a dict ``{name: (latitude, longitude)}`` or
    ``{name: (latitude, longitude, weight)}`` (list or tuple), or the compact string
    ``"name:latitude:longitude[:weight], ..."`` used by YAML and the TUI. Names must use letters, digits and
    underscores and be unique. Weights, when given, must be positive numbers, and must be given for every
    location or for none — a silent mix would leave some locations out of the weighted average while still
    fetching them.
    """
    if isinstance(locations, str):
        locations = _parse_compact_string(locations)
    if not isinstance(locations, dict) or not locations:
        raise ValueError("locations must be a non-empty dict of {name: {latitude, longitude, weight?}}")

    normalised = []
    seen = set()
    for name, spec in locations.items():
        if not isinstance(name, str) or not _LOCATION_NAME_RE.match(name):
            raise ValueError(f"location name must use letters, digits and underscores, got {name!r}")
        if name in seen:
            raise ValueError(f"duplicate location name {name!r}")
        seen.add(name)
        normalised.append({"name": name, **_parse_location_spec(name, spec)})

    weights = [loc["weight"] for loc in normalised]
    if any(w is not None for w in weights) and not all(w is not None for w in weights):
        missing = sorted(loc["name"] for loc in normalised if loc["weight"] is None)
        raise ValueError(f"weight must be given for every location or for none; missing for {missing}")

    return normalised


def _parse_compact_string(text: str) -> Dict[str, Any]:
    """ "name:latitude:longitude[:weight], ..." -> {name: (latitude, longitude) | (latitude, longitude, weight)}."""
    locations: Dict[str, Any] = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        parts = item.split(":")
        if len(parts) not in (3, 4):
            raise ValueError(f'location {item!r} must be "name:latitude:longitude" or "name:latitude:longitude:weight"')
        name = parts[0].strip()
        try:
            numbers = tuple(float(p) for p in parts[1:])
        except ValueError:
            raise ValueError(f'location {item!r} must be "name:latitude:longitude[:weight]" with numbers')
        if name in locations:
            raise ValueError(f"duplicate location name {name!r}")
        locations[name] = numbers
    return locations


def _parse_location_spec(name: str, spec: Any) -> Dict[str, Any]:
    """One location's `{"latitude", "longitude", "weight"}`, from a dict or a `(lat, lon[, weight])` sequence."""
    if isinstance(spec, dict):
        allowed = {"latitude", "longitude", "weight"}
        if not {"latitude", "longitude"} <= set(spec) or not set(spec) <= allowed:
            raise ValueError(
                f"location {name!r} must be a dict with latitude, longitude and optionally weight, got {spec!r}"
            )
        latitude, longitude, weight = spec["latitude"], spec["longitude"], spec.get("weight")
    elif isinstance(spec, (list, tuple)):
        if len(spec) == 2:
            (latitude, longitude), weight = spec, None
        elif len(spec) == 3:
            latitude, longitude, weight = spec
        else:
            raise ValueError(
                f"location {name!r} must be (latitude, longitude) or (latitude, longitude, weight), got {spec!r}"
            )
    else:
        raise ValueError(f"location {name!r} must be a dict or a (latitude, longitude[, weight]) tuple, got {spec!r}")

    if isinstance(latitude, bool) or not isinstance(latitude, (int, float)) or not -90 <= latitude <= 90:
        raise ValueError(f"latitude of {name!r} must be between -90 and 90, got {latitude!r}")
    if isinstance(longitude, bool) or not isinstance(longitude, (int, float)) or not -180 <= longitude <= 180:
        raise ValueError(f"longitude of {name!r} must be between -180 and 180, got {longitude!r}")
    if weight is not None:
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"weight of {name!r} must be a positive number, got {weight!r}")
        weight = float(weight)

    return {"latitude": float(latitude), "longitude": float(longitude), "weight": weight}


def _weighted_average(frames: List[pd.DataFrame], weights: np.ndarray) -> pd.DataFrame:
    """Weighted combination of same-columned, same-indexed frames, one row per shared index value.

    Most columns use a plain weighted mean; a column whose variable name (before `_d+i`) contains "direction"
    uses a weighted circular mean instead, so 350 deg and 10 deg average to about 0, not 180; a categorical
    variable (`_CATEGORICAL_VARIABLES`, e.g. `weather_code`) is skipped entirely — it has no meaningful average.
    A row is NaN wherever any location is NaN there: weights are never re-normalised over just the locations
    that have a value.
    """
    weighted = {}
    for column in frames[0].columns:
        variable = column.rsplit("_d+", 1)[0]
        if variable in _CATEGORICAL_VARIABLES:
            continue
        stack = np.column_stack([frame[column].to_numpy(dtype=float) for frame in frames])
        if _CIRCULAR_VARIABLE_RE.search(variable):
            radians = np.deg2rad(stack)
            sin = (np.sin(radians) * weights).sum(axis=1)
            cos = (np.cos(radians) * weights).sum(axis=1)
            values = np.mod(np.rad2deg(np.arctan2(sin, cos)), 360.0)
        else:
            values = (stack * weights).sum(axis=1)
        values[np.isnan(stack).any(axis=1)] = np.nan
        weighted[column] = values
    return pd.DataFrame(weighted, index=frames[0].index)


class OpenMeteoBase(DataSource):
    """Common behaviour of the Open-Meteo sources, for one location or several.

    Subclasses set their public attributes (latitude, longitude, locations, keep_locations, horizon, model,
    columns, prefix, proxy_url, timezone, issue_time, publication_delay_h, derived, hdd_base_c, cdd_base_c,
    strict), then call ``self._setup()`` and ``self._validate_config()``, which must call ``_validate_common()``.

    ``locations`` is mutually exclusive with ``latitude``/``longitude`` (see ``parse_locations``) and is resolved
    by ``_configure_locations()`` into ``self._locations`` (normalised) and ``self._location_sources`` (one
    instance of the same subclass per location, sharing every other setting — see ``_location_kwargs``).
    Subclasses whose own ``fetch()`` finds ``self._locations`` non-empty must delegate to ``_fetch_locations()``
    instead of the single-point path.
    """

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
    VALID_DERIVED = tuple(DERIVED_INPUTS)

    MAX_RETRIES = 5
    BACKOFF_BASE_S = 2.0
    BACKOFF_CAP_S = 60.0
    REQUEST_TIMEOUT_S = 60

    # Set by DataPipeline._bind_component_timezones() when `timezone` is left unset; never written by users.
    _pipeline_timezone: Optional[str] = None

    @property
    def market_timezone(self) -> Optional[str]:
        """Resolved zone: `timezone` if set, else the one a `DataPipeline` inherited it from (`None` if neither)."""
        return _resolve_timezone(self.timezone, self)

    def _require_market_timezone(self) -> str:
        zone = self.market_timezone
        if zone is not None:
            return zone
        raise ValueError(
            "timezone is required: set it to your market's zone, e.g. 'Europe/Warsaw' for PL, 'Europe/London' "
            "for GB, 'Europe/Berlin' for DE — either directly on this source, or once via "
            "DataPipeline(timezone=...). It defines the local day and the issue time used to pick a leak-free "
            "forecast vintage; a market outside CET that left it unset used to silently fall back to "
            "Europe/Warsaw, computing the issue time in the wrong zone (e.g. London 23:00 on D became 10:00 on "
            "D+1)."
        )

    def _setup(self):
        self.session = requests.Session()
        self._configure_proxy()
        self.console = Console()
        self.logger = get_logger("epftoolbox2.data.sources.open_meteo")

    def __del__(self):
        """Cleanup session on object destruction"""
        if hasattr(self, "session") and self.session:
            self.session.close()

    @property
    def _label(self) -> str:
        return f"Open-Meteo [{self._location_label()}]"

    def _location_label(self) -> str:
        if self._locations:
            names = ", ".join(loc["name"] for loc in self._locations)
            return f"{len(self._locations)} locations: {names}"
        return f"{self.latitude}, {self.longitude}"

    def _configure_proxy(self):
        if self.proxy_url:
            self.session.proxies = {"http": self.proxy_url, "https": self.proxy_url}

    def _now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")

    @staticmethod
    def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
        return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")

    def _validate_common(self, max_horizon: Optional[int] = None) -> None:
        """Checks shared by all Open-Meteo sources (everything except the API-specific parameters).

        `max_horizon` is enforced here only when given. A source whose usable horizon depends on the data (e.g.
        ``OpenMeteoSource``'s bucket availability) checks its own bound after calling this, with its own message.
        """
        self._configure_locations()

        if not self._locations:
            if not isinstance(self.latitude, (int, float)) or not -90 <= self.latitude <= 90:
                raise ValueError(f"Latitude must be between -90 and 90, got {self.latitude}")

            if not isinstance(self.longitude, (int, float)) or not -180 <= self.longitude <= 180:
                raise ValueError(f"Longitude must be between -180 and 180, got {self.longitude}")

        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon <= 0:
            raise ValueError(f"Horizon must be a positive integer, got {self.horizon}")
        if max_horizon is not None and self.horizon > max_horizon:
            raise ValueError(f"Horizon must be at most {max_horizon} for {type(self).__name__}, got {self.horizon}")

        if not self.columns or not all(isinstance(c, str) and c for c in self.columns):
            raise ValueError("At least one weather column must be specified")

        # `timezone` may be left unset here and inherited from DataPipeline.timezone instead (see
        # `market_timezone`); `_require_market_timezone()` raises once a concrete zone is actually needed.
        if self.timezone is not None:
            try:
                ZoneInfo(self.timezone)
            except (KeyError, ValueError, TypeError):
                raise ValueError(f"Invalid timezone: {self.timezone!r}")

        parse_issue_time(self.issue_time)

        delay = self.publication_delay_h
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not 0 <= delay <= 48:
            raise ValueError(f"publication_delay_h must be a number of hours between 0 and 48, got {delay!r}")

        self._validate_derived()

    def _validate_derived(self) -> None:
        unknown = [name for name in self.derived if name not in DERIVED_INPUTS]
        if unknown:
            raise ValueError(f"derived must be a subset of {list(DERIVED_INPUTS)}, got {unknown}")
        for name in self.derived:
            if DERIVED_INPUTS[name] not in self.columns:
                raise ValueError(f"derived column {name!r} needs {DERIVED_INPUTS[name]!r} in columns")
        for label, base in (("hdd_base_c", self.hdd_base_c), ("cdd_base_c", self.cdd_base_c)):
            if isinstance(base, bool) or not isinstance(base, (int, float)):
                raise ValueError(f"{label} must be a number, got {base!r}")

    def _configure_locations(self) -> None:
        """Parse `locations`, mutually exclusive with `latitude`/`longitude`, and build one instance of this
        class per location, sharing every other constructor parameter (see `_location_kwargs`).

        A single `latitude`/`longitude` pair keeps working exactly as before: `self._locations` and
        `self._location_sources` are left empty, `fetch()` takes the single-point path, and the cache key for an
        existing configuration does not change (see `_common_cache_config`).
        """
        lat_given = self.latitude is not None
        lon_given = self.longitude is not None
        if lat_given != lon_given:
            raise ValueError("latitude and longitude must both be set, or both left unset when using locations")
        has_point = lat_given and lon_given
        has_locations = self.locations is not None

        if has_point and has_locations:
            raise ValueError("locations is mutually exclusive with latitude/longitude; set only one of them")
        if not has_point and not has_locations:
            raise ValueError("set either latitude and longitude for one location, or locations for several")

        if not has_locations:
            self._locations = []
            self._location_sources = []
            return

        self._locations = parse_locations(self.locations)
        if not isinstance(self.keep_locations, bool):
            raise ValueError(f"keep_locations must be True or False, got {self.keep_locations!r}")

        kwargs = self._location_kwargs()
        self._location_sources = [
            type(self)(latitude=loc["latitude"], longitude=loc["longitude"], **kwargs) for loc in self._locations
        ]

    def _location_kwargs(self) -> Dict[str, Any]:
        """This instance's own constructor parameters, minus the ones `locations` replaces (`latitude`,
        `longitude`, `locations`, `prefix`, `keep_locations`) and runtime objects — used to build one sibling
        instance per location that shares every other setting."""
        return {
            key: value
            for key, value in vars(self).items()
            if not key.startswith("_") and key not in _RUNTIME_ATTRIBUTES and key not in _NON_LOCATION_KWARGS
        }

    @property
    def _locations_weighted(self) -> bool:
        """Whether every location has a weight (see `parse_locations`: it's all locations or none)."""
        return bool(self._locations) and all(loc["weight"] is not None for loc in self._locations)

    @property
    def _location_weights(self) -> np.ndarray:
        weights = np.array([loc["weight"] for loc in self._locations], dtype=float)
        return weights / weights.sum()

    def _fetch_locations(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Fetch every location (built by `_configure_locations`) and combine per weights / `keep_locations`.

        Every location is fetched by another instance of this class sharing every other setting, so all
        locations follow the same vintage rule. With weights this returns one weighted series per variable,
        `{prefix}_{variable}_d+i` (see `_weighted_average`); `keep_locations=True` also adds the per-location
        columns `{prefix}_{name}_{variable}_d+i`. Without weights, only the per-location columns are returned.
        One shared "now" is used for every location so a live fetch can't settle some locations as final and
        others as still provisional.
        """
        now = self._now()
        for source in self._location_sources:
            source._now = lambda now=now: now

        frames = []
        for location, source in zip(self._locations, self._location_sources):
            weight_note = f" (weight {location['weight']})" if location["weight"] is not None else ""
            self.logger.info(f"{self._label}: location {location['name']}{weight_note}")
            frames.append(source.fetch(start, end))
        if all(frame.empty for frame in frames):
            return pd.DataFrame()

        index = frames[0].index
        for frame in frames[1:]:
            index = index.union(frame.index)
        frames = [frame.reindex(index) for frame in frames]

        weighted_mode = self._locations_weighted
        parts = []
        if weighted_mode:
            weighted = _weighted_average(frames, self._location_weights)
            if self.prefix:
                weighted.columns = [f"{self.prefix}_{col}" for col in weighted.columns]
            parts.append(weighted)
            skipped = {
                col.rsplit("_d+", 1)[0]
                for col in frames[0].columns
                if col.rsplit("_d+", 1)[0] in _CATEGORICAL_VARIABLES
            }
            if skipped and not self.keep_locations:
                self.logger.info(
                    f"{self._label}: {', '.join(sorted(skipped))} is categorical and is not averaged; "
                    "use keep_locations=True to keep it per location"
                )
        if not weighted_mode or self.keep_locations:
            base = f"{self.prefix}_" if self.prefix else ""
            parts.extend(
                frame.add_prefix(f"{base}{location['name']}_") for location, frame in zip(self._locations, frames)
            )

        result = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]
        filled = result.notna().any(axis=1).to_numpy()
        keep = int(np.flatnonzero(filled)[-1]) + 1 if filled.any() else 0
        result = result.iloc[:keep]
        result.index.name = "datetime"
        return result

    def _backoff(self, attempt: int) -> float:
        return min(self.BACKOFF_CAP_S, self.BACKOFF_BASE_S * 2**attempt)

    def _request(self, url: str, params: Dict, unavailable: Optional[str] = None) -> Optional[Dict]:
        """GET with retries: capped exponential backoff on network errors and short-term rate limits.

        Returns None when the API reports an error containing `unavailable` (e.g. a model run it doesn't have).
        """
        for attempt in range(self.MAX_RETRIES):
            last_attempt = attempt == self.MAX_RETRIES - 1
            try:
                response = self.session.get(url, params=params, timeout=self.REQUEST_TIMEOUT_S)
                data = response.json()
                if not (isinstance(data, dict) and data.get("error")):
                    response.raise_for_status()
                    return data
            except requests.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}")
                if last_attempt:
                    raise
                time.sleep(self._backoff(attempt))
                continue

            reason = data.get("reason", "Unknown error")
            if unavailable and unavailable in reason:
                return None
            if "Minutely API request limit exceeded" in reason:
                wait = self.BACKOFF_CAP_S
            elif "Too many concurrent requests" in reason:
                wait = self._backoff(attempt)
            elif "Hourly API request limit exceeded" in reason or "Daily API request limit exceeded" in reason:
                raise RuntimeError(f"Open-Meteo: {reason} Retry later or fetch fewer variables per request.")
            else:
                raise ValueError(f"Open-Meteo API error: {reason}")

            self.logger.warning(f"API error: {reason} (attempt {attempt + 1}/{self.MAX_RETRIES})")
            if not last_attempt:
                time.sleep(wait)

        raise RuntimeError(f"Failed to fetch data after {self.MAX_RETRIES} retries")

    def _issue_instants(self, rows: pd.DatetimeIndex) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        """Issue instant per row and a mask of final rows (issue time passed).

        Rows whose issue time is still in the future (live runs) get "now" instead: they use the runs published so
        far, are provisional and change when fetched again later.
        """
        issue = local_issue_times(rows, self._require_market_timezone(), self.issue_time)
        now = self._now()
        final = np.asarray(issue <= now)
        if not final.all():
            self.logger.info(
                f"{self._label}: {int((~final).sum())} rows from {rows[~final][0]} have an issue time after now; "
                "they use runs published by now and will change when fetched again later"
            )
            issue = issue.where(final, now)
        return issue, final

    def _finish(self, result: pd.DataFrame, final: np.ndarray) -> pd.DataFrame:
        """Derived columns, trailing-row trim, prefix and coverage report, in that order."""
        if self.derived:
            result = add_derived_columns(result, self.derived, self.horizon, self.hdd_base_c, self.cdd_base_c)
        result, final, dropped_final = self._drop_unfilled_tail(result, final)
        if self.prefix:
            result.columns = [f"{self.prefix}_{col}" for col in result.columns]
        self._report_coverage(result, final, dropped_final)
        return result

    def _drop_unfilled_tail(self, result: pd.DataFrame, final: np.ndarray):
        """Drop trailing rows without any value (runs that do not exist yet, or the end of the archive).

        Returns the kept rows, their `final` flags and the number of dropped rows whose issue time has passed.
        """
        filled = result.notna().any(axis=1).to_numpy()
        keep = int(np.flatnonzero(filled)[-1]) + 1 if filled.any() else 0
        dropped_final = int(final[keep:].sum())
        if keep < len(result):
            self.logger.info(f"{self._label}: no data from {result.index[keep]} on; those rows are not returned")
        return result.iloc[:keep], final[:keep], dropped_final

    def _report_coverage(self, result: pd.DataFrame, final: np.ndarray, dropped_final: int):
        """Log per-column coverage and raise under `strict`.

        Only final rows count: provisional rows (issue time in the future) are expected to be incomplete. Final
        rows dropped because they had no value at all (`_drop_unfilled_tail`) count as missing but, having no
        timestamp, aren't part of the per-hour breakdown below.

        For each incomplete column this reports the timestamp the column is complete *from* (one step past its
        last missing row), not the first non-null value: a column whose coverage is restricted to some local
        hours can have an early first valid value while other hours of the same day stay NaN for years (this is
        exactly how JMA's `previous_day5`-`7` behave: missing for 2018-2020, but only for the hours that need
        that bucket). It also reports the missing share by local hour whenever it differs across hours, since the
        overall share hides which hours are affected.
        """
        final_rows = result[final]
        total = len(final_rows) + dropped_final
        if total == 0:
            return
        missing = (final_rows.isna().sum() + dropped_final) / total
        incomplete = missing[missing > 0]
        if incomplete.empty:
            return

        local_hour = final_rows.index.tz_convert(self._require_market_timezone()).hour
        lines = []
        for column, share in incomplete.items():
            complete_from = self._first_complete_timestamp(final_rows[column], dropped_final)
            line = f"  {column}: {share:.1%} missing, complete from {complete_from}"
            hourly = self._hourly_missing_share(final_rows[column], local_hour)
            if hourly is not None:
                by_hour = ", ".join(f"{hour:02d}h {hour_share:.0%}" for hour, hour_share in hourly.items())
                line += f"; missing share by local hour: {by_hour}"
            lines.append(line)

        message = (
            f"{self._label} model={self.model!r}: {len(incomplete)} of {len(missing)} columns have missing values "
            f"in {total} rows whose issue time has passed:\n" + "\n".join(lines)
        )
        if self.strict:
            raise ValueError(message)
        self.logger.warning(message)

    @staticmethod
    def _first_complete_timestamp(series: pd.Series, dropped_final: int) -> Optional[pd.Timestamp]:
        """Timestamp right after `series`'s last missing row, or None if it is still missing at the end.

        Rows dropped for having no value in any column follow the end of `series` and are always missing, so a
        positive `dropped_final` means the column never becomes complete within the fetched range.
        """
        if dropped_final > 0 or len(series) == 0:
            return None
        na = series.isna().to_numpy()
        if not na.any():
            return series.index[0]
        last_na = int(np.flatnonzero(na)[-1])
        return series.index[last_na + 1] if last_na + 1 < len(series) else None

    @staticmethod
    def _hourly_missing_share(series: pd.Series, local_hour: pd.Index) -> Optional[pd.Series]:
        """Missing share of `series` grouped by `local_hour`, or None when every hour has the same share."""
        if len(series) == 0:
            return None
        share = series.isna().groupby(local_hour).mean().sort_index()
        if np.isclose(share.max(), share.min()):
            return None
        return share

    def _common_cache_config(self) -> Dict:
        config = {
            "horizon": self.horizon,
            "model": self.model,
            "columns": sorted(self.columns),
            "prefix": self.prefix,
            "timezone": self._require_market_timezone(),
            "issue_time": self.issue_time,
            "publication_delay_h": self.publication_delay_h,
        }
        # Only when used, so a single-point configuration (the common case) keeps its existing cache key.
        if self._locations:
            config["locations"] = [{k: v for k, v in loc.items() if v is not None} for loc in self._locations]
            if self.keep_locations:
                config["keep_locations"] = True
        else:
            config["latitude"] = self.latitude
            config["longitude"] = self.longitude
        # Only when used, so configurations without derived columns keep their cache key.
        if self.derived:
            config["derived"] = sorted(self.derived)
        if "hdd" in self.derived:
            config["hdd_base_c"] = self.hdd_base_c
        if "cdd" in self.derived:
            config["cdd_base_c"] = self.cdd_base_c
        return config


def normalise_derived(derived: Optional[List[str]]) -> List[str]:
    return list(derived) if derived else []
