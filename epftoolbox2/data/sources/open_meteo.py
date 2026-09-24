import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from ._open_meteo_common import OpenMeteoBase, local_issue_times, normalise_derived, parse_hourly, shift_local_days

# The Previous Runs API serves `{variable}_previous_day1` .. `_previous_day7`; day 8 and later come back as nulls.
MAX_PREVIOUS_DAY = 7

# Time step (hours) at which Open-Meteo stores a model's `previous_dayK` values, as ((max_lead_h, step_h), ...)
# segments; the first segment whose max_lead_h >= 24K + 6 applies (None = no limit). Hourly output of a coarser
# store is interpolated across neighbouring stored points (4-point Hermite for temperature-like variables,
# backward sums/averages for precipitation and radiation). Those neighbours come from *different* runs, so the
# hourly value at T can depend on the run behind the stored point up to two steps after T. Probed against the
# Single Runs API: JMA GSM is stored 6-hourly; ICON is hourly to 78 h and 3-hourly beyond. The other entries
# follow the models' published output steps. Unknown models get the conservative 6 h.
# Changing this table changes outputs: bump SCHEMA_VERSION.
_STORED_STEP_H: Tuple[Tuple[str, Tuple[Tuple[Optional[int], int], ...]], ...] = (
    ("icon_d2", ((None, 1),)),
    ("icon_", ((78, 1), (None, 3))),
    ("ecmwf_ifs025", ((144, 3), (None, 6))),
    ("ecmwf_ifs04", ((144, 3), (None, 6))),
    ("ecmwf_aifs", ((None, 6),)),
    ("ecmwf_ifs", ((90, 1), (144, 3), (None, 6))),
    ("gfs_seamless", ((120, 1), (None, 3))),
    ("gfs_global", ((120, 1), (None, 3))),
    ("jma_", ((None, 6),)),
)
_DEFAULT_STORED_STEP_H = 6


def stored_step_hours(model: str, bucket: int) -> int:
    """Hours between the stored points behind `previous_day{bucket}` for `model` (see `_STORED_STEP_H`)."""
    lead_h = 24 * bucket + 6
    for prefix, segments in _STORED_STEP_H:
        if model.startswith(prefix):
            for max_lead_h, step_h in segments:
                if max_lead_h is None or lead_h <= max_lead_h:
                    return step_h
    return _DEFAULT_STORED_STEP_H


@lru_cache(maxsize=128)
def _bucket_requirements(
    model: str, timezone: str, issue_time: str, publication_delay_h: float, horizon: int
) -> Tuple[Tuple[int, int], ...]:
    """(smallest, largest) `previous_dayK` needed for each lead 1..horizon, over a full year of hourly rows.

    The largest K may exceed MAX_PREVIOUS_DAY; callers reject such a horizon.
    """
    rows = pd.date_range("2024-01-01", "2025-01-01", freq="1h", tz="UTC", inclusive="left")
    issue = local_issue_times(rows, timezone, issue_time)
    delay = pd.Timedelta(hours=publication_delay_h)
    needs = []
    for lead in range(1, horizon + 1):
        valid = shift_local_days(rows, timezone, lead)
        chosen = select_buckets(model, valid, issue, delay, lead, lead + MAX_PREVIOUS_DAY + 2)
        if (chosen == 0).any():
            raise ValueError(f"publication_delay_h={publication_delay_h} leaves no usable previous_dayK for d+{lead}")
        needs.append((int(chosen.min()), int(chosen.max())))
    return tuple(needs)


def select_buckets(
    model: str,
    valid: pd.DatetimeIndex,
    issue: pd.DatetimeIndex,
    delay: pd.Timedelta,
    lead: int,
    max_bucket: int,
) -> np.ndarray:
    """Smallest K in [lead, max_bucket] per cell whose value only depends on runs published by `issue`; 0 if none.

    `previous_dayK` at valid time T holds the latest run initialised at or before T - 24K h (Open-Meteo Previous
    Runs docs). With a stored step s > 1 h the hourly value also depends on stored points up to ceil_s(T) + s, so
    the latest run that can influence it was initialised at or before ceil_s(T) + s - 24K h.
    """
    chosen = np.zeros(len(valid), dtype=np.int64)
    for bucket in range(lead, max_bucket + 1):
        step = stored_step_hours(model, bucket)
        reach_end = valid if step == 1 else valid.ceil(f"{step}h") + pd.Timedelta(hours=step)
        latest_init = reach_end - pd.Timedelta(days=bucket)
        ok = np.asarray(latest_init + delay <= issue) & (chosen == 0)
        chosen[ok] = bucket
    return chosen


class OpenMeteoSource(OpenMeteoBase):
    """
    Open-Meteo weather forecasts as they were known when the day-ahead forecast was issued.

    Uses the Open-Meteo Previous Runs API, which stores each model's forecast at fixed lead-time offsets:
    `{variable}_previous_dayK` at valid time T comes from the latest run initialised at least 24K hours before T.

    With the default ``vintage="issue_time"``, the column ``{variable}_d+i`` on a row of local day D (in
    ``timezone``) holds the forecast for the same local wall-clock hour on D+i, taken from the smallest bucket
    K >= i whose runs were all published (initialisation + ``publication_delay_h``) by ``issue_time`` on D. Most
    D+1 hours therefore come from a run of D-1 or D-2: that is the price of reconstructing vintages from lead-time
    buckets. ``vintage="valid_time"`` restores the mapping of epftoolbox2 <= 2.2 (bucket i at t + 24 h * i), which
    uses runs published after the issue time; keep it only to reproduce old results.

    Example:
        >>> source = OpenMeteoSource(
        ...     latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", horizon=5, prefix="warsaw"
        ... )
        >>> df = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-02-01", tz="UTC"))
    """

    API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

    VALID_TEMPORAL_RESOLUTIONS = {None, "native", "hourly_3", "hourly_6"}
    VALID_VINTAGES = ("issue_time", "valid_time")
    SCHEMA_VERSION = 2

    def __init__(
        self,
        latitude: float,
        longitude: float,
        horizon: int = 5,
        model: str = "jma_seamless",
        columns: Optional[List[str]] = None,
        prefix: str = "",
        proxy_url: Optional[str] = None,
        temporal_resolution: Optional[str] = None,
        timezone: Optional[str] = None,
        issue_time: str = "10:00",
        publication_delay_h: float = 10,
        vintage: str = "issue_time",
        derived: Optional[List[str]] = None,
        hdd_base_c: float = 18.0,
        cdd_base_c: float = 21.0,
        strict: bool = False,
    ):
        """
        Initialize Open-Meteo data source

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            horizon: Days ahead (d+1 .. d+horizon), 1-7. With ``vintage="issue_time"`` the largest usable horizon
                depends on the model and issue settings (5 for the default JMA model); larger values raise.
            model: Weather model to use (default: "jma_seamless")
            columns: List of weather variables to fetch (default: DEFAULT_COLUMNS)
            prefix: Prefix to add to column names (default: "")
            proxy_url: Proxy URL (e.g., 'http://proxy.example.com:8080')
            temporal_resolution: Temporal resolution of the data. One of:
                None (default) — hourly (1 h), no parameter sent to API;
                "hourly_3" — 3-hourly; "hourly_6" — 6-hourly;
                "native" — model-native resolution (inferred from response).
            timezone: Market time zone. Defines the local calendar day D of each row, the local wall-clock hour
                that d+i refers to, and the time zone of ``issue_time``. None (default) inherits
                ``DataPipeline.timezone``; there is no other default, and fetching without either raises: a
                market outside CET that silently fell back to one would leak, since the issue time would be
                computed in the wrong zone (e.g. London 23:00 on D becoming 10:00 on D+1 in Warsaw time).
            issue_time: Local time ("HH:MM") on day D at which the forecast is issued (default "10:00").
            publication_delay_h: Hours between a run's initialisation and its availability (default 10). The
                Single Runs docs give 4-6 h for global models; Open-Meteo's model metadata showed 9.6 h for JMA
                GSM, 7-8 h for ECMWF IFS and 3.8 h for ICON.
            vintage: "issue_time" (default, leak-free) or "valid_time" (mapping of epftoolbox2 <= 2.2, look-ahead).
            derived: Extra columns computed from the forecast: "hdd", "cdd" (need temperature_2m) and
                "wind_speed_100m_cubed" (needs wind_speed_100m). Off by default.
            hdd_base_c: Base temperature of "hdd" (°C)
            cdd_base_c: Base temperature of "cdd" (°C)
            strict: Raise instead of warning when a returned column has missing values.
        """
        self.latitude = latitude
        self.longitude = longitude
        self.horizon = horizon
        self.model = model
        self.columns = list(columns) if columns else list(self.DEFAULT_COLUMNS)
        self.prefix = prefix
        self.proxy_url = proxy_url
        self.temporal_resolution = temporal_resolution
        self.timezone = timezone
        self.issue_time = issue_time
        self.publication_delay_h = publication_delay_h
        self.vintage = vintage
        self.derived = normalise_derived(derived)
        self.hdd_base_c = hdd_base_c
        self.cdd_base_c = cdd_base_c
        self.strict = strict

        self._setup()

        self._buckets: List[int] = []
        self._validate_config()

    def _validate_config(self) -> bool:
        """Validate the configuration parameters and work out which previous_dayK buckets to request."""
        self._validate_common()

        if self.horizon > MAX_PREVIOUS_DAY:
            raise ValueError(
                f"Horizon must be at most {MAX_PREVIOUS_DAY}: the Previous Runs API serves previous_day1.."
                f"previous_day{MAX_PREVIOUS_DAY} only, got {self.horizon}"
            )

        if self.temporal_resolution not in self.VALID_TEMPORAL_RESOLUTIONS:
            raise ValueError(
                f"temporal_resolution must be one of {self.VALID_TEMPORAL_RESOLUTIONS}, "
                f"got {self.temporal_resolution!r}"
            )

        if self.vintage not in self.VALID_VINTAGES:
            raise ValueError(f"vintage must be one of {self.VALID_VINTAGES}, got {self.vintage!r}")

        if self.vintage == "valid_time":
            self.logger.warning(
                f"{self._label}: vintage='valid_time' uses model runs published after the issue time (look-ahead). "
                "Use it only to reproduce results of epftoolbox2 <= 2.2."
            )
            self._buckets = list(range(1, self.horizon + 1))
            return True

        # The buckets depend on the market time zone, which may not be known yet (it can be inherited from
        # DataPipeline.timezone later): resolve them now if it is already known, else lazily at fetch() time.
        if self.timezone is not None:
            self._resolve_buckets(self.timezone)
        return True

    def _resolve_buckets(self, timezone: str) -> None:
        """Work out which previous_dayK buckets to request, for a now-known market `timezone`."""
        delay = self.publication_delay_h
        needs = _bucket_requirements(self.model, timezone, self.issue_time, float(delay), self.horizon)
        worst = max(high for _, high in needs)
        if worst > MAX_PREVIOUS_DAY:
            first_bad = next(lead for lead, (_, high) in enumerate(needs, 1) if high > MAX_PREVIOUS_DAY)
            raise ValueError(
                f"horizon={self.horizon} cannot be served without look-ahead: d+{first_bad} needs "
                f"previous_day{needs[first_bad - 1][1]} for some hours (model={self.model!r}, "
                f"issue_time={self.issue_time!r} {timezone}, publication_delay_h={delay}), but Open-Meteo "
                f"serves previous_day1..{MAX_PREVIOUS_DAY} only. Use horizon<={first_bad - 1}; a shorter "
                "publication_delay_h if it is justified for this model; or vintage='valid_time' (look-ahead) to "
                "reproduce results of epftoolbox2 <= 2.2."
            )
        low = min(lo for lo, _ in needs)
        # One spare bucket lets live runs made before the issue time fall back further (see _build_features).
        self._buckets = list(range(low, min(worst + 1, MAX_PREVIOUS_DAY) + 1))

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Fetch weather forecast features for rows in [start, end)

        Args:
            start: Start timestamp (inclusive)
            end: End timestamp (exclusive)

        Returns:
            DataFrame with a UTC index covering [start, end) at the response's time step and columns
            ``{prefix}_{variable}_d+{i}``. Trailing rows without any value (future runs that do not exist yet) are
            not returned.
        """
        start, end = self._to_utc(start), self._to_utc(end)

        if end <= start:
            raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")

        # `timezone` may have been unset at construction and inherited from DataPipeline.timezone since (see
        # `market_timezone`); the buckets could not be resolved until now in that case.
        tz = self._require_market_timezone()
        if self.vintage != "valid_time" and not self._buckets:
            self._resolve_buckets(tz)

        if self.vintage == "valid_time":
            self.logger.warning(
                f"{self._label}: vintage='valid_time' uses model runs published after the issue time (look-ahead). "
                "Use it only to reproduce results of epftoolbox2 <= 2.2."
            )

        start_time = time.time()
        # Valid times needed: rows' own times up to end + horizon local days (+1 day for DST and interpolation).
        chunks = self._generate_chunks(start, end + DateOffset(days=self.horizon + 1), months=3)

        self.logger.info(f"{self._label}: Start downloading weather data")

        tables = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"[cyan]{self._label}: Downloading...", total=len(chunks))

            for chunk_start, chunk_end in chunks:
                progress.update(task, description=f"[cyan]{self._label}: {chunk_start.date()} to {chunk_end.date()}")
                table = self._fetch_chunk(chunk_start, chunk_end)
                if table is not None and not table.empty:
                    tables.append(table)
                progress.advance(task)

        if not tables:
            return pd.DataFrame()

        table = pd.concat(tables)
        table = table[~table.index.duplicated(keep="last")].sort_index()

        result, final = self._build_features(table, start, end)
        result = self._finish(result, final)

        elapsed = time.time() - start_time
        self.logger.info(f"{self._label}: Download completed in {elapsed:.2f} sec")
        return result

    def _generate_chunks(self, start: pd.Timestamp, end: pd.Timestamp, months: int = 3) -> list:
        """Split [start, end] into contiguous request windows of at most `months` months."""
        if months <= 0:
            raise ValueError(f"months parameter must be positive, got {months}")

        chunks = []
        current = start
        while current < end:
            chunk_end = min(current + pd.DateOffset(months=months), end)
            chunks.append((current, chunk_end))
            current = chunk_end
        return chunks

    def _fetch_chunk(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Download the requested previous_dayK buckets for valid dates start..end (UTC, inclusive)."""
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": [f"{column}_previous_day{k}" for k in self._buckets for column in self.columns],
            "models": self.model,
            "timezone": "GMT",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
        }
        if self.temporal_resolution is not None:
            params["temporal_resolution"] = self.temporal_resolution
        return self._parse_response(self._request(self.API_URL, params))

    def _parse_response(self, response_data: Dict) -> pd.DataFrame:
        """Response → DataFrame indexed by UTC valid time, one float column per `{variable}_previous_dayK`."""
        return parse_hourly(response_data)

    def _build_features(self, table: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
        """Map the bucket table to `{variable}_d+{i}` columns for rows in [start, end).

        Returns the features and a boolean array marking rows whose issue time has passed (final rows). Rows whose
        issue time is still in the future (live runs) use the runs published by now instead: they are provisional
        and change when fetched again later.
        """
        rows = table.index[(table.index >= start) & (table.index < end)]
        n = len(rows)
        features = {}

        if self.vintage == "valid_time":
            final = np.ones(n, dtype=bool)
            for lead in range(1, self.horizon + 1):
                positions = table.index.get_indexer(rows + pd.Timedelta(days=lead))
                for column in self.columns:
                    features[f"{column}_d+{lead}"] = self._take(table, f"{column}_previous_day{lead}", positions)
            return pd.DataFrame(features, index=rows), final

        issue = local_issue_times(rows, self.market_timezone, self.issue_time)
        now = self._now()
        final = np.asarray(issue <= now)
        if not final.all():
            first = rows[~final][0]
            self.logger.info(
                f"{self._label}: {int((~final).sum())} rows from {first} have an issue time after now; they use runs "
                "published by now and will change when fetched again later"
            )
            issue = issue.where(final, now)

        delay = pd.Timedelta(hours=self.publication_delay_h)
        steps = np.diff(table.index.asi8)
        tolerance = pd.Timedelta(int(steps.min()) // 2 if len(steps) else 0)
        max_bucket = max(self._buckets)
        for lead in range(1, self.horizon + 1):
            target = shift_local_days(rows, self.market_timezone, lead)
            positions = table.index.get_indexer(target, method="nearest", tolerance=tolerance)
            valid = pd.DatetimeIndex(np.where(positions >= 0, table.index.asi8[positions], np.iinfo(np.int64).min))
            valid = valid.tz_localize("UTC")
            buckets = select_buckets(self.model, valid, issue, delay, lead, max_bucket)
            buckets[positions < 0] = 0
            for column in self.columns:
                values = np.full(n, np.nan)
                for bucket in np.unique(buckets[buckets > 0]):
                    mask = buckets == bucket
                    values[mask] = self._take(table, f"{column}_previous_day{bucket}", positions[mask])
                features[f"{column}_d+{lead}"] = values
        return pd.DataFrame(features, index=rows), final

    @staticmethod
    def _take(table: pd.DataFrame, key: str, positions: np.ndarray) -> np.ndarray:
        if key not in table.columns:
            return np.full(len(positions), np.nan)
        values = table[key].to_numpy()
        out = np.full(len(positions), np.nan)
        found = positions >= 0
        out[found] = values[positions[found]]
        return out

    def get_cache_config(self) -> dict:
        return {
            "source_type": "open_meteo",
            "schema_version": self.SCHEMA_VERSION,
            **self._common_cache_config(),
            "temporal_resolution": self.temporal_resolution,
            "vintage": self.vintage,
        }
