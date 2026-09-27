import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

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

from ._open_meteo_common import OpenMeteoBase, parse_hourly, parse_run, run_cutoff, shift_local_days

# The Previous Runs API serves `{variable}_previous_day1` .. `_previous_day7`; day 8 and later come back as nulls.
MAX_PREVIOUS_DAY = 7
# The Single Runs API's own limit: leads beyond a run's own length come back null (see OpenMeteoSource.horizon).
MAX_SINGLE_RUNS_HORIZON = 14

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


# Typical hours between a model run's initialisation and its availability, by model family — used only by
# epftoolbox2.information_set.source_rules to decide when an OpenMeteoSource column is actually known (the
# source itself does not use this: its own bucket/run selection uses the nominal cutoff "D at run:00 UTC" only,
# see run_cutoff). Not a constructor parameter: it is a conservative, best-effort estimate, not a promise about
# any particular historical run.
#
# Basis: Open-Meteo's Single Runs API docs (https://open-meteo.com/en/docs/single-runs-api) say "typically 4-6
# hours for global models (e.g. ECMWF IFS, GFS) and 1-3 hours for regional models"; the model metadata endpoint
# (https://api.open-meteo.com/data/<model>/static/meta.json, field last_run_availability_time minus
# last_run_initialisation_time) gave these figures when checked on 2026-09-25: DWD ICON global 3.4h, ICON-EU
# 2.9h, ICON-D2 1.4h; NCEP GFS 5.9h (013) / 6.9h (025); ECMWF IFS 6.9h (9 km) / 8.3h (0.25 deg) / 8.6h (0.4 deg);
# JMA GSM 9.6h. Round, conservative (never lower than measured) values are used below.
_PUBLICATION_DELAY_H: Tuple[Tuple[str, float], ...] = (
    ("icon_", 4.0),
    ("gfs_", 5.0),
    ("ecmwf_ifs", 8.0),
    ("jma_", 10.0),
)
_DEFAULT_PUBLICATION_DELAY_H = 10.0


def publication_delay_hours(model: str) -> float:
    """Typical hours from `model`'s run initialisation to its availability (see `_PUBLICATION_DELAY_H`)."""
    for prefix, delay in _PUBLICATION_DELAY_H:
        if model.startswith(prefix):
            return delay
    return _DEFAULT_PUBLICATION_DELAY_H


# Best-effort synoptic run-hour cycles (UTC), for a clear api="single_runs" construction-time error naming valid
# hours (see OpenMeteoSource.run). Approximate and not exhaustive: a `model` not matched here is not checked
# here at all, and simply gets whatever the Single Runs API itself returns for that run (an unavailable run is
# reported per row like any other, see OpenMeteoBase._report_coverage) — it is never silently accepted as
# "valid" by this table, only left unchecked by it.
_RUN_HOURS_H: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("icon_d2", tuple(range(0, 24, 3))),
    ("icon_eu", tuple(range(0, 24, 3))),
    ("icon_", (0, 6, 12, 18)),
    ("ecmwf_ifs", (0, 6, 12, 18)),
    ("gfs_", (0, 6, 12, 18)),
    ("jma_", (0, 6, 12, 18)),
)


def known_run_hours(model: str) -> Optional[Tuple[int, ...]]:
    """`model`'s known run hours (UTC), or None if `model` isn't covered by `_RUN_HOURS_H`."""
    for prefix, hours in _RUN_HOURS_H:
        if model.startswith(prefix):
            return hours
    return None


@lru_cache(maxsize=128)
def _bucket_requirements(model: str, timezone: str, run: str, horizon: int) -> Tuple[Tuple[int, int], ...]:
    """(smallest, largest) `previous_dayK` needed for each lead 1..horizon, over a full year of hourly rows.

    The largest K may exceed MAX_PREVIOUS_DAY; callers reject such a horizon. Rows whose target wall-clock hour
    does not exist (the spring DST gap) are excluded: they never need a bucket at all, see `shift_local_days`.
    """
    rows = pd.date_range("2024-01-01", "2025-01-01", freq="1h", tz="UTC", inclusive="left")
    cutoff = run_cutoff(rows, timezone, run)
    needs = []
    for lead in range(1, horizon + 1):
        valid = shift_local_days(rows, timezone, lead)
        chosen = select_buckets(model, valid, cutoff, lead, lead + MAX_PREVIOUS_DAY + 2)
        usable = chosen[~valid.isna()]
        if (usable == 0).any():
            raise ValueError(f"run={run!r} leaves no usable previous_dayK for d+{lead}")
        needs.append((int(usable.min()), int(usable.max())))
    return tuple(needs)


def select_buckets(
    model: str,
    valid: pd.DatetimeIndex,
    cutoff: pd.DatetimeIndex,
    lead: int,
    max_bucket: int,
) -> np.ndarray:
    """Smallest K in [lead, max_bucket] per cell whose value only depends on runs initialised at or before the
    row's own cutoff; 0 if none (including a `valid` cell that is `NaT`, see `shift_local_days`).

    `previous_dayK` at valid time T holds the latest run initialised at or before T - 24K h (Open-Meteo Previous
    Runs docs). With a stored step s > 1 h the hourly value also depends on stored points up to ceil_s(T) + s, so
    the latest run that can influence it was initialised at or before ceil_s(T) + s - 24K h.
    """
    chosen = np.zeros(len(valid), dtype=np.int64)
    for bucket in range(lead, max_bucket + 1):
        step = stored_step_hours(model, bucket)
        reach_end = valid if step == 1 else valid.ceil(f"{step}h") + pd.Timedelta(hours=step)
        latest_init = reach_end - pd.Timedelta(days=bucket)
        ok = np.asarray(latest_init <= cutoff) & (chosen == 0)
        chosen[ok] = bucket
    return chosen


class OpenMeteoSource(OpenMeteoBase):
    """
    Open-Meteo weather forecasts as they were known when the day-ahead forecast was issued.

    Row t belongs to local day D in `timezone`. Its `{variable}_d+i` columns hold the forecast for the same
    local wall-clock hour, i calendar days later. The **cutoff** is `D` at `run`:00 UTC (`run` is a UTC hour,
    e.g. ``"00"``): every value on D's rows comes only from a model run initialised at or before that instant.
    The source itself has no notion of a market's real issue time or a model's publication delay — that
    real-world lag is a model-side concern, handled by ``epftoolbox2.information_set`` (see
    ``publication_delay_hours``), not by this source's own run/bucket selection.

    ``api="previous_runs"`` (default) uses the `Open-Meteo Previous Runs API
    <https://open-meteo.com/en/docs/previous-runs-api>`_, which stores each model's forecast at fixed lead-time
    offsets: for each row and lead i, the source picks the smallest `previous_dayK` (K >= i) bucket whose value
    depends only on runs initialised at or before the cutoff (see ``select_buckets``); some hours therefore come
    from a run well before D. ``api="single_runs"`` uses the `Single Runs API
    <https://open-meteo.com/en/docs/single-runs-api>`_, which stores every run separately: the source requests
    exactly the run initialised at the cutoff and uses it for every row and lead of D. If the archive lacks that
    run, the affected cells are NaN and the coverage report warns — there is no fallback to another run.

    Pass ``locations`` instead of ``latitude``/``longitude`` to combine several locations into one source (a
    weighted average, per-location columns, or both) — see the ``locations`` parameter below.

    Example:
        >>> source = OpenMeteoSource(
        ...     latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", horizon=5, prefix="warsaw"
        ... )
        >>> df = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-02-01", tz="UTC"))
    """

    API_URLS = {
        "previous_runs": "https://previous-runs-api.open-meteo.com/v1/forecast",
        "single_runs": "https://single-runs-api.open-meteo.com/v1/forecast",
    }
    UNAVAILABLE_RUN = "requested model run is not available"

    VALID_TEMPORAL_RESOLUTIONS = {None, "native", "hourly_3", "hourly_6"}
    SCHEMA_VERSION = 3

    def __init__(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        locations: Optional[Union[str, Dict[str, Any]]] = None,
        keep_locations: bool = False,
        columns: Optional[List[str]] = None,
        horizon: int = 5,
        model: str = "jma_seamless",
        temporal_resolution: Optional[str] = None,
        api: str = "previous_runs",
        run: str = "00",
        timezone: Optional[str] = None,
        prefix: str = "",
        strict: bool = False,
        proxy_url: Optional[str] = None,
    ):
        """
        Initialize Open-Meteo data source

        Args:
            latitude: Latitude coordinate. Mutually exclusive with ``locations``; set this (with ``longitude``)
                for one location, or ``locations`` for several.
            longitude: Longitude coordinate. See ``latitude``.
            locations: Several locations combined into one source, mutually exclusive with ``latitude``/
                ``longitude``. A dict ``{name: {"latitude", "longitude", "weight"?}}`` (or ``{name: (latitude,
                longitude)}`` / ``(latitude, longitude, weight)`` tuples), or the compact string
                ``"name:latitude:longitude[:weight], ..."`` for YAML/the TUI. Names use letters, digits and
                underscores and must be unique; weights, when given, must be positive and given for every
                location or none. With weights this returns one weighted series per variable,
                `{prefix}_{variable}_d+i` (weights normalised; a circular mean for `*direction*` columns;
                `weather_code` not averaged; NaN wherever any location is NaN). Without weights it returns only
                the per-location columns, `{prefix}_{name}_{variable}_d+i`. Every location is fetched by
                another ``OpenMeteoSource`` sharing every other parameter here, so all locations follow the
                same cutoff; weights are fixed and must be known at every issue day D (use a reference date
                before the sample starts, e.g. population or installed capacity, not end-of-sample figures).
            keep_locations: With weighted ``locations``, also return each location's own columns,
                `{prefix}_{name}_{variable}_d+i` (default: False). Has no effect without weights, since the
                per-location columns are then already the only output.
            columns: List of weather variables to fetch (default: DEFAULT_COLUMNS)
            horizon: Days ahead (d+1 .. d+horizon). Under ``api="previous_runs"`` the largest usable horizon
                depends on the model, `timezone` and `run` (5 for the default JMA model at run="00"); a larger
                value raises. Under ``api="single_runs"``, 1-14; leads beyond the run's own length come back NaN
                (reported, see ``strict``).
            model: Weather model to use (default: "jma_seamless")
            temporal_resolution: Temporal resolution of the data. One of:
                None (default) — hourly (1 h), no parameter sent to API;
                "hourly_3" — 3-hourly; "hourly_6" — 6-hourly;
                "native" — model-native resolution (inferred from response).
            api: "previous_runs" (default, long archive, bucket-based) or "single_runs" (one exact run per day,
                shorter archive for most models) — see the class docstring.
            run: UTC hour ("HH", "00".."23") of the model run the leak-free cutoff is anchored to: `D`'s cutoff
                is `D` at `run`:00 UTC. Pick the latest run you know is published before your real issue time —
                see ``epftoolbox2.information_set.source_rules``/``publication_delay_hours`` for typical
                publication delays by model family.
            timezone: Market time zone. Defines the local calendar day D of each row and the wall-clock hour
                that `d+i` refers to. None (default) inherits ``DataPipeline.timezone``; there is no other
                default, and fetching without either raises: a market outside CET that relied on one would
                compute the cutoff in the wrong zone.
            prefix: Prefix to add to column names (default: "")
            strict: Raise instead of warning when a returned column has missing values.
            proxy_url: Proxy URL (e.g., 'http://proxy.example.com:8080')
        """
        self.latitude = latitude
        self.longitude = longitude
        self.locations = locations
        self.keep_locations = keep_locations
        self.columns = list(columns) if columns else list(self.DEFAULT_COLUMNS)
        self.horizon = horizon
        self.model = model
        self.temporal_resolution = temporal_resolution
        self.api = api
        self.run = run
        self.timezone = timezone
        self.prefix = prefix
        self.strict = strict
        self.proxy_url = proxy_url

        self._setup()

        self._buckets: List[int] = []
        self._validate_config()

    def _validate_config(self) -> bool:
        """Validate the configuration parameters and, for `api="previous_runs"`, work out which buckets to use."""
        self._validate_common(max_horizon=MAX_SINGLE_RUNS_HORIZON if self.api == "single_runs" else None)

        if self.temporal_resolution not in self.VALID_TEMPORAL_RESOLUTIONS:
            raise ValueError(
                f"temporal_resolution must be one of {self.VALID_TEMPORAL_RESOLUTIONS}, "
                f"got {self.temporal_resolution!r}"
            )

        if self.api == "single_runs":
            known_hours = known_run_hours(self.model)
            if known_hours is not None and parse_run(self.run) not in known_hours:
                hours = ", ".join(f"{h:02d}" for h in known_hours)
                raise ValueError(f"{self.model!r} has no run at {self.run!r} UTC; its runs are at {hours} UTC")
            return True

        if self.horizon > MAX_PREVIOUS_DAY:
            raise ValueError(
                f"Horizon must be at most {MAX_PREVIOUS_DAY}: the Previous Runs API serves previous_day1.."
                f"previous_day{MAX_PREVIOUS_DAY} only, got {self.horizon}"
            )

        # The buckets depend on the market time zone, which may not be known yet (it can be inherited from
        # DataPipeline.timezone later): resolve them now if it is already known, else lazily at fetch() time.
        if self.timezone is not None:
            self._resolve_buckets(self.timezone)
        return True

    def _resolve_buckets(self, timezone: str) -> None:
        """Work out which previous_dayK buckets to request, for a now-known market `timezone`."""
        needs = _bucket_requirements(self.model, timezone, self.run, self.horizon)
        worst = max(high for _, high in needs)
        if worst > MAX_PREVIOUS_DAY:
            first_bad = next(lead for lead, (_, high) in enumerate(needs, 1) if high > MAX_PREVIOUS_DAY)
            raise ValueError(
                f"horizon={self.horizon} cannot be served without look-ahead: d+{first_bad} needs "
                f"previous_day{needs[first_bad - 1][1]} for some hours (model={self.model!r}, run={self.run!r} "
                f"UTC, timezone={timezone!r}), but Open-Meteo serves previous_day1..{MAX_PREVIOUS_DAY} only. Use "
                f"horizon<={first_bad - 1}, an earlier run if this model publishes one, or api='single_runs' "
                "(horizon up to 14, leads beyond the run's own length come back NaN)."
            )
        low = min(lo for lo, _ in needs)
        # One spare bucket lets live runs made before the cutoff fall back further (see _build_features).
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

        if self._locations:
            return self._fetch_locations(start, end)

        if self.api == "single_runs":
            return self._fetch_single_runs(start, end)
        return self._fetch_previous_runs(start, end)

    # -- api="previous_runs" ---------------------------------------------------

    def _fetch_previous_runs(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # `timezone` may have been unset at construction and inherited from DataPipeline.timezone since (see
        # `market_timezone`); the buckets could not be resolved until now in that case.
        tz = self._require_market_timezone()
        if not self._buckets:
            self._resolve_buckets(tz)

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
        return self._parse_response(self._request(self.API_URLS["previous_runs"], params))

    def _parse_response(self, response_data: Dict) -> pd.DataFrame:
        """Response → DataFrame indexed by UTC valid time, one float column per `{variable}_previous_dayK`."""
        return parse_hourly(response_data)

    def _build_features(self, table: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
        """Map the bucket table to `{variable}_d+{i}` columns for rows in [start, end).

        Returns the features and a boolean array marking final rows (cutoff passed). Rows whose cutoff is still
        in the future (live runs) use the runs published by now instead: they are provisional and change when
        fetched again later.
        """
        rows = table.index[(table.index >= start) & (table.index < end)]
        n = len(rows)
        features = {}

        cutoff, final = self._cutoff_instants(rows)
        steps = np.diff(table.index.asi8)
        tolerance = pd.Timedelta(int(steps.min()) // 2 if len(steps) else 0)
        max_bucket = max(self._buckets)
        for lead in range(1, self.horizon + 1):
            target = shift_local_days(rows, self.market_timezone, lead)
            positions = table.index.get_indexer(target, method="nearest", tolerance=tolerance)
            valid = pd.DatetimeIndex(np.where(positions >= 0, table.index.asi8[positions], np.iinfo(np.int64).min))
            valid = valid.tz_localize("UTC")
            buckets = select_buckets(self.model, valid, cutoff, lead, max_bucket)
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

    # -- api="single_runs" -------------------------------------------------------

    def _fetch_single_runs(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        tz = self._require_market_timezone()
        rows = pd.date_range(start.ceil("1h"), end, freq="1h", inclusive="left")
        if rows.empty:
            return pd.DataFrame()

        started = time.time()
        cutoff, final = self._cutoff_instants(rows)
        targets = {lead: shift_local_days(rows, tz, lead) for lead in range(1, self.horizon + 1)}
        values = {
            f"{column}_d+{lead}": np.full(len(rows), np.nan)
            for lead in range(1, self.horizon + 1)
            for column in self.columns
        }

        groups = pd.Series(np.arange(len(rows))).groupby(cutoff.asi8).indices
        self.logger.info(f"{self._label}: Start downloading {len(groups)} model runs")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"[cyan]{self._label}: Downloading...", total=len(groups))
            for init_ns, positions in groups.items():
                init = pd.Timestamp(init_ns, tz="UTC")
                progress.update(task, description=f"[cyan]{self._label}: run {init:%Y-%m-%d %H}z")
                self._fill_from_run(init, positions, targets, values)
                progress.advance(task)

        result = pd.DataFrame(values, index=rows)
        result.index.name = "datetime"
        result = self._finish(result, final)
        self.logger.info(f"{self._label}: Download completed in {time.time() - started:.2f} sec")
        return result

    def _fill_from_run(
        self,
        init: pd.Timestamp,
        positions: np.ndarray,
        targets: Dict[int, pd.DatetimeIndex],
        values: Dict[str, np.ndarray],
    ) -> None:
        """Fill the rows at `positions` from exactly the run initialised at `init`.

        No fallback to another run: a valid time the run doesn't cover (missing from the archive, or beyond the
        run's own length) stays NaN and is reported by `_report_coverage`/`strict` like any other gap.
        """
        run = self._fetch_run(init)
        if run is None or run.empty:
            return
        steps = np.diff(run.index.asi8)
        tolerance = pd.Timedelta(int(steps.min()) // 2 if len(steps) else 0)
        for lead, target in targets.items():
            valid = target[positions]
            idx = run.index.get_indexer(valid, method="nearest", tolerance=tolerance)
            take = idx >= 0
            if not take.any():
                continue
            for column in self.columns:
                if column in run.columns:
                    values[f"{column}_d+{lead}"][positions[take]] = run[column].to_numpy()[idx[take]]

    def _fetch_run(self, init: pd.Timestamp) -> Optional[pd.DataFrame]:
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": list(self.columns),
            "models": self.model,
            "timezone": "GMT",
            "run": init.strftime("%Y-%m-%dT%H:%M"),
            "forecast_days": min(16, self.horizon + 3),
        }
        if self.temporal_resolution is not None:
            params["temporal_resolution"] = self.temporal_resolution
        data = self._request(self.API_URLS["single_runs"], params, unavailable=self.UNAVAILABLE_RUN)
        if data is None:
            self.logger.info(f"{self._label}: run {init:%Y-%m-%d %H}z of {self.model} is not in the archive")
            return None
        return parse_hourly(data)

    # -- caching ------------------------------------------------------------

    def get_cache_config(self) -> dict:
        return {
            "source_type": "open_meteo",
            "schema_version": self.SCHEMA_VERSION,
            **self._common_cache_config(),
            "temporal_resolution": self.temporal_resolution,
        }
