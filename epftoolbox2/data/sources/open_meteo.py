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
# What `horizon=None` means under api="single_runs": every lead up to 14 exists, but a run is often much shorter
# (ICON 7.5 days, JMA's 06/18 runs 5.5), so the cap is a modest, always-filled default rather than the maximum.
DEFAULT_SINGLE_RUNS_HORIZON = 5

# Deepest `previous_dayK` a model has data in. ICON-D2/EU and ICON global stop at their forecast length (verified
# 2026-10: the deeper buckets come back entirely null). Models not listed here are served down to MAX_PREVIOUS_DAY.
_MAX_PREVIOUS_DAY_BY_MODEL: Tuple[Tuple[str, int], ...] = (
    ("icon_d2", 1),
    ("icon_eu", 4),
    ("icon_", 6),
)

# Forecast length (hours) of every run a model has, by UTC initialisation hour. The Previous Runs API fills
# `previous_dayK` at valid time T from the latest run initialised at or before T - 24K h *that reaches T*, so the
# 06/18 UTC runs of ICON, ECMWF IFS and JMA (shorter than the 00/12 UTC ones) drop out of the deep buckets. Where
# a length is not known precisely, the longer (more runs, later init) value is used: a finer run grid can only
# make the cutoff check stricter. Unknown models get an hourly grid of unlimited runs, the strictest case.
_RUN_LENGTH_H: Tuple[Tuple[str, Dict[int, int]], ...] = (
    ("icon_d2", {hour: 48 for hour in range(0, 24, 3)}),
    ("icon_eu", {0: 120, 6: 120, 12: 120, 18: 120}),
    # icon_seamless blends ICON-D2 into the global and EU runs: inside the D2 area (e.g. Berlin) `previous_day1`
    # holds the newest D2 run, which starts every 3 h (verified against Single Runs 2026-10), outside it the
    # 00/06/12/18 UTC runs. The 3-hourly grid is the strictest of the two.
    ("icon_seamless", {hour: 180 if hour in (0, 12) else 120 if hour in (6, 18) else 48 for hour in range(0, 24, 3)}),
    ("icon_", {0: 180, 6: 120, 12: 180, 18: 120}),
    ("ecmwf_ifs", {0: 360, 6: 144, 12: 360, 18: 144}),
    ("gfs_", {0: 384, 6: 384, 12: 384, 18: 384}),
    ("jma_", {0: 264, 6: 132, 12: 264, 18: 132}),
)

# Time step (hours) of the points `previous_dayK` is stored at, as ((max_lead_h, step_h), ...) segments; the first
# segment whose max_lead_h >= 24K + 6 applies (None = no limit). This is the step of the Previous Runs files
# themselves, which is what `temporal_resolution="native"` returns, and not the output step of the model: the
# files of GFS and ICON are hourly (interpolated within each run when the run is stored), so their hourly values
# are pieces of single runs. JMA GSM (6 h) and ECMWF IFS 0.25 (3 h) are stored at their own coarser points and
# interpolated when read, with a 4-point Hermite spline (verified against Single Runs: it reproduces the hourly
# values of a run from its points). The points around one hour then come from different runs, so the hourly
# value at T depends on the points up to ceil_s(T) + s, up to two steps after T (see `select_buckets`).
# Unknown models get the conservative 6 h. Changing this table changes outputs: bump SCHEMA_VERSION.
_STORED_STEP_H: Tuple[Tuple[str, Tuple[Tuple[Optional[int], int], ...]], ...] = (
    ("ecmwf_ifs025", ((None, 3),)),
    ("ecmwf_ifs04", ((None, 3),)),
    ("ecmwf_aifs", ((None, 6),)),
    ("ecmwf_ifs", ((None, 1),)),
    ("icon_", ((None, 1),)),
    ("gfs_", ((None, 1),)),
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


def max_previous_day(model: str) -> int:
    """Deepest `previous_dayK` bucket that has data for `model` (see `_MAX_PREVIOUS_DAY_BY_MODEL`)."""
    for prefix, bucket in _MAX_PREVIOUS_DAY_BY_MODEL:
        if model.startswith(prefix):
            return bucket
    return MAX_PREVIOUS_DAY


def previous_run_lengths(model: str) -> Optional[Dict[int, int]]:
    """Forecast length (hours) of each run the Previous Runs API holds for `model`, by UTC init hour (see
    `_RUN_LENGTH_H`), or None for a model not covered there (then every hourly run counts, the strictest case)."""
    for prefix, lengths in _RUN_LENGTH_H:
        if model.startswith(prefix):
            return lengths
    return None


# Hours between a model run's initialisation and its availability, by model family. Used only by
# epftoolbox2.information_set.source_rules to decide when an OpenMeteoSource column is actually known (the
# source itself does not use this: its own bucket/run selection uses the nominal cutoff "D at run:00 UTC" only,
# see run_cutoff). Not a constructor parameter: it is a conservative estimate, not a promise about any particular
# historical run. The values err on the late side, so that the information-set check does not accept a run that
# may not have been published at the issue time.
#
# Basis: Open-Meteo's Single Runs API docs (https://open-meteo.com/en/docs/single-runs-api) say "typically 4-6
# hours for global models (e.g. ECMWF IFS, GFS) and 1-3 hours for regional models"; the model metadata endpoint
# (https://api.open-meteo.com/data/<model>/static/meta.json, field last_run_availability_time minus
# last_run_initialisation_time) gave these figures when checked on 2026-10-11: DWD ICON global 4.3h, ICON-EU
# 3.0h, ICON-D2 1.4h; NCEP GFS 5.5h (013) / 6.4h (025); ECMWF IFS 6.9h (9 km) / 8.6h (0.25 deg); JMA GSM 9.5h
# (9.53h, the 18 UTC run). The figures move a little from run to run, so the table rounds them up: ICON 4.5h,
# GFS 6.5h, ECMWF IFS 9.0h, JMA 9.6h. JMA at 9.6h puts its 00 UTC run at 09:36 UTC, i.e. 11:36 CEST, which decides
# whether the default weather model passes an issue time just before the 12:00 CET gate closure in summer. That is
# why the default issue time is 11:45.
_PUBLICATION_DELAY_H: Tuple[Tuple[str, float], ...] = (
    ("icon_", 4.5),
    ("gfs_", 6.5),
    ("ecmwf_ifs", 9.0),
    ("jma_", 9.6),
)
_DEFAULT_PUBLICATION_DELAY_H = 10.0


def publication_delay_hours(model: str) -> float:
    """Conservative hours from `model`'s run initialisation to its availability (see `_PUBLICATION_DELAY_H`)."""
    for prefix, delay in _PUBLICATION_DELAY_H:
        if model.startswith(prefix):
            return delay
    return _DEFAULT_PUBLICATION_DELAY_H


# Best-effort synoptic run-hour cycles (UTC) of the Single Runs API, for a clear api="single_runs" construction-time
# error naming valid hours (see OpenMeteoSource.run). Approximate and not exhaustive: a `model` not matched here is
# not checked here at all, and simply gets whatever the Single Runs API itself returns for that run (an unavailable
# run is reported per row like any other, see OpenMeteoBase._report_coverage) — it is never silently accepted as
# "valid" by this table, only left unchecked by it. The Previous Runs API has fewer runs for some models, see
# `_RUN_LENGTH_H`.
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


_HOUR_NS = 3600 * 10**9
_NO_RUN = np.iinfo(np.int64).max  # a stored point no run reaches: later than any cutoff


def _run_inits(model: str, point_hours: np.ndarray, bucket: int) -> np.ndarray:
    """Init (hours since the epoch) of the run `previous_day{bucket}` holds at each stored point.

    That is the latest run initialised at or before `point - 24 * bucket` h whose forecast reaches the point (the
    Previous Runs API takes every point from the run that is `bucket` days older, and the shorter 06/18 UTC runs
    of some models do not reach the deep buckets). `_NO_RUN` where no run within two days does.
    """
    lengths = previous_run_lengths(model)
    reach = np.full(24, np.iinfo(np.int64).max, dtype=np.int64) if lengths is None else np.zeros(24, dtype=np.int64)
    for hour, length in (lengths or {}).items():
        reach[hour] = length

    result = np.full(len(point_hours), _NO_RUN, dtype=np.int64)
    pending = np.ones(len(point_hours), dtype=bool)
    newest = point_hours - 24 * bucket
    for back in range(48):
        init = newest - back
        found = pending & (point_hours - init <= reach[init % 24])
        result[found] = init[found]
        pending &= ~found
        if not pending.any():
            break
    return result


def newest_run_init(model: str, valid: pd.DatetimeIndex, bucket: int) -> np.ndarray:
    """Init (UTC nanoseconds since the epoch) of the newest run that can influence `previous_day{bucket}` at each
    `valid` time, or the largest int64 where no run reaches it. `NaT` in `valid` also gives the largest int64.

    A stored point at P holds the run described in `_run_inits`. With a stored step of one hour the value at T is
    the point at T. With a coarser step s the file holds a point every s hours and Open-Meteo interpolates hourly
    values from the four points around T (a 4-point Hermite spline): a time exactly on the grid is its own point,
    any other time reads the points floor_s(T) - s, floor_s(T), floor_s(T) + s and floor_s(T) + 2s, so the newest
    run reaches ceil_s(T) + s, up to two steps after T.
    """
    missing = np.asarray(valid.isna())
    hours = np.where(missing, 0, valid.asi8 // _HOUR_NS)
    step = stored_step_hours(model, bucket)
    if step == 1:
        newest = _run_inits(model, hours, bucket)
    else:
        floor = hours - hours % step
        on_grid = hours % step == 0
        newest = _run_inits(model, floor, bucket)
        for offset in (-1, 1, 2):
            reach = np.maximum(newest, _run_inits(model, floor + offset * step, bucket))
            newest = np.where(on_grid, newest, reach)
    newest = np.where(newest == _NO_RUN, _NO_RUN, newest * _HOUR_NS)
    return np.where(missing, _NO_RUN, newest)


def select_buckets(
    model: str,
    valid: pd.DatetimeIndex,
    cutoff: pd.DatetimeIndex,
    lead: int,
    max_bucket: int,
) -> np.ndarray:
    """Smallest K in [lead, max_bucket] per cell whose value only depends on runs initialised at or before the
    row's own cutoff; 0 if none (including a `valid` cell that is `NaT`, see `shift_local_days`).

    `max_bucket` is capped at the deepest bucket the model has data in (`max_previous_day`). Which runs a bucket's
    value depends on is `newest_run_init`: `previous_dayK` at valid time T holds the latest run initialised at or
    before T - 24K h (Open-Meteo Previous Runs docs), taken point by point, plus the interpolation reach of a
    coarsely stored model.
    """
    chosen = np.zeros(len(valid), dtype=np.int64)
    cutoff_ns = np.asarray(cutoff.asi8)
    for bucket in range(lead, min(max_bucket, max_previous_day(model)) + 1):
        ok = (newest_run_init(model, valid, bucket) <= cutoff_ns) & (chosen == 0)
        chosen[ok] = bucket
    return chosen


@lru_cache(maxsize=512)
def _lead_buckets(model: str, timezone: str, run: str, lead: int) -> Optional[Tuple[int, int]]:
    """(smallest, largest) `previous_dayK` used for d+`lead` over a full year of hourly rows, or None if some hour
    of D+`lead` cannot be served leak-free by any bucket the model has.

    Rows whose target wall-clock hour does not exist (the spring DST gap) are excluded: they never need a bucket
    at all, see `shift_local_days`. A year covers both UTC offsets of a DST zone.
    """
    rows = pd.date_range("2024-01-01", "2025-01-01", freq="1h", tz="UTC", inclusive="left")
    cutoff = run_cutoff(rows, timezone, run)
    valid = shift_local_days(rows, timezone, lead)
    chosen = select_buckets(model, valid, cutoff, lead, MAX_PREVIOUS_DAY)
    usable = chosen[~valid.isna()]
    if (usable == 0).any():
        return None
    return int(usable.min()), int(usable.max())


@lru_cache(maxsize=128)
def max_previous_runs_horizon(model: str, timezone: str, run: str) -> int:
    """Largest `horizon` without look-ahead under api="previous_runs", for `model`, market `timezone` and `run`.

    Every local hour of D+horizon must be served by a `previous_dayK` bucket (K <= 7, and within what the model
    has) that only depends on runs initialised at or before D at `run`:00 UTC. `previous_dayK` at T is the run
    24K h older, so d+7 is only partly servable and never fully for an early `run`. 0 if even d+1 cannot be served.
    """
    limit = 0
    for lead in range(1, MAX_PREVIOUS_DAY + 1):
        if _lead_buckets(model, timezone, run, lead) is None:
            break
        limit = lead
    return limit


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
    from a run well before D. How many days ahead that works for depends on the model, `run` and `timezone`
    (``max_previous_runs_horizon``): ``horizon=None`` (default) takes the largest, an explicit larger one
    raises. ``api="single_runs"`` uses the `Single Runs API
    <https://open-meteo.com/en/docs/single-runs-api>`_, which stores every run separately: the source requests
    exactly the run initialised at the cutoff and uses it for every row and lead of D. If the archive lacks that
    run, the affected cells are NaN and the coverage report warns — there is no fallback to another run.

    Pass ``locations`` instead of ``latitude``/``longitude`` to combine several locations into one source:
    per-location columns, one or more aggregated series per variable (``agg``), or both (``keep_locations``) —
    see the ``locations`` and ``agg`` parameters below.

    Example:
        >>> source = OpenMeteoSource(
        ...     latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", prefix="warsaw"
        ... )
        >>> df = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-02-01", tz="UTC"))
    """

    API_URLS = {
        "previous_runs": "https://previous-runs-api.open-meteo.com/v1/forecast",
        "single_runs": "https://single-runs-api.open-meteo.com/v1/forecast",
    }
    UNAVAILABLE_RUN = "requested model run is not available"

    VALID_TEMPORAL_RESOLUTIONS = {None, "native", "hourly_3", "hourly_6"}
    SCHEMA_VERSION = 7

    def __init__(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        locations: Optional[Union[str, Dict[str, Any]]] = None,
        agg: Optional[Union[str, List[str]]] = None,
        keep_locations: bool = False,
        columns: Optional[List[str]] = None,
        horizon: Optional[int] = None,
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
                ``longitude``. A dict ``{name: {"latitude", "longitude", "weight"?}}``, or the compact string
                ``"name:latitude:longitude[:weight], ..."`` for YAML/the TUI. Names use letters, digits and
                underscores and must be unique; weights, when given, must be positive and given for every
                location or none, and are only meaningful with ``agg="mean"`` (or a list containing it) — see
                ``agg``. With ``agg=None`` (default), only the per-location columns are returned,
                `{prefix}_{name}_{variable}_d+i`. Every location is fetched by another ``OpenMeteoSource``
                sharing every other parameter here, so all locations follow the same cutoff; weights are fixed
                and must be known at every issue day D (use a reference date before the sample starts, e.g.
                population or installed capacity, not end-of-sample figures).
            agg: Only with ``locations`` (several points); raises with a single ``latitude``/``longitude``
                point. ``None`` (default), ``"mean"``, ``"median"``, ``"min"``, ``"max"``, ``"sum"``, or a list
                of these. ``None``: no aggregation, only the per-location columns (see ``locations``). A bare
                string or a list: one aggregated series per variable and entry, `{prefix}_{variable}_{agg}_d+i`
                (a bare string and a one-element list containing it name the same output the same way).
                ``"mean"`` is a weighted mean when every location has a ``weight`` (normalised to sum to 1),
                else a plain mean; giving weights when ``agg`` doesn't include ``"mean"`` raises, and in a list
                only ``"mean"`` is weighted, the rest are always unweighted. A `*direction*` column (e.g.
                `wind_direction_10m`) uses the circular (weighted) mean for ``"mean"`` and is skipped entirely
                for every other aggregation, which has no meaningful min/max/median/sum for a circular
                quantity. `weather_code` (an ordinal WMO code) is aggregated like any other numeric column. A
                row is NaN wherever any location is NaN there.
            keep_locations: With ``agg`` set, also return each location's own columns,
                `{prefix}_{name}_{variable}_d+i` (default: False). Raises if ``agg`` is ``None``: the
                per-location columns are then already the only output.
            columns: List of weather variables to fetch (default: DEFAULT_COLUMNS)
            horizon: Days ahead (d+1 .. d+horizon). ``None`` (default): under ``api="previous_runs"`` the
                largest horizon without look-ahead for `model`, `run` and `timezone` (see
                ``max_previous_runs_horizon``, 6 for the default JMA model at run="00" in Europe/Warsaw), under
                ``api="single_runs"`` 5. An explicit value above that limit raises under ``api="previous_runs"``
                and names the limit. Under ``api="single_runs"``, 1-14. Leads beyond the run's own length come
                back NaN (reported, see ``strict``).
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
        self.agg = agg
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

        if self.horizon is not None and self.horizon > MAX_PREVIOUS_DAY:
            raise ValueError(
                f"Horizon must be at most {MAX_PREVIOUS_DAY}: the Previous Runs API serves previous_day1.."
                f"previous_day{MAX_PREVIOUS_DAY} only, got {self.horizon}. Use api='single_runs' for a full week "
                "(horizon up to 14, leads beyond the run's own length come back NaN)."
            )

        # The buckets depend on the market time zone, which may not be known yet (it can be inherited from
        # DataPipeline.timezone later): resolve them now if it is already known, else lazily at fetch() time.
        if self.timezone is not None:
            self._resolve_buckets(self.timezone)
        return True

    def _effective_horizon(self) -> int:
        """The horizon actually fetched: `horizon`, or what ``horizon=None`` stands for (see ``horizon``)."""
        if self.horizon is not None:
            return self.horizon
        if self.api == "single_runs":
            return DEFAULT_SINGLE_RUNS_HORIZON
        return max_previous_runs_horizon(self.model, self._require_market_timezone(), self.run)

    def _resolve_buckets(self, timezone: str) -> None:
        """Work out which previous_dayK buckets to request, for a now-known market `timezone`."""
        limit = max_previous_runs_horizon(self.model, timezone, self.run)
        where = f"model={self.model!r}, run={self.run!r} UTC, timezone={timezone!r}"
        if limit == 0:
            raise ValueError(
                f"d+1 cannot be served without look-ahead ({where}): some of its hours need runs initialised after "
                f"D at {self.run}:00 UTC. Use a later run, or api='single_runs'."
            )
        horizon = limit if self.horizon is None else self.horizon
        if horizon > limit:
            raise ValueError(
                f"horizon={horizon} is above the largest horizon without look-ahead, {limit} ({where}): some hours "
                f"of d+{limit + 1} would need a run initialised after D at {self.run}:00 UTC, and "
                f"previous_day1..{MAX_PREVIOUS_DAY} are the only buckets Open-Meteo serves. Use horizon<={limit} "
                "(horizon=None picks it), a later run, or api='single_runs' for a full week (horizon up to 14, "
                "leads beyond the run's own length come back NaN)."
            )
        needs = [_lead_buckets(self.model, timezone, self.run, lead) for lead in range(1, horizon + 1)]
        low = min(lo for lo, _ in needs)
        worst = max(high for _, high in needs)
        # One spare bucket lets live runs made before the cutoff fall back further (see _build_features).
        self._buckets = list(range(low, min(worst + 1, max_previous_day(self.model)) + 1))

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
        horizon = self._effective_horizon()
        chunks = self._generate_chunks(start, end + DateOffset(days=horizon + 1), months=3)

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
        for lead in range(1, self._effective_horizon() + 1):
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
        horizon = self._effective_horizon()
        targets = {lead: shift_local_days(rows, tz, lead) for lead in range(1, horizon + 1)}
        values = {
            f"{column}_d+{lead}": np.full(len(rows), np.nan)
            for lead in range(1, horizon + 1)
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
            "forecast_days": min(16, self._effective_horizon() + 3),
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
