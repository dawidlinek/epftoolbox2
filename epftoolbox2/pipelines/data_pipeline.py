from datetime import date, datetime
from typing import List, Optional, Tuple, Union, Dict, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import hashlib
import json
import os
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import importlib

from epftoolbox2.data._market_timezone import bind as _bind_timezone
from epftoolbox2.data.sources.base import DataSource
from epftoolbox2.data.transformers.base import Transformer
from epftoolbox2.data.transformers.timezone import TimezoneTransformer
from epftoolbox2.data.validators.base import Validator
from epftoolbox2.data.validators.result import ValidationResult
from epftoolbox2.data.cache_manager import (
    CACHE_FORMAT_VERSION,
    CacheManager,
    align_to_local_days,
    atomic_write,
    clip_to_range,
    hold_lock,
    make_file_lock,
    settled_until_of,
    unsettled_boundary,
    without_ranges,
    without_settled_until,
)
from epftoolbox2.data.refresh import Refresh, RefreshError, RefreshRecord, SourceRefresh, last_valid, redact
from epftoolbox2.data.sources import _http
from epftoolbox2.data.sources._http import SourceTimeout
from epftoolbox2.logging import get_logger
from epftoolbox2._clock import current_clock, wall_time
from epftoolbox2._date_utils import _DATE_PATTERN, epoch_ns
from epftoolbox2._frame_csv import EXACT_FLOATS
from ._yaml import dump_yaml, to_yaml_value

logger = get_logger(__name__)

# Attributes that are runtime objects, not configuration.
_RUNTIME_ATTRIBUTES = {"session", "console", "logger"}
# Configuration that never changes the fetched data and must not end up in cache files.
_SECRET_ATTRIBUTES = {"api_key"}
_UNSET_ENV_VAR = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")

COMPONENT_REGISTRY = {
    "sources": {
        "EntsoeSource": "epftoolbox2.data.sources.entsoe",
        "OpenMeteoSource": "epftoolbox2.data.sources.open_meteo",
        "CsvSource": "epftoolbox2.data.sources.csv",
        "CalendarSource": "epftoolbox2.data.sources.calendar",
        "EpftoolboxBenchmarkSource": "epftoolbox2.data.sources.benchmark",
        "PseSource": "epftoolbox2.data.sources.pse",
        "EnergyChartsSource": "epftoolbox2.data.sources.energy_charts",
        "ForecastSource": "epftoolbox2.data.sources.forecast",
    },
    "transformers": {
        "TimezoneTransformer": "epftoolbox2.data.transformers.timezone",
        "ResampleTransformer": "epftoolbox2.data.transformers.resample",
        "LagTransformer": "epftoolbox2.data.transformers.lag",
        "DSTNormalizeTransformer": "epftoolbox2.data.transformers.dst_normalize",
        "DailyProfileTransformer": "epftoolbox2.data.transformers.daily_profile",
        "DailyStatsTransformer": "epftoolbox2.data.transformers.daily_stats",
        "PeriodStatsTransformer": "epftoolbox2.data.transformers.period_stats",
        "LambdaTransformer": "epftoolbox2.data.transformers.lambda_transformer",
        "ImputeTransformer": "epftoolbox2.data.transformers.impute",
        "SolarGeometryTransformer": "epftoolbox2.data.transformers.solar_geometry",
        "RollingStatsTransformer": "epftoolbox2.data.transformers.rolling_stats",
    },
    "validators": {
        "EdaValidator": "epftoolbox2.data.validators.eda",
        "DSTStructureValidator": "epftoolbox2.data.validators.dst_structure",
        "CoverageValidator": "epftoolbox2.data.validators.coverage",
        "PlausibilityValidator": "epftoolbox2.data.validators.plausibility",
    },
}


class DataValidationError(ValueError):
    """Raised by ``DataPipeline.run`` when a validator fails and ``on_invalid="raise"``."""


class DataPipeline:
    """Data pipeline that combines multiple sources and applies transformers.

    Args:
        sources: Data sources; their outputs are clipped to ``[start, end)`` and joined column-wise. Two sources
            producing the same column name raise ``ValueError`` (give one a prefix).
        transformers: Applied in order to the joined source output.
        validators: Run on the final frame (after the transformers).
        source_validators: Run on the joined source output before any transformer, e.g. to catch gaps that
            resampling would interpolate over.
        timezone: Market time zone (e.g. ``"Europe/Warsaw"``), the single place it needs to be set. ``start`` and
            a date-only ``end`` passed to ``run`` always cover a whole calendar day, local if set and UTC
            otherwise: ``start`` begins at that day's midnight and ``end`` runs through the end of its own day
            (inclusive). A ``datetime.date`` (what YAML makes of an unquoted ``end: 2024-01-31``) counts as a
            date-only string. If set, date strings and keywords passed to ``run`` are read as local calendar days and
            naive timestamps are local times. If ``None``, they are UTC calendar days and naive timestamps are
            UTC. Components whose own zone decides a local
            calendar day or an issue time — ``TimezoneTransformer.target_tz``,
            ``OpenMeteoSource.timezone`` and
            ``CalendarSource.timezone`` (but not ``CsvSource.timezone``, which is the input file's own zone, not
            the market's) — may leave it unset (``None``) and inherit ``timezone`` instead; an explicitly set,
            different zone still raises ``ValueError``, since a component computing its own local time in a
            different zone would silently disagree with it. A component that needs a zone and has neither raises
            when run.
        on_invalid: ``"warn"`` (default) logs failed validations and continues; ``"raise"`` raises
            ``DataValidationError`` after the failing validation stage. Results are kept in ``validation_results``.

    Example:
        >>> pipeline = DataPipeline(
        ...     sources=[CsvSource(...), EntsoeSource(...)],
        ...     transformers=[TimezoneTransformer()],
        ...     timezone="Europe/Warsaw",
        ... )
        >>> df = pipeline.run("2024-01-01", "2024-06-30")  # 2024-01-01 00:00 .. 2024-06-30 23:00 local

    Example:
        >>> pipeline = (
        ...     DataPipeline(timezone="Europe/Warsaw")
        ...     .add_source(EntsoeSource(country_code="PL", api_key="...", type=["load"]))
        ...     .add_source(OpenMeteoSource(latitude=52.23, longitude=21.01))
        ...     .add_transformer(TimezoneTransformer())
        ...     .add_validator(CoverageValidator(columns=["load_actual"], min_coverage=1.0))
        ... )
        >>> df = pipeline.run(start, end)
    """

    ON_INVALID = ("warn", "raise")

    def __init__(
        self,
        sources: Optional[List[DataSource]] = None,
        transformers: Optional[List[Transformer]] = None,
        validators: Optional[List[Validator]] = None,
        source_validators: Optional[List[Validator]] = None,
        timezone: Optional[str] = None,
        on_invalid: str = "warn",
    ):
        self.sources: List[DataSource] = sources or []
        self.transformers: List[Transformer] = transformers or []
        self.validators: List[Validator] = validators or []
        self.source_validators: List[Validator] = source_validators or []
        self.timezone = timezone
        self.on_invalid = on_invalid
        self.validation_results: Dict[str, ValidationResult] = {}
        # What the last `run` did, source by source (`RefreshRecord`); None before a run.
        self.last_refresh: Optional[RefreshRecord] = None
        self._validate_settings()

    def _validate_settings(self) -> None:
        if self.timezone is not None:
            try:
                ZoneInfo(self.timezone)
            except (ZoneInfoNotFoundError, ValueError, TypeError):
                raise ValueError(f"Invalid timezone: {self.timezone!r} (use an IANA name such as 'Europe/Warsaw')")
        self._bind_component_timezones()
        if self.on_invalid not in self.ON_INVALID:
            raise ValueError(f"on_invalid must be one of {self.ON_INVALID}, got {self.on_invalid!r}")

    def _bind_component_timezones(self) -> None:
        """Make ``self.timezone`` the single place the market zone needs to be written.

        Every ``TimezoneTransformer``/``CalendarSource``/``OpenMeteoSource`` whose own zone (``target_tz`` /
        ``timezone``) is left unset inherits ``self.timezone`` instead, resolved through
        ``_market_timezone.bind``: the component's private ``_pipeline_timezone`` attribute is set (never the
        public one, so ``save()`` keeps writing the value the user gave, and a YAML round trip re-resolves the
        same way given the same ``DataPipeline.timezone``); an explicitly set, different zone still raises.
        ``CsvSource.timezone`` is the input file's own zone, not the market's, and is not touched.

        ``EntsoeSource`` has no zone parameter of its own (nothing to conflict with, so ``own=None`` always):
        it only uses the bound zone to anchor a week-ahead daily value (A31, ``load_forecast_daily_min/max``)
        onto the market's own local day rather than this area's own — see ``EntsoeSource._daily_zone`` — which
        matters only when it is combined with another area in the same pipeline.

        ``EpftoolboxBenchmarkSource.index_timezone`` is different again: it is a fixed offset the source always
        has (derived from its ``dataset``, never inherited), so it can't be left unset to agree with
        ``self.timezone``. A mismatch there is never intentional — the source's UTC index already encodes the
        benchmark file's local wall-clock time through that exact offset, and re-interpreting it through a
        different (DST-observing) pipeline zone silently shifts every row's (date, hour) — so it raises too.
        """
        from epftoolbox2.data.sources.benchmark import EpftoolboxBenchmarkSource
        from epftoolbox2.data.sources.calendar import CalendarSource
        from epftoolbox2.data.sources.entsoe import EntsoeSource
        from epftoolbox2.data.sources.open_meteo import OpenMeteoSource

        for transformer in self.transformers:
            if isinstance(transformer, TimezoneTransformer):
                _bind_timezone(
                    transformer, transformer.target_tz, self.timezone, "TimezoneTransformer.target_tz", owner=self
                )

        for source in self.sources:
            if isinstance(source, (OpenMeteoSource, CalendarSource)):
                _bind_timezone(source, source.timezone, self.timezone, f"{type(source).__name__}.timezone", owner=self)
                # A source built with `locations` fetches through its own private per-location instances (see
                # OpenMeteoBase._configure_locations); their own `timezone` was copied from this source's at
                # construction (often None too), so each one needs the same binding.
                for child in getattr(source, "_location_sources", []):
                    where = f"{type(source).__name__}.timezone (via locations)"
                    _bind_timezone(child, child.timezone, self.timezone, where, owner=self)
            elif isinstance(source, EntsoeSource):
                _bind_timezone(
                    source, None, self.timezone, "EntsoeSource (for its week-ahead daily values)", owner=self
                )
            elif isinstance(source, EpftoolboxBenchmarkSource):
                if self.timezone is not None and self.timezone != source.index_timezone:
                    raise ValueError(
                        f"DataPipeline.timezone={self.timezone!r} does not match "
                        f"EpftoolboxBenchmarkSource({source.dataset!r}).index_timezone={source.index_timezone!r}. "
                        "That source's UTC index already encodes its file's local wall-clock time through this "
                        "fixed offset (see EpftoolboxBenchmarkSource's docs); a DataPipeline.timezone that "
                        "observes DST reads every row as a different (date, hour). Use "
                        f"timezone={source.index_timezone!r} for this pipeline."
                    )

    def add_source(self, source: DataSource) -> "DataPipeline":
        if not isinstance(source, DataSource):
            raise TypeError("source must be a DataSource instance")
        self.sources.append(source)
        return self

    def add_transformer(self, transformer: Transformer) -> "DataPipeline":
        if not isinstance(transformer, Transformer):
            raise TypeError("transformer must be a Transformer instance")
        self.transformers.append(transformer)
        return self

    def add_validator(self, validator: Validator) -> "DataPipeline":
        if not isinstance(validator, Validator):
            raise TypeError("validator must be a Validator instance")
        self.validators.append(validator)
        return self

    def add_source_validator(self, validator: Validator) -> "DataPipeline":
        """Add a validator that runs on the joined source output, before the transformers."""
        if not isinstance(validator, Validator):
            raise TypeError("validator must be a Validator instance")
        self.source_validators.append(validator)
        return self

    @staticmethod
    def _fetch_source(
        source: DataSource, start: pd.Timestamp, end: pd.Timestamp
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Timestamp]]:
        """``source.fetch(start, end)`` and the instant from which its rows are not settled (``DataSource``), or
        ``None`` if all of them are. The declaration is read and removed here, before any clip or concat can drop
        it or pass it on to the output."""
        df = source.fetch(start, end)
        if df is None:
            return None, None
        settled_until = settled_until_of(df)
        if settled_until is not None and settled_until >= end:
            settled_until = None
        return without_settled_until(df), settled_until

    def _fetch_with_cache(
        self, source: DataSource, start: pd.Timestamp, end: pd.Timestamp, cache_manager: CacheManager
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Timestamp]]:
        """The source's rows for ``[start, end)`` from the cache and the API, and the earliest instant a fetch
        declared its rows unsettled from (``None`` if none did; what the cache holds is settled)."""
        source_config = source.get_cache_config()
        if source_config is None:
            return self._fetch_source(source, start, end)

        source_class = _qualified_name(source)
        cache_key = cache_manager.get_cache_key(source_config, source_class)
        source_type = source_config.get("source_type", type(source).__name__)

        # Without a pipeline zone a source can still say which zone its one-day values are stamped at.
        day_zone = _cache_day_zone(source, self.timezone)

        def missing() -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
            # Holes are fetched from the start of their local day, see ``align_to_local_days``.
            return align_to_local_days(cache_manager.find_missing_ranges(cache_key, start, end), start, day_zone)

        # A full hit fetches nothing, so it holds the lock only inside each of its reads, never across them. The read comes
        # before the second look for holes: a file it finds broken leaves the metadata, and its range is then fetched.
        if not missing():
            cached = cache_manager.read_cached_data(cache_key, start, end)
            if not missing():
                logger.info(f"Cache: Full hit for {source_type} source")
                return (cached if cached is not None else pd.DataFrame()), None

        # Held around the fetch and its write: parallel processes wait for the one fetching this source, then read its
        # result. A fetch that raises (a timeout of the refresh's budget too) releases it as it leaves the block.
        with cache_manager.lock(cache_key):
            cached = cache_manager.read_cached_data(cache_key, start, end)
            missing_ranges = missing()

            if not missing_ranges:
                logger.info(f"Cache: Full hit for {source_type} source")
                return (cached if cached is not None else pd.DataFrame()), None

            if len(missing_ranges) == 1 and missing_ranges[0] == (start, end):
                logger.info(f"Cache: Miss for {source_type} source")
            else:
                logger.info(f"Cache: Partial hit for {source_type} source")

            parts = []
            unsettled_from = []
            # A range widened back to the start of its day also covers rows that are cached. Those are replaced only
            # by a refetch that returns rows (so a stale stamp inside it cannot stay next to the new one), and kept
            # when the source answers with nothing for the range (ENTSO-E's "No matching data" is an empty frame).
            replaced = []
            for missing_start, missing_end in missing_ranges:
                fresh_df, settled_until = self._fetch_source(source, missing_start, missing_end)
                if fresh_df is None or fresh_df.empty:
                    continue
                fresh_df = clip_to_range(fresh_df, missing_start, missing_end)
                if not fresh_df.empty:
                    replaced.append((missing_start, missing_end))
                # Records only the part the source actually returned, that is settled and that the source did not
                # declare unsettled; the rest is still returned now and fetched again next time. `timezone`
                # floors the unsettled-period cap to this pipeline's market zone's own local midnight (not UTC
                # midnight), so a later resumed fetch starts exactly there instead of strictly inside a source's
                # one-day period (e.g. ENTSO-E's A31 daily min/max) — a source that carries such a period's value
                # forward to cover its fetch's own start would otherwise re-stamp the same day's value a second
                # time on top of the one already cached.
                cache_manager.write_cache(
                    cache_key,
                    fresh_df,
                    missing_start,
                    missing_end,
                    source_config,
                    source_class,
                    timezone=self.timezone,
                    settled_until=settled_until,
                )
                if settled_until is not None:
                    unsettled_from.append(settled_until)
                parts.append(fresh_df)

            if cached is not None:
                parts.insert(0, without_ranges(cached, replaced))

        unsettled = min(unsettled_from) if unsettled_from else None
        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame(), unsettled
        # What is left of the cache and the refetched ranges are disjoint, so there are no duplicate timestamps.
        return pd.concat(parts, sort=False).sort_index(kind="stable"), unsettled  # columns in order of appearance

    def _parse_timestamp(self, ts: Union[str, date, pd.Timestamp], is_end: bool = False) -> pd.Timestamp:
        """UTC bound for ``run``. See the class docstring for how dates, keywords and timestamps are read."""
        tz = self.timezone or "UTC"
        if isinstance(ts, date) and not isinstance(ts, datetime):
            # A plain date, such as YAML makes of an unquoted `end: 2024-01-31`, names a whole day like "2024-01-31".
            ts = ts.isoformat()
        if isinstance(ts, str):
            day = _calendar_day(ts, self.timezone)
            if day is not None:
                if is_end:
                    # end is inclusive: the whole day is included (local if timezone is set, else UTC).
                    day += pd.Timedelta(days=1)
                return day.tz_localize(tz, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
        ts = pd.Timestamp(ts)
        return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize(tz).tz_convert("UTC")

    def resolve_range(
        self, start: Union[str, date, pd.Timestamp], end: Union[str, date, pd.Timestamp]
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """The half-open UTC range ``[start, end)`` that ``run(start, end)`` returns."""
        start_ts = self._parse_timestamp(start)
        end_ts = self._parse_timestamp(end, is_end=True)
        if end_ts <= start_ts:
            raise ValueError(f"End timestamp ({end_ts}) must be after start timestamp ({start_ts})")
        return start_ts, end_ts

    def run(
        self,
        start: Union[str, date, pd.Timestamp],
        end: Union[str, date, pd.Timestamp],
        cache: Union[bool, str] = False,
        *,
        budget: Optional[float] = None,
        deadline: Union[None, str, datetime, pd.Timestamp] = None,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
        on_failure: str = "raise",
    ) -> pd.DataFrame:
        """The sources' joined output for ``[start, end)`` (see ``resolve_range``), transformed and validated.

        Args:
            start, end: The range; see the class docstring for dates, keywords and timestamps.
            cache: ``False``, ``True`` (one cache per source, ``.cache/sources/``) or a path (one file for the whole
                source output); see the caching page of the docs.
            budget: Seconds the sources may take, from now on the clock (``epftoolbox2.clock``). The sources are asked
                one after another, each within what is left of it and of ``deadline``: a request is sent only while time
                is left, with its timeout cut to what is left, and a wait (a retry's backoff, a Retry-After, a rate
                limit) that would end at or after the deadline is never started. The source then raises
                ``SourceTimeout`` (``epftoolbox2.data.refresh``).
            deadline: An instant the sources must be done by; a naive one is a time of ``timezone`` (UTC without one).
            timeout: Seconds every HTTP request may take, in place of each source's own (60 for most).
            retries: How many times a failed HTTP request is asked again, in place of each source's own (0: once).
            on_failure: ``"raise"`` (the default): a source's error, a ``SourceTimeout`` included, is raised.
                ``"cache"`` (needs ``cache=True``): a source that failed, ran out of time or got no time at all gives
                only what its cache holds, which is settled rows, never the last two days before now or a tail the
                source declared unsettled; its recent cells are missing, never filled with an older value. The run goes
                on and ``last_refresh`` says what happened.

        Without ``budget``, ``deadline``, ``timeout``, ``retries`` and ``on_failure`` a run is what it was before
        they existed: the same requests, waits and frame. ``last_refresh`` holds a ``RefreshRecord`` of the run, each
        source's status, time, requests and retries, redacted error, rows and last valid timestamp per column; it is
        set when a source raised, too. Retry settings are arguments of the run, not of the sources, so no YAML file or
        cache key moves.
        """
        self.last_refresh = None
        if not self.sources:
            raise ValueError("At least one data source is required")
        self._validate_settings()  # settings (e.g. timezone) may have changed via add_transformer since __init__
        if on_failure == "cache" and cache is not True:
            raise ValueError(
                "on_failure='cache' gives a source that failed its rows from the per-source cache: pass cache=True"
            )
        refresh = Refresh(
            clock=current_clock(),
            budget=budget,
            deadline=deadline,
            timeout=timeout,
            retries=retries,
            on_failure=on_failure,
            cache="sources" if cache is True else "file" if isinstance(cache, (str, os.PathLike)) else "none",
            timezone=self.timezone,
        )

        start, end = self.resolve_range(start, end)
        self.validation_results = {}

        try:
            result = self._load_or_fetch_sources(start, end, cache, refresh)
        finally:
            self.last_refresh = refresh.record(start, end)
        if result.empty:
            return result

        self._run_validators(result, self.source_validators, "source_validators")
        result = self._run_transformers(result)
        self._run_validators(result, self.validators, "validators")

        logger.info(f"Pipeline: Completed with {len(result)} rows")
        return result

    def _load_or_fetch_sources(
        self, start: pd.Timestamp, end: pd.Timestamp, cache: Union[bool, str, os.PathLike], refresh: Refresh
    ) -> pd.DataFrame:
        if isinstance(cache, (str, os.PathLike)):
            return self._run_sources_with_file_cache(start, end, cache, refresh)
        return self._run_sources(start, end, cache, refresh)[0]

    def _run_sources_with_file_cache(
        self, start: pd.Timestamp, end: pd.Timestamp, cache: Union[str, os.PathLike], refresh: Refresh
    ) -> pd.DataFrame:
        """Cache the combined source output in one file, reused only for the same sources, dates and settledness
        (see ``_is_file_cache_stale``), which includes the instant from which a source declared its rows
        unsettled (``DataSource``)."""
        fingerprint = self._sources_fingerprint(start, end)
        digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
        path = _resolve_cache_file(cache, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = path.with_name(path.name + ".meta.json")

        with hold_lock(make_file_lock(path.with_name(path.name + ".lock")), str(path)):
            meta = _read_json(meta_path)
            fresh = path.exists() and meta.get("fingerprint") == digest
            stale = fresh and _is_file_cache_stale(meta, end)

            if fresh and not stale:
                try:
                    result = _read_frame(path, meta)
                    logger.info(f"Cache: Loading source data from {path}")
                    # No source is asked: each is ok, with what it gave the file unknown.
                    for name in self._source_names():
                        refresh.add(SourceRefresh(name, "ok"))
                    return result
                except Exception as e:  # noqa: BLE001 - an unreadable cache file is a cache miss
                    logger.warning(f"Cache: cannot read {path} ({type(e).__name__}: {e}); fetching again")
            elif stale:
                logger.info(
                    f"Cache: {path} was fetched while part of the range was not settled (still within the unsettled "
                    "period, or declared unsettled by a source); fetching again"
                )
            elif path.exists():
                logger.info(f"Cache: {path} holds data for {_describe_mismatch(meta, fingerprint)}; fetching again")

            result, settled_until = self._run_sources(start, end, cache=False, refresh=refresh)
            if not result.empty:
                # Invalidate first: a crash between the two writes must not pair old metadata with new data.
                meta_path.unlink(missing_ok=True)
                atomic_write(path, lambda tmp: _write_frame(result, tmp, path))
                meta = {
                    "fingerprint": digest,
                    "created_at": current_clock().now().isoformat(),
                    "settled_at": unsettled_boundary().isoformat(),
                    # None when every source's rows were settled up to `end`.
                    "settled_until": settled_until.isoformat() if settled_until is not None else None,
                    "columns": [str(c) for c in result.columns],
                    "dtypes": [str(t) for t in result.dtypes],
                    "index_name": result.index.name,
                    "config": fingerprint,
                }
                atomic_write(meta_path, lambda tmp: tmp.write_text(json.dumps(meta, indent=2, default=str)))
                logger.info(f"Cache: Saved source data to {path}")
            return result

    def _sources_fingerprint(self, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Any]:
        from epftoolbox2 import __version__

        return {
            "epftoolbox2": __version__,
            "cache_format": CACHE_FORMAT_VERSION,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "sources": [_component_fingerprint(source) for source in self.sources],
        }

    def _source_names(self) -> List[str]:
        """``"<class> #<position>"`` of every source, its name in warnings and in ``last_refresh``."""
        return [f"{type(source).__name__} #{number}" for number, source in enumerate(self.sources, start=1)]

    def _run_sources(
        self, start: pd.Timestamp, end: pd.Timestamp, cache: Union[bool, str], refresh: Refresh
    ) -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
        """The sources' joined output for ``[start, end)`` and the earliest instant from which any source declared
        its rows unsettled (``None`` if none did). Each source's part goes into ``refresh``."""
        cache_manager = CacheManager() if cache is True else None
        labelled = []
        unsettled_from = []
        secrets = _secrets(self.sources)
        failures: List[str] = []

        for name, source in zip(self._source_names(), self.sources):
            began = refresh.began()
            df, settled_until, status, failure, counted = self._refresh_source(
                name, source, start, end, cache_manager, refresh
            )
            error = None if failure is None else redact(f"{type(failure).__name__}: {failure}", secrets)
            if failure is not None and (refresh.on_failure == "raise" or not _http.is_unavailable(failure)):
                refresh.add(
                    SourceRefresh(
                        name,
                        status,
                        seconds=refresh.began() - began,
                        requests=counted.requests if counted is not None else 0,
                        retries=counted.retried if counted is not None else 0,
                        error=error,
                    )
                )
                raise failure
            if failure is not None:
                logger.warning(f"Pipeline: {name} {status} ({error}); it gives only the settled rows its cache holds")
                failures.append(f"{name} {status} ({error})")
                if df is None or df.empty:
                    columns = getattr(source, "output_columns", None)
                    named = f" ({', '.join(columns())})" if callable(columns) else ""
                    warnings.warn(
                        f"{name} {status} and its cache holds no row of [{start}, {end}): its columns{named} are "
                        f"missing from the frame ({error})",
                        RuntimeWarning,
                        stacklevel=4,
                    )
            if settled_until is not None:
                unsettled_from.append(settled_until)
            # Sources may return rows outside the request (ENTSO-E load from start - 1 day, Open-Meteo up to
            # end + horizon); every path returns exactly [start, end) so cached and uncached runs agree.
            if df is not None and not df.empty:
                df = clip_to_range(df, start, end)
            refresh.add(
                SourceRefresh(
                    name,
                    status,
                    seconds=refresh.began() - began,
                    requests=counted.requests if counted is not None else 0,
                    retries=counted.retried if counted is not None else 0,
                    error=error,
                    rows=0 if df is None else len(df),
                    last_valid=last_valid(df),
                )
            )
            if df is not None and not df.empty:
                labelled.append((name, df))

        unsettled = min(unsettled_from) if unsettled_from else None
        if not labelled and failures:
            raise RefreshError(f"no source gave a row of [{start}, {end}): {'; '.join(failures)}")
        if not labelled:
            logger.warning("Pipeline: No data returned from any source")
            return pd.DataFrame(), unsettled

        _check_unique_columns(labelled)
        frames = [df for _, df in labelled]
        # Frames on different indexes are joined on their sorted union and aligned ones keep their order, as pandas did
        # by default for DatetimeIndex frames before 3.1 deprecated it. The columns follow the sources in order. Aligned
        # means the same zone and instants, whatever the datetime unit (`Index.equals` says no across units).
        first = frames[0].index
        aligned = all(
            df.index.tz == first.tz and np.array_equal(epoch_ns(df.index), epoch_ns(first)) for df in frames[1:]
        )
        return pd.concat(frames, axis=1, sort=not aligned), unsettled

    def _refresh_source(
        self,
        name: str,
        source: DataSource,
        start: pd.Timestamp,
        end: pd.Timestamp,
        cache_manager: Optional[CacheManager],
        refresh: Refresh,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Timestamp], str, Optional[Exception], Optional[_http.Limits]]:
        """One source's fetch within the refresh's limits: its frame, the instant it declared its rows unsettled from,
        its status, what it raised (None when ok) and its request counts. A source that raised, or got no time, has the
        rows its cache holds (``on_failure="cache"``: settled ones only, see ``_cached_rows``)."""
        # Always started, also after the deadline: a full hit of its cache or work of its own needs no time, and a
        # request it really needs raises `SourceTimeout` (`_http.request`, or the cache lock's wait).
        try:
            with _http.limits(refresh.deadline, refresh.timeout, refresh.retries) as counted:
                if cache_manager is not None:
                    df, settled_until = self._fetch_with_cache(source, start, end, cache_manager)
                else:
                    df, settled_until = self._fetch_source(source, start, end)
            return df, settled_until, "ok", None, counted
        except Exception as exc:  # noqa: BLE001 - recorded, then raised or answered from the cache (on_failure)
            failure = exc
        if not isinstance(failure, SourceTimeout):
            status = "failed"
        else:  # out of time before it sent a request (a cache lock's wait too): it got no time at all
            status = "timeout" if counted.requests else "skipped"
        # Only a source out of reach for now is answered from its cache; an invalid key, a missing file or a bug is raised.
        if refresh.on_failure == "raise" or not _http.is_unavailable(failure):
            return None, None, status, failure, counted
        with _http.limits(refresh.deadline):  # the cache lock's wait keeps to what is left too
            try:
                cached = self._cached_rows(source, start, end, cache_manager)
            except SourceTimeout:
                logger.warning(f"Pipeline: {name}'s cache is locked by another process; it gives no row")
                cached = pd.DataFrame()
        return cached, None, status, failure, counted

    @staticmethod
    def _cached_rows(
        source: DataSource, start: pd.Timestamp, end: pd.Timestamp, cache_manager: Optional[CacheManager]
    ) -> pd.DataFrame:
        """What the source's per-source cache holds of ``[start, end)``: only settled rows, as the cache records
        nothing of the last ``UNSETTLED_PERIOD`` before now and nothing from a ``settled_until`` a source declared. Its
        recorded ranges are read after the failed fetch let go of the lock, so a range it wrote before failing counts."""
        config = source.get_cache_config() if cache_manager is not None else None
        if config is None:
            return pd.DataFrame()
        cached = cache_manager.read_cached_data(
            cache_manager.get_cache_key(config, _qualified_name(source)), start, end
        )
        return cached if cached is not None else pd.DataFrame()

    def _run_transformers(self, df: pd.DataFrame) -> pd.DataFrame:
        for transformer in self.transformers:
            df = transformer.transform(df)
        return df

    def _run_validators(self, df: pd.DataFrame, validators: List[Validator], stage: str) -> None:
        """Run one validation stage, store each result in ``validation_results`` and apply ``on_invalid``."""
        failed = []
        for validator in validators:
            label = type(validator).__name__
            name, n = f"{stage}.{label}", 2
            while name in self.validation_results:
                name, n = f"{stage}.{label}#{n}", n + 1
            result = validator.validate(df)
            self.validation_results[name] = result
            if not result.is_valid:
                failed.append(name)
                for error in result.errors:
                    logger.warning(f"Validation [{label}]: {error}")
            for warning in result.warnings:
                logger.info(f"Validation [{label}]: {warning}")

        if not failed:
            return
        summary = "; ".join(_summarize(name, self.validation_results[name]) for name in failed)
        if self.on_invalid == "raise":
            raise DataValidationError(f"Validation failed: {summary}")
        logger.warning(f"Validation: {len(failed)} validator(s) failed, continuing (on_invalid='warn'): {summary}")

    def _serialize_component(
        self, component: Any, include_secrets: bool = False, base_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        class_name = type(component).__name__
        refs = getattr(component, "_yaml_refs", {})
        params = {}
        for key, value in vars(component).items():
            if key.startswith("_") or key in _RUNTIME_ATTRIBUTES or _left_at_default(component, key, value):
                continue
            where = f"{class_name}.{key}"
            plain = to_yaml_value(value, where)
            ref = refs.get(key)
            unchanged = ref is not None and ref["value"] == plain
            if unchanged and _ENV_REF.search(str(ref["raw"])):
                params[key] = ref["raw"]  # written back as ${VAR}, as it was in the loaded YAML
            elif key in _SECRET_ATTRIBUTES and value and not include_secrets:
                params[key] = "${" + _secret_env_var(component, key) + "}"
            elif _is_path_param(key) and isinstance(plain, str) and plain:
                params[key] = _portable_path(plain, base_dir, was_relative=unchanged and ref["relative"])
            else:
                params[key] = plain
        return {"class": class_name, "params": params}

    def to_dict(self, include_secrets: bool = False, base_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Pipeline configuration as plain Python values (see ``save``)."""
        base = Path(base_dir) if base_dir is not None else None

        def serialize(components):
            return [self._serialize_component(c, include_secrets, base) for c in components]

        return {
            "timezone": self.timezone,
            "on_invalid": self.on_invalid,
            "sources": serialize(self.sources),
            "source_validators": serialize(self.source_validators),
            "transformers": serialize(self.transformers),
            "validators": serialize(self.validators),
        }

    def save(self, path: Union[str, Path], include_secrets: bool = False) -> None:
        """Write the configuration to YAML.

        ``api_key`` is written as ``${<SOURCE>_API_KEY}`` (e.g. ``${ENTSOE_API_KEY}``) and read from the environment
        on load, unless ``include_secrets=True``. Values loaded from ``${VAR}`` references are written back as the
        reference. Relative file paths are written relative to the YAML file. Parameters that cannot be represented
        in YAML (callables, sets, arbitrary objects) raise ``TypeError``.
        """
        path = Path(path)
        config = self.to_dict(include_secrets=include_secrets, base_dir=path.parent)
        path.write_text(dump_yaml(config))
        logger.info(f"Pipeline: Saved configuration to {path}")

    @classmethod
    def _load_component(
        cls, component_type: str, config: Dict[str, Any], base_dir: Optional[Union[str, Path]] = None
    ) -> Any:
        """Build one component; ``${VAR}`` in string values is expanded and, with ``base_dir``, relative file paths
        are resolved against it."""
        class_name = config["class"]
        raw_params = config.get("params") or {}

        if class_name not in COMPONENT_REGISTRY[component_type]:
            raise ValueError(f"Unknown {component_type[:-1]}: {class_name}")

        params, refs = {}, {}
        for key, raw in raw_params.items():
            value = _expand_env(raw, f"{class_name}.{key}")
            relative = False
            if base_dir is not None and _is_path_param(key) and isinstance(value, str) and value:
                local = os.path.expanduser(value)
                if not os.path.isabs(local):
                    value = os.path.normpath(os.path.join(os.path.abspath(base_dir), local))
                    relative = True
            if value != raw:
                refs[key] = {"raw": raw, "value": value, "relative": relative}
            params[key] = value

        module_path = COMPONENT_REGISTRY[component_type][class_name]
        module = importlib.import_module(module_path)
        component_class = getattr(module, class_name)
        component = component_class(**params)
        if refs:
            component._yaml_refs = refs
        return component

    @classmethod
    def from_dict(cls, config: Dict[str, Any], base_dir: Optional[Union[str, Path]] = None) -> "DataPipeline":
        """Build a pipeline from a configuration dict as written by ``to_dict``; relative file paths are resolved
        against ``base_dir`` (``load`` passes the YAML file's directory)."""
        config = config or {}
        unknown = set(config) - set(_PIPELINE_KEYS)
        if unknown:
            raise ValueError(f"Unknown DataPipeline setting(s) {sorted(unknown)}; expected {list(_PIPELINE_KEYS)}")

        def build(section: str, registry_section: str) -> list:
            return [cls._load_component(registry_section, c, base_dir) for c in config.get(section) or []]

        return cls(
            sources=build("sources", "sources"),
            source_validators=build("source_validators", "validators"),
            transformers=build("transformers", "transformers"),
            validators=build("validators", "validators"),
            timezone=config.get("timezone"),
            on_invalid=config.get("on_invalid", "warn"),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataPipeline":
        path = Path(path)
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        pipeline = cls.from_dict(config, base_dir=path.parent)
        logger.info(f"Pipeline: Loaded configuration from {path}")
        return pipeline


_PIPELINE_KEYS = ("timezone", "on_invalid", "sources", "source_validators", "transformers", "validators")
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _now(tz: Optional[str]) -> pd.Timestamp:
    """Current time in ``tz``, or naive machine-local time when ``tz`` is None, on the current clock
    (``epftoolbox2.clock``)."""
    return wall_time(tz if tz else None)


def _cache_day_zone(source: DataSource, timezone: Optional[str]) -> Optional[str]:
    """The zone whose local midnights a per-source cache fetches from: the pipeline's ``timezone``, else the zone a
    source says its one-day values are stamped at (``EntsoeSource._daily_zone``, the area's own zone for a source
    used without a pipeline zone), else ``None`` (no source-specific day to start from)."""
    if timezone is not None:
        return timezone
    daily_zone = getattr(source, "_daily_zone", None)
    return daily_zone() if callable(daily_zone) else None


def _calendar_day(text: str, tz: Optional[str]) -> Optional[pd.Timestamp]:
    """Naive midnight of the day a date string ("2024-01-01") or keyword ("today", "now_d-7") names, else None.

    Keywords count days in ``tz`` (the machine's local date when ``tz`` is None, as before).
    """
    s = text.strip()
    match = _DATE_PATTERN.match(s)
    if match:
        offset = int(match.group(2)) if match.group(2) else 0
        return _now(tz).tz_localize(None).normalize() + pd.Timedelta(days=offset)
    if _DATE_ONLY.match(s):
        return pd.Timestamp(s)
    if ":" in s:
        return None
    try:
        ts = pd.Timestamp(s)
    except ValueError:
        return None
    return ts if ts.tzinfo is None and ts == ts.normalize() else None


def _secrets(components: List[Any]) -> List[str]:
    """Every secret value of the components (``_SECRET_ATTRIBUTES``), for redaction."""
    found = []
    for component in components:
        for key in _SECRET_ATTRIBUTES:
            value = getattr(component, key, None)
            if isinstance(value, str) and value:
                found.append(value)
    return found


def _check_unique_columns(labelled: List[Tuple[str, pd.DataFrame]]) -> None:
    owners: Dict[Any, List[str]] = {}
    for label, df in labelled:
        duplicated = list(df.columns[df.columns.duplicated()].unique())
        if duplicated:
            raise ValueError(f"{label} returned duplicate column names {duplicated}")
        for column in df.columns:
            owners.setdefault(column, []).append(label)
    clashes = [f"{column!r} from {' and '.join(labels)}" for column, labels in owners.items() if len(labels) > 1]
    if clashes:
        shown = "; ".join(clashes[:10]) + ("; ..." if len(clashes) > 10 else "")
        raise ValueError(f"Sources produce the same column names ({shown}). Give the sources distinct prefixes.")


def _summarize(name: str, result: ValidationResult) -> str:
    errors = result.errors[:3] + ([f"... {len(result.errors) - 3} more"] if len(result.errors) > 3 else [])
    return f"{name}: {', '.join(errors) or 'invalid'}"


def _expand_env(value: Any, where: str) -> Any:
    """Replace ``${VAR}`` in strings (also inside lists and dicts) with the environment variable's value."""
    if isinstance(value, str):

        def substitute(match: "re.Match[str]") -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(
                    f"{where} refers to ${{{name}}}, but the environment variable {name} is not set "
                    "(set it, or save the pipeline with include_secrets=True)"
                )
            return os.environ[name]

        return _ENV_REF.sub(substitute, value)
    if isinstance(value, list):
        return [_expand_env(v, where) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v, f"{where}.{k}") for k, v in value.items()}
    return value


def _is_path_param(name: str) -> bool:
    return name == "path" or name.endswith(("_path", "_file", "_dir"))


def _portable_path(value: str, base_dir: Optional[Path], was_relative: bool) -> str:
    """Absolute paths stay absolute; relative ones (to the working directory, or to the YAML they were loaded from)
    are written relative to the directory of the YAML being saved."""
    local = os.path.expanduser(value)
    if base_dir is None or (os.path.isabs(local) and not was_relative):
        return value
    try:
        return Path(os.path.relpath(os.path.abspath(local), os.path.abspath(base_dir))).as_posix()
    except ValueError:  # e.g. different drives on Windows
        return os.path.abspath(local)


def _secret_env_var(component: Any, key: str) -> str:
    """Environment variable a secret is read from: ``api_key_env`` on the class, else e.g. ENTSOE_API_KEY."""
    explicit = getattr(type(component), f"{key}_env", None)
    if explicit:
        return explicit
    name = type(component).__name__
    if name.endswith("Source") and len(name) > len("Source"):
        name = name[: -len("Source")]
    return f"{name.upper()}_{key.upper()}"


def _qualified_name(obj: Any) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _fingerprint_value(value: Any) -> Any:
    """JSON-safe, deterministic stand-in for a parameter value. Files are identified by size and mtime."""
    if isinstance(value, Path):
        if value.exists():
            stat = value.stat()
            return {"path": str(value.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        return str(value)
    if isinstance(value, (list, tuple, range)):
        return [_fingerprint_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _fingerprint_value(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "tolist"):  # numpy scalars and arrays
        return _fingerprint_value(value.tolist())
    return repr(value)


def _left_at_default(component: Any, key: str, value: Any) -> bool:
    """Whether `key` is a parameter its class writes only when set (`_WRITTEN_WHEN_SET`, as `BaseModel`'s) and is at that
    default: a file saved now then loads in versions from before the parameter, and a fingerprint stays what it was."""
    written_when_set = getattr(type(component), "_WRITTEN_WHEN_SET", {})
    return key in written_when_set and value == written_when_set[key]


def _component_fingerprint(component: Any) -> Dict[str, Any]:
    params = {
        key: _fingerprint_value(value)
        for key, value in vars(component).items()
        if not key.startswith("_")
        and key not in _RUNTIME_ATTRIBUTES
        and key not in _SECRET_ATTRIBUTES
        and not _left_at_default(component, key, value)
    }
    fingerprint = {"class": _qualified_name(component), "params": params}
    # A component may inherit its market zone from DataPipeline.timezone (a private attribute, so it is not in
    # `params` above); include the value it actually resolved to, so the file cache is invalidated when only the
    # pipeline's zone changes.
    market_timezone = getattr(component, "market_timezone", None)
    if market_timezone is not None:
        fingerprint["market_timezone"] = market_timezone
    if isinstance(component, DataSource):
        fingerprint["cache_config"] = _fingerprint_value(component.get_cache_config())
        # What a source reads that its parameters don't say (`DataSource.content_digest`): another pipeline's results, say.
        # No key for a source that has none, so every other fingerprint is what it was.
        content = component.content_digest()
        if content is not None:
            fingerprint["content"] = content
    return fingerprint


def _resolve_cache_file(cache: Union[str, os.PathLike], digest: str) -> Path:
    """Expand ``$VAR`` and ``~`` in a cache path. A directory (existing, trailing separator, or a new path without a
    file suffix) gets one Parquet file per configuration; ``*.parquet`` is Parquet; any other file is CSV."""
    raw = os.fspath(cache)
    expanded = os.path.expanduser(os.path.expandvars(raw))
    unset = _UNSET_ENV_VAR.findall(expanded)
    if unset:
        raise ValueError(f"Cache path {raw!r} uses environment variables that are not set: {', '.join(unset)}")
    path = Path(expanded)
    if path.is_file():
        return path
    if path.is_dir() or raw.endswith(("/", os.sep)) or path.suffix == "":
        return path / f"data_{digest}.parquet"
    return path


def _is_parquet(path: Path) -> bool:
    return path.suffix.lower() in (".parquet", ".pq")


def _write_frame(df: pd.DataFrame, tmp: Path, path: Path) -> None:
    if _is_parquet(path):
        df.to_parquet(tmp)
    else:
        df.to_csv(tmp)


def _read_frame(path: Path, meta: Dict[str, Any]) -> pd.DataFrame:
    if _is_parquet(path):
        return pd.read_parquet(path)
    df = pd.read_csv(path, index_col=0, **EXACT_FLOATS)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = meta.get("index_name")
    # CSV loses column names (duplicates get ".1") and dtypes (an all-null column comes back as float).
    df.columns = meta["columns"]
    for i, dtype in enumerate(meta["dtypes"]):
        if str(df.dtypes.iloc[i]) != dtype:
            df.isetitem(i, df.iloc[:, i].astype(dtype))
    return df


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _is_file_cache_stale(meta: Dict[str, Any], end: pd.Timestamp) -> bool:
    """Whether a matching-fingerprint file cache might now be missing data that has become settled since it was
    written (a fixed date range whose `end` is in the future would otherwise cache NaN for not-yet-published rows
    forever, which the per-source cache avoids by never recording the unsettled tail before "now" as covered).

    A file is trusted only if either the whole requested range was already settled when it was fetched (`end` was
    at or before that fetch's settled boundary, so nothing about it could still change), or the settled boundary
    has not moved since (same day, or `unsettled_period` unchanged). A file without "settled_at" is always treated
    as stale.

    A source can also declare that its rows from some instant on are not settled (`DataSource`, e.g. school holidays
    the API has not published yet). The earliest such instant is stored as "settled_until", and a file whose
    "settled_until" is before the requested `end` is never reused: the fetch it holds was only settled up to there.
    A missing value means the sources declared nothing.
    """
    settled_until = meta.get("settled_until")
    if settled_until is not None:
        try:
            if pd.Timestamp(settled_until) < end:
                return True
        except (TypeError, ValueError):
            return True
    settled_at = meta.get("settled_at")
    if settled_at is None:
        return True
    settled_at = pd.Timestamp(settled_at)
    if end <= settled_at:
        return False
    return unsettled_boundary() > settled_at


def _describe_mismatch(meta: Dict[str, Any], fingerprint: Dict[str, Any]) -> str:
    old = meta.get("config")
    if not old:
        return "an unknown configuration"
    if (old.get("start"), old.get("end")) != (fingerprint["start"], fingerprint["end"]):
        return f"{old.get('start')} .. {old.get('end')}, requested {fingerprint['start']} .. {fingerprint['end']}"
    if old.get("sources") != fingerprint["sources"]:
        return "different sources or source parameters"
    return "a different epftoolbox2 version or cache format"
