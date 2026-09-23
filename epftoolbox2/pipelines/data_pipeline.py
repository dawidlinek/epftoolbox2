from typing import List, Optional, Tuple, Union, Dict, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import hashlib
import json
import os
import re
import pandas as pd
from pathlib import Path
import yaml
import importlib

from epftoolbox2.data.sources.base import DataSource
from epftoolbox2.data.transformers.base import Transformer
from epftoolbox2.data.validators.base import Validator
from epftoolbox2.data.validators.result import ValidationResult
from epftoolbox2.data.cache_manager import (
    CACHE_FORMAT_VERSION,
    CacheManager,
    atomic_write,
    clip_to_range,
    hold_lock,
    make_file_lock,
    unsettled_boundary,
)
from epftoolbox2.logging import get_logger
from epftoolbox2._date_utils import _DATE_PATTERN

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
    },
    "transformers": {
        "TimezoneTransformer": "epftoolbox2.data.transformers.timezone",
        "ResampleTransformer": "epftoolbox2.data.transformers.resample",
        "LagTransformer": "epftoolbox2.data.transformers.lag",
    },
    "validators": {
        "ContinuityValidator": "epftoolbox2.data.validators.continuity",
        "NullCheckValidator": "epftoolbox2.data.validators.null_check",
        "EdaValidator": "epftoolbox2.data.validators.eda",
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
        timezone: Market time zone (e.g. ``"Europe/Warsaw"``). If set, date strings passed to ``run`` are local
            calendar days: ``start`` begins at local midnight and ``end`` covers its whole local day; naive
            timestamps are local times. If ``None`` (default), date strings are midnight UTC, ``end`` is exclusive
            and naive timestamps are UTC (unchanged from before ``timezone`` existed).
        on_invalid: ``"warn"`` (default) logs failed validations and continues; ``"raise"`` raises
            ``DataValidationError`` after the failing validation stage. Results are kept in ``validation_results``.

    Example:
        >>> pipeline = DataPipeline(
        ...     sources=[CsvSource(...), EntsoeSource(...)],
        ...     transformers=[TimezoneTransformer(target_tz="Europe/Warsaw")],
        ...     timezone="Europe/Warsaw",
        ... )
        >>> df = pipeline.run("2024-01-01", "2024-06-30")  # 2024-01-01 00:00 .. 2024-06-30 23:00 local

    Example:
        >>> pipeline = (
        ...     DataPipeline()
        ...     .add_source(EntsoeSource(country_code="PL", api_key="...", type=["load"]))
        ...     .add_source(OpenMeteoSource(latitude=52.23, longitude=21.01))
        ...     .add_transformer(TimezoneTransformer(target_tz="Europe/Warsaw"))
        ...     .add_validator(NullCheckValidator(columns=["load_actual"]))
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
        self._validate_settings()

    def _validate_settings(self) -> None:
        if self.timezone is not None:
            try:
                ZoneInfo(self.timezone)
            except (ZoneInfoNotFoundError, ValueError, TypeError):
                raise ValueError(f"Invalid timezone: {self.timezone!r} (use an IANA name such as 'Europe/Warsaw')")
        if self.on_invalid not in self.ON_INVALID:
            raise ValueError(f"on_invalid must be one of {self.ON_INVALID}, got {self.on_invalid!r}")

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

    def _fetch_with_cache(
        self, source: DataSource, start: pd.Timestamp, end: pd.Timestamp, cache_manager: CacheManager
    ) -> pd.DataFrame:
        source_config = source.get_cache_config()
        if source_config is None:
            return source.fetch(start, end)

        source_class = _qualified_name(source)
        cache_key = cache_manager.get_cache_key(source_config, source_class)
        source_type = source_config.get("source_type", type(source).__name__)

        # Held across the fetch: parallel processes wait for the one fetching this source, then read its result.
        with cache_manager.lock(cache_key):
            cached = cache_manager.read_cached_data(cache_key, start, end)
            missing_ranges = cache_manager.find_missing_ranges(cache_key, start, end)

            if not missing_ranges:
                logger.info(f"Cache: Full hit for {source_type} source")
                return cached if cached is not None else pd.DataFrame()

            if len(missing_ranges) == 1 and missing_ranges[0] == (start, end):
                logger.info(f"Cache: Miss for {source_type} source")
            else:
                logger.info(f"Cache: Partial hit for {source_type} source")

            parts = [] if cached is None else [cached]
            for missing_start, missing_end in missing_ranges:
                fresh_df = source.fetch(missing_start, missing_end)
                if fresh_df is None or fresh_df.empty:
                    continue
                fresh_df = clip_to_range(fresh_df, missing_start, missing_end)
                # Records only the part the source actually returned and that is settled; the rest is still
                # returned now and fetched again next time.
                cache_manager.write_cache(cache_key, fresh_df, missing_start, missing_end, source_config, source_class)
                parts.append(fresh_df)

        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame()
        # Cached and missing ranges are disjoint, so there are no duplicate timestamps to resolve.
        return pd.concat(parts).sort_index(kind="stable")

    def _parse_timestamp(self, ts: Union[str, pd.Timestamp], is_end: bool = False) -> pd.Timestamp:
        """UTC bound for ``run``. See the class docstring for how dates, keywords and timestamps are read."""
        tz = self.timezone or "UTC"
        if isinstance(ts, str):
            day = _calendar_day(ts, self.timezone)
            if day is not None:
                if self.timezone is not None and is_end:
                    day += pd.Timedelta(days=1)  # the end date is included as a whole local day
                return day.tz_localize(tz, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
        ts = pd.Timestamp(ts)
        return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize(tz).tz_convert("UTC")

    def resolve_range(
        self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp]
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """The half-open UTC range ``[start, end)`` that ``run(start, end)`` returns."""
        start_ts = self._parse_timestamp(start)
        end_ts = self._parse_timestamp(end, is_end=True)
        if end_ts <= start_ts:
            raise ValueError(f"End timestamp ({end_ts}) must be after start timestamp ({start_ts})")
        return start_ts, end_ts

    def run(
        self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp], cache: Union[bool, str] = False
    ) -> pd.DataFrame:
        if not self.sources:
            raise ValueError("At least one data source is required")

        start, end = self.resolve_range(start, end)
        self.validation_results = {}

        result = self._load_or_fetch_sources(start, end, cache)
        if result.empty:
            return result

        self._run_validators(result, self.source_validators, "source_validators")
        result = self._run_transformers(result)
        self._run_validators(result, self.validators, "validators")

        logger.info(f"Pipeline: Completed with {len(result)} rows")
        return result

    def _load_or_fetch_sources(
        self, start: pd.Timestamp, end: pd.Timestamp, cache: Union[bool, str, os.PathLike]
    ) -> pd.DataFrame:
        if isinstance(cache, (str, os.PathLike)):
            return self._run_sources_with_file_cache(start, end, cache)
        return self._run_sources(start, end, cache)

    def _run_sources_with_file_cache(
        self, start: pd.Timestamp, end: pd.Timestamp, cache: Union[str, os.PathLike]
    ) -> pd.DataFrame:
        """Cache the combined source output in one file, reused only for the same sources, dates and settledness
        (see ``_is_file_cache_stale``)."""
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
                    return result
                except Exception as e:  # noqa: BLE001 - an unreadable cache file is a cache miss
                    logger.warning(f"Cache: cannot read {path} ({type(e).__name__}: {e}); fetching again")
            elif stale:
                logger.info(
                    f"Cache: {path} was fetched while part of the range was still unsettled and that boundary "
                    "has since moved; fetching again"
                )
            elif path.exists():
                logger.info(f"Cache: {path} holds data for {_describe_mismatch(meta, fingerprint)}; fetching again")

            result = self._run_sources(start, end, cache=False)
            if not result.empty:
                # Invalidate first: a crash between the two writes must not pair old metadata with new data.
                meta_path.unlink(missing_ok=True)
                atomic_write(path, lambda tmp: _write_frame(result, tmp, path))
                meta = {
                    "fingerprint": digest,
                    "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    "settled_at": unsettled_boundary().isoformat(),
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

    def _run_sources(self, start: pd.Timestamp, end: pd.Timestamp, cache: Union[bool, str]) -> pd.DataFrame:
        cache_manager = CacheManager() if cache is True else None
        labelled = []

        for number, source in enumerate(self.sources, start=1):
            if cache_manager is not None:
                df = self._fetch_with_cache(source, start, end, cache_manager)
            else:
                df = source.fetch(start, end)
            if df is None or df.empty:
                continue
            # Sources may return rows outside the request (ENTSO-E load from start - 1 day, Open-Meteo up to
            # end + horizon); every path returns exactly [start, end) so cached and uncached runs agree.
            df = clip_to_range(df, start, end)
            if not df.empty:
                labelled.append((f"{type(source).__name__} #{number}", df))

        if not labelled:
            logger.warning("Pipeline: No data returned from any source")
            return pd.DataFrame()

        _check_unique_columns(labelled)
        return pd.concat([df for _, df in labelled], axis=1)

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

    def _serialize_component(self, component: Any) -> Dict[str, Any]:
        class_name = type(component).__name__
        params = {}
        excluded = {"session", "console", "logger", "lat", "lon"}
        if hasattr(component, "__dict__"):
            for key, value in component.__dict__.items():
                if not key.startswith("_") and key not in excluded:
                    if isinstance(value, Path):
                        params[key] = str(value)
                    elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        params[key] = value
        return {"class": class_name, "params": params}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sources": [self._serialize_component(s) for s in self.sources],
            "transformers": [self._serialize_component(t) for t in self.transformers],
            "validators": [self._serialize_component(v) for v in self.validators],
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        config = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Pipeline: Saved configuration to {path}")

    @classmethod
    def _load_component(cls, component_type: str, config: Dict[str, Any]) -> Any:
        class_name = config["class"]
        params = config.get("params", {})

        if class_name not in COMPONENT_REGISTRY[component_type]:
            raise ValueError(f"Unknown {component_type[:-1]}: {class_name}")

        module_path = COMPONENT_REGISTRY[component_type][class_name]
        module = importlib.import_module(module_path)
        component_class = getattr(module, class_name)
        return component_class(**params)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataPipeline":
        path = Path(path)
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        pipeline = cls()

        for source_config in config.get("sources", []):
            pipeline.add_source(cls._load_component("sources", source_config))

        for transformer_config in config.get("transformers", []):
            pipeline.add_transformer(cls._load_component("transformers", transformer_config))

        for validator_config in config.get("validators", []):
            pipeline.add_validator(cls._load_component("validators", validator_config))

        logger.info(f"Pipeline: Loaded configuration from {path}")
        return pipeline


_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _now(tz: Optional[str]) -> pd.Timestamp:
    """Current time in ``tz``, or naive machine-local time when ``tz`` is None."""
    return pd.Timestamp.now(tz=tz) if tz else pd.Timestamp.now()


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


def _component_fingerprint(component: Any) -> Dict[str, Any]:
    params = {
        key: _fingerprint_value(value)
        for key, value in vars(component).items()
        if not key.startswith("_") and key not in _RUNTIME_ATTRIBUTES and key not in _SECRET_ATTRIBUTES
    }
    fingerprint = {"class": _qualified_name(component), "params": params}
    if isinstance(component, DataSource):
        fingerprint["cache_config"] = _fingerprint_value(component.get_cache_config())
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
    df = pd.read_csv(path, index_col=0, float_precision="round_trip")
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
    written (pipelines review, single-file cache finding: the per-source cache never records the unsettled tail
    before "now" as covered, but the single-file cache had no such boundary at all, so a fixed date range whose
    `end` was in the future cached NaN for not-yet-published rows forever).

    A file is trusted only if either the whole requested range was already settled when it was fetched (`end` was
    at or before that fetch's settled boundary, so nothing about it could still change), or the settled boundary
    has not moved since (same day, or `unsettled_period` unchanged). A file written before this check existed has
    no "settled_at" and is always treated as stale, so it is refetched once after an upgrade.
    """
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
        return "an unknown configuration (written by an older version)"
    if (old.get("start"), old.get("end")) != (fingerprint["start"], fingerprint["end"]):
        return f"{old.get('start')} .. {old.get('end')}, requested {fingerprint['start']} .. {fingerprint['end']}"
    if old.get("sources") != fingerprint["sources"]:
        return "different sources or source parameters"
    return "a different epftoolbox2 version or cache format"
