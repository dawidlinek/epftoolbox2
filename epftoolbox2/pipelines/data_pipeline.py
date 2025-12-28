from typing import List, Optional, Union, Dict, Any
import pandas as pd
from pathlib import Path
import yaml
import importlib

from epftoolbox2.data.sources.base import DataSource
from epftoolbox2.data.transformers.base import Transformer
from epftoolbox2.data.validators.base import Validator
from epftoolbox2.data.cache_manager import CacheManager
from epftoolbox2.logging import get_logger

logger = get_logger(__name__)

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
    },
    "validators": {
        "ContinuityValidator": "epftoolbox2.data.validators.continuity",
        "NullCheckValidator": "epftoolbox2.data.validators.null_check",
        "EdaValidator": "epftoolbox2.data.validators.eda",
    },
}


class DataPipeline:
    """Data pipeline that combines multiple sources and applies transformers.

    Example:
        >>> pipeline = DataPipeline(
        ...     sources=[CsvSource(...), EntsoeSource(...)],
        ...     transformers=[TimezoneTransformer(target_tz="Europe/Warsaw")]
        ... )
        >>> df = pipeline.run(start, end)

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

    def __init__(
        self,
        sources: Optional[List[DataSource]] = None,
        transformers: Optional[List[Transformer]] = None,
        validators: Optional[List[Validator]] = None,
    ):
        self.sources: List[DataSource] = sources or []
        self.transformers: List[Transformer] = transformers or []
        self.validators: List[Validator] = validators or []

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

    def _fetch_with_cache(self, source: DataSource, start: pd.Timestamp, end: pd.Timestamp, cache_manager: CacheManager) -> pd.DataFrame:
        source_config = source.get_cache_config()
        if source_config is None:
            return source.fetch(start, end)

        cache_key = cache_manager.get_cache_key(source_config)
        missing_ranges = cache_manager.find_missing_ranges(cache_key, start, end)
        source_type = source_config.get("source_type", "unknown")

        if not missing_ranges:
            logger.info(f"Cache: Full hit for {source_type} source")
            return cache_manager.read_cached_data(cache_key, start, end)

        if len(missing_ranges) == 1 and missing_ranges[0] == (start, end):
            logger.info(f"Cache: Miss for {source_type} source")
        else:
            logger.info(f"Cache: Partial hit for {source_type} source")

        for missing_start, missing_end in missing_ranges:
            fresh_df = source.fetch(missing_start, missing_end)
            if fresh_df is not None and not fresh_df.empty:
                cache_manager.write_cache(cache_key, fresh_df, missing_start, missing_end, source_config)

        df = cache_manager.read_cached_data(cache_key, start, end)
        return df if df is not None else pd.DataFrame()

    def _parse_timestamp(self, ts: Union[str, pd.Timestamp]) -> pd.Timestamp:
        if ts == "today":
            return pd.Timestamp("today", tz="UTC").normalize()
        if isinstance(ts, str):
            return pd.Timestamp(ts, tz="UTC")
        return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")

    def run(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp], cache: Union[bool, str] = False) -> pd.DataFrame:
        if not self.sources:
            raise ValueError("At least one data source is required")

        start = self._parse_timestamp(start)
        end = self._parse_timestamp(end)

        if end <= start:
            raise ValueError(f"End timestamp ({end}) must be after start timestamp ({start})")

        result = self._load_or_fetch_sources(start, end, cache)
        if result.empty:
            return result

        result = self._run_transformers(result)
        self._run_validators(result)

        logger.info(f"Pipeline: Completed with {len(result)} rows")
        return result

    def _load_or_fetch_sources(self, start: pd.Timestamp, end: pd.Timestamp, cache: Union[bool, str]) -> pd.DataFrame:
        if isinstance(cache, str):
            cache_file = Path(cache)
            if cache_file.exists():
                logger.info(f"Cache: Loading source data from {cache}")
                result = pd.read_csv(cache_file, index_col=0)
                result.index = pd.to_datetime(result.index, utc=True)
                return result
            result = self._run_sources(start, end, cache)
            if not result.empty:
                result.to_csv(cache)
                logger.info(f"Cache: Saved source data to {cache}")
            return result
        return self._run_sources(start, end, cache)

    def _run_sources(self, start: pd.Timestamp, end: pd.Timestamp, cache: Union[bool, str]) -> pd.DataFrame:
        cache_manager = CacheManager() if cache is True else None
        dataframes = []

        for source in self.sources:
            df = self._fetch_with_cache(source, start, end, cache_manager) if cache is True else source.fetch(start, end)
            if df is not None and not df.empty:
                dataframes.append(df)

        if not dataframes:
            logger.warning("Pipeline: No data returned from any source")
            return pd.DataFrame()

        return pd.concat(dataframes, axis=1)

    def _run_transformers(self, df: pd.DataFrame) -> pd.DataFrame:
        for transformer in self.transformers:
            df = transformer.transform(df)
        return df

    def _run_validators(self, df: pd.DataFrame) -> None:
        for validator in self.validators:
            validation_result = validator.validate(df)
            validator_name = type(validator).__name__
            if not validation_result.is_valid:
                for error in validation_result.errors:
                    logger.warning(f"Validation [{validator_name}]: {error}")
            for warning in validation_result.warnings:
                logger.info(f"Validation [{validator_name}]: {warning}")

    def _serialize_component(self, component: Any) -> Dict[str, Any]:
        class_name = type(component).__name__
        params = {}
        excluded = {"session", "console", "logger", "lat", "lon"}
        if hasattr(component, "__dict__"):
            for key, value in component.__dict__.items():
                if not key.startswith("_") and key not in excluded:
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
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
