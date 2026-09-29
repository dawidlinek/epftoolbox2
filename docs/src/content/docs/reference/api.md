---
title: API Reference
description: Complete API reference for epftoolbox2
---

# API Reference

## DataPipeline

```python
class DataPipeline:
    def __init__(
        self,
        sources: Optional[List[DataSource]] = None,
        transformers: Optional[List[Transformer]] = None,
        validators: Optional[List[Validator]] = None,
        source_validators: Optional[List[Validator]] = None,
        timezone: Optional[str] = None,  # the single place the market zone needs to be written; see Data Sources
        on_invalid: str = "warn",  # "warn" or "raise"
    ): ...

    def add_source(self, source: DataSource) -> "DataPipeline": ...
    def add_transformer(self, transformer: Transformer) -> "DataPipeline": ...
    def add_validator(self, validator: Validator) -> "DataPipeline": ...

    def run(
        self,
        start: Union[str, pd.Timestamp],  # Start date
        end: Union[str, pd.Timestamp],    # End date
        cache: Union[bool, str] = False,  # Caching option
    ) -> pd.DataFrame: ...

    def save(self, path: Union[str, Path]) -> None: ...       # Save to YAML

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataPipeline": ...  # Load from YAML
```

---

## ModelPipeline

```python
class ModelPipeline:
    def __init__(self): ...

    def add_model(self, model: BaseModel) -> "ModelPipeline": ...
    def add_evaluator(self, evaluator: Evaluator) -> "ModelPipeline": ...
    def add_exporter(self, exporter: Exporter) -> "ModelPipeline": ...

    def run(
        self,
        data: pd.DataFrame,              # Input DataFrame
        test_start: str = None,           # Test period start
        test_end: str = None,             # Test period end
        target: str = "price",            # Target column
        horizon: int = 7,                 # Max horizon
        freq: str = "1h",                 # "1h" or "15min"; must match the data
        save_dir: Optional[str] = None,   # Result cache directory (one JSONL per model)
        forecast_only: bool = False,      # Skip evaluators/exporters; dates default to today
    ) -> EvaluationReport: ...

    def report(                           # Report from the JSONL files a run wrote to save_dir
        self,
        save_dir: Union[str, Path],
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        horizon: int = 7,
        freq: str = "1h",
        forecast_only: bool = False,
        data: Optional[pd.DataFrame] = None,   # only for exporters that join source columns
        allow_missing: bool = False,           # else FileNotFoundError when a model has no results
    ) -> EvaluationReport: ...

    def save(self, path: Union[str, Path]) -> None: ...       # Save to YAML

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelPipeline": ...  # Load from YAML
```

---

## Workflow

```python
class Workflow:
    def __init__(
        self,
        data_start: str,                                      # Data fetch start date
        data_end: str,                                        # Data fetch end date
        model_test_start: str,                                # Test period start
        model_test_end: str,                                  # Test period end
        data_pipeline: Optional[Union[str, Path]] = None,    # Path to data_pipeline.yaml
        model_pipeline: Optional[Union[str, Path]] = None,   # Path to model_pipeline.yaml
        data_cache: Union[bool, str] = False,                 # Data caching option
        model_target: str = "price",                          # Target column
        model_horizon: int = 7,                               # Max forecast horizon
        max_processes: Optional[int] = None,                  # Worker process count
        threads_per_process: Optional[int] = None,            # Threads per process
        cache_path: Optional[str] = None,                     # Cache directory (overrides data_cache)
        model_index: Optional[int] = None,                    # Run only models[model_index]; all models if None
        model_save_dir: Optional[str] = None,                 # One JSONL per model; required with model_index
    ): ...

    def run(self) -> EvaluationReport: ...
    def report(self, allow_missing: bool = False) -> EvaluationReport: ...  # Merge save_dir, run evaluators/exporters

    def save(self, path: Union[str, Path]) -> None: ...

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Workflow": ...
```

`Workflow.run()` always sets `OMP_NUM_THREADS=1` and `PYTHON_GIL=0` before spawning workers.
Pipeline paths are resolved relative to the workflow YAML's directory. `$VAR` in paths is expanded at runtime.

---

## EvaluationReport

```python
class EvaluationReport:
    def summary(self) -> pd.DataFrame: ...          # model, forecasts, one column per evaluator
    def forecast_counts(self) -> pd.DataFrame: ...  # model, planned, produced, missing, skip reasons
    def by_hour(self) -> pd.DataFrame: ...
    def by_horizon(self) -> pd.DataFrame: ...
    def by_hour_horizon(self) -> pd.DataFrame: ...
    def by_year(self) -> pd.DataFrame: ...
    def by_year_horizon(self) -> pd.DataFrame: ...
```

---

## Data Sources

```python
class EntsoeSource(DataSource):
    def __init__(self, country_code: str, api_key: str, type: List[str],
                 prefer_15min: bool = False,
                 neighbours: Optional[Union[List[str], str]] = None,
                 prefix: str = ""): ...
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...  # [start, end), UTC

class OpenMeteoSource(DataSource):
    def __init__(self, latitude: Optional[float] = None, longitude: Optional[float] = None,
                 # locations: mutually exclusive with latitude/longitude — several locations combined into one
                 # source, weighted average and/or per-location columns; see Data Sources > OpenMeteoSource
                 locations: Optional[Union[str, Dict[str, Any]]] = None,
                 keep_locations: bool = False,
                 columns: Optional[List[str]] = None,
                 horizon: int = 5,
                 model: str = "jma_seamless",
                 temporal_resolution: Optional[str] = None,
                 api: str = "previous_runs",  # "previous_runs" | "single_runs"
                 run: str = "00",  # UTC hour "HH": the cutoff is D at run:00 UTC
                 timezone: Optional[str] = None,  # inherits DataPipeline.timezone; fetching w/o either raises
                 prefix: str = "", strict: bool = False, proxy_url: Optional[str] = None): ...
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...

class CalendarSource(DataSource):
    def __init__(self, country: str, timezone: str = None,
                 holidays: Union[str, bool] = "binary",
                 weekday: Union[str, bool] = "number",
                 hour: Union[str, bool] = False,
                 month: Union[str, bool] = False,
                 daylight: bool = False, lat: float = None, lon: float = None,
                 prefix: str = "", freq: str = "1h", region: str = None,
                 categories: Union[str, List[str]] = None,
                 bridge_days: bool = False, christmas_period: bool = False,
                 holiday_distance: int = 0, dst_switch: bool = False,
                 regimes: Union[bool, List[str], Dict[str, Any]] = False,
                 school_holidays: Union[bool, str] = False): ...
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...  # [start, end)

class CsvSource(DataSource):
    def __init__(self, file_path: str, datetime_column: str = "datetime",
                 columns: List[str] = None, prefix: str = "",
                 datetime_format: str = None, separator: str = ",",
                 timezone: str = "UTC"): ...
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...  # [start, end)

class EpftoolboxBenchmarkSource(DataSource):
    def __init__(self, dataset: str, cache_dir: Optional[str] = None, prefix: str = ""): ...
    index_timezone: str   # property: fixed-offset zone of the index
    test_period: tuple    # property: the paper's test period (delivery days)
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...
```

---

## Transformers

```python
class ResampleTransformer(Transformer):
    def __init__(
        self,
        freq: str = "1h",
        agg: Union[str, Dict[str, str]] = "mean",  # dict: {column: agg}, "default" key for the rest
        fill: Union[str, None, Dict[str, Optional[str]]] = "ffill",  # "ffill" | None
        method: Optional[str] = None,  # deprecated
        columns: Union[str, List[str], None] = None,  # deprecated
    ): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

class LagTransformer(Transformer):
    def __init__(
        self,
        columns: Union[str, List[str], None] = None,  # None = all columns
        lags: Union[int, List[int], range] = 1,  # range and lists both accepted; lag 0 is skipped
        freq: str = "1h",  # "day"/"1D" = local calendar days; hours and minutes = elapsed time
    ): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

class TimezoneTransformer(Transformer):
    def __init__(
        self,
        target_tz: str,
        daily_columns: Optional[List[str]] = None,  # None = ["load_forecast_daily_min", "load_forecast_daily_max"]
        daily_anchor: Optional[str] = None,  # None = the resolved target_tz
    ): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

---

## Validators

```python
# Every validator: validate(df: pd.DataFrame) -> ValidationResult

class NullCheckValidator(Validator):
    def __init__(self, columns: Union[str, List[str]] = None, allow_nulls: bool = False,
                 max_null_frac: float = 0.0): ...

class ContinuityValidator(Validator):
    def __init__(self, freq: Optional[str] = None): ...  # None = infer

class DSTStructureValidator(Validator):
    def __init__(self, tz: Optional[str] = None, freq: Optional[str] = None,
                 allow_partial_edges: bool = True): ...

class CoverageValidator(Validator):
    def __init__(self, columns: Union[str, List[str]] = None, start: Optional[str] = None,
                 end: Optional[str] = None, freq: Optional[str] = None,
                 min_coverage: float = 1.0, max_gap: Optional[str] = None): ...

class PlausibilityValidator(Validator):
    def __init__(self, ranges: Optional[Dict[str, List[Optional[float]]]] = None,
                 max_constant_run: Optional[int] = 24,
                 constant_columns: Optional[List[str]] = None): ...

class EdaValidator(Validator):
    def __init__(self, columns: List[str] = None, verbose: bool = True): ...

@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: List[str]
    warnings: List[str]
    info: Dict[str, Any]
    stats: pd.DataFrame
    validator: str = ""
    def add_error(self, message: str) -> None: ...
    def to_frame(self) -> pd.DataFrame: ...

def results_to_frame(results) -> pd.DataFrame: ...  # list, (name, result) pairs or dict
```

---

## Models

```python
class OLSModel(BaseModel):
    def __init__(self, predictors: List, training_window: int = 365,
                 name: str = "Model", scaler: ScalerSpec = "standard"): ...

class LassoCVModel(BaseModel):
    def __init__(self, predictors: List, training_window: int = 365,
                 cv: int = 5, max_iter: int = 10000, name: str = "Model",
                 scaler: ScalerSpec = "standard"): ...
```

`ScalerSpec` is a scaler name (`"standard"`, `"asinh"`, `"npit"`), a `{"name": ..., "params": {...}}` dict or a scaler instance.

### Scalers

```python
from epftoolbox2.scalers import StandardScaler, AsinhScaler, NPITScaler, make_scaler, scaler_spec

class StandardScaler(BaseScaler):
    def __init__(self): ...

class AsinhScaler(BaseScaler):
    def __init__(self, apply_to: Union[str, List[str]] = "all"): ...

class NPITScaler(BaseScaler):
    def __init__(self, apply_to: Union[str, List[str]] = "all"): ...

def make_scaler(spec: ScalerSpec = "standard") -> BaseScaler: ...          # fresh instance
def scaler_spec(spec: ScalerSpec = "standard") -> Union[str, dict]: ...    # validated, YAML-safe form
```

See [Scalers and VSTs](/epftoolbox2/models/scalers/).

---

## Evaluators

```python
class Evaluator(ABC):
    name: str
    def compute(self, df: pd.DataFrame, **kwargs) -> float: ...

class MAEEvaluator(Evaluator):
    name = "MAE"

class RMSEEvaluator(Evaluator):
    name = "RMSE"

class rMAEEvaluator(Evaluator):
    name = "rMAE"
    def __init__(self, base_model: str): ...  # Name of the benchmark model
```

---

## Exporters

```python
class TerminalExporter(Exporter):
    def __init__(self, show: List[str] = None): ...
    def export(self, report: EvaluationReport) -> None: ...

class ExcelExporter(Exporter):
    def __init__(self, path: str, sheets: List[str] = None): ...
    def export(self, report: EvaluationReport) -> None: ...

class CsvExporter(Exporter):
    def __init__(self, path: str, extra_columns: List[str] = None): ...
    def export(self, report: EvaluationReport) -> None: ...
```
