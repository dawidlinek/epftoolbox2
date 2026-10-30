---
title: API Reference
description: Complete API reference for epftoolbox2
---

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

    def save(
        self,
        path: Union[str, Path],
        include_secrets: bool = False,  # False writes api_key as a ${ENV_VAR} reference
    ) -> None: ...                      # Save to YAML

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataPipeline": ...  # Load from YAML
```

---

## ModelPipeline

```python
class ModelPipeline:
    def __init__(
        self,
        information_set: Optional[InformationSet] = None,  # When inputs become known and forecasts are issued
        issue_time: Optional[str] = None,                  # Shortcut for InformationSet(issue_time, issue_timezone)
        issue_timezone: str = "Europe/Brussels",           # Zone of issue_time
        availability: Optional[str] = None,                # "none", "warn" or "raise"; default "warn" with an issue time, else "none"
    ): ...

    def add_model(self, model: BaseModel) -> "ModelPipeline": ...
    def add_ensemble(self, ensemble: Ensemble) -> "ModelPipeline": ...  # Members are resolved when the pipeline runs
    def add_evaluator(self, *evaluators: Evaluator) -> "ModelPipeline": ...  # One or several; chainable
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
        allow_manifest_mismatch: bool = False,  # Overwrite a disagreeing manifest and resume anyway
        require_manifest: bool = False,   # Refuse results in save_dir that have no manifest
        validation: Union[Tuple[Any, Any], str, None] = None,  # (start, end) or "previous_year"; alone, only those days
        test: Optional[Tuple[Any, Any]] = None,  # (start, end), the same as test_start and test_end
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
        validation: Union[Tuple[Any, Any], str, None] = None,  # What run was told; each must be what the manifests record
        test: Optional[Tuple[Any, Any]] = None,  # None of the windows: every window the manifests record
    ) -> EvaluationReport: ...

    def verify(                           # Leakage probe and positive control on sampled issue days
        self,
        data: pd.DataFrame,               # The models' input frame, or source data with rebuild
        test_start: str,                  # First issue day to sample from
        test_end: str,                    # Last issue day to sample from
        target: str = "price",            # Target column
        horizon: int = 1,                 # Max horizon
        freq: str = "1h",                 # "1h" or "15min"; must match the data
        days: int = 5,                    # Issue days per model
        seed: int = 0,                    # Picks the days and the noise
        models: Optional[Sequence[str]] = None,  # Names of the models to verify; None verifies all
        positive_control: bool = True,    # False: `ok` only says that no forecast moved
        rebuild: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,  # Builds the input from source data
    ) -> VerifyReport: ...

    def explain(                          # What one forecast's predictors hold and when each is known; no side effects
        self,
        data: pd.DataFrame,               # The frame the models run on
        model: str,                       # The model's name
        test: Optional[Tuple[Any, Any]] = None,  # (first, last) issue day; required for a predictor function of ctx
        horizon: int = 1,                 # Days ahead of the issue day of the forecast shown
        hour: int = 12,                   # Its period of the target day: 0-23, or 0-95 for 15-minute data
        target: str = "price",            # Target column
        freq: str = "1h",                 # "1h" or "15min"; must match the data
        echo: bool = True,                # Print the tables
    ) -> Dict[str, pd.DataFrame]: ...     # "predictors", "grouped"; "lists" for a predictor function of ctx

    availability: str                     # The level in force; assign None to go back to the default

    def save(self, path: Union[str, Path]) -> None: ...       # Save to YAML

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelPipeline": ...  # Load from YAML
```

---

## VerifyReport

Returned by `ModelPipeline.verify()`, see [Verifying a Pipeline](/epftoolbox2/model-pipeline/verification/).

```python
class VerifyReport:
    issue_days: List[str]                 # The sampled issue days
    checks: List[IssueDayCheck]           # One model's probe and control on one issue day

    @property
    def ok(self) -> bool: ...             # Every model's status is "ok"
    @property
    def models(self) -> List[str]: ...

    def status(self, model: str) -> str: ...    # "ok", "leak" or "inconclusive"
    def table(self) -> pd.DataFrame: ...        # model, days, forecasts, changed, max change, control moved, status
    def problems(self) -> List[str]: ...        # One line for each thing that keeps a model from "ok"
    def assert_ok(self) -> None: ...            # Raises AssertionError naming them
```

---

## Workflow

```python
class Workflow:
    def __init__(
        self,
        data_start: str,                                      # Data fetch start date
        data_end: str,                                        # Data fetch end date
        model_test_start: Optional[str] = None,               # Test period start
        model_test_end: Optional[str] = None,                 # Test period end
        data_pipeline: Optional[Union[str, Path]] = None,    # Path to data_pipeline.yaml
        model_pipeline: Optional[Union[str, Path]] = None,   # Path to model_pipeline.yaml
        data_cache: Union[bool, str] = False,                 # Data caching option
        model_target: str = "price",                          # Target column
        model_horizon: int = 7,                               # Max forecast horizon
        model_freq: str = "1h",                               # "1h" or "15min"
        model_forecast_only: bool = False,                    # Skip evaluators/exporters
        max_processes: Optional[int] = None,                  # Worker process count
        threads_per_process: Optional[int] = None,            # Threads per process
        cache_path: Optional[str] = None,                     # Cache directory (overrides data_cache)
        model_index: Optional[int] = None,                    # Run only models[model_index], all models if None
        model_save_dir: Optional[str] = None,                 # One JSONL per model, required with model_index
        model_allow_manifest_mismatch: bool = False,          # Overwrite a disagreeing manifest and resume
        model_require_manifest: bool = False,                 # Refuse results that have no manifest
        model_validation: Any = None,                         # [start, end] or "previous_year"
        model_test: Optional[Sequence[Any]] = None,           # [start, end], the same as model_test_start and _end
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
    window: str                                     # "test" or "validation": what this report is on
    validation: EvaluationReport                    # The validation window; ValueError on a run with none
    windows: Optional[RunWindows]                   # The windows and segments of the run; None for a plain run

    def summary(self) -> pd.DataFrame: ...          # model, forecasts, one column per evaluator
    def forecast_counts(self) -> pd.DataFrame: ...  # model, planned, produced, missing, skip reasons
    def availability(self) -> pd.DataFrame: ...     # model, predictor, role, status, list, periods, horizons, rule, known, issued, note
    def choices(self) -> pd.DataFrame: ...          # ensemble, segment, chosen, k, validation_mae, validation_forecasts, validation_start, validation_end, test_start, test_end; attrs: member_mae, k_mae
    def by_hour(self) -> pd.DataFrame: ...
    def by_horizon(self) -> pd.DataFrame: ...
    def by_hour_horizon(self) -> pd.DataFrame: ...
    def by_year(self) -> pd.DataFrame: ...
    def by_year_horizon(self) -> pd.DataFrame: ...
    def by_quarter(self) -> pd.DataFrame: ...       # model, year, quarter, forecasts, metrics
    def by_day(self) -> pd.DataFrame: ...           # model, target_date, forecasts, metrics
    def by_price_level(
        self, spike: Optional[float] = None        # Price from which a period is a spike; default min(300, 99th percentile of the actuals)
    ) -> pd.DataFrame: ...                          # model, price_level (negative / normal / spike), forecasts, metrics; attrs: spike_threshold, spike_rule, spike_quantile
    def aggregate(
        self,
        freq: str = "1D",                          # The only aggregation so far
        agg: str = "max",                          # "max", "min", "mean" or "spread" (max - min) of the day
    ) -> EvaluationReport: ...                      # Whole days only; hour 0, freq "1D"; dropped days in its forecast_counts()
    def join(self, other: EvaluationReport) -> EvaluationReport: ...  # Models of both; refs kept, evaluators of this report

    def tests(
        self,
        benchmark: Optional[str] = None,   # Default: the benchmark of the report's DMEvaluator
        horizon: Union[int, str, None] = None,  # None: per horizon; a number; "all": daily losses averaged per run date
        variance: str = "hac",             # "hac" (Newey-West) or "plain" (epftoolbox)
        lag: Optional[int] = None,         # Newey-West lag; None: from the number of days
        correction: Optional[str] = "holm",  # Over the m models of a horizon; None for none
        by: Optional[str] = None,          # "hour": per-hour rows, descriptive, no p_holm or verdict
        interval: Optional[float] = 0.95,  # Coverage of the bootstrap interval of the ratio; None skips it
        block: int = 7,                    # Bootstrap block length in days
        reps: int = 4000,
        seed: int = 0,
    ) -> pd.DataFrame: ...  # model, benchmark, horizon, n_days, n_forecasts, mae, mae_benchmark, ratio, ci_low, ci_high, statistic, p_value, p_holm, m, mde, verdict

    def mcs(
        self,
        alpha: float = 0.10,
        statistic: str = "R",              # "R" (T_R) or "max" (T_max)
        block: int = 7,                    # Mean block length of the stationary bootstrap, in days
        reps: int = 1000,
        seed: int = 0,
        models: Optional[Sequence[str]] = None,  # Default: all; fewer than 3 gives NaN with a warning
        horizon: Union[int, str, None] = None,   # None or "all": daily losses averaged per run date; a number
    ) -> pd.DataFrame: ...  # model, n_days, n_forecasts, mae, ratio_to_best, p_value, in_90, in_75; attrs: n_days, n_forecasts, limited_by

    def ranks(
        self,
        block: int = 7,
        reps: int = 4000,
        seed: int = 0,
        level: float = 0.90,               # Coverage of the rank interval
        horizon: Union[int, str, None] = None,
        models: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame: ...  # model, n_days, n_forecasts, mae, rank, p_best, rank_low, rank_high; attrs: n_days, n_forecasts, limited_by

    def compare(
        self, a: str, b: str, horizon: Union[int, str, None] = None
    ) -> pd.DataFrame: ...  # One row; ratio = mae_a / mae_b: a, b, horizon, n_days, n_forecasts, mae_a, mae_b, ratio, ci_low, ci_high, statistic, p_plain, p_hac, hours_a_better, hours_b_better, hours_tested, mde

    def leaderboard(
        self, benchmark: Optional[str] = None, horizon: Union[int, str, None] = None
    ) -> pd.DataFrame: ...  # By MAE: model, n_forecasts, MAE, n_days, and with a benchmark n_common, ratio, ci_low, ci_high, p_value, p_holm
```

See [Comparing Models](/epftoolbox2/model-pipeline/comparing-models/). Every method keeps to the report's window, see [Validation and Test](/epftoolbox2/model-pipeline/validation-and-test/).

### Windows

```python
from epftoolbox2.results import RunWindows, Segment, read_windows

class RunWindows:
    validation: Optional[Tuple[str, str]]   # (first, last) issue day tagged "validation"
    test: Optional[Tuple[str, str]]         # (first, last) issue day tagged "test"
    mode: str                               # "fixed" or "previous_year"
    segments: Tuple[Segment, ...]           # Each test period and what serves it

class Segment:
    validation: Tuple[str, str]
    test: Tuple[str, str]
    def serves(self, run_date: str, target_date: str) -> bool: ...  # In its validation period, target before its test

def read_windows(path: Union[str, Path]) -> Optional[RunWindows]: ...  # From a results JSONL or its manifest; None for a plain run
```

---

## Ensembles

```python
from epftoolbox2.ensembles import AverageEnsemble, Ensemble

class AverageEnsemble(Ensemble):
    def __init__(
        self,
        members: Sequence[str],             # Names of models or ensembles, resolved when the pipeline runs
        top: Union[int, str, None] = None,  # None: all of them; k: the best k on validation; "validation": chooses k as well
        by: str = "validation",             # Where top chooses, the only value
        name: str = "avg",                  # Unique across models and ensembles
    ): ...
```

See [Ensembles](/epftoolbox2/model-pipeline/ensembles/).

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
                 # locations: mutually exclusive with latitude/longitude, several locations combined into one
                 # source, weighted average and/or per-location columns; see Data Sources > OpenMeteoSource
                 locations: Optional[Union[str, Dict[str, Any]]] = None,
                 agg: Optional[Union[str, List[str]]] = None,  # aggregate several locations; see Data Sources
                 keep_locations: bool = False,
                 columns: Optional[List[str]] = None,
                 horizon: Optional[int] = None,  # None: the largest horizon without look-ahead (6 for the default model)
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
        target_tz: Optional[str] = None,  # None = DataPipeline.timezone
        daily_columns: Optional[List[str]] = None,  # None = ["load_forecast_daily_min", "load_forecast_daily_max"]
        daily_anchor: Optional[str] = None,  # None = the resolved target_tz
    ): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

class DSTNormalizeTransformer(Transformer):
    def __init__(self, freq: str = "1h", tz: Optional[str] = None): ...  # naive wall clock, 24/96 slots a day
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

class DailyProfileTransformer(Transformer):
    def __init__(self, columns: Union[str, List[str]], freq: str = "1h"): ...  # {column}_h00 .. _h23
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

class DailyStatsTransformer(Transformer):
    def __init__(self, columns: Union[str, List[str]],
                 stats: List[str] = ["min", "max", "mean", "last"], freq: str = "1h"): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

class ImputeTransformer(Transformer):
    def __init__(self, columns: Union[str, List[str], None] = None,  # at least one column is required
                 method: str = "ffill",  # "ffill" | "previous_day" | "previous_week"
                 limit: Union[int, str, None] = None, flag: bool = True, suffix: str = ""): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

class LambdaTransformer(Transformer):
    def __init__(self, func: Union[Callable, str, None] = None,  # exactly one of func/columns
                 columns: Optional[dict] = None): ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

---

## Validators

```python
# Every validator: validate(df: pd.DataFrame) -> ValidationResult

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
    def __init__(self, predictors: Union[List, Callable], training_window: int = 365,
                 name: str = "Model", scaler: ScalerSpec = "standard"): ...

class LassoCVModel(BaseModel):
    def __init__(self, predictors: Union[List, Callable], training_window: int = 365,
                 cv: int = 5, max_iter: int = 10000, name: str = "Model",
                 scaler: ScalerSpec = "standard"): ...
```

`ScalerSpec` is a scaler name (`"standard"`, `"asinh"`, `"npit"`), a `{"name": ..., "params": {...}}` dict or a scaler instance.

### Predictor functions

`predictors` may be a function with one parameter named `ctx`, see [Predictor Functions](/epftoolbox2/model-pipeline/predictor-functions/).

```python
class PredictorContext:                     # What the function gets for one planned forecast
    horizon: int                            # Days ahead of the issue day, from 1
    period: int                             # 0-23, or 0-95 for 15-minute data
    hour: int                               # The same number as period
    issue_day: datetime.date                # The local calendar date D
    target_day: datetime.date               # D + horizon
    target_weekday: int                     # Weekday of target_day, 0 = Monday
    zone: Optional[str]                     # The data's time zone name
    freq: str                               # "1h" or "15min"

    def published(self, columns: Union[str, Iterable[str]]) -> bool: ...  # Known at the issue time and not NaN
    def value(self, column: str) -> Any: ...  # The cell of the forecast's row; raises if known only later

class Use:                                  # What the function may return instead of a plain list
    def __init__(self, predictors: Sequence[str], train_on: str = "all"): ...  # "matching" isn't available yet
```

`PredictorPlan` is what `ModelPipeline` builds before any fit (the distinct lists and which forecast uses which) and passes to `BaseModel.run(plan=...)`. `ModelResultRef.predictor_lists` describes those lists after a run, and `ModelResultRef.availability_listing` holds the rows behind `report.availability()`.

`BaseModel.run()` takes `availability` too, and without a pipeline it defaults to `"raise"`. It also takes `windows`, a `RunWindows`, which is what `ModelPipeline.run(validation=..., test=...)` passes; the returned ref then has a ref per window in `by_window`.

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
    def compute(self, df: pd.DataFrame, **kwargs) -> float: ...  # A float, or a dict for several columns
    def compute_group(self, model_dfs: Mapping[str, pd.DataFrame]) -> dict: ...  # Called per table group; loops compute by default
    def column_name(self, key: str) -> str: ...  # Column of a dict result: "{name} {key}"

class MAEEvaluator(Evaluator):
    name = "MAE"

class RMSEEvaluator(Evaluator):
    name = "RMSE"

class rMAEEvaluator(Evaluator):
    name = "rMAE"
    def __init__(self, base_model: str): ...  # Name of the benchmark model

class BiasEvaluator(Evaluator):
    name = "Bias"

class DMEvaluator(Evaluator):
    name = "DM"  # Columns "DM ratio", "DM ci_low", "DM ci_high", "DM p", "DM p_holm", "DM verdict", "DM n_days", "DM n_forecasts"
    def __init__(
        self,
        benchmark: str,                    # Name of the model the others are compared with
        variance: str = "hac",             # "hac" (Newey-West) or "plain" (epftoolbox)
        lag: Optional[int] = None,         # Newey-West lag; None: from the number of days
        correction: Optional[str] = "holm",  # Over the models of a group; None for none
        alpha: float = 0.05,               # Size behind "DM verdict"
        interval: Optional[float] = 0.95,  # Coverage of the bootstrap interval of the ratio; None skips it
        block: int = 7,                    # Bootstrap block length in days
        reps: int = 4000,
        seed: int = 0,
    ): ...

class MCSEvaluator(Evaluator):
    name = "MCS"  # Columns "MCS p", "in MCS", "MCS n_days", "MCS n_forecasts"
    def __init__(
        self,
        alpha: float = 0.10,               # Size of the set
        statistic: str = "R",              # "R" (T_R) or "max" (T_max)
        block: int = 7,                    # Mean block length of the stationary bootstrap, in days
        reps: int = 1000,
        seed: int = 0,
    ): ...

class EconomicEvaluator(Evaluator):
    name = "profit"  # Columns "profit", "profit % of perfect", "profit days"
    def __init__(self, strategy: Optional[BatteryArbitrage] = None): ...  # None: BatteryArbitrage()

class BatteryArbitrage:
    def __init__(
        self,
        power_mw: float = 1,               # Charge and discharge power
        energy_mwh: float = 2,             # Energy bought per cycle
        efficiency: float = 0.9,           # Round-trip efficiency
    ): ...
    def profit(self, forecast: np.ndarray, actual: np.ndarray, periods_per_hour: int = 1) -> np.ndarray: ...  # Days x periods in, EUR per day out
```

`epftoolbox2.evaluators.stats` holds the kernels behind them, for aligned loss series: `dm_test`, `ratio_interval`, `holm`, `model_confidence_set`, `rank_bootstrap`, `minimum_detectable_effect`, `newey_west_variance`, `default_lag`, `block_bootstrap_indices`.

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
    def __init__(self, path: str, extra_columns: List[str] = None, summary_path: str = None): ...
    def export(self, report: EvaluationReport) -> None: ...
```
