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
        engine: str = "off",                               # "on": engine groups, fits from shared sums; "check": and a sample on the model path too
    ): ...

    def add_model(self, model: Union[BaseModel, PooledModel]) -> "ModelPipeline": ...
    def add_ensemble(self, ensemble: Ensemble) -> "ModelPipeline": ...  # Members are resolved when the pipeline runs
    def add_evaluator(self, *evaluators: Evaluator) -> "ModelPipeline": ...  # One or several; chainable
    def add_exporter(self, exporter: Exporter) -> "ModelPipeline": ...

    def run(
        self,
        data: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],  # Input DataFrame, or {zone: frame}: a run over zones
        test_start: str = None,           # Test period start
        test_end: str = None,             # Test period end
        target: str = "price",            # Target column
        horizon: Optional[int] = None,    # Every horizon from 1 to it; 7 if neither it nor horizons is given
        freq: str = "1h",                 # "1h", "15min" or "1D" (one row per date at one time of day); must match the data
        save_dir: Optional[str] = None,   # Result cache directory (one JSONL file, or Parquet directory, per model)
        forecast_only: bool = False,      # Skip evaluators/exporters; dates default to today
        allow_manifest_mismatch: bool = False,  # Overwrite a disagreeing manifest and resume anyway
        require_manifest: bool = False,   # Refuse results in save_dir that have no manifest
        validation: Union[Tuple[Any, Any], str, None] = None,  # (start, end) or "previous_year"; alone, only those days
        test: Optional[Tuple[Any, Any]] = None,  # (start, end), the same as test_start and test_end
        horizons: Optional[Sequence[int]] = None,  # Only these horizons, e.g. [1, 3, 7] or range(3, 10); not with horizon
        issue_days: Optional[Sequence[str]] = None,  # Only these weekdays, e.g. ["friday"]; any case, all seven = every day
        save_format: str = "jsonl",       # "jsonl" or "parquet": <name>.jsonl or the directory <name>.parquet in save_dir
        save_coefficients: bool = True,   # False writes coefficients as []; recorded in the manifest, compared on resume
    ) -> EvaluationReport: ...            # A PanelReport for a run over zones

    def report(                           # Report from the results a run wrote to save_dir, JSONL or Parquet
        self,
        save_dir: Union[str, Path],
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        horizon: Optional[int] = None,         # As in run; neither: what the manifests record (7 without one)
        freq: Optional[str] = None,            # As in run; None: what the manifests (else the records) record, "1h" if none
        forecast_only: bool = False,
        data: Optional[pd.DataFrame] = None,   # only for exporters that join source columns; {zone: frame} for zone folders
        allow_missing: bool = False,           # else FileNotFoundError when a model has no results
        validation: Union[Tuple[Any, Any], str, None] = None,  # What run was told; each must be what the manifests record
        test: Optional[Tuple[Any, Any]] = None,  # None of the windows: every window the manifests record
        horizons: Optional[Sequence[int]] = None,  # As in run; only records of these horizons are read
        issue_days: Optional[Sequence[str]] = None,  # As in run; None reads them from the manifests
    ) -> EvaluationReport: ...            # A PanelReport for the save_dir of a run over zones (zone_run.json)

    def verify(                           # Leakage probe and positive control on sampled issue days
        self,
        data: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],  # The models' input frame, source data with rebuild, or {zone: frame}
        test_start: str,                  # First issue day to sample from
        test_end: str,                    # Last issue day to sample from
        target: str = "price",            # Target column
        horizon: Optional[int] = None,    # Every horizon from 1 to it; 1 if neither it nor horizons is given
        freq: str = "1h",                 # "1h", "15min" or "1D"; must match the data
        days: int = 5,                    # Issue days per model
        seed: int = 0,                    # Picks the days and the noise
        models: Optional[Sequence[str]] = None,  # Names of the models to verify; None verifies all
        positive_control: bool = True,    # False: `ok` only says that no forecast moved
        rebuild: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,  # Builds the input from source data
        horizons: Optional[Sequence[int]] = None,  # The horizons a run lists, in place of horizon
        issue_days: Optional[Sequence[str]] = None,  # The weekdays a run lists: only those are sampled
        device: Optional[str] = None,     # Set on every model with an attribute named device, e.g. "cpu"
        co_batch: Optional[bool] = None,  # Probe over runs holding the other items of a call; None: models that predict together, name them or have a card
        rerun: Optional[bool] = None,     # Run each day twice first, from other seeds; any difference is inconclusive. None: as co_batch
    ) -> VerifyReport: ...

    def explain(                          # What one forecast's predictors hold and when each is known; no side effects
        self,
        data: pd.DataFrame,               # The frame the models run on
        model: str,                       # The model's name
        test: Optional[Tuple[Any, Any]] = None,  # (first, last) issue day; required for a predictor function of ctx
        horizon: Optional[int] = None,    # Days ahead of the issue day of the forecast shown; 1 if not given
        hour: Optional[int] = None,       # Its period of the target day: 0-23, 0-95 for 15-minute data, 0 for "1D". None: 12, or 0 at "1D"
        target: str = "price",            # Target column
        freq: str = "1h",                 # "1h", "15min" or "1D"; must match the data
        echo: bool = True,                # Print the tables
        horizons: Optional[Sequence[int]] = None,  # The horizons a run lists, in place of horizon: shown at the largest
    ) -> Dict[str, pd.DataFrame]: ...     # "predictors", "grouped"; "lists" for a predictor function of ctx;
                                          # "newest_target" for newest_target="published"

    availability: str                     # The level in force; assign None to go back to the default
    engine: str                           # "off", "on" or "check"; saved to YAML only when not "off"

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
    @property
    def devices(self) -> Dict[str, str]: ...    # Model -> the device verify(device=) ran it on

    def status(self, model: str) -> str: ...    # "ok", "leak" or "inconclusive"
    def table(self) -> pd.DataFrame: ...        # model, days, forecasts, changed, max change, control moved, rerun, co_batch, status
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
        model_horizon: Optional[int] = None,                  # Every horizon from 1 to it; 7 if neither is given
        model_freq: str = "1h",                               # "1h", "15min" or "1D"
        model_forecast_only: bool = False,                    # Skip evaluators/exporters
        max_processes: Optional[int] = None,                  # Worker process count
        threads_per_process: Optional[int] = None,            # Threads per process
        cache_path: Optional[str] = None,                     # Cache directory (overrides data_cache)
        model_index: Optional[Union[int, str]] = None,        # Array task: models[model_index], or its block with models_per_task; all models if None. "$VAR" reads the variable
        model_save_dir: Optional[str] = None,                 # One JSONL per model, required with model_index
        model_allow_manifest_mismatch: bool = False,          # Overwrite a disagreeing manifest and resume
        model_require_manifest: bool = False,                 # Refuse results that have no manifest
        model_validation: Any = None,                         # [start, end] or "previous_year"
        model_test: Optional[Sequence[Any]] = None,           # [start, end], the same as model_test_start and _end
        models_per_task: Optional[int] = None,                # With model_index: task i runs models [i*k, (i+1)*k)
        model_horizons: Optional[Sequence[int]] = None,       # Only these horizons, e.g. [1, 3, 7]; not with model_horizon
        model_issue_days: Optional[Sequence[str]] = None,     # Only these weekdays, e.g. ["friday"]
        model_save_format: str = "jsonl",                     # "parquet": one Parquet directory per model (YAML: save_format)
        model_save_coefficients: bool = True,                 # False writes coefficients as [] (YAML: save_coefficients)
    ): ...

    def run(self) -> EvaluationReport: ...
    def report(self, allow_missing: bool = False) -> EvaluationReport: ...  # Merge save_dir, run evaluators/exporters
    def array_size(self) -> int: ...                          # How many array tasks cover the models: 0 .. size - 1

    def save(self, path: Union[str, Path]) -> None: ...

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Workflow": ...
```

`model_pipeline.engine: on` (or `check`) in a workflow YAML overrides the engine of the pipeline it loads, as `availability` does.

`Workflow.run()` always sets `OMP_NUM_THREADS=1` and `PYTHON_GIL=0` in the environment before starting workers. The GIL setting reaches only workers that start as new interpreters (Windows); to run the main process and its forked workers with the GIL off, start Python with `PYTHON_GIL=0` set.
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
    def vintages(self) -> pd.DataFrame: ...         # model, predictor, series, class, ctx
    def provenance(self) -> pd.DataFrame: ...       # model, family, licence, licence_url, commercial_use, revenue_cap, attribution, weights, sha256, runtime, corpus, corpus_end, released, validation_verdict, test_verdict
    def training_days(self) -> pd.DataFrame: ...    # model, list, predictors, forecasts, periods, horizons, smallest_days, median_days, largest_days, thin_forecasts
    def choices(self) -> pd.DataFrame: ...          # ensemble, segment, chosen, k, validation_mae, validation_forecasts, validation_share, validation_start, validation_end, test_start, test_end; attrs: member_mae, k_mae, limited_by
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
    def by_column(
        self,
        column: str,                               # A column of source_data, joined at each forecast's target date and period
        bins: Optional[Sequence[float]] = None,    # Increasing edges, intervals closed on the left; None: the values themselves
    ) -> pd.DataFrame: ...                          # model, <column>, forecasts, metrics; attrs: by_column, bins, no_value
    def aggregate(
        self,
        freq: str = "1D",                          # "1D": daily series; "1h": hourly ones from 15-minute results
        agg: str = "max",                          # "max", "min", "mean", "spread" (max - min); at 1D also "peak", "offpeak"
        peak_hours: Tuple[int, int] = (8, 20),     # Local hours of the peak block, the end left out (08:00-19:59)
    ) -> EvaluationReport: ...                      # Whole days (hours) only; dropped ones in its forecast_counts() as incomplete_day (incomplete_hour)
    def error_split(self, period: str = "1h") -> pd.DataFrame: ...  # model, days, forecasts, mse, level, hour, quarter (15-minute results)
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
        block: int = 7,                    # Bootstrap block length in issue days (observations)
        reps: int = 4000,
        seed: int = 0,
    ) -> pd.DataFrame: ...  # model, benchmark, horizon, n_days, n_forecasts, mae, mae_benchmark, ratio, ci_low, ci_high, statistic, p_value, p_holm, m, mde, verdict

    def mcs(
        self,
        alpha: float = 0.10,
        statistic: str = "R",              # "R" (T_R) or "max" (T_max)
        block: int = 7,                    # Mean block length of the stationary bootstrap, in issue days
        reps: int = 1000,
        seed: int = 0,
        models: Optional[Sequence[str]] = None,  # Default: all; fewer than 3 gives NaN with a warning
        horizon: Union[int, str, None] = None,   # None or "all": daily losses averaged per run date; a number
    ) -> pd.DataFrame: ...  # model, n_days, n_forecasts, mae, ratio_to_best, p_value, in_90, in_75; attrs: n_days, n_forecasts, limited_by, clones

    def ranks(
        self,
        block: int = 7,
        reps: int = 4000,
        seed: int = 0,
        level: float = 0.90,               # Coverage of the rank interval
        horizon: Union[int, str, None] = None,
        models: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame: ...  # model, n_days, n_forecasts, mae, rank, p_best, rank_low, rank_high; attrs: n_days, n_forecasts, limited_by, clones

    def compare(
        self, a: str, b: str, horizon: Union[int, str, None] = None
    ) -> pd.DataFrame: ...  # One row; ratio = mae_a / mae_b: a, b, horizon, n_days, n_forecasts, mae_a, mae_b, ratio, ci_low, ci_high, statistic, p_plain, p_hac, hours_a_better, hours_b_better, hours_tested, mde

    def leaderboard(
        self, benchmark: Optional[str] = None, horizon: Union[int, str, None] = None
    ) -> pd.DataFrame: ...  # By MAE: model, n_forecasts, MAE, n_days, and with a benchmark n_common, ratio, ci_low, ci_high, p_value, p_holm
```

See [Comparing Models](/epftoolbox2/model-pipeline/comparing-models/). Every method keeps to the report's window, see [Validation and Test](/epftoolbox2/model-pipeline/validation-and-test/).

### PanelReport

```python
class PanelReport:                          # What run(data={zone: frame}) returns: one EvaluationReport per zone
    def __init__(self, zones: Mapping[str, EvaluationReport], target: Optional[str] = None): ...

    zones: Dict[str, EvaluationReport]      # zone -> its report
    reports: Dict[Tuple[str, Optional[str]], EvaluationReport]  # (zone, target) -> its report
    targets: List[Optional[str]]

    @classmethod
    def concat(cls, panels: Iterable["PanelReport"]) -> "PanelReport": ...  # several targets in one panel

    def summary(self) -> pd.DataFrame: ...           # one row per (zone, model)
    def forecast_counts(self) -> pd.DataFrame: ...   # one row per (zone, model)
    def choices(self) -> pd.DataFrame: ...           # each zone's choices(), `zone` first
    def availability(self) -> pd.DataFrame: ...      # each zone's availability(), `zone` first
    def log_gain(self, benchmark: str, by: Sequence[str] = ("zone", "target", "horizon"),
                 models: Optional[Sequence[str]] = None, cells: bool = False,
                 common: bool = True) -> pd.DataFrame: ...   # common: every model on the same keys and cells
    def cells_lost(self, benchmark: str, by: Sequence[str] = ("zone", "target", "horizon"),
                   models: Optional[Sequence[str]] = None, common: bool = True) -> pd.DataFrame: ...
```

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
                 prefix: str = "",
                 price_exaa: bool = False): ...  # True: adds price_exaa (EXAA auction), AT and DE_LU only
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

class EnergyChartsSource(DataSource):
    def __init__(self, country: str = "de",
                 type: Optional[List[str]] = None,  # "forecast" | "power" | "price" | "cbet" | "cbpf"; None: forecast, power, price
                 bzn: Optional[str] = None,  # price zone; None: the country's single zone
                 forecast_types: Optional[List[str]] = None,  # "day-ahead" | "intraday" | "current"; None: ["day-ahead"]
                 production_types: Optional[List[str]] = None,  # "solar" | "wind_onshore" | "wind_offshore" | "load"; None: all
                 power: Optional[Union[List[str], str]] = None,  # series of /public_power to keep; None: all
                 allow_restricted_prices: bool = False,  # zones outside CC BY 4.0 are for private use only
                 prefix: str = "ec", cache_dir: Optional[str] = None, use_cache: bool = True): ...
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...  # [start, end), UTC, no key

class EpftoolboxBenchmarkSource(DataSource):
    def __init__(self, dataset: str, cache_dir: Optional[str] = None, prefix: str = ""): ...
    index_timezone: str   # property: fixed-offset zone of the index
    test_period: tuple    # property: the paper's test period (delivery days)
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...

class PseSource(DataSource):
    def __init__(
        self,
        datasets: Optional[List[str]] = None,  # None = ["kse-load", "csdac-pln"]
        columns: Optional[List[str]] = None,  # None = every column of the datasets
        prefix: str = "pse",
        pdgsz_cutoff: str = "19:00",  # "HH:MM" Warsaw on D-1
    ): ...
    def output_columns(self) -> List[str]: ...
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...  # [start, end), 15-minute grid

class ForecastSource(DataSource):
    def __init__(self, path: Any, model: str, column: str,
                 horizon: int = 1,  # the upstream horizon read: days from issue day to target day
                 available: Optional[str] = None,  # "HH:MM" in the upstream issue zone, not before its issue time
                 agg: Optional[str] = None): ...  # "min" | "max" | "mean" | "peak": emits {column}_day_{agg}
    output_column: str  # column, or {column}_day_{agg}
    def availability(self, upstream: Optional[Upstream] = None) -> Availability: ...  # the rule with_source_rules gives
    def content_digest(self) -> Optional[str]: ...  # the upstream's manifest digest and results identity
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...  # [start, end), UTC, at target instants
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
                 stats: List[str] = ["min", "max", "mean", "last"],  # or "peak": the mean over peak_hours
                 freq: str = "1h",
                 peak_hours: Tuple[int, int] = (8, 20)): ...  # local hours, the end left out
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

class PeriodStatsTransformer(Transformer):
    def __init__(self, columns: Union[str, List[str]],
                 period: str = "1h",  # names {column}_hour_mean, {column}_hour_dev
                 stats: List[str] = ["mean", "dev"],
                 freq: str = "15min"): ...  # must divide the period
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
    def __init__(self, predictors: Union[List, Callable], training_window: int = 365, *,
                 fit_intercept: bool = False, name: str = "Model", scaler: ScalerSpec = "standard",
                 newest_target: str = "d-1", refit: Optional[str] = None,
                 binary_mask: Optional[str] = None, normalise_by: Optional[str] = None,
                 bounds: Optional[Tuple[Optional[float], Optional[float]]] = None, zero_when: Optional[str] = None): ...

class LassoCVModel(BaseModel):
    def __init__(self, predictors: Union[List, Callable], training_window: int = 365,
                 cv: int = 5, max_iter: int = 10000, *, fit_intercept: bool = True,
                 name: str = "Model", scaler: ScalerSpec = "standard", newest_target: str = "d-1",
                 refit: Optional[str] = None, binary_mask: Optional[str] = None, normalise_by: Optional[str] = None,
                 bounds: Optional[Tuple[Optional[float], Optional[float]]] = None, zero_when: Optional[str] = None): ...

class RidgeModel(BaseModel):                # penalty chosen by leave-one-out (RidgeCV, cv=None)
    def __init__(self, predictors: Union[List, Callable], training_window: int = 365,
                 alphas: Optional[Sequence[float]] = None, *, fit_intercept: bool = True,
                 name: str = "Model", scaler: ScalerSpec = "standard", newest_target: str = "d-1",
                 refit: Optional[str] = None, binary_mask: Optional[str] = None, normalise_by: Optional[str] = None,
                 bounds: Optional[Tuple[Optional[float], Optional[float]]] = None, zero_when: Optional[str] = None): ...

class LEARModel(BaseModel):                 # Lasso, penalty by AIC/BIC on the LARS path + Lago's refit, or blocked CV
    def __init__(self, predictors: Union[List, Callable], training_window=(56, 84, 728, "expanding"),
                 criterion: str = "bic", solver: str = "lars", *, scaler: ScalerSpec = "asinh",
                 name: str = "LEAR", newest_target: str = "d-1", refit: Optional[str] = None,
                 binary_mask: Optional[str] = None, normalise_by: Optional[str] = None,
                 bounds: Optional[Tuple[Optional[float], Optional[float]]] = None, zero_when: Optional[str] = None): ...

    @classmethod
    def lago(cls, exogenous: Sequence[str], *, price: str = "price", calendar_prefix: str = "",
             name: str = "LEAR") -> "LEARModel": ...   # Lago et al. (2021): AIC, "cd", Invariant, [49, 77, 1085, 1449]

class NaiveModel(BaseModel):                # one column's value on the issue row, no fit
    def __init__(self, predictors: Union[List, Callable], name: str = "Naive", *,
                 known_only: bool = True, bounds=None): ...

class PooledModel:                          # one fit over the zones of run(data={zone: frame})
    def __init__(self, model: BaseModel, *, over: Union[str, Sequence[str]] = ("zone",),
                 zone_feature: Optional[str] = "onehot", missing: str = "raise",
                 name: Optional[str] = None): ...

class HistGBModel(BaseModel):               # scikit-learn's HistGradientBoostingRegressor, pooled over periods by default
    def __init__(self, predictors: List[str], training_window: Union[int, str] = "expanding",
                 refit: Optional[str] = "7D", periods: str = "pooled", loss: str = "l1", max_iter: int = 500,
                 learning_rate: float = 0.05, max_leaf_nodes: int = 31, min_samples_leaf: int = 20,
                 l2_regularization: float = 1.0, max_features: float = 0.5, random_state: int = 0,
                 name: str = "hgb", *, scaler: Any = INVARIANT_TARGET, newest_target: str = "d-1",
                 binary_mask: Optional[str] = None, normalise_by: Optional[str] = None,
                 bounds: Optional[Tuple[Optional[float], Optional[float]]] = None, zero_when: Optional[str] = None): ...

class LightGBMModel(BaseModel):             # LightGBM's LGBMRegressor, pooled over periods by default ([lightgbm])
    def __init__(self, predictors: List[str], training_window: Union[int, str] = "expanding",
                 refit: Optional[str] = "7D", periods: str = "pooled", objective: str = "l1", n_estimators: int = 500,
                 learning_rate: float = 0.05, num_leaves: int = 31, min_child_samples: int = 20,
                 subsample: float = 0.8, subsample_freq: int = 1, colsample_bytree: float = 0.5,
                 reg_lambda: float = 1.0, random_state: int = 0, threads: int = 1, name: str = "lgbm", *,
                 scaler: Any = INVARIANT_TARGET, newest_target: str = "d-1",
                 binary_mask: Optional[str] = None, normalise_by: Optional[str] = None,
                 bounds: Optional[Tuple[Optional[float], Optional[float]]] = None, zero_when: Optional[str] = None): ...

class XGBoostModel(BaseModel):              # XGBoost's XGBRegressor (tree_method="hist") ([xgboost])
    def __init__(self, predictors: List[str], training_window: Union[int, str] = "expanding",
                 refit: Optional[str] = "7D", periods: str = "pooled", objective: str = "l1", n_estimators: int = 500,
                 learning_rate: float = 0.05, max_depth: int = 6, subsample: float = 0.8,
                 colsample_bytree: float = 0.5, min_child_weight: float = 5.0, reg_lambda: float = 1.0,
                 random_state: int = 0, threads: int = 1, device: str = "cpu", name: str = "xgb", *,
                 scaler: Any = INVARIANT_TARGET, newest_target: str = "d-1",
                 binary_mask: Optional[str] = None, normalise_by: Optional[str] = None,
                 bounds: Optional[Tuple[Optional[float], Optional[float]]] = None, zero_when: Optional[str] = None): ...

class CatBoostModel(BaseModel):             # CatBoost's CatBoostRegressor ([catboost])
    def __init__(self, predictors: List[str], training_window: Union[int, str] = "expanding",
                 refit: Optional[str] = "7D", periods: str = "pooled", objective: str = "l1", iterations: int = 1000,
                 learning_rate: float = 0.05, depth: int = 8, l2_leaf_reg: float = 5.0, random_state: int = 0,
                 threads: int = 1, name: str = "catboost", *, scaler: Any = INVARIANT_TARGET,
                 newest_target: str = "d-1",
                 binary_mask: Optional[str] = None, normalise_by: Optional[str] = None,
                 bounds: Optional[Tuple[Optional[float], Optional[float]]] = None, zero_when: Optional[str] = None): ...

class ResidualModel(BaseModel):             # a base's out-of-sample forecasts + a learner fitted on their residuals
    def __init__(self, base: Union[BaseModel, str], learner: BaseModel,
                 training_window: Union[int, str] = "expanding", refit: Optional[str] = "7D",
                 warmup: Optional[int] = None, name: str = "residual"): ...

class TorchModel(BaseModel):                # a torch.nn.Module class of your own on the library's trainer ([torch])
    def __init__(self, module: Union[str, type], predictors: Union[List, Callable, None] = None,
                 inputs: Optional[Dict[str, List[str]]] = None, module_params: Optional[Dict[str, Any]] = None,
                 output: str = "day", training_window=728, refit: Optional[str] = "7D", seeds: int = 4,
                 loss: str = "l1", lr: float = 3e-4, max_epochs: int = 300, patience: int = 20, batch_size: int = 64,
                 early_stopping: float = 0.2, weight_decay: float = 0.0, stack: str = "auto",
                 pad_to: Optional[int] = None, device: str = "auto", tf32: bool = False, random_state: int = 0,
                 newest_target: str = "d-1", binary_mask: Optional[str] = None, scaler: ScalerSpec = "standard",
                 name: str = "torch", module_hash: Optional[str] = None, normalise_by: Optional[str] = None,
                 bounds: Optional[Tuple[Optional[float], Optional[float]]] = None, zero_when: Optional[str] = None): ...

class MLPHourlyModel(TorchModel):           # nets.MLP, output="period": one network per (period, horizon, refit day)
    def __init__(self, predictors: Union[List, Callable], hidden: Sequence[int] = (64,), dropout: float = 0.0,
                 activation: str = "relu", training_window=728, refit: Optional[str] = "7D", seeds: int = 3,
                 loss: str = "l1", lr: float = 3e-4, max_epochs: int = 300, patience: int = 20, batch_size: int = 64,
                 early_stopping: float = 0.2, weight_decay: float = 0.0, stack: str = "auto",
                 pad_to: Optional[int] = None, device: str = "auto", tf32: bool = False, random_state: int = 0,
                 newest_target: str = "d-1", binary_mask: Optional[str] = None, scaler="standard",
                 name: str = "mlp_hourly", module_hash: Optional[str] = None, normalise_by: Optional[str] = None,
                 bounds: Optional[Tuple[Optional[float], Optional[float]]] = None, zero_when: Optional[str] = None): ...

class MLPDailyModel(TorchModel):            # nets.MLP, output="day": one network per (horizon, refit day), P outputs
    def __init__(self, predictors: Union[List, Callable], hidden: Sequence[int] = (256, 128), dropout: float = 0.1,
                 activation: str = "relu", training_window=728, refit: Optional[str] = "7D", seeds: int = 4,
                 loss: str = "l1", lr: float = 3e-4, max_epochs: int = 300, patience: int = 20, batch_size: int = 64,
                 early_stopping: float = 0.2, weight_decay: float = 0.0, stack: str = "auto",
                 pad_to: Optional[int] = None, device: str = "auto", tf32: bool = False, random_state: int = 0,
                 newest_target: str = "d-1", binary_mask: Optional[str] = None, scaler="standard",
                 name: str = "mlp_daily", module_hash: Optional[str] = None): ...
```

`LightGBMModel`, `XGBoostModel` and `CatBoostModel` are in [Gradient-Boosted Trees](/epftoolbox2/models/trees/): they build, save and load without their package (one extra each), a fit's `threads`, the package's version and XGBoost's `device` are compared on resume, and `threads > 1` is refused where workers fork (Linux, macOS). `HistGBModel` and `ResidualModel` are in [Residual Model](/epftoolbox2/models/residual/): `INVARIANT_TARGET` is
`{"name": "asinh", "params": {"apply_to": "target", "standardise": False}}` (`epftoolbox2.models.hgb`), the predictors as they are
and the target as epftoolbox's Invariant; `base` is a model or the name of one added to the pipeline before the `ResidualModel`.

`TorchModel`, `MLPHourlyModel` and `MLPDailyModel` build, save and load without torch, and need the `[torch]` extra to fit (the
pre-flight names it). Their seeds are a hash of `random_state` and the fit alone and a forecast is their mean in target units; a
seed that diverged makes every forecast of its fit a `fit_error`. `module` is saved by its import path and `module_hash` (for a class
in a module file, a hash of that file and its package's version; for one from a notebook, `exec`, `__main__` or a function,
`session-` and a token of the class object, so a run of it resumes only within its session), compared with a warning on load and
in the manifest on resume; `save()` raises for a class from `__main__` or a function. See [Neural Networks](/epftoolbox2/models/neural-networks/).

With `ModelPipeline(engine="on")` these four run in engine groups (`BaseModel._engine_capable()`) and `NaiveModel` on its own: OLS, Ridge, LassoCV and LEAR with `scaler="standard"` or a variance-stabilising scaler (LEAR with `solver="lars"`; LassoCV and LEAR `"cv"` from the KFold blocks of their rows) fitted from shared exact sums, within 1e-8 (OLS, LassoCV, LEAR) and 1e-9 (Ridge) of `engine="off"`; an issue row far outside its training values (more than 30 training standard deviations along the eigenvectors of the fit's Gram for OLS and 10 for LEAR and LassoCV; for OLS, refined by one step of iterative refinement, also where the error estimated to remain in the forecast is above 5e-9 EUR/MWh; for Ridge, which has no distance bound, where scikit-learn's own rounding at the row exceeds its budget; with a variance-stabilising scaler also a forecast where its inverse is more than three times as steep as the median of its slopes at the training targets), a LEAR whose chosen columns' Gram has a condition number above 1e6, a LassoCV or LEAR with two near-copy inputs (1 − |correlation| at most 1e-5) and every other fit the sums can't give to those digits fitted as with `"off"` (worst gaps measured within the bounds: 4.8e-9 OLS, 4.6e-9 LassoCV, 6.1e-10 Ridge, 1.1e-9 LEAR), LEAR's `solver="cd"` with the same numbers; `engine="check"` also fits a sample of one fit in about 50 as `"off"` does and records the gaps in the manifest (`engine_check`, with gaps that aren't finite and fits `"off"` can't fit counted), warning about a model over its class's tolerance (see [The Engine](/epftoolbox2/model-pipeline/engine/)). `training_window` of `OLSModel`, `LassoCVModel`, `RidgeModel` and `LEARModel` is days from 1 (an int, default `365`; `LEARModel`'s `[56, 84, 728, "expanding"]`), `"expanding"`, or a list of them such as `[56, 84, 728, "expanding"]` whose windows are fitted separately and averaged inside the model; anything else raises (see [Training Windows](/epftoolbox2/model-pipeline/training-windows/)). `ScalerSpec` is a scaler name (`"standard"`, `"asinh"`, `"npit"`), a `{"name": ..., "params": {...}}` dict or a scaler instance. `fit_intercept` is keyword-only and scikit-learn's; a set value is saved and compared on resume, the default isn't written, so results and files made before it keep their manifests. `RidgeModel`'s `alphas` is the grid of penalties (`None`: `np.logspace(-3, 4, 36)`); its manifest holds the resolved grid, its YAML writes `alphas` only when set, a penalty below 1e-6 raises, and a fit needs at least 3 training rows. `LEARModel`'s `criterion` is `"bic"`, `"aic"` or `"cv"` (old `LassoLarsCV`, five blocked folds, the path's coefficients with no refit) and its `solver` `"lars"` (the exact Lasso at the chosen penalty) or `"cd"` (Lago's coordinate descent, at most 2500 sweeps), `"cv"` with `"lars"` only; both are always written to YAML and the manifest, which also records the kernels' backend as provenance (`kernels`); `LEARModel.lago(exogenous)` takes the exogenous series in the order of the dataset's file (see [LEARModel](/epftoolbox2/models/lear/)). `NaiveModel` takes a list of one name (or a function of `ctx` returning one), forecasts its value on the row of the issue day and has no coefficients; with `known_only` and an information set a value its rule puts after the issue time is no forecast, counted as `not_published` (see [Week-ahead forecasting](/epftoolbox2/model-pipeline/week-ahead/)). It has no `training_window`, `scaler`, `newest_target`, `refit` or `binary_mask`: setting `training_window`, `newest_target`, `refit` or `binary_mask` raises. `binary_mask` of the other four says which training rows decide the binary 0/1 columns a scaler leaves unscaled: `None` (the default) the window's first day without `refit` and each unit's rows with it, `"first_day"` the first by name (it raises with `refit`), `"fit"` each fit's own rows, so that no forecast depends on `test_start`; `"fit"` without `refit` is recorded in the manifest's `training_cut`, and YAML writes it only when set (see [Time and Calendars](/epftoolbox2/model-pipeline/time-and-calendars/#binary-columns-and-test_start)). `normalise_by`,
`bounds` and `zero_when` are a per-row target, `None` by default and written to YAML and the manifest's `model_params` only when
set (compared on resume): `normalise_by` divides each training row's target by its value of that column and multiplies the
forecast by the issue row's, `bounds=(low, high)` clips the forecast after that (in multiples of the scale with `normalise_by`,
either side `None`), `zero_when` makes a forecast 0 with no fit where the issue row's value is 1, with `detail: "zero_when"` in its
record; `{horizon}` may stand in either name. The columns are inputs (checked, hashed, perturbed by `verify`), not regressors, and a
model with any of them set never runs in an engine group. An issue row whose scale is missing is `missing_test_predictors` and one not
positive `non_positive_scale`; a `zero_when` value other than 0, 1 or missing on an issue row raises. `NaiveModel` takes `bounds`
alone; `MLPDailyModel`, a `TorchModel` with `output="day"` and a model that reads history refuse them, and they aren't supported on
a `ResidualModel` or the model a `PooledModel` pools (see [Common Parameters](/epftoolbox2/models/common-parameters/#generation-targets)).

### Predictor functions

`predictors` may be a function with one parameter named `ctx`, see [Predictor Functions](/epftoolbox2/model-pipeline/predictor-functions/).

```python
class PredictorContext:                     # What the function gets for one planned forecast
    horizon: int                            # Days ahead of the issue day, from 1
    period: int                             # 0-23, 0-95 for 15-minute data, 0 for a daily frame
    hour: int                               # The same number as period
    issue_day: datetime.date                # The local calendar date D
    target_day: datetime.date               # D + horizon
    target_weekday: int                     # Weekday of target_day, 0 = Monday
    zone: Optional[str]                     # The data's time zone name
    freq: str                               # "1h", "15min" or "1D"

    def published(self, columns: Union[str, Iterable[str]]) -> bool: ...  # Known at the issue time and not NaN
    def value(self, column: str) -> Any: ...  # The cell of the forecast's row; raises if known only later

class Use:                                  # What the function may return instead of a plain list
    def __init__(self, predictors: Sequence[str], train_on: str = "all"): ...  # "matching": only the past days given this list
```

`PredictorPlan` is what `ModelPipeline` builds before any fit (the distinct lists and which forecast uses which) and passes to `BaseModel.run(plan=...)`. `ModelResultRef.predictor_lists` describes those lists after a run, `ModelResultRef.availability_listing` holds the rows behind `report.availability()`, `ModelResultRef.vintages` those behind `report.vintages()` (None when no input has a class), and `ModelResultRef.matching_lists` the hash and number of predictors of each `train_on="matching"` list, which `report.training_days()` reads the records for. `ModelResultRef.zone` is the data's time zone, which says what periods a date has on the clock (see [Time and Calendars](/epftoolbox2/model-pipeline/time-and-calendars/#what-is-planned)). `ModelResultRef.horizon` is the largest horizon of a run and `ModelResultRef.horizons` the ones it lists, None for every one from 1 to `horizon`; only records of those horizons are read (see [Horizons](/epftoolbox2/model-pipeline/time-and-calendars/#some-horizons-only)). `ModelResultRef.issue_days` is the weekdays a run issued on, as `date.weekday()` numbers (Monday is 0), None for every day; only records issued on them (their `run_weekday`) are read (see [Weekdays](/epftoolbox2/model-pipeline/time-and-calendars/#some-weekdays-only)).

### Model cards

A family that needs an extra or pretrained weights declares a card as its `_card` (see
[Licences and Weights](/epftoolbox2/models/licences-and-weights/)); `model.card` reads it, None for a core model.

```python
from epftoolbox2.models import ModelCard, WeightsRef

class WeightsRef:                           # frozen dataclass
    def __init__(self, repo: str, revision: str, files: Mapping[str, Optional[str]],  # file -> sha256, None: recorded at the first fetch
                 bytes: Optional[int] = None): ...

class ModelCard:                            # frozen dataclass
    def __init__(self, family: str, extra: Optional[str], licence: str, licence_url: str, commercial_use: bool,
                 revenue_cap: Optional[str], attribution: Optional[str], weights: Optional[WeightsRef],
                 runtime: Tuple[str, ...],          # "pkg==1.2.3", "pkg>=1,<2", "pkg@<commit>" or a bare name
                 corpus: str, corpus_end: Optional[date], released: Optional[date],
                 stated_verdict: Optional[str] = None): ...  # overrides the verdict by dates, with its reason
```

The pre-flight raises `MissingWeightsError` (`epftoolbox2.models._weights`) when weights are among its problems, else
`MissingExtraError` (`epftoolbox2._extras`); both are `ImportError`s. `ModelResultRef.provenance` holds a carded model's `card`,
`runtime` and `weights`, which `report.provenance()` shows.

`BaseModel.run()` takes `availability` too, and without a pipeline it defaults to `"raise"`. It also takes `windows`, a `RunWindows`, which is what `ModelPipeline.run(validation=..., test=...)` passes; the returned ref then has a ref per window in `by_window`. And `frame_hashes`, what one `ModelPipeline.run` call shares between its models so that it hashes each column once. And `cut_rules`, an information set a model with `newest_target="published"` reads its newest targets from when `information_set` isn't given, without checking anything with it: the leakage probe runs models so. And `horizons`, `issue_days` and `save_coefficients`, as in `ModelPipeline.run`; its `horizon` defaults to 7 too, and its `save_to` names the format, `<name>.jsonl` or `<name>.parquet`. And `checked`, what `ModelPipeline` found when it checked the model's inputs before any model ran: a run it was found for uses it instead of checking again, and any other run checks its inputs itself. `leakage_probe` and `positive_control` take `horizon` (default 1) or `horizons` the same way. Their `freq`, and `BaseModel.run()`'s, takes `"1D"` as `ModelPipeline.run()`'s does, see [Daily frames](/epftoolbox2/model-pipeline/time-and-calendars/#daily-frames).

### Scalers

```python
from epftoolbox2.scalers import StandardScaler, AsinhScaler, MirrorLogScaler, PolyScaler, NPITScaler, make_scaler, scaler_spec

class StandardScaler(BaseScaler):
    def __init__(self, *, scale_binary: bool = False): ...   # True scales binary 0/1 columns too

class AsinhScaler(BaseScaler):
    def __init__(self, apply_to: Union[str, List[str]] = "all", *, standardise: bool = True): ...  # False: Lago's Invariant

class MirrorLogScaler(BaseScaler):  # "mlog": sgn(u) (log(|u| + 1/c) + log c) on the median/MAD-normalised u
    def __init__(self, apply_to: Union[str, List[str]] = "all", *, c: float = 1 / 3, standardise: bool = True): ...

class PolyScaler(BaseScaler):  # "poly": sgn(u) ((|u| + (c/lam)^(1/(lam-1)))^lam - (c/lam)^(lam/(lam-1)))
    def __init__(
        self, apply_to: Union[str, List[str]] = "all", *, lam: float = 0.125, c: float = 0.05, standardise: bool = True
    ): ...

class NPITScaler(BaseScaler):
    def __init__(self, apply_to: Union[str, List[str]] = "all", *, standardise: bool = True): ...

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

class ShapeMAEEvaluator(Evaluator):
    name = "ShapeMAE"  # Mean |(p - hour mean of p) - (y - hour mean of y)| on whole hours; NaN unless 15-minute results
    def __init__(self, period: str = "1h"): ...  # The only period so far

class NMAEEvaluator(Evaluator):
    name = "nMAE"  # MAE / mean actual, both on the forecasts with a finite prediction and actual

class ScaledMAEEvaluator(Evaluator):
    name = "MAE/{column}"  # Columns "MAE/{column}" (mean |e| / c) and "MAE/{column} left out"
    def __init__(self, column: str): ...  # Source-data column c, joined at each forecast's target date and period

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
        block: int = 7,                    # Bootstrap block length in issue days (observations)
        reps: int = 4000,
        seed: int = 0,
    ): ...

class MCSEvaluator(Evaluator):
    name = "MCS"  # Columns "MCS p", "in MCS", "MCS n_days", "MCS n_forecasts"
    def __init__(
        self,
        alpha: float = 0.10,               # Size of the set
        statistic: str = "R",              # "R" (T_R) or "max" (T_max)
        block: int = 7,                    # Mean block length of the stationary bootstrap, in issue days
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
