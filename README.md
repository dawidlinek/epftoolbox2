# epftoolbox2

A modern Python library for electricity price forecasting with modular data pipelines and model evaluation. Built as a complete rewrite of the original [epftoolbox](https://github.com/jeslago/epftoolbox), this library provides a flexible, extensible framework for downloading energy market data, building forecasting models, and evaluating their performance.

📖 **[Documentation](https://dawidlinek.github.io/epftoolbox2)** — upgrading from an earlier version? Read
[Upgrading](https://dawidlinek.github.io/epftoolbox2/getting-started/upgrading/) first: several fixes change
computed results, and old cached data / results directories need attention.

## Installation

```bash
pip install epftoolbox2
```

or with uv 

```bash
uv add epftoolbox2
```

## Verify Installation

After installation, you can verify your setup and check system information:

```python
import epftoolbox2

epftoolbox2.verify()
```
## Key Features

- **Data Sources**: ENTSOE (load, generation, day-ahead prices on an hourly or 15-minute grid, cross-border transfer capacity, installed capacity; `prefix` to combine zones), Open-Meteo (archived weather forecasts as known at the forecast issue time; OpenMeteoSingleRunsSource picks an exact model run through the Single Runs API; OpenMeteoWeightedSource aggregates several locations into one population- or capacity-weighted series), Calendar (national and regional holidays, bridge days, weekday, hour, month, daylight, market-regime dummies, opt-in school holidays via the OpenHolidays API), CSV files (CsvSource), EpftoolboxBenchmark (the Lago et al. 2021 datasets NP, PJM, FR, BE, DE, downloaded on demand)
- **Transformers**: Resample, Timezone conversion, Lag features (local calendar days), DST normalisation to 24 slots a day (DSTNormalize), daily profile and daily min/max/mean/last stats (DailyProfile, DailyStats), custom column arithmetic (Lambda), Impute (leak-safe gap filling)
- **Presets**: `epftoolbox2.presets` — ready-made predictor lists for the LEAR and expert-model benchmarks from the literature
- **Validators**: Null checks, Continuity (duplicates, gaps, mixed resolutions), DSTStructure (rows per local day), Coverage (per-column outages), Plausibility (SDAC price limits, frozen values), EDA statistics
- **Models**: OLS, LassoCV
- **Scalers**: standardisation (default) or variance-stabilising transformations for spiky prices (asinh, N-PIT), fitted per forecast on training data only
- **Evaluators**: MAE, RMSE, rMAE (relative to a base model)
- **Exporters**: Excel with conditional formatting, Terminal output
- **Caching**: Built-in per-source Parquet cache that fetches only missing ranges, refetches not-yet-published data and is safe for parallel jobs
- **Pipelines**: Data and model pipelines that can be saved and loaded with .yaml files; an optional market timezone makes dates mean local calendar days, duplicate column names across sources raise instead of colliding silently, and `api_key` is written to YAML as an `${ENV_VAR}` reference instead of plain text
- **Multiprocessing**: Process-based parallelism with inner thread pools; SLURM array mode (one model per task) with `epf report` to merge the results
- **Safeguards**: The model's input (resolution, time stamps, column names, predictors) is checked before any fit, and forecasts that could not be made are counted per model (`report.forecast_counts()`)
- **Information Set**: Optional `issue_time` check that every predictor is known at the moment the forecast is issued, with built-in availability rules for the shipped sources and an empirical leakage probe (`epftoolbox2.information_set`)
- **Result Manifests**: Resumable results (`save_dir`) are fingerprinted with the configuration that produced them (the predictors' ordered layout, data up to `test_end` and its time zone, model params, ...); resuming into results made with a different configuration raises instead of silently mixing them (`epftoolbox2.results.manifest`)
- **Extensibility**: Extensible base classes for data sources, transformers, validators, models, evaluators, and exporters

## Quick Start

### Standalone source use

```python
import pandas as pd
from epftoolbox2.data.sources import EntsoeSource

source = EntsoeSource("PL", api_key="YOUR_KEY", type=["load", "price"])
df = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-06-01", tz="UTC"))  # [start, end), UTC
```

### Standalone transformer use

```python
from epftoolbox2.data.transformers import TimezoneTransformer, ResampleTransformer

df = TimezoneTransformer(target_tz="Europe/Warsaw").transform(df)  # sources return UTC
df = ResampleTransformer(freq="1h").transform(df)  # aggregates or step-fills; never fills real gaps
```

### Standalone validator use

```python
from epftoolbox2.data.validators import NullCheckValidator

validator = NullCheckValidator(columns=["load_actual", "price"])
result = validator.validate(df)
print(result)  # VALID/INVALID with errors, warnings and details; result.stats has one row per column
```

### Standalone model use

```python
from epftoolbox2.models import OLSModel

seasonal_indicators = [
    "is_monday_d+{horizon}",
    "is_tuesday_d+{horizon}",
    "is_wednesday_d+{horizon}",
    "is_thursday_d+{horizon}",
    "is_friday_d+{horizon}",
    "is_saturday_d+{horizon}",
    "is_sunday_d+{horizon}",
    "is_holiday_d+{horizon}",
    "daylight_hours_d+{horizon}",
]

model = OLSModel(predictors=["load_actual_d-1", *seasonal_indicators], name="OLS")
report = model.run(df, test_start="2024-04-01", test_end="2024-06-01", target="price")
```

### Standalone evaluator use

```python
from epftoolbox2.evaluators import MAEEvaluator

evaluator = MAEEvaluator()
mae = evaluator.compute(predictions)  # DataFrame with "prediction" and "actual" columns
```

### Standalone exporter use

```python
from epftoolbox2.exporters import ExcelExporter

exporter = ExcelExporter("results.xlsx")
exporter.export(report)  # an EvaluationReport, e.g. from ModelPipeline.run() below
```

### Data Pipeline

Download and process electricity market data from ENTSOE, weather forecasts from Open-Meteo, and calendar features.

```python
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.data.sources import EntsoeSource, OpenMeteoSource, CalendarSource
from epftoolbox2.data.transformers import TimezoneTransformer, ResampleTransformer, LagTransformer
from epftoolbox2.data.validators import NullCheckValidator

pipeline = (
    # timezone is the single place the market zone needs to be written: sources/transformers below that
    # compute their own local time inherit it and leave their own timezone/target_tz unset.
    DataPipeline(timezone="Europe/Warsaw")  # optional: dates below then mean local calendar days
    .add_source(EntsoeSource("PL", api_key="YOUR_KEY", type=["load", "price"]))
    .add_source(OpenMeteoSource(latitude=52.23, longitude=21.01))
    .add_source(CalendarSource("PL", holidays="binary", weekday="onehot", daylight=True))
    .add_transformer(TimezoneTransformer())
    .add_transformer(ResampleTransformer(freq="1h"))
    # load_actual is only known for D once D is over; never offer its same-day (unlagged) value.
    .add_transformer(LagTransformer(columns=["load_actual"], lags=[1], freq="day"))
    # calendar features are known arbitrarily far in advance, so d+1..d+7 is safe.
    .add_transformer(
        LagTransformer(
            lags=range(-7, 0),
            freq="day",
            columns=[
                "is_monday",
                "is_tuesday",
                "is_wednesday",
                "is_thursday",
                "is_friday",
                "is_saturday",
                "is_sunday",
                "is_holiday",
                "daylight_hours",
            ],
        )
    )
    .add_validator(NullCheckValidator(columns=["load_actual", "price"]))
)

df = pipeline.run("2024-01-01", "2024-06-01", cache=True)
```

### Model Pipeline

Train and evaluate forecasting models with built-in metrics and export capabilities.

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import ExcelExporter

pipeline = (
    ModelPipeline()
    .add_model(OLSModel(predictors=["load_actual_d-1", *seasonal_indicators], name="OLS"))
    .add_model(LassoCVModel(predictors=["load_actual_d-1", *seasonal_indicators], name="Lasso"))
    .add_evaluator(MAEEvaluator())
    .add_exporter(ExcelExporter("results.xlsx"))
)

report = pipeline.run(df, test_start="2024-04-01", test_end="2024-06-01", target="price")
```

Pass `issue_time` (and optionally `issue_timezone`) to check every predictor against the time the forecast is
actually issued, before any model is fitted — `ModelPipeline(issue_time="11:30", issue_timezone="Europe/Brussels")`
raises `InformationSetError` if a predictor uses a value not known yet (e.g. same-day `load_actual`). Built-in
availability rules cover the shipped sources; see `epftoolbox2.information_set` for custom rules, accepted
assumptions, and `leakage_probe`, an empirical check that perturbs unknown values and asserts forecasts don't move.



## Examples

See the `examples/` folder for complete working examples.