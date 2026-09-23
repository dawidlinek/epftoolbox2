# epftoolbox2

A modern Python library for electricity price forecasting with modular data pipelines and model evaluation. Built as a complete rewrite of the original [epftoolbox](https://github.com/jeslago/epftoolbox), this library provides a flexible, extensible framework for downloading energy market data, building forecasting models, and evaluating their performance.

📖 **[Documentation](https://dawidlinek.github.io/epftoolbox2)** 

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

- **Data Sources**: ENTSOE (load, generation, prices), Open-Meteo (weather forecasts), Calendar (holidays, weekday)
- **Transformers**: Resample, Timezone conversion, Lag features
- **Validators**: Null checks, Continuity checks, EDA statistics
- **Models**: OLS, LassoCV
- **Evaluators**: MAE
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
from epftoolbox2.data.sources import EntsoeSource

source = EntsoeSource("PL", api_key="YOUR_KEY", type=["load", "price"])
df = source.run("2024-01-01", "2024-06-01", cache=True)
```

### Standalone transformer use

```python
from epftoolbox2.data.transformers import ResampleTransformer

transformer = ResampleTransformer(freq="1h")
df = transformer.run(df)
```

### Standalone validator use

```python
from epftoolbox2.data.validators import NullCheckValidator

validator = NullCheckValidator(columns=["load_actual", "price"])
df = validator.run(df)
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

model = OLSModel(predictors=["load_actual", *seasonal_indicators], name="OLS")
report = model.run(df, test_start="2024-04-01", test_end="2024-06-01", target="price")
```

### Standalone evaluator use

```python
from epftoolbox2.evaluators import MAEEvaluator

evaluator = MAEEvaluator()
report = evaluator.run(df, test_start="2024-04-01", test_end="2024-06-01", target="price")
```

### Standalone exporter use

```python
from epftoolbox2.exporters import ExcelExporter

exporter = ExcelExporter("results.xlsx")
report = exporter.run(df, test_start="2024-04-01", test_end="2024-06-01", target="price")
```

### Data Pipeline

Download and process electricity market data from ENTSOE, weather forecasts from Open-Meteo, and calendar features.

```python
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.data.sources import EntsoeSource, OpenMeteoSource, CalendarSource
from epftoolbox2.data.transformers import ResampleTransformer
from epftoolbox2.data.validators import NullCheckValidator

pipeline = (
    DataPipeline(timezone="Europe/Warsaw")  # optional: dates below then mean local calendar days
    .add_source(EntsoeSource("PL", api_key="YOUR_KEY", type=["load", "price"]))
    .add_source(OpenMeteoSource(latitude=52.23, longitude=21.01))
    .add_source(CalendarSource("PL", holidays="binary", weekday="number"))
    .add_transformer(ResampleTransformer(freq="1h"))
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
    .add_model(OLSModel(predictors=["load_actual", *seasonal_indicators], name="OLS"))
    .add_model(LassoCVModel(predictors=["load_actual", *seasonal_indicators], name="Lasso"))
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