---
title: CsvExporter
description: Export results to a wide-format CSV
---

Exports detailed prediction results to a wide-format CSV file. Each row represents a single forecast point, with per-model prediction and error columns.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | Required | Output CSV file path |
| `extra_columns` | List[str] | `[]` | Columns from the source dataset to include |
| `summary_path` | str | `None` | Also write `report.summary()` to this CSV file: one row per model with every evaluator's columns |

## Basic Usage

```python
from epftoolbox2.exporters import CsvExporter

exporter = CsvExporter("results.csv")
```

The file at `path` holds forecasts, not metrics. To get the metric columns (MAE, `DM ratio`, `in MCS`, `profit` and so on) in a CSV file too, pass `summary_path`:

```python
exporter = CsvExporter("results.csv", summary_path="summary.csv")
```

## Output Format

The CSV contains the following columns:

**Base columns:**
- `run_date`: date the forecast was made
- `target_date`: date being forecasted
- `hour`: period of the day: the hour (0-23) for hourly results, the quarter-hour (0-95) for `freq="15min"` results
- `target_time`: only for `freq="15min"` results, placed right after `hour`: local start time of that quarter-hour as `HH:MM` (period 5 is `01:15`)
- `horizon`: forecast horizon (1 to max)
- `day_in_test`: day index in the test period
- `actual`: actual observed value

**Per-model columns:**
- `{model}_prediction`: model's prediction
- `{model}_error`: residual (prediction - actual)

**Extra columns** (optional):
- Any columns from the source dataset, joined by `target_date` + `hour` (the period of the day, so a quarter-hour row gets the source value of that quarter-hour)

### Example Output

For a pipeline with models `OLS` and `LassoCV`, and `extra_columns=["is_holiday"]`:

| run_date | target_date | hour | horizon | actual | OLS_prediction | OLS_error | LassoCV_prediction | LassoCV_error | is_holiday |
|----------|-------------|------|---------|--------|---------------|-----------|-------------------|---------------|------------|
| 2024-02-01 | 2024-02-02 | 0 | 1 | 48.50 | 45.23 | -3.27 | 46.10 | -2.40 | 0 |

### 15-minute results

With `freq="15min"` every quarter-hour is its own row: `hour` runs from 0 to 95 and `target_time` names the time of day, so `target_date` + `target_time` is the local timestamp the forecast is for.

| run_date | target_date | hour | target_time | horizon | actual | OLS_prediction | OLS_error |
|----------|-------------|------|-------------|---------|--------|---------------|-----------|
| 2024-02-01 | 2024-02-02 | 5 | 01:15 | 1 | 48.50 | 45.23 | -3.27 |

All models in one export must have run at the same frequency; mixing hourly and 15-minute models raises a `ValueError`.

## Extra Columns

Use `extra_columns` to include columns from the source dataset (e.g., calendar features, weather data). Columns are joined by matching `target_date` and `hour` from the results to the source dataset's DatetimeIndex, using the same period of the day: the clock hour for hourly results, the quarter-hour for 15-minute results. The source dataset must have the resolution of the results.

```python
exporter = CsvExporter(
    "results.csv",
    extra_columns=["is_holiday", "load_forecast", "warsaw_temperature_2m"],
)
```

If a requested column does not exist in the source dataset, it is silently skipped.

## In Pipeline

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator
from epftoolbox2.exporters import CsvExporter

pipeline = (
    ModelPipeline()
    .add_model(OLSModel(predictors=predictors, name="OLS"))
    .add_model(LassoCVModel(predictors=predictors, cv=7, name="LassoCV"))
    .add_evaluator(MAEEvaluator())
    .add_evaluator(RMSEEvaluator())
    .add_exporter(CsvExporter(
        "results.csv",
        extra_columns=["is_holiday", "load_forecast"],
    ))
)

report = pipeline.run(data=df, test_start="2024-02-01", test_end="2024-03-01", target="price", horizon=7)
# Results saved to results.csv
```
