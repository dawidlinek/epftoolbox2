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
| `summary_path` | str | `None` | Also write `report.summary()` to this CSV file: one row per model with its number of forecasts (`forecasts`) and every evaluator's columns |

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
- `hour`: period of the day: the hour (0-23) for hourly results, the quarter-hour (0-95) for `freq="15min"` results, 0 for `freq="1D"` results (one row a forecast)
- `target_time`: only for `freq="15min"` results, placed right after `hour`: local start time of that quarter-hour as `HH:MM` (period 5 is `01:15`)
- `horizon`: forecast horizon (1 to max)
- `day_in_test`: day index in the test period (in a run with validation, in the window the report is on)
- `actual`: actual observed value

**Per-model columns:**
- `{model}_prediction`: model's prediction
- `{model}_error`: residual (prediction - actual)

A model with no forecasts in the report's window (an array task that stopped before its test days, say) keeps its two columns, empty, in its place among the models, and the rows of the other models are unchanged. If no model has forecasts, the file holds the header only.

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

All models in one export must have run at the same frequency; mixing hourly, 15-minute and daily (`freq="1D"`) models raises a `ValueError`.

## Extra Columns

Use `extra_columns` to include columns from the source dataset (e.g., calendar features, weather data). Columns are joined by matching `target_date` and `hour` from the results to the source dataset's DatetimeIndex, using the same period of the day: the clock hour for hourly results, the quarter-hour for 15-minute results.

The source dataset must have the resolution of the results, one row per local date and period. For `freq="1D"` results that is a daily source, one row per local date at one time of day (`df.at_time("12:00")`; see [Daily frames](/epftoolbox2/model-pipeline/time-and-calendars/#daily-frames)), else every hour of a date would join period 0. A source of another resolution raises a `ValueError` (`extra_columns need source data at the resolution of the 1h results`) instead of writing a forecast once per source row (a finer source) or leaving most cells empty (a coarser one). Resample or aggregate it first, for example with [ResampleTransformer](/epftoolbox2/transformers/resample/). Only the source rows on the target dates of the file are checked for their spacing.

The join reads the source's local wall clock, so when the results know their time zone (a report of a run, or from files whose manifests record it), the source's index must be in that zone: a frame indexed in UTC, or with no zone, under Europe/Berlin results raises a `ValueError` instead of giving every forecast another hour's value. Results passed as plain lists have no zone, and the source's wall clock is read as given.

When the clocks go back, a source in local time holds the repeated hour twice, once per UTC offset. A forecast for that hour takes the values of the second occurrence, the row a run forecasts from, and stays one row in the file. A source with a naive index cannot tell the two apart and raises.

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
