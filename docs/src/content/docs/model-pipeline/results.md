---
title: Working with Results
description: Using EvaluationReport to analyze model performance
---

# Working with Results

The `ModelPipeline.run()` method returns an `EvaluationReport` object with various analysis methods.

## EvaluationReport Methods

```python
report = pipeline.run(...)

# Overall summary per model (with the number of forecasts each metric is based on)
summary = report.summary()

# Planned vs produced forecasts per model, and why tasks gave no forecast
counts = report.forecast_counts()

# Breakdown by hour of day
by_hour = report.by_hour()

# Breakdown by forecast horizon
by_horizon = report.by_horizon()

# Combined hour × horizon
by_hour_horizon = report.by_hour_horizon()

# Breakdown by year
by_year = report.by_year()

# Combined year × horizon
by_year_horizon = report.by_year_horizon()

# Combined issue-day weekday × horizon, and target-day weekday × horizon
by_run_weekday_horizon = report.by_run_weekday_horizon()
by_target_weekday_horizon = report.by_target_weekday_horizon()

# Iterate (model_name, DataFrame of per-forecast run_date/target_date/hour/horizon/day_in_test/actual/prediction)
for model_name, details in report.iter_details():
    ...

# summary() plus every by_*() breakdown above, computed in one pass over the results (what ExcelExporter uses)
sheets = report.compute_all()  # {"summary": ..., "hour": ..., ..., "target_weekday_horizon": ...}
```

## Example Output

```python
print(report.summary())
#    model  forecasts      MAE
# 0    OLS      5038  26.0199
# 1  Lasso      5038  28.1098

print(report.forecast_counts())
#    model  planned  produced  missing  missing_test_predictors
# 0    OLS     5040      5038        2                        2
# 1  Lasso     5040      5038        2                        2

print(report.by_horizon())
#    model  horizon      MAE
# 0    OLS        1  18.5432
# 1    OLS        2  22.1234
# 2    OLS        7  35.4567
```

## Forecast counts

A task (period × horizon × run day) that cannot give a forecast is skipped and counted, never dropped silently. `report.forecast_counts()` has one row per model with `planned`, `produced`, `missing` and one column per skip reason:

| Reason | Meaning |
|---|---|
| `missing_test_predictors` | A predictor is NaN on the issue day's row |
| `missing_test_row` | The issue day has no row for that period (a gap in the data) |
| `too_few_rows` | The training window has no rows before the issue day |
| `no_finite_training_rows` | Every training row has a NaN predictor or target |
| `fit_error` | The model raised `ValueError` (scikit-learn does this on degenerate input) |

A shortfall is logged as a warning when the model runs, and a model that produces nothing raises. Any other exception inside a task stops the run (results already written to `save_dir` are kept, so a rerun resumes).

Models are only comparable on the forecasts they all produced: check that `produced` agrees before comparing metrics.

## Accessing Raw Predictions

```python
# All predictions as a tidy DataFrame (no actuals needed, e.g. in forecast-only mode)
predictions = report.predictions()

print(predictions.head())
#      run_date target_date  hour  horizon  prediction model
# 0  2024-02-01  2024-02-02     0        1       45.23   OLS
```

## Reports from saved results

`ModelPipeline.report(save_dir, test_start=..., test_end=..., horizon=..., freq=...)` builds a report from the per-model JSONL files a previous run wrote to `save_dir` and runs the evaluators and exporters. `Workflow.report()` / `epf report` use it to merge the results of a SLURM array (see [Workflow](/epftoolbox2/workflow/overview/#run-only-one-model-slurm-arrays)).

If any model's file has fewer lines than the planned count for `test_start`..`test_end` × `horizon` × `freq` (an array task that failed or never ran), `report()` warns, naming the models and their produced/planned/missing counts, and notes that metrics are then computed on different forecast sets for those models.
