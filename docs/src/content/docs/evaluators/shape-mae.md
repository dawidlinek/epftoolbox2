---
title: ShapeMAEEvaluator
description: Error of the shape within each hour, for 15-minute forecasts
---

Measures how well a 15-minute forecast gets the movement **inside** each hour, apart from the hour's level: the mean absolute
difference between the forecast's and the actual's deviations from their own hourly means. A forecast that gets every hourly mean
wrong but the four quarter-hours around it right scores 0, and one that repeats its hourly value four times scores the size of the
actual's within-hour movement. Use it next to the MAE of 15-minute results and the MAE of their hourly means
(`report.aggregate("1h", agg="mean")`): together they say whether the error is in the hourly level or in the shape.

## Formula

ShapeMAE = mean over the quarter-hours q of whole hours of |(ŷ_q − ŷ̄_h) − (y_q − ȳ_h)|

Where:
- ŷ_q, y_q = predicted and actual value of quarter-hour q
- ŷ̄_h, ȳ_h = the means of the prediction and of the actual over the four quarter-hours of q's hour

Only **whole hours** count: an hour of one forecast day with all four quarter-hours there, each with a finite prediction and actual.
The hour that occurs twice when clocks fall back has no actual in a run's records, so it never counts; the hour the spring switch
skips has no quarter-hours at all.

## Basic Usage

```python
from epftoolbox2.evaluators import ShapeMAEEvaluator

evaluator = ShapeMAEEvaluator()  # period="1h", the only one so far
```

## In Pipeline

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel
from epftoolbox2.evaluators import MAEEvaluator, ShapeMAEEvaluator

pipeline = (
    ModelPipeline()
    .add_model(OLSModel(predictors=predictors, name="OLS"))
    .add_evaluator(MAEEvaluator(), ShapeMAEEvaluator())
)

report = pipeline.run(df_15min, test=("2026-04-01", "2026-06-30"), freq="15min")
print(report.summary())                                # MAE and ShapeMAE of the quarter-hours
print(report.aggregate("1h", agg="mean").summary())    # MAE of the hourly means
print(report.error_split())                            # level, hour profile and quarter shape of the squared error
```

## Notes

- **15-minute results only.** On hourly or daily results (and in a report from `aggregate()`) the column is NaN: there is no shape
  within the hour. The report tells the evaluator each model's frequency; called directly on a frame, it reads it from the period
  ids (an id above 24 means quarter-hours).
- In `by_hour()` the column is NaN too, since each group holds one quarter-hour of every hour. Breakdowns that keep whole hours
  together (`by_year()`, `by_quarter()`, `by_day()`) have it.
- It is taken on each model's own forecasts, like the MAE: compare models on the same keys (`report.forecast_counts()`).
- `report.error_split()` divides the squared error into the same three parts: the level of the day, the hour profile and the shape
  within the hour. See [Working with Results](/epftoolbox2/model-pipeline/results/#hourly-aggregates-shape-and-the-error-split).
- This is the shape MAE of the resolution study, whose R1 table it reproduces from the stored forecasts: see
  [Resolution Study](/epftoolbox2/reproductions/resolution/#tier-s-the-published-tables-from-the-stored-forecasts).
- Serializable to YAML: `{class: ShapeMAEEvaluator, params: {period: 1h}}`.
