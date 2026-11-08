---
title: ScaledMAEEvaluator
description: Mean of each error divided by a source-data column, such as installed capacity
---

The mean of each forecast's absolute error divided by a value of the data at the forecast's target row: with installed capacity
(or a capacity proxy) it is the MAE as a share of capacity, which stays comparable while capacity grows. Unlike
[NMAEEvaluator](/epftoolbox2/evaluators/nmae/), each error is divided by its own row's value.

## Formula

MAE/c = (1/n) × Σ |ŷᵢ − yᵢ| / cᵢ

Where cᵢ is the value of `column` in the report's source data at forecast i's target date and period, and the sum runs over the
forecasts with a finite prediction and actual and a finite positive cᵢ.

## Basic Usage

```python
from epftoolbox2.evaluators import ScaledMAEEvaluator

evaluator = ScaledMAEEvaluator("installed_capacity_solar")
```

## In Pipeline

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import RidgeModel
from epftoolbox2.evaluators import MAEEvaluator, ScaledMAEEvaluator

pipeline = (
    ModelPipeline()
    .add_model(RidgeModel(predictors=predictors, name="Ridge"))
    .add_evaluator(MAEEvaluator(), ScaledMAEEvaluator("cap"))
)

report = pipeline.run(df, test=("2025-07-01", "2026-06-30"), target="solar")
print(report.summary())  # columns MAE/cap (a fraction) and "MAE/cap left out"
```

## Notes

- **Two columns.** `MAE/{column}` and `MAE/{column} left out`, the number of forecasts with a finite error but no finite positive
  value of the column, which are not in the mean. Each column name carries the column, so two of these evaluators on different
  columns are two columns.
- **Joined like `CsvExporter(extra_columns=)`**: by the forecast's target date and period, at the results' frequency, so the source
  data must be at that frequency (hourly results need an hourly column). On the day clocks fall back, the repeated hour takes its
  second row, the one a run forecasts from. A report of `ModelPipeline` has the data it ran on as its `source_data`; an
  `EvaluationReport` built by hand needs `source_data=`. `aggregate()` leaves the evaluator out with a warning, since its report has
  no source data.
- **Checked before any fit.** `run()` (and each task of a SLURM array) raises when the data lacks the column, so a typo costs no
  fits. A report from files needs the data: `ModelPipeline.report(save_dir, data=...)` raises without it, and
  `Workflow.report()` (`epf report`) runs the data pipeline for it.
- **The source's time zone is the results'.** The join reads the source's wall clock, so with results that know their zone a
  source indexed in another one (or in none) raises.
- The value is read at the **target** row: for a capacity that is known only later, use a column that holds, on each row, what was
  known when the forecast was issued (the renewables study's proxy: the largest actual of the 365 days up to D-1).
- A fraction; multiply by 100 for the percentage the renewables study prints as `MAE_cap_pct`.
- Serializable to YAML: `{class: ScaledMAEEvaluator, params: {column: cap}}`.
