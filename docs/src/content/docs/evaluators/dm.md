---
title: DMEvaluator
description: Each model against a benchmark, with the MAE ratio, its interval and a Diebold-Mariano test
---

Adds a comparison with one benchmark model to the report's tables: the MAE ratio, a bootstrap interval around it and a two-sided Diebold-Mariano (DM) test. Each model is compared on the forecasts it shares with the benchmark, never on a global intersection, so one model with missing forecasts does not change the numbers of the others.

[`report.tests()`](/epftoolbox2/model-pipeline/comparing-models/#testing-against-a-benchmark) gives the same comparison as a table of its own, with the statistic and the minimum detectable effect. See [Comparing Models](/epftoolbox2/model-pipeline/comparing-models/) for how to read it.

## What it computes

For the forecasts a model and the benchmark share (same `run_date`, `target_date`, `hour`, `horizon`, with a finite prediction and actual on both sides):

- **Daily loss:** the mean absolute error over the periods of one (run date, horizon), whatever their number (23, 24 or 25; 92, 96 or 100). With several horizons in a group, a run date's loss is the mean of its horizons' losses.
- **DM test:** on the daily losses, with a Newey-West variance (lag ⌊4(n/100)^(2/9)⌋, 5 at 365 days). `variance="plain"` reproduces epftoolbox. Two-sided; a positive statistic means the model is better.
- **Ratio:** MAE of the model / MAE of the benchmark, both forecast-weighted on the shared forecasts, so below 1 means the model is better. The interval resamples daily error sums in weekly blocks, so it agrees with the ratio on DST days.
- **Holm:** over the models compared with the benchmark in that group.

## Output columns

| Column | Meaning |
|--------|---------|
| `DM ratio` | MAE ratio (1.0 on the benchmark's own row) |
| `DM ci_low`, `DM ci_high` | Bootstrap interval of the ratio |
| `DM p` | Two-sided p-value |
| `DM p_holm` | Holm-adjusted p-value |
| `DM verdict` | `better`, `worse` or `no difference`, from `p_holm` at `alpha` |
| `DM n_days`, `DM n_forecasts` | Days and forecasts of this pair |

In the per-hour tables (`by_hour`, `by_hour_horizon`, and the sheets of the Excel exporter that follow them) `DM p_holm` and `DM verdict` stay empty: a per-hour DM is a heatmap, never a verdict. It called two seeds of one model different 37 % of the time.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `benchmark` | str | Required | Name of the model the others are compared with |
| `variance` | str | `"hac"` | `"hac"` (Newey-West) or `"plain"` (epftoolbox's variance) |
| `lag` | int | `None` | Newey-West lag; `None` picks it from the number of days |
| `correction` | str | `"holm"` | `"holm"`, or `None` for no correction (`DM p_holm` empty, `DM verdict` from `DM p`) |
| `alpha` | float | `0.05` | Two-sided size behind `DM verdict` |
| `interval` | float | `0.95` | Coverage of the interval; `None` skips the bootstrap |
| `block` | int | `7` | Block length of the bootstrap, in days |
| `reps` | int | `4000` | Bootstrap repetitions |
| `seed` | int | `0` | Bootstrap seed: same data and seed, same numbers |

## Basic Usage

```python
from epftoolbox2.evaluators import DMEvaluator

evaluator = DMEvaluator(benchmark="OLS")
```

## In Pipeline

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator, DMEvaluator

pipeline = (
    ModelPipeline()
    .add_model(OLSModel(predictors=predictors, name="OLS"))
    .add_model(LassoCVModel(predictors=predictors, name="Lasso"))
    .add_evaluator(MAEEvaluator(), DMEvaluator(benchmark="OLS"))
)

report = pipeline.run(...)
print(report.summary()[["model", "MAE", "DM ratio", "DM ci_low", "DM ci_high", "DM p_holm", "DM verdict", "DM n_days"]])
#    model      MAE  DM ratio  DM ci_low  DM ci_high  DM p_holm     DM verdict  DM n_days
# 0    OLS  26.0199    1.0000        NaN         NaN        NaN           None        365
# 1  Lasso  24.8100    0.9535     0.9210      0.9858     0.0071         better        365
```

## Notes

- One `DMEvaluator` per pipeline: a second one would write the same columns, which raises `ValueError`. For another benchmark call `report.tests(benchmark=...)`.
- In a group where the benchmark has no forecasts the columns are NaN, with a warning.
- A group is tested only with at least 2 days (and more days than `lag`); below that the p-values are NaN, and the ratio is still given.
- In a group with several horizons (`summary()`, `by_hour()`) the test is `report.tests(horizon="all")`; `by_horizon()` holds the per-horizon tests of `report.tests()`.
- In `compute_all()` (and so the Excel exporter) the test and bootstrap run once per group, which adds up on many models and horizons (with `MCSEvaluator` as well, about 40 seconds for 50 models over a year). Pass `interval=None` or a smaller `reps` if that matters.
- Serializable to YAML and works with `ModelPipeline.save()`/`load()`.
