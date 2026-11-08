---
title: PooledModel
description: One model fitted over the zones of a run of several markets, each zone's rows standardised within its own fit
---

A model fitted once over several zones: the forecast of each zone is made from one fit on every zone's training rows. Markets of very
different size share it because each zone's rows are standardised within that zone's own fit, so none dominates. It runs only in a
[run over several zones](/epftoolbox2/model-pipeline/zones/), `ModelPipeline.run(data={zone: frame})`, and writes each zone's
forecasts into that zone's folder of `save_dir`, as that zone's own model would.

## Basic Usage

```python
from epftoolbox2.models import LassoCVModel, PooledModel
from epftoolbox2.pipelines import ModelPipeline

lasso = LassoCVModel(predictors=["load_forecast_daily_max_d+{horizon}", "is_holiday_d+{horizon}"], training_window=1092)
pipeline = ModelPipeline(issue_time="12:00", issue_timezone="data").add_model(
    PooledModel(lasso, over=("zone", "horizon"), name="pooled_lasso")
)
panel = pipeline.run(
    {"PL": df_pl, "DE": df_de, "FI": df_fi}, target="actual_daily_max", freq="1D", horizons=range(3, 10),
    issue_days=["friday"], test=("2024-01-05", "2025-12-26"), save_dir="runs/pooled",
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | model | Required | The model fitted: `OLSModel`, `LassoCVModel`, `RidgeModel` or `LEARModel` (a model with `_fit_model` and `_predict_fitted`), with its predictors, training window, scaler and `refit` |
| `over` | tuple | `("zone",)` | Keyword-only. The axes one fit pools: `("zone",)`, one fit per period, issue day and horizon over every zone; `("zone", "horizon")`, one fit per period and issue day over every zone and horizon |
| `zone_feature` | str or None | `"onehot"` | Keyword-only. What tells the zones apart in a fit: `"onehot"`, a 0/1 column per zone; `"code"`, one column of the zone's place in the run's order (for a tree learner that reads it as a category); None, nothing |
| `missing` | str | `"raise"` | Keyword-only. A predictor a zone's frame lacks: `"raise"` refuses the run before any fit; `"zero"` gives it 0 in that zone, its standardised mean |
| `name` | str | the model's name | Keyword-only. The name of the pooled model's results in each zone |

## What one fit is

A **unit** is (period, issue day, horizon), or with `refit` (period, refit anchor, horizon), and without the horizon when `over`
holds `"horizon"`. Its rows are the union of what each zone's own model would train on there:

1. **Each zone's rows** are its model's rows: those dated in `training_span(D, W, h)` in that zone's own local dates (a zone's
   frame can start on another date than the others'), filtered as the model filters them (a missing predictor or target drops a row).
2. **Each zone's rows are scaled by a scaler of the model's kind fitted on them alone**, the target too, with that zone's own
   binary-column mask. No scale is fitted over the stack: a whole-sample scale would reach across zones and dates.
3. **Stacked** zone by zone in the run's order, horizon by horizon, row by row in date order, with the zone feature (and with
   `"horizon"` in `over` a one-hot horizon), and fitted once with the model's own estimator.
4. **Each zone's forecast** is its issue row, scaled by its own zone's scaler, forecast on its own from the fit and inverted by that
   scaler. Every zone's rows train every unit, whether or not the zone has a forecast left to make there, so a resume equals a fresh run.

With one zone (and one horizon) nothing is added: a pooled model over one zone is the model itself, to the bit. With `"horizon"` in
`over` the predictors of each horizon are stacked by their place in the list (`load_d+3` under `load_d+4`), so every horizon's list
must have the same length, or the run raises before any fit. The coefficients of a record are the fit's, the zone and horizon columns
last.

## Issue time

One fit serves every zone, so it is made by the earliest issue of its day over the zones. With
`issue_timezone="data"` every zone issues on its own clock, and each zone's training rows are judged as of the earliest of them
before any fit: refused at `availability="raise"`, listed at `"warn"` with a note naming the zone that issues earlier. A zone's issue
row is judged on its own clock. See [Several Zones](/epftoolbox2/model-pipeline/zones/#the-issue-time-of-each-zone).

## Results and resuming

Each zone's folder holds `<name>.jsonl` and its manifest, with the records of that zone's forecasts under the usual keys. The manifest
records `pooled`: the zones in their order (the order of the stack), `over`, `zone_feature`, `missing` and the other zones' data
hashes. A resume compares it, so another set or order of zones, another setting or another zone's data refuses with
`ManifestMismatchError`.

## Not in this step

A predictor function of `ctx`, a list of training windows, `newest_target="published"` (each zone issues on its own clock and one
fit serves them all), `NaiveModel` (it fits nothing), validation windows, and `Workflow` or SLURM array jobs over zones. A model with
`periods="pooled"` (one model over the periods of a day) inside a `PooledModel` comes with the step-6 learners.
