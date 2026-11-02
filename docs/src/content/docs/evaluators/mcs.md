---
title: MCSEvaluator
description: Model confidence set of the models in a table
---

Adds the model confidence set (Hansen, Lunde & Nason 2011) to the report's tables: the set of models the data cannot tell from the best one. Models are eliminated one at a time, starting from the worst, until the rest cannot be told apart. A model's MCS p-value is the smallest size at which it is out of the set.

[`report.mcs()`](/epftoolbox2/model-pipeline/comparing-models/#model-confidence-set) gives the same result as a table of its own, with the 90 % and 75 % sets and the models that limited the sample.

## What it computes

- **Keys:** here the set of models is the comparison, so every model is on the forecasts **all** the models of the group hold. One model with few forecasts shrinks the sample for all of them: read `MCS n_days`, or use `report.mcs()`, which names it and takes a subset.
- **Loss:** the daily mean absolute error, with the daily losses of several horizons averaged per run date.
- **Test:** the T_R statistic and a stationary bootstrap with mean blocks of 7 days, with the definitions of `arch.bootstrap.MCS`. T_max is available, but in the study behind these defaults it changed with the seed and kept models up to 17 % worse than the best.

## Output columns

| Column | Meaning |
|--------|---------|
| `MCS p` | MCS p-value of the model |
| `in MCS` | Whether `MCS p` is above `alpha` |
| `MCS n_days`, `MCS n_forecasts` | Days and forecasts all the models of the group share |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | `0.10` | Size of the set (0.10 gives the 90 % set) |
| `statistic` | str | `"R"` | `"R"` (T_R) or `"max"` (T_max) |
| `block` | int | `7` | Mean block length of the stationary bootstrap, in days |
| `reps` | int | `1000` | Bootstrap repetitions (at least 1,000 recommended) |
| `seed` | int | `0` | Bootstrap seed: same data and seed, same numbers |

## Basic Usage

```python
from epftoolbox2.evaluators import MCSEvaluator

evaluator = MCSEvaluator(alpha=0.10)
```

## In Pipeline

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator, MCSEvaluator

pipeline = (
    ModelPipeline()
    .add_model(OLSModel(predictors=predictors, name="OLS"))
    .add_model(LassoCVModel(predictors=predictors, name="Lasso"))
    .add_model(LassoCVModel(predictors=predictors, cv=3, name="Lasso3"))
    .add_evaluator(MAEEvaluator(), MCSEvaluator())
)

report = pipeline.run(...)
print(report.summary()[["model", "MAE", "MCS p", "in MCS", "MCS n_days"]])
#     model      MAE  MCS p  in MCS  MCS n_days
# 0     OLS  26.0199  0.012   False         365
# 1   Lasso  24.8100  1.000    True         365
# 2  Lasso3  24.8702  0.640    True         365
```

## Notes

- Needs at least 3 models, counting exact clones once. With fewer, or fewer than 2 common days, the columns are NaN, with a warning.
- [Exact clones](/epftoolbox2/model-pipeline/comparing-models/#exact-clones), models with the same predictions, are one model to the set, and each gets its result. Other models with the same losses on every day have no valid test: in a table the columns are then NaN with a warning that names the models, and `report.mcs()` raises.
- In the per-hour tables `MCS p` is a heatmap, like the per-hour DM, and `in MCS` stays empty. It stays empty too below 30 days.
- `summary()` warns below a span of 182 calendar days, from the first run date to the last: `n_days` when every day has a forecast (see [Test length](/epftoolbox2/model-pipeline/comparing-models/#test-length)). Results at different frequencies raise `ValueError`.
- Serializable to YAML and works with `ModelPipeline.save()`/`load()`.
