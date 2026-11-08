---
title: NMAEEvaluator
description: MAE divided by the mean actual, on the same forecasts
---

The normalised MAE: the MAE of a model's forecasts divided by the mean of their actual values. It puts the errors of zones, years
or targets of different size on one scale, which is how generation and load forecasts are usually compared (a solar MAE of
1,000 MW means little without the zone's mean output).

## Formula

nMAE = (1/n) × Σ|ŷᵢ − yᵢ| / ((1/n) × Σ yᵢ)

Where the sums run over the same n forecasts: those with a finite prediction and actual.

## Basic Usage

```python
from epftoolbox2.evaluators import NMAEEvaluator

evaluator = NMAEEvaluator()
```

## In Pipeline

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import RidgeModel
from epftoolbox2.evaluators import MAEEvaluator, NMAEEvaluator

pipeline = (
    ModelPipeline()
    .add_model(RidgeModel(predictors=predictors, name="Ridge"))
    .add_evaluator(MAEEvaluator(), NMAEEvaluator())
)

report = pipeline.run(df, test=("2025-07-01", "2026-06-30"), target="solar")
print(report.summary())  # nMAE is a fraction: 0.101 is 10.1 % of the mean output
```

## Notes

- **On the keys of the MAE.** The mean actual is taken on the forecasts the MAE is taken on, never on every actual of the data, so
  a model that has no forecast for some hours is divided by its own mean. Compare models on identical keys (`forecast_counts()`):
  with different keys both the MAE and the mean differ.
- A fraction, not a percentage. Meant for quantities whose mean is positive; a mean actual of 0 gives inf, and prices with a
  negative mean give a negative number.
- In grouped views it is taken on that group's forecasts, with that group's mean.
- No parameters; serializable to YAML.
