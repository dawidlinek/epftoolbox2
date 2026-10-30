---
title: BiasEvaluator
description: Mean error of the forecasts
---

Calculates the mean error of a model's forecasts: positive when it forecasts too high on average. A model can have a low MAE and still be biased, and the bias ranks models quite differently from the MAE, so it is worth a column next to it.

## Formula

Bias = (1/n) × Σ(ŷᵢ - yᵢ)

Where:
- ŷᵢ = predicted value
- yᵢ = actual value
- n = number of forecasts

## Basic Usage

```python
from epftoolbox2.evaluators import BiasEvaluator

evaluator = BiasEvaluator()
```

## In Pipeline

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel
from epftoolbox2.evaluators import BiasEvaluator, MAEEvaluator

pipeline = (
    ModelPipeline()
    .add_model(OLSModel(predictors=predictors, name="OLS"))
    .add_evaluator(MAEEvaluator(), BiasEvaluator())
)

report = pipeline.run(...)
print(report.summary())
#   model  forecasts      MAE    Bias
# 0   OLS       5038  26.0199  1.2041
```

## Notes

- Like the MAE, it is taken on each model's own forecasts, and in grouped views on that group's forecasts.
- No parameters; serializable to YAML.
