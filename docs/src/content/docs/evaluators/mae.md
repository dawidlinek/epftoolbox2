---
title: MAEEvaluator
description: Mean Absolute Error metric
---

# MAEEvaluator

Calculates Mean Absolute Error (MAE) between predictions and actual values.

## Formula

MAE = (1/n) × Σ|yᵢ - ŷᵢ|

Where:
- yᵢ = actual value
- ŷᵢ = predicted value
- n = number of observations

## Basic Usage

```python
from epftoolbox2.evaluators import MAEEvaluator

evaluator = MAEEvaluator()
```

## In Pipeline

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import TerminalExporter

pipeline = (
    ModelPipeline()
    .add_model(OLSModel(predictors=predictors, name="OLS"))
    .add_evaluator(MAEEvaluator())
    .add_exporter(TerminalExporter())
)

report = pipeline.run(...)
print(report.summary())
#    model       MAE
# 0    OLS  26.0199
```

## Creating Custom Evaluators

See [Extending](/epftoolbox2/reference/extending/) for how to create custom evaluators like RMSE or MAPE.
