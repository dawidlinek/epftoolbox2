---
title: RMSEEvaluator
description: Root Mean Squared Error metric
---

# RMSEEvaluator

Calculates Root Mean Squared Error (RMSE) between predictions and actual values. RMSE penalizes larger errors more heavily than MAE.

## Formula

RMSE = √((1/n) × Σ(yᵢ - ŷᵢ)²)

Where:
- yᵢ = actual value
- ŷᵢ = predicted value
- n = number of observations

## Basic Usage

```python
from epftoolbox2.evaluators import RMSEEvaluator

evaluator = RMSEEvaluator()
```

## In Pipeline

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel
from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator
from epftoolbox2.exporters import TerminalExporter

pipeline = (
    ModelPipeline()
    .add_model(OLSModel(predictors=predictors, name="OLS"))
    .add_evaluator(MAEEvaluator())
    .add_evaluator(RMSEEvaluator())
    .add_exporter(TerminalExporter())
)

report = pipeline.run(...)
print(report.summary())
#    model       MAE     RMSE
# 0    OLS  26.0199  32.4512
```
