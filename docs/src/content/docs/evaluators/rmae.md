---
title: rMAEEvaluator
description: Relative Mean Absolute Error metric
---

# rMAEEvaluator

Calculates relative Mean Absolute Error (rMAE) — the ratio of a model's MAE to a base model's MAE. This is a standard metric in the electricity price forecasting literature (Lago et al., 2021) for benchmarking model performance.

## Formula

rMAE = MAE(model) / MAE(base_model)

Where both MAEs are computed on the same data slice (same time period, same grouping).

**Interpretation:**
- rMAE < 1 — model **outperforms** the base model
- rMAE = 1 — model performs **equally** to the base model
- rMAE > 1 — model **underperforms** the base model

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_model` | str | Required | Name of the model to use as the benchmark |

## Basic Usage

```python
from epftoolbox2.evaluators import rMAEEvaluator

evaluator = rMAEEvaluator(base_model="OLS")
```

## In Pipeline

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator, rMAEEvaluator
from epftoolbox2.exporters import TerminalExporter

pipeline = (
    ModelPipeline()
    .add_model(OLSModel(predictors=predictors, name="OLS"))
    .add_model(LassoCVModel(predictors=predictors, cv=7, name="LassoCV"))
    .add_evaluator(MAEEvaluator())
    .add_evaluator(rMAEEvaluator(base_model="OLS"))
    .add_exporter(TerminalExporter())
)

report = pipeline.run(...)
print(report.summary())
#      model       MAE   rMAE
# 0      OLS  26.0199  1.0000
# 1  LassoCV  24.8100  0.9535
```

The rMAE is computed per data slice, so grouped views (`by_hour`, `by_horizon`, etc.) each show the relative performance for that specific group.

## Notes

- The `base_model` name must match the `name` parameter of one of the models in the pipeline.
- If the base model's MAE is zero for a given group, rMAE returns `inf`.
- rMAE is serializable to YAML and works with `ModelPipeline.save()`/`load()`.
