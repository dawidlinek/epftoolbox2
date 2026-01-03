---
title: Model Pipeline Examples
description: Complete ModelPipeline usage examples
---

# Model Pipeline Examples

## Basic Example

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import TerminalExporter

predictors = [
    "load_actual",
    "weekday",
    "hour",
    "is_holiday",
]

pipeline = (
    ModelPipeline()
    .add_model(OLSModel(predictors=predictors, training_window=365, name="OLS"))
    .add_evaluator(MAEEvaluator())
    .add_exporter(TerminalExporter())
)

report = pipeline.run(
    data=df,
    test_start="2024-02-01",
    test_end="2024-03-01",
    target="price",
    horizon=7,
)
```

## Comparing Multiple Models

```python
from epftoolbox2.models import OLSModel, LassoCVModel

pipeline = (
    ModelPipeline()
    .add_model(OLSModel(predictors=predictors, name="OLS"))
    .add_model(LassoCVModel(predictors=predictors, cv=5, name="LassoCV"))
    .add_evaluator(MAEEvaluator())
    .add_exporter(TerminalExporter())
    .add_exporter(ExcelExporter("comparison.xlsx"))
)
```

## With Many Predictors

```python
predictors = [
    "load_actual",
    "weekday",
    "hour",
    "is_holiday",
    # Hourly lags (1 week = 168 hours)
    *[f"load_actual_h-{i}" for i in range(1, 169)],
    # Daily price lags
    *[f"price_d-{i}" for i in range(1, 8)],
    # Weather forecasts
    *[f"warsaw_temperature_2m_d+{h}" for h in range(1, 8)],
]

# LassoCV is recommended for many predictors
model = LassoCVModel(predictors=predictors, cv=5, name="Lasso_Full")
```
