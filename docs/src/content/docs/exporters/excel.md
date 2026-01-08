---
title: ExcelExporter
description: Export results to Excel with formatting
---

# ExcelExporter

Exports evaluation results to Excel files with conditional formatting.

## Basic Usage

```python
from epftoolbox2.exporters import ExcelExporter

exporter = ExcelExporter("results.xlsx")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | Required | Output file path |
| `sheets` | List[str] | All | Sheets to include |

### Sheets Options

- `"summary"` - Overall metrics per model
- `"hour"` - Breakdown by hour
- `"horizon"` - Breakdown by forecast horizon
- `"hour_horizon"` - Hour × Horizon matrix
- `"year"` - Breakdown by year
- `"year_horizon"` - Year × Horizon matrix
- `"details"` - Details of each prediction

## Example

```python
# Include all sheets
exporter = ExcelExporter("results.xlsx")

# Include only specific sheets
exporter = ExcelExporter("results.xlsx", sheets=["summary", "horizon"])
```

## Conditional Formatting

The Excel file includes color-coded metrics in each horizon context separate:
- 🟢 Green: Low error
- 🟡 Yellow: Medium error
- 🔴 Red: High error

## In Pipeline

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import ExcelExporter

pipeline = (
    ModelPipeline()
    .add_model(OLSModel(predictors=predictors, name="OLS"))
    .add_evaluator(MAEEvaluator())
    .add_exporter(ExcelExporter("results.xlsx"))
)

report = pipeline.run(...)
# Results saved to results.xlsx
```
