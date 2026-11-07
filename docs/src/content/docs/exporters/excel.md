---
title: ExcelExporter
description: Export results to Excel with formatting
---

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
- `"run_weekday_horizon"` - Issue day's weekday × Horizon matrix
- `"target_weekday_horizon"` - Target day's weekday × Horizon matrix
- `"details"` - Details of each prediction

A model with no forecasts in the report's window has a row with 0 forecasts in `summary` and an empty `prediction_{model}` column in `details`; it has no row in the breakdowns, which hold only forecasts.

An evaluator with several columns, such as `DMEvaluator`, gets one sheet per column in the breakdowns (`horizon_DM ratio`, `horizon_DM p`, ...). Empty cells are values that do not exist for that group, such as `DM p_holm` in a per-hour sheet.

Each breakdown also gets a sheet of counts after its metric sheets (`horizon_forecasts`, `HourHorizon_forecasts`, `RunWeekdayHorizon_forecasts`, ...): the number of
forecasts of each model in each group, laid out like the metric sheets. A model with no forecast in a group has 0 there, and in the sheets with a block for
each model every block has the groups of the whole table, where the metric sheets leave an empty cell or no cell at all. The sheets of counts have no
colours, since a large count is not a bad value. Compare them with a metric sheet before reading differences between models: models with different counts
in a group are scored on different forecasts there ([Forecasts per group](/epftoolbox2/model-pipeline/results/#forecasts-per-group)). The `Summary` sheet has
the model's total in its `forecasts` column, which its colour scale leaves out.

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
