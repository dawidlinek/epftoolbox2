---
title: TerminalExporter
description: Display results in the terminal
---

Displays evaluation results as formatted tables in the terminal using Rich.

## Basic Usage

```python
from epftoolbox2.exporters import TerminalExporter

exporter = TerminalExporter()
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show` | List[str] | `["summary", "horizon"]` | Tables to display |

### Show Options

- `"summary"` - Overall metrics per model
- `"hour"` - Breakdown by hour
- `"horizon"` - Breakdown by forecast horizon
- `"hour_horizon"` - Hour × Horizon matrix
- `"year"` - Breakdown by year
- `"year_horizon"` - Year × Horizon matrix

## Example

```python
# Show all available tables
exporter = TerminalExporter(show=["summary", "hour", "horizon", "year"])

# Show only summary
exporter = TerminalExporter(show=["summary"])
```

## Output

If the run has anything in [`report.availability()`](/epftoolbox2/model-pipeline/information-set/#what-warn-lists), it comes first,
whatever `show` says: inputs not known at the issue time, or without a rule, with their periods and horizons. The rule itself is left
out for width. With nothing to list this table isn't printed.

```
Availability: inputs not known to be out at the issue time
┏━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
┃ model ┃ predictor   ┃ status ┃ periods ┃ horizons ┃ known     ┃ issued  ┃
┡━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
│ OLS   │ load_actual │ late   │ 10-23   │ 1-7      │ D+1 01:00 │ D 11:45 │
└───────┴─────────────┴────────┴─────────┴──────────┴───────────┴─────────┘

Summary
┏━━━━━━━━━━┳━━━━━━━━━┓
┃ model    ┃ MAE     ┃
┡━━━━━━━━━━╇━━━━━━━━━┩
│ OLS      │ 26.0199 │
│ LassoCV  │ 28.1098 │
└──────────┴─────────┘

By Horizon
┏━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ model    ┃ horizon  ┃ MAE     ┃
┡━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ OLS      │ 1        │ 18.5432 │
│ OLS      │ 2        │ 22.1234 │
└──────────┴──────────┴─────────┘
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
    .add_exporter(TerminalExporter(show=["summary", "horizon"]))
)
```
