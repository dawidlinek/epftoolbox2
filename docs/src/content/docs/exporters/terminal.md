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

After it come the inputs read from an archive of revised values, the rows of
[`report.vintages()`](/epftoolbox2/model-pipeline/results/#vintage-classes-of-the-inputs) whose class is `latest_revision` or
`last_write_stamp`, also whatever `show` says, and only if there are any: a backtest on them is more accurate than a live run would
have been.

```
Availability: inputs not known to be out at the issue time
┏━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
┃ model ┃ predictor   ┃ status ┃ periods ┃ horizons ┃ known     ┃ issued  ┃
┡━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
│ OLS   │ load_actual │ late   │ 10-23   │ 1-7      │ D+1 01:00 │ D 11:45 │
└───────┴─────────────┴────────┴─────────┴──────────┴───────────┴─────────┘

Summary
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
┃ model    ┃ forecasts ┃ MAE     ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
│ OLS      │ 5038      │ 26.0199 │
│ LassoCV  │ 5038      │ 28.1098 │
└──────────┴───────────┴─────────┘

By Horizon
┏━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
┃ model    ┃ horizon  ┃ forecasts ┃ MAE     ┃
┡━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
│ OLS      │ 1        │ 720       │ 18.5432 │
│ OLS      │ 2        │ 720       │ 22.1234 │
└──────────┴──────────┴───────────┴─────────┘
```

Every table has a `forecasts` column: the number of forecasts of the model in that row's group (after the group keys), so a model that
has fewer in a group than another is visible in the table. When the models of a table differ in their counts the exporter also emits the
`UserWarning` of the report: see [Forecasts per group](/epftoolbox2/model-pipeline/results/#forecasts-per-group).

## Model Provenance

A report with a model that has a card (see [Licences and Weights](/epftoolbox2/models/licences-and-weights/)) prints a "Model
provenance" table before the metric tables, whatever `show` lists: each such model's licence, attribution, weights, runtime and
contamination verdicts. Without one nothing more is printed.

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
