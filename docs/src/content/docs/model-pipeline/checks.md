---
title: Checks Before Export
description: ForecastChecks on each model's forecasts of a target day (periods by the clock, finite, range, not constant, max step), check_failed skip records that move a fallback ladder, applied in backtests too when set
---

A delivered curve can be wrong in ways no table shows until it is too late: a period missing, a NaN, a forecast of 9,000 EUR/MWh,
a whole day stuck at one value, a jump of 800 between two hours. `ForecastChecks` are the tests a model's forecasts of one target
day must pass before anything uses them. A day that fails has none of its forecasts kept, so a
[`FallbackEnsemble`](/epftoolbox2/model-pipeline/fallback/) serves that day from its next rung, an average counts it as missing,
and every table counts it as skipped.

```python
from epftoolbox2.ensembles import FallbackEnsemble
from epftoolbox2.models import NaiveModel, OLSModel
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results import ForecastChecks

full = ["load_forecast_d+{horizon}", "price_exaa_d+{horizon}", "price_d-1", "is_sunday_d+{horizon}"]

pipeline = (
    ModelPipeline(
        issue_time="11:45",
        issue_timezone="Europe/Berlin",
        checks=ForecastChecks(range=(-500, 4000), max_step=300),
    )
    .add_model(OLSModel(predictors=full, training_window=364, name="OLS"))
    .add_model(NaiveModel(predictors=["price_d-7"], name="naive"))
    .add_ensemble(FallbackEnsemble(["OLS", "naive"], name="live"))
)
report = pipeline.run(df, test=("2024-01-05", "2025-12-26"), horizon=1)
print(report.forecast_counts())   # check_failed: the forecasts of days that failed a check
print(report.rungs("live"))       # the days rung 1 served, those OLS failed included
```

| Parameter | Default | What it checks of each target day of a model |
|---|---|---|
| `periods` | `True` | Every period the clock gives the day has a forecast (see below) |
| `finite` | `True` | No prediction is NaN, infinite or missing |
| `range` | `None` | `(low, high)`: every finite prediction within it, both included; either bound may be `None` |
| `not_constant` | `True` | The finite predictions of a day of two or more periods aren't all one value |
| `max_step` | `None` | No step between neighbouring periods of the day larger than this, in the target's units |

All keyword only. **Without checks (`checks=None`, the default) nothing is checked and nothing changes:** every record, manifest,
table and YAML file is what it was, bit for bit. `ModelPipeline(checks=ForecastChecks())` turns on the three checks that need no
number (`periods`, `finite`, `not_constant`).

## What a day is

A check is of one model's forecasts of one (issue day, horizon), so of one target day: every period of that day at once. The checks
read the predictions and the clock alone, never an actual, so no check can let an outcome decide which forecasts are kept.

- **The periods are the clock's.** A day needs the periods it is planned with
  ([what is planned](/epftoolbox2/model-pipeline/time-and-calendars/#what-is-planned)): those its target day has, less one its issue
  day lacks. A spring day has 23 hours (92 quarter-hours), an autumn day 25 hours (100 quarter-hours) in 24 periods (96), the
  repeated hour being one period that covers both its occurrences, an ordinary day 24 (96), and a `"1D"` day one. A forecast issued
  on a spring switch day has no row to issue 02:00 from, so its target days need 23 hours: no failure, but no model delivers that
  hour either.
- **Neighbours are in clock order:** 01:00 and 03:00 on a spring day, a day's last period and the next day's first never.
- **A key skipped for another reason** (`missing_test_predictors`, `not_published`, ...) keeps its own skip record, and with
  `periods` makes its day fail: the other periods of that day aren't kept either.

## What a failing day becomes

Each forecast of the day becomes a **skip record** with status `check_failed`, its `prediction` null, its `actual` kept and the
failing checks in `detail`, such as `range: 1 outside [-500, 4000] (19 at 5123.4)` or `periods: 23 of 24, missing 5; not_constant: all
23 at 51.2`. So:

- `report.forecast_counts()` has a `check_failed` column, the manifest's `skipped` counts them per window, and a run warns about the
  shortfall as for any skip (and raises if a model kept no forecast at all).
- A `FallbackEnsemble` moves past the day to its next rung, as it does past any skip record; `report.rungs()` counts what the lower
  rungs served there. An `AverageEnsemble` counts the day's keys as `member_missing`.
- Ensembles aren't checked themselves: they are computed from their members' checked forecasts.

## In backtests

Checks set on a pipeline apply to every run of it, backtests included, so a rung's backtest score counts the days it would have
failed live. A run checks each day once every task of it has come back from the pool (an engine group sends a day back period by
period) and writes nothing of a failing day but its skip records.

- **Resuming.** The manifest records the checks, and a resume with other checks, or none, raises `ManifestMismatchError`
  (`allow_manifest_mismatch=True` overrides it, and leaves the records on file as they were). A failed day is tried again by every
  run, as every skipped key is, and counted again in that run's skips. A day a run completes (some of its keys were on file, from a
  run that crashed or skipped them) is checked with the forecasts on file; when it fails, those are overruled by a `check_failed`
  skip record one revision above them, which the readers take.
- **What checks don't do.** They don't look at inputs: the data validators stay advisory and turn no value into a missing one.
  `verify()` runs without them, and a `ResidualModel`'s base isn't checked, only what the residual model forecasts.

## Saving

Checks are saved with the pipeline, only when set, and a `Workflow` can set them over the pipeline's file:

```yaml
checks:
  class: ForecastChecks
  params: {periods: true, finite: true, range: [-500.0, 4000.0], not_constant: true, max_step: 300.0}
```

In a `Workflow` file the same key goes under `model_pipeline:`, and `checks: null` there turns a file's checks off.
