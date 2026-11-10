---
title: Fallbacks
description: A ladder of forecasts, each target day taken whole from the first member that has all of it, scored per rung on identical keys, and top= choices frozen for live use
---

A live forecast needs a plan for the morning an input is late. `FallbackEnsemble` is that plan written down: a ladder of members,
best first, and each target day taken from the first member that has a forecast of all of it. It is an ensemble like
[`AverageEnsemble`](/epftoolbox2/model-pipeline/ensembles/): it fits nothing, it is computed from what its members stored, the same
way in a backtest and live, and from then on it is one more model in every table. So the backtest says how often each rung would
have served and how well, before a live day ever depends on it.

```python
from epftoolbox2.ensembles import AverageEnsemble, FallbackEnsemble
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.models import NaiveModel, OLSModel
from epftoolbox2.pipelines import ModelPipeline

full = ["load_forecast_d+{horizon}", "price_exaa_d+{horizon}", "price_d-1", "is_sunday_d+{horizon}"]
without_exaa = ["load_forecast_d+{horizon}", "price_d-1", "is_sunday_d+{horizon}"]

pipeline = (
    ModelPipeline(issue_time="11:45", issue_timezone="Europe/Berlin")
    .add_model(OLSModel(predictors=full, training_window=364, name="OLS_364"))
    .add_model(OLSModel(predictors=full, training_window=728, name="OLS_728"))
    .add_model(OLSModel(predictors=without_exaa, training_window=728, name="OLS_no_exaa"))
    .add_model(NaiveModel(predictors=["price_d-7"], known_only=True, name="naive"))
    .add_ensemble(AverageEnsemble(members=["OLS_364", "OLS_728"], top=1, name="best"))
    .add_ensemble(FallbackEnsemble(["best", "OLS_no_exaa", "naive"], per="day", name="live"))
    .add_evaluator(MAEEvaluator())
)
report = pipeline.run(df, validation="previous_year", test=("2024-01-05", "2025-12-26"), horizon=1)
print(report.rungs("live"))       # per rung: the days and forecasts it served, its MAE there, rung 0's on the same keys
print(report.forecast_counts())   # a day no rung serves is missing: member_missing
```

| Parameter | Default | What it does |
|---|---|---|
| `members` | required | Names of the models or ensembles of the pipeline, best first: the order is the ladder, rung 0 first. A `top` ensemble may be one of them |
| `per` | `"day"` | `"day"` takes each target day whole from one member; `"key"` takes each forecast key from the first member that has it. Keyword only |
| `name` | required | Names the ensemble in every table and its results file. Keyword only |

## Which member serves

With `per="day"`, for each issue day, horizon and target day, **the first member whose forecasts of every planned key of that day
exist with a finite prediction** serves all of them. So a delivered day is never stitched from two models: a day's curve is one
model's curve, whatever went wrong with the others.

- **Planned keys are the clock's.** A key is planned when its period exists on the issue day and on the target day, as for every
  [planned count](/epftoolbox2/model-pipeline/time-and-calendars/#what-is-planned): a spring day needs 23 hours (92 quarter-hours),
  an autumn day 24 (96), the repeated hour being one period. A key no member has still counts, so it leaves its day to no one.
- **A skip record is no forecast.** A member with `missing_test_predictors` on one hour of a day, or a `check_failed` record from
  checks before export, doesn't have that day, and the ladder moves on. A NaN or infinite prediction counts as missing too.
- **A day no member serves is missing.** It is counted in `forecast_counts()` under `member_missing`, as for an average, and the
  fallback never fills it in from parts of several members.

With `per="key"` each key comes from the first member that has it, so one day can mix rungs. That keeps more forecasts in a backtest
and fits a product that is delivered key by key; for a day-ahead curve keep `per="day"`.

The records are the member's forecasts, prediction and actual, with two more keys: `rung` (0-based) and `member` (its name). Their
`status` is `"ok"`, so every table, comparison, exporter and `ForecastSource` reads them as forecasts. Members are checked as an
average's are (one target, horizon, frequency, zone, information set and data), and the file is recomputed every time, never resumed.

A fallback chooses nothing on validation, so it has forecasts in every window on file and is in `report.validation` like a model,
unless a member has test forecasts only (a `top` ensemble, or an average over one): then it has test forecasts only, as a plain
average over one has, or a lower rung would serve every validation day the top one lacks. It may be a member of a plain average or of
another fallback, but not of a `top` ensemble.

## Scoring the rungs

```python
rungs = report.rungs("live")
```

For example (the numbers are made up, the counts add up: 722 test days of 24 hours, all served):

```text
 rung       member  days  forecasts   mae  common_forecasts  common_mae  rung0_mae  member_forecasts  member_mae
    0         best   706      16944  9.12             16944        9.12       9.12             17256        9.14
    1  OLS_no_exaa    14        336 11.80               312       11.75       9.60             17328       10.41
    2        naive     2         48 23.10                 0         NaN        NaN             17328       24.05
```

One row per member, in rung order, rungs that served nothing included, in the report's window:

| Column | What it is |
|---|---|
| `rung`, `member` | Its place on the ladder and its name |
| `days` | The (issue day, horizon, target day) it served; with `per="day"` all of each |
| `forecasts` | The keys it served |
| `mae` | Its MAE on those keys with an actual |
| `common_forecasts` | Of those, the keys where the rung-0 member has a forecast too, both with an actual |
| `common_mae`, `rung0_mae` | This rung's and the rung-0 member's MAE on exactly the `common_forecasts` keys |
| `member_forecasts`, `member_mae` | The member's own forecasts in the window, all of them, and its MAE on those with an actual |

**Compare a rung with rung 0 on `common_mae` against `rung0_mae`**, which are on identical keys with their count. `mae` is on the
days the rung served, the days rung 0 failed, which are often harder (a late input, a holiday), and `member_mae` on all of the
member's own days: neither is a comparison with rung 0. With `per="day"` rung 0 often has most of a day it lost by one hour, so
`common_forecasts` is close to `forecasts`; with `per="key"` a rung above 0 serves only keys rung 0 lacks, and has none in common.
`df.attrs["missing"]` is how many planned forecasts no rung served. Every MAE is an exact sum, the same whatever order the records
are in. The members must be in the report, as they are in what `run` and `ModelPipeline.report` return; `rungs()` of a report made
by `aggregate()`, or of anything but a fallback, raises.

## Freezing choices for live use

A `top=` ensemble chooses per test period on the validation period before it, and a live pipeline has no test period to choose for.
`ModelPipeline.frozen(save_dir)` gives the pipeline for live use: every `top=` ensemble replaced by a plain `AverageEnsemble` of the
members it chose for the newest segment of the study in `save_dir`, under the same name and in the same place, everything else kept.

```python
live = pipeline.frozen("results/de_price")
print(live.ensembles[0].members, live.frozen_choices["best"])
live.save("live_pipeline.yaml")
```

- **No look-ahead.** The choices are made again from the members' stored results, as `report(save_dir)` makes them: on validation
  forecasts alone, each segment on the 12 months before its test period with `validation="previous_year"`. A stale ensemble file
  can't decide them. It is a report of the `top` ensembles and their members, so it writes their files again, runs no evaluator or
  exporter, and raises as a report would (a member without results, results made without validation).
- **The newest segment** is the last of `report.windows.segments`, or the one validation period of a study without a test.
- **Bit for bit.** The frozen average's forecasts equal the `top` ensemble's on that segment's test period, since a `top=k` ensemble
  is `AverageEnsemble(chosen)` there.
- **What was frozen** is in `frozen_choices`, by ensemble: the original `members` and `top`, `chosen`, `segment`, its `validation` and
  `test` periods, `validation_mae`, `validation_forecasts` and `save_dir`. It is not saved to YAML, where the frozen ensemble is a
  plain average. A pipeline without a `top` ensemble comes back as a copy with an empty `frozen_choices`.
- **Zones.** Each zone of a run over zones chose on its own, so `frozen()` of such a `save_dir` raises: freeze each zone's folder.

A frozen pipeline reported into the backtest's own `save_dir` writes its plain average over the `top` ensemble's file, which the
backtest pipeline's next report writes back: keep live results in a directory of their own.

## YAML

```yaml
ensembles:
- class: AverageEnsemble
  params:
    members: [OLS_364, OLS_728]
    top: 1
    by: validation
    name: best
- class: FallbackEnsemble
  params:
    members: [best, OLS_no_exaa, naive]
    name: live
    per: day
```
