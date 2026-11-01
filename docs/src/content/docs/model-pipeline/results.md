---
title: Working with Results
description: Using EvaluationReport to analyze model performance
---

The `ModelPipeline.run()` method returns an `EvaluationReport` object with various analysis methods.

## EvaluationReport Methods

```python
report = pipeline.run(...)

# Overall summary per model (with the number of forecasts each metric is based on)
summary = report.summary()

# Planned vs produced forecasts per model, and why tasks gave no forecast
counts = report.forecast_counts()

# What each ensemble chose on validation, per test period served (empty without any that chooses)
choices = report.choices()

# Breakdown by hour of day
by_hour = report.by_hour()

# Breakdown by forecast horizon
by_horizon = report.by_horizon()

# Combined hour × horizon
by_hour_horizon = report.by_hour_horizon()

# Breakdown by year
by_year = report.by_year()

# Combined year × horizon
by_year_horizon = report.by_year_horizon()

# Combined issue-day weekday × horizon, and target-day weekday × horizon
by_run_weekday_horizon = report.by_run_weekday_horizon()
by_target_weekday_horizon = report.by_target_weekday_horizon()

# Calendar quarter of the target date (with the year), target date, and price level of the actual price
by_quarter = report.by_quarter()
by_day = report.by_day()
by_price_level = report.by_price_level()

# Daily maximum, minimum, mean or spread of the forecasts, as a report of its own
daily_max = report.aggregate("1D", agg="max")

# The models of two reports (say, of two runs) in one report
joined = report.join(other_report)

# Iterate (model_name, DataFrame of per-forecast run_date/target_date/hour/horizon/day_in_test/actual/prediction)
for model_name, details in report.iter_details():
    ...

# summary() plus the by_*() breakdowns up to the weekday ones, computed in one pass over the results (what the
# exporters use; the quarter, day and price-level tables are not among them)
sheets = report.compute_all()  # {"summary": ..., "hour": ..., ..., "target_weekday_horizon": ...}

# Is a difference real? Each model against a benchmark, the confidence set, rank uncertainty, a pair, an overview
tests = report.tests(benchmark="OLS")
confidence_set = report.mcs()
ranks = report.ranks()
pair = report.compare("Lasso", "OLS")
board = report.leaderboard(benchmark="OLS")
```

The last five are explained on [Comparing Models](/epftoolbox2/model-pipeline/comparing-models/), the quarter, day and price-level tables, `aggregate()` and `join()` below.

An evaluator may write several columns (`DM ratio`, `DM p`, ...). They appear in `summary()` and every `by_*()` table like any other column, and the exporters write them.

## Example Output

```python
print(report.summary())
#    model  forecasts      MAE
# 0    OLS      5038  26.0199
# 1  Lasso      5038  28.1098

print(report.forecast_counts())
#    model  planned  produced  missing  missing_test_predictors
# 0    OLS     5040      5038        2                        2
# 1  Lasso     5040      5038        2                        2

print(report.by_horizon())
#    model  horizon      MAE
# 0    OLS        1  18.5432
# 1    OLS        2  22.1234
# 2    OLS        7  35.4567
```

## Forecast counts

A task (period × horizon × run day) that cannot give a forecast is skipped and counted, never dropped silently. `report.forecast_counts()` has one row per model with `planned`, `produced`, `missing` and one column per skip reason:

| Reason | Meaning |
|---|---|
| `missing_test_predictors` | A predictor is NaN on the issue day's row |
| `missing_test_row` | The issue day has no row for that period (a gap in the data) |
| `member_missing` | An [ensemble](/epftoolbox2/model-pipeline/ensembles/) only: a member it averages has no finite prediction for that key |
| `too_few_rows` | The training window has no rows before the issue day |
| `no_matching_days` | A [`train_on="matching"` list](/epftoolbox2/model-pipeline/predictor-functions/#training-only-on-the-days-with-the-same-list): no past day in the window got this list |
| `no_finite_training_rows` | Every training row has a NaN predictor or target |
| `fit_error` | The model raised `ValueError` (scikit-learn does this on degenerate input) |

A shortfall is logged as a warning when the model runs, and a model that produces nothing raises. Any other exception inside a task stops the run (results already written to `save_dir` are kept, so a rerun resumes).

Models are only comparable on the forecasts they all produced: check that `produced` agrees before comparing metrics.

## Inputs not known at the issue time

`report.availability()` lists what a run with an issue time could not vouch for: every predictor whose rule puts it after the
issue time and every column without a rule, per model, with its periods and horizons. It is empty, with the same columns, when
there is nothing to list. See [Information Set](/epftoolbox2/model-pipeline/information-set/#what-warn-lists).

```python
report.availability()  # model, predictor, role, status, list, periods, horizons, rule, known, issued, note
```

## Breakdowns by quarter, day and price level

```python
by_quarter = report.by_quarter()
by_day = report.by_day()
by_price_level = report.by_price_level()
by_price_level = report.by_price_level(spike=250)  # your own spike threshold, in EUR/MWh

print(report.by_quarter())
#    model  year  quarter  forecasts       MAE      rMAE
# 0    OLS  2023        1       2136  6.3279  1.0000
# 1  Lasso  2023        1       2126  4.7142  0.7458
# 2    OLS  2023        2       2184  6.2534  1.0000

print(report.by_price_level())
#    model price_level  forecasts      MAE      rMAE
# 0    OLS    negative         25   5.3657  1.0000
# 1  Lasso    negative         25   4.8619  0.9061
# 2    OLS      normal       8647   6.2718  1.0000
# 3  Lasso      normal       8606   4.8062  0.7671
# 4    OLS       spike         88   6.0166  1.0000
# 5  Lasso       spike         88   4.8244  0.8018
```

They are cut like `by_year()`: each model's forecasts are split into groups, and every evaluator runs on a group alone. A relative metric (`rMAE`) or a test (`DM ...`, `MCS ...`) therefore compares the models on the forecasts they share inside the group, and prints its own `n_days` and `n_forecasts`. These tables also have a `forecasts` column, the model's own number of forecasts in the group: in the example Lasso lacks 10 forecasts in the first quarter and `rMAE` is taken without them. A group is short by nature, so the tests say nothing below 30 days (`DM verdict` and `in MCS` stay empty). They are not in `compute_all()`, so the exporters do not write them.

| Method | Groups | Columns |
|---|---|---|
| `by_quarter()` | The calendar quarter of the **target date**, with its year | `year`, `quarter` |
| `by_day()` | The target date. All horizons that forecast it are in its group, so a group holds one run date per horizon and the tests have little to run on | `target_date` |
| `by_price_level(spike=None)` | `negative` (actual price below 0), `spike` (at or above the threshold), `normal` (the rest) | `price_level` |

**Price levels come from the actual price alone.** A prediction never decides which group a forecast is in, so a model that misses a spike is scored on it, not excused. A forecast is in exactly one group; one without a finite actual price is in none. The spike threshold is `min(300, 99th percentile)` of the actual prices (the definition of the evaluation-protocol study, which gives about 88 spike hours a year), taken once from the actuals of all the models together, with each period (target date and hour) counted once however many horizons forecast it. It is the same number for every model, so the groups are the same periods for all of them. `df.attrs` holds `spike_threshold`, `spike_rule` and `spike_quantile` (the 99th percentile, NaN when `spike` is given), and `no_level`: the forecasts per model that are in no group for lack of a finite actual, which are the difference between the groups and `summary()`. `spike` replaces the rule with a price you choose; it must be positive, and a negative price is never a spike.

## Daily aggregates

```python
daily_max = report.aggregate("1D", agg="max")  # "max", "min", "mean" or "spread" (maximum minus minimum)

print(daily_max.summary())
#    model  forecasts      MAE    rMAE
# 0    OLS        365  8.1879  1.0000
# 1  Lasso        324  6.0205  0.7390

print(daily_max.forecast_counts())
#    model  planned  produced  missing  incomplete_day
# 0    OLS      365       365        0               0
# 1  Lasso      365       324       41              41
```

`aggregate()` turns hourly or 15-minute forecasts into forecasts of a daily quantity: for each (run date, target date, horizon) the prediction and the actual are each reduced by `agg` over the periods of the day. The result is a new `EvaluationReport` with `hour` 0 and frequency `"1D"` and the evaluators of the first report, so `summary()`, `by_*()`, `tests()`, `mcs()`, `ranks()`, `compare()` and `leaderboard()` score the daily maximum (or minimum, mean, spread) the same way as they score hours.

- **Only whole days count.** A day is whole when it has all its periods with a finite prediction and actual: 23, 24 or 25 hours, or 92, 96 or 100 quarter-hours. This is judged by the period ids and not by how many there are, with the rule `EconomicEvaluator` uses for a battery's day: a real run numbers periods by the clock, so a 23-hour day lacks the hour it skips (between 01:00 and 04:00) and a 25-hour day has ids 0-24. A day of 24 forecasts that are not the hours 0-23 is not a day, and ids 0-22 count as a 23-hour day only on a Sunday, when clocks change.
- **Against the other models.** The periods of a target date are every period that any model has a record for, at any horizon, and a day with fewer finite periods is dropped. So a model that lacks 02:00 on an ordinary Tuesday is not aggregated over 23 hours while another is aggregated over 24 (they would be scored against different actuals on one key), and the horizon-1 forecasts of the day after the spring switch, which lack the hour the switch day skipped, are dropped when horizon 2 has all 24. A period that no model has at any horizon is taken at its word, because that is how a clock change looks in the ids: in a run of horizon 1 alone the day after the spring switch is aggregated over its 23 forecast hours, and a period with a record but no actual price (the ambiguous 02:00 of the autumn switch day in a real run) takes the day out for every model.
- **Each model on its own days.** A model that lacks a period loses that day for itself and for nobody else. The comparison tables then use the days the models share and print how many.
- **Dropped days are counted.** They are in `forecast_counts()` of the new report as `missing` and under `incomplete_day`, and a warning names the models. `planned` there is the number of days with any forecast of the model; days without a single forecast are in the `forecast_counts()` of the report you aggregated. As in the example, models can then have different numbers of forecasts, so compare them with `rMAE` or `tests()`, which use the common days, and not with the raw `MAE` column.
- **What differs from an hourly report.** `tests(by="hour")` raises (a day has no hours), and the `hours_*` columns of `compare()` are NaN. `EconomicEvaluator` is left out with a warning, since a schedule needs the periods of a day. The new report has no `source_data`, and `day_in_test` counts days from the earliest run date. It is a daily report, so it cannot be joined to an hourly one.

## Joining reports

```python
joined = report.join(other_report)
```

`join()` returns a report with the models of both, for results of separate runs that are to be compared (a SLURM array, a model added later). The references to the results are kept as they are, and the evaluators are the ones of the report you call it on: if the other report has different ones, a warning says so, since the columns of its evaluators are lost and its models are scored by yours (an `rMAEEvaluator` with a base model is computed against the left report's base). It raises `ValueError` when a model name is in both reports, or when the results have different frequencies. Every table is computed on forecast keys, so a joined report gives the tables one report over all the models would, bit for bit.

## Accessing Raw Predictions

```python
# All predictions as a tidy DataFrame (no actuals needed, e.g. in forecast-only mode)
predictions = report.predictions()

print(predictions.head())
#      run_date target_date  hour  horizon  prediction model
# 0  2024-02-01  2024-02-02     0        1       45.23   OLS
```

## Order of results

A [predictor function of `ctx`](/epftoolbox2/model-pipeline/predictor-functions/#results) adds `predictor_list` to each record and `predictor_lists` to the model's ref, which says which list its forecasts used. A [`train_on="matching"` list](/epftoolbox2/model-pipeline/predictor-functions/#training-only-on-the-days-with-the-same-list) adds `training_days` and `matching_days` to its records, and `report.training_days()` sums them up per list. It is empty, with the same columns, without such a list.

Workers write records as their tasks finish, so the order of the lines in a results file varies with the number of processes, scheduling and resuming. The report does not depend on it: each model's forecasts are evaluated in the order of their forecast key (`run_date`, `target_date`, `horizon`, `hour`), so the same results give the same metrics bit for bit (floating-point sums depend on the order of their terms). `iter_details()` lists a model's rows in that order, and `predictions()` is sorted by model, horizon, hour, then run date.

## Reports from saved results

`ModelPipeline.report(save_dir, test_start=..., test_end=..., horizon=..., freq=...)` builds a report from the per-model JSONL files a previous run wrote to `save_dir` and runs the evaluators and exporters. `Workflow.report()` / `epf report` use it to merge the results of a SLURM array (see [Workflow](/epftoolbox2/workflow/overview/#slurm-arrays-one-model-or-a-block-per-task)).

If any model's file has fewer forecasts inside `test_start`..`test_end` than the planned count for that period × `horizon` × `freq` (an array task that failed or never ran, or forecasts skipped), `report()` warns, naming the models and their produced/planned/missing counts, and notes that metrics are then computed on different forecast sets for those models.

What the runs listed in `report.availability()` comes back from their manifests, whichever pipeline builds the report. So do the skip reasons in `forecast_counts()`, per window: a run that skipped forecasts records how many in its manifest when it finishes, and a resume, or a run naming one window, replaces that window's counts only. So what `missing` holds beyond the reasons is a task that crashed or never ran. The counts can't be split by day, so a report on part of a plain run's days has `missing` and no reasons.

After runs with validation, `report(save_dir, validation=..., test=..., horizon=..., freq=...)` takes what `run` was told and returns what `run` returns, with `report.validation`. Each window must be what the manifests record, and with no window at all the report is on every window they record. Results whose manifests record different windows raise `ValueError`. See [Validation and Test](/epftoolbox2/model-pipeline/validation-and-test/).

## Windows

The report of a run with validation is on the test window, and `report.validation` on the validation window. `report.window` says which one you hold, and every method keeps to it, so nothing of one window is reachable from the other. `forecast_counts()` of a validation report counts as `target_in_test` the forecasts left out because their target day is not before the first test issue day. `report.windows` holds the windows and segments of the run. A plain run's `report.window` is `"test"`, its `report.windows` is `None`, and its `report.validation` raises `ValueError`.
