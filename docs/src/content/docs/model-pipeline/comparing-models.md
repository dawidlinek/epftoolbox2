---
title: Comparing Models
description: Tests, confidence sets, rank uncertainty and how to read them
---

A difference in MAE between two models is not yet a finding. `EvaluationReport` has five methods that say whether it is one, built on a study of what these tests do on a large set of stored forecasts (including pairs of seeds of one model, where every "difference" is a false alarm).

| Method | Question |
|--------|----------|
| [`leaderboard()`](#leaderboard) | How do the models line up, and how far is each from the benchmark? |
| [`tests()`](#testing-against-a-benchmark) | Which models differ from the benchmark, after correcting for testing several? |
| [`mcs()`](#model-confidence-set) | Which models can the data not tell from the best one? |
| [`ranks()`](#ranking-uncertainty) | How decisive is the ranking? |
| [`compare()`](#two-models) | What do we know about these two models? |

The same numbers can go into `summary()` and the `by_*()` tables through [`DMEvaluator`](/epftoolbox2/evaluators/dm/) and [`MCSEvaluator`](/epftoolbox2/evaluators/mcs/), and so into the exporters.

## The rules they share

- **Common keys.** A forecast is identified by `(run_date, target_date, hour, horizon)`. Each comparison uses the keys its models all hold, with a finite prediction and actual, and nothing else. In `tests()`, `compare()` and `leaderboard()` that is per pair, so a model with missing forecasts leaves the other pairs alone. In `mcs()` and `ranks()` the whole set of models is the comparison, so it is the keys all of them hold.
- **Every table prints n.** `n_days` and `n_forecasts` are the days and forecasts behind each row. Start with `report.forecast_counts()` to see what each model produced.
- **Daily loss.** The tests run on the mean absolute error of each (run date, horizon) over the periods that exist on those keys, so DST days with 23 or 25 hours (92 or 100 quarter-hours) work as they are.
- **Horizons.** `tests()` has one set of rows per horizon. `horizon="all"` averages the daily losses of all horizons per run date, which is also what the other methods do by default (`horizon=2` picks one).
- **Two-sided at 5 %.** A positive statistic means the model is better than the benchmark.
- **Fixed seeds.** Same data, same numbers. The seed is a parameter of every method that resamples.

## Test length

One test year resolves differences of about 5 % of the MAE. Gaps of 1–2 % need about 2,700 days, so with one year most small gaps will read `no difference`, and that is the right answer. The `mde` column gives the minimum detectable effect for each pair: the difference (as a share of the benchmark's MAE) this test length would detect with 80 % power.

| `n_days` | What happens |
|----------|--------------|
| below 182 | A warning, and a note in `df.attrs["note"]` |
| below 365 | A note in `df.attrs["note"]` |

## Testing against a benchmark

```python
report.tests(benchmark="OLS")
#    model benchmark  horizon  n_days  n_forecasts    mae  mae_benchmark  ratio  ci_low  ci_high  statistic  p_value  p_holm  m    mde        verdict
# 0  Lasso       OLS        1     365         8760  24.81          26.02  0.954   0.921    0.986       2.90   0.0037  0.0074  2  0.052         better
# 1   Mean       OLS        1     365         8760  25.77          26.02  0.990   0.958    1.024       0.61   0.5440  0.5440  2  0.050  no difference
```

- **`ratio`** is `mae / mae_benchmark` on the pair's forecasts, with a 95 % bootstrap interval in weekly blocks. Read this first: how large is the difference, and how well is it known?
- **`p_value`** is the DM test on the daily loss with a Newey-West variance (lag 5 at 365 days). On pairs of seeds of one model it rejected 7.5 % of the time, near the nominal 5 %. `variance="plain"` reproduces the tables of epftoolbox.
- **`p_holm`** corrects for testing `m` models against one reference, and **`verdict`** reads it at 5 %. In the study the correction turned 12 "better than LEAR" into 8. Holm runs within a horizon; testing several horizons is one more family of tests to keep in mind.
- `benchmark` defaults to the one of the report's `DMEvaluator`. The same comparison on another benchmark is one more call.

### Per-hour DM is a heatmap

```python
report.tests(benchmark="OLS", by="hour")
```

One row per (model, horizon, hour), without `p_holm` and `verdict`, and with a note in `df.attrs["descriptive"]`. Use it to see where in the day two models differ, never to say that one is better: on pairs of seeds of one model, per-hour tests with Holm over the hours flagged a difference 37 % of the time. Seeds of a network differ persistently in hour-specific errors, which cancels out in the daily loss.

## Model confidence set

```python
report.mcs()
#     model  n_days  n_forecasts    mae  ratio_to_best  p_value  in_90  in_75
# 0   Lasso     365         8760  24.81          1.000    1.000   True   True
# 1    Mean     365         8760  25.77          1.039    0.170   True  False
# 2     OLS     365         8760  26.02          1.049    0.012  False  False
```

The set of models the data cannot tell from the best: `in_90` is the 90 % set and `in_75` the 75 % set. It takes at least 3 models. Use it when there is no natural reference, and read the width of the set: several members means the data does not pick a winner.

Since the whole set is the comparison, one model with few forecasts cuts the sample of all. `df.attrs` holds `n_days`, `n_forecasts` and `limited_by`, the models that lack forecasts the others have, with how many. Leave such a model out with `models=[...]`.

## Ranking uncertainty

```python
report.ranks()
#     model  n_days  n_forecasts    mae  rank  p_best  rank_low  rank_high
# 0   Lasso     365         8760  24.81     1   0.610         1          2
# 1    Mean     365         8760  25.77     2   0.300         1          3
# 2     OLS     365         8760  26.02     3   0.090         2          3
```

`p_best` is the share of bootstrap samples of days in which a model has the lowest MAE, and `rank_low`, `rank_high` a 90 % interval of its rank. In the study the best single model's `p_best` was only 0.50–0.65, so a first place in a table is a weaker claim than it looks. As in `mcs()`, it runs on the keys all the models hold, and `models=[...]` chooses the set.

## Two models

```python
report.compare("Lasso", "OLS")
#        a    b  horizon  n_days  n_forecasts  mae_a  mae_b  ratio  ci_low  ci_high  statistic  p_plain   p_hac  hours_a_better  hours_b_better  hours_tested    mde
# 0  Lasso  OLS        1     365         8760  24.81  26.02  0.954   0.921    0.986       2.90   0.0041  0.0037               4               0            24  0.052
```

One row. `ratio` is `mae_a / mae_b`, so below 1 means `a` is better, and a positive `statistic` says the same. `p_plain` and `p_hac` are the DM p-values with the two variances. `hours_a_better` and `hours_b_better` count the hours where a per-hour DM (plain variance, Holm over the hours) rejects: descriptive, see above.

## Leaderboard

```python
report.leaderboard(benchmark="OLS")
#    model  n_forecasts    MAE  n_days  n_common  ratio  ci_low  ci_high  p_value  p_holm
# 0  Lasso         8760  24.81     365      8760  0.954   0.921    0.986   0.0037  0.0074
# 1   Mean         8760  25.77     365      8760  0.990   0.958    1.024   0.5440  0.5440
# 2    OLS         8760  26.02     365      8760  1.000     NaN      NaN      NaN     NaN
```

One row per model, sorted by MAE. `n_forecasts` and `MAE` are the model's own, and `n_days`, `n_common` and the rest are on the forecasts it shares with the benchmark. When the models have different forecast sets the MAE column compares different samples: a warning says so, and `ratio` is the column to read. Without a benchmark (none given, no `DMEvaluator`) it has only `model`, `n_forecasts`, `MAE` and `n_days`.

## Which one to report

1. `report.forecast_counts()`: did every model produce what it should?
2. `report.leaderboard(benchmark=...)` for the overview, with one reference that means something (a weekly naive, LEAR).
3. `report.tests(benchmark=...)` for the claims, with their intervals and `m`.
4. `report.mcs()` and `report.ranks()` when several models are close.
5. `report.tests(by="hour")` and `report.by_hour()` to understand them, not to prove anything.
