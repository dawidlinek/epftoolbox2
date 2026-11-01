---
title: Release Notes
description: What changed between releases, breaking changes first
---

Notes for changes that need something from you. The full list of changes of a release is in the commit log.

## Unreleased

### Breaking changes

- **A reversed test period now raises.** `ModelPipeline.run(df, test_start=later, test_end=earlier)` used to run nothing and
  return an empty report, and now raises `ValueError`, as a reversed `test=` does. So does `ModelPipeline.report()`.
- **An issue time now warns instead of raising.** `ModelPipeline(issue_time=...)` and `ModelPipeline(information_set=...)` used
  to refuse a predictor known after the issue time with `InformationSetError` before any fit. They now default to
  `availability="warn"`: the run goes on, and the predictor is listed in `report.availability()`, a warning, the terminal exporter and
  the manifest. A column without a rule is listed as `no rule`, whatever `unknown` says. **The forecasts of a pipeline that passed
  before don't change.** To keep refusing, pass `availability="raise"`, which is the old behaviour with the same messages.
  See [Information Set](/epftoolbox2/model-pipeline/information-set/#three-levels).
- **So do YAML files.** A `ModelPipeline` or `Workflow` file with an `information_set` and no `availability` key loads with the new
  default, so it warns where it used to refuse. Add `availability: raise` to keep refusing. `epf model --issue-time` warns too; add
  `--availability raise`.
- **Exact clones no longer stop a comparison.** `report.mcs()` used to raise and `MCSEvaluator` to give NaN for every model when two
  models had the same predictions, and `report.ranks()` to split their chance of being best by their order. They now treat them as one
  model and give each its result, and in `tests()`, `leaderboard()` and `DMEvaluator` they count once in `m`. `df.attrs["clones"]` names
  them. Tables without clones don't change. See [Comparing Models](/epftoolbox2/model-pipeline/comparing-models/#exact-clones).
- **A predictor function behaves accordingly.** At the new default `ctx.value()` of a late column returns the value and the column is
  listed, where it used to raise, and a column without a rule no longer raises in `ctx.published()` or `ctx.value()`.
  `ctx.published()` answers as before for every column with a rule.

Unchanged: `BaseModel.run(information_set=...)`, without a pipeline, and `leakage_probe` still default to refusing, and
`pipeline.verify()` still reports a late predictor as a `leak` at every level, and now also what a predictor function uses late on
an issue day it didn't sample.

### Fixed

- `ModelPipeline.report(test=narrower)` counted every line of a results file as produced, so a shorter period looked complete. It now
  counts the forecasts inside the period, and `forecast_counts()` and its warning say what is missing.

### New

- A merged `report()` has the skip reasons in `forecast_counts()`. A run that skipped forecasts records them per window in its
  manifest (`skipped`), when it finishes; a run with no skips has no such key, and a resume or a run naming one window replaces
  that window's counts only.
- A predictor function returning the same names in another order warns: the order is part of a list.
- `Use(list, train_on="matching")` in a predictor function of `ctx`: a forecast trains only on the past days for which the function
  returned that list and rule, per (period, horizon), after the usual training cut. The function is evaluated on those days too, once
  per (day, period, horizon), before any fit and as of each day's own issue time, which doubles the calls for a year of test and a
  window of 364. The window isn't extended, so such a forecast trains on fewer rows. A forecast with no such day is counted as
  `no_matching_days`, and as many days as predictors or fewer warns, also in the terminal exporter. Records of such a list gain
  `training_days` and `matching_days`, and resuming compares the set. `report.training_days()` and the `lists` table of `explain()` show how many days
  they train on. Anything without such a list doesn't change. See
  [Predictor Functions](/epftoolbox2/model-pipeline/predictor-functions/#training-only-on-the-days-with-the-same-list).
- A manifest gains `column_hashes` (each column a run reads, on its own) and `frame_hash` (the whole frame), through `test_end`,
  on every run, and `data_unverified` on a resume that accepted other data or had no manifest. They never refuse a resume, and
  manifests without them load. Records and tables don't change.
- `AverageEnsemble(members, top=None | k | "validation", by="validation", name=...)` and `pipeline.add_ensemble()`, from
  `epftoolbox2.ensembles`: the mean of stored forecasts on identical keys, with the members and their number chosen on validation
  per test period, `member_missing` in `forecast_counts()` and `report.choices()`. `ModelPipeline.report()` and `epf report` compute
  ensembles from stored results, and YAML has an `ensembles:` section. Members must share their issue time and what they read. See
  [Ensembles](/epftoolbox2/model-pipeline/ensembles/).
- `run(df, validation=(start, end) | "previous_year", test=(start, end))`: one run per model over both windows, into one results
  file and one manifest. Each forecast is bit for bit what a plain run over its window gives. `report.validation`, `report.window`
  and `report.windows`, also in `ModelPipeline.report()`, `Workflow` YAML and `epf model --validation`. Plain runs and their files
  don't change, `test_start` and `test_end` keep working. See
  [Validation and Test](/epftoolbox2/model-pipeline/validation-and-test/).
- `availability="none" | "warn" | "raise"` on `ModelPipeline` (and in YAML, and `epf model --availability`).
- `report.availability()`: the inputs a run can't vouch for, per model, list, period and horizon.
- `pipeline.explain()`: what a forecast's predictors hold relative to the target day, and when each is known; for a predictor
  function, the distinct lists with their days and what each adds or drops.
