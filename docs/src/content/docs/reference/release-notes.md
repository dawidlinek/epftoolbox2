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

### New

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

### Fixed

- **A column that is constant up to rounding noise is no longer divided by its standard deviation.** The scalers took a column
  for constant only when its standard deviation was exactly 0. A column of 0.3 with floating-point error, or 0.3 repeated 30
  times (numpy gives a standard deviation of 5.6e-17, not 0), was divided by about 1e-16 and became a regressor: on a narrow
  training window a test row of 0.31 gave a forecast of 3.5e13 with `StandardScaler`, and `AsinhScaler` and `NPITScaler` turned
  the noise into an input. A standard deviation at or below 1e-12 of the column's largest finite absolute value now counts as
  constant, for every scaler and for the target, and the scale is 1.0 as for 0 before. A constant predictor's training values are
  set to exactly 0, so the model gives it the coefficient 0 whichever solver it uses. **Forecasts change only for a model that had
  such a column in a training window**; all other columns scale bit for bit as before. On the 247 LEAR inputs of the five
  epftoolbox benchmark datasets (NP, PJM, FR, BE, DE) no column of any training window of 7 to 728 days was within a factor of
  about 1e9 of the threshold, so those results don't change. Results stored by an earlier version resume without a warning:
  re-run a model whose training windows could hold a constant column. See
  [Constant columns](/epftoolbox2/models/scalers/#constant-columns).
