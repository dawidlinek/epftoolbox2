---
title: Release Notes
description: What changed between releases, breaking changes first
---

Notes for changes that need something from you. The full list of changes of a release is in the commit log.

## Unreleased

### Breaking changes

- **Training rows are chosen by date, and results made before that are refused.** A forecast's training rows were found by counting
  back from the issue day, which went wrong around a clock change: on the days after an autumn switch the forecast of the switch hour
  trained on a target dated on the issue day or later, a look-ahead, and on a spring switch day the key of the skipped hour was built
  from the day before's row and carried another day's `actual`. Now a forecast (period, h, D) trains on the rows dated from D-W-h to
  D-1-h, forecasts from the last row dated D, and exists only if the clock has its period on D and on D+h.
  - **What moves.** Per year, for a run of horizons 1 to H at 1 h: H(H+1)/2 forecasts after the autumn switch, H(H-1)/2 after the spring
    one, and the keys of the skipped hour go, with a slot the clock doesn't have no longer counted in `planned` or as missing. At
    `freq="15min"` that is 4 times as many. Every other forecast is the same, bit for bit, and so is every forecast of a frame without a
    switch or a gap. A row lost to a gap in the last days before the issue no longer costs a newer row, so such a key moves. `test_golden_metrics.py` went from 140
    forecasts to 139, with one mislabelled key gone.
  - **Old results are refused on resume.** A manifest without the new field `training_cut` raises `ManifestMismatchError`, because the
    stale records would stay in the file and be read. Use a new `save_dir`. Results accepted anyway, with `allow_manifest_mismatch=True`
    or from a folder with no manifest, are marked `{"rows": "dates", "stored": "unverified"}` for good, and every `report()` of them warns;
    a record of a slot the clock doesn't have is left out of every table and count, with a warning.
  - **`aggregate()` reads a whole day from the clock.** Every target day of a forecast issued on a spring switch day is now dropped, also
    the day after it in a run of horizon 1 alone, where it was aggregated over 23 of its 24 hours. The short day of a zone whose clock
    skips midnight, 22:00 or half an hour is now whole. Reports of plain lists keep the old rule.
  - **A manifest names its zone.** `data_timezone` of a dateutil zone is its name (`Europe/Warsaw`), not `tzfile(...)`, so `report()`
    knows its clock. A zone with no such name warns when a run writes its files.
  - **`verify` no longer calls those days a leak**, and a verify over a spring switch day counts 23 forecasts per horizon.
  See [Time and Calendars](/epftoolbox2/model-pipeline/time-and-calendars/).
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
- A workflow file with `model_index: $SLURM_ARRAY_TASK_ID` couldn't be loaded where that variable isn't set, so `epf report` and
  `epf validate` failed on it outside the array with `invalid literal for int()`. They now work, and only `run()` raises, saying so.
- `CsvExporter` and `ExcelExporter` raised `KeyError` when a model had no forecasts in the report's window (an array task that stopped
  during validation, say), so one unfinished model cost the whole CSV or the `details` sheet. The model now keeps its columns, empty,
  where its place among the models puts them, and the other models' rows are as they were; a file of a report with no such model doesn't
  change by a byte. When no model has forecasts, the CSV is the header alone and the workbook holds the `Summary` and `Details` sheets only.

### New

- `environment.models_per_task: k` in `Workflow` YAML (`Workflow(models_per_task=k)`): SLURM array task `i` runs the block of models
  `[i*k, (i+1)*k)` instead of one model. `epf array experiment.yaml` prints the range for `sbatch --array=`, and
  `Workflow.array_size()` the number of tasks. `epf report` merges the blocks and computes the ensembles as before. The block size is no
  part of a result: nothing stored records it, and results resume at another block size and merge to what one run gives, bit for bit.
  Files without the key don't change. See [Workflow](/epftoolbox2/workflow/overview/#slurm-arrays-one-model-or-a-block-per-task).
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
