---
title: Release Notes
description: What changed between releases, breaking changes first
---

Notes for changes that need something from you. The full list of changes of a release is in the commit log.

## Unreleased

### Breaking changes

- **`ModelPipeline.report()` reads `freq` from the manifests and refuses one they contradict.** Its `freq` defaulted to `"1h"`
  and was taken at its word: daily results reported at the default planned 24 forecasts a day and called almost all of them missing,
  and hourly results reported with `freq="1D"`, or 15-minute ones with `"1h"`, showed no missing forecast at all, gaps included.
  `freq` now defaults to `None`, the one the manifests record (without one, the one the records carry; `"1h"` if none); models run at different frequencies raise
  `ValueError`, and so does a `freq` the manifests contradict. A report told the freq its runs had doesn't change.
- **`ModelPipeline.report()` reads only the horizons it plans.** `report(save_dir, horizon=3)` of results run with `horizon=7` used to
  plan horizons 1 to 3 and read every record on file, so its tables held horizons 4 to 7 and `planned` equalled `produced` with no
  warning. It now reads horizons 1 to 3 alone, and so do the ensembles it computes, which planned what their members ran. A report told
  the horizon its runs had doesn't change. A report told other horizons than its runs had has no skip reasons in `forecast_counts()`,
  only `missing`: the runs counted them over all their horizons.
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
- **A `weather_cutoff` rule is in UTC.** A model run's time is a UTC hour, but a rule built with `timezone=None`, or from a dict or
  YAML entry without a zone, read it in the data's zone or in `Europe/Brussels`: run `"06"` plus `"5h"` was known at 11:00 Warsaw
  time in summer, two hours before the 13:00 that `describe()` says, a look-ahead. Both now mean UTC, and any other zone raises.
  Rules made by `Availability.weather_cutoff(run, delay)` and by the data sources were already in UTC and don't change.
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

- **Nothing after the issue day decides how a column is scaled.** A model leaves a 0/1 column unscaled and decides which columns
  those are once per list, period and horizon (and window), from the training rows of one day. A column with no value in those rows,
  such as one that starts inside the test, can't tell. It stayed unscaled for good in a fixed list and in a function's list used on
  every issue day, and was decided on a later issue day of the list otherwise; whether a list was used on every day depended on the
  last issue day and on what a predictor function returned for a slot the clock skips. So a later `test_end`, or a value a function
  read after D, moved earlier forecasts: in a run of every day the forecasts of 5 April moved when the test went on from 7 to 12
  April, 22 of 24 by up to 0.14 with `LassoCVModel` and 24 of 24 by up to 0.25 with `OLSModel` on two drivers. Now such a column is
  decided on the first calendar day after that day whose rows have a value of it, for every list, never after the window's last day,
  and a slot the clock skips has no say in which day a function's list starts from. See [Binary columns](/epftoolbox2/models/scalers/#binary-columns).
  - **What moves.** Only forecasts of a list holding a column with no value on the day its mask is read and values later in the
    window: from the day it has one, that column may now be standardised where it stayed raw, or decided on another day. Every other
    forecast is bit for bit as before: `test_golden_metrics.py` doesn't move, nor do the 2,928 forecasts of `LassoCVModel` on
    `lear_predictors` for Lago's DE data (W=364, 61 days, horizon 2). In a run of Fridays, a function's list first used on the second
    Friday, at a period and horizon whose key on the first Friday the clock skips, now reads its mask on the window's first day, so
    its records there differ from a run of every day's ([where a function's records differ](/epftoolbox2/model-pipeline/time-and-calendars/#some-weekdays-only)).
  - **Resuming.** Results whose stored forecasts now get another mask raise `ManifestMismatchError`: run them into a new `save_dir`.
    For a predictor function and for a run with validation windows that is the mask digest of each stored forecast. Plain results of
    fixed lists stored no digest, so a plain run of fixed lists now records its digests (`binary_masks`, under the test window's key)
    when a column was read on a later day and that changed a mask, and resuming results whose manifest has none then raises, saying the
    scaling rule changed; `allow_manifest_mismatch=True` resumes anyway. Where nothing was read again the manifest is as it was, byte
    for byte apart from provenance, and resumes.
- **Availability compares instants, not clock readings.** `InformationSet` worked in wall-clock minutes of the data's zone, so around
  a clock change a value known after the issue could read as known before it: `realised("1h")` at 01:00 on the Warsaw spring day read
  as known at 03:00, though 01:00 CET plus two hours is 04:00 CEST, and a value known in the second 01:00 read as before an issue in the
  first 01:30. Delays, period lengths and `_h` and `_min` shifts now are elapsed time, each `_d` shift keeps the wall clock and the
  occurrence, shifts apply from the end of the name, and a repeated autumn hour is checked at its second occurrence, also against a
  rule in another zone. **No forecast of a fixed list moves.** Next to a switch `availability="raise"` may now refuse what this error
  let through, `report.availability()`, `pipeline.verify()` and `explain()` may now call it late, and what it wrongly called late can
  pass. Listings show the real time, with its UTC offset inside the repeated hour, and the manifest's `minutes_late` counts elapsed
  minutes, so across a switch another day can be a column's worst. A predictor function that asks `ctx.published()` about such a cell
  gets the new answer and may take another list there, which moves that forecast, and resuming its stored forecast then raises
  `ManifestMismatchError`, as for any changed answer. A rule's clock time that the clocks repeat counts at its second occurrence,
  and a period they skip has no row, so nothing is late there. See
  [Clock Changes](/epftoolbox2/model-pipeline/information-set/#clock-changes).
- **Listings stored before this change are read with a warning.** They were made on the wall clock, so next to a clock change
  `report.availability()` of such results can list other periods than a fresh run. A run at `availability="warn"` lists them anew,
  in place where the results resume (results made before rows were chosen by date don't, see above), else into a new `save_dir`;
  until then `report()` says so.
- `ModelPipeline.report(test=narrower)` counted every line of a results file as produced, so a shorter period looked complete. It now
  counts the forecasts inside the period, and `forecast_counts()` and its warning say what is missing.
- A workflow file with `model_index: $SLURM_ARRAY_TASK_ID` couldn't be loaded where that variable isn't set, so `epf report` and
  `epf validate` failed on it outside the array with `invalid literal for int()`. They now work, and only `run()` raises, saying so.
- `CsvExporter` and `ExcelExporter` raised `KeyError` when a model had no forecasts in the report's window (an array task that stopped
  during validation, say), so one unfinished model cost the whole CSV or the `details` sheet. The model now keeps its columns, empty,
  where its place among the models puts them, and the other models' rows are as they were; a file of a report with no such model doesn't
  change by a byte. When no model has forecasts, the CSV is the header alone and the workbook holds the `Summary` and `Details` sheets only.

### New

- **`refit=`: fitting on a fixed calendar.** `OLSModel(..., refit="7D")` and `LassoCVModel(..., refit="1MS")` fit once per unit (a
  period, a horizon, a refit day and a predictor list) instead of once per forecast: `"kD"` every k days, `"kMS"` on the first of every
  k-th month, counted from Monday 1970-01-05, so a forecast's refit day depends on its issue day alone, never on where a run starts. A
  forecast issued on D trains on the rows of the forecast issued on its refit day a (dated a-W-h to a-1-h; with
  `newest_target="published"` the newest target known at a's issue time), finds its binary columns on those rows, and is predicted from
  its own row on its own, so it is bit for bit the same whichever run, process count or resume made it. Records gain `refit_anchor`,
  the manifest's `training_cut` gains `refit` and `refit_epoch` (changing `refit` refuses a resume), resume compares each stored
  anchor, and YAML writes `refit` only when set. A frame that doesn't hold W+h days before the refit day of the first issue day raises `ValueError` naming the day it
  must start on (one sized for `refit=None` is too short whenever the test doesn't start on a refit day), and a run uses
  at most one process per refit day. Other values raise, `"auto"` included, and so does `refit` on a custom model with only
  `_fit_predict`. **Nothing moves without the keyword**: `refit=None` is every record, manifest and table as before. With it the
  numbers move, since fits are older: on Lago DE in 2017 at horizon 1 (LassoCV on the LEAR inputs, W = 364, the same 8712
  forecasts), MAE was 4.6665 refitting daily, 4.6560 (-0.23 %) with `"7D"` from 1248 fits and 4.7134 (+1.00 %) with `"1MS"` from 288,
  with naive standard errors of the daily difference of 0.6 % and 1.4 % of the MAE; forecasts issued on a refit day equal the daily
  refit's bit for bit. See
  [Refitting on a calendar](/epftoolbox2/model-pipeline/time-and-calendars/#refitting-on-a-calendar).
- **`freq="1D"`: one model per day.** `pipeline.run(df.at_time("12:00"), ..., freq="1D")` fits one model per day on a frame with one row
  per local date, every row at one time of day. It is accepted wherever `freq` is: `ModelPipeline.run`, `report`, `verify` and
  `explain`, `BaseModel.run`, `leakage_probe` and `positive_control`, `Workflow` YAML (`model_pipeline.freq: 1D`), `epf model` and
  `epf verify` (`--freq 1D`) and the interactive wizard. Records have `hour` 0 and `freq` `"1D"`, the shape `report.aggregate()` gives.
  The row's time of day is read from the index and is what its names mean (`x_h-1` is one elapsed hour before the stamp), so stamp
  the rows at the time the forecast is made: a frame stamped at midnight that holds what is true at noon reads `load_actual_h-1` as
  23:00 on D-1, and neither the check nor `verify` can see the look-ahead. A run warns when the rows are stamped at or near midnight and a predictor counts hours from the stamp. A plain `realised` column such as `actual_daily_max` is the
  value of its whole local date, known `delay` after that date ends, so it is late on its own date at any issue time. A plain `published` or `weekly` column (a `daily_close` too) is read at its date's last slot, so a date that spans two of the rule's days is known when the later one is. A daily run of
  `df.at_time("12:00")` is period 12 of the hourly run of `df`, bit for bit. **Nothing else moves**: `"1h"` and `"15min"`
  runs are what they were. `EconomicEvaluator` with `freq="1D"` raises before any model is planned or run, since a schedule needs the
  periods of a day. `ModelPipeline.explain(hour=...)` now defaults to `None`, which is 12 at `"1h"` and `"15min"` as before and 0 at
  `"1D"`. See [Daily frames](/epftoolbox2/model-pipeline/time-and-calendars/#daily-frames).
- **`horizons=`: some horizons only.** `pipeline.run(df, test=..., horizons=[1, 3, 7])` (or `range(3, 10)`) forecasts the listed
  horizons and nothing else, and `report`, `verify`, `explain`, `BaseModel.run`, `leakage_probe` and `positive_control` take it too, as do
  `Workflow` YAML (`model_pipeline.horizons: [1, 3, 7]`) and `epf model` / `epf verify` (`--horizons 3-9`, `1,3,7` or `1-3,7`). Each
  forecast reads its own rows, so every record of a model, and of a plain average, is bit for bit the one `horizon=max(horizons)` gives
  that key. A `top=` ensemble chooses on validation pooled over the horizons the run made, so it may choose other members than a run of
  every horizon. `horizon=H` keeps meaning
  every horizon from 1 to H, and **nothing moves without the keyword**: `range(1, H + 1)` is `horizon=H` down to the bytes of the files.
  Giving both raises, so `horizon` now defaults to `None`, which is 7 for `run` and `report` and 1 for `verify`, `explain` and the probes,
  and `Workflow(model_horizon=...)` defaults to `None` (7). Targets are still built for 1 to the largest horizon, so `price_d+2` stays
  refused as an input at `horizons=[3]`. The manifest records `horizons` when they aren't 1 to the largest and a resume with another list
  refuses. `report()` told neither `horizon` nor `horizons` plans what the manifests record (7 without one, as before) and raises when
  the models ran different lists; it reads only the records of the horizons it plans. See
  [Some horizons only](/epftoolbox2/model-pipeline/time-and-calendars/#some-horizons-only).
- **`issue_days=`: some weekdays only.** `pipeline.run(df, test=..., horizons=range(3, 10), issue_days=["friday"])` issues forecasts on
  the listed weekdays and no others, as a week-ahead study does. `report`, `verify` and `BaseModel.run` take it too, as do `Workflow` YAML
  (`model_pipeline.issue_days: [friday]`) and `epf model` / `epf verify` (`--issue-days friday,monday`). Names in any case; all seven are
  every day, and an empty list, an unknown or repeated name, a bare string or numbers raise. Training is untouched: every date's row
  trains, a list's mask of binary columns is read on its window's first calendar day whatever its weekday, so for a fixed list a Friday's
  record is bit for bit the one a run of every day gives it (a predictor function's can differ where the two runs read a list's mask on
  different days, neither after the issue day), and `day_in_test` still counts calendar days. The other days are neither
  planned nor missing. **Nothing moves without the keyword.** The manifest records `issue_days` when it isn't every day and a resume
  with other weekdays refuses; `report()` reads the weekdays from the manifests, raises for models run on different ones, and told
  `issue_days` reads only the records issued on them. The information-set check and the `availability="warn"` listing look at the
  listed days alone, so a column published weekly (A31, Fridays 10:00) used at horizon 9 passes on a Friday and is late on a Monday. See
  [Some weekdays only](/epftoolbox2/model-pipeline/time-and-calendars/#some-weekdays-only).
- **A choice on few issue days warns.** An ensemble with `top=` warns when its validation forecasts were issued on fewer than 28
  distinct days, next to the warning on fewer than 28 target days, which four Fridays at horizons 1 to 7 pass. A run of every day
  that warned before warns once more; nothing is chosen differently. See [Ensembles](/epftoolbox2/model-pipeline/ensembles/).
- **The test-length messages read the calendar span.** The note below 365 days and the warning below 182, in `tests()`, `compare()`,
  `leaderboard()`, the set tables of `mcs()` and `ranks()`, `DMEvaluator` and `MCSEvaluator`, now count the days from the first run date
  to the last. `n_days` is still the days with forecasts, and the verdict (30 days), the Newey-West lag and the bootstrap intervals read
  it. With a forecast on every day the two are equal and the message is word for word as before; a run of every day with holes warns a
  little less, and 104 Fridays span 722 days and don't warn. No number changes. See
  [Test length](/epftoolbox2/model-pipeline/comparing-models/#test-length).
- **`newest_target` on every model.** `OLSModel(..., newest_target="published")` (any model) trains each forecast on every target
  that the target's availability rule says is known by the issue time on D, period by period, instead of stopping at D-1: at 11:45
  Brussels that is D's day-ahead prices, and D's realised load up to 10:00. A row dated D or later never trains. It needs an
  information set with a rule for the target, at every level, and reads nothing else from it. `"d-1"` stays the default and
  **nothing moves without the keyword**: records, manifests and YAML of such a model are byte for byte what they were. A published
  model's records have `newest_target_day` (the newest target date of its training span, which needn't have trained), its manifest's `training_cut` names the target's rule and the issue time and zone (so
  changing them refuses a resume, and changing the level doesn't), `explain()` shows the newest target, and `verify()` takes the rule
  as its boundary. See [The newest target](/epftoolbox2/model-pipeline/time-and-calendars/#the-newest-target).
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
