---
title: Release Notes
description: What changed between releases, breaking changes first
---

Notes for changes that need something from you. The full list of changes of a release is in the commit log.

## Unreleased

### Breaking changes

- **A results file holds skip records, and every record ends with `status`, `list_hash`, `refit_anchor` and `revision`.** A key a
  run can't forecast is now a line too, with its reason as `status` (`"missing_test_predictors"`, `"not_published"`, ...), `prediction`
  null, the `actual` of its issue row and `detail` where the reason has one; a forecast has `status` `"ok"`, the hash of its predictor
  list (fixed lists too), the date of its fit (its run date without `refit`) and `revision` 1, after the keys it had, which keep their
  order and values. A script that reads the JSONL lines itself must keep the records with `status` `"ok"` or none.
  - **Nothing the library reads moves.** Tables, `forecast_counts()`, ensembles, exporters and resuming read the forecasts alone, so
    their numbers are bit for bit as before, and files written before resume and report as they did. A skipped key is tried again by
    every run, as before, and gets a new skip record only when its reason changed. Records held in memory are the forecasts alone.
    Manifests add `record_schema: 2`, provenance like `git_sha`, never compared. About 90 bytes more per record (60 with `refit`).
  - **One forecast per key.** A forecast repeated in a file with the same numbers is read once, with a warning, where a table used to
    count it twice; a key with two different forecasts raises `ValueError` naming the file and a key, in every table, ensemble and
    resume. The highest `revision` of a key wins, and a run writes at it.
  - **A half-written last line no longer stops a resume.** Reading skips it with a warning, and the next run cuts it off (a warning
    names its bytes) before appending; a damaged line before the last raises `ValueError` with its number.
  - **One writer per results file.** A run holds `<model>.lock` next to `<model>.jsonl` while it writes, and a second run into the
    same model's results raises `ResultsLockedError` naming it; where the file system refuses locks (an NFS mount without a lock
    daemon, say) the run warns and goes on, whatever the installed filelock.
  See [Incremental Training](/epftoolbox2/model-pipeline/caching/#results-file-format).
- **The week-ahead load forecast's rule follows each market's own week.** The built-in rule of `load_forecast_daily_min`/`_max`
  (ENTSO-E A31) read the delivery week in Brussels dates, while A31 holds one value per local date of each market's Monday-to-Sunday
  week. It is now `Availability.weekly("friday", "10:00", week_zone="data")`: the week of the data's local dates, published on
  Friday at 10:00 Brussels as before. `week_zone` is new on `Availability.weekly`: None, the default, keeps a rule's week in its own
  `timezone`, as every rule of yours had it, and `"data"` or a zone's name reads another zone's dates.
  - **What moves.** Nothing where the data's dates are Brussels's (Warsaw, Berlin, Paris and every other CET zone). On hourly and
    15-minute frames east of Brussels (Helsinki, Athens, Bucharest) the first hour of a Monday was taken as published a week early, a
    look-ahead: `NaiveModel(known_only=True)` loses those keys (12 of two January weeks in Helsinki at horizons 1 to 9), and a model
    that reads them is listed at `"warn"` and refused at `"raise"`. Daily frames there were read at a date's last slot and don't move.
    West of Brussels (Lisbon, Dublin), issued in the market's zone, the last hour of a Sunday, and a daily row of a Sunday, was taken
    as published a week late: keys
    called `not_published` come back (14 in the same two weeks), so on Fridays at noon the TSO has D+9 again, all 728 forecasts of a
    statistic in the [week-ahead study](/epftoolbox2/model-pipeline/week-ahead/).
  - **The source's rule publishes at 10:00 Brussels too.** `with_source_rules` (so every `Workflow`) gave an `EntsoeSource`'s A31
    columns Friday 10:00 on the bound market's clock: an hour early in an EET market, a look-ahead for an issue from 09:00 to 10:00
    Brussels on a Friday, and an hour late in Lisbon or Dublin. It is 10:00 Brussels now, for the week of that market's zone.
  - **Issue in the market's zone for an exact week.** On a frame in another zone than the issue time's (UTC, a fixed offset) the
    week is also read in the issue zone's dates and the later counts, so it is never early: exact when the issue zone is the market's,
    a week late on the last hour of a Sunday where neither is (a Lisbon frame issued on Brussels time, which keeps the old rule's
    verdicts there). The keys that come back west of Brussels need the issue in the market's zone, as in the week-ahead study.
  - **Resuming.** A `NaiveModel(known_only=True)` of these columns records the rules that decide its keys, so its results from before
    refuse to resume with `ManifestMismatchError` on `training_cut`. Use a new `save_dir`.
- **`training_window` is checked where it is set.** A float, a string other than `"expanding"`, a bool, None, or a value below 1 now
  raises `ValueError`. An int window ran before with any of these (a float such as `45.0` trained as 45 days and checked the whole frame);
  write the int.
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

- **A `Workflow` file's inline sources read a relative path from the file's directory.** `data_pipeline: sources:` in a Workflow
  YAML resolved `file_path` (or `path`) against the working directory, while the same source in a data pipeline file, and the model
  pipeline's `save_dir`, were read from the YAML's directory. They are all the YAML's now: a workflow run from another directory
  reads the same files.
- **Pool workers no longer start idle BLAS threads.** Limiting a forked worker to one BLAS thread started OpenBLAS's thread pool
  again in it (OpenBLAS stops the pool before a fork and starts it when its thread count is set): one thread per CPU for numpy's
  OpenBLAS and as many for scipy's, each spinning for about a tenth of a second before it slept, in every worker of every pool. On
  32 CPUs that was 62 threads and 3 to 4 s of CPU per worker, and a fit at a worker's start cost 5 to 50 times its wall time in
  CPU. Now your own process's OpenBLAS runs on one thread while a run's pools exist (from the first model to the last of
  `ModelPipeline.run` or `verify`, and so of a `Workflow`; one pool in a model's own `run`), the workers inherit that count and
  start no thread, and your counts come back after the last model, before ensembles, evaluators and exporters; on Windows, where
  the workers are spawned, the thread variables are also set to 1 for that time. A 766-forecast OLS, Ridge or LEAR run at 32
  processes took 44 to 58 core-seconds, now 6 to 9, of which 4 to 5 are your process's OpenBLAS starting its threads again,
  once, when it gets its count back (LassoCV 63 to 79, now 30 to 32); the same on numpy 2.5. Only an OpenBLAS with its own thread
  pool (numpy's and scipy's wheels) is held; other BLAS and OpenMP libraries start no thread in a worker and keep your settings.
  Every forecast is bit for bit as before, and `OMP_NUM_THREADS=1` before Python starts is no longer needed to avoid the cost.

- **Every `by_*()` table counts the forecasts of each model, and warns when the models of a group differ.** `by_hour()`, `by_horizon()`,
  `by_hour_horizon()`, `by_year()`, `by_year_horizon()`, `by_run_weekday_horizon()` and `by_target_weekday_horizon()` printed a metric per
  (model, group) and not how many forecasts it rests on, so a `NaiveModel(known_only=True)` benchmark at horizon 7 could show a MAE on 24
  forecasts next to an OLS model's on 56 with nothing in the table to say so. They now have the `forecasts` column of `summary()`, `by_quarter()`,
  `by_day()` and `by_price_level()`, after the group keys, and a `UserWarning` when the models of a group have different counts (a model with no row
  in a group counts 0), with `df.attrs["unequal_counts"]` listing those groups. See [Forecasts per group](/epftoolbox2/model-pipeline/results/#forecasts-per-group).
  - **What moves.** Only the shape of these tables, `compute_all()` and what the exporters write: every metric value is bit for bit as before.
    The tables have one more column (`model, <group columns>, forecasts, <metrics>`; `metric_columns()` does not list it), so code that
    picks columns by position must change, not code that picks them by name. `TerminalExporter` prints the column, and `ExcelExporter` writes a
    sheet of counts after the metric sheets of each breakdown (`horizon_forecasts`, `HourHorizon_forecasts`, ...), with 0 where a model has no
    forecast and no colours, which shifts the sheets after it. The `Summary` sheet's colour scale no longer covers the `forecasts` column, which
    it coloured like an error. `CsvExporter` is unchanged. `summary()` does not warn.

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
- **`CsvExporter(extra_columns=...)` wrote a forecast once per source row.** The columns are joined by target date and period, and
  only 15-minute and daily results checked the source. A quarter-hour source on hourly results wrote each forecast four times (96
  hourly forecasts became 388 rows), and even an hourly source repeated the hour that occurs twice on an autumn switch day, once per source row, so any
  count or Diebold-Mariano test run on the file was inflated. The source is now checked at every frequency, and one that is finer or
  coarser than the results raises `ValueError` (`extra_columns need source data at the resolution of the 1h results`), saying to
  resample or aggregate it first, where a coarser one used to leave most cells empty. The spacing is checked on the file's target dates
  only. The repeated hour takes the values of its second occurrence, the row a run forecasts from, at 1 h and at 15 min. A file whose
  results have no repeated hour and whose source has the results' resolution is the same, byte for byte, and so is every file written
  without `extra_columns`.

### New

- **`ForecastSource`: another pipeline's stored forecasts as an input.** `ForecastSource(path, model, column, horizon=1,
  available=None)` reads one horizon of the forecasts a `ModelPipeline` stored in `path` (JSONL or Parquet, bit for bit the same
  frame) and puts each at its target instant, the slot of its target date in the upstream's data zone (both rows of a repeated autumn
  slot, none of a skipped one; a daily run's value at its new `daily_stamp`, which every `freq="1D"` manifest records now, as
  provenance). Lag it like any other column: `solar_fc_d+1` is the forecast for D+1 issued on D. `with_source_rules` (so every
  `Workflow`) gives it the rule `published(issue time, days_before=horizon, timezone=issue zone, day_zone=data zone)` from the
  upstream's manifest and registers its upstream (`InformationSet.chained_inputs()`); that rule comes before every rule, assumption
  and pattern, and a rule naming the column is refused. A downstream model that reads it counts the rows its fits read (of the periods
  that read it) without a value of it, warns once per column at every level and refuses at `"raise"`, naming the upstream issue days
  that would cover them; an upstream that listed late inputs, or whose listing doesn't vouch for every forecast it stored (level
  `"none"`, `"raise"` with `unknown` other than `"raise"`, resumed under other rules or unverified, forecasts outside the test period
  it describes), makes its column `late upstream` (listed at `"warn"`, refused at `"raise"`). Its manifest records `chained_inputs`,
  and a resume refuses another upstream model, horizon, issue time or zone, data zone or `available`. A resume under another
  information set keeps the earlier listing under `availability["stored_under_other_rules"]`, for every model. The source refuses an upstream without an issue time or a named zone, live records (`issued_at`), and a horizon it
  lacks (`UpstreamError`). It has no per-source cache; a new `DataSource.content_digest()` hook makes the single-file cache fetch
  again after the upstream changed. See [ForecastSource](/epftoolbox2/data-sources/forecast/) and
  [Chaining Pipelines](/epftoolbox2/model-pipeline/chaining/). No existing record, listing or table moves; only daily manifests gain
  `daily_stamp`.
- **`day_zone` on `Availability.published` and `daily_close`.** The delivery day as a date of another zone than `timezone`: a zone's
  name, or `"data"` for the data's (the later of the data's and the issue zone's date on a frame in another zone), the publication
  instant staying `time` in `timezone`; validity periods read the same day. `None`, the default, is every rule as it was, and YAML
  writes it only when set. See [Information Set](/epftoolbox2/model-pipeline/information-set/#delivery-days-of-another-zone).
- **Generation targets: `normalise_by`, `bounds` and `zero_when` on every model that fits period or pooled units.**
  `normalise_by="capacity_d+{horizon}"` divides each training row's target by its value of the column (a row without a positive one
  drops out) and multiplies the forecast by the issue row's (a forecast without one is `missing_test_predictors`, with one at or
  below 0 the new skip reason `non_positive_scale`); `bounds=(low, high)` clips the forecast after that, in multiples of the scale
  with `normalise_by` (the renewables study's `(0, 1.05)`), in target units without, either side `None`;
  `zero_when="is_night_d+{horizon}"` makes a forecast 0 with no fit where the issue row's value is 1, its record `coefficients`
  `[]` and `detail` `"zero_when"`, decided after the skips of the training rows and of the issue row's inputs and before the scale
  and the fit (so a night with no capacity reported, or a fit that fails, still gives 0); a value other than 0, 1 or missing on an
  issue row raises. They are model parameters, so a benchmark beside the model stays as published; `NaiveModel` takes `bounds`
  alone. The columns are inputs the model reads, never regressors: `normalise_by`'s judged by the information set as a predictor
  (on the training rows too, for a rule over time), `zero_when`'s on the issue row alone, both listed at `"warn"` and refused at
  `"raise"` when late; hashed in the manifest, whose `model_params` record the three where set (compared on resume), and listed by
  `explain()`. `verify` perturbs the scale as a predictor and flips the flag's unknown values (0 to 1 and back), runs its copies
  without `bounds`, and doesn't ask a forecast of 0 to move under its control. A model with any of them never runs in an engine
  group. Day units, models that read series and models that fit nothing (bar `bounds`) refuse them, and they aren't supported on a
  `ResidualModel` or on the model a `PooledModel` pools. With all three `None`, the default, no record, manifest or table moves.
  See [Common Parameters](/epftoolbox2/models/common-parameters/#generation-targets).
- **What a sequence model reads, cut by the information set (internal).** The foundation models to come read series, not lagged
  columns: `epftoolbox2.models._history` gives each forecast the target's context of a fixed number of steps in physical time,
  ending at its newest value known at the issue time (at the end of D-1 at the latest under `newest_target="d-1"`, the default),
  its past and known-ahead covariates on the same steps (the known-ahead ones to the end of the target day), and the (date, period)
  each step ahead forecasts. The cut reads every value through the per-cell rule `verify`'s probe perturbs by, and `verify` runs
  such a model on the frame cut after each target day. Its `report.availability()` has one row per series it reads, with how far
  it is read and the series' vintage class in two more columns, `newest` and `vintage`, which only such a model's rows bring; the
  terminal exporter shows its series read in time in a table of their own. A resume compares every series' rule, the issue time
  and zone, and what the last issue day reads after `test_end`. Nothing of a model of predictors moves. See [Extending](/epftoolbox2/reference/extending/#sequence-models-series-instead-of-rows-internal).
- **The `[torch]` extra and `epftoolbox2.nets`.** `pip install 'epftoolbox2[torch]'` installs torch (`uv.lock` takes it from
  PyTorch's CPU index); `epftoolbox2.nets` has `MLP(n_inputs, n_outputs, hidden=(256, 128), dropout=0.0, activation="relu")`, a
  counter-based `Dropout(p)` (its mask a hash of the member's key, the step, the layer and the element), `Linear(in_features,
  out_features)` and `Activation(name)`, the layers the neural-network families to come stack. Their trainer is in place: members
  of a fit in a loop or in padded stacks, each member's numbers its own (alone, in a stack of any size or order, with other
  learning rates), early stopping on a random share of the member's own training days with the best weights restored (a member
  that never had a finite loss there counts as failed), TF32 off and torch's threads set for each fit and put back after it,
  torch's threads (and its bundled MKL's) pinned in each worker process, one per worker where the pool forks. Nothing of the core imports torch, and no
  record, manifest or table of a core model moves. See [Neural Networks](/epftoolbox2/models/neural-networks/).
- **`verify` checks determinism and what else is in a forecast's call.** `ModelPipeline.verify(..., rerun=None, co_batch=None,
  device=None)`: the rerun runs each sampled issue day twice first, each run from other seeds of the global generators, and a
  forecast that differs at all makes the model `inconclusive` (`nondeterministic`), never `ok`; on such a day a move under the
  probe is a `leak` only beyond 100 times the day's rerun difference (`NOISE_MARGIN`). The co-batch check runs the probe once more
  over runs widened to the other items of the call that predicts the issue day's forecasts (the other days of a refit block a
  model predicts in one call, or of a batch), with the probe's perturbation (a row after D read as of its own issue day, so a
  later row's `weather_cutoff` or `as_of` forecast is unknown), so a forecast that reads what it can't know through
  another item of its call is a `leak`, which the probe alone misses where the call is made of the run's own days. Both are on
  by default for the models that predict several items in one call (`_predicts_together`, or naming them with
  `_co_batch_items(run_day)`) or have a card, the model families to come, and off for every other. `device` sets an attribute
  named `device` on the models that have one (`report.devices`; a warning when none has). `epf verify` takes `--device`,
  `--rerun`/`--no-rerun` and `--co-batch`/`--no-co-batch`. `VerifyReport.table()` gains the columns `rerun` and `co_batch`,
  between `control moved` and `status`, empty for every model of today: **no status or other column moves**. `co_batch_probe`
  joins `leakage_probe` in `epftoolbox2.information_set`, and `leakage_probe` and `positive_control` take `span=`. See
  [Verifying a Pipeline](/epftoolbox2/model-pipeline/verification/#rerun-and-co-batch).
- **Several zones in one run, and `PooledModel`.** `ModelPipeline.run(data={"PL": df_pl, "FI": df_fi, ...})` runs every model once
  per zone on that zone's frame, in its own time zone, into `save_dir/<zone>/`, and returns a `PanelReport` (one `EvaluationReport`
  per zone, `summary()`, `forecast_counts()`, `choices()` and `availability()` with a `zone` column, `log_gain(benchmark, by=...)` and
  `cells_lost(benchmark)` with every model scored on the same cells and keys by default (`common=True`), `PanelReport.concat` for
  several targets); the run writes `save_dir/zone_run.json`, which alone makes `report(save_dir)` (and `epf report`) one of zones,
  refusing a zone whose folder is gone, and `verify` takes the dict too, each zone with its own positive control. A plain model in such a run is its run on the zone's frame alone, bit for bit. `issue_timezone="data"`
  issues each zone at the issue time on its own clock (on one frame, the frame's zone written out). `PooledModel(model, over=("zone",),
  zone_feature="onehot", missing="raise")` fits one model over every zone, each zone's rows standardised within its own fit, and
  judges every zone's training rows as of the earliest issue of the day; each zone's manifest records `pooled`, compared on resume.
  **No existing number moves**: a run of one frame, its records, manifests and tables are as before, and so are the goldens. See
  [Several Zones](/epftoolbox2/model-pipeline/zones/) and [PooledModel](/epftoolbox2/models/pooled/).
- **A model can predict a refit block in one call.** A model class that defines `_predict_block(fitted, rows)` (a prediction and a
  coefficient list per row) has each refit unit fitted once and predicted in one call: the issue rows of every day of the unit's block
  that the frame has with finite inputs, whichever days the run forecasts, padded to the rule's nominal length (7 rows for `"7D"`, 31
  for `"1MS"`) with copies of the first, each forecast taken from its own row. A forecast is then the same to the bit alone, in its
  block, from any `test_start` and after a resume. It is for the in-context models to come, whose call costs as much for one row as
  for a week of them; `block_predict=False` predicts one row at a time, and the manifest records which. Such a class gets
  `_predicts_together` and `_co_batch_items` (the other days of the block), so `verify`'s co-batch check runs on it by default,
  and that check also runs the issue day on the frame cut after its date and requires the same forecasts to the bit
  (`issue_day_cut`, `CutResult` in `epftoolbox2.information_set`): a forecast that depends on which later days its call holds
  is a `leak`. **No model of today defines it, and none of their numbers moves.** See [Extending](/epftoolbox2/reference/extending/#predicting-a-refit-block-in-one-call).
- **Day and pooled units: one fit of every period of a day at once.** A model class sets `_unit_kind = "day"` (rows are local dates,
  the outputs the P periods of the target day: `_fit_days(x, y, y_mask)` and `_predict_days(fitted, rows)`) or `"pooled"` (the
  (date, period) rows of every period, one output, with `period` a feature when a list names it: `_fit_model` and `_predict_block`,
  and `target_transform="per_period"` or `"pooled"`), for the neural networks and tree models to come. A unit is fitted on the refit
  calendar, with dates as rows and `training_span` as for any model (under `newest_target="published"` each period on the W dates
  of its own period unit); a slot with no target (the hour a spring switch skips, the repeated autumn hour) is masked and drops out
  of its period alone;
  each period's targets get their own Invariant (`epftoolbox2.scalers.target.MultiTargetAsinh`); and each unit predicts every issue
  day of its block that the frame has in one call of the block's nominal length, so a forecast is the same alone, in its block,
  after a resume and whatever the test start or the data's end. A day unit's predictors must hold one value per local date: a
  column that changes with the period (`price_d-1`) raises before any fit, naming its daily profile (`price_h00_d-1 …
  price_h23_d-1`), on every date the run's fits train on (an issue date's input is its last row's value; `verify` checks the whole
  frame). A list may change with the horizon and the issue day, not with the period. Manifests of such models record
  `unit_kind` and `target_transform`, compared on resume; `verify` runs the co-batch check and the cut frame on them by default.
  **No model of today is one, and none of their numbers, records or manifests moves.** See [Extending](/epftoolbox2/reference/extending/#day-and-pooled-units).
- **Model cards, weights and the pre-flight, for the families that need an extra or pretrained weights.** A family declares a
  `ModelCard` (licence, commercial use, revenue cap, attribution, corpus and its end, release, runtime packages with their pins, weight
  files with their sha256). `ModelPipeline.run`, each `Workflow` array task (before its data pipeline), `epf run` and `verify` (each
  model verified, before any probe) check every model first and raise one error naming every missing extra, runtime package outside its pin and missing or changed weight file
  (`MissingWeightsError` or `MissingExtraError`; `MissingExtraError` takes an optional message and pickles with it). Weights never
  download during a run: `epf weights fetch` downloads a card's files into `EPFTOOLBOX2_WEIGHTS_DIR` (default
  `~/.cache/epftoolbox2/weights`), verifies their sha256 and needs `--accept <licence>` for a non-commercial or revenue-capped
  licence; `epf weights list` and `epf weights verify` show and re-hash the cache. A carded model's manifest adds `card` (provenance)
  and `runtime` and `weights` (compared: another runtime version, commit, device type, thread count or weight hash refuses a resume).
  `report.provenance()` lists each carded model's licence, weights, runtime and the contamination verdict of each window (`clean`,
  `clean (released before)`, `possibly contaminated`, or the card's own), and the terminal, CSV (`<stem>.provenance.csv` next to its file) and Excel
  ("Provenance" sheet) exporters print it. `epftoolbox2.verify()` adds a table of extras and of the weights in the cache.
  `BaseModel._engine_ok = False` keeps a family out of the engine whatever it defines. **Nothing of the core models moves**: no
  shipped model has a card, so their records, manifests, tables and exporter files are as before. See
  [Licences and Weights](/epftoolbox2/models/licences-and-weights/).
- **Scoring by period: hourly and block aggregates, the shape within the hour, normalised MAEs, breakdowns by a data column.**
  `report.aggregate("1h", agg="mean")` turns 15-minute results into hourly ones (an hour counts with all four quarter-hours finite;
  dropped hours are `incomplete_hour` in its `forecast_counts()`), and `aggregate("1D", agg="peak")` / `"offpeak"` scores the mean over
  `peak_hours=(8, 20)` (local, the end left out) or the rest of the day, a block counting when every period of it the day's clock has
  is finite. `report.error_split()` divides a 15-minute model's squared error into the day's level, the hour profile and the shape
  within the hour. New evaluators: `ShapeMAEEvaluator()` (the error of the within-hour shape; NaN on hourly and daily results),
  `NMAEEvaluator()` (MAE over the mean actual on the same forecasts) and `ScaledMAEEvaluator(column)` (mean |error| over a source-data
  column such as capacity, with a `left out` count), all in YAML and the TUI. `report.by_column(column, bins=None)` groups forecasts by
  a column of the source data at the target row (a 0/1 flag such as `is_night`, or bins of a price), joined as
  `CsvExporter(extra_columns=)` joins, from a source in the results' time zone, which `CsvExporter(extra_columns=)` now checks too (a UTC frame under Europe/Berlin results gave every row another hour's value). A column an evaluator reads is checked before any fit, and `Workflow.report()` (`epf report`) loads the data for it. **No existing number moves**: the four daily aggregates, every table and the goldens are bit
  for bit as before. They reproduce the resolution study's R1 table (MAE15, MAEh, shape MAE, the error split), its DM list and R3's
  derived daily rows, and the renewables study's price and generation tables, from the studies' stored forecasts
  (`tests/reproductions/`). See [Working with Results](/epftoolbox2/model-pipeline/results/#hourly-aggregates-shape-and-the-error-split).
- **`ResidualModel` and `HistGBModel`: a base model's forecasts corrected by gradient boosting.** `ResidualModel(base, learner,
  training_window="expanding", refit="7D", warmup=None, name="residual")` runs its own copy of a base model (a model, or the name of
  one added to the pipeline before it) over a warm-up, stores its out-of-sample forecasts in `<name>.base.jsonl`, and fits the
  learner, `HistGBModel` by default (scikit-learn's histogram gradient boosting over the pooled (date, hour) rows, with the
  research's defaults), on their residuals; the learner's predictors may name the base's forecasts (`base_fc_d+{horizon}`,
  `base_fc_hHH_d+{horizon}`), and `presets.tree_predictors` gives the research's inputs. On the research's DE_LU and AT years a LEAR-BIC
  base with boosted residuals was about 4 % below LEAR's window average. A base that is a `ResidualModel`, a `PooledModel` or reads
  series is refused. Nothing of a model without it moves. See [Residual Model](/epftoolbox2/models/residual/).
- **`LightGBMModel`, `XGBoostModel` and `CatBoostModel`: gradient-boosted trees, one extra each.** `pip install
  'epftoolbox2[lightgbm]'` (or `[xgboost]`, `[catboost]`; `[gbm]` is the first two). Each fits one tree model per horizon and refit
  day over the pooled (date, hour) rows of every hour (`periods="pooled"`, the default) or one per hour (`"each"`), at the research's
  settings: absolute error, an expanding window, weekly refits. They build, save and load without their package, and the pre-flight
  names a missing one. The manifest compares the package's version, the thread count of a fit (`threads`, 1; more than one is
  refused where workers fork, on Linux and macOS) and XGBoost's `device`; a package's own error stops the run.
  Each is also a `ResidualModel` learner, whose learner stage now runs with its learner's processes. See
  [Gradient-Boosted Trees](/epftoolbox2/models/trees/).
- **Torch stacks: the networks of an `add_model` loop trained together.** `ModelPipeline.run` trains the torch models of a run
  whose fits are stacks on their device (`stack="vmap"`, or `"auto"` on a GPU) and that share a stack key (the module and its
  parameters but `dropout`, `output`, training window, refit, loss, batch size, epochs, patience, early stopping, weight decay,
  device, `tf32`, `pad_to`) in stacks of `pad_to` networks on one device, after the engine groups and whatever `engine`. Speed
  only: each model's records and manifest are its own run's to the bit, alone, in any loop, order or `models_per_task` split and
  after a resume, on a GPU for any module and on a CPU for `nets.MLP` (the presets) with the `"l1"` or `"mse"` loss, the only
  models a CPU stacks (a module's own `torch.sigmoid` or power can give a CPU stack's last slot other last bits); a module whose
  structure is drawn from torch's generator when built trains alone; a network that diverged makes its own fit's forecasts
  `fit_error`s and moves no other. Nothing moves for a model whose fits are a loop (`stack="auto"` on a CPU, the default there)
  or whose key is its own. `TORCH_STACKS=off` runs each model on its own, `TORCH_STACKS=check` also trains a sample of the
  stacked fits alone and records the comparison as `stack_check` in the manifest, and `TORCH_STACK_MEMBERS` sets the networks
  a job holds at least; `ModelPipeline.run` refuses either variable it can't read before any model writes. See
  [Neural Networks](/epftoolbox2/models/neural-networks/#many-networks-at-once-torch-stacks).
- **`TorchModel`, `MLPHourlyModel` and `MLPDailyModel`: neural networks on the library's trainer.** `TorchModel(module, ...)` trains
  a `torch.nn.Module` class of your own (flat predictors or named blocks), with one output per period (`output="period"`) or a day of
  outputs on day-level predictors (`output="day"`); `MLPHourlyModel` and `MLPDailyModel` are its presets with `epftoolbox2.nets.MLP`.
  A forecast is the mean of `seeds` members in target units, and is the same alone, in a loop, in a stack and after a resume. They
  build, save and load without torch and need the `[torch]` extra to fit (the pre-flight names it). The manifest compares the torch
  version, device, dtype, `tf32`, threads per fit, `stack` and `pad_to`. A module is saved by its import path and `module_hash`, a
  hash of its source file and its package's version; a network defined in a notebook, by `exec`, in a function or in the script being
  run has a `session-` token instead and resumes only within its session (`save()` raises for one from `__main__` or a function).
  `examples/mlp_search.py` is the default search. See [Neural Networks](/epftoolbox2/models/neural-networks/).
- **`ModelPipeline(engine="on")`: engine groups for studies of many models.** With `engine="on"` (`model_pipeline.engine: on` in a
  workflow) the models the engine takes, OLS, Ridge, LassoCV and LEAR, run in groups, one per refit rule: each group reads the
  data once for all its models, runs one pool of processes whose jobs are one day (or refit block) and period of every model, and
  writes every model's results into its own file as before, through one writer. A model's per-run cost beyond its fits drops (the
  pool start, the `gc.collect()` and the read of the data are once per group, and the checks of the frame and of the information
  set and each column's binary mask once for the models that share them): about 0.23 s per OLS model on its own, 0.005 s in a
  group of 1,000 (one test day, 24 periods). **OLS, Ridge, LassoCV and LEAR are fitted from exact sums shared across the group**,
  with `scaler="standard"` or a variance-stabilising scaler (`asinh`, LEAR's default, `npit`, `mlog`, `poly`: the scaler's own
  transform runs once per set of models with the same training rows, and every transformed column is the model's own to the bit),
  LassoCV and LEAR's `criterion="cv"` scored on the KFold blocks of those rows: the same bits alone and in any group, order or
  process count, within 1e-8 EUR/MWh of `"off"` for OLS, LassoCV and LEAR and 1e-9 for Ridge (the same penalty, the same kink),
  not to the bit, with the same counts of forecasts and skips; a fit the sums
  can't give to those digits is fitted as with `"off"`, to the bit, and so is the forecast of an issue row far outside its
  training values (more than 30 training standard deviations out for OLS and 10 for LEAR and LassoCV, along the eigenvectors of
  the fit's Gram; for OLS, whose normal equations are refined by one step of iterative refinement, also where the error that
  remains at the row, estimated per fit and row, is above 5e-9 EUR/MWh, 0.1 to 0.5 % more of OLS's forecasts on Lago's 247 DE
  inputs with a 364-day window; for Ridge, which has no
  distance bound, past scikit-learn's rounding budget at the row; with a
  variance-stabilising scaler also where its inverse is more than three times as steep as the median of its slopes at the
  training targets), a LEAR fit whose chosen columns are nearly collinear (their Gram's condition number above 1e6) and a
  LassoCV or LEAR fit with two near-copy inputs (1 − |correlation| at most 1e-5); within those bounds the largest gaps measured
  were 4.8e-9 (OLS), 4.6e-9 (LassoCV), 6.1e-10 (Ridge) and 1.1e-9 (LEAR) EUR/MWh. Their manifests record
  `fit_path: "engine-2"`, compared on resume, so results written with the engine on refuse a resume with it off and the other way
  round (`allow_manifest_mismatch=True` overrides), and as provenance how many fits came from the sums; `"engine-1"` results
  of a development build (before OLS's refinement step) are refused too, for every class, although only OLS's numbers moved,
  so for LEAR, Ridge and LassoCV `allow_manifest_mismatch=True` resumes them safely. LEAR's `solver="cd"`
  (`LEARModel.lago()`'s) is fitted as with `"off"`, to the bit. **`engine="check"`** is `"on"` with one fit in about 50 also
  fitted as `"off"` fits it: the results are `"on"`'s, the manifest records the gaps as provenance (`engine_check`: fits, largest
  absolute and relative gap, gaps that aren't finite, how many chose another kink or penalty, fits `"off"` couldn't fit), and a
  warning names a model whose largest gap is above its class's tolerance or that has either of the last two. On 1,000 LEAR
  variants (asinh) the engine spends 1.45-1.54 times fewer core-seconds than `"off"` where the inputs have no gaps, and runs at
  0.84-0.98 times its speed (2-19 % more core-seconds) where every variant's gaps give it rows of its own; LassoCV spends 1.64
  times fewer; a run's wall time is 5-14 times shorter. `NaiveModel`, a `train_on="matching"` list, a subclass with its own fit and a `forecast_only` run go through their own run as
  with `"off"`, the default, which runs nothing new. `ENGINE_MAX_MODELS` (20,000)
  and `ENGINE_MAX_COLUMNS` (4,096) bound how many models and columns a group reads at once; `verify` runs each model through the
  engine too. YAML writes `engine` only when it is `"on"` or `"check"`. At scale (10,000 OLS, 1,000 Ridge, 1,000 LEAR and 200 LassoCV
  variants of a gappy 40-column weather union, 24 periods, 7 issue days) every model's records, manifest and `epf report` tables
  are the same bytes over 64 array tasks, at 1 and 12 processes and alone, and `binary_mask="fit"` keeps every shared forecast of
  two test starts; the 10,000 OLS variants take 7.2 minutes on 12 cores, against about 48 with `"off"`. See
  [The Engine](/epftoolbox2/model-pipeline/engine/), a page of its own now.
- **`PseSource`: the Polish TSO's reports API, no key.** `PseSource(datasets=["kse-load", "csdac-pln"], columns=None, prefix="pse")` loads
  `kse-load` (`load_actual`, `load_fcst`), `his-wlk-cal` (`demand`, `wi`, `pv`, `jg` and more on request), `csdac-pln` (the SDAC price in
  PLN), `rce-pln` (the prosumer net-billing price, another series), `pdgsz` (recommended-usage code, the version stamped by
  `pdgsz_cutoff` on D-1, NaN where none was) and `przeplywy-mocy` (`flow_*`) on the 15-minute grid, UTC start of the period (PSE
  stamps the end), every column prefixed `pse_` by default. It loads from YAML, is offered in the TUI and has built-in availability
  rules (`load_fcst` 10:00 and the prices 15:00 on D-1, `usage_fcst` at the cutoff, the actuals 13:00 on D+1) with the classes
  `snapshot` (prices, `pdgsz`) and `last_write_stamp`. Nothing existing moves. See [PseSource](/epftoolbox2/data-sources/pse/).
- **`EntsoeSource(price_exaa=True)` adds the EXAA auction; CH prices are the Swiss auction.** For AT and DE_LU the new column
  `price_exaa` (A44, `classificationSequence` position 2) comes beside `price` on the same grid; other zones raise. Its built-in rule
  is `published("11:15", 1)` in Vienna or Berlin and the Swiss `price` gets `published("12:00", 1)` in Zurich, both the legal
  bounds and `snapshot`s (first-seen times unmeasured). A source without `price_exaa` returns and requests exactly what it did, and its
  cache key is unchanged. The [ENTSO-E page](/epftoolbox2/data-sources/entsoe/#early-auctions-exaa-and-switzerland) shows how
  to keep a model that works when the column is missing.
- **`EnergyChartsSource`: Energy-Charts (Fraunhofer ISE) without a key.** `EnergyChartsSource("de", type=["forecast", "power", "price"])`
  reads the TSOs' solar, wind onshore, wind offshore and load forecasts by vintage (`ec_solar_forecast_day_ahead`, `_intraday`, `_current`; every column is prefixed `ec_` by default so none takes another source's built-in rule),
  the actual generation and load by type (`actual_*`), the day-ahead price (`price_day_ahead`) and the cross-border flows (`cbet_*`,
  `cbpf_*`). Timestamps are the API's start-of-period unix seconds, so both DST days come out exact at 15 minutes and 1 hour. Built-in
  availability rules: `*_forecast_day_ahead` published 18:00 on D-1, `*_forecast_intraday` 08:00 on the delivery day (both `latest_revision`), `actual_*` 1 h
  after the period, `price_day_ahead` 13:00 on D-1 (CH 12:00), and `*_forecast_current`, filled in after the day, has no rule that
  holds on any day, so `availability="raise"` refuses it at every shift. The vintages are matched to ENTSO-E's by name and error ordering, not
  documentation. Data: Energy-Charts.info, CC BY 4.0, attribution required; the prices of zones outside the CC BY list (AT, BE, CH, CZ,
  DE-LU, DE-AT-LU, DK1, DK2, FR, HU, IT-North, NL, NO2, PL, SE4, SI) are for private use, and the source refuses them unless
  `allow_restricted_prices=True`. The API allows 2 requests a minute per IP and endpoint: the source throttles itself, waits
  `Retry-After` and caches on disk. See [EnergyChartsSource](/epftoolbox2/data-sources/energy-charts/).
- **`binary_mask=`: binary columns read on each fit's own rows, so no forecast depends on `test_start`.** Every model that fits takes
  `binary_mask=None | "first_day" | "fit"`. `None`, the default, is today's rule: without `refit` a list's binary 0/1 columns (left
  unscaled) are read on the window's first day and kept for the window, so a column that is 0/1 there and not later (night-time
  solar at 06:00 from a winter start) makes a forecast depend on where the test starts; with `refit` each unit reads its own rows.
  `"first_day"` names the first and raises with `refit`. `"fit"` reads each fit's own training rows, before any row is dropped, as a
  refit unit does: its records are `refit="1D"`'s, and runs from any start share their forecasts bit for bit. It moves only forecasts
  whose own rows disagree with the first day's (none of 2,928 on Lago's DE data with OLS, LassoCV or LEAR; on the Polish week-ahead
  Fridays none at the study's W = 1092, and 78 and 101 of 301 of a 91-day LassoCV, whose cooling degree days are 0 in January's
  rows). The manifest's `training_cut` records `"binary_mask": "fit"` without
  `refit`, so a resume under the other rule refuses; YAML writes it only when set. **Nothing moves without it.** See
  [Time and Calendars](/epftoolbox2/model-pipeline/time-and-calendars/#binary-columns-and-test_start).
- **Parquet results: `run(save_format="parquet")`, `save_coefficients=False` and `epf results`.** With `save_format="parquet"`
  (`model_pipeline.save_format: parquet` in a workflow) each model's results are a directory of zstd Parquet parts,
  `save_dir/<model>.parquet/`, with a catalog of the committed parts, in place of `<model>.jsonl`: the same records (a record
  converts both ways exactly, keys, their order, nulls and float bits), read by the same rules, so every table and count is the same
  to the last bit, and resuming, skip records, the lock and array tasks work as with JSONL. A crash leaves at most an orphan part that
  no reader sees; a write that fails raises, and its keys are forecast again. Part files differ between identical runs (how
  records split into parts depends on timing), the records don't. `save_coefficients=False` writes `coefficients` as `[]` and
  changes no other key; the manifest records it (`save_coefficients: false`) and a resume with the other setting raises
  `ManifestMismatchError`. A writer of many models names its own lock file in each model's lock file, which an older epftoolbox2
  doesn't read: don't write one save_dir with two library versions at once. `epf results convert SAVE_DIR
  --to parquet|jsonl` converts a save_dir record for record (JSONL to Parquet and back gives the same lines), `epf results compact`
  merges each (model, year)'s parts, and `epf results verify` checks catalogs, part checksums, orphans and repeated keys. **Nothing
  moves without it**: JSONL stays the default and every file is as before; a run in the other format than a model's results on disk
  raises before any fit, naming `epf results convert`. The manifest of Parquet results records `save_format: "parquet"`
  (provenance). See [Parquet Results](/epftoolbox2/model-pipeline/caching/#parquet-results).
- **`MirrorLogScaler` and `PolyScaler`: the mirror-log and polynomial VSTs.** `scaler="mlog"` and `scaler="poly"` are the
  transformations of Uniejewski, Weron & Ziel (2018) on `AsinhScaler`'s median/MAD normalisation, then standardised like every VST:
  `sgn(u) (log(|u| + 1/c) + log c)` with `c=1/3`, and `sgn(u) ((|u| + (c/λ)^(1/(λ-1)))^λ - (c/λ)^(λ/(λ-1)))` with `lam=0.125`,
  `c=0.05` (keyword-only; at their defaults left out of the scaler's parameters, so `"mlog"` is the spec of the default). Both are odd,
  strictly increasing and invertible for every value; they take `apply_to` and `standardise` as the other VSTs do, and the same
  near-constant rule. They are two of the four scalers of the recommended LEAR. **Nothing else moves**: `AsinhScaler` and every other
  scaler give the same numbers, bit for bit. See [Scalers and VSTs](/epftoolbox2/models/scalers/).
- **The recommended LEAR, `examples/lear_lago.py` and the benchmark page.** The LEAR page now documents two settings: `LEARModel.lago()`,
  the paper's, and the recommended one, four `LEARModel`s on `[56, 84, 728, "expanding"]` with BIC and `newest_target="published"`
  that differ only in their scaler (Lago's Invariant asinh, mirror-log, polynomial, N-PIT), averaged by an `AverageEnsemble`, with the evidence for
  each setting. `examples/lear_lago.py` runs both on a benchmark market (`main(market, test_start, test_end, output_dir)`; no API key,
  the data come from Zenodo). [Benchmark Reproduction](/epftoolbox2/models/lear-benchmark/) holds every number measured on the
  benchmark, each on identical forecast keys with their counts: the original epftoolbox, the published forecasts, the criteria and
  solvers, the newest target, `refit="auto"`, blocked CV, Ridge, and the recommended settings and the defaults against `lago()`.
- **`LEARModel`: LEAR, and `LEARModel.lago()`, Lago et al. (2021) exactly.** `LEARModel(predictors, training_window=[56, 84, 728,
  "expanding"], criterion="bic", solver="lars", *, scaler="asinh", name="LEAR", newest_target="d-1", refit=None)` fits a Lasso per
  period whose penalty is chosen on every fit as old scikit-learn `LassoLarsIC(normalize=True)` chose it, AIC or BIC on the LARS path of
  unit-norm columns, then refits it on the columns as they are (Lago's refit), exactly (`"lars"`) or by Lago's coordinate descent of at
  most 2500 sweeps (`"cd"`). Its kernels follow scikit-learn 0.24.2, which the paper ran, and run in numpy or, with the new `[fast]`
  extra (`pip install 'epftoolbox2[fast]'`, numba), compiled (`EPFTOOLBOX2_KERNELS=auto|numpy|numba`); exact twin columns (copies, exact
  negatives, a daily value on 24 slots) are fitted as one, so the two backends agree within 1e-9 with the same kinks; a manifest
  records which ran as provenance (`kernels`, a list once a resume used the other), and a resume warns while a file mixes them. `LEARModel.lago(exogenous)` is the paper's
  model: AIC, `"cd"`, the Invariant scaler, the windows `[49, 77, 1085, 1449]` (the paper's 56 to 1456 days less the first week),
  `newest_target="published"` and its inputs in the original's order; on the first DE and BE test days it is within 1e-9 EUR/MWh of the
  original epftoolbox. Two options come with it: `AsinhScaler(standardise=False)` (on every VST scaler) is epftoolbox's Invariant
  scaler, and `lear_predictors(..., order="lago")` lists the same 247 inputs in the original's order. **Nothing moves for the other
  models or for the defaults of those two**: no existing manifest, scaler spec, YAML file or golden number changes. See
  [LEARModel](/epftoolbox2/models/lear/).
- **`LEARModel(criterion="cv")`: the penalty by blocked cross-validation.** Old scikit-learn `LassoLarsCV(cv=5,
  normalize=True, max_iter=2500)`: five contiguous blocks of the training rows in date order, each fold's LARS path on its own
  centred, normalised training part, α\* the first minimum of the folds' mean held-out squared error over their kinks (at most
  about 1,000), and the coefficients of the full window's path at α\*, with no refit. It takes `solver="lars"`; `"cd"` with it
  raises. On the first DE and BE test days of `lago()`'s setup it is 0.24.2's `LassoLarsCV` within 1e-9 in α\* and every
  coefficient. Over the paper's test periods the average of its four windows is within −0.1 to +0.9 % of the MAE of `"bic"` with
  Lago's refit (one-sided DM, unadjusted: BIC better in NP and PJM), at 1.7 to 1.9 times the cost of a run of `"bic"` with the
  default `solver="lars"` (1.5 to 1.7 against `"cd"`). It differs from `"bic"` in the penalty and in the coefficients: with the
  path's coefficients for both, CV's penalty has a 9 to 25 % lower MAE than BIC's; with BIC's penalty, the refit a 9 to 26 %
  lower MAE than the path. Each fold drops the exact twins and the columns constant up to rounding of its own training rows, which
  0.24.2 didn't; its blocks are cut by rows, as 0.24.2's were, so a doubled autumn hour can straddle a block edge (a known issue,
  about 0.05 % of hourly keys). **Nothing else moves**: `"aic"` and `"bic"` fit as before, bit for bit. See
  [LEARModel](/epftoolbox2/models/lear/#blocked-cross-validation).
- **`RidgeModel`: ridge regression with a leave-one-out penalty.** `RidgeModel(predictors, training_window=365, alphas=None, *,
  fit_intercept=True)` is scikit-learn's `RidgeCV` with `cv=None`: on every fit it chooses the penalty from a grid by the closed-form
  leave-one-out error of the training rows and fits with it, on the same standardised rows as `OLSModel` and `LassoCVModel`. `alphas=None`
  is `np.logspace(-3, 4, 36)`; the manifest records the grid in full (so a resume with another one is refused) and YAML writes `alphas` only
  when set. It takes `scaler`, `newest_target` and `refit` like the other models, a penalty below 1e-6 raises (scikit-learn's leave-one-out route loses digits as 1/penalty; use `OLSModel` for an unpenalised
  fit), and a fit needs at least 3 training rows (fewer is a counted `fit_error` skip). **Nothing moves for the other models**: no existing manifest, file or golden number changes. See
  [RidgeModel](/epftoolbox2/models/ridge/).
- **Training windows: days, `"expanding"`, or a list averaged inside the model.** `LassoCVModel(..., training_window=[56, 84, 728,
  "expanding"])` fits one member per window, each on its own rows with its own scaler and binary columns, exactly the model with that
  window alone, and forecasts their mean in target units (`math.fsum(...) / M`); a key is forecast only when every member is, else it is
  skipped with the first failing member's reason and window (`no_finite_training_rows: window 56`). Records add `member_predictions`
  and have `coefficients: []`. `"expanding"` trains on every row from the frame's first local date, which the manifest records as
  `training_cut["expanding_from"]`, and the data checks then read the whole frame. A list is sorted, and `[W]` is `W` in records,
  manifests and YAML, so **nothing moves for a single window**. Not yet with `train_on="matching"`, which raises. See
  [Training Windows](/epftoolbox2/model-pipeline/training-windows/).
- **`NaiveModel`: a published forecast as a benchmark.** `NaiveModel(predictors=["load_forecast_daily_max_d+{horizon}"], name="TSO")`
  forecasts that column's value on the row of the issue day: no fit, scaler or training rows, no coefficients, so a forecast needs no
  history. `{horizon}`, callables and functions of `ctx` work; more than one name raises, and so do `refit`, `newest_target` and a
  matching list. With an information set, `known_only=True` (the default) makes a value its rule puts after the issue time no
  forecast: counted as `not_published` in `forecast_counts()`, neither listed nor refused, at every availability level and in `verify`.
  The issue time and the column's rule are recorded in the manifest's `training_cut`, so a resume with others refuses. See
  [NaiveModel](/epftoolbox2/models/naive/).
- **Estimator options.** `LassoCVModel(fit_intercept=True)`, `OLSModel(fit_intercept=False)` and `StandardScaler(scale_binary=False)`, all
  keyword-only, whose defaults are what the models did before: **nothing moves without them**, and a default isn't written to the
  manifest or to YAML, so results and files made before keep resuming and loading as they were. `scale_binary=True` standardises the
  binary 0/1 columns too. A published week-ahead study needs all three; see
  [What the study needs](/epftoolbox2/model-pipeline/week-ahead/#what-the-study-needs).
- **Week-ahead forecasting, documented.** A new page builds the daily minimum and maximum load of D+3 to D+9 on Fridays at noon, against
  the TSO's A31 forecast, and the daily form for D+1 to D+7, whose lists drop the TSO's forecast where it isn't published yet. With the
  study's settings the library refits a published 20-market study's forecasts within 0.0005 MW (the stored rounding), and its list's
  `load_actual_h-1`, known at 13:00, is late at a noon issue. See [Week-ahead Forecasting](/epftoolbox2/model-pipeline/week-ahead/).
- **Vintage classes and rules over time.** `Availability(..., vintage=...)` (and `vintage=` on every constructor) says what the archive
  of a column holds: `snapshot`, `row_stamp`, `last_write_stamp` or `latest_revision`. The built-in ENTSO-E load and generation rules are
  `latest_revision` and `price` and Open-Meteo's are `snapshot`; a rule of yours has none unless you say. A class moves no number:
  `report.vintages()` lists every input's class, a run records them in its manifest as `vintages` (never compared), the terminal
  exporter prints the revised ones, and `tests()`, `compare()`, `leaderboard()` and a DM column warn when compared models read one
  series with different classes (`df.attrs["vintages"]`). `verify` calls a predictor function `inconclusive` when a
  `train_on="matching"` list reads a `latest_revision` or `last_write_stamp` column on a day it trains on, naming it: a revised value
  can move the set of days and no probe sees it. `.during(valid_from=, valid_until=)` gives a rule the days it holds on, and a list of
  rules with disjoint periods is a column's rule over time, for a lag that changed (observed: PSE's settlement prices went from D+4 to
  D+1 on business date 8 October 2026; the library has no PSE source yet). For such a column the check also reads the training rows
  (with `refit`, those of the refit day's fit) and every slot of a day aggregate, since an older value can come after a newer one;
  `check()` and `late_uses()` take them as `training_rows=TrainingRows(...)` and warn without; a model passes the frame's first date
  (`first_day`), before which no row exists or is judged, so a rule that starts on the frame's first date passes whatever the window.
  A value no rule holds for is never known. The check, `ctx`, `explain`, the probes and `newest_target="published"`
  all read it the same way. **Nothing moves without them**: records and tables are as before, and YAML writes the new keys only when
  set, so an old file is written back byte for byte. What moves: manifests of runs that read an input with a built-in class gain
  `vintages` (an ensemble's manifest doesn't), and a matching function reading a revised built-in column on a training day goes from
  `ok` to `inconclusive`. See [Vintage Classes](/epftoolbox2/model-pipeline/information-set/#vintage-classes) and
  [Rules Over Time](/epftoolbox2/model-pipeline/information-set/#rules-over-time).
- **`refit=`: fitting on a fixed calendar.** `OLSModel(..., refit="7D")` and `LassoCVModel(..., refit="1MS")` fit once per unit (a
  period, a horizon, a refit day and a predictor list) instead of once per forecast: `"kD"` every k days, `"kMS"` on the first of every
  k-th month, counted from Monday 1970-01-05, so a forecast's refit day depends on its issue day alone, never on where a run starts. A
  forecast issued on D trains on the rows of the forecast issued on its refit day a (dated a-W-h to a-1-h; with
  `newest_target="published"` the newest target known at a's issue time), finds its binary columns on those rows, and is predicted from
  its own row on its own, so it is bit for bit the same whichever run, process count or resume made it. Records gain `refit_anchor`,
  the manifest's `training_cut` gains `refit` and `refit_epoch` (changing `refit` refuses a resume), resume compares each stored
  anchor, and YAML writes `refit` only when set. A frame that doesn't hold W+h days before the refit day of the first issue day, and the predictors' deepest lag before
  that (`price_d-6` 6 days, an hour lag a day: else the first training rows miss their lags and drop out), raises `ValueError` naming
  the day it must start on (one sized for `refit=None` is too short whenever the test doesn't start on a refit day), and a run uses
  at most one process per refit day. Other values raise, and so does `refit` on a custom model with only
  `_fit_predict`, or whose `_fit_predict`, `_fit_model` and `_predict_fitted` aren't all defined in one class (a subclass that
  overrides some of them would fit one model with `refit=None` and another with `refit`). `verify`, `leakage_probe` and `positive_control` run a refit model from the issue day to the end of its block,
  so a leak that needs several days of a block shows, and cost that many more days of forecasts. **Nothing moves without the keyword**: `refit=None` is every record, manifest and table as before. With it the
  numbers move, since fits are older: on Lago DE in 2017 at horizon 1 (LassoCV on the LEAR inputs, W = 364, the same 8712
  forecasts), MAE was 4.6665 refitting daily, 4.6560 (-0.23 %) with `"7D"` from 1248 fits and 4.7134 (+1.00 %) with `"1MS"` from 288,
  with naive standard errors of the daily difference of 0.6 % and 1.4 % of the MAE; forecasts issued on a refit day equal the daily
  refit's bit for bit. See
  [Refitting on a calendar](/epftoolbox2/model-pipeline/time-and-calendars/#refitting-on-a-calendar).
- **`refit="auto"`: a refit rule per training window.** `LEARModel(..., training_window=[56, 84, 728, "expanding"], refit="auto")`
  (any model that takes `refit`) fits the windows of up to 84 days on every issue day and the longer ones and `"expanding"` on Mondays,
  each exactly as the single-window model with `refit="1D"` or `"7D"`, and forecasts their mean; a single window under `"auto"` is
  its rule in every record. A record's `refit_anchor` is the Monday, the oldest fit it uses, and `newest_target_day` the newest of its
  windows'; `training_cut` gains `refit_rules`, the rule of each window, so a resume with another `refit` is refused. A worker runs a
  Monday-to-Sunday block, the frame must reach back from each window's own refit day (a long window asks up to 6 days more than with
  `refit=None`, and the error names it), the training target is checked on every window's refit days, and a column whose rule changes
  over time on the rows of every window's own fit. **Nothing moves for `None`, `"kD"` or `"kMS"`**: records, manifests and YAML are as
  before. On `LEARModel.lago()` with `newest_target="d-1"` over DE (Lago's test period without its first day: 727 issue days, the same
  17,448 forecasts) the window average's MAE went from 3.6224 to 3.6252 (+0.08 %, DM p 0.09) at 57 % of the fits and about 40 % of the
  compute. See
  [`refit="auto"`](/epftoolbox2/model-pipeline/time-and-calendars/#refitauto).
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
