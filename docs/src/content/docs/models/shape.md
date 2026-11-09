---
title: Shape Model
description: A stored hourly forecast plus a learned within-hour shape for 15-minute targets (ShapeModel), trained on out-of-sample hourly forecasts
---

`ShapeModel(base, learner)` forecasts a 15-minute target as an **hourly forecast plus each quarter-hour's deviation from its hour's
mean**. The hourly forecast is another pipeline's, stored and read back into the 15-minute frame (a
[chained column](/epftoolbox2/model-pipeline/chaining/)); the learner forecasts the deviation. On the resolution study's DE_LU
and AT prices (R1: April to October 2026, 182-day window, daily refit) this structure, 24 hourly models plus a model of the
within-hour shape, tied 96 per-quarter-hour models (DM p .16 to .75) with the best shape error, at a quarter of the fits.

## Basic Usage

Two pipelines: the hourly one forecasts the hourly mean price and stores its forecasts over the training years and the test year;
the 15-minute one reads them, steps each hour's value over its quarter-hours, and fits the shape on them.

```python
from epftoolbox2.data.sources import CsvSource, ForecastSource
from epftoolbox2.data.transformers import LagTransformer, PeriodStatsTransformer, ResampleTransformer
from epftoolbox2.models import LEARModel, OLSModel, ShapeModel
from epftoolbox2.pipelines import DataPipeline, ModelPipeline

# 1. Hourly: 24 LEARs on the hourly mean price, issued at 11:45 Berlin, stored over the shape model's training rows too.
hourly = ModelPipeline(issue_time="11:45", issue_timezone="Europe/Berlin")
hourly.add_model(LEARModel(["price_d-1", "load_forecast_d+{horizon}"], training_window=728, name="LEAR_1h"))
hourly.run(df_1h, test_start="2025-09-30", test_end="2026-10-11", horizon=1, save_dir="runs/de_1h")

# 2. The 15-minute frame: the hourly forecast stepped over its quarter-hours, its D+1 and the hour before it (the ramp), and
#    the load forecast's within-hour deviation.
data15 = (
    DataPipeline(timezone="Europe/Berlin")
    .add_source(CsvSource("de_15min.csv"))
    .add_source(ForecastSource("runs/de_1h", model="LEAR_1h", column="price_1h_fc"))
    .add_transformer(ResampleTransformer(freq="15min"))
    .add_transformer(PeriodStatsTransformer(columns="load_forecast", stats=["dev"]))
    .add_transformer(LagTransformer(["price_1h_fc", "load_forecast_hour_dev"], [-1], "day"))
    .add_transformer(LagTransformer(["price_1h_fc_d+1"], [1], "hour"))
)
df15 = data15.run("2025-09-30", "2026-10-12")

# 3. The shape: per quarter-hour, an OLS of the deviation on the forecast hourly ramp and the load forecast's deviation.
shape_x = ["price_1h_fc_d+{horizon}", "price_1h_fc_d+{horizon}_h-1", "load_forecast_hour_dev_d+{horizon}"]
shape = ModelPipeline(issue_time="11:45", issue_timezone="Europe/Berlin")
shape.information_set = shape.information_set.with_source_rules(data15.sources)
shape.add_model(
    ShapeModel(
        base="price_1h_fc_d+{horizon}",
        learner=OLSModel(shape_x, training_window=182, fit_intercept=True),
        name="B24_dyn",
    )
)
report = shape.run(df15, test_start="2026-04-02", test_end="2026-10-11", horizon=1, freq="15min", save_dir="runs/de_15")
```

The records are quarter-hour price forecasts, on the keys of any 15-minute model of the run, so a 96-model run beside it
(`LEARModel` per quarter-hour on the same frame) compares on identical keys with `report.tests()`, and `report.aggregate("1h")`
and `ShapeMAEEvaluator` score the hourly level and the shape apart. The [chaining page](/epftoolbox2/model-pipeline/chaining/#a-15-minute-shape-on-an-hourly-forecast)
shows the whole chain with its checks.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base` | str | Required | The column of the hourly forecast on every quarter-hour of its hour, `{horizon}` allowed: a chained column, such as `price_1h_fc_d+{horizon}` from a `ForecastSource` stepped by `ResampleTransformer` and lagged by `LagTransformer`. Read on the issue row only |
| `learner` | model | Required | The model of the deviation, such as `OLSModel(shape_x, training_window=182, fit_intercept=True)`: its estimator, predictors (a list of names, `{horizon}` allowed), `training_window`, `refit`, `scaler`, `newest_target` and `binary_mask` |
| `period` | str | `"1h"` | Keyword-only. The period whose mean the shape is taken from: the hour |
| `native` | str or date | `"distinct"` | Keyword-only. Which target dates train: `"distinct"`, a date where some hour has quarter-hours that aren't all equal; or the first natively 15-minute date, such as `"2025-10-01"`, and the dates after it |
| `min_native_days` | int or None | `None` | Keyword-only. The fewest native target dates a fit may train on, counted in its window from the first native date on: a forecast whose fit has fewer is skipped as `too_few_native_days`. `None` is each window's own length (an `"expanding"` window has no minimum); `0` skips nothing |
| `name` | str | `"Shape"` | Keyword-only. The model's name, which names its results |

The learner's settings are the model's: `shape.training_window`, `shape.refit` and the others read the learner's, and are set on
the learner. A ShapeModel runs at `freq="15min"` only.

## What the learner trains on

On a training row the learner's target is **y − ȳ**: the row's target less the mean of the target over the four quarter-hours of
its hour on its target date. That mean is built like the target, by the calendar shift of `{target}_d+{h}`, so it reads no date
the training cut excludes; it is NaN unless all four quarter-hours have a value, and the row then drops. The hour repeated in autumn
is two hours of four, as in [PeriodStatsTransformer](/epftoolbox2/transformers/period-stats/); its target on the autumn day is
missing, as for any model, so no training row reads it.

A row whose target date isn't **natively 15-minute** drops too. Before the 2025-10-01 switch the day-ahead market cleared hours,
and a 15-minute series of that history (`EntsoeSource(prefer_15min=True)`) holds each hour's price on its four quarter-hours: a
deviation of 0 that would teach the learner that hours are flat (the study's "shape dilution", R2). `native="distinct"` keeps a
date where some hour's four quarter-hours aren't all equal, which a step-filled date never has; `native="2025-10-01"` keeps that
date and the later ones. The hourly model can still train on the whole history: only the shape is restricted.

A fit whose window reaches before the first native date (`native`'s date, or the frame's first native date with `"distinct"`)
would train on the few native days inside it: a fit of 91 days issued a week after the switch, on 7 rows. By default
(`min_native_days=None`) such a forecast is skipped as `too_few_native_days`, decided before any fit and counted in
`forecast_counts()`, so a short-trained shape never enters a comparison; one warning per run names the issue days. Give
`min_native_days=` the fewest native days a fit may train on to forecast them anyway (the skip record's `detail` says how many a
fit had), or start the test a window after the switch. A native date missing later on (a day stored hourly) drops its rows but
doesn't count against the minimum.

## The forecast

The forecast is the issue row's `base` plus the learner's forecast of the deviation. A missing `base` on the issue row skips the
forecast as `missing_test_predictors`. A learner with no predictor and an intercept forecasts the base plus the mean deviation of its
quarter-hour over its window (the study's B24_static): `LEARModel([], training_window=91, scaler="standard")` is one, as LEAR
forecasts the training mean of a fit of no column.

## Why the base must be chained

The learner's regressors are built from the base (the forecast hourly ramp above). If they held the actual hourly prices on the
training rows and forecasts on the issue row, the learner would weigh a ramp it never sees in use: the resolution study trained
B24_dyn so, and its audit found the mismatch. With a chained base the training rows hold the hourly model's out-of-sample
forecasts, issued as of each row's date, and the issue row holds the forecast of the issue day: the same thing. Run the hourly
pipeline over the shape model's training rows too: from W+2 days before the first issue day, W the learner's window (the rows of
a forecast issued on D are dated from D−W−1 at horizon 1, and the ramp's hour lag reads the day before at midnight). The coverage
check counts the rows that miss it, those whose target date isn't native left out, as the fits drop them anyway.

A base that isn't a chained column (`InformationSet.chained_of`, which `with_source_rules` fills) warns, and so does any base of
a run without an information set, which can't tell. A chained base is checked as
a chained input: an hourly upstream that read something late makes the base `late upstream`, listed at `availability="warn"` and
refused at `"raise"`, and the manifest's `chained_inputs` records its upstream. The base is also judged by the information set on the
issue row: a base known after the issue time, such as the actual hour mean 24 hours ahead, is refused at `"raise"`; the target's
actual hour mean of a later date by its name, `{target}_hour_mean_d+k`, is refused whatever the rules, as `{target}_d+k` is as a
predictor. `explain()` shows the base apart from the learner's inputs (`out["base"]`), judged on the issue row.

## Newest targets

With the learner's `newest_target="published"`, a period's newest training target is the oldest of its hour's quarter-hours (its
deviation reads all four), and with `native="distinct"` the oldest of its target date's (the flag reads every hour of the date). For
day-ahead prices, published for a whole day at once, this changes nothing.

## Results

The records are the target's: `actual` is the quarter-hour's price, `coefficients` the learner's. The manifest's `model_params`
record `base`, `period`, `native` and the learner's class and parameters, and its list, window, refit rule, newest target and mask
rule are recorded as the model's: a resume with another of them refuses. The base column is an input the model reads (hashed in
the manifest, perturbed by `verify`), never a regressor unless the learner lists it. A record doesn't carry the base's value (unlike
a `ResidualModel`'s `base_prediction`): the hourly model's own records hold it, and `prediction` less it is the learner's part.

## Not in this step

An engine group (`ModelPipeline(engine="on")` runs it on its own), `PooledModel` over zones, a per-row target (`normalise_by`,
`bounds`, `zero_when`) on the model or its learner, learners of day units, sequence models, `TorchModel` and predictor functions.
