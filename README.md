# epftoolbox2

A modern Python library for electricity price forecasting with modular data pipelines and model evaluation. Built as a complete rewrite of the original [epftoolbox](https://github.com/jeslago/epftoolbox), this library provides a flexible, extensible framework for downloading energy market data, building forecasting models, and evaluating their performance.

📖 **[Documentation](https://dawidlinek.github.io/epftoolbox2)**

## Installation

```bash
pip install epftoolbox2
```

or with uv 

```bash
uv add epftoolbox2
```

## Verify Installation

After installation, you can verify your setup and check system information:

```python
import epftoolbox2

epftoolbox2.verify()
```
## Key Features

- **Data Sources**: ENTSOE (load, generation, day-ahead prices on an hourly or 15-minute grid, with the EXAA auction of AT and DE_LU as `price_exaa` and the Swiss auction for CH, cross-border transfer capacity, installed capacity, with `prefix` to combine zones), Open-Meteo (`OpenMeteoSource`, archived weather forecasts as known at a UTC cutoff, with `api="previous_runs"` for a long lead-time-bucket archive or `api="single_runs"` for one exact model run per day, and a `locations` dict to combine several locations into per-location columns, one or more aggregated series per variable via `agg` (mean, optionally population- or capacity-weighted, plus median, min, max and sum), or both), Calendar (national and regional holidays, bridge days, weekday, hour, month, daylight, a 31-entry market-regime catalog with ex-ante dates and sources, opt-in national or regional-share school holidays via the OpenHolidays API), CSV files (CsvSource), EpftoolboxBenchmark (the Lago et al. 2021 datasets NP, PJM, FR, BE, DE, downloaded on demand), PSE (`PseSource`, the Polish TSO's reports API without a key: load and its day-ahead forecast, SDAC price in PLN, wind and PV, cross-border flows, on 15-minute periods; source: PSE), Energy-Charts (`EnergyChartsSource`, no key: TSO wind, solar and load forecasts by vintage (day-ahead, intraday, current), actual generation, day-ahead prices and cross-border flows; data: Energy-Charts.info, Fraunhofer ISE, CC BY 4.0)
- **Transformers**: Resample, Timezone conversion, Lag features (local calendar days), DST normalisation to 24 slots a day (DSTNormalize), daily profile and daily min/max/mean/last stats (DailyProfile, DailyStats), custom column arithmetic (Lambda), Impute (leak-safe gap filling)
- **Presets**: `epftoolbox2.presets`, ready-made predictor lists for the LEAR and expert-model benchmarks from the literature
- **Validators**: DSTStructure (rows per local day), Coverage (duplicate and unsorted timestamps, per-column outages, missing timestamps and nulls), Plausibility (SDAC price limits, frozen values), EDA statistics
- **Models**: OLS, LassoCV (`fit_intercept=` on both, `StandardScaler(scale_binary=True)` to standardise dummies too), Ridge (the penalty chosen by leave-one-out from a grid on every fit), LEAR (Lago et al.'s Lasso, its penalty chosen by AIC or BIC on the LARS path or by blocked cross-validation; `LEARModel.lago()` reproduces the original epftoolbox's forecasts, the recommended LEAR averages four of them, one per VST, and the `[fast]` extra compiles its kernels with numba), Naive (one column's value on the issue row, such as a TSO's published forecast as a benchmark, with no forecast where it wasn't out at the issue time); `training_window` is days, `"expanding"` or a list such as `[56, 84, 728, "expanding"]` whose windows are fitted separately and averaged inside the model; with `refit="7D"` or `"1MS"` they fit once a week or a month on a fixed calendar instead of once per forecast, and a forecast issued between two refits trains on what the one issued on the refit day trains on; `binary_mask="fit"` reads which columns are binary (left unscaled) from each fit's own rows rather than the test's first day, so no forecast depends on where the test starts
- **Horizons**: `horizon=7` forecasts every day ahead from 1 to 7, `horizons=[1, 3, 7]` (or `range(3, 10)`, `epf model --horizons 3-9`) only those, each model's forecast (and a plain average's) bit for bit what the full run gives it; a `top=` ensemble chooses on the horizons the run made
- **Issue days**: `issue_days=["friday"]` (or `epf model --issue-days friday`) forecasts only on those weekdays, as a week-ahead study does; training is untouched, a fixed list's forecast is bit for bit what a run of every day gives it, and the information set and `verify` look at the listed days alone
- **Week-ahead forecasts**: daily minimum and maximum load for D+3 to D+9 on Fridays (or D+1 to D+7 every day) against the TSO's A31 forecast, documented end to end and reproducing a published 20-market study's forecasts to the stored digits
- **Daily forecasts**: `freq="1D"` (or `epf model --freq 1D`) fits one model per day on a frame with one row per local date at one time of day, such as `df.at_time("12:00")`; the row's stamp says what `x_h-1` holds and when a value is known, a daily run is period 12 of the hourly run bit for bit, and `report.join()` puts it next to an hourly model's `report.aggregate()`
- **Validation and test**: `run(df, validation=(start, end) or "previous_year", test=(start, end))` forecasts both windows in one run and one results file, with each forecast bit for bit what a plain run over its window gives; `report.validation` holds only forecasts whose target day is before the first test issue day, and every report method keeps to its window
- **Residual hybrid**: `ResidualModel(base, learner)` corrects a base model's out-of-sample forecasts (LEAR, Ridge, any model, run by it over a warm-up into `<name>.base.jsonl`) with a learner fitted weekly on their residuals, by default `HistGBModel`, scikit-learn's histogram gradient boosting over the pooled (date, hour) rows of `presets.tree_predictors` and the base's forecasts; the research's best classical model (LEAR-BIC + boosted residuals, 4 % below LEAR's window average on DE and AT)
- **Several zones**: `ModelPipeline.run(data={"PL": df_pl, "FI": df_fi, ...})` runs every model per zone, each zone's forecasts bit for bit those of its frame alone, `issue_timezone="data"` issues each market on its own clock, `PooledModel` fits one model over every zone with each zone's rows standardised within its own fit, and a `PanelReport` scores the log gain over a benchmark per cell on its common keys
- **Ensembles**: `AverageEnsemble` averages stored forecasts on identical keys, `top=k` or `top="validation"` chooses the members on validation per test period, without look-ahead and frozen for that period, `report.choices()` says what was chosen, and an ensemble is one more model in every table; `ModelPipeline.report()` computes them after array jobs
- **Predictor functions**: a function of `ctx` picks each forecast's list (by horizon, period, weekday, or whether an input is out at the issue time), planned before any fit, checked against the issue time, recorded in the results and saved to YAML by import path; `Use(list, train_on="matching")` trains a forecast only on the past days given that list, each as of its own issue time
- **Scalers**: standardisation (default) or variance-stabilising transformations for spiky prices (asinh, mirror-log, polynomial, N-PIT), fitted per forecast on training data only
- **Evaluators**: MAE, RMSE, rMAE (relative to a base model), Bias, ShapeMAE (the shape within the hour of 15-minute forecasts), NMAE (MAE over the mean actual) and ScaledMAE (each error over a source column such as capacity), DM (MAE ratio, bootstrap interval and Diebold-Mariano test against a benchmark), MCS (model confidence set), Economic value (battery arbitrage profit)
- **Comparing models**: `report.tests()`, `mcs()`, `ranks()`, `compare()` and `leaderboard()`, each on the forecast keys its models share, printing how many days and forecasts it used, with Holm correction, fixed seeds and the minimum detectable effect; `tests()`, `compare()`, `leaderboard()` and a DM column also warn when the compared models read one series with different vintage classes
- **Breakdowns and daily aggregates**: `report.by_quarter()`, `by_day()` and `by_price_level()` (negative, spike and normal prices, read from the actual price only, with one spike threshold for all models), `report.by_column(column, bins=None)` (a flag or binned column of the data at the target row), `report.aggregate("1D", agg="max")` to score forecasts of the daily maximum, minimum, mean, spread, peak or offpeak mean from hourly or 15-minute ones (whole days only, dropped days counted), `aggregate("1h")` for the hourly means of 15-minute forecasts, `report.error_split()` (day level, hour profile and quarter shape of the squared error), and `report.join(other)` to put the models of separate runs in one report
- **Exporters**: Excel with conditional formatting, Terminal output, CSV
- **Caching**: Built-in per-source Parquet cache that fetches only missing ranges, refetches not-yet-published data and is safe for parallel jobs
- **Pipelines**: Data and model pipelines that can be saved and loaded with .yaml files. An optional market timezone makes dates mean local calendar days, duplicate column names across sources raise instead of colliding silently, and `api_key` is written to YAML as an `${ENV_VAR}` reference instead of plain text
- **Multiprocessing**: Process-based parallelism with inner thread pools. SLURM array mode (one model or a block of models per task) with `epf array` for the range and `epf report` to merge the results
- **Safeguards**: The model's input (resolution, time stamps, column names, predictors) is checked before any fit, and forecasts that could not be made are counted per model (`report.forecast_counts()`), and every `summary()` and `by_*()` table has a `forecasts` count per model and group, with a warning when the models of a group differ
- **Information Set**: Optional `issue_time` check that every predictor is known at the moment the forecast is issued, at three levels (`availability="none"`, `"warn"`, `"raise"`), with built-in availability rules for the shipped sources (with the vintage class of what their archive holds, listed by `report.vintages()`, and rules over time for a lag that changed), `pipeline.explain()` to read a model's predictors relative to the target day, and an empirical leakage probe (`epftoolbox2.information_set`); `newest_target="published"` on a model trains it on every target published by the issue time, D's prices included, instead of stopping at D-1
- **Verification**: `ModelPipeline.verify()` and `epf verify` run the leakage probe on sampled issue days with a positive control: perturbing unknown values must not move a forecast, and perturbing the freshest known ones must, so a check that can't reach a model reports `inconclusive` instead of passing
- **Result Manifests**: Resumable results (`save_dir`) are fingerprinted with the configuration that produced them (the predictors' ordered layout, data up to `test_end` and its time zone, model params, ...). Resuming into results made with a different configuration raises instead of silently mixing them (`epftoolbox2.results.manifest`)
- **Neural networks** (`[torch]` extra): `TorchModel` trains a `torch.nn.Module` of your own (flat predictors or named blocks, saved by import path and a hash of its source) with one output per period (`output="period"`) or a day of outputs on day-level predictors (`output="day"`); `MLPHourlyModel` and `MLPDailyModel` are its presets with `epftoolbox2.nets.MLP`. The library's trainer, a port of the research's stacked trainer, owns the rows and their cut, scaling, early stopping on each fit's own training days and seeds from the fit alone (their mean in EUR/MWh), and keeps a forecast the same alone, in a loop and after a resume: one torch thread and TF32 off per fit, both recorded and compared. The networks of a loop that share a stack key train together in padded stacks on one device (a search on a GPU), each forecast its model's alone to the bit. `examples/mlp_search.py` is the default search: a random loop of configurations and `AverageEnsemble(top=4, by="validation")`
- **Licences and weights**: a model family that needs an extra or pretrained weights declares a `ModelCard` (licence, revenue cap, attribution, corpus, runtime pins, weight hashes); every run checks every model's extra, runtime and weights before any fit, never downloads (`epf weights fetch`, with `--accept` for restricted licences, `list`, `verify`), refuses a resume with another runtime or other weights, and `report.provenance()` and the exporters show each model's licence and contamination verdict per window
- **Extensibility**: Extensible base classes for data sources, transformers, validators, models, evaluators, and exporters

## Quick Start

### Standalone source use

```python
import pandas as pd
from epftoolbox2.data.sources import EntsoeSource

source = EntsoeSource("PL", api_key="YOUR_KEY", type=["load", "price"])
df = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-06-01", tz="UTC"))  # [start, end), UTC
```

### Standalone transformer use

```python
from epftoolbox2.data.transformers import TimezoneTransformer, ResampleTransformer

df = TimezoneTransformer(target_tz="Europe/Warsaw").transform(df)  # sources return UTC
df = ResampleTransformer(freq="1h").transform(df)  # aggregates or step-fills; never fills real gaps
```

### Standalone validator use

```python
from epftoolbox2.data.validators import CoverageValidator

validator = CoverageValidator(columns=["load_actual", "price"], min_coverage=1.0)
result = validator.validate(df)
print(result)  # VALID/INVALID with errors, warnings and details; result.stats has one row per column
```

### Data Pipeline

Download and process electricity market data from ENTSOE, weather forecasts from Open-Meteo, and calendar features.

```python
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.data.sources import EntsoeSource, OpenMeteoSource, CalendarSource
from epftoolbox2.data.transformers import TimezoneTransformer, ResampleTransformer, LagTransformer
from epftoolbox2.data.validators import CoverageValidator

pipeline = (
    # timezone is the single place the market zone needs to be written: sources/transformers below that
    # compute their own local time inherit it and leave their own timezone/target_tz unset.
    DataPipeline(timezone="Europe/Warsaw")  # optional: dates below then mean local calendar days
    .add_source(EntsoeSource("PL", api_key="YOUR_KEY", type=["load", "price"]))
    .add_source(OpenMeteoSource(latitude=52.23, longitude=21.01))
    .add_source(CalendarSource("PL", holidays="binary", weekday="onehot", daylight=True))
    .add_transformer(TimezoneTransformer())
    .add_transformer(ResampleTransformer(freq="1h"))
    # load_actual is only known for D once D is over; never offer its same-day (unlagged) value.
    .add_transformer(LagTransformer(columns=["load_actual"], lags=[1], freq="day"))
    # calendar features are known arbitrarily far in advance, so d+1..d+7 is safe.
    .add_transformer(
        LagTransformer(
            lags=range(-7, 0),
            freq="day",
            columns=[
                "is_monday",
                "is_tuesday",
                "is_wednesday",
                "is_thursday",
                "is_friday",
                "is_saturday",
                "is_sunday",
                "is_holiday",
                "daylight_hours",
            ],
        )
    )
    .add_validator(CoverageValidator(columns=["load_actual", "price"], min_coverage=1.0))
)

df = pipeline.run("2024-01-01", "2024-06-08", cache=True)
```

### Standalone model use

```python
from epftoolbox2.models import OLSModel

seasonal_indicators = [
    "is_monday_d+{horizon}",
    "is_tuesday_d+{horizon}",
    "is_wednesday_d+{horizon}",
    "is_thursday_d+{horizon}",
    "is_friday_d+{horizon}",
    "is_saturday_d+{horizon}",
    "is_sunday_d+{horizon}",
    "is_holiday_d+{horizon}",
    "daylight_hours_d+{horizon}",
]

model = OLSModel(predictors=["load_actual_d-1", *seasonal_indicators], name="OLS")
ref = model.run(df, test_start="2024-04-01", test_end="2024-06-01", target="price")  # where the forecasts are
```

### Standalone evaluator use

```python
import pandas as pd
from epftoolbox2.evaluators import MAEEvaluator

predictions = pd.DataFrame({"prediction": [48.0, 52.5], "actual": [50.0, 51.0]})
mae = MAEEvaluator().compute(predictions)  # 1.75
```

### Standalone exporter use

```python
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import ExcelExporter
from epftoolbox2.results import EvaluationReport

report = EvaluationReport({"OLS": ref}, [MAEEvaluator()])  # the object ModelPipeline.run() returns
ExcelExporter("results.xlsx").export(report)
```

### Model Pipeline

Train and evaluate forecasting models with built-in metrics and export capabilities.

```python
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import ExcelExporter

pipeline = (
    ModelPipeline(issue_time="11:45", issue_timezone="Europe/Warsaw")  # issued before the 12:00 CET gate closure
    .add_model(OLSModel(predictors=["load_actual_d-1", *seasonal_indicators], name="OLS"))
    .add_model(LassoCVModel(predictors=["load_actual_d-1", *seasonal_indicators], name="Lasso"))
    .add_evaluator(MAEEvaluator())
    .add_exporter(ExcelExporter("results.xlsx"))
)

report = pipeline.run(df, test_start="2024-04-01", test_end="2024-06-01", target="price")
```

`issue_time` (and optionally `issue_timezone`) compares every predictor with the time the forecast is actually
issued, before any model is fitted. A predictor that uses a value not known yet (e.g. same-day `load_actual`) is listed in
`report.availability()`, a warning, the terminal exporter and the manifest, and the run goes on. With
`availability="raise"` it raises `InformationSetError` instead, which suits published results and CI. The level doesn't
change forecasts. Built-in availability rules cover the shipped sources. See `epftoolbox2.information_set` for
custom rules, accepted assumptions, and `leakage_probe`, an empirical check that perturbs unknown values and asserts
forecasts don't move. `pipeline.verify(df, test_start, test_end)` (or `epf verify`) runs it on sampled issue days, and
`pipeline.explain(df, model="OLS", test=("2024-04-01", "2024-06-01"), horizon=1, hour=12)` shows what each predictor holds
relative to the target day.



## Examples

See the `examples/` folder for complete working examples. Each script defines a `main()` that takes the date ranges it uses (and a directory for its output files when it writes any), so it can also be imported and called on a short window, for example `main(data_start="2024-01-01", data_end="2024-03-01")`. `examples/lear_lago.py` runs LEAR on the Lago et al. (2021) benchmark, exactly as the paper did and with the recommended settings, and needs no API key. `examples/mlp_search.py` runs the default network search on the same benchmark (a random loop of `MLPDailyModel` configurations and an `AverageEnsemble` of the 4 best on validation; `[torch]` extra), with no key either.