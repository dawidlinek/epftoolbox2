"""
Example: LEAR on the Lago et al. (2021) benchmark, exactly as the paper ran it and with the recommended settings

The two documented LEAR settings (docs page models/lear) on one of the paper's five open-access datasets, one day ahead
over its test period:

- "LEAR_lago": LEARModel.lago(), the paper's LEAR. It reproduces the original epftoolbox's forecasts.
- "LEAR": the recommended LEAR, four LEARModels that differ only in their scaler (Lago's Invariant asinh, mirror-log,
  polynomial, N-PIT), each on the window average [56, 84, 728, "expanding"] with BIC, averaged by an AverageEnsemble.

No API key is needed: EpftoolboxBenchmarkSource downloads the dataset from Zenodo once and checks its md5. Forecasts are
issued at 11:45 Brussels on the day before each target day, and the newest training target of every model is the issue
day's price, published at 13:00 the day before. The full run (728 issue days, about 6 core-hours with the numba kernels of
the [fast] extra, many more without them) is resumable: results go to OUTPUT_DIR/results_lear_{market}.
"""

import os

os.environ["MAX_PROCESSES"] = "4"

from pathlib import Path

import pandas as pd
from epftoolbox2.pipelines import DataPipeline, ModelPipeline
from epftoolbox2.data.sources import CalendarSource, EpftoolboxBenchmarkSource
from epftoolbox2.data.transformers import (
    DailyProfileTransformer,
    DSTNormalizeTransformer,
    LagTransformer,
    TimezoneTransformer,
)
from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.evaluators import DMEvaluator, MAEEvaluator, RMSEEvaluator
from epftoolbox2.exporters import CsvExporter, TerminalExporter
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import LEARModel
from epftoolbox2.presets import WEEKDAYS, lear_predictors, profile_columns

# The two exogenous series of each dataset in the order of its file, which LEARModel.lago() needs (its coordinate descent
# stops early, so the column order matters).
EXOGENOUS = {
    "NP": ["load_forecast", "wind_forecast"],
    "PJM": ["load_forecast", "zonal_load_forecast"],
    "FR": ["generation_forecast", "load_forecast"],
    "BE": ["generation_forecast_fr", "load_forecast_fr"],
    "DE": ["load_forecast_amprion", "wind_solar_forecast"],
}
# Only the weekday dummies are taken from CalendarSource (holidays=False), and they depend on the local date alone; Nord Pool
# and PJM have no calendar of their own, so they borrow one.
CALENDAR_COUNTRY = {"NP": "NO", "PJM": "DE", "FR": "FR", "BE": "BE", "DE": "DE"}
# The recommended LEAR's four scalers: Lago's Invariant asinh (no standardisation), then mirror-log, polynomial and N-PIT.
VSTS = {"asinh": {"name": "asinh", "params": {"standardise": False}}, "mlog": "mlog", "poly": "poly", "npit": "npit"}


def main(market="DE", test_start=None, test_end=None, output_dir="."):
    """Forecast the issue days test_start..test_end (default: the paper's test period, each target day's day before)."""
    output = Path(output_dir)
    source = EpftoolboxBenchmarkSource(market)
    exogenous = EXOGENOUS[market]
    first_target, last_target = (pd.Timestamp(day) for day in source.test_period)
    test_start = test_start or (first_target - pd.Timedelta(days=1)).date().isoformat()
    test_end = test_end or (last_target - pd.Timedelta(days=1)).date().isoformat()
    # The file holds the 1456 days before its test period. Its first day is the expanding window's first row, and the
    # paper's longest window (1449 training rows) reaches back that far for the first test day.
    first_day = (first_target - pd.Timedelta(days=1456)).date().isoformat()
    last_day = (pd.Timestamp(test_end) + pd.Timedelta(days=1)).date().isoformat()

    # LEAR's inputs (Predictor Presets): the 24-hour profiles of the price on D, D-1, D-2 and D-6, of both exogenous
    # series on the target day, D and D-6, and the target day's weekday.
    zone = source.index_timezone
    data = (
        DataPipeline(timezone=zone)
        .add_source(source)
        .add_source(CalendarSource(CALENDAR_COUNTRY[market], weekday="onehot", holidays=False, timezone=zone))
        .add_transformer(TimezoneTransformer())
        .add_transformer(DSTNormalizeTransformer(freq="1h"))
        .add_transformer(DailyProfileTransformer(columns=["price", *exogenous]))
        .add_transformer(LagTransformer(columns=profile_columns("price"), lags=[1, 2, 6], freq="day"))
        .add_transformer(
            LagTransformer(columns=[c for x in exogenous for c in profile_columns(x)], lags=[-1, 6], freq="day")
        )
        .add_transformer(LagTransformer(columns=[f"is_{d}" for d in WEEKDAYS], lags=[-1], freq="day"))
    )
    df = data.run(first_day, last_day)
    print(f"{market}: {len(df)} rows, {first_day} to {last_day}")

    # Prices of D are published at 13:00 on D-1 (the files' clock, Etc/GMT-1 in Europe). PJM's file is on Etc/GMT+5, so its
    # issue time and rule are read on that clock.
    clock = "Etc/GMT+5" if market == "PJM" else "Europe/Brussels"
    rule_zone = "Etc/GMT+5" if market == "PJM" else "Etc/GMT-1"
    information_set = InformationSet(
        "11:45", clock, rules={"price": Availability.published("13:00", 1, timezone=rule_zone)}
    ).with_source_rules([source])

    pipeline = ModelPipeline(information_set=information_set, availability="raise")
    pipeline.add_model(LEARModel.lago(exogenous, name="LEAR_lago"))  # Lago et al. (2021), exactly
    lear_x = lear_predictors(exogenous)
    for vst, scaler in VSTS.items():  # recommended
        pipeline.add_model(
            LEARModel(
                predictors=lear_x,
                training_window=[56, 84, 728, "expanding"],
                criterion="bic",
                scaler=scaler,
                newest_target="published",
                name=f"LEAR_{vst}",
            )
        )
    pipeline.add_ensemble(AverageEnsemble(members=[f"LEAR_{v}" for v in VSTS], name="LEAR"))
    pipeline.add_evaluator(MAEEvaluator()).add_evaluator(RMSEEvaluator())
    pipeline.add_evaluator(DMEvaluator(benchmark="LEAR_lago"))
    pipeline.add_exporter(TerminalExporter()).add_exporter(CsvExporter(output / f"lear_{market}.csv"))

    report = pipeline.run(
        df, test_start=test_start, test_end=test_end, horizon=1, save_dir=output / f"results_lear_{market}"
    )

    print("\n=== Forecasts per model (identical keys) ===")
    print(report.forecast_counts())
    print("\n=== Summary ===")
    print(report.summary())
    return report


if __name__ == "__main__":
    main()
