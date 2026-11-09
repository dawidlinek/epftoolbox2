"""
Example: a random search of day-level MLPs, averaged: the four best on the validation year (the documented default search)

A loop of MLPDailyModel configurations drawn with a fixed seed (hidden sizes, dropout, learning rate), each refitted weekly with
4 seeds, and an AverageEnsemble of the 4 with the lowest validation MAE, chosen on validation only and scored on test. In the
research, the top 4 configurations x 4 seeds beat the single best configuration x 4 seeds, the top 8 or 16 were no better, and
validation and test ranks within the top 16 barely agreed: a small search averaged, not a large search picked.

On the Lago et al. (2021) DE benchmark (no API key: EpftoolboxBenchmarkSource downloads it from Zenodo once), one day ahead,
with LEAR's 247 day-level inputs. Forecasts are issued at 11:45 Brussels on the day before each target day. Needs the [torch]
extra (pip install 'epftoolbox2[torch]'). The full run (48 configurations, a validation year and the paper's two test years, 4
seeds, weekly refits: about 32,000 networks of 256 to 1024 units) takes a few CPU-days on one thread per fit, so run it on many
cores (MAX_PROCESSES) or shard it with a Workflow array job; results go to OUTPUT_DIR/results_mlp_search and resume.
"""

import os

os.environ.setdefault("MAX_PROCESSES", "8")

import random
from pathlib import Path

import pandas as pd
from epftoolbox2.data.sources import CalendarSource, EpftoolboxBenchmarkSource
from epftoolbox2.data.transformers import (
    DailyProfileTransformer,
    DSTNormalizeTransformer,
    LagTransformer,
    TimezoneTransformer,
)
from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator
from epftoolbox2.exporters import TerminalExporter
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import MLPDailyModel
from epftoolbox2.pipelines import DataPipeline, ModelPipeline
from epftoolbox2.presets import WEEKDAYS, lear_predictors, profile_columns

EXOGENOUS = ["load_forecast_amprion", "wind_solar_forecast"]  # the DE file's two series


def main(n_configs=48, top=4, seeds=4, validation=None, test=None, output_dir="."):
    """Search `n_configs` configurations; average the `top` best on `validation` (default: the year before the test
    period) and score them on `test` (default: the paper's two test years). Both are (first, last) issue days."""
    output = Path(output_dir)
    source = EpftoolboxBenchmarkSource("DE")
    first_target, last_target = (pd.Timestamp(day) for day in source.test_period)
    test = test or (
        (first_target - pd.Timedelta(days=1)).date().isoformat(),
        (last_target - pd.Timedelta(days=1)).date().isoformat(),
    )
    validation = validation or "previous_year"
    first_day = (first_target - pd.Timedelta(days=1456)).date().isoformat()
    last_day = (pd.Timestamp(test[1]) + pd.Timedelta(days=1)).date().isoformat()

    # LEAR's day-level inputs: the 24-hour profiles of the price on D, D-1, D-2 and D-6, of both exogenous series on the target
    # day, D and D-6, and the target day's weekday: one value per local date, as a day-level network needs.
    zone = source.index_timezone
    data = (
        DataPipeline(timezone=zone)
        .add_source(source)
        .add_source(CalendarSource("DE", weekday="onehot", holidays=False, timezone=zone))
        .add_transformer(TimezoneTransformer())
        .add_transformer(DSTNormalizeTransformer(freq="1h"))
        .add_transformer(DailyProfileTransformer(columns=["price", *EXOGENOUS]))
        .add_transformer(LagTransformer(columns=profile_columns("price"), lags=[1, 2, 6], freq="day"))
        .add_transformer(
            LagTransformer(columns=[c for x in EXOGENOUS for c in profile_columns(x)], lags=[-1, 6], freq="day")
        )
        .add_transformer(LagTransformer(columns=[f"is_{d}" for d in WEEKDAYS], lags=[-1], freq="day"))
    )
    df = data.run(first_day, last_day)

    # Prices of D are published at 13:00 on D-1 (the file's clock, Etc/GMT-1).
    information_set = InformationSet(
        "11:45", "Europe/Brussels", rules={"price": Availability.published("13:00", 1, timezone="Etc/GMT-1")}
    ).with_source_rules([source])
    pipeline = ModelPipeline(information_set=information_set, availability="raise")

    # The search: a fixed seed, so the loop is the same every time; the learning rate in 1e-4..3e-4, where the research found
    # the best networks (the other settings moved the median validation MAE by 0.05-0.1 EUR/MWh).
    lear_x = lear_predictors(EXOGENOUS)
    rng, names = random.Random(0), []
    for i in range(n_configs):
        name = f"mlp_{i:02d}"
        pipeline.add_model(
            MLPDailyModel(
                lear_x,
                hidden=rng.choice([[256, 128], [512, 256], [1024, 512]]),
                dropout=round(rng.uniform(0, 0.5), 2),
                lr=round(10 ** rng.uniform(-4, -3.5), 6),
                training_window=728,
                refit="7D",
                seeds=seeds,
                name=name,
            )
        )
        names.append(name)
    pipeline.add_ensemble(
        AverageEnsemble(members=names, top=min(top, n_configs), by="validation", name=f"mlp_top{top}")
    )
    pipeline.add_evaluator(MAEEvaluator()).add_evaluator(RMSEEvaluator()).add_exporter(TerminalExporter())

    report = pipeline.run(df, validation=validation, test=test, horizon=1, save_dir=output / "results_mlp_search")
    print("\n=== Chosen on validation ===")
    print(report.choices())
    print("\n=== Forecasts per model (identical keys) ===")
    print(report.forecast_counts())
    print("\n=== Test ===")
    print(report.summary())
    return report


if __name__ == "__main__":
    main()
