"""
Example 4: Workflow - YAML-based experiment configuration

This example shows how to define a full experiment as YAML config files and
run it via the Workflow class.

Two-phase pattern:
  1. Build pipelines once, save to YAML (setup phase)
  2. Load Workflow from YAML and run (execution phase)

Note: predictors must be strings or "{horizon}" templates, because callables cannot
be serialized. Use "col_d+{horizon}" template syntax instead of lambdas.

main() runs phase 1 only, which needs no network: it writes the three YAML files to output_dir.
Call main(run=True) to run phase 2 as well, which downloads the data and trains the models.
"""

import os

os.environ["PYTHON_GIL"] = "0"

from pathlib import Path

from epftoolbox2.pipelines import DataPipeline, ModelPipeline, Workflow
from epftoolbox2.data.sources import EntsoeSource, OpenMeteoSource, CalendarSource
from epftoolbox2.data.transformers import ResampleTransformer, LagTransformer, TimezoneTransformer
from epftoolbox2.data.validators import CoverageValidator
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator
from epftoolbox2.exporters import ExcelExporter, TerminalExporter

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "YOUR_ENTSOE_API_KEY")


def main(
    data_start="2023-01-01",
    data_end="2024-04-01",
    test_start="2024-02-01",
    test_end="2024-03-01",
    output_dir=".",
    run=False,
):
    output = Path(output_dir)

    # -----------------------------------------------------------------------
    # Phase 1: Build pipelines and save to YAML (run once)
    # -----------------------------------------------------------------------
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

    predictors = [
        *seasonal_indicators,
        "load_actual_d-1",
        "load_actual_d-2",
        "price_d-1",
        "price_d-2",
        "warsaw_temperature_2m_d+{horizon}",
    ]

    (
        # timezone makes data_start/data_end (in experiment.yaml) whole local days
        DataPipeline(timezone="Europe/Warsaw")
        .add_source(EntsoeSource(country_code="PL", api_key=ENTSOE_API_KEY, type=["load", "price"]))
        .add_source(OpenMeteoSource(latitude=52.2297, longitude=21.0122, prefix="warsaw"))
        .add_source(CalendarSource(country="PL", holidays="binary", daylight=True, weekday="onehot"))
        .add_transformer(TimezoneTransformer())
        .add_transformer(ResampleTransformer(freq="1h"))
        .add_transformer(LagTransformer(columns=["load_actual", "price"], lags=[1, 2, 7], freq="day"))
        .add_transformer(
            LagTransformer(
                lags=range(-7, 1),
                freq="day",
                columns=[
                    "is_monday",
                    "is_tuesday",
                    "daylight_hours",
                    "is_wednesday",
                    "is_thursday",
                    "is_friday",
                    "is_saturday",
                    "is_sunday",
                    "is_holiday",
                ],
            )
        )
        .add_transformer(LagTransformer(lags=range(-1, 0), freq="day", columns=["load_forecast"]))
        .add_validator(CoverageValidator(columns=["load_actual", "price"], min_coverage=1.0))
        .save(output / "data_pipeline.yaml")
    )

    (
        # Forecasts are issued at 11:45, before the 12:00 CET day-ahead gate closure
        ModelPipeline(issue_time="11:45", issue_timezone="Europe/Warsaw")
        .add_model(OLSModel(predictors=predictors, training_window=365, name="OLS"))
        .add_model(LassoCVModel(predictors=predictors, training_window=365, cv=5, name="LassoCV"))
        .add_evaluator(MAEEvaluator())
        .add_evaluator(RMSEEvaluator())
        .add_exporter(TerminalExporter())
        .add_exporter(ExcelExporter(output / "workflow_results.xlsx"))
        .save(output / "model_pipeline.yaml")
    )

    Workflow(
        data_pipeline="data_pipeline.yaml",
        model_pipeline="model_pipeline.yaml",
        data_start=data_start,
        data_end=data_end,
        data_cache=True,
        model_test_start=test_start,
        model_test_end=test_end,
        model_target="price",
        model_horizon=6,  # as far ahead as the weather forecasts go
        max_processes=4,
        threads_per_process=8,
    ).save(output / "experiment.yaml")

    # -----------------------------------------------------------------------
    # Phase 2: Load and run
    # -----------------------------------------------------------------------

    if run:
        report = Workflow.load(output / "experiment.yaml").run()

        print(report.summary())
        print(report.by_horizon())
        return report


if __name__ == "__main__":
    main()
