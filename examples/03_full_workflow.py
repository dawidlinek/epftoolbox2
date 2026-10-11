"""
Example 3: Full Workflow - Data + Model Pipelines

This example demonstrates a complete electricity price forecasting workflow:
1. Download and process data using DataPipeline
2. Train and evaluate models using ModelPipeline
"""

import os

os.environ["THREADS_PER_PROCESS"] = "16"
os.environ["MAX_PROCESSES"] = "2"

from pathlib import Path

from epftoolbox2.pipelines import DataPipeline, ModelPipeline
from epftoolbox2.data.sources import EntsoeSource, OpenMeteoSource, CalendarSource
from epftoolbox2.data.transformers import ResampleTransformer, LagTransformer, TimezoneTransformer
from epftoolbox2.data.validators import CoverageValidator
from epftoolbox2.information_set import InformationSet
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator
from epftoolbox2.exporters import ExcelExporter, TerminalExporter, CsvExporter

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "YOUR_API_KEY_HERE")


def main(
    data_start="2023-01-01",
    data_end="2024-04-01",
    test_start="2024-02-01",
    test_end="2024-03-01",
    output_dir=".",
):
    output = Path(output_dir)
    data_pipeline = (
        # timezone makes data_start/data_end whole local days
        DataPipeline(timezone="Europe/Warsaw")
        .add_source(EntsoeSource(country_code="PL", api_key=ENTSOE_API_KEY, type=["load", "price"]))
        .add_source(OpenMeteoSource(latitude=52.2297, longitude=21.0122, prefix="warsaw"))
        .add_source(CalendarSource(country="PL", holidays="binary", weekday="onehot", daylight=True))
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
    )
    df = data_pipeline.run(start=data_start, end=data_end, cache=True)
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
        "load_actual_d-7",
        "price_d-1",
        "price_d-7",
        lambda h: f"warsaw_temperature_2m_d+{h}",
    ]

    # Forecasts are issued at 11:45, before the 12:00 CET day-ahead gate closure
    information_set = InformationSet("11:45", "Europe/Warsaw").with_source_rules(data_pipeline.sources)

    model_pipeline = (
        ModelPipeline(information_set=information_set)
        .add_model(
            OLSModel(
                predictors=predictors,
                training_window=365,
                name="OLS",
            )
        )
        .add_model(LassoCVModel(predictors=predictors, training_window=365, cv=7, name="LassoCV"))
        .add_evaluator(MAEEvaluator())
        .add_evaluator(RMSEEvaluator())
        .add_exporter(TerminalExporter())
        .add_exporter(ExcelExporter(output / "full_workflow_results.xlsx"))
        .add_exporter(CsvExporter(output / "full_workflow_results.csv"))
    )

    report = model_pipeline.run(
        data=df,
        test_start=test_start,
        test_end=test_end,
        target="price",
        horizon=6,  # as far ahead as the weather forecasts go
        save_dir=output / "results",
    )
    return report


if __name__ == "__main__":
    main()
