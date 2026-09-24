"""
Example 3: Full Workflow - Data + Model Pipelines

This example demonstrates a complete electricity price forecasting workflow:
1. Download and process data using DataPipeline
2. Train and evaluate models using ModelPipeline
"""

import os

os.environ["THREADS_PER_PROCESS"] = "16"
os.environ["MAX_PROCESSES"] = "2"

from epftoolbox2.pipelines import DataPipeline, ModelPipeline
from epftoolbox2.data.sources import EntsoeSource, OpenMeteoSource, CalendarSource
from epftoolbox2.data.transformers import ResampleTransformer, LagTransformer, TimezoneTransformer
from epftoolbox2.data.validators import NullCheckValidator
from epftoolbox2.information_set import InformationSet
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator
from epftoolbox2.exporters import ExcelExporter, TerminalExporter, CsvExporter

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "YOUR_API_KEY_HERE")

DATA_START = "2023-01-01"
DATA_END = "2024-04-01"

TEST_START = "2024-02-01"
TEST_END = "2024-03-01"
if __name__ == "__main__":
    data_pipeline = (
        # timezone makes DATA_START/DATA_END mean whole local calendar days, not UTC midnight, so the last
        # local day of the fetch isn't left partial once TimezoneTransformer converts to Europe/Warsaw below.
        DataPipeline(timezone="Europe/Warsaw")
        .add_source(EntsoeSource(country_code="PL", api_key=ENTSOE_API_KEY, type=["load", "price"]))
        .add_source(
            # jma_seamless (default model) is only leak-free up to 5 days ahead at the default issue_time/
            # publication_delay_h; a longer horizon needs a different model or publication_delay_h. timezone is
            # left unset: it inherits DataPipeline(timezone=...) above.
            OpenMeteoSource(latitude=52.2297, longitude=21.0122, horizon=5, prefix="warsaw")
        )
        .add_source(CalendarSource(country="PL", holidays="binary", weekday="onehot", daylight=True))
        .add_transformer(TimezoneTransformer())
        .add_transformer(ResampleTransformer(freq="1h"))
        # load_actual/price are only known for D once D is over; never offer their same-day (unlagged) value.
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
        # load_forecast (A65/A01) is published 10:00 CET the day before delivery: only d+1 is ever safe.
        .add_transformer(LagTransformer(lags=range(-1, 0), freq="day", columns=["load_forecast"]))
        .add_validator(NullCheckValidator(columns=["load_actual", "price"]))
    )
    df = data_pipeline.run(start=DATA_START, end=DATA_END, cache=True)
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

    # issue_time is 30 min before the 12:00 CET day-ahead gate closure. with_source_rules() teaches the check
    # that "warsaw_temperature_2m" is an Open-Meteo issue-time vintage column (see docs/model-pipeline/
    # information-set); Workflow does this automatically, but a standalone ModelPipeline needs it spelled out.
    information_set = InformationSet("11:30", "Europe/Warsaw").with_source_rules(data_pipeline.sources)

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
        .add_exporter(ExcelExporter("full_workflow_results.xlsx"))
        .add_exporter(CsvExporter("full_workflow_results.csv"))
    )

    report = model_pipeline.run(
        data=df,
        test_start=TEST_START,
        test_end=TEST_END,
        target="price",
        horizon=5,  # matches warsaw_temperature_2m's leak-free horizon (see the OpenMeteoSource above)
        save_dir="results",
    )
