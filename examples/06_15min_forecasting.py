"""
Example 6: 15-Minute Granularity Forecasting

Demonstrates running the full workflow at 15-minute resolution instead of
hourly. The model trains 96 sub-models per day (one per 15-min slot) rather
than the default 24.

Key differences vs. example 03:
- EntsoeSource(..., prefer_15min=True) to pick 15-min prices when available
- ResampleTransformer(freq="15min") to align everything to 15-min bins
- CalendarSource(..., freq="15min") so calendar features are at 15-min too
- ModelPipeline.run(..., freq="15min") selects the 96-period training loop

Note: DE 15-min data from ENTSOE only includes load, not price.
This example forecasts load_actual as the target.
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
from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator, rMAEEvaluator
from epftoolbox2.exporters import ExcelExporter, TerminalExporter, CsvExporter

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "YOUR_API_KEY_HERE")

DATA_START = "2023-01-01"
DATA_END = "2024-04-01"

TEST_START = "2024-02-01"
TEST_END = "2024-03-01"

if __name__ == "__main__":
    data_pipeline = (
        DataPipeline()
        .add_source(EntsoeSource(country_code="DE", api_key=ENTSOE_API_KEY, type=["load", "price"], prefer_15min=True))
        .add_source(
            # jma_seamless (default model) is only leak-free up to 5 days ahead at the default issue_time/
            # publication_delay_h; a longer horizon needs a different model or publication_delay_h.
            OpenMeteoSource(latitude=52.5200, longitude=13.4050, horizon=5, prefix="berlin", timezone="Europe/Berlin")
        )
        .add_source(CalendarSource(country="DE", holidays="binary", weekday="onehot", daylight=True, freq="15min"))
        .add_transformer(TimezoneTransformer(target_tz="Europe/Berlin"))
        # Hourly values become a step over their four quarter-hours; daily min/max cover their local day
        .add_transformer(ResampleTransformer(freq="15min"))
        # load_actual is only known for D once D is over; never offer its same-day (unlagged) value —
        # it is also this example's forecast target, so leaving it unlagged would hand the model the answer.
        .add_transformer(LagTransformer(columns=["load_actual"], lags=[1, 2, 7], freq="day"))
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
        .add_validator(NullCheckValidator(columns=["load_actual"]))
    )
    df = data_pipeline.run(start=DATA_START, end=DATA_END, cache=True)
    df.to_csv("15min_forecasting_data.csv")

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
        lambda h: f"berlin_temperature_2m_d+{h}",
    ]

    # issue_time is 30 min after the 12:00 CET day-ahead gate closure. with_source_rules() teaches the check
    # that "berlin_temperature_2m" is an Open-Meteo issue-time vintage column; Workflow does this
    # automatically, but a standalone ModelPipeline needs it spelled out.
    information_set = InformationSet("12:30", "Europe/Berlin").with_source_rules(data_pipeline.sources)

    model_pipeline = (
        ModelPipeline(information_set=information_set)
        .add_model(OLSModel(predictors=predictors, training_window=365, name="OLS"))
        .add_model(LassoCVModel(predictors=predictors, training_window=365, cv=7, name="LassoCV"))
        .add_evaluator(MAEEvaluator())
        .add_evaluator(RMSEEvaluator())
        .add_evaluator(rMAEEvaluator(base_model="OLS"))
        .add_exporter(TerminalExporter())
        .add_exporter(ExcelExporter("15min_workflow_results.xlsx"))
        .add_exporter(CsvExporter("15min_workflow_results.csv"))
    )

    report = model_pipeline.run(
        data=df,
        test_start=TEST_START,
        test_end=TEST_END,
        target="load_actual",
        horizon=5,  # matches berlin_temperature_2m's leak-free horizon (see the OpenMeteoSource above)
        freq="15min",
        save_dir="results_15min",
    )
