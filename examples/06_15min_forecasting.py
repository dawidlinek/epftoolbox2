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
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator, rMAEEvaluator
from epftoolbox2.exporters import ExcelExporter, TerminalExporter, CsvExporter

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "YOUR_API_KEY_HERE")

DATA_START = "2023-01-01"
DATA_END = "2024-04-01"

TEST_START = "2024-02-01"
TEST_END = "2024-03-01"

if __name__ == "__main__":

    df = (
        DataPipeline()
        .add_source(EntsoeSource(country_code="DE", api_key=ENTSOE_API_KEY, type=["load", "price"], prefer_15min=True))
        .add_source(OpenMeteoSource(latitude=52.5200, longitude=13.4050, horizon=7, prefix="berlin"))
        .add_source(CalendarSource(country="DE", holidays="binary", weekday="onehot", daylight=True, freq="15min"))
        .add_transformer(TimezoneTransformer(target_tz="Europe/Berlin"))
        .add_transformer(ResampleTransformer(freq="15min", columns=["load_forecast_daily_min","load_forecast_daily_max"], method="ffill"))
        .add_transformer(ResampleTransformer(freq="15min"))
        .add_transformer(LagTransformer(columns=["load_actual"], lags=[1, 2, 7], freq="day"))
        .add_transformer(LagTransformer(lags=range(-7, 1), freq="day", columns=["is_monday", "is_tuesday", "daylight_hours", "is_wednesday", "is_thursday", "is_friday", "is_saturday", "is_sunday", "load_forecast", "is_holiday"]))
        .add_validator(NullCheckValidator(columns=["load_actual"]))
        .run(start=DATA_START, end=DATA_END, cache=True)
    )
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
        "load_actual",
        *seasonal_indicators,
        "load_actual_d-1",
        "load_actual_d-7",
        lambda h: f"berlin_temperature_2m_d+{h}",
    ]

    model_pipeline = (
        ModelPipeline()
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
        horizon=7,
        freq="15min",
        save_dir="results_15min",
    )
