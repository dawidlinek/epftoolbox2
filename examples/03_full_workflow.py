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
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import ExcelExporter, TerminalExporter

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "YOUR_API_KEY_HERE")

DATA_START = "2023-01-01"
DATA_END = "2024-04-01"

TEST_START = "2024-02-01"
TEST_END = "2024-03-01"
if __name__ == "__main__":

    df = (
        DataPipeline()
        .add_source(EntsoeSource(country_code="PL", api_key=ENTSOE_API_KEY, type=["load", "price"]))
        .add_source(OpenMeteoSource(latitude=52.2297, longitude=21.0122, horizon=7, prefix="warsaw"))
        .add_source(CalendarSource(country="PL", holidays="binary", weekday="onehot",daylight=True))
        .add_transformer(TimezoneTransformer(target_tz="Europe/Warsaw"))
        .add_transformer(ResampleTransformer(freq="1h"))
        .add_transformer(LagTransformer(columns=["load_actual", "price"], lags=[1, 2, 7],freq="day"))
        .add_transformer(LagTransformer(lags=range(-7, 1), freq="day", columns=["is_monday", "is_tuesday", "daylight_hours", "is_wednesday", "is_thursday", "is_friday", "is_saturday", "is_sunday", "load_forecast", "is_holiday"]))
        .add_validator(NullCheckValidator(columns=["load_actual", "price"]))
        .run(start=DATA_START, end=DATA_END, cache=True)
    )
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
        "price_d-1",
        "price_d-7",
        lambda h: f"warsaw_temperature_2m_d+{h}",
    ]

    model_pipeline = (
        ModelPipeline()
        .add_model(OLSModel(predictors=predictors, training_window=365, name="OLS",))
        .add_model(LassoCVModel(predictors=predictors, training_window=365, cv=7, name="LassoCV"))
        .add_evaluator(MAEEvaluator())
        .add_exporter(TerminalExporter())
        .add_exporter(ExcelExporter("full_workflow_results.xlsx"))
    )

    report = model_pipeline.run(
        data=df,
        test_start=TEST_START,
        test_end=TEST_END,
        target="price",
        horizon=7,
        save_dir="results",
    )
