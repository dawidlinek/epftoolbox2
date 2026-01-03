"""
Example 3: Full Workflow - Data + Model Pipelines

This example demonstrates a complete electricity price forecasting workflow:
1. Download and process data using DataPipeline
2. Train and evaluate models using ModelPipeline
"""

import os

os.environ["PYTHON_GIL"] = "0"
os.environ["MAX_THREADS"] = "16"  # set to number of cores
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from epftoolbox2.pipelines import DataPipeline, ModelPipeline
from epftoolbox2.data.sources import EntsoeSource, OpenMeteoSource, CalendarSource
from epftoolbox2.data.transformers import ResampleTransformer, LagTransformer
from epftoolbox2.data.validators import NullCheckValidator
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import ExcelExporter, TerminalExporter

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "YOUR_API_KEY_HERE")

DATA_START = "2023-05-01"
DATA_END = "2024-08-01"

TEST_START = "2024-06-01"
TEST_END = "2024-07-01"

data_pipeline = (
    DataPipeline()
    .add_source(
        EntsoeSource(
            country_code="PL",
            api_key=ENTSOE_API_KEY,
            type=["load", "price"],
        )
    )
    .add_source(
        OpenMeteoSource(
            latitude=52.2297,
            longitude=21.0122,
            horizon=7,
            prefix="warsaw",
        )
    )
    .add_source(
        CalendarSource(
            country="PL",
            holidays="binary",
        )
    )
    .add_transformer(ResampleTransformer(freq="1h"))
    .add_transformer(
        LagTransformer(
            columns=["load_actual", "price"],
            lags=[1, 2, 7],
            freq="day",
        )
    )
    .add_validator(NullCheckValidator(columns=["load_actual", "price"]))
)

df = data_pipeline.run(start=DATA_START, end=DATA_END, cache=True)


predictors = [
    "load_actual",
    "is_holiday",
    "load_actual_d-1",
    "load_actual_d-2",
    "price_d-1",
    "price_d-2",
    lambda h: f"warsaw_temperature_2m_d+{h}",
]
model_pipeline = (
    ModelPipeline()
    .add_model(
        OLSModel(
            predictors=predictors,
            training_window=365,
            name="OLS",
        )
    )
    .add_model(
        LassoCVModel(
            predictors=predictors,
            training_window=365,
            cv=5,
            name="LassoCV",
        )
    )
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

print(report.summary())
print(report.by_horizon())
