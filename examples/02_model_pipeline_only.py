"""
Example 2: Model Pipeline Only

This example demonstrates how to use the ModelPipeline to train and evaluate
forecasting models on pre-existing data (loaded from CSV).
"""

import os

os.environ["PYTHON_GIL"] = "0"
os.environ["MAX_THREADS"] = "16"  # set to number of cores
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import pandas as pd
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import ExcelExporter, TerminalExporter

df = pd.read_csv("data_output.csv", index_col=0, parse_dates=True)

print(f"Loaded {len(df)} rows")

predictors = [
    "load_actual",
    "weekday",
    "hour",
    "is_holiday",
    "load_actual_d-1",
    "load_actual_d-7",
    "warsaw_temperature_2m_d+{horizon}",
]

pipeline = (
    ModelPipeline()
    .add_model(
        OLSModel(
            predictors=predictors,
            training_window=365,
            name="OLS Baseline",
        )
    )
    .add_model(
        LassoCVModel(
            predictors=predictors,
            training_window=365,
            cv=5,  # 5-fold cross-validation
            name="Lasso CV",
        )
    )
    .add_evaluator(MAEEvaluator())
    .add_exporter(TerminalExporter())
    .add_exporter(ExcelExporter("model_results.xlsx"))
)

report = pipeline.run(
    data=df,
    test_start="2024-02-01",
    test_end="2024-03-01",
    target="price",  # Column to predict
    horizon=7,  # Forecast up to 7 days ahead
    save_dir="results",  # Directory to save intermediate results
)

print("\n=== Summary ===")
print(report.summary())

print("\n=== Results by Hour ===")
print(report.by_hour())

print("\n=== Results by Horizon ===")
print(report.by_horizon())
