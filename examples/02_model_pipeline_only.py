"""
Example 2: Model Pipeline Only

This example demonstrates how to use the ModelPipeline to train and evaluate
forecasting models on pre-existing data (loaded from CSV).
"""

import os

os.environ["THREADS_PER_PROCESS"] = "16" 
os.environ["MAX_PROCESSES"] = "2" 

import pandas as pd
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator, rMAEEvaluator
from epftoolbox2.exporters import ExcelExporter, TerminalExporter, CsvExporter

if __name__ == "__main__":
    df = pd.read_csv("data_output.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Warsaw")
    print(f"Loaded {len(df)} rows")

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
                cv=7,  # 7-fold cross-validation
                name="Lasso CV",
            )
        )
        .add_evaluator(MAEEvaluator())
        .add_evaluator(RMSEEvaluator())
        .add_evaluator(rMAEEvaluator(base_model="OLS Baseline"))
        .add_exporter(TerminalExporter(['horizon']))
        .add_exporter(ExcelExporter("model_results.xlsx"))
        .add_exporter(CsvExporter("model_results.csv"))
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
