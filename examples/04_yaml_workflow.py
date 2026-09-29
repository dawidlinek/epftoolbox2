"""
Example 4: Workflow - YAML-based experiment configuration

This example shows how to define a full experiment as YAML config files and
run it via the Workflow class.

Two-phase pattern:
  1. Build pipelines once, save to YAML (setup phase)
  2. Load Workflow from YAML and run (execution phase)

Note: predictors must be strings or "{horizon}" templates — callables cannot
be serialized. Use "col_d+{horizon}" template syntax instead of lambdas.
"""

import os

os.environ["PYTHON_GIL"] = "0"

from epftoolbox2.pipelines import DataPipeline, ModelPipeline, Workflow
from epftoolbox2.data.sources import EntsoeSource, OpenMeteoSource, CalendarSource
from epftoolbox2.data.transformers import ResampleTransformer, LagTransformer, TimezoneTransformer
from epftoolbox2.data.validators import CoverageValidator
from epftoolbox2.models import OLSModel, LassoCVModel
from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator
from epftoolbox2.exporters import ExcelExporter, TerminalExporter

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "YOUR_ENTSOE_API_KEY")
# ---------------------------------------------------------------------------
# Phase 1: Build pipelines and save to YAML (run once)
# ---------------------------------------------------------------------------

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
    # timezone makes data_start/data_end (in experiment.yaml) mean whole local calendar days, not UTC midnight.
    DataPipeline(timezone="Europe/Warsaw")
    .add_source(EntsoeSource(country_code="PL", api_key=ENTSOE_API_KEY, type=["load", "price"]))
    .add_source(
        # jma_seamless (default model) is only leak-free up to 5 days ahead at the default
        # api="previous_runs"/run="00"; a longer horizon needs a different model or api="single_runs".
        # timezone is left unset: it inherits DataPipeline(timezone=...) above.
        OpenMeteoSource(latitude=52.2297, longitude=21.0122, horizon=5, prefix="warsaw")
    )
    .add_source(CalendarSource(country="PL", holidays="binary", daylight=True, weekday="onehot"))
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
    .add_validator(CoverageValidator(columns=["load_actual", "price"], min_coverage=1.0))
    .save("data_pipeline.yaml")
)

(
    # issue_time is 30 min before the 12:00 CET day-ahead gate closure: every predictor above is checked
    # against it before any model is fit (see docs/model-pipeline/information-set).
    ModelPipeline(issue_time="11:30", issue_timezone="Europe/Warsaw")
    .add_model(OLSModel(predictors=predictors, training_window=365, name="OLS"))
    .add_model(LassoCVModel(predictors=predictors, training_window=365, cv=5, name="LassoCV"))
    .add_evaluator(MAEEvaluator())
    .add_evaluator(RMSEEvaluator())
    .add_exporter(TerminalExporter())
    .add_exporter(ExcelExporter("workflow_results.xlsx"))
    .save("model_pipeline.yaml")
)

Workflow(
    data_pipeline="data_pipeline.yaml",
    model_pipeline="model_pipeline.yaml",
    data_start="2023-05-01",
    data_end="2024-08-01",
    data_cache=True,
    model_test_start="2024-06-01",
    model_test_end="2024-07-01",
    model_target="price",
    model_horizon=5,  # matches warsaw_temperature_2m's leak-free horizon (see the OpenMeteoSource above)
    max_processes=4,
    threads_per_process=8,
).save("experiment.yaml")

# ---------------------------------------------------------------------------
# Phase 2: Load and run
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     report = Workflow.load("experiment.yaml").run()

#     print(report.summary())
#     print(report.by_horizon())
