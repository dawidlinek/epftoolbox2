"""
Example 5: Forecast-Only Mode

Produce a 5-day-ahead electricity price forecast from today's data, with no back-test,
no evaluation and no scores.

Key patterns:
- Date keywords: "today", "now_d-730", "today_d+5" resolve at runtime, so this
  script can be run every day without changing any dates.
- DataPipeline is given timezone="Europe/Warsaw", so "today"/"end" mean whole
  local calendar days: end="today_d+5" fetches weather forecasts and calendar
  features through the full 5-day prediction window (needed as predictors).
  `end` always includes the whole day it names (local here because `timezone`
  is set, UTC otherwise).
- ModelPipeline uses forecast_only=True to skip evaluators and exporters.
- ModelPipeline is given an InformationSet (issue_time + the data sources' rules), so every predictor is
  checked against what is known at issue time before any fit.
- report.predictions() returns a clean DataFrame with no "actual" column.

Output: 120 rows per model (24 hours × 5 horizons).
"""

import os

os.environ["THREADS_PER_PROCESS"] = "8"
os.environ["MAX_PROCESSES"] = "2"

from epftoolbox2.pipelines import DataPipeline, ModelPipeline
from epftoolbox2.data.sources import EntsoeSource, OpenMeteoSource, CalendarSource
from epftoolbox2.data.transformers import TimezoneTransformer, ResampleTransformer, LagTransformer
from epftoolbox2.information_set import InformationSet
from epftoolbox2.models import OLSModel, LassoCVModel

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "YOUR_ENTSOE_API_KEY")

if __name__ == "__main__":
    # Fetch 2 years of history + 5-day weather/calendar forecast.
    # Adjust "now_d-730" to match your largest model's training_window.
    data_pipeline = (
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
                    "is_wednesday",
                    "is_thursday",
                    "is_friday",
                    "is_saturday",
                    "is_sunday",
                    "is_holiday",
                    "daylight_hours",
                ],
            )
        )
        .add_transformer(LagTransformer(lags=range(-1, 0), freq="day", columns=["load_forecast"]))
    )
    df = data_pipeline.run(start="now_d-735", end="today_d+5", cache=True)

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

    # Forecasts are issued at 11:45, before the 12:00 CET day-ahead gate closure
    information_set = InformationSet("11:45", "Europe/Warsaw").with_source_rules(data_pipeline.sources)

    report = (
        ModelPipeline(information_set=information_set)
        .add_model(OLSModel(predictors=predictors, training_window=365, name="OLS"))
        .add_model(LassoCVModel(predictors=predictors, training_window=365, cv=7, name="LassoCV"))
        .run(data=df, target="price", horizon=5, forecast_only=True)
    )

    # 120 rows per model: target_date, hour, horizon, prediction, model
    forecast = report.predictions()
    print(forecast.to_string())
