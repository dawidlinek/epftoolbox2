"""
Example 5: Forecast-Only Mode

Produce a 7-day-ahead electricity price forecast from today's data — no back-test,
no evaluation, no scores.

Key patterns:
- Date keywords: "today", "now_d-730", "today_d+7" resolve at runtime, so this
  script can be run every day without changing any dates.
- DataPipeline end is set to "today_d+7" so that weather forecasts and calendar
  features are fetched for the full prediction window (needed as predictors).
- ModelPipeline uses forecast_only=True to skip evaluators and exporters.
- report.predictions() returns a clean DataFrame with no "actual" column.

Output: 168 rows per model (24 hours × 7 horizons).
"""

import os

os.environ["THREADS_PER_PROCESS"] = "8"
os.environ["MAX_PROCESSES"] = "2"

from epftoolbox2.pipelines import DataPipeline, ModelPipeline
from epftoolbox2.data.sources import EntsoeSource, OpenMeteoSource, CalendarSource
from epftoolbox2.data.transformers import TimezoneTransformer, ResampleTransformer, LagTransformer
from epftoolbox2.models import OLSModel, LassoCVModel

ENTSOE_API_KEY = "fade2e5f-6d62-4354-9f95-e8629acec0e9"

if __name__ == "__main__":

    # Fetch 2 years of history + 7-day weather/calendar forecast.
    # Adjust "now_d-730" to match your largest model's training_window.
    df = (
        DataPipeline()
        .add_source(EntsoeSource(country_code="PL", api_key=ENTSOE_API_KEY, type=["load", "price"]))
        .add_source(OpenMeteoSource(latitude=52.2297, longitude=21.0122, horizon=7, prefix="warsaw"))
        .add_source(CalendarSource(country="PL", holidays="binary", weekday="onehot", daylight=True))
        .add_transformer(TimezoneTransformer(target_tz="Europe/Warsaw"))
        .add_transformer(ResampleTransformer(freq="1h"))
        .add_transformer(LagTransformer(columns=["load_actual", "price"], lags=[1, 2, 7], freq="day"))
        .add_transformer(LagTransformer(
            lags=range(-7, 1), freq="day",
            columns=["is_monday", "is_tuesday", "is_wednesday", "is_thursday",
                     "is_friday", "is_saturday", "is_sunday", "is_holiday",
                     "daylight_hours", "load_forecast"],
        ))
        .run(start="now_d-735", end="today_d+8", cache=True)
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
        "load_actual_d-2",
        "price_d-1",
        "price_d-2",
        "warsaw_temperature_2m_d+{horizon}",
    ]

    report = (
        ModelPipeline()
        .add_model(OLSModel(predictors=predictors, training_window=365, name="OLS"))
        .add_model(LassoCVModel(predictors=predictors, training_window=365, cv=7, name="LassoCV"))
        .run(data=df, target="price", horizon=7, forecast_only=True)
    )

    # 168 rows per model: target_date, hour, horizon, prediction, model
    forecast = report.predictions()
    print(forecast.to_string())
