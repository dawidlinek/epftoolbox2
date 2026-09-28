"""
Example 1: Data Pipeline Only

This example demonstrates how to use the DataPipeline to download and process
electricity market data from ENTSOE, weather data from Open-Meteo, and
calendar features - without running any models.

Every lagged/lead column below is built to be usable as a predictor: `load_actual` and `price` are only
offered as `_d-k` (their same-day value is future information relative to most of the day), and
`load_forecast` is only offered as `_d+1`, the one horizon ENTSO-E actually publishes before D
(day-ahead load forecasts for D+2 and later are published on D+1 or later, so day D never has them yet).
"""

import os
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.data.sources import EntsoeSource, OpenMeteoSource, CalendarSource
from epftoolbox2.data.transformers import ResampleTransformer, LagTransformer, TimezoneTransformer
from epftoolbox2.data.validators import NullCheckValidator, ContinuityValidator

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "YOUR_API_KEY_HERE")

pipeline = (
    DataPipeline(timezone="Europe/Warsaw")
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
            # jma_seamless (default model) is only leak-free up to 5 days ahead at the default
            # api="previous_runs"/run="00"; a longer horizon needs a different model (e.g. icon_seamless) or
            # api="single_runs" (horizon up to 14, leads beyond the run's own length come back NaN).
            horizon=5,
            prefix="warsaw",
            # timezone is left unset: it inherits DataPipeline(timezone=...) above.
        )
    )
    .add_source(
        CalendarSource(
            country="PL",
            holidays="binary",  # 1 if holiday, 0 otherwise
            weekday="onehot",  # One-hot encode weekdays (is_monday, is_tuesday, etc.)
            daylight=True,  # Add daylight hours feature
        )
    )
    .add_transformer(TimezoneTransformer())
    # Averages 15-min series, spreads the daily min/max over their local day, leaves real gaps as NaN
    .add_transformer(ResampleTransformer(freq="1h"))
    # load_actual/price are only known for D once D is over; never offer their same-day (unlagged) value.
    .add_transformer(
        LagTransformer(
            columns=["load_actual", "price"],
            lags=range(1, 8),
            freq="day",
        )
    )
    .add_transformer(
        LagTransformer(
            lags=range(-10, 1),
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
    # load_forecast_daily_min/max (A31) is published Fridays 10:00 CET for the whole next week: d+1..d+3
    # is always safe, d+4 and beyond only from Friday on. Cap at d+3 so the shipped columns are always safe.
    .add_transformer(
        LagTransformer(
            lags=range(-3, 1),
            freq="day",
            columns=["load_forecast_daily_min", "load_forecast_daily_max"],
        )
    )
    # load_forecast (A65/A01) is published 10:00 CET the day before delivery: only d+1 is ever safe.
    .add_transformer(LagTransformer(lags=range(-1, 0), freq="day", columns=["load_forecast"]))
    .add_validator(NullCheckValidator(columns=["load_actual", "price"]))
    .add_validator(ContinuityValidator())
)

df = pipeline.run(
    start="2021-01-01",
    end="2026-05-01",
    cache=False,
)

print(f"\nDownloaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

output_path = "data_output.csv"
df.to_csv(output_path)
