"""
Example 1: Data Pipeline Only

This example demonstrates how to use the DataPipeline to download and process
electricity market data from ENTSOE, weather data from Open-Meteo, and
calendar features - without running any models.
"""

import os
from pathlib import Path

from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.data.sources import EntsoeSource, OpenMeteoSource, CalendarSource
from epftoolbox2.data.transformers import ResampleTransformer, LagTransformer, TimezoneTransformer
from epftoolbox2.data.validators import CoverageValidator

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "YOUR_API_KEY_HERE")


def main(data_start="2021-01-01", data_end="2026-05-01", output_dir="."):
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
                prefix="warsaw",
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
        .add_transformer(ResampleTransformer(freq="1h"))
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
        .add_transformer(
            LagTransformer(
                lags=range(-3, 1),
                freq="day",
                columns=["load_forecast_daily_min", "load_forecast_daily_max"],
            )
        )
        .add_transformer(LagTransformer(lags=range(-1, 0), freq="day", columns=["load_forecast"]))
        .add_validator(CoverageValidator(columns=["load_actual", "price"], min_coverage=1.0))
    )

    df = pipeline.run(
        start=data_start,
        end=data_end,
        cache=False,
    )

    print(f"\nDownloaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())

    # Example 2 reads this file
    df.to_csv(Path(output_dir) / "data_output.csv")
    return df


if __name__ == "__main__":
    main()
