"""
Example 1: Data Pipeline Only

This example demonstrates how to use the DataPipeline to download and process
electricity market data from ENTSOE, weather data from Open-Meteo, and
calendar features - without running any models.
"""

import os
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.data.sources import EntsoeSource, OpenMeteoSource, CalendarSource
from epftoolbox2.data.transformers import ResampleTransformer, LagTransformer
from epftoolbox2.data.validators import NullCheckValidator, ContinuityValidator

ENTSOE_API_KEY = os.environ.get("ENTSOE_API_KEY", "YOUR_API_KEY_HERE")

pipeline = (
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
            holidays="binary",  # 1 if holiday, 0 otherwise
            weekday="number",  # 0=Monday, 6=Sunday
            hour="number",  # Hour of day (0-23)
        )
    )
    .add_transformer(ResampleTransformer(freq="1h"))
    .add_transformer(
        LagTransformer(
            columns=["load_actual"],
            lags=[1, 2, 7],
            freq="day",
        )
    )
    .add_validator(NullCheckValidator(columns=["load_actual", "price"]))
    .add_validator(ContinuityValidator())
)

df = pipeline.run(
    start="2023-01-01",
    end="2024-04-01",
    cache=True,
)

print(f"\nDownloaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

output_path = "data_output.csv"
df.to_csv(output_path)
