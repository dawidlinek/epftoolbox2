---
title: Pipeline Examples
description: DataPipeline examples, from a minimal ENTSO-E pipeline to a full Polish one with weather and validators
---

Each example adds to the one before it. The last two are complete pipelines you can copy as they are.

## 1. ENTSO-E and calendar

```python
import os
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.data.sources import EntsoeSource, CalendarSource
from epftoolbox2.data.transformers import TimezoneTransformer, ResampleTransformer

pipeline = (
    DataPipeline(timezone="Europe/Warsaw")
    .add_source(EntsoeSource(country_code="PL", api_key=os.environ["ENTSOE_API_KEY"], type=["load", "price"]))
    .add_source(CalendarSource(country="PL", holidays="binary", weekday="number"))
    .add_transformer(TimezoneTransformer())
    .add_transformer(ResampleTransformer(freq="1h"))
)

df = pipeline.run(start="2024-01-01", end="2024-12-31", cache=True)
```

`end` is inclusive, so this covers all of 31 December in Warsaw time. The result is an hourly frame with a Warsaw
index and the columns `load_actual`, `load_forecast`, `load_forecast_daily_min`, `load_forecast_daily_max`,
`price`, `is_holiday` and `weekday`.

## 2. Lagged features

```python
from epftoolbox2.data.transformers import LagTransformer

pipeline.add_transformer(LagTransformer(columns=["load_actual", "price"], lags=[1, 2, 7], freq="day"))
pipeline.add_transformer(LagTransformer(columns=["load_forecast", "is_holiday", "weekday"], lags=[-1], freq="day"))
```

A positive lag looks back, so `price_d-1` is the price at the same local hour yesterday. A negative lag looks
forward, so `is_holiday_d+1` is the calendar of the day being forecast and `load_forecast_d+1` is its day-ahead
load forecast.

## 3. Weather for the whole country

```python
from epftoolbox2.data.sources import OpenMeteoSource

pipeline.add_source(OpenMeteoSource(
    locations={
        "warsaw": {"latitude": 52.2297, "longitude": 21.0122, "weight": 1.86},
        "krakow": {"latitude": 50.0647, "longitude": 19.9450, "weight": 0.80},
        "gdansk": {"latitude": 54.3520, "longitude": 18.6466, "weight": 0.47},
    },
    agg="mean",
    columns=["temperature_2m", "wind_speed_10m", "cloud_cover"],
    prefix="pl",
))
```

The three cities become one population-weighted series per variable, with the columns
`pl_temperature_2m_mean_d+1` to `pl_temperature_2m_mean_d+5` and the same for the other variables. They already
hold the forecast for the target day, so they need no `LagTransformer`. See
[OpenMeteoSource](/epftoolbox2/data-sources/openmeteo/) for the other options.

## 4. A neighbouring zone

```python
pipeline.add_source(EntsoeSource(country_code="DE_LU", api_key=os.environ["ENTSOE_API_KEY"], type=["price"], prefix="de"))
pipeline.add_transformer(LagTransformer(columns=["de_price"], lags=[1, 2, 7], freq="day"))
```

The `prefix` keeps the German price (`de_price`) apart from the Polish one. Day-ahead prices of all zones are
published together, so the neighbour's price is used only through its lags.

## 5. Validators

```python
from epftoolbox2.data.validators import (
    CoverageValidator,
    ContinuityValidator,
    DSTStructureValidator,
    PlausibilityValidator,
)

pipeline.add_validator(CoverageValidator(columns=["load_actual", "price"], min_coverage=1.0))
pipeline.add_validator(ContinuityValidator())
pipeline.add_validator(DSTStructureValidator())
pipeline.add_validator(PlausibilityValidator())
```

Validators do not change the data. A failure is logged as a warning, and `DataPipeline(on_invalid="raise")` makes it
raise `DataValidationError` instead.

## 6. A 15-minute grid

```python
pipeline = (
    DataPipeline(timezone="Europe/Warsaw")
    .add_source(EntsoeSource(
        country_code="PL",
        api_key=os.environ["ENTSOE_API_KEY"],
        type=["load", "price"],
        prefer_15min=True,
    ))
    .add_source(CalendarSource(country="PL", holidays="binary", weekday="number", freq="15min"))
    .add_transformer(TimezoneTransformer())
    .add_transformer(ResampleTransformer(freq="15min"))
    .add_transformer(LagTransformer(columns=["load_actual", "price"], lags=[1, 7], freq="day"))
)

df = pipeline.run(start="2025-10-01", end="2025-12-31", cache=True)
```

`prefer_15min=True` gives a 15-minute price, and `freq="15min"` on the calendar and the resample puts everything on
the same grid, 96 rows per local day (92 or 100 on DST days). See [EntsoeSource](/epftoolbox2/data-sources/entsoe/#price-resolution) for
the price resolution.

## 7. A full Polish pipeline

```python
import os
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.data.sources import EntsoeSource, CalendarSource, OpenMeteoSource
from epftoolbox2.data.transformers import TimezoneTransformer, ResampleTransformer, LagTransformer
from epftoolbox2.data.validators import (
    CoverageValidator,
    ContinuityValidator,
    DSTStructureValidator,
    PlausibilityValidator,
)

api_key = os.environ["ENTSOE_API_KEY"]

pipeline = (
    DataPipeline(timezone="Europe/Warsaw")
    .add_source(EntsoeSource(country_code="PL", api_key=api_key, type=["load", "price"]))
    .add_source(EntsoeSource(country_code="DE_LU", api_key=api_key, type=["price"], prefix="de"))
    .add_source(CalendarSource(
        country="PL",
        holidays="binary",
        weekday="number",
        bridge_days=True,
        daylight=True,
        school_holidays="all",
    ))
    .add_source(OpenMeteoSource(
        locations={
            "warsaw": {"latitude": 52.2297, "longitude": 21.0122, "weight": 1.86},
            "krakow": {"latitude": 50.0647, "longitude": 19.9450, "weight": 0.80},
            "lodz": {"latitude": 51.7592, "longitude": 19.4560, "weight": 0.65},
            "wroclaw": {"latitude": 51.1079, "longitude": 17.0385, "weight": 0.67},
            "poznan": {"latitude": 52.4064, "longitude": 16.9252, "weight": 0.54},
            "gdansk": {"latitude": 54.3520, "longitude": 18.6466, "weight": 0.47},
        },
        agg="mean",
        columns=["temperature_2m", "wind_speed_10m", "cloud_cover", "relative_humidity_2m"],
        prefix="pl",
    ))
    .add_transformer(TimezoneTransformer())
    .add_transformer(ResampleTransformer(freq="1h"))
    .add_transformer(LagTransformer(columns=["load_actual", "price", "de_price"], lags=range(1, 8), freq="day"))
    .add_transformer(LagTransformer(columns=["load_forecast"], lags=[-1], freq="day"))
    .add_transformer(LagTransformer(
        columns=["is_holiday", "is_bridge_day", "weekday", "daylight_hours", "school_holiday_share"],
        lags=range(-5, 0),
        freq="day",
    ))
    .add_validator(CoverageValidator(columns=["load_actual", "price", "de_price"], min_coverage=1.0))
    .add_validator(ContinuityValidator())
    .add_validator(DSTStructureValidator())
    .add_validator(PlausibilityValidator())
)

df = pipeline.run(start="2023-01-01", end="2025-12-31", cache=True)
pipeline.save("data_pipeline.yaml")
```

The saved YAML keeps the API key as `${ENTSOE_API_KEY}`, so it can be committed and reloaded with
`DataPipeline.load("data_pipeline.yaml")`. See [Serialization](/epftoolbox2/data-pipeline/serialization/).
