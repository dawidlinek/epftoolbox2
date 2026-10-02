---
title: Pipeline Examples
description: Complete DataPipeline examples
---

## Basic Example: Polish Electricity Data

```python
import os
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.data.sources import EntsoeSource, CalendarSource
from epftoolbox2.data.transformers import ResampleTransformer, LagTransformer
from epftoolbox2.data.validators import CoverageValidator, EdaValidator

pipeline = (
    DataPipeline()
    .add_source(EntsoeSource(
        country_code="PL",
        api_key=os.environ["ENTSOE_API_KEY"],
        type=["load", "price"],
    ))
    .add_source(CalendarSource(
        country="PL",
        holidays="binary",
        weekday="number",
        hour="number",
    ))
    .add_transformer(ResampleTransformer(freq="1h"))
    .add_transformer(LagTransformer(
        columns=["load_actual"],
        lags=[1, 7],
        freq="day",
    ))
    .add_validator(CoverageValidator(columns=["load_actual", "price"], min_coverage=1.0))
    .add_validator(EdaValidator())
)

df = pipeline.run(start="2023-01-01", end="2024-01-01", cache=True)
```

## With Weather Forecasts

```python
from epftoolbox2.data.sources import OpenMeteoSource

pipeline = (
    DataPipeline(timezone="Europe/Warsaw")
    .add_source(EntsoeSource(
        country_code="PL",
        api_key=os.environ["ENTSOE_API_KEY"],
        type=["load", "price"],
    ))
    .add_source(OpenMeteoSource(
        latitude=52.2297,
        longitude=21.0122,
        # 5 is the largest horizon jma_seamless (the default model) can serve leak-free at the default
        # api="previous_runs"/run="00".
        horizon=5,
        prefix="warsaw",
        # timezone is left unset: it inherits DataPipeline.timezone above.
    ))
    .add_transformer(ResampleTransformer(freq="1h"))
)

df = pipeline.run(start="2023-01-01", end="2024-01-01", cache=True)
```

## Multiple Weather Locations

```python
cities = {
    "warsaw": (52.2297, 21.0122),
    "krakow": (50.0647, 19.9450),
    "gdansk": (54.3520, 18.6466),
}

pipeline = DataPipeline(timezone="Europe/Warsaw")

# Add ENTSOE source
pipeline.add_source(EntsoeSource(
    country_code="PL",
    api_key=os.environ["ENTSOE_API_KEY"],
    type=["load", "price"],
))

# Add weather for each city; timezone is left unset on each and inherits DataPipeline.timezone above.
for name, (lat, lon) in cities.items():
    pipeline.add_source(OpenMeteoSource(
        latitude=lat,
        longitude=lon,
        horizon=5,
        prefix=name,
    ))

df = pipeline.run(start="2023-01-01", end="2024-01-01", cache=True)
```
