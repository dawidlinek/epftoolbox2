---
title: Pipeline Examples
description: Complete DataPipeline examples
---

import { Tabs, TabItem, Aside } from '@astrojs/starlight/components';

# Pipeline Examples

## Basic Example: Polish Electricity Data

```python
import os
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.data.sources import EntsoeSource, CalendarSource
from epftoolbox2.data.transformers import ResampleTransformer, LagTransformer
from epftoolbox2.data.validators import NullCheckValidator, EdaValidator

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
    .add_validator(NullCheckValidator(columns=["load_actual", "price"]))
    .add_validator(EdaValidator())
)

df = pipeline.run(start="2023-01-01", end="2024-01-01", cache=True)
```

## With Weather Forecasts

```python
from epftoolbox2.data.sources import OpenMeteoSource

pipeline = (
    DataPipeline()
    .add_source(EntsoeSource(
        country_code="PL",
        api_key=os.environ["ENTSOE_API_KEY"],
        type=["load", "price"],
    ))
    .add_source(OpenMeteoSource(
        latitude=52.2297,
        longitude=21.0122,
        horizon=7,
        prefix="warsaw",
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

pipeline = DataPipeline()

# Add ENTSOE source
pipeline.add_source(EntsoeSource(
    country_code="PL",
    api_key=os.environ["ENTSOE_API_KEY"],
    type=["load", "price"],
))

# Add weather for each city
for name, (lat, lon) in cities.items():
    pipeline.add_source(OpenMeteoSource(
        latitude=lat,
        longitude=lon,
        horizon=7,
        prefix=name,
    ))

df = pipeline.run(start="2023-01-01", end="2024-01-01", cache=True)
```
