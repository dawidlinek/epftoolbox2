---
title: Pipeline Serialization
description: Save and load pipeline configurations
---

# Pipeline Serialization

Save and load pipeline configurations for reproducibility.

## Save Pipeline

```python
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.data.sources import EntsoeSource, CalendarSource
from epftoolbox2.data.transformers import ResampleTransformer

pipeline = (
    DataPipeline()
    .add_source(EntsoeSource(country_code="PL", api_key="...", type=["load"]))
    .add_source(CalendarSource(country="PL"))
    .add_transformer(ResampleTransformer(freq="1h"))
)

# Save to YAML
pipeline.save("pipeline_config.yaml")
```

## Load Pipeline

```python
from epftoolbox2.pipelines import DataPipeline

pipeline = DataPipeline.load("pipeline_config.yaml")
df = pipeline.run(start="2024-01-01", end="2024-06-01")
```

## Configuration Format

```yaml
sources:
  - type: EntsoeSource
    country_code: PL
    api_key: ${ENTSOE_API_KEY}
    type:
      - load
      - price
  - type: CalendarSource
    country: PL
    holidays: binary

transformers:
  - type: ResampleTransformer
    freq: 1h

validators:
  - type: NullCheckValidator
    columns:
      - load_actual
      - price
```

## Environment Variables

Use `${VAR_NAME}` syntax to reference environment variables:

```yaml
sources:
  - type: EntsoeSource
    api_key: ${ENTSOE_API_KEY}
```
