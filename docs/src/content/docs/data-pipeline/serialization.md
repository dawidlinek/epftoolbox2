---
title: Serialization
description: Save and load DataPipeline configurations as YAML
---

# DataPipeline Serialization

Save and load pipeline configurations for reproducibility and version control.

## Save

```python
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.data.sources import EntsoeSource, CalendarSource
from epftoolbox2.data.transformers import ResampleTransformer, LagTransformer
from epftoolbox2.data.validators import NullCheckValidator

pipeline = (
    DataPipeline()
    .add_source(EntsoeSource(country_code="PL", api_key="...", type=["load", "price"]))
    .add_source(CalendarSource(country="PL", holidays="binary", weekday="onehot", daylight=True))
    .add_transformer(ResampleTransformer(freq="1h"))
    .add_transformer(LagTransformer(columns=["load_actual", "price"], lags=[1, 2, 7], freq="day"))
    .add_validator(NullCheckValidator(columns=["load_actual", "price"]))
)

pipeline.save("data_pipeline.yaml")
```

## Load

```python
pipeline = DataPipeline.load("data_pipeline.yaml")
df = pipeline.run(start="2024-01-01", end="2024-06-01", cache=True)
```

## YAML Format

```yaml
sources:
- class: EntsoeSource
  params:
    country_code: PL
    api_key: YOUR_KEY
    type: [load, price]
- class: CalendarSource
  params:
    country: PL
    holidays: binary
    weekday: onehot
    daylight: true
transformers:
- class: ResampleTransformer
  params:
    freq: 1h
- class: LagTransformer
  params:
    columns: [load_actual, price]
    lags: [1, 2, 7]
    freq: day
validators:
- class: NullCheckValidator
  params:
    columns: [load_actual, price]
```

## Serialization Rules

| Type | Behavior |
|------|----------|
| `str`, `int`, `float`, `bool`, `list`, `dict`, `None` | Serialized as-is |
| `Path` | Converted to `str` |
| `range` (lags) | Converted to `list` at construction — serializes correctly |
| Private attrs (`_name`) | Excluded |
| `session`, `console`, `logger`, `lat`, `lon` | Excluded |

## Registering Custom Sources/Transformers/Validators

Add to `COMPONENT_REGISTRY` in `data_pipeline.py`:

```python
COMPONENT_REGISTRY["sources"]["MySource"] = "mypackage.sources.my_source"
COMPONENT_REGISTRY["transformers"]["MyTransformer"] = "mypackage.transformers.my_transformer"
COMPONENT_REGISTRY["validators"]["MyValidator"] = "mypackage.validators.my_validator"
```
