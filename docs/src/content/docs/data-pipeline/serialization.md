---
title: Serialization
description: Save and load DataPipeline configurations as YAML
---

Save and load pipeline configurations for reproducibility and version control.

## Save

```python
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.data.sources import EntsoeSource, CalendarSource
from epftoolbox2.data.transformers import ResampleTransformer, LagTransformer
from epftoolbox2.data.validators import CoverageValidator

pipeline = (
    DataPipeline()
    .add_source(EntsoeSource(country_code="PL", api_key="...", type=["load", "price"]))
    .add_source(CalendarSource(country="PL", holidays="binary", weekday="onehot", daylight=True))
    .add_transformer(ResampleTransformer(freq="1h"))
    .add_transformer(LagTransformer(columns=["load_actual", "price"], lags=[1, 2, 7], freq="day"))
    .add_validator(CoverageValidator(columns=["load_actual", "price"], min_coverage=1.0))
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
timezone: Europe/Warsaw    # optional; see Timezone below
on_invalid: warn           # "warn" (default) or "raise"
sources:
- class: EntsoeSource
  params:
    country_code: PL
    api_key: ${ENTSOE_API_KEY}   # written by save() by default — see Secrets below
    type: [load, price]
- class: CalendarSource
  params:
    country: PL
    holidays: binary
    weekday: onehot
    daylight: true
source_validators:         # optional; run on raw source output, before any transformer
- class: ContinuityValidator
  params:
    freq: 1h
transformers:
- class: ResampleTransformer
  params:
    freq: 1h
- class: LagTransformer
  params:
    columns: [load_actual, price]
    lags: [1, 2, 7]
    freq: day
validators:                 # run on the final (post-transformer) frame
- class: CoverageValidator
  params:
    columns: [load_actual, price]
    min_coverage: 1.0
```

`DataPipeline.load()` raises `ValueError` on an unknown top-level key (e.g. a typo'd `timezon:`), and on an
unregistered `class` name.

## Timezone

`DataPipeline(timezone="Europe/Warsaw")` makes date strings and keywords passed to `run()` mean **local calendar
days**: `start` begins at local midnight and `end` covers its *whole local day* (inclusive), correctly sized on
DST-transition days. Without `timezone` (the default), `start`/`end` mean midnight UTC instead, with `end`
likewise covering the whole UTC day it names. See
[Date Format Options](/epftoolbox2/data-pipeline/overview/#date-format-options).

It is also the single place the market zone needs to be written: `TimezoneTransformer.target_tz`,
`CalendarSource`/`OpenMeteoSource`'s own zone may all be
left unset (`None`, the default) and they then inherit it, so `save()` writes the zone once and the components'
own `timezone`/`target_tz` come back `null`. See
[Timezone (local calendar days)](/epftoolbox2/data-pipeline/overview/#timezone-local-calendar-days).

## Validators and on_invalid

`validators` run on the final frame; `source_validators` run on the raw, joined source output, before any
transformer (useful for checks that a transformer such as `ResampleTransformer` would otherwise paper over).
Every result is kept in `pipeline.validation_results`, keyed `"source_validators.<Class>"` / `"validators.<Class>"`.
`on_invalid="warn"` (the default) logs a failure and continues; `on_invalid="raise"` raises `DataValidationError`
after the failing stage's validators have all run.

## Secrets

`api_key` (and any component's declared secret) is written as `${<SOURCE>_API_KEY}` (e.g. `${ENTSOE_API_KEY}`) by
default, and expanded from the environment on load. Pass `pipeline.save(path, include_secrets=True)` to write the
literal value instead — only do this for files you don't commit. Any string parameter may itself contain a
`${VAR}` reference (not just `api_key`); it is expanded on load, and preserved as the reference (not flattened to
the expanded value) when the pipeline is saved again.

## Relative Paths

File-like parameters (`path`, `*_path`, `*_file`, `*_dir`, e.g. `CsvSource.file_path`) loaded from a relative path
are resolved against the YAML file's directory, not the process's working directory — so a pipeline YAML and the
data it points to can be moved together. On save, a path under the YAML's directory is written relative to it
again.

## Serialization Rules

| Type | Behavior |
|------|----------|
| `str`, `int`, `float`, `bool`, `list`, `dict`, `None` | Serialized as-is |
| `Path` | Converted to `str` |
| numpy scalar / array | Converted to plain `int`/`float`/`list` |
| `tuple`, `range` (e.g. lags) | Converted to `list` |
| Private attrs (`_name`) | Excluded |
| `session`, `console`, `logger` | Excluded (runtime objects, not configuration) |
| Anything else (`set`, callable, arbitrary object) | Raises `TypeError` naming the parameter — nothing is dropped silently |

## Registering Custom Sources/Transformers/Validators

Add to `COMPONENT_REGISTRY` in `data_pipeline.py`:

```python
COMPONENT_REGISTRY["sources"]["MySource"] = "mypackage.sources.my_source"
COMPONENT_REGISTRY["transformers"]["MyTransformer"] = "mypackage.transformers.my_transformer"
COMPONENT_REGISTRY["validators"]["MyValidator"] = "mypackage.validators.my_validator"
```
