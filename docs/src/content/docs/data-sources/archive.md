---
title: ArchiveSource
description: A snapshot archive replayed as an honest vintage, each delivery day as it had been downloaded by a stated time
---

A provider's API answers with today's values. A TSO that revises its load forecast after the gate, or corrects an actual a week
later, has rewritten what a backtest reads, and the backtest then uses values no forecast issued on the day could have seen. A
[snapshot archive](/epftoolbox2/operations/snapshots/) that a scheduled job filled keeps every download as it came, with the
instant it came. `ArchiveSource` reads one source's downloads back as a data source: each delivery day as the archive had it at a
stated time, with the availability rule that says exactly that. A backtest over the days the archive covers then reads nothing
before the archive had it.

## Basic Usage

```python
from epftoolbox2.data.sources import ArchiveSource, EntsoeSource
from epftoolbox2.data.transformers import LagTransformer, TimezoneTransformer
from epftoolbox2.pipelines import DataPipeline, ModelPipeline

data = (
    DataPipeline(timezone="Europe/Brussels")
    .add_source(EntsoeSource("BE", api_key="your-key", type=["price"]))
    # The day-ahead load forecast as the archive held it at 11:45 on the day before each delivery day
    .add_source(ArchiveSource("archive/be", "EntsoeSource", known_at="11:45", days_before=1, columns=["load_forecast"]))
    .add_transformer(TimezoneTransformer())
    .add_transformer(LagTransformer(["archive_load_forecast"], [-1], "day"))  # the forecast for D+1, as of 11:45 on D
)
df = data.run("2026-12-01", "2027-03-31")  # days the archive covers

pipeline = ModelPipeline(issue_time="11:45", issue_timezone="Europe/Brussels")
# Outside a Workflow, tell the information set about the sources: only it knows the archive's rule.
pipeline.information_set = pipeline.information_set.with_source_rules(data.sources)
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | str | Yes | - | The archive's directory, as `SnapshotArchive(path)` filled it. Relative to the working directory, or in YAML to the file |
| `source` | str | Yes | - | The archived source's key (`SnapshotArchive(path).keys()`, such as `EntsoeSource-3f2a9c41d07e5b18`), or its class name when the archive holds one configuration of that class |
| `known_at` | str | No | `"11:45"` | The time of day, `"HH:MM"` in `timezone`, each delivery day is read at |
| `days_before` | int | No | `1` | Days from the delivery day back to the day it is read on: `1` the day before, `0` the day itself, `-1` the day after |
| `timezone` | str | No | `"Europe/Brussels"` | The zone of the delivery days and of `known_at` |
| `columns` | list[str] | No | `None` | The archived columns to emit, by their names in the archive. `None` emits every column the source's fetches have |
| `prefix` | str | No | `"archive"` | Joined to every emitted name with `_`: `load_forecast` comes back as `archive_load_forecast`. `""` keeps the archived names |

`fetch(start, end)` returns the cells in **[start, end)** with a UTC index.

## What Each Day Holds

For a delivery day d, a local date of `timezone`, every cell of d (each instant whose date in `timezone` is d) takes its value
from **the newest fetch that had returned by `known_at` on d − `days_before`** and that asked for the cell:

- *Returned by*: a fetch counts from the instant its answer was in hand (`fetched_at` in the catalog), never from when its
  request started.
- *Asked for the cell*: each fetch asked for a range of whole local days; one that didn't ask for d says nothing about it, and an
  older one that did is used.
- Where that fetch has no value (NaN, or no row), the cell is missing, even if an older fetch had a value: it is what a refresh at
  that time saw. Days before the archive's first fetch have no values.
- The cut-off is a wall-clock time of `timezone`. A time the clocks repeat (02:30 on the day they go back) is read at its second
  occurrence, one they skip (02:30 on the day they go forward) at the switch, as the information set reads a publication time.
  A 25-hour day has its 25 hours, each from the same fetch.

With recordings every day at 09:00, 10:30, 11:30 and 12:30, and a load forecast published at 10:00 on the day before delivery:

| `known_at`, `days_before` | Delivery day d is read from | It holds |
|---|---|---|
| `"11:45"`, `1` | the 11:30 fetch on d − 1 | the forecast with its revisions up to about 11:30 |
| `"11:45"`, `1`, the 11:30 run missed | the 10:30 fetch on d − 1 | the first publication, without the 11:05 revision |
| `"10:15"`, `1` | the 09:00 fetch on d − 1 | nothing: not yet published |
| `"12:00"`, `-1` | the 11:30 fetch on d + 1 | the forecast as revised by then; for an actual, d's values as reported by then |

Each day is read on its own, so a sub-range holds the same values as a larger range.

## Its Rule

Every column it emits has the rule `Availability.published(known_at, days_before, timezone=timezone, vintage="snapshot")`:
published at `known_at` on the day `days_before` before the delivery day, its archive a snapshot (one value per cell, as fetched
by then). `InformationSet.with_source_rules(data.sources)` gives it, and a `Workflow` applies it. It comes before every rule,
assumption and pattern of the information set, as the values were chosen by it: a rule of yours that names one of its columns
(or a lag, profile or day statistic of one) raises `InformationSetError`, and a pattern such as `load_*` is left alone for the
other columns it matches.

Choose `known_at` and `days_before` by what the column is:

- **A forecast for the next day** (load, wind and solar day-ahead forecasts): `days_before=1` at the issue time of the model that
  reads it, so `archive_load_forecast_d+1` on the row of D is the forecast for D+1 as of the issue.
- **An actual** (actual load, generation): it isn't out the day before. `days_before=-1` reads d's values on d + 1, with the
  corrections made by then and none after; lag it as `_d-1` or older, which is then known on the issue day.
- Several vintages of one archive are several `ArchiveSource`s with different `columns` or `prefix`.

The default `prefix` matters outside a `Workflow`: the archived names (`load_forecast`, `price`) have built-in rules of the live
API, and most prefixed names have none, so a run that forgot `with_source_rules` reports the column as having no rule instead of
judging the archive's values by the API's publication times. A built-in pattern can still match a prefixed name:
`*_price_exaa` catches `archive_price_exaa`. So the first `fetch` warns about every emitted name a bare `InformationSet()`
gives a rule (exact or by a pattern), with `prefix=""` or without: with `with_source_rules` the warning says nothing new, as the
archive's rule comes first. A pattern of yours that matches an archive column warns in `with_source_rules`: it is ignored for
that column, also where it is stricter.

## What It Refuses

- A `source` the archive doesn't hold, or a class name with several configurations in it (`ArchiveError`, naming the keys).
- `columns` that no fetch of the source holds (`ArchiveError`, listing its columns).
- A `known_at` that isn't `"HH:MM"`, a `days_before` that isn't a whole number, an unknown `timezone`, a `prefix` ending in `_`.
- A file whose bytes don't match the sha256 its catalog line recorded.

## Caching

`get_cache_config()` is `None`: the archive grows with every recording, so a per-source cache would serve what it held before.
`content_digest()` is the sha256 of the archive's catalog, so the single-file pipeline cache (`cache="file.parquet"`) fetches
again after a new recording and reuses its file otherwise.

## YAML

```yaml
sources:
  - class: ArchiveSource
    params:
      path: archive/be
      source: EntsoeSource
      known_at: "11:45"
      days_before: 1
      timezone: Europe/Brussels
      columns: [load_forecast]
      prefix: archive
```

A relative `path` is read against the YAML file's directory.
