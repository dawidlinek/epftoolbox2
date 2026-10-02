---
title: ContinuityValidator
description: Check that the time index is a sorted, gap-free grid
---

Checks the `DatetimeIndex` itself: it must be sorted, free of duplicates, and step through time at one frequency without missing timestamps.

## Basic Usage

```python
from epftoolbox2.data.validators import ContinuityValidator

validator = ContinuityValidator()            # infer the step from the data
validator = ContinuityValidator(freq="15min")  # or state it

result = validator.validate(df)
print(result)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `freq` | str or None | `None` | Expected step (`"15min"`, `"1h"`, `"1D"`). `None` infers it as the most common step between consecutive timestamps. |

Leave `freq=None` unless you want to assert a resolution: a fixed `"1h"` on 15-minute data is reported as an error, because the data isn't on the hourly grid.

## What It Reports

All of these are errors:

| Problem | Message / info |
|---------|----------------|
| Duplicate timestamps | `duplicate_count`, `duplicates` (first 10) |
| Index not sorted | `unsorted_count` |
| Gaps (a step longer than `freq`) | `gaps` (start, end, duration, missing), `gap_count`, `missing_periods` |
| Steps that are not a multiple of `freq` (mixed resolutions, off-grid rows) | `irregular_steps`, `irregular` (first 10) |

`info` also holds `expected_freq` and `freq_inferred`. At most 10 gaps are listed in `errors`, followed by a line counting the rest. `info["gaps"]` has all of them.

```python
result = ContinuityValidator().validate(df)
print(result.errors[0])
# Gap detected (expected 15min frequency): no rows between 2024-01-01 01:00:00+00:00
# and 2024-01-01 02:00:00+00:00 (3 missing timestamps)
print(result.info["gap_count"], result.info["missing_periods"])
```

## Time Zones and DST

- Steps are measured in elapsed time, so a **tz-aware local index** is continuous across DST switches: the local clock skips or repeats an hour, the elapsed step stays one hour.
- Steps of a day or more on a tz-aware index are measured on the local calendar (a local day lasts 23 or 25 hours on switch days), so daily data at local midnight is continuous too.
- A **naive** index is compared as it is. A naive local wall-clock index therefore shows each spring switch as a gap and each autumn switch as a duplicate, which is correct: such an index can't represent the repeated hour.

## When to Use

- On raw source data, to find missing rows and mixed resolutions (for example around the 15-minute SDAC switch on 2025-10-01).
- After timezone conversion and resampling, to confirm a regular grid.

It only sees rows that are missing from the index. A column that is empty on a complete index needs [CoverageValidator](/epftoolbox2/validators/coverage/). Rows per local day are checked by [DSTStructureValidator](/epftoolbox2/validators/dst-structure/).
