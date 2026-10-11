---
title: DSTStructureValidator
description: Check rows per local day (23, 24 or 25 hours times periods per hour)
---

Checks that every local calendar day has one row per period. The number of hours in each day comes from the time zone: in Europe the spring switch day has 23 hours and the autumn one 25, so an hourly frame must have 23/24/25 rows per day and a 15-minute frame 92/96/100.

## Basic Usage

```python
from epftoolbox2.data.validators import DSTStructureValidator

# Frame after TimezoneTransformer: local days come from the index
result = DSTStructureValidator().validate(df)

# Raw source frame (UTC index): say which local days to use
result = DSTStructureValidator(tz="Europe/Warsaw").validate(raw_df)
print(result.info["rows_per_day"])  # e.g. {23: 1, 24: 364, 25: 1}
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tz` | str or None | `None` | Time zone that defines the local days. `None` uses the index's own time zone. A naive index counts as UTC (the source contract). |
| `freq` | str or None | `None` | Period length (`"15min"`, `"1h"`). `None` infers it from the data. |
| `allow_partial_edges` | bool | `True` | Report a first or last day with too few rows as a warning instead of an error. |

Pipeline dates are UTC midnights, so after conversion to local time the first local day usually starts at 01:00 and the last one is cut short. That is why partial edge days are only warnings by default.

## What It Catches

- A **naive wall-clock index**: its spring day has 23 rows and its autumn day 25, but a naive index has no DST, so both are wrong.
- A **filled-in or duplicated DST hour** (e.g. 24 rows on the spring switch day).
- Missing or duplicated rows on ordinary days, and whole missing days (0 rows).
- A `freq` that doesn't match the data.

## Output

- **Errors:** days whose row count differs from the expected one (first 10 listed).
- **Warnings:** partial first/last days (with `allow_partial_edges=True`) and daily data, which is skipped.
- **info:** `tz`, `freq`, `rows_per_day` (row count → number of days), `dst_days` (days not 24 hours long), `bad_days` and `partial_days` (date, rows, expected).
- **stats:** one row per local day: `date`, `hours`, `rows`, `expected`.

```python
result = DSTStructureValidator().validate(df)
print(result.errors[0])
# Local day 2024-03-31 (24 h in UTC) has 23 rows, expected 24
```

## When to Use

- On raw source data with `tz=` set to the market's time zone, to see whether any delivery day is incomplete.
- After `TimezoneTransformer` and `ResampleTransformer`, to confirm the frame keeps 23/25-row DST days. Forecasting models build one model per period of the local day, so a wrong row count shifts periods.
