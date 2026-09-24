---
title: LagConsistencyValidator
description: Check that x_d-k columns hold x from k local calendar days earlier
---

# LagConsistencyValidator

Checks the column naming contract: `x_d-k` must hold `x` from **k local calendar days earlier** at the same wall-clock time, and `x_d+k` the value k calendar days later. On a local-time index "the same hour yesterday" is one calendar day back, not 24 elapsed hours; the two differ around DST switches, where a lag built from elapsed time takes the wrong hour.

## Basic Usage

```python
from epftoolbox2.data.validators import LagConsistencyValidator

# Every x_d-k / x_d+k column whose base column x is in the frame
result = LagConsistencyValidator().validate(df)
print(result.info["mismatch_counts"])  # e.g. {'price_d-1': 0, 'price_d-7': 0}

# Only some columns; a UTC index whose lags were built on Warsaw days
result = LagConsistencyValidator(columns=["price_d-1"], tz="Europe/Warsaw").validate(df)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | str, List[str] or None | `None` | Lag columns to check. `None` checks every `x_d-k` / `x_d+k` column whose base column `x` is in the frame. |
| `tz` | str or None | `None` | Time zone that defines the local days. `None` uses the index's own time zone. |
| `tolerance` | float | `1e-9` | Absolute difference still counted as equal (numeric columns). |

## Lookup Rule

For a row at local wall-clock time *w*, `x_d-k` must equal `x` at *w* − k days (and `x_d+k` at *w* + k days):

- **Repeated autumn hour:** the first occurrence is used when the source day has that hour only once.
- **Skipped spring hour:** the source time doesn't exist, so the lag must be missing there.
- **Source outside the frame:** can't be verified and is skipped (`unverifiable` in `stats`).
- A lag value where the base column is missing counts as a mismatch.

Only day lags are checked. Hour and minute lags (`price_h-24`) are elapsed-time lags by definition. Columns whose base isn't in the frame, such as Open-Meteo's `temperature_2m_d+1` forecasts, are skipped; don't list a forecast column in `columns` when a realised column with the base name exists.

## What It Catches

A day lag computed as "the value 24 h × k earlier" is wrong around every DST switch. On a year of hourly Europe/Warsaw data it reports:

| Column | Rows reported (hourly) | 15-minute |
|--------|------------------------|-----------|
| `price_d-1` | 49 (the switch days and the early hours of the next day) | 196 |
| `price_d-7` | 337 (a week after each switch) | 1348 |

A lag built on local calendar days passes.

## Output

- **Errors:** one per column with mismatches, with the first wrong row, its value and the expected value; columns in `columns` that aren't day-lag names or whose base column is missing.
- **Warnings:** no checkable lag column (with `columns=None`); duplicate timestamps.
- **info:** `tz`, `checked` and `mismatch_counts` (per column), `mismatches` (first 10 rows per column: timestamp, actual, expected).
- **stats:** `column`, `base`, `days` (signed shift), `checked`, `unverifiable`, `mismatches`, `first_mismatch`.

```python
print(result.errors[0])
# Column 'price_d-1' is not 'price' 1 local calendar day earlier at the same wall-clock time on
# 49 of 8760 rows (first: 2024-03-31 03:00:00+02:00 has 2137, expected 2138)
```

## When to Use

After `LagTransformer` (and any transformer that runs after it), before training. The model builds its targets `{target}_d+{h}` with calendar-day shifts, so features built another way disagree with the targets around DST.
