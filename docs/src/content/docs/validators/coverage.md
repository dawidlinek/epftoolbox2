---
title: CoverageValidator
description: Per-column coverage, longest missing run and first/last value over a range
---

# CoverageValidator

Measures, per column, which timestamps of a regular grid carry a value: coverage, the longest run of missing timestamps and the first and last value. It finds what the other checks can't:

- an **outage of one column on a complete index** (for example three days without prices while load keeps coming), which `ContinuityValidator` can't see;
- **rows missing from the index entirely**, which a null check on the rows present can't see;
- a column that **starts late or ends early** compared with the range you asked for.

## Basic Usage

```python
from epftoolbox2.data.validators import CoverageValidator

# Internal holes only: each column is measured between its own first and last value
result = CoverageValidator(columns=["price", "load_actual"]).validate(df)

# Against the range passed to DataPipeline.run (end exclusive)
result = CoverageValidator(
    columns=["price"],
    start="2024-01-01",
    end="2025-01-01",
    freq="1h",
    min_coverage=0.999,
    max_gap="6h",
).validate(df)
print(result.stats)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | str, List[str] or None | `None` | Columns to check. `None` checks every column. |
| `start` | str or None | `None` | First expected timestamp. `None` uses each column's first value. |
| `end` | str or None | `None` | End of the range, **exclusive**. `None` uses each column's last value. |
| `freq` | str or None | `None` | Grid step (`"15min"`, `"1h"`). `None` infers it per column from the column's own values, so a raw frame with hourly prices and 15-minute load measures each on its own grid. |
| `min_coverage` | float | `1.0` | Smallest share of expected timestamps that must have a value. |
| `max_gap` | str or None | `None` | Longest missing run allowed, as a duration (`"6h"`, `"2D"`). `None` means no limit. |

`start` and `end` accept dates, timestamps and the `today` / `today_d-1` keywords of `DataPipeline.run`. They are resolved each time the validator runs, and naive values are UTC, like pipeline dates. The grid follows the phase of the column's values, so a UTC-midnight start works for a local-time frame.

With `start`/`end` left at `None`, lag columns pass even though they are empty at the edges of the frame by construction, and so do sources that pad the frame (A31 starting the evening before, Open-Meteo running past `end`). Set `start`/`end` when late starts and early ends matter.

## Output

- **Errors:** missing columns; columns with no values; columns below `min_coverage` or with a missing run longer than `max_gap`.
- **Warnings:** columns with missing timestamps that are still within both limits.
- **info:** `freq`, `coverage`, `longest_missing_run` (in timestamps), `first_valid`, `last_valid` (all per column).
- **stats:** `column`, `freq`, `start`, `end`, `expected`, `present`, `missing`, `coverage`, `longest_missing_run`, `longest_missing_start`, `longest_missing_end`, `first_valid`, `last_valid`.

```python
print(result.errors[0])
# Column 'price' has values at 168 of 240 expected 1h timestamps in [2024-01-01 00:00:00+00:00,
# 2024-01-11 00:00:00+00:00) (70.00%, min 100.00%); longest missing run 72 timestamps (3D)
# from 2024-01-03 00:00:00+00:00 to 2024-01-05 23:00:00+00:00
```

## Time Zones

Sub-daily grids run in elapsed time, so DST days have 23/25 hourly timestamps on a local index. Daily grids on a local index follow the local calendar.

## When to Use

- On raw source data, before any transformer fills holes: interpolation hides outages.
- On the columns the model uses (target and predictors), with the backtest range.
