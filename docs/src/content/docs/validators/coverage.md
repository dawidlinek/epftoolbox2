---
title: CoverageValidator
description: Index checks plus per-column coverage, longest missing run and first/last value over a range
---

Checks the `DatetimeIndex` and measures, per column, which timestamps of a regular grid carry a value: coverage, the longest run of missing timestamps and the first and last value. It finds:

- an **outage of one column on a complete index** (for example three days without prices while load keeps coming), which a check of the index alone can't see.
- **rows missing from the index entirely**, which a null check on the rows present can't see.
- a column that **starts late or ends early** compared with the range you asked for.
- **duplicate timestamps** and an **unsorted index**.
- values that **lie between the grid points**, such as 15-minute data checked against `freq="1h"`.

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

With `end` left at `None`, the range ends one step after the column's last value on the grid.

With `start`/`end` left at `None`, lag columns pass even though they are empty at the edges of the frame by construction, and so do sources that pad the frame (A31 starting the evening before, Open-Meteo running past `end`). Set `start`/`end` when late starts and early ends matter.

## Output

- **Errors:** duplicate timestamps, an unsorted index, missing columns, columns with no values, columns with values between the grid points, and columns below `min_coverage` or with a missing run longer than `max_gap`.
- **Warnings:** columns with missing timestamps that are still within both limits.
- **info:** `duplicate_count`, `unsorted_count` and, when there are duplicates, `duplicates` (the first 10 repeated timestamps). Per column: `freq`, `coverage`, `longest_missing_run` (in timestamps), `first_valid`, `last_valid`, `off_grid` (values between the grid points).
- **stats:** `column`, `freq`, `start`, `end`, `expected`, `present`, `missing`, `coverage`, `longest_missing_run`, `longest_missing_start`, `longest_missing_end`, `first_valid`, `last_valid`, `off_grid`.

The index checks look at the whole index, whichever `columns` you select. A duplicate timestamp or an unsorted index is an error even when every grid point has a value.

```python
print(result.errors[0])
# Column 'price' has values at 168 of 240 expected 1h timestamps in [2024-01-01 00:00:00+00:00,
# 2024-01-11 00:00:00+00:00) (70.00%, min 100.00%); longest missing run 72 timestamps (3D)
# from 2024-01-03 00:00:00+00:00 to 2024-01-05 23:00:00+00:00
```

## Time Zones

Sub-daily grids run in elapsed time, so a tz-aware local index is on the grid across DST switches: the local clock skips or repeats an hour, the elapsed step stays one hour, and DST days have 23 or 25 hourly timestamps. Steps of a day or more on a tz-aware index follow the local calendar, so daily data at local midnight is on the grid too and a missing local day counts as one missing timestamp even on a 23 or 25 hour day.

A **naive** index is taken as it is. A naive local wall-clock index therefore shows each autumn switch as a duplicate timestamp and each spring switch as a missing one, which is correct: such an index can't represent the repeated hour. [DSTStructureValidator](/epftoolbox2/validators/dst-structure/) checks the rows per local day.

## When to Use

- On raw source data, to find missing rows, duplicates and mixed resolutions (for example around the 15-minute SDAC switch on 2025-10-01), before any transformer fills holes: interpolation hides outages.
- After timezone conversion and resampling, to confirm a regular grid.
- On the columns the model uses (target and predictors), with the backtest range.
