---
title: FreshnessValidator
description: Check that every column has values up to an expected end
---

# FreshnessValidator

Checks the last value of each column against an expected end. Use it before a live forecast to make sure every input has been published, or to find a source that stopped updating while the others went on.

## Basic Usage

```python
from epftoolbox2.data.validators import FreshnessValidator

# Every column must reach the last row of the frame
result = FreshnessValidator().validate(df)

# Live forecasting: data needed up to tomorrow 00:00 UTC, three hours of slack
result = FreshnessValidator(
    columns=["price", "load_forecast"],
    expected_end="today_d+1",
    max_delay="3h",
).validate(df)
print(result.info["stale_columns"])
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | str, List[str] or None | `None` | Columns to check. `None` checks every column. |
| `expected_end` | str or None | `None` | Timestamp of the last value you need. `None` uses the last row of the frame. |
| `max_delay` | str | `"0h"` | How far before `expected_end` a column's last value may be. |

`expected_end` accepts a timestamp or the `today` / `today_d+N` keywords of `DataPipeline.run`, resolved each time the validator runs. Keywords give a date, i.e. midnight UTC of the machine's local date; naive timestamps are UTC.

## Output

- **Errors:** missing columns; columns with no values; stale columns, whose last value is earlier than `expected_end - max_delay`.
- **info:** `expected_end`, `last_valid` (per column), `stale_columns`.
- **stats:** `column`, `last_valid`, `behind` (`expected_end` minus the last value; negative when the column goes further).

```python
print(result.errors[0])
# Column 'price' ends at 2024-06-03 23:00:00+02:00, 2 days 00:00:00 before the expected end 2024-06-05 23:00:00+02:00
```

## When to Use

- Right before issuing a forecast for today.
- After merging several sources, to see whether one of them lags behind.
