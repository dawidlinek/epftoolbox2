---
title: NullCheckValidator
description: Check that columns exist and have few enough null values
---

# NullCheckValidator

Checks that the listed columns exist and counts their missing values, with the longest run of consecutive missing rows.

## Basic Usage

```python
from epftoolbox2.data.validators import NullCheckValidator

validator = NullCheckValidator(
    columns=["load_actual", "price"],
    allow_nulls=False,
    max_null_frac=0.0,
)

result = validator.validate(df)
print(result.is_valid)  # True/False
print(result.stats)     # one row per checked column
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | str, List[str] or None | `None` | Columns to check. `None` checks every column. |
| `allow_nulls` | bool | `False` | Report missing values as warnings only; they never fail the check. |
| `max_null_frac` | float | `0.0` | Largest share of missing values allowed per column (0 to 1). A column with missing values within the limit produces a warning. |

Only the listed columns are null-checked. List the columns that matter: lag columns such as `price_d-7` are empty for the first week of the frame by construction, so `columns=None` on a frame with lags always reports them.

## Output

- **Errors:** missing columns; columns whose share of missing values is above `max_null_frac`.
- **Warnings:** columns with missing values within the limit (or any missing values with `allow_nulls=True`).
- **info:** `missing_columns`, and for columns with missing values `null_counts`, `null_fraction`, `longest_null_run` (in rows).
- **stats:** `column`, `rows`, `null_count`, `null_frac`, `longest_null_run`, `longest_null_run_start`, `longest_null_run_end`.

```python
result = NullCheckValidator(columns=["price"]).validate(df)
print(result.errors[0])
# Column 'price' has 72 null values (30.00% of 240 rows, max 0.00%); longest missing run
# 72 rows from 2024-01-03 00:00:00+00:00 to 2024-01-05 23:00:00+00:00
```

## When to Use

- Before training, on the columns the model uses.
- On raw source data, before any transformer can fill the holes.

NullCheck only sees the rows that are in the frame. Rows missing from the index are found by [CoverageValidator](/epftoolbox2/validators/coverage/) and [ContinuityValidator](/epftoolbox2/validators/continuity/).
