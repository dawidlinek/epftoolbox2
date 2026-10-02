---
title: PlausibilityValidator
description: Check value ranges (e.g. SDAC price limits) and frozen constant runs
---

Checks that values lie in plausible ranges and that they don't stay frozen for too long.

## Basic Usage

```python
from epftoolbox2.data.validators import PlausibilityValidator

# Defaults: prices within the SDAC limits, load at least 1 MW, price/load frozen for over 24 rows
result = PlausibilityValidator().validate(df)

result = PlausibilityValidator(
    ranges={"price*": [-600, 4000], "temperature*": [-40, 50], "generation_*": [0, None]},
    max_constant_run=24,
    constant_columns=["price", "load_actual"],
).validate(df)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ranges` | dict or None | `None` | `{column or glob pattern: [min, max]}`, bounds inclusive, `None` for no bound. `None` uses the defaults below; `{}` turns the range check off. |
| `max_constant_run` | int or None | `24` | Longest run of identical consecutive values allowed, in rows. `None` turns the check off. |
| `constant_columns` | List[str] or None | `None` | Columns or glob patterns checked for constant runs. `None` uses `price*` and `load_actual*`. `ImputeTransformer` flag columns (`*_imputed`) are always skipped. |

A column uses the **first** pattern in `ranges` it matches, so put specific patterns before general ones. Only numeric columns are checked. A pattern you pass that matches no column gives a warning.

### Defaults

| Pattern | Range | Why |
|---------|-------|-----|
| `price*` | -600 to 4000 EUR/MWh | SDAC harmonised minimum and maximum clearing prices. The maximum has been 4000 since 2022; the minimum was -500 until trading day 2026-05-28 and -600 since. |
| `load*` | at least 1 MW | Load can't be zero or negative; some TSOs report zeros during outages. |

The SDAC limits move automatically under ACER's HMMCP methodology when prices come close to them. Pass `ranges` explicitly if they change, for non-SDAC markets, or for prices in another unit or currency.

Constant runs are checked only in `price*` and `load_actual*` by default: calendar dummies, daily values upsampled to hours and night-time solar output are constant by design.

## Output

- **Errors:** values outside their range (count, first timestamp, min and max).
- **Warnings:** runs of identical values longer than `max_constant_run`. They are warnings because a price can genuinely repeat for hours, but a long run usually means a filled outage or a frozen feed.
- **info:** `out_of_range` (per column), `longest_constant_run` (per column), `checked_columns`.
- **stats:** one row per checked column: `rule`, `low`, `high`, `min`, `max`, `out_of_range`, `longest_constant_run`, `constant_value`, `constant_start`.

```python
print(result.warnings[0])
# Column 'price' repeats 92.54 for 30 consecutive rows from 2024-01-03 01:00:00+01:00
# to 2024-01-04 06:00:00+01:00 (more than 24)
```

## When to Use

- On raw source data, to catch parsing errors (wrong unit, a mixed-up series) and outages filled with a constant.
- After resampling, where a flat run is often what interpolation or forward fill left behind.
