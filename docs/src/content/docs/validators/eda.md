---
title: EdaValidator
description: Display exploratory data analysis statistics
---

# EdaValidator

Calculates and displays EDA statistics for numeric columns. Uses Rich for formatted console output.

## Basic Usage

```python
from epftoolbox2.data.validators import EdaValidator

validator = EdaValidator()
result = validator.validate(df)
```

## Console Output

```
┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Column      ┃ Min      ┃ Max      ┃ Mean     ┃ Std      ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ load_actual │ 8500.12  │ 25000.45 │ 15234.67 │ 3456.78  │
│ price       │ -5.23    │ 450.00   │ 85.45    │ 45.23    │
└─────────────┴──────────┴──────────┴──────────┴──────────┘
```

## Statistics Included

| Statistic | Description |
|-----------|-------------|
| Min | Minimum value |
| Max | Maximum value |
| Mean | Average value |
| Std | Standard deviation |
| Q25 | 25th percentile |
| Q50 | Median (50th percentile) |
| Q75 | 75th percentile |
| Nulls | Count of null values |

## Accessing Statistics Programmatically

```python
result = validator.validate(df)

# Get stats as DataFrame
stats_df = result.stats
print(stats_df)
```

## When to Use

- Initial data exploration
- Checking for outliers
- Verifying data ranges are reasonable
