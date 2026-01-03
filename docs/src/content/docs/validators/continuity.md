---
title: ContinuityValidator
description: Check for gaps in time series data
---

# ContinuityValidator

Validates that the time series has no gaps based on expected frequency.

## Basic Usage

```python
from epftoolbox2.data.validators import ContinuityValidator

validator = ContinuityValidator(expected_freq="1h")

result = validator.validate(df)
print(result.is_valid)  # True/False
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expected_freq` | str | Required | Expected frequency (e.g., "1h", "15min") |

## Example Output

```python
result = validator.validate(df)

if not result.is_valid:
    for error in result.errors:
        print(f"Gap detected: {error}")
    # Gap detected: Missing 3 hours between 2024-01-15 12:00 and 2024-01-15 16:00
    
print(result.info)
# {'gaps': [{'start': Timestamp(...), 'end': Timestamp(...), 'missing': 3}]}
```

## When to Use

- After downloading data from external APIs
- After timezone conversions (to detect DST gaps)
- Before training models that expect continuous data
