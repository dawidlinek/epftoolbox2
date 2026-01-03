---
title: NullCheckValidator
description: Check for null values in data
---

# NullCheckValidator

Validates that specified columns exist and optionally checks for null values.

## Basic Usage

```python
from epftoolbox2.data.validators import NullCheckValidator

validator = NullCheckValidator(
    columns=["load_actual", "price"],
    allow_nulls=False,
)

result = validator.validate(df)
print(result.is_valid)  # True/False
print(result.errors)    # List of error messages
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | List[str] | Required | Columns to check |
| `allow_nulls` | bool | `False` | Whether nulls are allowed |

## ValidationResult

| Field | Type | Description |
|-------|------|-------------|
| `is_valid` | bool | True if validation passed |
| `errors` | List[str] | Error messages |
| `warnings` | List[str] | Warning messages |
| `info` | dict | Additional information |

## Example Output

```python
result = validator.validate(df)

if not result.is_valid:
    for error in result.errors:
        print(f"Error: {error}")
    # Error: Column 'price' has 5 null values
```
