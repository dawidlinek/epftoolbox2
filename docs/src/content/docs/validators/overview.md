---
title: Validators overview
description: Which validator catches what, and how to read validation results
sidebar:
  order: 0
---

Validators check a DataFrame without changing it. `validate(df)` returns a `ValidationResult`: **errors** for data that can't be used, **warnings** for data that is suspicious. Invalid settings raise `ValueError` when the validator is created.

## Which Validator Catches What

| Problem | Validator |
|---------|-----------|
| A local day with the wrong number of rows (DST days, naive wall-clock index) | [DSTStructureValidator](/epftoolbox2/validators/dst-structure/) |
| Duplicate or unsorted timestamps, missing timestamps, mixed resolutions, one column missing for a stretch on a complete index, a late start or early end, too many nulls in the columns the model uses | [CoverageValidator](/epftoolbox2/validators/coverage/) |
| Prices outside the SDAC limits, zero load, frozen values | [PlausibilityValidator](/epftoolbox2/validators/plausibility/) |
| Summary statistics for a first look | [EdaValidator](/epftoolbox2/validators/eda/) |

Data gaps are easiest to see **before** transformers run: resampling with interpolation or forward fill hides them. Validators that work on raw source data (UTC index) take a `tz` parameter where local days matter.

## ValidationResult

```python
from epftoolbox2.data.validators import ValidationResult

@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: List[str] = []
    warnings: List[str] = []
    info: Dict[str, Any] = {}          # counts, timestamps and details per check
    stats: pd.DataFrame = pd.DataFrame()  # a table per column or per day, depending on the validator
    validator: str = ""                 # class name of the validator that produced it
```

- `bool(result)` is `result.is_valid`.
- `print(result)` shows the status, errors, warnings, `info` (long lists shortened) and the size of `stats`.
- `result.add_error(message)` appends an error and marks the result invalid (useful in custom validators).
- `result.to_frame()` returns the messages as a table.

## Printing Several Results as a Table

```python
from epftoolbox2.data.validators import (
    CoverageValidator,
    DSTStructureValidator,
    results_to_frame,
)

validators = [DSTStructureValidator(), CoverageValidator(columns=["price"], min_coverage=1.0)]
results = [v.validate(df) for v in validators]
print(results_to_frame(results).to_string(index=False))
```

```
            validator  valid   level  message
DSTStructureValidator   True      ok
    CoverageValidator  False   error  Column 'price' has values at 168 of 240 expected 1h timestamps ...
```

`results_to_frame` takes a list of results, a list of `(name, result)` pairs or a dict `{name: result}`, and returns one row per error and warning (columns `validator`, `valid`, `level`, `message`). A result without messages gets one `ok` row.
