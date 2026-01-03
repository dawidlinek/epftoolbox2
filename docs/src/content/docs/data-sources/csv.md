---
title: CsvSource
description: Load time-series data from CSV files
---

import { Aside } from '@astrojs/starlight/components';

# CsvSource

Loads time-series data from CSV files with automatic datetime parsing.

## Basic Usage

```python
import pandas as pd
from epftoolbox2.data.sources import CsvSource

source = CsvSource(
    file_path="data/prices.csv",
    datetime_column="datetime",
)

df = source.fetch(
    start=pd.Timestamp("2024-01-01", tz="UTC"),
    end=pd.Timestamp("2024-06-01", tz="UTC"),
)
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | str | Yes | - | Path to CSV file |
| `datetime_column` | str | No | `"datetime"` | Name of datetime column |
| `columns` | List[str] | No | All | Columns to include |
| `prefix` | str | No | `""` | Prefix for column names |
| `datetime_format` | str | No | Auto | strftime format |
| `separator` | str | No | `","` | CSV separator |

---

## CSV File Format

Expected format:

```csv
datetime,price,load,temperature
2024-01-01 00:00:00,50.5,12000,5.2
2024-01-01 01:00:00,48.3,11500,5.0
2024-01-01 02:00:00,45.2,10800,4.8
```

---

## Timezone Handling

- Timezone-naive datetimes are localized to UTC
- Timezone-aware datetimes are converted to UTC
- Output always has a **UTC DatetimeIndex**

<Aside type="tip">
  Use `TimezoneTransformer` to convert to local time after loading.
</Aside>
