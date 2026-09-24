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

`fetch(start, end)` returns the rows in **[start, end)** with a UTC index.

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | str | Yes | - | Path to the file: `.csv`, or compressed `.csv.gz`, `.csv.bz2`, `.csv.xz`, `.csv.zip` |
| `datetime_column` | str | No | `"datetime"` | Name of datetime column |
| `columns` | List[str] | No | All | Columns to include |
| `prefix` | str | No | `""` | Prefix for column names (`prefix="c"` gives `c_price`) |
| `datetime_format` | str | No | Auto | strftime format |
| `separator` | str | No | `","` | CSV separator |
| `timezone` | str | No | `"UTC"` | Time zone of **naive** timestamps; ignored for timestamps with a UTC offset |

A relative `file_path` is resolved from the working directory.

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

- Timestamps with a UTC offset (`2024-01-01T00:00:00Z`, `2024-10-27 02:00:00+01:00`) are converted to UTC,
  including files with mixed offsets, which is what pandas writes for a local-time index spanning a DST switch.
- Naive timestamps are read as local times in `timezone` (default `"UTC"`) and converted to UTC.
- Output always has a **UTC DatetimeIndex**.

```python
# A spreadsheet export in Polish local time
source = CsvSource(file_path="data/pl_prices.csv", timezone="Europe/Warsaw")
```

<Aside type="caution">
  A naive local-time file read with the default `timezone="UTC"` is shifted by one or two hours. The error is
  silent unless the file contains a fall-back day, whose repeated hour then shows up as a duplicated timestamp.
</Aside>

For naive local times around DST switches:
- The fall-back hour occurs twice. The rows must be in chronological order (oldest first, as written by most
  exporters) so that the first 02:00 is placed before the second. A file with repeated times that is not in
  that order (e.g. newest first) raises an error, and so does a file with only one 02:00 on that day.
- A time inside the spring-forward gap (02:30 on the last Sunday of March in Central Europe) raises an error.

In both cases, exporting with a UTC offset or in UTC avoids the ambiguity.

## Validation

- **Duplicated timestamps** (after conversion to UTC) raise a `ValueError`.
- Rows without a timestamp raise a `ValueError`.
- A file mixing naive and offset timestamps raises a `ValueError`.

<Aside type="tip">
  Use `TimezoneTransformer` to convert to local time after loading.
</Aside>
