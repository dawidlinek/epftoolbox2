from typing import List, Optional, Union

import pandas as pd

from ._utils import longest_run, select_columns
from .base import Validator
from .result import ValidationResult


class NullCheckValidator(Validator):
    """Checks that columns exist and have few enough missing values.

    Args:
        columns: Columns to check. None checks every column; lag columns (`price_d-7`) are empty at the start of the
            frame by construction, so list the columns that matter instead.
        allow_nulls: If True, missing values are reported as warnings and never fail the check.
        max_null_frac: Largest share of missing values allowed per column (0.0 = none). A column with missing values
            within the limit produces a warning.

    `stats` has one row per checked column: rows, null_count, null_frac and the longest run of consecutive missing
    rows (length, first and last timestamp).
    """

    def __init__(
        self,
        columns: Optional[Union[str, List[str]]] = None,
        allow_nulls: bool = False,
        max_null_frac: float = 0.0,
    ):
        if not 0.0 <= max_null_frac <= 1.0:
            raise ValueError(f"max_null_frac must be between 0 and 1, got {max_null_frac}")
        self.columns = [columns] if isinstance(columns, str) else columns
        self.allow_nulls = allow_nulls
        self.max_null_frac = max_null_frac

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = self._result()

        if df.empty:
            result.warnings.append("DataFrame is empty")
            return result

        columns, missing = select_columns(df, self.columns)
        if missing:
            result.add_error(f"Missing required columns: {sorted(missing)}")
            result.info["missing_columns"] = sorted(missing)

        limit = 1.0 if self.allow_nulls else self.max_null_frac
        rows = []
        for column, is_null in df[columns].isna().items():
            mask = is_null.to_numpy()
            count = int(mask.sum())
            run, run_start = longest_run(mask)
            rows.append(
                {
                    "column": column,
                    "rows": len(mask),
                    "null_count": count,
                    "null_frac": count / len(mask),
                    "longest_null_run": run,
                    "longest_null_run_start": df.index[run_start] if run else None,
                    "longest_null_run_end": df.index[run_start + run - 1] if run else None,
                }
            )
        result.stats = pd.DataFrame(rows)

        with_nulls = [row for row in rows if row["null_count"] > 0]
        if with_nulls:
            result.info["null_counts"] = {row["column"]: row["null_count"] for row in with_nulls}
            result.info["null_fraction"] = {row["column"]: row["null_frac"] for row in with_nulls}
            result.info["longest_null_run"] = {row["column"]: row["longest_null_run"] for row in with_nulls}
        for row in with_nulls:
            message = (
                f"Column '{row['column']}' has {row['null_count']} null values "
                f"({row['null_frac']:.2%} of {row['rows']} rows, max {limit:.2%}); longest missing run "
                f"{row['longest_null_run']} rows from {row['longest_null_run_start']} to {row['longest_null_run_end']}"
            )
            if row["null_frac"] > limit:
                result.add_error(message)
            else:
                result.warnings.append(message)

        return result
