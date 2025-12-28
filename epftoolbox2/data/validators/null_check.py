from typing import Optional, List
import pandas as pd
from .base import Validator
from .result import ValidationResult


class NullCheckValidator(Validator):
    def __init__(self, columns: Optional[List[str]] = None, allow_nulls: bool = False):
        self.columns = columns
        self.allow_nulls = allow_nulls

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = ValidationResult()

        if df.empty:
            result.warnings.append("DataFrame is empty")
            return result

        if self.columns:
            missing = set(self.columns) - set(df.columns)
            if missing:
                result.is_valid = False
                result.errors.append(f"Missing required columns: {sorted(missing)}")
                result.info["missing_columns"] = sorted(missing)

        if not self.allow_nulls:
            null_counts = df.isnull().sum()
            cols_with_nulls = null_counts[null_counts > 0]
            if len(cols_with_nulls) > 0:
                result.is_valid = False
                null_info = cols_with_nulls.to_dict()
                result.info["null_counts"] = null_info
                for col, count in null_info.items():
                    result.errors.append(f"Column '{col}' has {count} null values")

        return result
