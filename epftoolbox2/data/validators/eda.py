from typing import Optional, List
import pandas as pd
from .base import Validator
from .result import ValidationResult


class EdaValidator(Validator):
    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = ValidationResult()

        if df.empty:
            result.warnings.append("DataFrame is empty")
            return result

        cols = self.columns if self.columns else df.select_dtypes(include=["number"]).columns.tolist()

        if not cols:
            result.warnings.append("No numeric columns to analyze")
            return result

        stats_data = []
        for col in cols:
            if col not in df.columns:
                result.warnings.append(f"Column '{col}' not found")
                continue

            series = df[col]
            null_count = series.isnull().sum()
            stats_data.append(
                {
                    "column": col,
                    "dtype": str(series.dtype),
                    "count": series.count(),
                    "null_count": null_count,
                    "null_pct": (null_count / len(series) * 100) if len(series) > 0 else 0,
                    "min": series.min(),
                    "max": series.max(),
                    "mean": series.mean(),
                    "std": series.std(),
                    "25%": series.quantile(0.25),
                    "50%": series.quantile(0.50),
                    "75%": series.quantile(0.75),
                }
            )

        result.stats = pd.DataFrame(stats_data)
        result.info["columns_analyzed"] = len(stats_data)
        return result
