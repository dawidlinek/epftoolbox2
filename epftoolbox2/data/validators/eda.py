from typing import Optional, List
import pandas as pd
from rich.console import Console
from rich.table import Table
from .base import Validator
from .result import ValidationResult


class EdaValidator(Validator):
    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns
        self.console = Console()

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

        self._print_table(stats_data)
        return result

    def _print_table(self, stats_data: List[dict]) -> None:
        table = Table(title="EDA Statistics", show_lines=True)
        table.add_column("Column", style="cyan", no_wrap=True)
        table.add_column("Type", style="dim")
        table.add_column("Count", justify="right")
        table.add_column("Nulls", justify="right")
        table.add_column("Null%", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("Std", justify="right")

        for row in stats_data:
            table.add_row(
                row["column"][:30],
                row["dtype"],
                str(row["count"]),
                str(row["null_count"]),
                f"{row['null_pct']:.1f}%",
                f"{row['min']:.2f}" if pd.notna(row["min"]) else "-",
                f"{row['max']:.2f}" if pd.notna(row["max"]) else "-",
                f"{row['mean']:.2f}" if pd.notna(row["mean"]) else "-",
                f"{row['std']:.2f}" if pd.notna(row["std"]) else "-",
            )

        self.console.print(table)
