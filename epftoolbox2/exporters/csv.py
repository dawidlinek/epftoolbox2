from pathlib import Path
from typing import List, Optional

import pandas as pd

from .base import Exporter
from ..results.report import EvaluationReport


class CsvExporter(Exporter):
    def __init__(self, path: str, extra_columns: Optional[List[str]] = None):
        """
        Args:
            path: Path to the output CSV file.
            extra_columns: Column names from the source dataset to include
                           (e.g. ["is_holiday", "load_forecast"]). Joined by
                           target_date + hour.
        """
        self.path = Path(path)
        self.extra_columns = extra_columns or []

    def export(self, report: EvaluationReport) -> None:
        if not report.refs:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)

        sort_keys = ["target_date", "hour", "horizon"]
        base_cols = ["run_date", "target_date", "hour", "horizon", "day_in_test", "actual"]

        base_df: Optional[pd.DataFrame] = None
        model_names: List[str] = []

        for model_name, model_df in report.iter_details():
            model_df = model_df.sort_values(by=sort_keys).reset_index(drop=True)
            if base_df is None:
                base_df = model_df[base_cols].copy()
            base_df[f"{model_name}_prediction"] = model_df["prediction"].values
            base_df[f"{model_name}_error"] = (
                model_df["prediction"].values - base_df["actual"].values
            )
            model_names.append(model_name)
            del model_df

        if base_df is None:
            return

        if self.extra_columns and report.source_data is not None:
            base_df = self._join_extra_columns(base_df, report.source_data)

        base_df = base_df.sort_values(by=sort_keys).reset_index(drop=True)
        base_df.to_csv(self.path, index=False)

    def _join_extra_columns(
        self, base_df: pd.DataFrame, source_data: pd.DataFrame
    ) -> pd.DataFrame:
        available = [c for c in self.extra_columns if c in source_data.columns]
        if not available:
            return base_df

        src = source_data[available].copy()
        idx = src.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)
        src["_target_date"] = idx.date.astype(str)
        src["_hour"] = idx.hour
        src = src.reset_index(drop=True)

        base_df = base_df.merge(
            src,
            left_on=["target_date", "hour"],
            right_on=["_target_date", "_hour"],
            how="left",
        ).drop(columns=["_target_date", "_hour"])

        return base_df
