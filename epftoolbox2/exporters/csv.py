from pathlib import Path
from typing import List, Optional

import pandas as pd

from .base import Exporter
from ..models._contract import check_index, period_index
from ..results.report import EvaluationReport

_MERGE_KEYS = ["run_date", "target_date", "hour", "horizon"]


def _clock_time(period: int) -> str:
    """Local wall-clock start of a quarter-hour period (0-95) as HH:MM."""
    return f"{period // 4:02d}:{period % 4 * 15:02d}"


class CsvExporter(Exporter):
    def __init__(self, path: str, extra_columns: Optional[List[str]] = None, summary_path: Optional[str] = None):
        """
        Args:
            path: Path to the output CSV file.
            extra_columns: Column names from the source dataset to include
                           (e.g. ["is_holiday", "load_forecast"]). Joined by
                           target_date + period of the day (`hour`), which is the
                           hour at 1h and the quarter-hour at 15min.
            summary_path: Also write `report.summary()`, one row per model with every
                          evaluator's columns, to this CSV file. `path` holds forecasts, not
                          metrics, so this is where metric columns go.

        With 15min results `hour` holds the period of the day (0-95) and a `target_time`
        column next to it gives that period's local start time as HH:MM.
        """
        self.path = Path(path)
        self.extra_columns = extra_columns or []
        self.summary_path = Path(summary_path) if summary_path is not None else None

    def export(self, report: EvaluationReport) -> None:
        if not report.refs:
            return

        if self.summary_path is not None:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            report.summary().to_csv(self.summary_path, index=False)

        freq = self._results_freq(report)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        base_cols = ["run_date", "target_date", "hour", "horizon", "day_in_test", "actual"]
        base_df: Optional[pd.DataFrame] = None

        for model_name, model_df in report.iter_details():
            model_df = model_df.rename(
                columns={
                    "prediction": f"{model_name}_prediction",
                }
            )
            model_df[f"{model_name}_error"] = model_df[f"{model_name}_prediction"] - model_df["actual"]
            keep = _MERGE_KEYS + [f"{model_name}_prediction", f"{model_name}_error"]
            if base_df is None:
                base_df = model_df[base_cols + [f"{model_name}_prediction", f"{model_name}_error"]].copy()
            else:
                base_df = base_df.merge(
                    model_df[keep],
                    on=_MERGE_KEYS,
                    how="outer",
                )
            del model_df

        if base_df is None:
            return

        if freq == "15min":
            base_df.insert(base_df.columns.get_loc("hour") + 1, "target_time", base_df["hour"].map(_clock_time))

        if self.extra_columns and report.source_data is not None:
            base_df = self._join_extra_columns(base_df, report.source_data, freq)

        sort_keys = ["target_date", "hour", "horizon"]
        base_df = base_df.sort_values(by=sort_keys).reset_index(drop=True)
        base_df.to_csv(self.path, index=False)

    @staticmethod
    def _results_freq(report: EvaluationReport) -> str:
        """The one frequency all models of the report ran at; period ids of different ones don't line up."""
        freqs = sorted(set(report._result_freqs().values()))
        if len(freqs) > 1:
            raise ValueError(
                f"CsvExporter cannot put results of different frequencies ({', '.join(freqs)}) in one file: "
                "their period ids (hour 0-23 vs quarter-hour 0-95) name different times of day. "
                "Export each frequency from its own report."
            )
        return freqs[0] if freqs else "1h"

    def _join_extra_columns(self, base_df: pd.DataFrame, source_data: pd.DataFrame, freq: str) -> pd.DataFrame:
        available = [c for c in self.extra_columns if c in source_data.columns]
        if not available:
            return base_df

        if freq == "15min":
            # Quarter-hour results only line up with quarter-hour source rows; any other resolution would fill
            # most rows with NaN (hourly source) or repeat them (finer source) without saying so.
            try:
                check_index(source_data, freq)
            except ValueError as exc:
                raise ValueError(
                    f"extra_columns need source data at the resolution of the 15min results: {exc}"
                ) from exc

        src = source_data[available].copy()
        idx = src.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)
        src["_target_date"] = idx.date.astype(str)
        src["_hour"] = period_index(idx, freq)
        src = src.reset_index(drop=True)

        base_df = base_df.merge(
            src,
            left_on=["target_date", "hour"],
            right_on=["_target_date", "_hour"],
            how="left",
        ).drop(columns=["_target_date", "_hour"])

        return base_df
