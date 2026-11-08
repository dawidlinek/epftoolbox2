from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import Exporter
from ..models._calendar import DAILY
from ..models._contract import check_index, period_index
from ..results.breakdowns import check_source_zone
from ..results.report import EvaluationReport

_MERGE_KEYS = ["run_date", "target_date", "hour", "horizon"]


def _clock_time(period: int) -> str:
    """Local wall-clock start of a quarter-hour period (0-95) as HH:MM."""
    return f"{period // 4:02d}:{period % 4 * 15:02d}"


class CsvExporter(Exporter):
    _KEY_DTYPES = {"hour": "int64", "horizon": "int64", "day_in_test": "int64", "actual": "float64"}

    def __init__(self, path: str, extra_columns: Optional[List[str]] = None, summary_path: Optional[str] = None):
        """
        Args:
            path: Path to the output CSV file.
            extra_columns: Column names from the source dataset to include
                           (e.g. ["is_holiday", "load_forecast"]). Joined by
                           target_date + period of the day (`hour`), which is the
                           hour at 1h, the quarter-hour at 15min and 0 at 1D.
            summary_path: Also write `report.summary()`, one row per model with every
                          evaluator's columns, to this CSV file. `path` holds forecasts, not
                          metrics, so this is where metric columns go.

        With 15min results `hour` holds the period of the day (0-95) and a `target_time`
        column next to it gives that period's local start time as HH:MM.

        A report with a model that has a card (licence, corpus, weights: see the docs page
        Licences and Weights) also writes `report.provenance()` to `provenance.csv` next to
        `path`; without one nothing more is written.
        """
        self.path = Path(path)
        self.extra_columns = extra_columns or []
        self.summary_path = Path(summary_path) if summary_path is not None else None

    @property
    def provenance_path(self) -> Path:
        """Where a report with a carded model's provenance goes: `provenance.csv` next to `path`."""
        return self.path.parent / "provenance.csv"

    def export(self, report: EvaluationReport) -> None:
        if not report.refs:
            return

        if self.summary_path is not None:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            report.summary().to_csv(self.summary_path, index=False)

        provenance = report.provenance()
        if (
            not provenance.empty
        ):  # only with a carded model: the file is there whenever a licence must travel with the numbers
            self.path.parent.mkdir(parents=True, exist_ok=True)
            provenance.to_csv(self.provenance_path, index=False)

        freq = self._results_freq(report)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        base_cols = ["run_date", "target_date", "hour", "horizon", "day_in_test", "actual"]
        base_df: Optional[pd.DataFrame] = None
        model_names: List[str] = []

        for model_name, model_df in report.iter_details():
            model_names.append(model_name)
            if model_df.empty:
                # No forecast in the report's window (a task that stopped before its test days): nothing to merge,
                # and its columns are put back below, empty, where its place in the report puts them.
                continue
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

        model_cols = [col for name in model_names for col in (f"{name}_prediction", f"{name}_error")]
        if base_df is None:  # every model is empty: the header, with the dtypes a join on the keys needs
            base_df = pd.DataFrame({col: pd.Series(dtype=self._KEY_DTYPES.get(col, "object")) for col in base_cols})
        if list(base_df.columns) != base_cols + model_cols:
            base_df = base_df.reindex(columns=base_cols + model_cols)

        if freq == "15min":
            base_df.insert(base_df.columns.get_loc("hour") + 1, "target_time", base_df["hour"].map(_clock_time))

        if self.extra_columns and report.source_data is not None:
            zones = [ref.zone for ref in report.refs.values()]
            base_df = self._join_extra_columns(base_df, report.source_data, freq, zones)

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

    def _join_extra_columns(
        self, base_df: pd.DataFrame, source_data: pd.DataFrame, freq: str, zones: Sequence[Any] = ()
    ) -> pd.DataFrame:
        available = [c for c in self.extra_columns if c in source_data.columns]
        if not available:
            return base_df
        # The join reads the source's wall clock: with results that know their zone, the source must be in it.
        check_source_zone(source_data, zones, "CsvExporter extra_columns")

        # The join is on (target date, period of the day), so the source must have one row per date and period at the
        # resolution of the results. A finer source would write each forecast once per source row (96 hourly forecasts
        # became 388 rows from a quarter-hour source), a coarser one would fill most rows with NaN, and at 1D every hour of
        # a date is period 0. Only the target dates of the file are read, as a run reads only the days it can use.
        try:
            check_index(source_data, freq, self._target_days(base_df))
        except ValueError as exc:
            hint = ""
            if freq != DAILY:  # at 1D the message of check_index already says how to take one row a date
                hint = (
                    f" Each forecast takes the source row of its target date and period, so resample or aggregate the "
                    f"source to {freq} first (ResampleTransformer(freq='{freq}'))."
                )
            raise ValueError(
                f"extra_columns need source data at the resolution of the {freq} results: {exc}{hint}"
            ) from exc

        # check_index lets one wall-clock time repeat: the hour that occurs twice when clocks fall back. Both rows are
        # (date, period) of one forecast; the later one is kept, the row a run forecasts from (`issue_row`).
        src = source_data[available].copy()
        src = src.iloc[np.argsort(src.index.asi8, kind="stable")]  # in time order, so "last" is the second occurrence
        idx = src.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)
        src["_target_date"] = idx.date.astype(str)
        src["_hour"] = period_index(idx, freq)
        src = src.drop_duplicates(["_target_date", "_hour"], keep="last").reset_index(drop=True)

        base_df = base_df.merge(
            src,
            left_on=["target_date", "hour"],
            right_on=["_target_date", "_hour"],
            how="left",
        ).drop(columns=["_target_date", "_hour"])

        return base_df

    @staticmethod
    def _target_days(base_df: pd.DataFrame) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """The first and last target date of the file, or None if it has none: the days of the source that can be joined."""
        dates = pd.to_datetime(base_df["target_date"], errors="coerce").dropna()
        return (dates.min(), dates.max()) if len(dates) else None
