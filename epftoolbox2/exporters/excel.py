import re
from typing import List
from pathlib import Path
import pandas as pd
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter

from .base import Exporter
from ..results.report import EvaluationReport, metric_columns


# The column of the report's tables with each model's number of forecasts in a group. A breakdown gets a sheet of it after the
# sheets of its metrics, with 0 where a model has no forecast in the group (it has no row in the table there) and every model on
# the groups of the whole table, and no colors: a large count is not a bad value.
COUNTS = "forecasts"

# The characters an Excel sheet title can't hold; an evaluator's column may (`ScaledMAEEvaluator`'s "MAE/cap").
_SHEET_FORBIDDEN = re.compile(r"[\\/?*\[\]:]")


def _sheet_name(title: str) -> str:
    """A sheet title Excel takes: the forbidden characters as "_", at most 31 characters. Titles without them are as before."""
    return _SHEET_FORBIDDEN.sub("_", title)[:31]


def _cell(value):
    """A table value as openpyxl should write it: NaN as an empty cell, since its `nan` is not a valid Excel number."""
    return None if pd.isna(value) else value


def _sheet_columns(data: pd.DataFrame) -> List[str]:
    """What a breakdown has a sheet for: each metric, then the number of forecasts."""
    return [*metric_columns(data), *([COUNTS] if COUNTS in data.columns else [])]


def _pivot(data: pd.DataFrame, index: str, columns: str, column: str, whole: pd.DataFrame = None) -> pd.DataFrame:
    """`column` of a breakdown (or of one model's rows of it) with `index` down and `columns` across.

    A group where a model has no row is NaN for a metric (no value) and 0 for the count (no forecast). The count also has
    every row and column that `whole`, the table of all models, has, so that a model without a forecast at a horizon or on a
    weekday still shows its zero there, and the blocks of the models line up."""
    pivot = data.pivot(index=index, columns=columns, values=column)
    if column != COUNTS:
        return pivot
    if whole is not None:
        pivot = pivot.reindex(index=sorted(whole[index].unique()), columns=sorted(whole[columns].unique()))
    return pivot.fillna(0).astype(int)


class ExcelExporter(Exporter):
    def __init__(self, path: str, sheets: List[str] = None):
        """
        Args:
            path: Path to the output Excel file.
            sheets: Which sheets to include. Options: "summary", "hour", "horizon",
                    "hour_horizon", "year", "year_horizon", "details". Defaults to all.
        """
        self.path = Path(path)
        self.sheets = sheets or [
            "summary",
            "hour",
            "horizon",
            "hour_horizon",
            "year",
            "year_horizon",
            "run_weekday_horizon",
            "target_weekday_horizon",
            "details",
        ]

    def export(self, report: EvaluationReport) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        agg_sheets = {
            "summary",
            "hour",
            "horizon",
            "hour_horizon",
            "year",
            "year_horizon",
            "run_weekday_horizon",
            "target_weekday_horizon",
        }
        needs_agg = bool(set(self.sheets) & agg_sheets)
        agg = report.compute_all() if needs_agg else {}

        with pd.ExcelWriter(self.path, engine="openpyxl") as writer:
            if "summary" in self.sheets:
                self._write_summary(writer, report, agg["summary"])
            if "hour" in self.sheets:
                self._write_pivot_sheet(writer, report, "hour", agg["hour"], "hour")
            if "horizon" in self.sheets:
                self._write_pivot_sheet(writer, report, "horizon", agg["horizon"], "horizon")
            if "hour_horizon" in self.sheets:
                self._write_matrix_sheet(writer, report, "HourHorizon", agg["hour_horizon"], "hour", "horizon")
            if "year" in self.sheets:
                self._write_pivot_sheet(writer, report, "year", agg["year"], "year")
            if "year_horizon" in self.sheets:
                self._write_matrix_sheet(writer, report, "YearHorizon", agg["year_horizon"], "year", "horizon")
            if (
                "run_weekday_horizon" in self.sheets
                and "run_weekday_horizon" in agg
                and not agg["run_weekday_horizon"].empty
            ):
                self._write_weekday_horizon_sheet(
                    writer, report, agg["run_weekday_horizon"], "run_weekday", "RunWeekdayHorizon"
                )
            if (
                "target_weekday_horizon" in self.sheets
                and "target_weekday_horizon" in agg
                and not agg["target_weekday_horizon"].empty
            ):
                self._write_weekday_horizon_sheet(
                    writer, report, agg["target_weekday_horizon"], "target_weekday", "TgtWeekdayHorizon"
                )
            if "details" in self.sheets:
                self._write_details(writer, report)

    def _write_summary(self, writer: pd.ExcelWriter, report: EvaluationReport, summary: pd.DataFrame) -> None:
        sheet_name = "Summary"
        summary.to_excel(writer, sheet_name=sheet_name, index=False)
        metrics = metric_columns(summary)  # the number of forecasts is a count, not a good or a bad value
        if metrics:
            first = list(summary.columns).index(metrics[0]) + 1
            self._apply_color_scale(writer, sheet_name, 2, first, len(summary), len(metrics))

    def _write_pivot_sheet(
        self, writer: pd.ExcelWriter, report: EvaluationReport, sheet_prefix: str, data: pd.DataFrame, col_col: str
    ) -> None:
        for metric in _sheet_columns(data):
            pivot = _pivot(data, "model", col_col, metric)
            pivot.columns = [f"{col_col}_{c}" for c in pivot.columns]
            pivot.index.name = metric

            sheet_name = _sheet_name(f"{sheet_prefix}_{metric}")
            pivot.to_excel(writer, sheet_name=sheet_name)

            if metric != COUNTS:
                per_column = col_col == "horizon"
                self._apply_color_scale(writer, sheet_name, 2, 2, len(pivot), len(pivot.columns), per_column=per_column)

    def _write_matrix_sheet(
        self,
        writer: pd.ExcelWriter,
        report: EvaluationReport,
        sheet_prefix: str,
        data: pd.DataFrame,
        row_col: str,
        col_col: str,
    ) -> None:
        if data.empty:  # no model has a forecast in the window, so the table has no columns to read
            return
        models = list(data["model"].unique())

        for metric in _sheet_columns(data):
            sheet_name = _sheet_name(f"{sheet_prefix}_{metric}")
            ws = writer.book.create_sheet(sheet_name)

            col_offset = 1
            for model in models:
                model_df = data[data["model"] == model]
                pivot = _pivot(model_df, row_col, col_col, metric, whole=data)
                ws.cell(row=1, column=col_offset, value=model)
                for c_idx, col_val in enumerate(pivot.columns, start=1):
                    ws.cell(row=2, column=col_offset + c_idx, value=f"{col_col}_{col_val}")
                for r_idx, row_val in enumerate(pivot.index, start=1):
                    ws.cell(row=2 + r_idx, column=col_offset, value=f"{row_col}_{row_val}")
                    for c_idx, col_val in enumerate(pivot.columns, start=1):
                        ws.cell(row=2 + r_idx, column=col_offset + c_idx, value=_cell(pivot.loc[row_val, col_val]))

                if metric != COUNTS:
                    self._apply_color_scale(writer, sheet_name, 3, col_offset + 1, len(pivot), len(pivot.columns))
                col_offset += len(pivot.columns) + 2

    def _apply_color_scale(
        self,
        writer: pd.ExcelWriter,
        sheet_name: str,
        start_row: int,
        start_col: int,
        num_rows: int,
        num_cols: int,
        per_column: bool = False,
    ) -> None:
        ws = writer.book[sheet_name]
        end_row = start_row + num_rows - 1
        end_col = start_col + num_cols - 1

        if num_rows <= 0 or num_cols <= 0:
            return

        if per_column:
            for col in range(start_col, end_col + 1):
                start_cell = f"{get_column_letter(col)}{start_row}"
                end_cell = f"{get_column_letter(col)}{end_row}"
                ws.conditional_formatting.add(f"{start_cell}:{end_cell}", self._COLOR_SCALE_RULE)
        else:
            start_cell = f"{get_column_letter(start_col)}{start_row}"
            end_cell = f"{get_column_letter(end_col)}{end_row}"
            ws.conditional_formatting.add(f"{start_cell}:{end_cell}", self._COLOR_SCALE_RULE)

    _WEEKDAY_LABELS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

    _COLOR_SCALE_RULE = ColorScaleRule(
        start_type="min",
        start_color="63BE7B",
        mid_type="percentile",
        mid_value=50,
        mid_color="FFEB84",
        end_type="max",
        end_color="F8696B",
    )

    def _write_weekday_horizon_sheet(
        self,
        writer: pd.ExcelWriter,
        report: EvaluationReport,
        data: pd.DataFrame,
        weekday_col: str,
        sheet_prefix: str,
    ) -> None:
        models = list(data["model"].unique())

        for metric in _sheet_columns(data):
            sheet_name = _sheet_name(f"{sheet_prefix}_{metric}")
            ws = writer.book.create_sheet(sheet_name)

            col_offset = 1
            for model in models:
                model_df = data[data["model"] == model]
                pivot = _pivot(model_df, weekday_col, "horizon", metric, whole=data)
                pivot = pivot.sort_index()
                ws.cell(row=1, column=col_offset, value=model)
                for c_idx, col_val in enumerate(pivot.columns, start=1):
                    ws.cell(row=2, column=col_offset + c_idx, value=f"h+{col_val}")
                for r_idx, row_val in enumerate(pivot.index, start=1):
                    label = self._WEEKDAY_LABELS.get(row_val, str(row_val))
                    ws.cell(row=2 + r_idx, column=col_offset, value=label)
                    for c_idx, col_val in enumerate(pivot.columns, start=1):
                        ws.cell(row=2 + r_idx, column=col_offset + c_idx, value=_cell(pivot.loc[row_val, col_val]))

                if metric != COUNTS:
                    self._apply_color_scale(writer, sheet_name, 3, col_offset + 1, len(pivot), len(pivot.columns))
                col_offset += len(pivot.columns) + 2

    def _write_details(self, writer: pd.ExcelWriter, report: EvaluationReport) -> None:
        if not report.refs:
            return

        sort_keys = ["run_date", "target_date", "hour", "horizon"]
        details_df = None
        model_names = []

        for model_name, model_df in report.iter_details():
            model_names.append(model_name)
            if model_df.empty:  # no forecast in the report's window: its column is added below, empty
                continue
            preds = model_df[sort_keys + ["prediction", "actual", "day_in_test"]].copy()
            preds = preds.rename(columns={"prediction": f"prediction_{model_name}"})

            if details_df is None:
                details_df = preds
            else:
                details_df = pd.merge(details_df, preds, on=sort_keys, how="outer", suffixes=("", "_drop"))
                if "actual_drop" in details_df.columns:
                    details_df["actual"] = details_df["actual"].fillna(details_df["actual_drop"])
                    details_df["day_in_test"] = details_df["day_in_test"].fillna(details_df["day_in_test_drop"])
                    details_df = details_df.drop(columns=["actual_drop", "day_in_test_drop"])

        final_cols = ["run_date", "target_date", "hour", "horizon", "day_in_test", "actual"] + [
            f"prediction_{name}" for name in model_names
        ]
        if details_df is None:  # every model is empty: the header only
            details_df = pd.DataFrame(columns=final_cols)

        details_df = details_df.reindex(columns=final_cols)
        details_df = details_df.sort_values(by=["target_date", "hour", "horizon"]).reset_index(drop=True)
        details_df.to_excel(writer, sheet_name="Details", index=False)
