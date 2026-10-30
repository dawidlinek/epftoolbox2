"""Exporters format what the report computed: the columns of set-level evaluators reach every file, and nothing is
computed on the way. What a report with only MAE, RMSE and rMAE writes is pinned in `tests/results/test_report_pin.py`."""

import warnings

import pandas as pd
import pytest
from openpyxl import load_workbook

from epftoolbox2.evaluators import (
    BiasEvaluator,
    DMEvaluator,
    EconomicEvaluator,
    MAEEvaluator,
    MCSEvaluator,
    rMAEEvaluator,
)
from epftoolbox2.exporters import CsvExporter, ExcelExporter, TerminalExporter
from epftoolbox2.results.report import EvaluationReport
from tests.forecast_records import forecast_records

pytestmark = pytest.mark.filterwarnings("ignore:The comparison rests on")

RUN_DATES = pd.date_range("2023-12-01", periods=70, freq="D")


def study(horizons=(1,), run_dates=RUN_DATES):
    return {
        "base": forecast_records(1, run_dates, noise=6.0, horizons=horizons),
        "better": forecast_records(2, run_dates, noise=4.0, horizons=horizons),
        "same": forecast_records(3, run_dates, noise=6.0, horizons=horizons),
    }


@pytest.fixture(scope="module")
def full_report():
    evaluators = [
        MAEEvaluator(),
        BiasEvaluator(),
        rMAEEvaluator("base"),
        DMEvaluator("base", reps=100),
        MCSEvaluator(reps=100),
        EconomicEvaluator(),
    ]
    return EvaluationReport(study(horizons=(1, 2)), evaluators)


def test_excel_writes_every_dict_column(tmp_path, full_report):
    path = tmp_path / "report.xlsx"
    ExcelExporter(str(path)).export(full_report)
    book = load_workbook(path)

    summary = full_report.summary()
    rows = list(book["Summary"].iter_rows(values_only=True))
    assert list(rows[0]) == list(summary.columns)
    written = pd.DataFrame(rows[1:], columns=rows[0]).set_index("model")
    for column in ("DM ratio", "DM ci_low", "MCS p", "profit", "profit % of perfect"):
        for model in written.index:
            expected = summary.set_index("model").loc[model, column]
            if pd.isna(expected):
                assert pd.isna(written.loc[model, column])
            else:
                assert written.loc[model, column] == pytest.approx(expected, rel=1e-12), column

    for sheet in (
        "horizon_DM ratio",
        "horizon_in MCS",
        "horizon_profit % of perfect",
        "hour_DM p",
        "HourHorizon_DM p",
        "YearHorizon_profit",
        "RunWeekdayHorizon_DM ratio",
    ):
        assert sheet in book.sheetnames


def test_excel_matrix_cells_are_empty_for_nan(tmp_path, full_report):
    path = tmp_path / "report.xlsx"
    ExcelExporter(str(path), sheets=["hour_horizon"]).export(full_report)
    book = load_workbook(path)
    # `DM p_holm` is empty in per-hour tables; a `nan` number would make Excel call the file damaged.
    cells = [c for row in book["HourHorizon_DM p_holm"].iter_rows(min_row=3, values_only=True) for c in row[1:3]]
    assert set(cells) == {None}


def test_terminal_prints_the_dict_columns(capsys, monkeypatch, full_report):
    monkeypatch.setenv("COLUMNS", "400")
    TerminalExporter(show=["summary", "horizon"]).export(full_report)
    out = capsys.readouterr().out
    for column in ("DM ratio", "DM p_holm", "MCS p", "in MCS", "profit % of perfect"):
        assert column in out
    assert "better" in out


def test_csv_summary_file_holds_the_dict_columns(tmp_path, full_report):
    forecasts, summary_path = tmp_path / "forecasts.csv", tmp_path / "metrics" / "summary.csv"
    CsvExporter(str(forecasts), summary_path=str(summary_path)).export(full_report)

    written = pd.read_csv(summary_path)
    pd.testing.assert_frame_equal(
        written, pd.read_csv(pd.io.common.StringIO(full_report.summary().to_csv(index=False)))
    )
    assert "DM p_holm" in written.columns and "in MCS" in written.columns and "profit days" in written.columns
    # The forecast file is what it was: forecasts, no metric columns.
    assert not {c for c in pd.read_csv(forecasts).columns if c.startswith(("DM", "MCS", "profit"))}


def test_a_report_without_set_evaluators_writes_no_summary_file_by_default(tmp_path):
    report = EvaluationReport(study(), [MAEEvaluator()])
    CsvExporter(str(tmp_path / "forecasts.csv")).export(report)
    assert [p.name for p in tmp_path.iterdir()] == ["forecasts.csv"]


def test_no_warning_on_a_healthy_table():
    run_dates = pd.date_range("2023-01-01", periods=365, freq="D")
    report = EvaluationReport(study(run_dates=run_dates), [DMEvaluator("base", reps=20), MCSEvaluator(reps=20)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        report.summary()
