"""A report with only MAE, RMSE and rMAE gives the same tables and exporter files as before set-level evaluators.

The expected outputs in `tests/fixtures/report_pin/` were written by the code before `Evaluator.compute_group` existed.
Refresh them with `UPDATE_REPORT_PIN=1 uv run pytest tests/results/test_report_pin.py` only for an intended change, and
say in the commit message why they moved.
"""

import json
import math
import os
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook

from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator, rMAEEvaluator
from epftoolbox2.exporters import CsvExporter, ExcelExporter, TerminalExporter
from epftoolbox2.results.report import GROUP_CONFIGS, EvaluationReport

FIXTURE = Path(__file__).parent.parent / "fixtures" / "report_pin"
UPDATE = bool(os.environ.get("UPDATE_REPORT_PIN"))


def records(wobble: float, skip_every: int = 0) -> list[dict]:
    """Three weeks over a new year, two horizons, no random numbers (the file must not depend on a generator)."""
    rows = []
    for day_in_test, run_date in enumerate(pd.date_range("2023-12-20", "2024-01-09", freq="D")):
        for horizon in (1, 2):
            for hour in range(24):
                n = (day_in_test * 2 + horizon) * 24 + hour
                if skip_every and n % skip_every == 0:
                    continue
                actual = 50 + 10 * math.sin(hour / 24 * 2 * math.pi) + 5 * math.cos(day_in_test / 3)
                rows.append(
                    {
                        "run_date": run_date.date().isoformat(),
                        "target_date": (run_date + pd.Timedelta(days=horizon)).date().isoformat(),
                        "run_weekday": run_date.dayofweek,
                        "hour": hour,
                        "horizon": horizon,
                        "day_in_test": day_in_test,
                        "prediction": actual + wobble * math.sin(n * 0.7) + 0.1 * wobble,
                        "actual": actual,
                    }
                )
    return rows


def build_report() -> EvaluationReport:
    evaluators = [MAEEvaluator(), RMSEEvaluator(), rMAEEvaluator(base_model="base")]
    study = {"base": records(4.0), "better": records(2.5, skip_every=11), "worse": records(6.0)}
    return EvaluationReport(study, evaluators)


def tables(report: EvaluationReport) -> dict:
    out = {name: frame.to_dict(orient="split") for name, frame in report.compute_all().items()}
    for name, frame in {"by_" + name: getattr(report, "by_" + name)() for name, _ in GROUP_CONFIGS}.items():
        out[name] = frame.to_dict(orient="split")
    return out


def workbook_cells(path: Path) -> dict:
    book = load_workbook(path)
    return {sheet.title: [list(row) for row in sheet.iter_rows(values_only=True)] for sheet in book.worksheets}


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (pd.Timestamp,)) or hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def assert_same(actual, expected, where: str = "") -> None:
    """Equal, floats within 1e-10 relative: exact text would tie the pin to one platform's last bit."""
    if isinstance(expected, dict):
        assert list(actual) == list(expected), where
        for key in expected:
            assert_same(actual[key], expected[key], f"{where}/{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), where
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_same(a, e, f"{where}/{i}")
    elif isinstance(expected, float) and not isinstance(actual, str) and actual is not None:
        assert actual == pytest.approx(expected, rel=1e-10, nan_ok=True), where
    else:
        assert actual == expected, where


def compare_or_update(name: str, actual) -> None:
    path = FIXTURE / name
    if UPDATE:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(actual, indent=0, allow_nan=True) + "\n")
    assert_same(actual, json.loads(path.read_text()), name)


def test_tables_are_pinned():
    compare_or_update("tables.json", jsonable(tables(build_report())))


def test_excel_file_is_pinned(tmp_path):
    path = tmp_path / "report.xlsx"
    ExcelExporter(str(path)).export(build_report())
    compare_or_update("excel.json", jsonable(workbook_cells(path)))


def test_csv_file_is_pinned(tmp_path):
    path = tmp_path / "report.csv"
    CsvExporter(str(path)).export(build_report())
    frame = pd.read_csv(path)
    compare_or_update("forecasts.json", jsonable(frame.to_dict(orient="split")))


def test_terminal_output_is_pinned(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    TerminalExporter(show=["summary", "horizon", "year", "year_horizon"]).export(build_report())
    compare_or_update("terminal.json", capsys.readouterr().out.splitlines())
