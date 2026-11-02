"""Workers write result records in the order tasks finish, so the same study can leave its records in a different
order from one run to the next (process count, scheduling, resume). Every number and file the report produces must
depend on the set of records only: float sums are not associative, so a metric taken over the rows in file order
can differ in the last bit between two runs of the same configuration.
"""

import functools
import json
import operator
import random

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator, rMAEEvaluator
from epftoolbox2.exporters import CsvExporter, ExcelExporter, TerminalExporter
from epftoolbox2.results import ModelResultRef
from epftoolbox2.results.report import GROUP_CONFIGS, EvaluationReport

BY_METHODS = [f"by_{name}" for name, _ in GROUP_CONFIGS]
RUN_DATES = pd.date_range("2023-12-18", "2024-01-14", freq="D")
HORIZONS = (1, 2, 3)


def make_records(seed: int, skip_every: int = 0) -> list[dict]:
    """Records of one model whose errors span eleven orders of magnitude (1e-3 to 1e8), so that the order in which
    they are summed shows in the last bits of any mean."""
    rng = np.random.default_rng(seed)
    records = []
    for day_in_test, run_date in enumerate(RUN_DATES):
        for horizon in HORIZONS:
            for hour in range(24):
                n = (day_in_test * len(HORIZONS) + horizon - 1) * 24 + hour
                if skip_every and n % skip_every == 0:
                    continue
                actual = float(50 + 10 * np.sin(hour / 24 * 2 * np.pi) + rng.normal(0, 3))
                error = float(rng.choice([-1, 1]) * 10 ** rng.uniform(-3, 8))
                records.append(
                    {
                        "run_date": run_date.date().isoformat(),
                        "target_date": (run_date + pd.Timedelta(days=horizon)).date().isoformat(),
                        "run_weekday": int(run_date.dayofweek),
                        "hour": hour,
                        "horizon": horizon,
                        "day_in_test": day_in_test,
                        "prediction": actual + error,
                        "actual": actual,
                    }
                )
    return records


STUDY = {"base": make_records(1), "other": make_records(2, skip_every=7), "third": make_records(3, skip_every=11)}


def shuffled(records: list[dict], seed: int) -> list[dict]:
    out = list(records)
    random.Random(seed).shuffle(out)
    return out


def build_report(order_seed: int | None, kind: str, tmp_path) -> EvaluationReport:
    """The study with each model's records in the order of `order_seed` (None: key order, the way a sequential run
    would write them), as plain lists or as JSONL files."""
    evaluators = [MAEEvaluator(), RMSEEvaluator(), rMAEEvaluator(base_model="base")]
    results = {}
    for name, records in STUDY.items():
        ordered = records if order_seed is None else shuffled(records, order_seed + len(name))
        if kind == "list":
            results[name] = ordered
            continue
        path = tmp_path / f"order{order_seed}" / f"{name}.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in ordered))
        results[name] = ModelResultRef(
            name=name,
            path=path,
            count=len(ordered),
            test_start="2023-12-18",
            test_end="2024-01-14",
            horizon=max(HORIZONS),
        )
    return EvaluationReport(results, evaluators)


def assert_bitwise_equal(left: pd.DataFrame, right: pd.DataFrame, what: str) -> None:
    assert list(left.columns) == list(right.columns), what
    assert len(left) == len(right), what
    for column in left.columns:
        a, b = left[column].reset_index(drop=True), right[column].reset_index(drop=True)
        if a.dtype.kind == "f":
            assert a.dtype == b.dtype, f"{what}: {column} dtype"
            assert np.array_equal(a.to_numpy().view("i8"), b.to_numpy().view("i8")), f"{what}: {column} differs"
        else:
            pd.testing.assert_series_equal(a, b, check_exact=True, obj=f"{what}: {column}")


@pytest.fixture(params=["list", "jsonl"])
def kind(request):
    return request.param


@pytest.fixture
def reports(kind, tmp_path):
    """The same study in key order and in two shuffled orders."""
    return [build_report(seed, kind, tmp_path) for seed in (None, 11, 12)]


@pytest.fixture
def file_reports(tmp_path):
    """The same, always from JSONL files: what the exporters write is what the report holds, whichever way it was
    given the records, and the tests above compare the report's tables for both."""
    return [build_report(seed, "jsonl", tmp_path) for seed in (None, 11, 12)]


def test_the_test_data_makes_summation_order_matter():
    """Guards the tests below: if shuffling the records changed no sum, they would pass without the fix."""
    for records in STUDY.values():
        errors = [abs(r["prediction"] - r["actual"]) for r in records]
        sums = {functools.reduce(operator.add, shuffled(errors, seed)) for seed in range(6)}
        assert len(sums) > 1


def test_summary_does_not_depend_on_record_order(reports):
    for other in reports[1:]:
        assert_bitwise_equal(reports[0].summary(), other.summary(), "summary()")


def test_compute_all_does_not_depend_on_record_order(reports):
    expected = reports[0].compute_all()
    for other in reports[1:]:
        tables = other.compute_all()
        assert tables.keys() == expected.keys()
        for name, table in expected.items():
            assert not table.empty
            assert_bitwise_equal(table, tables[name], f"compute_all()[{name!r}]")


@pytest.mark.parametrize("method", BY_METHODS)
def test_grouped_tables_do_not_depend_on_record_order(reports, method):
    expected = getattr(reports[0], method)()
    assert not expected.empty
    for other in reports[1:]:
        assert_bitwise_equal(expected, getattr(other, method)(), f"{method}()")


def test_predictions_and_details_do_not_depend_on_record_order(reports):
    for other in reports[1:]:
        assert_bitwise_equal(reports[0].predictions(), other.predictions(), "predictions()")
        for (name, expected), (other_name, details) in zip(reports[0].iter_details(), other.iter_details()):
            assert name == other_name
            assert_bitwise_equal(expected, details, f"iter_details()[{name!r}]")


def test_records_without_dates_or_with_repeated_keys_still_have_one_order():
    """Records passed as plain lists may lack the date keys, and a forecast written twice repeats its key: the key
    cannot settle the order there, so the values do."""
    records = STUDY["base"]
    undated = [{k: r[k] for k in ("hour", "horizon", "prediction", "actual")} for r in records]
    repeated = records + [{**r, "day_in_test": r["day_in_test"] + 100} for r in records[::5]]
    repeated += [{**r, "prediction": r["prediction"] + 0.1} for r in records[::7]]

    for results in (undated, repeated):
        a = EvaluationReport({"m": shuffled(results, 1)}, [MAEEvaluator(), RMSEEvaluator()])
        b = EvaluationReport({"m": shuffled(results, 2)}, [MAEEvaluator(), RMSEEvaluator()])
        tables_a, tables_b = a.compute_all(), b.compute_all()
        assert not tables_a["summary"].empty
        for name, table in tables_a.items():
            assert_bitwise_equal(table, tables_b[name], f"compute_all()[{name!r}]")
        assert_bitwise_equal(a.predictions(), b.predictions(), "predictions()")
        for (_, details_a), (_, details_b) in zip(a.iter_details(), b.iter_details()):
            assert_bitwise_equal(details_a, details_b, "iter_details()")


def test_details_are_in_forecast_key_order(reports):
    keys = ["run_date", "target_date", "horizon", "hour"]
    for report in reports:
        for name, details in report.iter_details():
            assert details[keys].equals(details[keys].sort_values(keys).reset_index(drop=True)), name


def test_csv_does_not_depend_on_record_order(file_reports, tmp_path):
    for i, report in enumerate(file_reports):
        CsvExporter(str(tmp_path / f"out{i}.csv")).export(report)

    expected = (tmp_path / "out0.csv").read_bytes()
    assert expected
    for i in range(1, len(file_reports)):
        assert (tmp_path / f"out{i}.csv").read_bytes() == expected


def test_excel_does_not_depend_on_record_order(file_reports, tmp_path):
    for i, report in enumerate(file_reports):
        ExcelExporter(str(tmp_path / f"out{i}.xlsx")).export(report)

    expected = pd.read_excel(tmp_path / "out0.xlsx", sheet_name=None)
    assert "Summary" in expected and "Details" in expected
    for i in range(1, len(file_reports)):
        sheets = pd.read_excel(tmp_path / f"out{i}.xlsx", sheet_name=None)
        assert sheets.keys() == expected.keys()
        for name, sheet in expected.items():
            pd.testing.assert_frame_equal(sheet, sheets[name], check_exact=True, obj=name)


def test_terminal_does_not_depend_on_record_order(file_reports, capsys):
    shown = []
    for report in file_reports:
        TerminalExporter(show=["summary", "horizon", "hour_horizon", "year"]).export(report)
        shown.append(capsys.readouterr().out)

    assert shown[0]
    assert shown[1:] == [shown[0]] * (len(file_reports) - 1)
