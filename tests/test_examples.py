"""End-to-end runs of every script in examples/ against the live APIs, on a short window.

Each example has a `main(...)` that takes its date ranges and an output directory, so a test can run the real
script on a few weeks of data. These tests need ENTSOE_API_KEY and the network, which is why they are marked
`integration` (CI's fast suite skips them) and `examples` (run only these with `uv run pytest -m examples`).

Every test runs in its own `live_dir` (see conftest.py): it is the working directory, so the source cache
(`.cache/sources`) and the output files land there, and the example is loaded from its file like a user would run
it. A test asserts real outcomes: a non-empty data frame with the expected columns, forecasts for every model on
the same keys, no unexpected missing forecasts, a passing information-set check and the output files.
"""

import datetime as dt
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from tests.live import (
    DATA_END,
    DATA_START,
    HORIZON,
    LOAD_15MIN_COLUMNS,
    PRICE_COLUMNS,
    RUN_DAYS,
    TEST_END,
    TEST_START,
    assert_csv,
    assert_data_frame,
    assert_excel,
    assert_forecasts,
    assert_information_set_checked,
    assert_metrics,
    switch_days,
)

pytestmark = [pytest.mark.integration, pytest.mark.examples]

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load_example(filename: str, monkeypatch):
    """Import examples/<filename> from its path, then set one process and one thread again.

    The examples set MAX_PROCESSES and THREADS_PER_PROCESS while they are imported.
    """
    spec = importlib.util.spec_from_file_location(f"example_{Path(filename).stem}", EXAMPLES / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    return module


def test_01_data_pipeline_only(live_dir, monkeypatch):
    example = load_example("01_data_pipeline_only.py", monkeypatch)

    df = example.main(data_start=DATA_START, data_end=DATA_END, output_dir=live_dir)

    assert_data_frame(
        df,
        PRICE_COLUMNS,
        "Europe/Warsaw",
        dt.date.fromisoformat(DATA_START),
        dt.date.fromisoformat(DATA_END),
    )
    written = live_dir / "data_output.csv"
    assert written.is_file()
    reread = pd.read_csv(written, index_col=0)
    assert reread.shape == df.shape
    assert list(reread.columns) == list(df.columns)


def test_02_model_pipeline_only(live_dir, monkeypatch, information_set_checks):
    # Example 2 reads the CSV that example 1 writes
    data_example = load_example("01_data_pipeline_only.py", monkeypatch)
    data_example.main(data_start=DATA_START, data_end=DATA_END, output_dir=live_dir)
    example = load_example("02_model_pipeline_only.py", monkeypatch)
    models = ["OLS Baseline", "Lasso CV"]
    planned = RUN_DAYS * HORIZON * 24

    report = example.main(
        data_path=live_dir / "data_output.csv", test_start=TEST_START, test_end=TEST_END, output_dir=live_dir
    )

    assert_data_frame(
        report.source_data,
        PRICE_COLUMNS,
        "Europe/Warsaw",
        dt.date.fromisoformat(DATA_START),
        dt.date.fromisoformat(DATA_END),
    )
    assert_forecasts(report, models, planned)
    assert_metrics(report, models, planned, columns=("MAE", "RMSE", "rMAE"))
    assert_information_set_checked(information_set_checks, models)
    assert_excel(live_dir / "model_results.xlsx", models, planned)
    assert_csv(live_dir / "model_results.csv", models, planned)
    assert sorted(p.name for p in (live_dir / "results").glob("*.jsonl")) == ["lasso_cv.jsonl", "ols_baseline.jsonl"]


def test_03_full_workflow(live_dir, monkeypatch, information_set_checks):
    example = load_example("03_full_workflow.py", monkeypatch)
    models = ["OLS", "LassoCV"]
    planned = RUN_DAYS * HORIZON * 24

    report = example.main(
        data_start=DATA_START, data_end=DATA_END, test_start=TEST_START, test_end=TEST_END, output_dir=live_dir
    )

    assert_data_frame(
        report.source_data,
        PRICE_COLUMNS,
        "Europe/Warsaw",
        dt.date.fromisoformat(DATA_START),
        dt.date.fromisoformat(DATA_END),
    )
    assert_forecasts(report, models, planned)
    assert_metrics(report, models, planned)
    assert_information_set_checked(information_set_checks, models)
    assert_excel(live_dir / "full_workflow_results.xlsx", models, planned)
    assert_csv(live_dir / "full_workflow_results.csv", models, planned)
    assert sorted(p.name for p in (live_dir / "results").glob("*.jsonl")) == ["lassocv.jsonl", "ols.jsonl"]
    assert (live_dir / ".cache" / "sources").is_dir()


def test_04_yaml_workflow(live_dir, monkeypatch, information_set_checks, entsoe_api_key):
    example = load_example("04_yaml_workflow.py", monkeypatch)
    models = ["OLS", "LassoCV"]
    planned = RUN_DAYS * HORIZON * 24

    report = example.main(
        data_start=DATA_START,
        data_end=DATA_END,
        test_start=TEST_START,
        test_end=TEST_END,
        output_dir=live_dir,
        run=True,
    )

    yamls = {
        name: (live_dir / name).read_text() for name in ("data_pipeline.yaml", "model_pipeline.yaml", "experiment.yaml")
    }
    assert "${ENTSOE_API_KEY}" in yamls["data_pipeline.yaml"]
    assert all(entsoe_api_key not in text for text in yamls.values()), "the API key was written to a YAML file"
    assert_data_frame(
        report.source_data,
        PRICE_COLUMNS,
        "Europe/Warsaw",
        dt.date.fromisoformat(DATA_START),
        dt.date.fromisoformat(DATA_END),
    )
    assert_forecasts(report, models, planned)
    assert_metrics(report, models, planned)
    assert_information_set_checked(information_set_checks, models)
    assert_excel(live_dir / "workflow_results.xlsx", models, planned)


def test_05_forecast_only(live_dir, monkeypatch, information_set_checks):
    example = load_example("05_forecast_only.py", monkeypatch)
    models = ["OLS", "LassoCV"]
    horizon = 5
    planned = horizon * 24

    # The example's dates are relative to today, so the window is too: 60 days of history through today + 5.
    before = dt.date.today()
    report = example.main(data_start="now_d-60")
    after = dt.date.today()

    predictions = report.predictions()
    run_dates = {dt.date.fromisoformat(d) for d in predictions["run_date"]}
    assert len(run_dates) == 1 and before <= run_dates.pop() <= after
    today = before
    # Weather and calendar rows run through the whole 5-day prediction window
    assert_data_frame(
        report.source_data,
        PRICE_COLUMNS,
        "Europe/Warsaw",
        today - dt.timedelta(days=60),
        today + dt.timedelta(days=5),
    )
    # A DST switch among the target days costs at most one forecast per model and switch day
    switching = switch_days(today + dt.timedelta(days=1), today + dt.timedelta(days=horizon), "Europe/Warsaw")
    assert_forecasts(report, models, planned, max_missing=len(switching))
    assert len(predictions) == report.forecast_counts()["produced"].sum()
    assert set(predictions["horizon"]) == set(range(1, horizon + 1))
    assert set(predictions["hour"]) == set(range(24))
    assert_information_set_checked(information_set_checks, models)
    # forecast_only skips the evaluators and the exporters
    assert report.evaluators == []
    assert not list(live_dir.glob("*.xlsx")) and not list(live_dir.glob("*.csv"))


def test_06_15min_forecasting(live_dir, monkeypatch, information_set_checks):
    example = load_example("06_15min_forecasting.py", monkeypatch)
    models = ["OLS", "LassoCV"]
    planned = RUN_DAYS * HORIZON * 96

    report = example.main(
        data_start=DATA_START, data_end=DATA_END, test_start=TEST_START, test_end=TEST_END, output_dir=live_dir
    )

    assert_data_frame(
        report.source_data,
        LOAD_15MIN_COLUMNS,
        "Europe/Berlin",
        dt.date.fromisoformat(DATA_START),
        dt.date.fromisoformat(DATA_END),
    )
    assert report.source_data.index.to_series().diff().dropna().eq(pd.Timedelta("15min")).all()
    assert_forecasts(report, models, planned)
    assert_metrics(report, models, planned, columns=("MAE", "RMSE", "rMAE"))
    assert_information_set_checked(information_set_checks, models, zone="Europe/Berlin")
    assert pd.read_csv(live_dir / "15min_forecasting_data.csv", index_col=0).shape == report.source_data.shape
    assert_excel(live_dir / "15min_workflow_results.xlsx", models, planned)
    assert_csv(live_dir / "15min_workflow_results.csv", models, planned)
    assert sorted(p.name for p in (live_dir / "results_15min").glob("*.jsonl")) == ["lassocv.jsonl", "ols.jsonl"]
