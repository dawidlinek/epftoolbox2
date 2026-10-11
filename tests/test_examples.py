"""End-to-end runs of every script in examples/ against the live APIs, on a short window.

Each example has a `main(...)` that takes its date ranges and an output directory, so a test can run the real
script on a few weeks of data. These tests need ENTSOE_API_KEY and the network, which is why they are marked
`integration` (CI's fast suite skips them) and `examples` (run only these with `uv run pytest -m examples`).

Every test runs in its own `tmp_path`: it is the working directory, so the source cache (`.cache/sources`) and
the output files land there, and the example is loaded from its file like a user would run it. A test asserts
real outcomes: a non-empty data frame with the expected columns, forecasts for every model on the same keys,
no unexpected missing forecasts, a passing information-set check and the output files.
"""

import datetime as dt
import importlib.util
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from epftoolbox2.information_set import InformationSet

pytestmark = [pytest.mark.integration, pytest.mark.examples]

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# A short back-test inside winter, away from every DST switch, so no forecast may be missing. The data starts two
# months before the test and ends past test_end + horizon, as the models need the d+1..d+6 targets and weather.
DATA_START, DATA_END = "2023-12-01", "2024-02-12"
TEST_START, TEST_END = "2024-02-01", "2024-02-03"
RUN_DAYS, HORIZON = 3, 6

# Environment variables that the examples (at import) and Workflow.run() set on the whole process.
PROCESS_ENV = (
    "PYTHON_GIL",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLAS_NUM_THREADS",
)

PRICE_COLUMNS = [
    "price",
    "load_actual",
    "price_d-1",
    "price_d-7",
    "load_actual_d-1",
    "load_actual_d-7",
    "load_forecast_d+1",
    "is_holiday_d+1",
    "is_monday_d+1",
    "daylight_hours_d+1",
    "warsaw_temperature_2m_d+1",
    "warsaw_temperature_2m_d+6",
]
LOAD_15MIN_COLUMNS = [
    "load_actual",
    "load_actual_d-1",
    "load_actual_d-7",
    "load_forecast_d+1",
    "is_holiday_d+1",
    "daylight_hours_d+1",
    "berlin_temperature_2m_d+1",
    "berlin_temperature_2m_d+6",
]


@pytest.fixture
def sandbox(tmp_path, monkeypatch, entsoe_api_key):
    """Work in tmp_path with one process and one thread, and put the process environment back afterwards.

    The examples read the key when they are imported and set MAX_PROCESSES, THREADS_PER_PROCESS and a few more
    variables for the whole process, so each is registered with monkeypatch first to be restored on teardown.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENTSOE_API_KEY", entsoe_api_key)
    for name in PROCESS_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    return tmp_path


@pytest.fixture
def information_set_checks(monkeypatch):
    """Record every InformationSet.check call that returns, so a test can show the check ran and passed.

    The check raises InformationSetError on a predictor that is not known at issue time, so a recorded call
    means it did not.
    """
    calls = []
    original = InformationSet.check

    def recording_check(self, expanded, *args, **kwargs):
        original(self, expanded, *args, **kwargs)
        calls.append((kwargs.get("model_name"), self.issue_time, self.issue_timezone))

    monkeypatch.setattr(InformationSet, "check", recording_check)
    return calls


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


def assert_data_frame(df: pd.DataFrame, columns, zone: str, first_day: dt.date, last_day: dt.date) -> None:
    """A non-empty local-time frame over the requested days, with the key columns and no all-NaN column."""
    assert not df.empty
    assert str(df.index.tz) == zone
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert (df.index[0].date(), df.index[-1].date()) == (first_day, last_day)
    missing = [c for c in columns if c not in df.columns]
    assert not missing, f"columns missing from the data frame: {missing}"
    empty = [c for c in df.columns if df[c].isna().all()]
    assert not empty, f"columns without a single value: {empty}"


def forecast_keys(report) -> dict:
    """The set of (run_date, target_date, hour, horizon) each model produced."""
    predictions = report.predictions()
    keys = ["run_date", "target_date", "hour", "horizon"]
    return {name: set(group[keys].itertuples(index=False, name=None)) for name, group in predictions.groupby("model")}


def switch_days(first: dt.date, last: dt.date, zone: str) -> list:
    """Local dates in [first, last] whose UTC offset differs from the previous day's (DST switch days)."""
    tz = ZoneInfo(zone)

    def offset(day: dt.date):
        return dt.datetime.combine(day, dt.time(12), tzinfo=tz).utcoffset()

    return [
        d
        for d in (first + dt.timedelta(days=i) for i in range((last - first).days + 1))
        if offset(d) != offset(d - dt.timedelta(days=1))
    ]


def assert_forecasts(report, models, planned: int, max_missing: int = 0) -> None:
    """Every model planned `planned` forecasts, made all but `max_missing` of them, and made the same ones.

    `max_missing` is nonzero only for the documented DST cases (see .claude/rules/models.md): a window with no
    DST switch must be complete.
    """
    counts = report.forecast_counts().set_index("model")
    assert list(counts.index) == models
    assert (counts["planned"] == planned).all(), counts
    assert (counts["missing"] <= max_missing).all(), counts
    assert (counts["produced"] == counts["planned"] - counts["missing"]).all(), counts
    keys = forecast_keys(report)
    assert sorted(keys) == sorted(models)
    assert len({frozenset(k) for k in keys.values()}) == 1, "the models did not forecast the same keys"
    predictions = report.predictions()
    assert predictions["prediction"].notna().all()
    assert predictions["prediction"].abs().max() < 100_000


def assert_information_set_checked(calls, models, issue_time="11:45", zone="Europe/Warsaw") -> None:
    assert {name for name, _, _ in calls} == set(models)
    assert {(t, z) for _, t, z in calls} == {(issue_time, zone)}


def assert_metrics(report, models, planned: int, columns=("MAE", "RMSE")) -> None:
    summary = report.summary().set_index("model")
    assert list(summary.index) == models
    assert (summary["forecasts"] == planned).all()
    for column in columns:
        assert column in summary.columns
        assert summary[column].notna().all() and (summary[column] > 0).all()


def assert_excel(path: Path, models, planned: int) -> None:
    """The workbook has the summary sheet with every model's forecast count, and the per-forecast details."""
    assert path.is_file()
    sheets = pd.read_excel(path, sheet_name=None)
    assert {"Summary", "Details"} <= set(sheets)
    summary = sheets["Summary"].set_index("model")
    assert list(summary.index) == models
    assert (summary["forecasts"] == planned).all()


def assert_csv(path: Path, models, planned: int) -> None:
    """One row per forecast key with every model's prediction and error next to the actual."""
    assert path.is_file()
    exported = pd.read_csv(path)
    assert len(exported) == planned
    assert exported["actual"].notna().all()
    for model in models:
        prediction, error = exported[f"{model}_prediction"], exported[f"{model}_error"]
        assert prediction.notna().all()
        assert ((prediction - exported["actual"]) - error).abs().max() < 1e-9


def test_01_data_pipeline_only(sandbox, monkeypatch):
    example = load_example("01_data_pipeline_only.py", monkeypatch)

    df = example.main(data_start=DATA_START, data_end=DATA_END, output_dir=sandbox)

    assert_data_frame(
        df,
        PRICE_COLUMNS,
        "Europe/Warsaw",
        dt.date.fromisoformat(DATA_START),
        dt.date.fromisoformat(DATA_END),
    )
    written = sandbox / "data_output.csv"
    assert written.is_file()
    reread = pd.read_csv(written, index_col=0)
    assert reread.shape == df.shape
    assert list(reread.columns) == list(df.columns)


def test_02_model_pipeline_only(sandbox, monkeypatch, information_set_checks):
    # Example 2 reads the CSV that example 1 writes
    data_example = load_example("01_data_pipeline_only.py", monkeypatch)
    data_example.main(data_start=DATA_START, data_end=DATA_END, output_dir=sandbox)
    example = load_example("02_model_pipeline_only.py", monkeypatch)
    models = ["OLS Baseline", "Lasso CV"]
    planned = RUN_DAYS * HORIZON * 24

    report = example.main(
        data_path=sandbox / "data_output.csv", test_start=TEST_START, test_end=TEST_END, output_dir=sandbox
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
    assert_excel(sandbox / "model_results.xlsx", models, planned)
    assert_csv(sandbox / "model_results.csv", models, planned)
    assert sorted(p.name for p in (sandbox / "results").glob("*.jsonl")) == ["lasso_cv.jsonl", "ols_baseline.jsonl"]


def test_03_full_workflow(sandbox, monkeypatch, information_set_checks):
    example = load_example("03_full_workflow.py", monkeypatch)
    models = ["OLS", "LassoCV"]
    planned = RUN_DAYS * HORIZON * 24

    report = example.main(
        data_start=DATA_START, data_end=DATA_END, test_start=TEST_START, test_end=TEST_END, output_dir=sandbox
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
    assert_excel(sandbox / "full_workflow_results.xlsx", models, planned)
    assert_csv(sandbox / "full_workflow_results.csv", models, planned)
    assert sorted(p.name for p in (sandbox / "results").glob("*.jsonl")) == ["lassocv.jsonl", "ols.jsonl"]
    assert (sandbox / ".cache" / "sources").is_dir()


def test_04_yaml_workflow(sandbox, monkeypatch, information_set_checks, entsoe_api_key):
    example = load_example("04_yaml_workflow.py", monkeypatch)
    models = ["OLS", "LassoCV"]
    planned = RUN_DAYS * HORIZON * 24

    report = example.main(
        data_start=DATA_START,
        data_end=DATA_END,
        test_start=TEST_START,
        test_end=TEST_END,
        output_dir=sandbox,
        run=True,
    )

    yamls = {
        name: (sandbox / name).read_text() for name in ("data_pipeline.yaml", "model_pipeline.yaml", "experiment.yaml")
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
    assert_excel(sandbox / "workflow_results.xlsx", models, planned)


def test_05_forecast_only(sandbox, monkeypatch, information_set_checks):
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
    assert not list(sandbox.glob("*.xlsx")) and not list(sandbox.glob("*.csv"))


def test_06_15min_forecasting(sandbox, monkeypatch, information_set_checks):
    example = load_example("06_15min_forecasting.py", monkeypatch)
    models = ["OLS", "LassoCV"]
    planned = RUN_DAYS * HORIZON * 96

    report = example.main(
        data_start=DATA_START, data_end=DATA_END, test_start=TEST_START, test_end=TEST_END, output_dir=sandbox
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
    assert pd.read_csv(sandbox / "15min_forecasting_data.csv", index_col=0).shape == report.source_data.shape
    assert_excel(sandbox / "15min_workflow_results.xlsx", models, planned)
    assert_csv(sandbox / "15min_workflow_results.csv", models, planned)
    assert sorted(p.name for p in (sandbox / "results_15min").glob("*.jsonl")) == ["lassocv.jsonl", "ols.jsonl"]
