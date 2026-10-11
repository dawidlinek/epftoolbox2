"""What the live tests (tests/test_examples.py, tests/test_docs_code.py) share: the short back-test window, the
columns the price pipelines must produce and the assertions on a data frame, a report and the files it exports."""

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# A short back-test inside winter, away from every DST switch, so no forecast may be missing. The data starts two
# months before the test and ends past test_end + horizon, as the models need the d+1..d+6 targets and weather.
DATA_START, DATA_END = "2023-12-01", "2024-02-12"
TEST_START, TEST_END = "2024-02-01", "2024-02-03"
RUN_DAYS, HORIZON = 3, 6

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
