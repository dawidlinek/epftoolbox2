"""Every `by_*()` table says how many forecasts each model has in each group, and warns when the models of a group differ.

Two models are only comparable on the same forecasts. A table of MAEs by horizon in which one model has 6 forecasts at
horizon 7 and another 14 compares different samples, and used to say nothing about it. The count is the `forecasts` column
of `summary()`, after the group keys; the warning and `df.attrs["unequal_counts"]` name the groups that differ. Nothing
numeric moves: the metric columns are what they were.
"""

import re
import warnings

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator, rMAEEvaluator
from epftoolbox2.exporters import ExcelExporter, TerminalExporter
from epftoolbox2.information_set import InformationSet
from epftoolbox2.models import NaiveModel
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.report import GROUP_CONFIGS, EvaluationReport
from tests.forecast_records import forecast_records

WARNING = "different numbers of forecasts"

# method -> its grouping columns, for every table that has groups: the exporter tables and the breakdowns.
GROUPINGS = {
    **{f"by_{name}": keys for name, keys in GROUP_CONFIGS},
    "by_quarter": ["year", "quarter"],
    "by_day": ["target_date"],
    "by_price_level": ["price_level"],
}

# Run days over a quarter end, so that the quarters, months and weekdays have more than one group.
RUN_DATES = pd.date_range("2023-03-01", "2023-04-30", freq="D")


def lacks_h3_on_odd_days(record: dict) -> bool:
    """A source that isn't out yet for the furthest horizon on half of the days."""
    return record["horizon"] == 3 and record["day_in_test"] % 2 == 1


def study(**drops):
    """A model with every key, and one without some: the same days, hours and horizons otherwise."""
    return {
        "full": forecast_records(1, RUN_DATES, noise=5.0, horizons=(1, 2, 3)),
        "gap": forecast_records(2, RUN_DATES, noise=4.0, horizons=(1, 2, 3), drop=drops.get("gap")),
    }


def silently(call):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


def expected_counts(results: dict, method: str, threshold: float = float("nan")) -> dict:
    """{(model, *group): number of records} counted from the raw records, with the group of each record read from its
    dates, hour and actual by the definition in the docs and not by the report's code."""
    counts = {}
    for model, records in results.items():
        frame = pd.DataFrame(records)
        run, target = pd.to_datetime(frame["run_date"]), pd.to_datetime(frame["target_date"])
        keys = {
            "hour": frame["hour"],
            "horizon": frame["horizon"],
            "year": target.dt.year,
            "quarter": target.dt.quarter,
            "run_weekday": run.dt.dayofweek,
            "target_weekday": target.dt.dayofweek,
            "target_date": target,
            "price_level": pd.Series(
                np.where(frame["actual"] < 0, "negative", np.where(frame["actual"] >= threshold, "spike", "normal")),
                index=frame.index,
            ),
        }
        group = GROUPINGS[method]
        for key, part in frame.groupby([keys[k] for k in group], sort=False):
            key = key if isinstance(key, tuple) else (key,)
            counts[(model, *key)] = len(part)
    return counts


def as_counts(table: pd.DataFrame, method: str) -> dict:
    keys = GROUPINGS[method]
    return {(row["model"], *(row[k] for k in keys)): row["forecasts"] for row in table.to_dict("records")}


# ---------------------------------------------------------------------------------------------------------------------
# The count: in every table, counted from the records
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", list(GROUPINGS))
def test_every_by_table_counts_each_models_forecasts_in_each_group(method):
    results = study(gap=lacks_h3_on_odd_days)
    report = EvaluationReport(results, [MAEEvaluator(), RMSEEvaluator(), rMAEEvaluator("full")])

    table = silently(getattr(report, method))
    keys = GROUPINGS[method]

    # After the group keys, before the metrics, as in `by_quarter()`: `model, <keys>, forecasts, MAE, ...`
    assert list(table.columns[: len(keys) + 2]) == ["model", *keys, "forecasts"]
    assert list(table.columns[len(keys) + 2 :]) == ["MAE", "RMSE", "rMAE"]
    threshold = table.attrs.get("spike_threshold", float("nan"))
    assert as_counts(table, method) == expected_counts(results, method, threshold)
    # The groups cover every forecast, so the counts add up to `summary()`.
    totals = table.groupby("model")["forecasts"].sum()
    assert totals.to_dict() == silently(report.summary).set_index("model")["forecasts"].to_dict()


def test_compute_all_has_the_counts_of_the_by_tables():
    """The exporters read `compute_all()`, so its tables carry the count too, and equal the `by_*()` ones."""
    report = EvaluationReport(study(gap=lacks_h3_on_odd_days), [MAEEvaluator()])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tables = report.compute_all()
        by_tables = {name: getattr(report, f"by_{name}")() for name, _ in GROUP_CONFIGS}

    for name, _ in GROUP_CONFIGS:
        pd.testing.assert_frame_equal(tables[name], by_tables[name], check_exact=True)
        assert tables[name].attrs == by_tables[name].attrs
    # One warning for each of the seven tables, from each way of building them.
    assert sum(WARNING in str(w.message) for w in caught) == 2 * len(GROUP_CONFIGS)


# ---------------------------------------------------------------------------------------------------------------------
# The warning
# ---------------------------------------------------------------------------------------------------------------------


def test_by_horizon_shows_a_model_with_fewer_forecasts_at_one_horizon_and_warns():
    """Ten run days of two hours: 20 forecasts a horizon. `gap` lacks horizon 3 after the fourth run day, as a TSO forecast
    that isn't out yet would: 8 forecasts to the full model's 20, and nothing else differs."""
    days = pd.date_range("2023-03-01", periods=10)
    results = {
        "full": forecast_records(1, days, horizons=(1, 2, 3), periods=lambda d: 2),
        "gap": forecast_records(
            2, days, horizons=(1, 2, 3), periods=lambda d: 2, drop=lambda r: r["horizon"] == 3 and r["day_in_test"] >= 4
        ),
    }
    report = EvaluationReport(results, [MAEEvaluator()])

    with pytest.warns(UserWarning, match=WARNING) as caught:
        table = report.by_horizon()

    assert table[["model", "horizon", "forecasts"]].to_dict("records") == [
        {"model": "full", "horizon": 1, "forecasts": 20},
        {"model": "gap", "horizon": 1, "forecasts": 20},
        {"model": "full", "horizon": 2, "forecasts": 20},
        {"model": "gap", "horizon": 2, "forecasts": 20},
        {"model": "full", "horizon": 3, "forecasts": 20},
        {"model": "gap", "horizon": 3, "forecasts": 8},
    ]
    assert table.attrs["unequal_counts"] == [(3,)]
    message = str(caught[0].message)
    assert "1 of 3 groups by horizon" in message
    assert "horizon=3: full: 20, gap: 8" in message
    assert caught[0].filename == __file__  # raised at the call of the public method, not inside the report


@pytest.mark.parametrize("method", list(GROUPINGS))
def test_every_by_table_warns_on_the_groups_where_counts_differ(method):
    results = study(gap=lacks_h3_on_odd_days)
    report = EvaluationReport(results, [MAEEvaluator()])

    with pytest.warns(UserWarning, match=WARNING) as caught:
        table = getattr(report, method)()

    counts = expected_counts(results, method, table.attrs.get("spike_threshold", float("nan")))
    groups = {key[1:] for key in counts}
    differing = sorted(group for group in groups if counts.get(("full", *group), 0) != counts.get(("gap", *group), 0))
    assert differing  # the study has some in every table
    assert sorted(table.attrs["unequal_counts"]) == differing
    assert len([w for w in caught if WARNING in str(w.message)]) == 1
    assert f"in {len(differing)} of {len(groups)} groups by {', '.join(GROUPINGS[method])}" in str(caught[0].message)


@pytest.mark.parametrize("method", list(GROUPINGS))
def test_models_with_the_same_forecast_counts_do_not_warn(method):
    report = EvaluationReport(study(), [MAEEvaluator(), rMAEEvaluator("full")])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = getattr(report, method)()

    assert "unequal_counts" not in table.attrs
    per_group = table.pivot(index=GROUPINGS[method], columns="model", values="forecasts")
    assert (per_group["full"] == per_group["gap"]).all()


def test_one_model_has_nothing_to_differ_from():
    report = EvaluationReport({"only": study()["full"]}, [MAEEvaluator()])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name, _ in GROUP_CONFIGS:
            getattr(report, f"by_{name}")()


def test_a_model_without_a_forecast_in_a_group_counts_zero_there():
    """`late` has no forecast at horizon 1 at all, so it has no row there. It is still a difference."""
    days = pd.date_range("2023-03-01", periods=10)
    results = {
        "full": forecast_records(1, days, horizons=(1, 2), periods=lambda d: 2),
        "late": forecast_records(2, days, horizons=(1, 2), periods=lambda d: 2, drop=lambda r: r["horizon"] == 1),
    }
    report = EvaluationReport(results, [MAEEvaluator()])

    with pytest.warns(UserWarning, match=WARNING) as caught:
        table = report.by_horizon()

    assert table[table["horizon"] == 1]["model"].tolist() == ["full"]
    assert table.attrs["unequal_counts"] == [(1,)]
    assert "horizon=1: full: 20, late: 0" in str(caught[0].message)


def test_a_model_with_no_forecast_at_all_is_zero_in_every_group():
    report = EvaluationReport({"full": study()["full"], "none": []}, [MAEEvaluator()])

    with pytest.warns(UserWarning, match=WARNING) as caught:
        table = report.by_hour()

    assert table["model"].unique().tolist() == ["full"]
    assert len(table.attrs["unequal_counts"]) == 24
    assert "none: 0" in str(caught[0].message)


def test_a_row_without_a_finite_actual_is_in_the_count_and_not_in_the_mean():
    """The count is of rows, as in `summary()`: MAE skips the row, the count keeps it."""
    days = pd.date_range("2023-03-01", periods=10)
    results = {name: forecast_records(i, days, periods=lambda d: 2) for i, name in enumerate(["a", "b"])}
    results["b"][0]["actual"] = float("nan")  # run day 1, hour 0
    report = EvaluationReport(results, [MAEEvaluator()])

    table = silently(report.by_hour).set_index(["model", "hour"])
    summary = silently(report.summary).set_index("model")

    assert table.loc[("b", 0), "forecasts"] == 10 and table.loc[("a", 0), "forecasts"] == 10
    assert summary.loc["b", "forecasts"] == 20
    frame = pd.DataFrame(results["b"])
    errors = (frame["prediction"] - frame["actual"]).abs()[frame["hour"] == 0].dropna()
    assert len(errors) == 9
    assert table.loc[("b", 0), "MAE"] == pytest.approx(errors.mean(), rel=1e-12)


def test_the_metrics_are_on_the_models_own_forecasts_whatever_the_counts():
    """The count adds a column and a warning: MAE of a group is still the mean over the model's own rows there."""
    results = study(gap=lacks_h3_on_odd_days)
    report = EvaluationReport(results, [MAEEvaluator()])

    table = silently(report.by_horizon).set_index(["model", "horizon"])

    for model, records in results.items():
        frame = pd.DataFrame(records)
        error = (frame["prediction"] - frame["actual"]).abs()
        for horizon, errors in error.groupby(frame["horizon"]):
            assert table.loc[(model, horizon), "MAE"] == pytest.approx(errors.mean(), rel=1e-12)


# ---------------------------------------------------------------------------------------------------------------------
# A NaiveModel whose source isn't out yet, next to a model with every key
# ---------------------------------------------------------------------------------------------------------------------

ZONE = "Europe/Warsaw"
TSO = "load_forecast_daily_max_d+{horizon}"
# Two whole weeks, Monday to Sunday: the built-in rule of the TSO's week-ahead forecast publishes it on Friday at 10:00, so
# at noon Monday lacks h=7, Tuesday h=6-7, Wednesday h=5-7, Thursday h=4-7.
START, END = "2024-03-25", "2024-04-07"
TSO_PER_HORIZON = {1: 14, 2: 14, 3: 14, 4: 12, 5: 10, 6: 8, 7: 6}


def daily_frame(first="2024-03-18", days=40):
    dates = pd.date_range(first, periods=days, freq="D")
    index = (dates + pd.Timedelta(hours=12)).tz_localize(ZONE)
    frame = pd.DataFrame({"actual_daily_max": 20000 + 50 * np.arange(days, dtype=float)}, index=index)
    for k in range(1, 8):
        frame[f"load_forecast_daily_max_d+{k}"] = 1000.0 * k + np.arange(days)
    return frame


@pytest.fixture(scope="module")
def naive_report():
    """`TSO` is a `NaiveModel(known_only=True)`: no forecast where its value isn't published at the issue time. `all` reads the
    same column with `known_only=False` (no information set check at `availability="none"`), so it has every key."""
    mp = pytest.MonkeyPatch()
    mp.setenv("MAX_PROCESSES", "1")
    mp.setenv("THREADS_PER_PROCESS", "1")
    try:
        pipeline = (
            ModelPipeline(information_set=InformationSet("12:00", ZONE), availability="none")
            .add_model(NaiveModel([TSO], name="TSO"))
            .add_model(NaiveModel([TSO], name="all", known_only=False))
            .add_evaluator(MAEEvaluator())
            .add_evaluator(rMAEEvaluator("all"))
        )
        yield pipeline.run(daily_frame(), START, END, target="actual_daily_max", horizon=7, freq="1D")
    finally:
        mp.undo()


def test_by_horizon_of_a_naive_benchmark_shows_the_keys_it_lacks(naive_report):
    summary = naive_report.summary().set_index("model")["forecasts"]
    assert summary.to_dict() == {"TSO": 78, "all": 98}  # as `forecast_counts()` says: 20 `not_published`

    with pytest.warns(UserWarning, match=WARNING) as caught:
        table = naive_report.by_horizon()

    forecasts = table.set_index(["model", "horizon"])["forecasts"]
    assert forecasts.loc["TSO"].to_dict() == TSO_PER_HORIZON
    assert forecasts.loc["all"].to_dict() == {h: 14 for h in range(1, 8)}
    assert table.attrs["unequal_counts"] == [(4,), (5,), (6,), (7,)]
    assert "horizon=7: TSO: 6, all: 14" in str(caught[0].message)
    # rMAE is on the keys the pair shares, so the TSO's own count is not what its rMAE rests on: the table says which is which.
    assert forecasts.loc["TSO"].sum() == summary["TSO"]


@pytest.mark.parametrize(
    "method", ["by_hour", "by_year", "by_run_weekday_horizon", "by_target_weekday_horizon", "by_quarter"]
)
def test_the_other_tables_of_a_naive_benchmark_add_up_to_its_forecasts(naive_report, method):
    with pytest.warns(UserWarning, match=WARNING):
        table = getattr(naive_report, method)()

    assert table.groupby("model")["forecasts"].sum().to_dict() == {"TSO": 78, "all": 98}
    assert table.attrs["unequal_counts"]


# ---------------------------------------------------------------------------------------------------------------------
# The exporters
# ---------------------------------------------------------------------------------------------------------------------


def table_cells(output: str, title: str) -> list:
    """The cells of the rich table printed under `title`, header first."""
    lines = output.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == title)
    rows = []
    for line in lines[start + 1 :]:
        if line.startswith("│") or line.startswith("┃"):
            rows.append([cell.strip() for cell in re.split(r"[│┃]", line)[1:-1]])
        elif rows and line.startswith("└"):
            break
    return rows


def test_the_terminal_prints_the_counts_of_every_table(naive_report, capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")

    with pytest.warns(UserWarning, match=WARNING):
        TerminalExporter(show=["summary", "horizon", "hour"]).export(naive_report)

    output = capsys.readouterr().out
    summary = table_cells(output, "Summary")
    assert summary[0][:2] == ["model", "forecasts"]
    by_horizon = table_cells(output, "By Horizon")
    assert by_horizon[0][:3] == ["model", "horizon", "forecasts"]
    rows = {(row[0], row[1]): row[2] for row in by_horizon[1:]}
    assert rows[("TSO", "7")] == "6" and rows[("all", "7")] == "14"
    assert table_cells(output, "By Hour")[0][:3] == ["model", "hour", "forecasts"]


@pytest.fixture(scope="module")
def late_report():
    """`late` lacks every forecast at horizon 1 and every forecast issued on a Monday; `full` has all of them."""
    days = pd.date_range("2023-03-01", periods=14)
    results = {
        "full": forecast_records(1, days, horizons=(1, 2), periods=lambda d: 3),
        "late": forecast_records(
            2, days, horizons=(1, 2), periods=lambda d: 3, drop=lambda r: r["horizon"] == 1 or r["run_weekday"] == 0
        ),
    }
    return EvaluationReport(results, [MAEEvaluator()])


def sheet(book, name):
    return [list(row) for row in book[name].iter_rows(values_only=True)]


def test_excel_has_a_sheet_of_counts_after_the_sheets_of_the_metrics_of_each_breakdown(tmp_path, late_report):
    path = tmp_path / "report.xlsx"
    with pytest.warns(UserWarning, match=WARNING):
        ExcelExporter(str(path)).export(late_report)
    book = load_workbook(path)

    names = book.sheetnames
    assert names == [
        "Summary",
        "hour_MAE",
        "hour_forecasts",
        "horizon_MAE",
        "horizon_forecasts",
        "HourHorizon_MAE",
        "HourHorizon_forecasts",
        "year_MAE",
        "year_forecasts",
        "YearHorizon_MAE",
        "YearHorizon_forecasts",
        "RunWeekdayHorizon_MAE",
        "RunWeekdayHorizon_forecasts",
        "TgtWeekdayHorizon_MAE",
        "TgtWeekdayHorizon_forecasts",
        "Details",
    ]
    # Horizon 1: `late` has no forecast, so no row in the table, 0 in the count sheet and nothing in the metric sheet.
    assert sheet(book, "horizon_forecasts") == [
        ["forecasts", "horizon_1", "horizon_2"],
        ["full", 42, 42],
        ["late", 0, 36],
    ]
    metric = sheet(book, "horizon_MAE")
    assert metric[2][1] is None and metric[2][2] is not None


def test_excel_counts_per_model_in_the_matrix_sheets_and_does_not_colour_them(tmp_path, late_report):
    path = tmp_path / "report.xlsx"
    with pytest.warns(UserWarning, match=WARNING):
        ExcelExporter(str(path), sheets=["hour_horizon", "run_weekday_horizon"]).export(late_report)
    book = load_workbook(path)

    # One block per model with the hours down and the horizons across, each on the groups of the whole table: `late` has
    # no forecast at horizon 1, so 0 there, and 12 at horizon 2 (14 run days, two of them Mondays).
    blocks = sheet(book, "HourHorizon_forecasts")
    assert blocks[0][0] == "full" and "late" in blocks[0]
    assert blocks[1][:3] == [None, "horizon_1", "horizon_2"]
    assert blocks[2][:3] == ["hour_0", 14, 14]
    first = blocks[0].index("late")
    assert blocks[1][first : first + 3] == [None, "horizon_1", "horizon_2"]
    assert blocks[2][first : first + 3] == ["hour_0", 0, 12]
    weekdays = sheet(book, "RunWeekdayHorizon_forecasts")
    assert weekdays[0][0] == "full" and weekdays[2][:3] == ["Mon", 6, 6]  # two Mondays of three hours
    late = weekdays[0].index("late")
    assert [row[late] for row in weekdays[2:]] == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assert [row[late + 1] for row in weekdays[2:]] == [0] * 7  # no forecast at horizon 1
    assert [row[late + 2] for row in weekdays[2:]] == [0] + [6] * 6  # and none issued on a Monday
    assert len(book["HourHorizon_forecasts"].conditional_formatting) == 0
    assert len(book["HourHorizon_MAE"].conditional_formatting) > 0
    assert len(book["RunWeekdayHorizon_forecasts"].conditional_formatting) == 0


def test_excel_summary_colours_the_metrics_and_not_the_count(tmp_path, late_report):
    path = tmp_path / "report.xlsx"
    ExcelExporter(str(path), sheets=["summary"]).export(late_report)
    book = load_workbook(path)

    assert [cell.value for cell in book["Summary"][1]] == ["model", "forecasts", "MAE"]
    ranges = [str(rule.sqref) for rule in book["Summary"].conditional_formatting]
    assert ranges == ["C2:C3"]  # the MAE of the two models


def test_excel_count_sheets_are_numbers_that_equal_the_tables(tmp_path, late_report):
    path = tmp_path / "report.xlsx"
    with pytest.warns(UserWarning, match=WARNING):
        ExcelExporter(str(path), sheets=["hour", "year"]).export(late_report)
        tables = late_report.compute_all()
    book = load_workbook(path)

    for name, key in (("hour", "hour"), ("year", "year")):
        rows = sheet(book, f"{name}_forecasts")
        written = pd.DataFrame(rows[1:], columns=rows[0]).set_index("forecasts")
        expected = tables[name].pivot(index="model", columns=key, values="forecasts").fillna(0).astype(int)
        for model in expected.index:
            assert [int(v) for v in written.loc[model]] == expected.loc[model].tolist()
