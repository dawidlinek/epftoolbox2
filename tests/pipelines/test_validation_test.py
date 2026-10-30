"""`run(df, validation=..., test=...)`: one run over both windows gives what a plain run over each window gives.

A forecast does not depend on the role it plays. The main trap is the scaler's binary-column mask, which is read from the
training window of the first day a list is used: with one mask per run, adding validation days would change test forecasts
without a word. So each test here that compares numbers compares them bit for bit, as text.
"""

import json
import re
import warnings
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from epftoolbox2 import cli
from epftoolbox2.evaluators import DMEvaluator, MAEEvaluator, RMSEEvaluator, rMAEEvaluator
from epftoolbox2.exporters import CsvExporter, ExcelExporter, TerminalExporter
from epftoolbox2.information_set import Availability, InformationSet, InformationSetError, leakage_probe
from epftoolbox2.models import BaseModel, LassoCVModel, OLSModel
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline, Workflow
from epftoolbox2.results import EvaluationReport, RunWindows, Segment, read_windows
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

RULES = {
    # The synthetic drivers stand for forecasts published well before their delivery.
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published("13:00", 1),
    "flag": Availability.static(),
}
INFO = InformationSet("11:45", "Europe/Warsaw", rules=RULES)
V = ("2023-03-28", "2023-03-30")
T = ("2023-04-03", "2023-04-05")
FLAG_TURNS = "2023-03-30"  # after what the first validation day's training window holds, before the first test day's
SMALLER = ["load_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]
WIND_ONLY = ["wind_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]


class InProcessPool:
    """Stands in for the worker pool, whose start is most of the time these runs would take. The tests about the
    pool itself use the real one."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        _worker_init(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, task, *args):
        future = Future()
        try:
            future.set_result(task(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)


def with_flag(frame, day):
    """A column that is 0/1 before `day` and takes other values from then on, so a list's mask depends on when it is read."""
    out = frame.copy()
    flag = (np.random.default_rng(5).random(len(frame)) > 0.5).astype(float)
    rows = out.index.tz_localize(None).normalize() >= pd.Timestamp(day)
    flag[rows] = flag[rows] * 0.5 + 0.25
    out["flag"] = flag
    return out


@pytest.fixture(scope="module")
def df():
    return with_flag(make_market_data(horizon=3, noise=5.0, seed=3), FLAG_TURNS)


def by_day_and_value(ctx):
    """A list that depends on the day, and on a value it reads: nothing here may depend on the window, which it never sees."""
    names = list(PREDICTORS)
    if ctx.issue_day.day % 2:
        names.append("flag")
    if ctx.published("load_d+{horizon}") and ctx.value("load_d+{horizon}") > 1000:
        names.remove("wind_d+{horizon}")
    return names


FIXED = PREDICTORS + ["flag"]


def lines(records, *drop):
    """Records as text, which compares them bit for bit (and NaN equal to NaN), whatever order they finished in."""
    return sorted(json.dumps({k: v for k, v in r.items() if k not in drop}, sort_keys=True) for r in records)


def of_window(records, window):
    return [r for r in records if r["window"] == window]


def stored(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def pipeline(*models, info=None, level=None, evaluators=()):
    built = ModelPipeline(information_set=info, availability=level)
    for model in models:
        built.add_model(model)
    built.add_evaluator(*evaluators)
    return built


def ols(predictors=FIXED, name="m", window=30):
    return OLSModel(predictors, training_window=window, name=name)


def all_records(report, name="m"):
    return report.refs[name]._results


def assert_identity(make, frame, validation, test, horizon, freq="1h", info=None, level=None):
    """Test and validation records of the combined run are bitwise the plain runs', apart from the tag."""
    combined = all_records(pipeline(make(), info=info, level=level).run(frame, validation=validation, test=test, horizon=horizon, freq=freq))  # fmt: skip
    plain_test = all_records(pipeline(make(), info=info, level=level).run(frame, test=test, horizon=horizon, freq=freq))
    plain_validation = all_records(
        pipeline(make(), info=info, level=level).run(frame, test=validation, horizon=horizon, freq=freq)
    )
    assert lines(of_window(combined, "test"), "window") == lines(plain_test)
    assert lines(of_window(combined, "validation"), "window") == lines(plain_validation)
    assert all("window" not in r for r in plain_test)
    return combined


# ---------------------------------------------------------------------------
# Bitwise identity with the plain runs
# ---------------------------------------------------------------------------

SWITCHES = {
    "spring": dict(
        start="2023-01-01",
        days=120,
        validation=("2023-03-25", "2023-03-27"),
        test=("2023-04-03", "2023-04-05"),
        turns="2023-03-30",  # fmt: skip
    ),
    "autumn": dict(
        start="2023-08-01",
        days=120,
        validation=("2023-09-20", "2023-09-22"),
        test=("2023-10-28", "2023-10-30"),
        turns="2023-10-10",  # fmt: skip
    ),
}
LASSO_DAYS = {
    "spring": (("2023-03-26", "2023-03-26"), ("2023-04-03", "2023-04-03")),
    "autumn": (("2023-09-20", "2023-09-20"), ("2023-10-29", "2023-10-29")),
}


def switch_data(spec, freq):
    frame = make_market_data(start=spec["start"], days=spec["days"], freq=freq, horizon=3, noise=5.0, seed=3)
    return with_flag(frame, spec["turns"])


@pytest.mark.parametrize("switch", list(SWITCHES))
@pytest.mark.parametrize("freq", ["1h", "15min"])
@pytest.mark.parametrize("horizon", [1, 3])
@pytest.mark.parametrize("predictors", [FIXED, by_day_and_value], ids=["fixed list", "ctx function"])
def test_ols_records_are_the_plain_runs_records(switch, freq, horizon, predictors):
    spec = SWITCHES[switch]
    frame = switch_data(spec, freq)
    make = lambda: ols(predictors, window=45)  # noqa: E731
    info, level = (INFO, "raise") if callable(predictors) else (None, None)
    combined = assert_identity(make, frame, spec["validation"], spec["test"], horizon, freq, info, level)
    assert {r["run_date"] for r in of_window(combined, "validation")} <= set(pd.date_range(*spec["validation"]).strftime("%Y-%m-%d"))  # fmt: skip


@pytest.mark.parametrize(
    ("predictors", "switch", "freq", "horizon"),
    [
        (FIXED, "spring", "1h", 1),
        (FIXED, "autumn", "1h", 3),
        (FIXED, "autumn", "15min", 1),
        (FIXED, "spring", "15min", 3),
        (by_day_and_value, "autumn", "1h", 3),
        (by_day_and_value, "spring", "15min", 1),
    ],
    ids=lambda value: value if isinstance(value, (str, int)) else ("ctx function" if callable(value) else "fixed list"),
)
def test_lasso_records_are_the_plain_runs_records(switch, freq, horizon, predictors):
    frame = switch_data(SWITCHES[switch], freq)
    validation, test = LASSO_DAYS[switch]
    make = lambda: LassoCVModel(predictors, training_window=14, cv=2, max_iter=200, name="m")  # noqa: E731
    info, level = (INFO, "raise") if callable(predictors) else (None, None)
    assert_identity(make, frame, validation, test, horizon, freq, info, level)


def test_the_trap_each_window_has_its_own_masks(df):
    """The flag is binary in the training window of the first validation day and isn't in the first test day's. With one
    mask for the run, as one plain run over the whole span has, the test forecasts come out different."""
    report = pipeline(ols()).run(df, validation=V, test=T, horizon=1)
    plain = pipeline(ols()).run(df, test=T, horizon=1)
    assert lines(of_window(all_records(report), "test"), "window") == lines(all_records(plain))

    one_mask = pipeline(ols()).run(df, test=(V[0], T[1]), horizon=1)
    in_test = [r for r in all_records(one_mask) if r["run_date"] >= T[0]]
    assert sorted(r["prediction"] for r in in_test) != sorted(r["prediction"] for r in all_records(plain))


def test_a_column_read_on_a_later_day_of_the_window_is_read_from_that_window(tmp_path):
    """A list used on some days reads a column without values on the first of them from a later day of its window."""
    frame = make_market_data(days=140, horizon=3, noise=5.0, seed=3)
    day = frame.index.tz_localize(None).normalize()
    rng = np.random.default_rng(1)
    late = (rng.random(len(frame)) > 0.5).astype(float)  # binary up to the end of February
    late[day >= pd.Timestamp("2023-03-01")] = (
        np.nan
    )  # nothing in the training window of the list's first test day, Apr 4
    on = day >= pd.Timestamp("2023-04-03")
    late[on] = rng.normal(0, 3, int(on.sum()))  # continuous from Apr 3 on: first read for Apr 5
    frame["late"] = late
    info = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "late": Availability.static()})

    def uses_late(ctx):
        names = ["load_d+{horizon}"]
        return names + ["late"] if ctx.issue_day.isoformat() in ("2023-03-27", "2023-04-04", "2023-04-05") else names

    make = lambda: ols(uses_late)  # noqa: E731
    assert_identity(make, frame, ("2023-03-25", "2023-03-27"), T, 1, info=info, level="raise")


def test_the_manifest_holds_a_digest_per_window(df, tmp_path):
    pipeline(ols()).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))
    manifest = json.loads((tmp_path / "m.manifest.json").read_text())
    masks = manifest["binary_masks"]
    assert len(masks) == 2 * 24
    assert {key.split("/")[0] for key in masks if "/" in key} == {"validation"}
    # Scaled in the test window (the flag isn't binary there), left alone in validation.
    assert masks["5-1"] != masks["validation/5-1"]


def test_a_function_keeps_a_digest_per_window_under_its_list(df, tmp_path):
    def same_list(ctx):
        return FIXED

    pipeline(ols(same_list), info=INFO, level="raise").run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))
    manifest = json.loads((tmp_path / "m.manifest.json").read_text())
    masks = manifest["predictor_lists"][0]["binary_masks"]
    assert masks["5-1"] != masks["validation/5-1"]
    assert "binary_masks" not in manifest


def test_a_plain_manifest_has_nothing_new(df, tmp_path):
    pipeline(ols()).run(df, test=T, horizon=1, save_dir=str(tmp_path))
    manifest = json.loads((tmp_path / "m.manifest.json").read_text())
    assert not {"windows", "binary_masks", "actuals"} & set(manifest)
    assert all("window" not in r for r in stored(tmp_path / "m.jsonl"))


def test_a_plain_ctx_manifest_keeps_its_flat_digests(df, tmp_path):
    def same_list(ctx):
        return FIXED

    pipeline(ols(same_list), info=INFO, level="raise").run(df, test=T, horizon=1, save_dir=str(tmp_path))
    masks = json.loads((tmp_path / "m.manifest.json").read_text())["predictor_lists"][0]["binary_masks"]
    assert set(masks) == {f"{period}-1" for period in range(24)}


def test_previous_year_records_are_the_plain_runs_records(tmp_path):
    frame = with_flag(make_market_data(start="2022-01-01", days=500, horizon=3, noise=5.0, seed=3), "2023-03-01")
    test = ("2023-03-25", "2023-03-28")
    windows = RunWindows.resolve("previous_year", test)
    make = lambda: ols(window=30)  # noqa: E731
    combined = pipeline(make()).run(
        frame, validation="previous_year", test=test, horizon=1, save_dir=str(tmp_path / "a")
    )
    records = stored(tmp_path / "a" / "m.jsonl")
    plain_test = all_records(pipeline(make()).run(frame, test=test, horizon=1))
    plain_validation = all_records(pipeline(make()).run(frame, test=windows.validation, horizon=1))
    assert lines(of_window(records, "test"), "window") == lines(plain_test)
    assert lines(of_window(records, "validation"), "window") == lines(plain_validation)
    assert combined.windows == windows


class MeanModel(BaseModel):
    """The cheapest model there is, for runs that are about days and tags and need a year or two of them."""

    def _fit_predict(self, train_x, train_y, test_x):
        return float(train_y.mean()), []


def test_previous_year_from_a_leap_day(tmp_path):
    frame = make_market_data(start="2022-11-01", days=540, horizon=2, noise=5.0, seed=3)
    test = ("2024-02-29", "2024-03-02")
    report = pipeline(MeanModel(PREDICTORS, training_window=14, name="m")).run(
        frame, validation="previous_year", test=test, horizon=1, save_dir=str(tmp_path)
    )
    assert report.windows.mode == "previous_year"
    assert report.windows.validation == ("2023-02-28", "2024-02-28")
    assert read_windows(tmp_path / "m.jsonl") == report.windows
    assert read_windows(tmp_path / "m.manifest.json") == report.windows
    records = stored(tmp_path / "m.jsonl")
    assert {r["run_date"] for r in of_window(records, "test")} == {"2024-02-29", "2024-03-01", "2024-03-02"}
    assert min(r["run_date"] for r in records) == "2023-02-28"
    assert {r["window"] for r in records if r["run_date"] >= "2024-02-29"} == {"test"}
    assert len({r["run_date"] for r in of_window(records, "validation")}) == 366


def test_a_later_segments_validation_is_the_test_before_it_and_stays_tagged_test():
    """It is what the plain test run gives, and the first test issue day cuts `report.validation`, not later ones."""
    frame = make_market_data(start="2022-12-01", days=850, horizon=2, noise=5.0, seed=3)
    test = ("2024-01-05", "2025-01-06")
    make = lambda: MeanModel(PREDICTORS, training_window=14, name="m")  # noqa: E731
    report = pipeline(make()).run(frame, validation="previous_year", test=test, horizon=2)
    assert report.windows.segments == (
        Segment(("2023-01-05", "2024-01-04"), ("2024-01-05", "2025-01-04")),
        Segment(("2024-01-05", "2025-01-04"), ("2025-01-05", "2025-01-06")),
    )
    second = report.windows.segments[1]
    serving = [r for r in all_records(report) if second.serves(r["run_date"], r["target_date"])]
    assert serving and {r["window"] for r in serving} == {"test"}
    assert max(r["target_date"] for r in serving) == "2025-01-04"
    plain = all_records(pipeline(make()).run(frame, test=test, horizon=2))
    assert lines(of_window(all_records(report), "test"), "window") == lines(plain)
    assert {r["day_in_test"] for r in of_window(all_records(report), "test")} == set(range(368))
    assert report.validation.predictions()["target_date"].max() == "2024-01-04"


def test_a_test_period_shorter_than_a_year_is_one_segment_through_the_run():
    frame = make_market_data(start="2022-01-01", days=500, horizon=2, noise=5.0, seed=3)
    test = ("2023-03-10", "2023-03-12")
    make = lambda: MeanModel(PREDICTORS, training_window=14, name="m")  # noqa: E731
    report = pipeline(make()).run(frame, validation="previous_year", test=test, horizon=1)
    assert report.windows.segments == (Segment(("2022-03-10", "2023-03-09"), test),)
    plain = all_records(pipeline(make()).run(frame, test=test, horizon=1))
    assert lines(of_window(all_records(report), "test"), "window") == lines(plain)


def test_previous_year_needs_the_year_in_the_data(df):
    with pytest.raises(ValueError, match="validation start"):
        pipeline(ols()).run(df, validation="previous_year", test=T, horizon=1)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def test_each_records_day_in_test_counts_from_its_own_window(df):
    records = all_records(pipeline(ols()).run(df, validation=V, test=T, horizon=1))
    for window, days in (("validation", pd.date_range(*V)), ("test", pd.date_range(*T))):
        expected = {day.strftime("%Y-%m-%d"): i for i, day in enumerate(days)}
        found = {r["run_date"]: r["day_in_test"] for r in of_window(records, window)}
        assert found == expected


def test_a_record_is_tagged_by_its_issue_day_whatever_its_target(df):
    """With horizon 3 a forecast issued on the last validation day targets a test day, and stays a validation record."""
    validation, test = ("2023-04-01", "2023-04-02"), ("2023-04-03", "2023-04-04")
    records = all_records(pipeline(ols()).run(df, validation=validation, test=test, horizon=3))
    late = [r for r in records if r["run_date"] == "2023-04-02" and r["horizon"] == 3]
    assert {r["target_date"] for r in late} == {"2023-04-05"} and {r["window"] for r in late} == {"validation"}


def test_days_between_the_windows_are_not_forecast(df):
    records = all_records(pipeline(ols()).run(df, validation=V, test=T, horizon=1))
    days = {r["run_date"] for r in records}
    assert days == set(pd.date_range(*V).strftime("%Y-%m-%d")) | set(pd.date_range(*T).strftime("%Y-%m-%d"))


def test_validation_alone_computes_nothing_of_the_test_period(df, tmp_path):
    report = pipeline(ols()).run(df, validation=V, horizon=2, save_dir=str(tmp_path))
    records = stored(tmp_path / "m.jsonl")
    assert {r["window"] for r in records} == {"validation"}
    assert max(r["run_date"] for r in records) == V[1]
    assert report.window == "validation"
    assert report.windows == RunWindows.resolve(V)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def test_test_start_and_test_end_are_an_alias_also_positionally(df):
    keyword = pipeline(ols()).run(df, test=T, horizon=1)
    aliased = pipeline(ols()).run(df, test_start=T[0], test_end=T[1], horizon=1)
    positional = pipeline(ols()).run(df, T[0], T[1], "price", 1)
    assert lines(all_records(keyword)) == lines(all_records(aliased)) == lines(all_records(positional))


def test_validation_works_with_the_alias_too(df):
    aliased = pipeline(ols()).run(df, T[0], T[1], "price", 1, validation=V)
    assert aliased.validation.window == "validation"
    assert aliased.windows == RunWindows.resolve(V, T)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(test=T, test_start=T[0]), "once"),
        (dict(test=T, test_end=T[1]), "once"),
        (dict(test=T, test_start=T[0], test_end=T[1]), "once"),
        (dict(test_start=T[0]), "test_start and test_end are required"),
        (dict(test_end=T[1]), "test_start and test_end are required"),
        (dict(), "No window"),
        (dict(validation=("2023-03-25", "2023-04-03"), test=T), "must end before"),
        (dict(validation=("2023-03-27", "2023-03-25"), test=T), "reversed"),
        (dict(validation=V, test=("2023-04-05", "2023-04-03")), "reversed"),
        (dict(test=("2023-04-05", "2023-04-03")), "reversed"),
        (dict(validation="last_year", test=T), "previous_year"),
        (dict(validation="previous_year"), "test=\\(start, end\\)"),
    ],
)
def test_what_is_refused_before_anything_runs(df, kwargs, message):
    with pytest.raises(ValueError, match=message):
        pipeline(ols()).run(df, horizon=1, **kwargs)


def test_forecast_only_has_no_validation(df):
    with pytest.raises(ValueError, match="forecast_only"):
        pipeline(ols()).run(df, validation=V, test=T, horizon=1, forecast_only=True)


def test_forecast_only_still_takes_test(df):
    report = pipeline(ols()).run(df, test=T, horizon=1, forecast_only=True)
    assert {r["run_date"] for r in all_records(report)} == set(pd.date_range(*T).strftime("%Y-%m-%d"))


def test_a_window_the_data_does_not_have_is_refused(df):
    with pytest.raises(ValueError, match="validation start"):
        pipeline(ols()).run(df, validation=("2022-01-01", "2022-01-02"), test=T, horizon=1)
    with pytest.raises(ValueError, match="test end"):
        pipeline(ols()).run(df, validation=V, test=("2023-04-03", "2025-01-01"), horizon=1)


# ---------------------------------------------------------------------------
# The report keeps to its window
# ---------------------------------------------------------------------------

MODELS = (("m", FIXED), ("small", SMALLER), ("wind", WIND_ONLY))
EVALUATORS = [MAEEvaluator(), RMSEEvaluator(), rMAEEvaluator("small"), DMEvaluator("small")]
# Long enough for the tests to say something, with a horizon that reaches over the boundary.
LONG_V = ("2023-03-01", "2023-04-01")
LONG_T = ("2023-04-02", "2023-04-30")


def models():
    return [OLSModel(predictors, training_window=30, name=name) for name, predictors in MODELS]


@pytest.fixture(scope="module")
def reports(df, tmp_path_factory):
    """The combined run, and the plain runs over each window, on saved results."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MAX_PROCESSES", "1")
        patch.setenv("THREADS_PER_PROCESS", "1")
        patch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
        base = tmp_path_factory.mktemp("reports")
        combined = pipeline(*models(), evaluators=EVALUATORS).run(
            df, validation=LONG_V, test=LONG_T, horizon=2, save_dir=str(base / "combined")
        )
        plain_test = pipeline(*models(), evaluators=EVALUATORS).run(
            df, test=LONG_T, horizon=2, save_dir=str(base / "plain_test")
        )
        plain_validation = pipeline(*models(), evaluators=EVALUATORS).run(
            df, test=LONG_V, horizon=2, save_dir=str(base / "plain_validation")
        )
    records = {name: stored(base / "plain_validation" / f"{name}.jsonl") for name, _ in MODELS}
    kept = {name: [r for r in rows if r["target_date"] < LONG_T[0]] for name, rows in records.items()}
    filtered = EvaluationReport(kept, EVALUATORS)
    return dict(combined=combined, plain_test=plain_test, plain_validation=plain_validation, filtered=filtered)


def details(report):
    return dict(report.iter_details())


TABLES = {
    "summary": lambda r: r.summary(),
    "by_hour": lambda r: r.by_hour(),
    "by_horizon": lambda r: r.by_horizon(),
    "by_hour_horizon": lambda r: r.by_hour_horizon(),
    "by_year": lambda r: r.by_year(),
    "by_year_horizon": lambda r: r.by_year_horizon(),
    "by_run_weekday_horizon": lambda r: r.by_run_weekday_horizon(),
    "by_target_weekday_horizon": lambda r: r.by_target_weekday_horizon(),
    "by_quarter": lambda r: r.by_quarter(),
    "by_day": lambda r: r.by_day(),
    "by_price_level": lambda r: r.by_price_level(),
    "tests": lambda r: r.tests(benchmark="small", reps=100),
    "mcs": lambda r: r.mcs(reps=100),
    "ranks": lambda r: r.ranks(reps=100),
    "compare": lambda r: r.compare("m", "small"),
    "leaderboard": lambda r: r.leaderboard(benchmark="small"),
    "aggregate": lambda r: r.aggregate().summary(),
    "aggregate_mcs": lambda r: r.aggregate(agg="mean").mcs(reps=100),
    "predictions": lambda r: r.predictions(),
    "compute_all": lambda r: pd.concat(r.compute_all(), names=["table"]),
}


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("table", list(TABLES))
def test_every_table_of_the_test_report_is_the_plain_runs(reports, table):
    pd.testing.assert_frame_equal(TABLES[table](reports["combined"]), TABLES[table](reports["plain_test"]))


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("table", list(TABLES))
def test_every_table_of_the_validation_report_is_the_plain_runs_without_later_targets(reports, table):
    pd.testing.assert_frame_equal(TABLES[table](reports["combined"].validation), TABLES[table](reports["filtered"]))


def test_details_keep_to_their_window(reports):
    for name, table in reports["combined"].iter_details():
        pd.testing.assert_frame_equal(table, details(reports["plain_test"])[name])
    for name, table in reports["combined"].validation.iter_details():
        pd.testing.assert_frame_equal(table, details(reports["filtered"])[name])


def test_no_record_of_one_window_is_reachable_from_the_other(reports):
    combined = reports["combined"]
    for report, (first, last), (other_first, other_last) in (
        (combined, LONG_T, LONG_V),
        (combined.validation, LONG_V, LONG_T),
    ):
        for table in [report.predictions(), *details(report).values(), *[f for f in report._model_frames().values()]]:
            run_dates = pd.to_datetime(table["run_date"])
            assert run_dates.min() >= pd.Timestamp(first) and run_dates.max() <= pd.Timestamp(last)
            assert not run_dates.between(pd.Timestamp(other_first), pd.Timestamp(other_last)).any()
    days = pd.to_datetime(combined.validation.predictions()["target_date"])
    assert days.max() < pd.Timestamp(LONG_T[0])  # a forecast for the first test issue day would be look-ahead


def test_evaluators_only_see_their_window(df):
    seen = {}

    class Spy(MAEEvaluator):
        name = "spy"

        def compute(self, frame, **kwargs):
            seen.setdefault(id(self), []).append(frame["run_date"])
            return super().compute(frame, **kwargs)

    spy = Spy()
    report = pipeline(ols(), evaluators=[spy]).run(df, validation=V, test=T, horizon=1)
    report.summary()
    assert pd.concat(seen[id(spy)]).min() >= pd.Timestamp(T[0])
    seen.clear()
    report.validation.summary()
    assert pd.concat(seen[id(spy)]).max() <= pd.Timestamp(V[1])


def test_exporters_write_what_the_plain_run_writes(reports, tmp_path):
    for report, expected, name in (
        (reports["combined"], reports["plain_test"], "test"),
        (reports["combined"].validation, reports["filtered"], "validation"),
    ):
        CsvExporter(tmp_path / f"{name}.csv", summary_path=tmp_path / f"{name}_summary.csv").export(report)
        CsvExporter(tmp_path / f"{name}_e.csv", summary_path=tmp_path / f"{name}_e_summary.csv").export(expected)
        assert (tmp_path / f"{name}.csv").read_text() == (tmp_path / f"{name}_e.csv").read_text()
        assert (tmp_path / f"{name}_summary.csv").read_text() == (tmp_path / f"{name}_e_summary.csv").read_text()
        ExcelExporter(tmp_path / f"{name}.xlsx", sheets=["summary", "details"]).export(report)
        ExcelExporter(tmp_path / f"{name}_e.xlsx", sheets=["summary", "details"]).export(expected)
        found = pd.read_excel(tmp_path / f"{name}.xlsx", sheet_name=None)
        wanted = pd.read_excel(tmp_path / f"{name}_e.xlsx", sheet_name=None)
        assert list(found) == list(wanted)
        for sheet in found:
            pd.testing.assert_frame_equal(found[sheet], wanted[sheet])


def test_the_terminal_exporter_says_which_window_it_shows(reports, capsys):
    exporter = TerminalExporter(show=["summary"])
    exporter.export(reports["combined"])
    assert "validation window" not in capsys.readouterr().out
    exporter.export(reports["combined"].validation)
    assert "The validation window" in capsys.readouterr().out


def test_the_pipeline_exports_the_test_window(df, tmp_path):
    built = pipeline(ols(), evaluators=[MAEEvaluator()])
    built.add_exporter(CsvExporter(tmp_path / "out.csv"))
    built.run(df, validation=V, test=T, horizon=1)
    days = pd.read_csv(tmp_path / "out.csv")["run_date"]
    assert days.min() == T[0] and days.max() == T[1]


def test_the_pipeline_exports_the_validation_window_when_it_is_all_there_is(df, tmp_path):
    built = pipeline(ols(), evaluators=[MAEEvaluator()])
    built.add_exporter(CsvExporter(tmp_path / "out.csv"))
    built.run(df, validation=V, horizon=1)
    days = pd.read_csv(tmp_path / "out.csv")["run_date"]
    assert days.min() == V[0] and days.max() == V[1]


def test_which_window_a_report_is_on(df):
    both = pipeline(ols(), evaluators=[MAEEvaluator()]).run(df, validation=V, test=T, horizon=1)
    assert both.window == "test"
    assert both.validation.window == "validation"
    assert both.validation.validation is both.validation

    alone = pipeline(ols(), evaluators=[MAEEvaluator()]).run(df, validation=V, horizon=1)
    assert alone.window == "validation"
    assert alone.validation is alone
    assert alone.summary().loc[0, "forecasts"] == 72


def test_a_plain_run_is_the_test_window_with_no_validation(df):
    report = pipeline(ols(), evaluators=[MAEEvaluator()]).run(df, test=T, horizon=1)
    assert report.window == "test"
    assert report.windows is None
    with pytest.raises(ValueError, match="no validation window"):
        report.validation


def test_a_report_of_plain_lists_has_no_validation_either():
    report = EvaluationReport({"m": []}, [])
    assert report.window == "test"
    with pytest.raises(ValueError, match="no validation window"):
        report.validation


def test_validation_alone_has_no_cut_since_it_knows_no_test(df):
    """Nothing is known about a later test window, so the report holds every forecast of the days it was told."""
    report = pipeline(ols(), evaluators=[MAEEvaluator()]).run(df, validation=("2023-04-01", "2023-04-02"), horizon=3)
    assert report.forecast_counts().loc[0, "produced"] == 2 * 3 * 24
    assert "target_in_test" not in report.forecast_counts()


def test_later_targets_are_counted_under_their_own_reason(df):
    validation, test = ("2023-03-30", "2023-04-02"), ("2023-04-03", "2023-04-04")
    report = pipeline(ols(), evaluators=[MAEEvaluator()]).run(df, validation=validation, test=test, horizon=3)
    counts = report.validation.forecast_counts().iloc[0]
    # Issued on 04-02: horizons 1, 2 and 3 reach the test. On 04-01: 2 and 3. On 03-31: 3.
    assert counts["target_in_test"] == (3 + 2 + 1) * 24
    assert counts["planned"] == 4 * 3 * 24
    assert counts["produced"] == (4 * 3 - 6) * 24
    assert counts["missing"] == 6 * 24
    assert report.validation.summary().loc[0, "forecasts"] == (4 * 3 - 6) * 24
    assert "target_in_test" not in report.forecast_counts()
    assert report.forecast_counts().iloc[0]["planned"] == 2 * 3 * 24


def test_skip_counts_are_split_by_window(df):
    frame = df.copy()
    wall = frame.index.tz_localize(None)
    frame.loc[wall.normalize() == pd.Timestamp("2023-03-29"), "load_d+1"] = np.nan  # a validation day
    frame = frame[~((wall.normalize() == pd.Timestamp("2023-04-04")) & (wall.hour == 5))]  # a row of a test day
    report = pipeline(ols(), evaluators=[MAEEvaluator()]).run(frame, validation=V, test=T, horizon=1)
    plain_test = pipeline(ols()).run(frame, test=T, horizon=1)
    plain_validation = pipeline(ols()).run(frame, test=V, horizon=1)

    assert report.refs["m"].skipped == plain_test.refs["m"].skipped == {"missing_test_row": 1}
    assert report.validation.refs["m"].skipped == plain_validation.refs["m"].skipped
    assert report.validation.refs["m"].skipped["missing_test_predictors"] == 24
    assert "missing_test_predictors" not in report.forecast_counts()
    assert "missing_test_row" not in report.validation.forecast_counts()
    assert list(report.forecast_counts()["missing"]) == [1]


def test_worker_skips_and_absent_rows_land_in_their_own_window_either_way_round(df):
    frame = df.copy()
    wall = frame.index.tz_localize(None)
    frame.loc[wall.normalize() == pd.Timestamp("2023-04-04"), "load_d+1"] = np.nan  # a test day, skipped in the worker
    frame = frame[~((wall.normalize() == pd.Timestamp("2023-03-29")) & (wall.hour == 5))]  # a row of a validation day
    report = pipeline(ols(), evaluators=[MAEEvaluator()]).run(frame, validation=V, test=T, horizon=1)
    assert report.refs["m"].skipped == {"missing_test_predictors": 24}
    assert report.validation.refs["m"].skipped == {"missing_test_row": 1}
    assert report.refs["m"].skipped == pipeline(ols()).run(frame, test=T, horizon=1).refs["m"].skipped
    assert report.validation.refs["m"].skipped == pipeline(ols()).run(frame, test=V, horizon=1).refs["m"].skipped


def test_a_whole_runs_ref_in_a_report_is_split_too(df):
    """`BaseModel.run` with windows gives the ref of the whole run: a report built from it must not mix the windows."""
    ref = ols().run(df, horizon=1, windows=RunWindows.resolve(V, T))
    report = EvaluationReport({"m": ref}, [MAEEvaluator()])
    assert report.window == "test"
    assert report.summary().loc[0, "forecasts"] == 3 * 24
    assert report.predictions()["run_date"].min() == T[0]
    assert report.validation.predictions()["run_date"].max() == V[1]


def test_plain_counts_keep_their_shape(df):
    report = pipeline(ols(), evaluators=[MAEEvaluator()]).run(df, test=T, horizon=1)
    assert list(report.forecast_counts().columns) == ["model", "planned", "produced", "missing"]


def test_a_saved_report_counts_its_own_window(df, tmp_path):
    report = pipeline(ols()).run(df, validation=V, test=T, horizon=2, save_dir=str(tmp_path))
    assert report.refs["m"].produced == report.validation.refs["m"].produced == 3 * 2 * 24
    assert report.refs["m"].count == report.validation.refs["m"].count == 3 * 2 * 24
    assert "target_in_test" not in report.validation.forecast_counts()  # the horizon doesn't reach over the gap


# -- join -------------------------------------------------------------------


def test_join_of_reports_on_different_windows_raises(df):
    both = pipeline(ols(), evaluators=[MAEEvaluator()]).run(df, validation=V, test=T, horizon=1)
    other = pipeline(ols(name="other"), evaluators=[MAEEvaluator()]).run(df, validation=V, test=T, horizon=1)
    with pytest.raises(ValueError, match="different windows"):
        both.join(other.validation)
    with pytest.raises(ValueError, match="different windows"):
        both.validation.join(other)
    moved = pipeline(ols(name="moved"), evaluators=[MAEEvaluator()]).run(
        df, test=("2023-04-04", "2023-04-05"), horizon=1
    )
    with pytest.raises(ValueError, match="different test periods"):
        both.join(moved)
    moved_validation = pipeline(ols(name="v"), evaluators=[MAEEvaluator()]).run(
        df, validation=("2023-03-24", V[1]), horizon=1
    )
    with pytest.raises(ValueError, match="different validation periods"):
        both.validation.join(moved_validation)


def test_join_of_aggregates_of_other_periods_raises(df):
    a = pipeline(ols(name="a")).run(df, validation=V, test=T, horizon=1)
    b = pipeline(ols(PREDICTORS, name="b")).run(df, validation=V, test=("2023-04-10", "2023-04-12"), horizon=1)
    with pytest.raises(ValueError, match="different test periods"):
        a.aggregate().join(b.aggregate())
    with pytest.raises(ValueError, match="different validation periods"):  # the same days, cut at another test start
        a.validation.aggregate().join(b.validation.aggregate())
    c = pipeline(ols(PREDICTORS, name="c")).run(df, validation=V, test=T, horizon=1)
    assert set(a.aggregate().join(c.aggregate()).summary()["model"]) == {"a", "c"}
    assert set(a.validation.aggregate().join(c.validation.aggregate()).summary()["model"]) == {"a", "c"}


def test_join_of_validation_reports_with_other_cuts_raises(df):
    validation = ("2023-03-25", "2023-04-01")
    a = pipeline(ols(name="a")).run(df, validation=validation, test=("2023-04-02", "2023-04-05"), horizon=3)
    b = pipeline(ols(PREDICTORS, name="b")).run(df, validation=validation, test=("2023-04-04", "2023-04-05"), horizon=3)
    with pytest.raises(ValueError, match="different validation periods"):
        a.validation.join(b.validation)


def test_join_keeps_the_windows_and_joins_the_validation_reports(df):
    both = pipeline(ols(), evaluators=[MAEEvaluator()]).run(df, validation=V, test=T, horizon=1)
    other = pipeline(ols(SMALLER, name="small"), evaluators=[MAEEvaluator()]).run(df, validation=V, test=T, horizon=1)
    together = both.join(other)
    assert set(together.summary()["model"]) == {"m", "small"}
    assert together.window == "test"
    assert together.validation.window == "validation"
    assert set(together.validation.summary()["model"]) == {"m", "small"}
    assert (together.validation.summary()["forecasts"] == 3 * 24).all()
    assert (together.summary()["forecasts"] == 3 * 24).all()


def test_join_with_a_plain_report_on_the_same_days_keeps_working(df):
    both = pipeline(ols(), evaluators=[MAEEvaluator()]).run(df, validation=V, test=T, horizon=1)
    plain = pipeline(ols(SMALLER, name="small"), evaluators=[MAEEvaluator()]).run(df, test=T, horizon=1)
    together = both.join(plain)
    assert (together.summary()["forecasts"] == 3 * 24).all()
    with pytest.raises(ValueError, match="no validation window"):
        together.validation


def test_two_plain_reports_on_different_days_still_join(df):
    one = pipeline(ols(), evaluators=[MAEEvaluator()]).run(df, test=T, horizon=1)
    two = pipeline(ols(name="two"), evaluators=[MAEEvaluator()]).run(df, test=("2023-04-04", "2023-04-06"), horizon=1)
    assert set(one.join(two).summary()["model"]) == {"m", "two"}


# -- aggregate --------------------------------------------------------------


def test_aggregate_stays_on_its_window(df):
    both = pipeline(*models(), evaluators=[MAEEvaluator()]).run(df, validation=V, test=T, horizon=1)
    daily = both.aggregate()
    assert daily.window == "test"
    assert set(daily.predictions()["run_date"]) <= set(pd.date_range(*T).strftime("%Y-%m-%d"))
    daily_validation = both.validation.aggregate()
    assert daily_validation.window == "validation"
    assert daily_validation.validation is daily_validation
    assert set(daily_validation.predictions()["run_date"]) <= set(pd.date_range(*V).strftime("%Y-%m-%d"))
    with pytest.raises(ValueError, match="no validation window"):
        daily.validation


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def manifest_of(path):
    manifest = json.loads(manifest_path(path).read_text())
    return {k: v for k, v in manifest.items() if k not in ("created_at", "updated_at", "git_sha")}


def test_validation_first_and_test_later_equals_a_fresh_combined_run(df, tmp_path):
    resumed, fresh = tmp_path / "resumed", tmp_path / "fresh"
    pipeline(ols()).run(df, validation=V, horizon=2, save_dir=str(resumed))
    before = (resumed / "m.jsonl").read_text()
    calls = []
    fit_predict = OLSModel._fit_predict

    def counted(self, *args):
        calls.append(1)
        return fit_predict(self, *args)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(OLSModel, "_fit_predict", counted)
        pipeline(ols()).run(df, validation=V, test=T, horizon=2, save_dir=str(resumed))
    assert len(calls) == 3 * 2 * 24  # only the test days
    assert (resumed / "m.jsonl").read_text().startswith(before)

    pipeline(ols()).run(df, validation=V, test=T, horizon=2, save_dir=str(fresh))
    assert lines(stored(resumed / "m.jsonl")) == lines(stored(fresh / "m.jsonl"))


def test_the_manifest_after_validation_then_test_is_the_fresh_ones(df, tmp_path):
    resumed, fresh = tmp_path / "resumed", tmp_path / "fresh"
    pipeline(ols()).run(df, validation=V, horizon=2, save_dir=str(resumed))
    report = pipeline(ols()).run(df, validation=V, test=T, horizon=2, save_dir=str(resumed))
    fresh_report = pipeline(ols()).run(df, validation=V, test=T, horizon=2, save_dir=str(fresh))
    assert manifest_of(resumed / "m.jsonl") == manifest_of(fresh / "m.jsonl")
    assert lines(stored(resumed / "m.jsonl")) == lines(stored(fresh / "m.jsonl"))
    assert report.refs["m"].produced == fresh_report.refs["m"].produced
    assert report.validation.refs["m"].produced == fresh_report.validation.refs["m"].produced
    assert report.validation.refs["m"].skipped == fresh_report.validation.refs["m"].skipped


def test_a_function_with_lists_only_the_test_days_use_resumes_too(df, tmp_path):
    """The stored forecasts' data is compared on what they read, so a column only a pending list uses is no mismatch."""

    def more_in_the_test(ctx):
        return PREDICTORS + (["flag"] if ctx.issue_day >= pd.Timestamp(T[0]).date() else [])

    make = lambda: ols(more_in_the_test)  # noqa: E731
    resumed, fresh = tmp_path / "resumed", tmp_path / "fresh"
    pipeline(make(), info=INFO, level="raise").run(df, validation=V, horizon=1, save_dir=str(resumed))
    assert "flag" not in json.loads((resumed / "m.manifest.json").read_text())["predictors"]
    pipeline(make(), info=INFO, level="raise").run(df, validation=V, test=T, horizon=1, save_dir=str(resumed))
    pipeline(make(), info=INFO, level="raise").run(df, validation=V, test=T, horizon=1, save_dir=str(fresh))
    assert lines(stored(resumed / "m.jsonl")) == lines(stored(fresh / "m.jsonl"))
    assert manifest_of(resumed / "m.jsonl") == manifest_of(fresh / "m.jsonl")


def test_validation_and_test_then_validation_alone_keeps_what_is_on_file(df, tmp_path):
    pipeline(ols()).run(df, validation=V, test=T, horizon=2, save_dir=str(tmp_path))
    before, manifest = lines(stored(tmp_path / "m.jsonl")), manifest_of(tmp_path / "m.jsonl")
    report = pipeline(ols()).run(df, validation=V, horizon=2, save_dir=str(tmp_path))
    assert lines(stored(tmp_path / "m.jsonl")) == before
    assert manifest_of(tmp_path / "m.jsonl") == manifest
    assert report.window == "validation"
    assert report.predictions()["run_date"].max() == V[1]


def test_validation_alone_after_a_run_with_test_cuts_at_the_test_on_file(df, tmp_path):
    """The file knows the test window, so the same validation report comes out, whichever way it was asked for."""
    validation, test = ("2023-04-01", "2023-04-02"), ("2023-04-03", "2023-04-04")
    full = pipeline(ols()).run(df, validation=validation, test=test, horizon=3, save_dir=str(tmp_path))
    alone = pipeline(ols()).run(df, validation=validation, horizon=3, save_dir=str(tmp_path))
    pd.testing.assert_frame_equal(alone.predictions(), full.validation.predictions())
    assert alone.forecast_counts().equals(full.validation.forecast_counts())


def test_a_plain_run_into_results_with_windows_resumes_and_keeps_them(df, tmp_path):
    pipeline(ols()).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))
    before, windows = lines(stored(tmp_path / "m.jsonl")), read_windows(tmp_path / "m.jsonl")
    report = pipeline(ols()).run(df, test=T, horizon=1, save_dir=str(tmp_path))
    assert lines(stored(tmp_path / "m.jsonl")) == before
    assert read_windows(tmp_path / "m.jsonl") == windows
    assert report.predictions()["run_date"].min() == T[0]
    with pytest.raises(ValueError, match="no validation window"):
        report.validation


def test_a_plain_run_then_validation_adds_the_validation_days(df, tmp_path):
    """Stored test records have no tag, which means test: only the validation days are computed."""
    pipeline(ols()).run(df, test=T, horizon=1, save_dir=str(tmp_path / "a"))
    plain_lines = lines(stored(tmp_path / "a" / "m.jsonl"))
    report = pipeline(ols()).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path / "a"))
    records = stored(tmp_path / "a" / "m.jsonl")
    assert lines([r for r in records if "window" not in r]) == plain_lines
    assert {r["window"] for r in records if "window" in r} == {"validation"}
    pipeline(ols()).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path / "b"))
    assert lines(records, "window") == lines(stored(tmp_path / "b" / "m.jsonl"), "window")
    assert report.summary().loc[0, "forecasts"] == 3 * 24
    assert report.validation.summary().loc[0, "forecasts"] == 3 * 24


@pytest.mark.parametrize("predictors", [FIXED, by_day_and_value], ids=["fixed list", "ctx function"])
def test_interrupted_then_resumed_equals_fresh(df, tmp_path, predictors):
    info, level = (INFO, "raise") if callable(predictors) else (None, None)
    make = lambda: pipeline(ols(predictors), info=info, level=level)  # noqa: E731
    make().run(df, validation=V, test=T, horizon=2, save_dir=str(tmp_path / "fresh"))
    make().run(df, validation=V, test=T, horizon=2, save_dir=str(tmp_path / "cut"))
    path = tmp_path / "cut" / "m.jsonl"
    kept = path.read_text().splitlines()
    path.write_text("\n".join(kept[::3]) + "\n")
    report = make().run(df, validation=V, test=T, horizon=2, save_dir=str(tmp_path / "cut"))
    assert lines(stored(path)) == lines(stored(tmp_path / "fresh" / "m.jsonl"))
    assert manifest_of(path) == manifest_of(tmp_path / "fresh" / "m.jsonl")
    assert report.refs["m"].produced == 3 * 2 * 24


# A model that stops the whole pool in a worker after some fits, the way a killed run does: what was saved stays.
class StopsAfter(OLSModel):
    stop_after = None
    _fits = 0

    def _fit_predict(self, train_x, train_y, test_x):
        limit = StopsAfter.stop_after
        if limit is not None:
            type(self)._fits += 1
            if type(self)._fits > limit:
                raise RuntimeError("stopped on purpose")
        return super()._fit_predict(train_x, train_y, test_x)


@pytest.mark.parametrize("predictors", [FIXED, by_day_and_value], ids=["fixed list", "ctx function"])
def test_a_run_stopped_in_the_real_pool_then_resumed_equals_fresh(df, tmp_path, monkeypatch, predictors):
    monkeypatch.undo()  # the real pool, and a resume with another number of processes
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    info, level = (INFO, "raise") if callable(predictors) else (None, None)
    make = lambda: pipeline(  # noqa: E731
        StopsAfter(predictors, training_window=30, name="m"), info=info, level=level
    )
    monkeypatch.setenv("MAX_PROCESSES", "1")
    make().run(df, validation=V, test=T, horizon=2, save_dir=str(tmp_path / "fresh"))

    monkeypatch.setattr(StopsAfter, "stop_after", 70)  # inherited by the forked workers
    monkeypatch.setenv("MAX_PROCESSES", "2")
    with pytest.raises(RuntimeError, match="forecasting failed"):
        make().run(df, validation=V, test=T, horizon=2, save_dir=str(tmp_path / "cut"))
    path = tmp_path / "cut" / "m.jsonl"
    assert path.exists() and 0 < len(stored(path)) < 6 * 2 * 24

    monkeypatch.setattr(StopsAfter, "stop_after", None)
    monkeypatch.setenv("MAX_PROCESSES", "3")
    report = make().run(df, validation=V, test=T, horizon=2, save_dir=str(tmp_path / "cut"))
    assert lines(stored(path)) == lines(stored(tmp_path / "fresh" / "m.jsonl"))
    assert report.refs["m"].produced == report.refs["m"].count == 3 * 2 * 24
    assert manifest_of(path) == manifest_of(tmp_path / "fresh" / "m.jsonl")


def test_a_stored_key_that_would_change_window_is_refused(df, tmp_path):
    pipeline(ols()).run(df, validation=V, save_dir=str(tmp_path / "v"), horizon=1)
    with pytest.raises(ManifestMismatchError, match="change window"):
        pipeline(ols()).run(df, test=V, save_dir=str(tmp_path / "v"), horizon=1)

    pipeline(ols()).run(df, test=V, save_dir=str(tmp_path / "t"), horizon=1)
    with pytest.raises(ManifestMismatchError, match="change window"):
        pipeline(ols()).run(df, validation=V, save_dir=str(tmp_path / "t"), horizon=1)

    pipeline(ols()).run(df, validation=V, test=T, save_dir=str(tmp_path / "both"), horizon=1)
    later = ("2023-04-06", "2023-04-07")
    with pytest.raises(ManifestMismatchError, match="change window"):
        pipeline(ols()).run(df, validation=T, test=later, save_dir=str(tmp_path / "both"), horizon=1)


def test_a_refused_resume_touches_nothing(df, tmp_path):
    pipeline(ols()).run(df, validation=V, save_dir=str(tmp_path), horizon=1)
    results, manifest = (tmp_path / "m.jsonl").read_text(), (tmp_path / "m.manifest.json").read_text()
    with pytest.raises(ManifestMismatchError):
        pipeline(ols()).run(df, test=V, save_dir=str(tmp_path), horizon=1)
    assert (tmp_path / "m.jsonl").read_text() == results
    assert (tmp_path / "m.manifest.json").read_text() == manifest


def always_fixed(ctx):
    return FIXED


@pytest.mark.parametrize("predictors", [FIXED, always_fixed], ids=["fixed list", "ctx function"])
def test_a_stored_key_whose_mask_would_change_is_refused(tmp_path, predictors):
    """The window's first day moves past the day the flag stops being binary, and a stored forecast's mask with it."""
    info, level = (INFO, "raise") if callable(predictors) else (None, None)
    make = lambda: pipeline(ols(predictors), info=info, level=level)  # noqa: E731
    frame = with_flag(make_market_data(horizon=3, noise=5.0, seed=3), "2023-03-27")
    validation, test = ("2023-03-10", "2023-03-15"), ("2023-03-20", "2023-03-30")
    make().run(frame, validation=validation, test=test, horizon=1, save_dir=str(tmp_path))
    with pytest.raises(ManifestMismatchError, match="binary columns"):
        make().run(frame, validation=validation, test=("2023-03-30", "2023-03-30"), horizon=1, save_dir=str(tmp_path))


@pytest.mark.parametrize("predictors", [FIXED, always_fixed], ids=["fixed list", "ctx function"])
def test_a_stored_validation_key_whose_mask_would_change_is_refused(tmp_path, predictors):
    info, level = (INFO, "raise") if callable(predictors) else (None, None)
    make = lambda: pipeline(ols(predictors), info=info, level=level)  # noqa: E731
    frame = with_flag(make_market_data(horizon=3, noise=5.0, seed=3), "2023-03-27")
    make().run(frame, validation=("2023-03-20", "2023-03-30"), horizon=1, save_dir=str(tmp_path))
    with pytest.raises(ManifestMismatchError, match="binary columns"):
        make().run(frame, validation=("2023-03-30", "2023-03-30"), horizon=1, save_dir=str(tmp_path))


def test_the_same_mask_after_the_first_day_moved_resumes_and_reads_as_fresh(df, tmp_path):
    """Nothing about a stored forecast changes if the column reads the same from the new first day. Its `day_in_test` in the
    file keeps the old count, but a report reads it from the window as it is now."""
    pipeline(ols(PREDICTORS)).run(df, validation=V, save_dir=str(tmp_path / "a"), horizon=1)
    before = lines(stored(tmp_path / "a" / "m.jsonl"))
    later = ("2023-03-29", V[1])
    moved = pipeline(ols(PREDICTORS)).run(df, validation=later, save_dir=str(tmp_path / "a"), horizon=1)
    assert lines(stored(tmp_path / "a" / "m.jsonl")) == before
    fresh = pipeline(ols(PREDICTORS)).run(df, validation=later, save_dir=str(tmp_path / "b"), horizon=1)
    pd.testing.assert_frame_equal(dict(moved.iter_details())["m"], dict(fresh.iter_details())["m"])
    pd.testing.assert_frame_equal(moved.predictions(), fresh.predictions())


@pytest.mark.parametrize("predictors", [["load_d+{horizon}", "flag"], lambda ctx: ["load_d+{horizon}", "flag"]], ids=["fixed list", "ctx function"])  # fmt: skip
def test_a_file_keeps_one_mask_per_key_whatever_days_its_records_are_on(tmp_path, predictors):
    """A digest says nothing about which records used it. A run on other days with another mask would overwrite it, and the
    older records would pass a later check for what they are not."""
    info, level = (INFO, "raise") if callable(predictors) else (None, None)
    make = lambda: pipeline(ols(predictors), info=info, level=level)  # noqa: E731
    frame = with_flag(make_market_data(horizon=3, noise=5.0, seed=3), "2023-04-03")
    make().run(frame, validation=("2023-04-04", "2023-04-08"), horizon=1, save_dir=str(tmp_path))
    with pytest.raises(ManifestMismatchError, match="binary columns"):
        make().run(frame, validation=("2023-04-20", "2023-04-22"), horizon=1, save_dir=str(tmp_path))
    assert max(r["run_date"] for r in stored(tmp_path / "m.jsonl")) == "2023-04-08"


def test_plain_results_whose_last_actuals_were_missing_are_refused(df, tmp_path):
    """A plain manifest holds no hash of them, so the records are compared with the data."""
    short = df[df.index.tz_localize(None).normalize() <= pd.Timestamp(T[1])]
    pipeline(ols()).run(short, test=T, horizon=1, save_dir=str(tmp_path))
    with pytest.raises(ManifestMismatchError, match="another `actual`"):
        pipeline(ols()).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))


def test_plain_results_whose_actuals_were_revised_are_refused(df, tmp_path):
    pipeline(ols()).run(df, test=T, horizon=1, save_dir=str(tmp_path))
    revised = df.copy()
    revised.loc[revised.index.tz_localize(None).normalize() == pd.Timestamp("2023-04-06"), "price"] += 50.0
    with pytest.raises(ManifestMismatchError, match="another `actual`"):
        pipeline(ols()).run(revised, validation=V, test=T, horizon=1, save_dir=str(tmp_path))


def test_a_plain_run_on_another_period_may_not_replace_the_test_window_on_file(df, tmp_path):
    pipeline(ols()).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))
    before = lines(stored(tmp_path / "m.jsonl")), read_windows(tmp_path / "m.jsonl")
    for kwargs in (
        dict(test=("2023-04-20", "2023-04-20")),
        dict(test=("2023-04-03", "2023-04-04")),
        dict(test_start="2023-04-20", test_end="2023-04-20", forecast_only=True),
    ):
        with pytest.raises(ManifestMismatchError, match="would change them"):
            pipeline(ols()).run(df, horizon=1, save_dir=str(tmp_path), **kwargs)
    assert (lines(stored(tmp_path / "m.jsonl")), read_windows(tmp_path / "m.jsonl")) == before


def test_a_plain_run_may_add_the_test_window_to_validation_on_file(df, tmp_path):
    pipeline(ols()).run(df, validation=V, horizon=1, save_dir=str(tmp_path / "a"))
    report = pipeline(ols()).run(df, test=T, horizon=1, save_dir=str(tmp_path / "a"))
    assert read_windows(tmp_path / "a" / "m.jsonl") == RunWindows.resolve(V, T)
    pipeline(ols()).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path / "b"))
    assert lines(stored(tmp_path / "a" / "m.jsonl")) == lines(stored(tmp_path / "b" / "m.jsonl"))
    assert report.window == "test"


def test_forecast_only_may_not_add_a_test_window_to_validation_on_file(df, tmp_path):
    pipeline(ols()).run(df, validation=V, horizon=1, save_dir=str(tmp_path))
    with pytest.raises(ManifestMismatchError, match="would change them"):
        pipeline(ols()).run(df, test=T, horizon=1, save_dir=str(tmp_path), forecast_only=True)


def test_plain_results_of_fixed_lists_whose_first_day_moved_are_refused(df, tmp_path):
    """A plain manifest of fixed lists has no digests, only the day their masks were read on."""
    pipeline(ols()).run(df, test=("2023-04-01", T[1]), save_dir=str(tmp_path), horizon=1)
    with pytest.raises(ManifestMismatchError, match="binary columns"):
        pipeline(ols()).run(df, validation=V, test=T, save_dir=str(tmp_path), horizon=1)


def test_other_data_under_the_stored_forecasts_is_refused(df, tmp_path):
    pipeline(ols()).run(df, validation=V, save_dir=str(tmp_path), horizon=1)
    changed = df.copy()
    changed.loc[changed.index.tz_localize(None).normalize() == pd.Timestamp("2023-03-10"), "load_d+1"] += 1.0
    with pytest.raises(ManifestMismatchError, match="data_columns_hash"):
        pipeline(ols()).run(changed, validation=V, test=T, save_dir=str(tmp_path), horizon=1)


def test_other_data_after_the_stored_forecasts_is_not_refused(df, tmp_path):
    pipeline(ols()).run(df, validation=V, save_dir=str(tmp_path), horizon=1)
    changed = df.copy()
    changed.loc[changed.index.tz_localize(None).normalize() == pd.Timestamp(T[0]), "load_d+1"] += 1.0
    pipeline(ols()).run(changed, validation=V, test=T, save_dir=str(tmp_path), horizon=1)
    assert {r["window"] for r in stored(tmp_path / "m.jsonl")} == {"validation", "test"}
    # From here on the test records hang on it too.
    with pytest.raises(ManifestMismatchError, match="data_columns_hash"):
        pipeline(ols()).run(df, validation=V, test=T, save_dir=str(tmp_path), horizon=1)


def test_data_that_grew_under_the_stored_actuals_is_refused(df, tmp_path):
    """The last days' `actual` read rows after the last issue day: with fewer rows then, a fresh run would differ."""
    short = df[df.index.tz_localize(None) < pd.Timestamp("2023-03-31")]
    pipeline(ols(PREDICTORS)).run(short, validation=V, save_dir=str(tmp_path), horizon=2)
    assert any(np.isnan(r["actual"]) for r in stored(tmp_path / "m.jsonl"))
    with pytest.raises(ManifestMismatchError, match="actuals"):
        pipeline(ols(PREDICTORS)).run(df, validation=V, test=T, save_dir=str(tmp_path), horizon=2)


def test_data_cut_just_after_the_stored_actuals_resumes(df, tmp_path):
    """Everything the stored records read is there, and the same: a frame kept short of the test period is no problem."""
    short = df[df.index.tz_localize(None) < pd.Timestamp("2023-04-02")]
    pipeline(ols(PREDICTORS)).run(short, validation=V, save_dir=str(tmp_path / "a"), horizon=2)
    pipeline(ols(PREDICTORS)).run(df, validation=V, test=T, save_dir=str(tmp_path / "a"), horizon=2)
    pipeline(ols(PREDICTORS)).run(df, validation=V, test=T, save_dir=str(tmp_path / "b"), horizon=2)
    assert lines(stored(tmp_path / "a" / "m.jsonl")) == lines(stored(tmp_path / "b" / "m.jsonl"))


def test_a_changed_configuration_is_refused_before_the_records_are_looked_at(df, tmp_path):
    pipeline(ols()).run(df, validation=V, save_dir=str(tmp_path), horizon=1)
    with pytest.raises(ManifestMismatchError, match="predictor_layout_hash"):
        pipeline(ols(PREDICTORS)).run(df, validation=V, test=T, save_dir=str(tmp_path), horizon=1)
    with pytest.raises(ManifestMismatchError, match="training_window"):
        pipeline(ols(window=20)).run(df, validation=V, test=T, save_dir=str(tmp_path), horizon=1)


def test_allow_manifest_mismatch_goes_on_and_the_records_keep_their_tag(df, tmp_path):
    pipeline(ols()).run(df, validation=V, save_dir=str(tmp_path), horizon=1)
    before = lines(stored(tmp_path / "m.jsonl"))
    with pytest.warns(UserWarning, match="no forecasts in the test window"):
        pipeline(ols()).run(df, test=V, save_dir=str(tmp_path), horizon=1, allow_manifest_mismatch=True)
    assert lines(stored(tmp_path / "m.jsonl")) == before


# ---------------------------------------------------------------------------
# Planning and checks cover both windows, once
# ---------------------------------------------------------------------------

ASKED = []


def remembers(ctx):
    ASKED.append((ctx.issue_day, ctx.period, ctx.horizon))
    return FIXED


def test_the_function_is_asked_once_per_forecast_of_both_windows_and_never_in_between(df):
    ASKED.clear()
    pipeline(ols(remembers), info=INFO, level="raise").run(df, validation=V, test=T, horizon=2)
    assert len(ASKED) == len(set(ASKED)) == 6 * 2 * 24
    days = {day.isoformat() for day, _, _ in ASKED}
    assert days == set(pd.date_range(*V).strftime("%Y-%m-%d")) | set(pd.date_range(*T).strftime("%Y-%m-%d"))


def test_the_function_does_not_see_the_window(df):
    seen = set()

    def look(ctx):
        seen.update(name for name in dir(ctx) if not name.startswith("_"))
        return FIXED

    pipeline(ols(look), info=INFO, level="raise").run(df, validation=V, test=T, horizon=1)
    assert not {name for name in seen if "window" in name or "test" in name or "validation" in name}


# The value's rule puts it after the issue time in winter only: 12:00 Brussels is 11:00 UTC in winter, 10:00 UTC in summer.
SEASONAL = {**RULES, "seasonal": Availability.published("12:00", 0, "Europe/Brussels")}
SEASONAL_INFO = InformationSet("10:30", "UTC", rules=SEASONAL)
SUMMER_V, SUMMER_T = ("2023-07-03", "2023-07-04"), ("2024-07-01", "2024-07-02")
WINTER_V = ("2023-12-04", "2023-12-05")


@pytest.fixture(scope="module")
def long_frame():
    frame = make_market_data(start="2023-05-01", days=460, horizon=2, noise=5.0, seed=3)
    frame["seasonal"] = np.random.default_rng(1).normal(0, 1, len(frame))
    return frame


def seasonal_model():
    return OLSModel(PREDICTORS + ["seasonal"], training_window=14, name="m")


def test_a_rule_broken_only_between_the_windows_is_no_violation(long_frame):
    report = pipeline(seasonal_model(), info=SEASONAL_INFO, level="raise").run(
        long_frame, validation=SUMMER_V, test=SUMMER_T, horizon=1
    )
    assert report.summary().loc[0, "forecasts"] == 2 * 24
    with pytest.raises(InformationSetError, match="seasonal"):  # the plain run over the whole span would refuse it
        pipeline(seasonal_model(), info=SEASONAL_INFO, level="raise").run(
            long_frame, test=(SUMMER_V[0], SUMMER_T[1]), horizon=1
        )


def test_a_rule_broken_in_a_window_is_refused_before_any_fit(long_frame, monkeypatch):
    monkeypatch.setattr(OLSModel, "_fit_predict", lambda *a: pytest.fail("a fit ran"))
    with pytest.raises(InformationSetError, match="seasonal"):
        pipeline(seasonal_model(), info=SEASONAL_INFO, level="raise").run(
            long_frame, validation=WINTER_V, test=SUMMER_T, horizon=1
        )


def test_each_window_lists_what_it_uses(long_frame):
    with pytest.warns(UserWarning, match="seasonal"):
        report = pipeline(seasonal_model(), info=SEASONAL_INFO, level="warn").run(
            long_frame, validation=WINTER_V, test=SUMMER_T, horizon=1
        )
    assert report.availability().empty
    listed = report.validation.availability()
    assert set(listed["predictor"]) == {"seasonal"} and (listed["status"] == "late").all()


def test_a_list_a_function_uses_only_in_one_window_is_listed_only_there(long_frame):
    def seasonal_in_winter(ctx):
        return PREDICTORS + (["seasonal"] if ctx.issue_day.month == 12 else [])

    with pytest.warns(UserWarning, match="seasonal"):
        report = pipeline(ols(seasonal_in_winter, window=14), info=SEASONAL_INFO, level="warn").run(
            long_frame, validation=WINTER_V, test=SUMMER_T, horizon=1
        )
    assert report.availability().empty
    assert set(report.validation.availability()["predictor"]) == {"seasonal"}
    assert {len(entry["predictors"]) for entry in report.refs["m"].predictor_lists} == {4}
    assert {len(entry["predictors"]) for entry in report.validation.refs["m"].predictor_lists} == {5}


def test_a_value_a_function_reads_late_is_listed_for_its_window_only(long_frame):
    def reads_it_in_winter(ctx):
        if ctx.issue_day.month == 12:
            ctx.value("seasonal")
        return PREDICTORS

    with pytest.warns(UserWarning, match="seasonal"):
        report = pipeline(ols(reads_it_in_winter, window=14), info=SEASONAL_INFO, level="warn").run(
            long_frame, validation=WINTER_V, test=SUMMER_T, horizon=1
        )
    assert report.availability().empty
    listed = report.validation.availability()
    assert list(listed["predictor"].unique()) == ["seasonal"] and set(listed["role"]) == {"ctx.value"}


def test_the_manifest_lists_per_window_and_a_merged_report_shows_it(long_frame, tmp_path):
    make = lambda: pipeline(seasonal_model(), info=SEASONAL_INFO, level="warn")  # noqa: E731
    with pytest.warns(UserWarning, match="seasonal"):
        make().run(long_frame, validation=WINTER_V, test=SUMMER_T, horizon=1, save_dir=str(tmp_path))
    manifest = json.loads((tmp_path / "m.manifest.json").read_text())
    assert manifest["availability"]["windows"]["test"] == []
    assert {row["predictor"] for row in manifest["availability"]["windows"]["validation"]} == {"seasonal"}
    assert {row["predictor"] for row in manifest["availability"]["listing"]} == {"seasonal"}
    with pytest.warns(UserWarning, match="seasonal"):
        merged = make().report(tmp_path, horizon=1)
    assert merged.availability().empty
    assert set(merged.validation.availability()["predictor"]) == {"seasonal"}


NO_RULE_INFO = InformationSet("10:30", "UTC", rules=SEASONAL, unknown="ignore")


def test_what_has_no_rule_is_listed_only_for_the_window_that_uses_it(long_frame):
    def in_winter(ctx):
        if ctx.issue_day.month == 12:
            ctx.value("read_no_rule")
            return PREDICTORS + ["listed_no_rule"]
        return PREDICTORS

    frame = long_frame.assign(listed_no_rule=long_frame["seasonal"], read_no_rule=long_frame["seasonal"])
    with pytest.warns(UserWarning, match="no rule"):
        report = pipeline(ols(in_winter, window=14), info=NO_RULE_INFO, level="warn").run(
            frame, validation=WINTER_V, test=SUMMER_T, horizon=1
        )
    assert report.availability().empty
    listed = report.validation.availability()
    assert set(listed["status"]) == {"no rule"}
    assert set(listed["predictor"]) == {"listed_no_rule", "read_no_rule"}


def test_a_late_training_target_is_listed_for_its_window_only(long_frame):
    rules = {**SEASONAL, "price": Availability.published("12:00", -1, "Europe/Brussels")}
    info = InformationSet("10:30", "UTC", rules=rules)
    with pytest.warns(UserWarning, match="price"):
        report = pipeline(OLSModel(PREDICTORS, training_window=14, name="m"), info=info, level="warn").run(
            long_frame, validation=WINTER_V, test=SUMMER_T, horizon=1
        )
    assert report.availability().empty
    assert set(report.validation.availability()["role"]) == {"training target"}

    def same(ctx):
        return PREDICTORS

    with pytest.warns(UserWarning, match="price"):
        report = pipeline(ols(same, window=14), info=info, level="warn").run(
            long_frame, validation=WINTER_V, test=SUMMER_T, horizon=1
        )
    assert report.availability().empty
    assert set(report.validation.availability()["role"]) == {"training target"}


def flagged(frame, tmp_path, level):
    model = OLSModel(PREDICTORS + ["no_rule_column"], training_window=14, name="m")
    return pipeline(model, info=NO_RULE_INFO, level=level).run(
        frame, validation=SUMMER_V, test=SUMMER_T, horizon=1, save_dir=str(tmp_path)
    )


@pytest.mark.parametrize("later_level", ["raise", "none", "warn"])
def test_what_a_run_listed_per_window_stays_on_file(long_frame, tmp_path, later_level):
    frame = long_frame.assign(no_rule_column=long_frame["seasonal"])
    with pytest.warns(UserWarning, match="no rule"):
        flagged(frame, tmp_path, "warn")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flagged(frame, tmp_path, later_level)
    merged = pipeline(OLSModel(PREDICTORS, training_window=14, name="m"))
    with pytest.warns(UserWarning, match="no rule"):
        report = merged.report(tmp_path, horizon=1)
    assert set(report.availability()["predictor"]) == {"no_rule_column"}
    assert set(report.validation.availability()["predictor"]) == {"no_rule_column"}


@pytest.mark.parametrize("later_level", ["none", "warn"])
def test_what_a_plain_run_listed_stays_on_file_as_the_test_windows(long_frame, tmp_path, later_level):
    frame = long_frame.assign(no_rule_column=long_frame["seasonal"])
    model = lambda: OLSModel(PREDICTORS + ["no_rule_column"], training_window=14, name="m")  # noqa: E731
    with pytest.warns(UserWarning, match="no rule"):
        pipeline(model(), info=NO_RULE_INFO, level="warn").run(frame, test=SUMMER_T, horizon=1, save_dir=str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline(model(), info=NO_RULE_INFO, level=later_level).run(
            frame, validation=SUMMER_V, test=SUMMER_T, horizon=1, save_dir=str(tmp_path)
        )
    merged = pipeline(OLSModel(PREDICTORS, training_window=14, name="m"))
    with pytest.warns(UserWarning, match="no rule"):
        report = merged.report(tmp_path, horizon=1)
    assert set(report.availability()["predictor"]) == {"no_rule_column"}
    assert set(report.validation.availability()["predictor"]) == (
        {"no_rule_column"} if later_level == "warn" else set()
    )


def guarded_in_winter_only(ctx):
    return PREDICTORS + (["seasonal"] if ctx.published("seasonal") and ctx.issue_day.month != 12 else [])


def test_an_input_that_never_switches_on_in_a_window_warns_like_a_plain_run(long_frame):
    """Over the run the guard is True on some days, but a plain run over the winter window would say it never is."""
    with pytest.warns(UserWarning, match="never True") as plain:
        pipeline(ols(guarded_in_winter_only, window=14), info=SEASONAL_INFO, level="warn").run(
            long_frame, test=WINTER_V, horizon=1
        )
    assert len([w for w in plain if "never True" in str(w.message)]) == 1
    with pytest.warns(UserWarning, match="never True .*in the validation window"):
        pipeline(ols(guarded_in_winter_only, window=14), info=SEASONAL_INFO, level="warn").run(
            long_frame, validation=WINTER_V, test=SUMMER_T, horizon=1
        )


def guarded_never(ctx):
    return PREDICTORS + (["seasonal"] if ctx.published("all_nan") else [])


def test_an_input_that_never_switches_on_anywhere_warns_once(long_frame):
    frame = long_frame.assign(all_nan=np.nan)
    rules = {**SEASONAL, "all_nan": Availability.static()}
    info = InformationSet("10:30", "UTC", rules=rules)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline(ols(guarded_never, window=14), info=info, level="raise").run(
            frame, validation=SUMMER_V, test=SUMMER_T, horizon=1
        )
    assert len([w for w in caught if "never True" in str(w.message)]) == 1


def coarse_between(frame, first, last):
    """Only every third hour between these days, which the data contract refuses wherever a run reads it."""
    wall = frame.index.tz_localize(None)
    inside = (wall >= pd.Timestamp(first)) & (wall < pd.Timestamp(last))
    return frame[~inside | (wall.hour % 3 == 0)]


def test_a_coarse_stretch_no_window_reads_is_not_refused_and_moves_nothing(df):
    validation, test = ("2023-02-01", "2023-02-02"), ("2023-05-01", "2023-05-02")
    frame = coarse_between(df, "2023-03-10", "2023-03-20")
    make = lambda: OLSModel(PREDICTORS, training_window=14, name="m")  # noqa: E731
    combined = all_records(pipeline(make()).run(frame, validation=validation, test=test, horizon=1))
    clean = all_records(pipeline(make()).run(df, validation=validation, test=test, horizon=1))
    assert lines(combined) == lines(clean)


@pytest.mark.parametrize(
    "stretch", [("2023-01-25", "2023-01-30"), ("2023-04-20", "2023-04-25")], ids=["validation", "test"]
)
def test_a_coarse_stretch_a_window_reads_is_refused(df, stretch):
    validation, test = ("2023-02-01", "2023-02-02"), ("2023-05-01", "2023-05-02")
    frame = coarse_between(df, *stretch)
    with pytest.raises(ValueError, match="coarser|most commonly"):
        pipeline(OLSModel(PREDICTORS, training_window=14, name="m")).run(
            frame, validation=validation, test=test, horizon=1
        )


def test_a_plan_made_for_other_windows_is_refused(df):
    model = ols(remembers)
    plan = model._plan_predictors(df, "price", 1, "1h", INFO, windows=RunWindows.resolve(V, T))
    with pytest.raises(ValueError, match="test period"):
        model.run(df, *T, horizon=1, information_set=INFO, plan=plan)
    with pytest.raises(ValueError, match="test period"):
        model.run(df, horizon=1, information_set=INFO, plan=plan, windows=RunWindows.resolve(None, T))


# ---------------------------------------------------------------------------
# The merge after a SLURM array, Workflow YAML and the CLI
# ---------------------------------------------------------------------------


def test_report_from_saved_results_is_what_the_run_returned(df, tmp_path):
    make = lambda: pipeline(*models(), evaluators=[MAEEvaluator(), rMAEEvaluator("small")])  # noqa: E731
    returned = make().run(df, validation=LONG_V, test=LONG_T, horizon=2, save_dir=str(tmp_path))
    merged = make().report(tmp_path, validation=LONG_V, test=LONG_T, horizon=2)
    from_manifests = make().report(tmp_path, horizon=2)
    for report in (merged, from_manifests):
        assert report.window == "test"
        pd.testing.assert_frame_equal(report.summary(), returned.summary())
        pd.testing.assert_frame_equal(report.validation.summary(), returned.validation.summary())
        # What the run skipped and why is not on file, except what the report leaves out itself.
        columns = ["model", "planned", "produced", "missing"]
        pd.testing.assert_frame_equal(report.forecast_counts()[columns], returned.forecast_counts()[columns])
        columns.append("target_in_test")
        pd.testing.assert_frame_equal(
            report.validation.forecast_counts()[columns], returned.validation.forecast_counts()[columns]
        )
        pd.testing.assert_frame_equal(report.predictions(), returned.predictions())
        assert report.windows == returned.windows


def test_report_says_what_run_was_told(df, tmp_path):
    make = lambda: pipeline(*models(), evaluators=[MAEEvaluator()])  # noqa: E731
    make().run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))

    test_only = make().report(tmp_path, test=T, horizon=1)
    assert test_only.window == "test"
    with pytest.raises(ValueError, match="no validation window"):
        test_only.validation
    assert make().report(tmp_path, *T, horizon=1).predictions()["run_date"].min() == T[0]

    validation_only = make().report(tmp_path, validation=V, horizon=1)
    assert validation_only.window == "validation"

    with pytest.raises(ValueError, match="not what the results were made with"):
        make().report(tmp_path, validation=V, test=("2023-04-03", "2023-04-04"), horizon=1)
    with pytest.raises(ValueError, match="not what the results were made with"):
        make().report(tmp_path, validation=("2023-03-20", "2023-03-22"), test=T, horizon=1)


def test_report_of_validation_alone_cuts_at_the_test_on_file(df, tmp_path):
    validation, test = ("2023-04-01", "2023-04-02"), ("2023-04-03", "2023-04-04")
    make = lambda: pipeline(ols(), evaluators=[MAEEvaluator()])  # noqa: E731
    full = make().run(df, validation=validation, test=test, horizon=3, save_dir=str(tmp_path))
    alone = make().report(tmp_path, validation=validation, horizon=3)
    assert alone.window == "validation"
    assert alone.predictions()["target_date"].max() == "2023-04-02"
    pd.testing.assert_frame_equal(alone.predictions(), full.validation.predictions())


def test_report_of_a_plain_run_has_no_windows_to_ask_for(df, tmp_path):
    make = lambda: pipeline(ols(), evaluators=[MAEEvaluator()])  # noqa: E731
    make().run(df, test=T, horizon=1, save_dir=str(tmp_path))
    assert make().report(tmp_path, test=T, horizon=1).summary().loc[0, "forecasts"] == 3 * 24
    with pytest.raises(ValueError, match="made without validation"):
        make().report(tmp_path, validation=V, test=T, horizon=1)
    with pytest.raises(ValueError, match="No window"):
        make().report(tmp_path, horizon=1)


def test_report_of_a_validation_run_is_the_validation_report(df, tmp_path):
    make = lambda: pipeline(ols(), evaluators=[MAEEvaluator()])  # noqa: E731
    make().run(df, validation=V, horizon=1, save_dir=str(tmp_path))
    report = make().report(tmp_path, validation=V, horizon=1)
    assert report.window == "validation"
    assert report.summary().loc[0, "forecasts"] == 3 * 24


def test_shards_with_different_windows_raise(df, tmp_path):
    a, b = models()[:2]
    pipeline(a).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))
    pipeline(b).run(df, validation=("2023-03-20", V[1]), test=T, horizon=1, save_dir=str(tmp_path))
    with pytest.raises(ValueError, match="different windows"):
        pipeline(a, b).report(tmp_path, horizon=1)


def test_a_shard_without_windows_next_to_one_with_them_raises(df, tmp_path):
    a, b = models()[:2]
    pipeline(a).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))
    pipeline(b).run(df, test=T, horizon=1, save_dir=str(tmp_path))
    with pytest.raises(ValueError, match="different windows"):
        pipeline(a, b).report(tmp_path, test=T, horizon=1)


def test_a_missing_shard_is_refused_or_left_out(df, tmp_path):
    a, b = models()[:2]
    pipeline(a).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="small"):
        pipeline(a, b).report(tmp_path, horizon=1)
    partial = pipeline(a, b).report(tmp_path, horizon=1, allow_missing=True)
    assert list(partial.refs) == ["m"] and list(partial.validation.refs) == ["m"]


def test_report_warns_about_a_shortfall_in_a_window_but_not_about_later_targets(df, tmp_path):
    validation, test = ("2023-04-01", "2023-04-02"), ("2023-04-03", "2023-04-04")
    pipeline(ols()).run(df, validation=validation, test=test, horizon=3, save_dir=str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipeline(ols()).report(tmp_path, horizon=3)

    path = tmp_path / "m.jsonl"
    kept = [line for line in path.read_text().splitlines() if '"hour": 5, "horizon": 1' not in line]
    path.write_text("\n".join(kept) + "\n")
    with pytest.warns(UserWarning, match="m in the test window .*m in the validation window|m in the validation window .*m in the test window"):  # fmt: skip
        pipeline(ols()).report(tmp_path, horizon=3)


@pytest.fixture
def experiment(tmp_path, df):
    frame = df[[c for c in df.columns if c in ("price", "load", "wind", "is_weekend", "is_weekday")]]
    index = frame.index.tz_convert("UTC")
    csv = frame.reset_index(drop=True)
    csv.insert(0, "datetime", index.strftime("%Y-%m-%d %H:%M:%S"))
    csv.to_csv(tmp_path / "market.csv", index=False)
    predictors = ["price_d-1", "price_d-7"]
    config = {
        "data_pipeline": {
            "start": "2023-01-01",
            "end": "2023-05-30",
            "sources": [{"class": "CsvSource", "params": {"file_path": str(tmp_path / "market.csv")}}],
            "transformers": [
                {"class": "TimezoneTransformer", "params": {"target_tz": "Europe/Warsaw"}},
                {"class": "LagTransformer", "params": {"columns": ["price"], "lags": [1, 7], "freq": "day"}},
            ],
        },
        "model_pipeline": {
            "target": "price",
            "horizon": 1,
            "save_dir": "results",
            "models": [
                {"class": "OLSModel", "params": {"predictors": predictors, "training_window": 30, "name": "OLS"}},
                {"class": "OLSModel", "params": {"predictors": predictors[:1], "training_window": 30, "name": "Short"}},
            ],
            "evaluators": [
                {"class": "MAEEvaluator", "params": {}},
                {"class": "rMAEEvaluator", "params": {"base_model": "OLS"}},
            ],
            "exporters": [{"class": "CsvExporter", "params": {"path": str(tmp_path / "out.csv")}}],
        },
    }
    path = tmp_path / "experiment.yaml"

    def write(validation=None, test=None, **environment):
        model = dict(config["model_pipeline"])
        if validation is not None:
            model["validation"] = validation
        if test is not None:
            model["test"] = test
        cfg = {**config, "model_pipeline": model}
        if environment:
            cfg["environment"] = environment
        path.write_text(yaml.safe_dump(cfg))
        return path

    return write


def test_a_workflow_runs_both_windows(tmp_path, experiment):
    report = Workflow.load(experiment(validation=list(V), test=list(T))).run()
    assert report.window == "test"
    assert report.summary()["forecasts"].tolist() == [3 * 24, 3 * 24]
    assert report.validation.summary()["forecasts"].tolist() == [3 * 24, 3 * 24]
    days = pd.read_csv(tmp_path / "out.csv")["run_date"]
    assert days.min() == T[0] and days.max() == T[1]


def test_a_workflow_keeps_its_windows_in_yaml(tmp_path, experiment):
    workflow = Workflow.load(experiment(validation="previous_year", test=list(T)))
    assert workflow.model_validation == "previous_year"
    assert workflow.model_test == list(T)
    workflow.save(tmp_path / "copy.yaml")
    saved = yaml.safe_load((tmp_path / "copy.yaml").read_text())["model_pipeline"]
    assert saved["validation"] == "previous_year" and saved["test"] == list(T)
    assert Workflow.load(tmp_path / "copy.yaml").model_test == list(T)


def test_a_workflow_takes_unquoted_dates_from_yaml(tmp_path, experiment):
    path = experiment()
    config = yaml.safe_load(path.read_text())
    config["model_pipeline"]["validation"] = [pd.Timestamp(V[0]).date(), pd.Timestamp(V[1]).date()]
    config["model_pipeline"]["test"] = [pd.Timestamp(T[0]).date(), pd.Timestamp(T[1]).date()]
    path.write_text(yaml.safe_dump(config))
    report = Workflow.load(path).run()
    assert report.windows == RunWindows.resolve(V, T)


def test_a_workflow_refuses_a_test_period_given_twice(experiment):
    path = experiment(test=list(T))
    config = yaml.safe_load(path.read_text())
    config["model_pipeline"]["test_start"] = T[0]
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="once"):
        Workflow.load(path).run()


def test_an_array_with_windows_merges_to_a_single_run(tmp_path, experiment):
    for index in (0, 1):
        shard = Workflow.load(experiment(validation=list(V), test=list(T), model_index=index)).run()
        assert shard.window == "test"
    merged = Workflow.load(experiment(validation=list(V), test=list(T))).report()
    assert merged.window == "test" and merged.validation.window == "validation"
    assert read_windows(tmp_path / "results" / "ols.jsonl") == read_windows(tmp_path / "results" / "short.jsonl")
    counts = merged.forecast_counts().set_index("model")
    assert (counts["planned"] == 3 * 24).all() and (counts["produced"] == 3 * 24).all()

    (tmp_path / "results").rename(tmp_path / "array_results")
    single = Workflow.load(experiment(validation=list(V), test=list(T))).run()
    pd.testing.assert_frame_equal(single.summary(), merged.summary(), check_like=True)
    pd.testing.assert_frame_equal(single.validation.summary(), merged.validation.summary(), check_like=True)
    for name in ("ols", "short"):
        assert lines(stored(tmp_path / "results" / f"{name}.jsonl")) == lines(
            stored(tmp_path / "array_results" / f"{name}.jsonl")
        )


def test_an_array_whose_tasks_ran_on_different_windows_raises_on_merge(experiment):
    Workflow.load(experiment(validation=list(V), test=list(T), model_index=0)).run()
    Workflow.load(experiment(validation=["2023-03-20", V[1]], test=list(T), model_index=1)).run()
    with pytest.raises(ValueError, match="different windows"):
        Workflow.load(experiment(validation=list(V), test=list(T))).report()


def test_the_cli_report_merges_an_array_with_windows(tmp_path, experiment, capsys):
    for index in (0, 1):
        path = experiment(validation=list(V), test=list(T), model_index=index)
        cli.cmd_run(cli.build_parser().parse_args(["run", str(path)]))
    path = experiment(validation=list(V), test=list(T))
    cli.cmd_report(cli.build_parser().parse_args(["report", str(path)]))
    assert "planned" in capsys.readouterr().out


@pytest.fixture
def csv_frame(tmp_path, df):
    from epftoolbox2._frame_csv import save_frame_csv

    path = tmp_path / "frame.csv"
    save_frame_csv(df[["price", "load", "load_d+1", "is_weekend_d+1", "is_weekday_d+1"]], path)
    built = pipeline(OLSModel(["load_d+1", "is_weekend_d+1", "is_weekday_d+1"], training_window=30, name="OLS"))
    built.save(tmp_path / "mp.yaml")
    return path, tmp_path / "mp.yaml"


def model_args(paths, *flags):
    data, model = paths
    return cli.build_parser().parse_args(["model", str(model), "--data", str(data), "--horizon", "1", *flags])


def test_epf_model_runs_validation_and_test(tmp_path, csv_frame):
    flags = ["--validation", *V, "--test-start", T[0], "--test-end", T[1], "--save-dir", str(tmp_path / "out")]
    cli.cmd_model(model_args(csv_frame, *flags))
    assert read_windows(tmp_path / "out" / "ols.jsonl") == RunWindows.resolve(V, T)


def test_epf_model_runs_validation_alone(tmp_path, csv_frame):
    cli.cmd_model(model_args(csv_frame, "--validation", *V, "--save-dir", str(tmp_path / "out")))
    assert {r["window"] for r in stored(tmp_path / "out" / "ols.jsonl")} == {"validation"}


def test_epf_model_with_a_plain_test_period_stays_plain(tmp_path, csv_frame):
    flags = ["--test-start", T[0], "--test-end", T[1], "--save-dir", str(tmp_path / "out")]
    cli.cmd_model(model_args(csv_frame, *flags))
    assert all("window" not in r for r in stored(tmp_path / "out" / "ols.jsonl"))


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ([], "No window"),
        (["--validation", "last_year", "--test-start", T[0], "--test-end", T[1]], "previous_year"),
        (["--validation", "previous_year"], "test="),
        (["--test-start", T[0]], "test"),
        (["--validation", "2023-03-27", "2023-03-25"], "reversed"),
    ],
)
def test_epf_model_refuses_what_makes_no_sense(tmp_path, csv_frame, capsys, flags, message):
    with pytest.raises(SystemExit):
        cli.cmd_model(model_args(csv_frame, *flags))
    assert re.search(message, capsys.readouterr().out)


# ---------------------------------------------------------------------------
# The checks for look-ahead when validation is in use
# ---------------------------------------------------------------------------

VERIFY_RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published("13:00", 1),
    "from_tomorrow": Availability.realised("1h"),
}
VERIFY_INFO = InformationSet("11:45", "Europe/Warsaw", rules=VERIFY_RULES)


@pytest.fixture(scope="module")
def verify_frame():
    frame = make_market_data(horizon=1, noise=5.0, seed=3)
    frame["from_tomorrow"] = calendar_shift(frame["price"], 1)  # the day-ahead price of tomorrow, a planted leak
    return frame


def verify_clean_function(ctx):
    return PREDICTORS + (["price_d-1"] if ctx.published("price_d-1") else [])


def verify_leaky_function(ctx):
    return PREDICTORS + (["from_tomorrow"] if ctx.value("from_tomorrow") > 80 else [])


@pytest.mark.parametrize("predictors", [PREDICTORS + ["price_d-1"], verify_clean_function], ids=["fixed", "ctx"])
def test_verify_and_the_probe_pass_a_clean_model_on_each_window(verify_frame, predictors):
    built = pipeline(OLSModel(predictors, training_window=30, name="m"), info=VERIFY_INFO, level="warn")
    report = built.run(verify_frame, validation=V, test=T, horizon=1)
    assert report.window == "test"
    for period in (V, T):
        checked = built.verify(verify_frame, *period, horizon=1, days=2)
        assert checked.ok, checked.problems()
        probe = leakage_probe(
            lambda: OLSModel(predictors, training_window=30, name="m"), verify_frame, period[0], VERIFY_INFO, horizon=1
        )
        probe.assert_ok()


@pytest.mark.parametrize("predictors", [PREDICTORS + ["from_tomorrow"], verify_leaky_function], ids=["fixed", "ctx"])
def test_verify_and_the_probe_fail_a_planted_leak_on_each_window(verify_frame, predictors):
    built = pipeline(OLSModel(predictors, training_window=30, name="m"), info=VERIFY_INFO, level="warn")
    with pytest.warns(UserWarning, match="from_tomorrow"):
        built.run(verify_frame, validation=V, test=T, horizon=1)
    for period in (V, T):
        checked = built.verify(verify_frame, *period, horizon=1, days=2)
        assert not checked.ok
        assert checked.table().loc[0, "status"] == "leak"
        probe = leakage_probe(
            lambda: OLSModel(predictors, training_window=30, name="m"),
            verify_frame,
            period[0],
            VERIFY_INFO,
            horizon=1,
            availability="warn",
        )
        assert not probe.ok


def test_a_run_with_validation_leaves_nothing_on_the_model_that_verify_sees(verify_frame):
    built = pipeline(OLSModel(PREDICTORS, training_window=30, name="m"), info=VERIFY_INFO, level="warn")
    before = built.verify(verify_frame, *T, horizon=1, days=2).table()
    built.run(verify_frame, validation=V, test=T, horizon=1)
    pd.testing.assert_frame_equal(built.verify(verify_frame, *T, horizon=1, days=2).table(), before)


def test_a_late_predictor_at_level_raise_stops_a_run_with_validation(verify_frame):
    built = pipeline(OLSModel(PREDICTORS + ["from_tomorrow"], training_window=30), info=VERIFY_INFO, level="raise")
    with pytest.raises(InformationSetError, match="from_tomorrow"):
        built.run(verify_frame, validation=V, test=T, horizon=1)
