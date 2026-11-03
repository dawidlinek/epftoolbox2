"""`NaiveModel`: a forecast that is one column's value on the issue row, with no fit, scaler or training rows.

The column here is a TSO's week-ahead forecast (`load_forecast_daily_max_d+k`, the built-in rule: published Fridays 10:00
Brussels for the next Monday-Sunday week), so with an issue at noon some cells aren't out yet on Monday to Thursday. With
`known_only=True` (the default) such a key is no forecast, `not_published`, at every availability level and in `verify`: Monday
lacks h=7, Tuesday 6-7, Wednesday 5-7, Thursday 4-7, so 10 of the 49 keys of a week at horizons 1 to 7.
"""

import datetime
import json
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.information_set import Availability, InformationSet, InformationSetError
from epftoolbox2.models import NaiveModel, OLSModel, Use
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from tests.synthetic import make_market_data

ZONE = "Europe/Warsaw"
TSO = "load_forecast_daily_max_d+{horizon}"
RULES = {"actual_daily_max": Availability.realised("1h")}
INFO = InformationSet("12:00", ZONE, rules=RULES)
# Two whole weeks, Monday to Sunday, over the spring switch of 31 March.
START, END = "2024-03-25", "2024-04-07"
LATE = {0: {7}, 1: {6, 7}, 2: {5, 6, 7}, 3: {4, 5, 6, 7}, 4: set(), 5: set(), 6: set()}  # weekday -> late horizons


class InProcessPool:
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


def tso_list(ctx):
    """The fixed list as a function of `ctx`: the same one name on every forecast."""
    return [f"load_forecast_daily_max_d+{ctx.horizon}"]


def tso_of_the_next_day(ctx):
    return ["load_forecast_daily_max_d+1"]


def two_names(ctx):
    return ["load_forecast_daily_max_d+1", "load_forecast_daily_max_d+2"]


def matching(ctx):
    return Use(["load_forecast_daily_max_d+1"], train_on="matching")


def daily_frame(first="2024-03-18", days=40):
    """One row per local date at noon. Each `_d+k` column holds a value no other cell has (1000 k + the row's number), so a
    forecast read from another row or another horizon can't pass; `actual_daily_max` is the target."""
    dates = pd.date_range(first, periods=days, freq="D")
    index = (dates + pd.Timedelta(hours=12)).tz_localize(ZONE)
    frame = pd.DataFrame({"actual_daily_max": 20000 + 50 * np.arange(days, dtype=float)}, index=index)
    for k in range(1, 8):
        frame[f"load_forecast_daily_max_d+{k}"] = 1000.0 * k + np.arange(days)
    return frame


def keyed(records, *drop):
    return {
        (r["run_date"], r["hour"], r["horizon"]): json.dumps(
            {k: v for k, v in r.items() if k not in drop}, sort_keys=True
        )
        for r in records
    }


def published_keys(start=START, end=END, horizons=range(1, 8)):
    days = pd.date_range(start, end, freq="D")
    return {(day.date().isoformat(), 0, h) for day in days for h in horizons if h not in LATE[day.weekday()]}


# --- the forecast ------------------------------------------------------------------------------------------------------------


def test_a_forecast_is_the_column_on_its_issue_row_with_no_history():
    """Read from the row dated D, at the horizon's column, whatever came before: the run starts on the frame's first date."""
    frame = daily_frame()
    first = frame.index[0].date().isoformat()
    ref = NaiveModel([TSO], name="TSO").run(frame, first, END, target="actual_daily_max", horizon=7, freq="1D")
    assert ref.produced == ref.count == 21 * 7 and not ref.skipped
    local = frame.index.tz_localize(None).normalize()
    for record in ref._results:
        day = pd.Timestamp(record["run_date"])
        row = frame[local == day].iloc[0]
        assert record["prediction"] == row[f"load_forecast_daily_max_d+{record['horizon']}"]
        target_day = day + pd.Timedelta(days=record["horizon"])
        assert record["actual"] == frame.loc[local == target_day, "actual_daily_max"].iloc[0]
        assert record["coefficients"] == [] and record["hour"] == 0 and record["freq"] == "1D"


def test_a_record_has_the_keys_of_a_fitted_model_s():
    frame = daily_frame()
    naive = NaiveModel([TSO]).run(frame, "2024-03-28", "2024-03-28", target="actual_daily_max", horizon=1, freq="1D")
    ols = OLSModel([TSO], training_window=5).run(
        frame, "2024-03-28", "2024-03-28", target="actual_daily_max", horizon=1, freq="1D"
    )
    assert list(naive._results[0]) == list(ols._results[0])


def test_an_hourly_forecast_reads_its_period_and_the_second_occurrence_of_a_repeated_hour():
    hourly = make_market_data(start="2023-10-20", days=16, horizon=2)
    ref = NaiveModel(["load_d+{horizon}"]).run(hourly, "2023-10-27", "2023-10-31", target="price", horizon=2)
    local = hourly.index.tz_localize(None)
    for record in ref._results:
        rows = hourly[(local.normalize() == pd.Timestamp(record["run_date"])) & (local.hour == record["hour"])]
        assert record["prediction"] == rows[f"load_d+{record['horizon']}"].iloc[-1]
    switch = [r for r in ref._results if r["run_date"] == "2023-10-29" and r["hour"] == 2]
    assert len(switch) == 2  # one key per horizon, from the later 02:00


def test_a_missing_value_is_a_missing_predictor_and_a_missing_row_is_a_missing_row():
    frame = daily_frame()
    frame.loc[frame.index[10], "load_forecast_daily_max_d+2"] = np.nan  # 2024-03-28
    frame = frame.drop(frame.index[11])  # 2024-03-29
    ref = NaiveModel([TSO]).run(frame, "2024-03-27", "2024-03-30", target="actual_daily_max", horizon=2, freq="1D")
    assert ref.skipped == {"missing_test_predictors": 1, "missing_test_row": 2}
    assert ref.produced == 8 - 3


def test_a_function_of_ctx_gives_the_fixed_list_s_forecasts():
    frame = daily_frame()
    fixed = NaiveModel([TSO]).run(frame, START, END, target="actual_daily_max", horizon=7, freq="1D")
    function = NaiveModel(tso_list).run(frame, START, END, target="actual_daily_max", horizon=7, freq="1D")
    assert keyed(function._results, "predictor_list") == keyed(fixed._results)
    assert len({r["predictor_list"] for r in function._results}) == 7  # one list per horizon


@pytest.mark.parametrize(
    "build, error",
    [
        (lambda: NaiveModel(["a", "b"]), ValueError),
        (lambda: NaiveModel("load"), TypeError),
        (lambda: NaiveModel([TSO], known_only="yes"), TypeError),
        (lambda: NaiveModel([TSO], refit="7D"), TypeError),
        (lambda: NaiveModel([TSO], newest_target="published"), TypeError),
    ],
    ids=["two names", "a string", "known_only not a bool", "refit", "newest_target"],
)
def test_what_a_naive_model_refuses_at_construction(build, error):
    with pytest.raises(error):
        build()


def test_refit_and_newest_target_raise_when_set_later_too():
    model = NaiveModel([TSO])
    with pytest.raises(ValueError, match="nothing to refit"):
        model.refit = "1MS"
    with pytest.raises(ValueError, match="no newest training target"):
        model.newest_target = "published"
    model.refit, model.newest_target = None, "d-1"  # the defaults are fine


@pytest.mark.parametrize(
    "predictors, match",
    [([], "0 names"), (two_names, "returned 2 names"), (matching, "train_on='matching'")],
    ids=["no name", "a function of two names", "a matching list"],
)
def test_what_a_naive_model_refuses_at_run_time(predictors, match):
    frame = daily_frame()
    with pytest.raises(ValueError, match=match):
        NaiveModel(predictors).run(frame, START, END, target="actual_daily_max", horizon=2, freq="1D")


# --- known_only -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("level", ["none", "warn", "raise"])
def test_a_late_cell_is_not_published_at_every_level(level, recwarn):
    frame = daily_frame()
    pipeline = ModelPipeline(information_set=INFO, availability=level).add_model(NaiveModel([TSO], name="TSO"))
    report = pipeline.run(frame, START, END, target="actual_daily_max", horizon=7, freq="1D")
    ref = report.refs["TSO"]
    assert {(r["run_date"], r["hour"], r["horizon"]) for r in ref._results} == published_keys()
    assert ref.count == 14 * 7 and ref.produced == 14 * 7 - 20
    assert ref.skipped == {"not_published": 20}
    counts = report.forecast_counts().set_index("model").loc["TSO"]
    assert (counts["planned"], counts["produced"], counts["not_published"]) == (98, 78, 20)
    assert report.availability().empty  # neither listed nor refused


def test_the_forecasts_are_the_same_at_every_level_and_with_a_function():
    frame = daily_frame()
    found = []
    for level in ("none", "warn", "raise"):
        for predictors in ([TSO], tso_list):
            pipeline = ModelPipeline(information_set=INFO, availability=level)
            report = pipeline.add_model(NaiveModel(predictors, name="TSO")).run(
                frame, START, END, target="actual_daily_max", horizon=7, freq="1D"
            )
            found.append(keyed(report.refs["TSO"]._results, "predictor_list"))
    assert all(records == found[0] for records in found)


def test_without_an_information_set_every_cell_is_a_forecast():
    frame = daily_frame()
    ref = NaiveModel([TSO]).run(frame, START, END, target="actual_daily_max", horizon=7, freq="1D")
    assert ref.produced == 98 and not ref.skipped


def test_known_only_false_judges_the_column_as_any_input():
    frame = daily_frame()
    pipeline = ModelPipeline(information_set=INFO, availability="warn")
    report = pipeline.add_model(NaiveModel([TSO], name="TSO", known_only=False)).run(
        frame, START, END, target="actual_daily_max", horizon=7, freq="1D"
    )
    assert report.refs["TSO"].produced == 98
    listed = report.availability()
    assert set(listed["predictor"]) == {f"load_forecast_daily_max_d+{h}" for h in (4, 5, 6, 7)}
    assert set(listed["status"]) == {"late"} and set(listed["role"]) == {"predictor"}

    pipeline = ModelPipeline(information_set=INFO, availability="raise")
    with pytest.raises(InformationSetError, match="load_forecast_daily_max_d\\+7"):
        pipeline.add_model(NaiveModel([TSO], name="TSO", known_only=False)).run(
            frame, START, END, target="actual_daily_max", horizon=7, freq="1D"
        )


def test_a_column_without_a_rule_follows_unknown_and_the_level():
    frame = daily_frame()
    frame["own_forecast_d+1"] = frame["load_forecast_daily_max_d+1"]
    strict = ModelPipeline(information_set=INFO, availability="raise").add_model(NaiveModel(["own_forecast_d+1"]))
    with pytest.raises(InformationSetError, match="no availability rule"):
        strict.run(frame, START, END, target="actual_daily_max", horizon=1, freq="1D")
    report = (
        ModelPipeline(information_set=INFO, availability="warn")
        .add_model(NaiveModel(["own_forecast_d+1"], name="own"))
        .run(frame, START, END, target="actual_daily_max", horizon=1, freq="1D")
    )
    assert report.refs["own"].produced == 14  # counts as known, and listed as having no rule
    assert set(report.availability()["status"]) == {"no rule"}


def test_the_rules_that_decide_which_forecasts_exist_are_compared_on_resume(tmp_path):
    frame = daily_frame()
    path = tmp_path / "tso.jsonl"
    NaiveModel([TSO]).run(
        frame, START, END, target="actual_daily_max", horizon=7, freq="1D", save_to=str(path), information_set=INFO
    )
    manifest = json.loads(manifest_path(path).read_text())
    assert manifest["training_cut"]["known_only"] == {
        "issue_time": "12:00",
        "issue_timezone": ZONE,
        "rules": {
            f"load_forecast_daily_max_d+{h}": {
                "kind": "weekly",
                "weekday": "friday",
                "time": "10:00",
                "timezone": "Europe/Brussels",
            }
            for h in range(1, 8)
        },
    }
    assert manifest["skipped"] == {"test": {"not_published": 20}}
    first = path.read_text()
    again = NaiveModel([TSO]).run(
        frame, START, END, target="actual_daily_max", horizon=7, freq="1D", save_to=str(path), information_set=INFO
    )
    assert path.read_text() == first and again.produced == 78

    # Another issue time would publish other cells: the stored set isn't this run's.
    later = InformationSet("10:30", "Europe/Brussels", rules=RULES)
    with pytest.raises(ManifestMismatchError, match="training_cut"):
        NaiveModel([TSO]).run(
            frame, START, END, target="actual_daily_max", horizon=7, freq="1D", save_to=str(path), information_set=later
        )
    # Without an information set every cell is a forecast, and the cut is the plain one.
    plain = tmp_path / "plain.jsonl"
    NaiveModel([TSO]).run(frame, START, END, target="actual_daily_max", horizon=7, freq="1D", save_to=str(plain))
    assert json.loads(manifest_path(plain).read_text())["training_cut"] == {"rows": "dates"}


def test_a_naive_model_saves_and_loads(tmp_path):
    path = tmp_path / "p.yaml"
    pipeline = ModelPipeline(information_set=INFO).add_model(NaiveModel([TSO], name="TSO"))
    pipeline.add_model(NaiveModel(tso_of_the_next_day, name="next", known_only=False))
    saved = pipeline.to_dict()["models"]
    assert saved[0] == {"class": "NaiveModel", "params": {"predictors": [TSO], "name": "TSO", "known_only": True}}
    assert saved[1]["params"]["predictors"] == {"function": f"{__name__}:tso_of_the_next_day"}
    pipeline.save(path)
    first, second = ModelPipeline.load(path).models
    assert (first.predictors, first.name, first.known_only, first.training_window) == ([TSO], "TSO", True, 0)
    assert second._ctx_fn is tso_of_the_next_day and second.known_only is False


# --- verify -----------------------------------------------------------------------------------------------------------------


def test_verify_is_ok_on_every_weekday_with_its_control():
    """A forecast that isn't out has no forecast in either run of the probe, so nothing moves; each one that is moves under the
    control, which perturbs the TSO's value of every day from D-1 on."""
    frame = daily_frame()
    pipeline = ModelPipeline(information_set=INFO).add_model(NaiveModel([TSO], name="TSO"))
    report = pipeline.verify(frame, START, "2024-03-31", target="actual_daily_max", horizon=7, freq="1D", days=7)
    assert sorted(datetime.date.fromisoformat(day).weekday() for day in report.issue_days) == list(range(7))
    assert report.status("TSO") == "ok", report.problems()
    checks = {check.run_day: check for check in report.checks}
    for day, check in checks.items():
        weekday = datetime.date.fromisoformat(day).weekday()
        assert check.probe.forecasts == 7 - len(LATE[weekday])
        assert len(check.control.moved) == 7 - len(LATE[weekday])


def test_verify_sees_a_late_cell_let_through(monkeypatch):
    """The mutant `known_only` lets a late cell through: on Monday to Thursday the TSO's next-week value is perturbed and the
    forecast moves."""
    monkeypatch.setattr(NaiveModel, "_known_only", property(lambda self: False))
    frame = daily_frame()
    pipeline = ModelPipeline(information_set=INFO).add_model(NaiveModel([TSO], name="TSO"))
    report = pipeline.verify(frame, START, "2024-03-31", target="actual_daily_max", horizon=7, freq="1D", days=7)
    assert report.status("TSO") == "leak"
