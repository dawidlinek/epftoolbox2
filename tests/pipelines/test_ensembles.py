"""`AverageEnsemble`: an average of stored forecasts on identical keys, chosen on validation without look-ahead.

Most tests here write what a run leaves behind by hand and call `ModelPipeline.report()`: the numbers are then known, a year
of days costs nothing, and the ensemble is computed by the same code as after a real run. The tests that need a real run use
OLS on a few days. Numbers are compared bit for bit.
"""

import argparse
import datetime
import json
import os
import random
import warnings
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from epftoolbox2 import cli
from epftoolbox2.ensembles import AverageEnsemble, Ensemble
from epftoolbox2.ensembles._compute import MIN_VALIDATION_DAYS, MIN_VALIDATION_ISSUES, MIN_VALIDATION_SHARE
from epftoolbox2.evaluators import (
    BiasEvaluator,
    DMEvaluator,
    EconomicEvaluator,
    MAEEvaluator,
    MCSEvaluator,
    RMSEEvaluator,
    rMAEEvaluator,
)
from epftoolbox2.evaluators._compare import LossPanel, compare_set, confidence_set
from epftoolbox2.exporters import CsvExporter, ExcelExporter, TerminalExporter
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import BaseModel, OLSModel
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline, Workflow
from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY
from epftoolbox2.results import EvaluationReport, ManifestMismatchError, RunWindows, read_windows
from epftoolbox2.results import manifest as result_manifest
from epftoolbox2.results.manifest import manifest_path
from tests.synthetic import PREDICTORS, make_market_data

V = ("2023-03-28", "2023-03-31")
T = ("2023-04-03", "2023-04-05")
SMALL = ["load_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]
WIND = ["wind_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]
RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published("13:00", 1),
}
INFO = InformationSet("11:45", "Europe/Warsaw", rules=RULES)
KEY = ("run_date", "target_date", "hour", "horizon")

# The windows of the results written by hand below.
HV = ("2024-05-01", "2024-05-08")
HT = ("2024-05-09", "2024-05-12")
HAND = RunWindows.fixed(HV, HT)
DAY = 24  # periods of a day
HOURS = tuple(range(DAY))
# What a horizon-1 forecast can serve of `HV`: the last validation day's forecast is for the first test issue day.
SERVED = 7 * DAY


class InProcessPool:
    """Stands in for the worker pool, whose start is most of the time these runs would take."""

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


@pytest.fixture(scope="module")
def df():
    return make_market_data(horizon=3, noise=5.0, seed=3)


class MeanModel(BaseModel):
    """The cheapest model there is, for runs that are about days and tags and need a year of them."""

    def _fit_predict(self, train_x, train_y, test_x):
        return float(train_y.mean()), []


def ols(name, predictors=PREDICTORS, window=30):
    return OLSModel(predictors, training_window=window, name=name)


def full_small_wind():
    return [ols("full"), ols("small", SMALL), ols("wind", WIND)]


def built(*components, info=None):
    pipeline = ModelPipeline(information_set=info)
    for component in components:
        if isinstance(component, Ensemble):
            pipeline.add_ensemble(component)
        else:
            pipeline.add_model(component)
    return pipeline


def records_of(report, name):
    return list(report.refs[name].iter_records())


def stored(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def key_of(record):
    return tuple(record[k] for k in KEY)


def keyed(records):
    return {key_of(r): r for r in records}


def forecast_lines(records):
    """Records without what only a model has, as text, which compares them bit for bit and NaN equal to NaN."""
    return sorted(
        json.dumps({k: v for k, v in r.items() if k not in ("coefficients", "predictor_list")}, sort_keys=True)
        for r in records
    )


def without_timestamps(path):
    return {k: v for k, v in json.loads(Path(path).read_text()).items() if k not in ("created_at", "updated_at")}


def counts_of(report):
    return report.forecast_counts().set_index("model")


# ---------------------------------------------------------------------------------------------------------------------
# Results written by hand: what a run leaves behind, with known numbers
# ---------------------------------------------------------------------------------------------------------------------


def days_of(first, last):
    return RunWindows(test=(first, last)).issue_days()


def plus_days(day, count):
    return (datetime.date.fromisoformat(day) + datetime.timedelta(days=count)).isoformat()


def actual_of(target_date, hour):
    """One series that every member reads its `actual` from, as members of one frame do. Whole numbers keep errors exact."""
    return 100.0 + datetime.date.fromisoformat(target_date).toordinal() % 50 + hour


def forecasts(days, error=0.0, hours=HOURS, horizons=(1,), windows=None, freq="1h"):
    """A member's records: `prediction` is `actual` + `error`, a number or a function of the record's key."""
    records = []
    starts = dict(windows.periods) if windows else {}
    for run_date in days:
        for horizon in horizons:
            target_date = plus_days(run_date, horizon)
            for hour in hours:
                off = error(run_date, target_date, hour, horizon) if callable(error) else error
                window = windows.window_of(run_date) if windows else None
                first = starts[window][0] if window else days[0]
                record = {
                    "run_date": run_date,
                    "target_date": target_date,
                    "run_weekday": datetime.date.fromisoformat(run_date).weekday(),
                    "hour": hour,
                    "horizon": horizon,
                    "day_in_test": (datetime.date.fromisoformat(run_date) - datetime.date.fromisoformat(first)).days,
                    "prediction": actual_of(target_date, hour) + off,
                    "actual": actual_of(target_date, hour),
                    "coefficients": [1.0],
                    "freq": freq,
                }
                if window is not None:
                    record["window"] = window
                records.append(record)
    return records


def member_manifest(windows=None, **fields):
    manifest = {
        "model_class": "OLSModel",
        "predictors": ["load_d+1"],
        "target": "price",
        "horizon": 1,
        "freq": "1h",
        "training_window": 30,
        "data_columns_hash": "same data",
        "data_timezone": "Europe/Warsaw",
        "test_start": windows.first_day if windows else "2024-05-01",
        "test_end": windows.last_day if windows else "2024-05-12",
        "column_hashes": {"price": "one target", "load_d+1": "one load"},
        "frame_hash": "one frame",
        "package_version": "2.3.0",
        "git_sha": None,
        "created_at": "then",
        "updated_at": "then",
    }
    if windows:
        manifest["windows"] = windows.to_dict()
    return {**manifest, **fields}


def write_member(directory, name, records, windows=None, **fields):
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = Path(directory) / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    manifest_path(path).write_text(json.dumps(member_manifest(windows, **fields)))
    return path


def write_members(directory, windows=HAND, hours=HOURS, horizons=(1,), **errors):
    """Members over every day of `windows`, each off the actual by the error given under its name."""
    days = windows.issue_days()
    for name, error in errors.items():
        write_member(directory, name, forecasts(days, error, hours, horizons, windows), windows, horizon=max(horizons))


def reporting(*ensembles, members=("a", "b", "c")):
    return built(*[OLSModel(["x"], name=name) for name in members], *ensembles)


def report_of(directory, *ensembles, members=("a", "b", "c"), horizon=1, **kwargs):
    return reporting(*ensembles, members=members).report(directory, horizon=horizon, **kwargs)


def choice_of(report, name="e", segment=0):
    table = report.choices()
    return table[(table["ensemble"] == name) & (table["segment"] == segment)].iloc[0]


# ---------------------------------------------------------------------------------------------------------------------
# The average, bit for bit
# ---------------------------------------------------------------------------------------------------------------------


def test_a_one_member_average_is_the_member(df, tmp_path):
    report = built(ols("a"), AverageEnsemble(["a"], name="e")).run(df, test=T, horizon=2, save_dir=str(tmp_path))
    assert forecast_lines(stored(tmp_path / "e.jsonl")) == forecast_lines(stored(tmp_path / "a.jsonl"))
    assert forecast_lines(records_of(report, "e")) == forecast_lines(records_of(report, "a"))
    windowed = built(ols("a"), AverageEnsemble(["a"], name="e")).run(df, validation=V, test=T, horizon=2)
    assert forecast_lines(records_of(windowed, "e")) == forecast_lines(records_of(windowed, "a"))
    assert forecast_lines(records_of(windowed.validation, "e")) == forecast_lines(records_of(windowed.validation, "a"))


def test_the_average_of_two_is_their_sum_over_two_and_of_three_their_np_mean(df):
    report = built(*full_small_wind(), AverageEnsemble(["full", "small"], name="two")).run(df, test=T, horizon=2)
    full, small = keyed(records_of(report, "full")), keyed(records_of(report, "small"))
    two = keyed(records_of(report, "two"))
    assert set(two) == set(full) == set(small) and len(two) == 3 * DAY * 2
    for key, record in two.items():
        assert record["prediction"] == (full[key]["prediction"] + small[key]["prediction"]) / 2
        assert record["actual"] == full[key]["actual"]

    report = built(*full_small_wind(), AverageEnsemble(["wind", "full", "small"], name="three")).run(
        df, test=T, horizon=1
    )
    full, small, wind = (keyed(records_of(report, name)) for name in ("full", "small", "wind"))
    for key, record in keyed(records_of(report, "three")).items():
        by_hand = np.mean([wind[key]["prediction"], full[key]["prediction"], small[key]["prediction"]])
        assert record["prediction"] == by_hand


def test_top_k_equals_the_plain_average_of_what_it_chose_on_test(df):
    members = ["wind", "small", "full", "no_weekend"]
    models = [*full_small_wind(), ols("no_weekend", ["load_d+{horizon}", "wind_d+{horizon}", "is_weekday_d+{horizon}"])]
    chosen = built(*models, AverageEnsemble(members, top=3, name="top")).run(df, validation=V, test=T, horizon=1)
    row = choice_of(chosen, "top")
    assert row["chosen"] == [m for m in members if m in row["chosen"]] and row["k"] == 3
    # The ranking is not member order here, so an average in ranking order would differ in the last bit somewhere.
    scores = chosen.choices().attrs["member_mae"]["top"][0]
    assert sorted(row["chosen"], key=scores.get) != row["chosen"]
    plain = built(*models, AverageEnsemble(row["chosen"], name="top")).run(df, validation=V, test=T, horizon=1)
    assert forecast_lines(records_of(chosen, "top")) == forecast_lines(records_of(plain, "top"))
    assert {r["run_date"] for r in records_of(chosen, "top")} == set(RunWindows(test=T).issue_days())


def test_the_average_is_in_member_order_whatever_the_ranking(tmp_path):
    """0.3 + (0.2 + 0.1) and 0.1 + (0.2 + 0.3) differ in the last bit, so this sees the order of the mean."""
    days = HAND.issue_days()
    off = {"a": 0.3, "b": 0.2, "c": 0.1}
    for name, value in off.items():
        records = [{**r, "prediction": value, "actual": 0.0} for r in forecasts(days, windows=HAND)]
        write_member(tmp_path, name, records, HAND)
    report = report_of(tmp_path, AverageEnsemble(["a", "b", "c"], top=3, name="e"))
    assert choice_of(report)["chosen"] == ["a", "b", "c"]
    assert report.choices().attrs["member_mae"]["e"][0] == pytest.approx(off)
    assert float(np.mean([0.3, 0.2, 0.1])) != float(np.mean([0.1, 0.2, 0.3]))
    assert {r["prediction"] for r in records_of(report, "e")} == {float(np.mean([0.3, 0.2, 0.1]))}


def test_an_ensembles_record_has_the_keys_of_a_models(tmp_path):
    write_members(tmp_path, a=1.0, b=2.0)
    report_of(tmp_path, AverageEnsemble(["a", "b"], name="plain"), members=("a", "b"))
    model, ensemble = stored(tmp_path / "a.jsonl")[0], stored(tmp_path / "plain.jsonl")[0]
    assert list(ensemble) == list(model)
    assert ensemble["coefficients"] == []
    assert {k: v for k, v in ensemble.items() if k not in ("prediction", "coefficients")} == {
        k: v for k, v in model.items() if k not in ("prediction", "coefficients")
    }


# ---------------------------------------------------------------------------------------------------------------------
# Choosing: top, ties, what counts
# ---------------------------------------------------------------------------------------------------------------------


def test_top_keeps_the_members_with_the_lowest_validation_mae(tmp_path):
    write_members(tmp_path, a=3.0, b=1.0, c=2.0)
    report = report_of(
        tmp_path,
        AverageEnsemble(["a", "b", "c"], top=1, name="one"),
        AverageEnsemble(["a", "b", "c"], top=2, name="two"),
        AverageEnsemble(["a", "b", "c"], name="all"),
    )
    assert choice_of(report, "one")["chosen"] == ["b"]
    assert choice_of(report, "one")["k"] == 1
    assert choice_of(report, "one")["validation_mae"] == 1.0
    assert choice_of(report, "two")["chosen"] == ["b", "c"]
    assert choice_of(report, "two")["validation_mae"] == 1.5
    assert list(report.choices()["ensemble"]) == ["one", "two"], "a plain average chooses nothing"
    assert choice_of(report, "one")["validation_forecasts"] == SERVED
    assert report.choices().attrs["member_mae"]["two"][0] == {"a": 3.0, "b": 1.0, "c": 2.0}
    row = choice_of(report, "two")
    assert (row["validation_start"], row["validation_end"], row["test_start"], row["test_end"]) == (*HV, *HT)
    assert all(r["prediction"] == r["actual"] + 1.0 for r in records_of(report, "one"))


def test_top_validation_chooses_k_on_the_average_of_the_best_k(tmp_path):
    # b and c err in opposite directions, so their average beats both; a is the third best and would spoil it.
    write_members(tmp_path, a=3.0, b=1.0, c=-1.0)
    report = report_of(tmp_path, AverageEnsemble(["a", "b", "c"], top="validation", name="e"))
    row = choice_of(report)
    assert row["chosen"] == ["b", "c"] and row["k"] == 2 and row["validation_mae"] == 0.0
    assert report.choices().attrs["k_mae"]["e"][0] == {"1": 1.0, "2": 0.0, "3": 1.0}
    assert all(r["prediction"] == r["actual"] for r in records_of(report, "e"))


def test_a_tie_in_ranking_goes_to_the_earlier_member_and_in_k_to_the_smaller(tmp_path):
    write_members(tmp_path, a=1.0, b=-1.0, c=1.0)
    report = report_of(tmp_path, AverageEnsemble(["c", "b", "a"], top=1, name="first"))
    assert choice_of(report, "first")["chosen"] == ["c"]
    report = report_of(tmp_path, AverageEnsemble(["a", "b", "c"], top=1, name="first"))
    assert choice_of(report, "first")["chosen"] == ["a"]
    report = report_of(tmp_path, AverageEnsemble(["c", "a"], top="validation", name="smaller"))
    assert choice_of(report, "smaller")["chosen"] == ["c"], "k=1 and k=2 tie, and the smaller k wins"
    assert choice_of(report, "smaller")["k"] == 1


def test_identical_members_tie_through_the_noise_of_averaging():
    """Averaging identical members gives another float, by rounding noise alone, sometimes a lower MAE. That is still a tie."""
    ensemble = AverageEnsemble(["a", "b", "c"], top="validation")
    noise_won = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        actual = rng.normal(50, 20, 240)
        prediction = actual + rng.normal(0, 5, 240)
        choice = ensemble.choose(np.column_stack([prediction] * 3), actual)
        noise_won += choice.k_mae[3] < choice.k_mae[1]
        assert choice.chosen == ["a"]
    assert noise_won, "this needs noise that would have won to mean anything"


def test_a_difference_within_twelve_significant_digits_is_a_tie_and_beyond_them_is_not():
    actual = np.zeros(4)
    ensemble = AverageEnsemble(["a", "b"], top=1)
    assert ensemble.choose(np.column_stack([np.full(4, 1.0), np.full(4, 1.0 - 1e-14)]), actual).chosen == ["a"]
    assert ensemble.choose(np.column_stack([np.full(4, 1.0), np.full(4, 1.0 - 1e-11)]), actual).chosen == ["b"]
    assert ensemble.choose(np.column_stack([np.full(4, 1000.0), np.full(4, 1000.0 - 1e-11)]), actual).chosen == ["a"]
    # The last digit that counts and the first that doesn't.
    assert ensemble.choose(np.column_stack([np.full(4, 1.0), np.full(4, 1.0 - 1e-12)]), actual).chosen == ["b"]
    assert ensemble.choose(np.column_stack([np.full(4, 1.0), np.full(4, 1.0 - 1e-13)]), actual).chosen == ["a"]


def test_every_k_is_scored_on_its_average_in_member_order():
    """0.3 + (0.2 + 0.1) and 0.1 + (0.2 + 0.3) differ in the last bit, and the ranking here is the reverse of member order."""
    choice = AverageEnsemble(["a", "b", "c"], top="validation").choose(np.array([[0.3, 0.2, 0.1]]), np.zeros(1))
    assert choice.k_mae[3] == float(np.mean([0.3, 0.2, 0.1])) != float(np.mean([0.1, 0.2, 0.3]))


def test_validation_only_gives_a_row_with_no_test_period_and_no_forecasts(tmp_path):
    windows = RunWindows.fixed(HV, None)
    # With no test period known, even forecasts for days after the validation period count.
    days = windows.issue_days()
    for name, error in (("a", 2.0), ("b", 1.0)):
        write_member(tmp_path, name, forecasts(days, error, horizons=(1, 2), windows=windows), windows, horizon=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = report_of(
            tmp_path,
            AverageEnsemble(["a", "b"], top=1, name="e"),
            AverageEnsemble(["e", "b"], name="outer"),
            AverageEnsemble(["a", "b"], name="plain"),
            members=("a", "b"),
            horizon=2,
        )
    assert not [w for w in caught if "no forecasts" in str(w.message)]
    row = choice_of(report)
    assert row["chosen"] == ["b"] and row["validation_forecasts"] == 8 * DAY * 2
    assert row["test_start"] is None and row["test_end"] is None
    assert report.window == "validation"
    assert set(report.refs) == {"a", "b", "plain"}, "no forecast to show, not a row of zeros in every table"
    assert stored(tmp_path / "e.jsonl") == []
    assert json.loads((tmp_path / "e.manifest.json").read_text())["choices"][0]["test"] is None


def test_validation_only_through_a_run_and_then_the_test_comes(df, tmp_path):
    make = lambda: built(*full_small_wind(), AverageEnsemble(["full", "small"], top=1, name="e"))  # noqa: E731
    report = make().run(df, validation=V, horizon=1, save_dir=str(tmp_path))
    assert report.window == "validation" and "e" not in report.refs
    first = choice_of(report)
    assert first["test_start"] is None and first["validation_forecasts"] == 4 * DAY
    report = make().run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))
    assert choice_of(report)["test_start"] == T[0]
    assert {r["run_date"] for r in records_of(report, "e")} == set(RunWindows(test=T).issue_days())


def test_validation_test_and_top_k_ensemble_on_identical_keys(df):
    def by_hour(ctx):
        return PREDICTORS if ctx.period < 12 else SMALL

    pipeline = built(
        ols("full"),
        ols("no_wind", SMALL),
        ols("dyn", by_hour),
        AverageEnsemble(["full", "no_wind", "dyn"], top=2, by="validation", name="top2"),
        info=INFO,
    ).add_evaluator(MAEEvaluator())
    report = pipeline.run(df, horizon=1, validation=V, test=T)
    assert set(report.refs) == {"full", "no_wind", "dyn", "top2"}
    validation = report.validation.leaderboard().sort_values("MAE").model.tolist()
    assert set(choice_of(report, "top2")["chosen"]) == set(validation[:2])
    assert choice_of(report, "top2")["validation_forecasts"] == 4 * DAY, (
        "days lie between the windows, so all of it serves"
    )
    assert all(r["run_date"] >= T[0] for r in records_of(report, "top2"))
    board = report.leaderboard(benchmark="full")
    assert (board["n_common"] == 3 * DAY).all()
    counts = counts_of(report)
    assert counts.loc["top2", "planned"] == counts.loc["full", "planned"] == 3 * DAY
    assert counts.loc["top2", "produced"] == 3 * DAY


def test_ensemble_does_not_rank_on_targets_inside_the_test_period(df):
    pipeline = built(ols("a"), ols("b", SMALL), AverageEnsemble(["a", "b"], top=1, name="top1"))
    report = pipeline.run(df, horizon=2, validation=("2023-03-28", "2023-04-02"), test=T)
    # Of the 6 validation days, what has its target before the first test issue day: 5 days at horizon 1, 4 at horizon 2.
    assert choice_of(report, "top1")["validation_forecasts"] == (5 + 4) * DAY


# ---------------------------------------------------------------------------------------------------------------------
# No look-ahead in choosing
# ---------------------------------------------------------------------------------------------------------------------


def perturb(directory, names, when, by):
    """Move the `actual` of every record `when(record)` says, in every member, as new data would."""
    for name in names:
        path = Path(directory) / f"{name}.jsonl"
        records = stored(path)
        for r in records:
            if when(r):
                r["actual"] += by
        path.write_text("".join(json.dumps(r) + "\n" for r in records))


def is_for_the_first_test_day(r):
    """Issued in validation for the first test issue day: on the last validation day at horizon 1, the one before at 2."""
    return r["run_date"] <= HV[1] and r["target_date"] == HT[0]


def members_a_moved_actual_can_swing(directory):
    """`a` errs 1 and `b` 1.5, but `b` 1000 on the keys the next tests move, where it comes out right if their actuals move
    by 1000 and `a` then errs 999: enough to win the whole validation period for `b`, if they count."""

    def b_error(run_date, target_date, hour, horizon):
        far_off = run_date == HV[0] or (run_date <= HV[1] and target_date >= HT[0])
        return 1000.0 if far_off else 1.5

    write_members(directory, horizons=(1, 2), a=1.0, b=b_error)


def two_day_ahead_choice(directory):
    return choice_of(report_of(directory, AverageEnsemble(["a", "b"], top=1, name="e"), members=("a", "b"), horizon=2))


def test_a_validation_issue_day_with_its_target_on_the_first_test_issue_day_does_not_count(tmp_path):
    """The one case a `<=` would let through: issued in validation, for the first test issue day exactly."""
    members_a_moved_actual_can_swing(tmp_path)
    before = two_day_ahead_choice(tmp_path)
    assert before["chosen"] == ["a"]
    assert before["validation_forecasts"] == (7 + 6) * DAY
    perturb(tmp_path, ("a", "b"), is_for_the_first_test_day, by=1000.0)
    after = two_day_ahead_choice(tmp_path)
    assert after["chosen"] == ["a"]
    assert after["validation_mae"] == before["validation_mae"] == 1.0
    assert after["validation_forecasts"] == before["validation_forecasts"]


def test_actuals_from_the_first_test_issue_day_on_never_touch_the_choice(tmp_path):
    members_a_moved_actual_can_swing(tmp_path)
    before = two_day_ahead_choice(tmp_path)
    perturb(tmp_path, ("a", "b"), lambda r: r["target_date"] >= HT[0], by=1000.0)
    after = two_day_ahead_choice(tmp_path)
    assert after["chosen"] == ["a"]
    assert after["validation_mae"] == before["validation_mae"]
    assert after["validation_forecasts"] == before["validation_forecasts"]


def test_the_control_an_earlier_actual_does_change_the_choice(tmp_path):
    """Otherwise the two tests above could pass because nothing ever moved the choice."""
    members_a_moved_actual_can_swing(tmp_path)
    assert two_day_ahead_choice(tmp_path)["chosen"] == ["a"]
    perturb(tmp_path, ("a", "b"), lambda r: r["run_date"] == HV[0], by=1000.0)
    assert two_day_ahead_choice(tmp_path)["chosen"] == ["b"]


def test_no_look_ahead_on_a_real_run(df):
    """Prices from the first test issue day on change what the run stores as `actual`, and must leave the choice as it was."""
    validation = ("2023-03-28", "2023-04-02")
    make = lambda: built(ols("a"), ols("b", SMALL), ols("c", WIND), AverageEnsemble(["a", "b", "c"], top=2, name="e"))  # noqa: E731
    zone = df.index.tz
    before = make().run(df, validation=validation, test=T, horizon=2)
    moved = df.copy()
    moved.loc[moved.index >= pd.Timestamp(T[0], tz=zone), "price"] += 500.0
    after = make().run(moved, validation=validation, test=T, horizon=2)
    assert choice_of(after)["chosen"] == choice_of(before)["chosen"]
    assert choice_of(after)["validation_mae"] == choice_of(before)["validation_mae"]
    assert before.choices().attrs["member_mae"] == after.choices().attrs["member_mae"]
    issued_in_validation = [r for r in after.refs["a"]._results if r["run_date"] < T[0] and r["target_date"] >= T[0]]
    assert issued_in_validation, "there are such forecasts"
    assert all(r["actual"] > 300 for r in issued_in_validation), "and the perturbation reached them"

    control = df.copy()
    control.loc[(control.index >= pd.Timestamp(V[1], tz=zone)) & (control.index < pd.Timestamp(T[0], tz=zone)), "price"] += 500.0  # fmt: skip
    assert choice_of(make().run(control, validation=validation, test=T, horizon=2))["validation_mae"] != choice_of(before)["validation_mae"]  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# previous_year
# ---------------------------------------------------------------------------------------------------------------------


def yearly_members(directory, test, a_better_in, hours=(0,)):
    """`a` errs 1 and `b` 2 on the validation days of the segments `a_better_in` names, and the other way round elsewhere."""
    windows = RunWindows.of_previous_year(test)

    def segment_of(run_date):
        return next((i for i, s in enumerate(windows.segments) if s.validation[0] <= run_date <= s.validation[1]), None)

    def error_a(run_date, target_date, hour, horizon):
        return 1.0 if segment_of(run_date) in a_better_in else 2.0

    def error_b(run_date, target_date, hour, horizon):
        return 2.0 if segment_of(run_date) in a_better_in else 1.0

    days = windows.issue_days()
    write_member(directory, "a", forecasts(days, error_a, hours, windows=windows), windows)
    write_member(directory, "b", forecasts(days, error_b, hours, windows=windows), windows)
    return windows


def test_previous_year_chooses_per_segment_and_a_forecast_uses_its_own_segments_choice(tmp_path):
    test = ("2024-01-05", "2025-12-26")
    windows = yearly_members(tmp_path, test, a_better_in={0})
    assert len(windows.segments) == 2
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], top=1, name="e"), members=("a", "b"))
    first, second = choice_of(report, segment=0), choice_of(report, segment=1)
    assert first["chosen"] == ["a"] and second["chosen"] == ["b"]
    assert (first["validation_start"], first["test_start"], first["test_end"]) == (
        "2023-01-05",
        "2024-01-05",
        "2025-01-04",
    )
    assert (second["validation_start"], second["validation_end"]) == ("2024-01-05", "2025-01-04")
    assert (second["test_start"], second["test_end"]) == ("2025-01-05", "2025-12-26")

    a, b = keyed(stored(tmp_path / "a.jsonl")), keyed(stored(tmp_path / "b.jsonl"))
    records = stored(tmp_path / "e.jsonl")
    assert {r["window"] for r in records} == {"test"}
    assert {r["run_date"] for r in records} == set(RunWindows(test=test).issue_days())
    for key, r in keyed(records).items():
        member = a if r["run_date"] <= "2025-01-04" else b
        assert r["prediction"] == member[key]["prediction"]
    counts = counts_of(report)
    assert counts.loc["e", "planned"] == counts.loc["a", "planned"]
    assert len(records) == counts.loc["e", "produced"]


def test_a_later_segments_validation_days_are_tagged_test_and_count_for_its_choices(tmp_path):
    test = ("2024-01-05", "2025-12-26")
    windows = yearly_members(tmp_path, test, a_better_in={0})
    second = windows.segments[1]
    serving = [r for r in stored(tmp_path / "a.jsonl") if second.serves(r["run_date"], r["target_date"])]
    assert {r["window"] for r in serving} == {"test"}
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], top=1, name="e"), members=("a", "b"))
    assert choice_of(report, segment=1)["validation_forecasts"] == len(serving)
    # The days tagged "validation" belong to the first segment, and they would have chosen `a`.
    assert choice_of(report, segment=1)["chosen"] == ["b"]


def test_a_top_validation_ensemble_chooses_k_per_segment(tmp_path):
    windows = RunWindows.of_previous_year(("2024-01-05", "2025-12-26"))
    days = windows.issue_days()
    first_year_end = windows.segments[0].validation[1]
    write_member(tmp_path, "a", forecasts(days, 1.0, (0,), windows=windows), windows)
    # In the first year b cancels a's error; afterwards it adds to it.
    b_error = lambda run_date, *_: -1.5 if run_date <= first_year_end else 1.5  # noqa: E731
    write_member(tmp_path, "b", forecasts(days, b_error, (0,), windows=windows), windows)
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], top="validation", name="e"), members=("a", "b"))
    assert choice_of(report, segment=0)["k"] == 2
    assert choice_of(report, segment=1)["k"] == 1


def test_previous_year_from_a_leap_day(tmp_path):
    test = ("2024-02-29", "2026-03-10")
    windows = yearly_members(tmp_path, test, a_better_in={0, 2})
    assert [segment.test[0] for segment in windows.segments] == ["2024-02-29", "2025-02-28", "2026-02-28"]
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], top=1, name="e"), members=("a", "b"))
    assert list(report.choices()["chosen"]) == [["a"], ["b"], ["a"]]
    assert list(report.choices()["validation_start"]) == ["2023-02-28", "2024-02-29", "2025-02-28"]
    assert list(report.choices()["validation_end"]) == ["2024-02-28", "2025-02-27", "2026-02-27"]
    a, b = keyed(stored(tmp_path / "a.jsonl")), keyed(stored(tmp_path / "b.jsonl"))
    for key, r in keyed(stored(tmp_path / "e.jsonl")).items():
        member = b if "2025-02-28" <= r["run_date"] <= "2026-02-27" else a
        assert r["prediction"] == member[key]["prediction"]


def test_previous_year_through_a_real_run(tmp_path):
    frame = make_market_data(start="2022-01-01", days=500, horizon=2, noise=5.0, seed=3)
    test = ("2023-03-10", "2023-03-11")
    make = lambda: built(  # noqa: E731
        MeanModel(PREDICTORS, training_window=14, name="short"),
        MeanModel(PREDICTORS, training_window=28, name="long"),
        AverageEnsemble(["short", "long"], top=1, name="e"),
    )
    report = make().run(frame, validation="previous_year", test=test, horizon=1, save_dir=str(tmp_path))
    row = choice_of(report)
    assert (row["validation_start"], row["validation_end"]) == ("2022-03-10", "2023-03-09")
    assert (row["test_start"], row["test_end"]) == test
    served = [r for r in stored(tmp_path / "short.jsonl") if r["target_date"] < test[0] and np.isfinite(r["actual"])]
    assert row["validation_forecasts"] == len(served) > 360 * DAY
    assert {r["run_date"] for r in records_of(report, "e")} == {"2023-03-10", "2023-03-11"}
    chosen = keyed(records_of(report, row["chosen"][0]))
    assert all(r["prediction"] == chosen[key_of(r)]["prediction"] for r in records_of(report, "e"))
    assert report.validation.windows.mode == "previous_year"
    assert "e" not in report.validation.refs


# ---------------------------------------------------------------------------------------------------------------------
# Identical keys and counts
# ---------------------------------------------------------------------------------------------------------------------


def test_member_gaps_are_missing_for_the_ensemble_and_counted(tmp_path):
    days = HAND.issue_days()
    write_member(tmp_path, "a", forecasts(days, 1.0, windows=HAND), HAND)
    held = lambda r: not (r["run_date"] in (HV[0], HT[0]) and r["hour"] == 1)  # noqa: E731
    write_member(tmp_path, "b", [r for r in forecasts(days, 3.0, windows=HAND) if held(r)], HAND)
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))
    counts = counts_of(report)
    assert counts.loc["e", "member_missing"] == 1
    assert counts.loc["a", "member_missing"] == counts.loc["b", "member_missing"] == 0
    assert list(counts.loc["e", ["planned", "produced", "missing"]]) == [4 * DAY, 4 * DAY - 1, 1]
    validation = counts_of(report.validation)
    assert validation.loc["e", "member_missing"] == 1
    assert list(validation.loc["e", ["planned", "produced", "target_in_test"]]) == [8 * DAY, SERVED - 1, DAY]
    assert keyed(records_of(report, "e")).keys() == keyed(records_of(report, "b")).keys()
    assert all(r["prediction"] == r["actual"] + 2.0 for r in records_of(report, "e")), "never averaged over the rest"


def test_member_gaps_are_counted_in_a_plain_run_too(tmp_path):
    days = days_of(*HT)
    write_member(tmp_path, "a", forecasts(days, 1.0))
    write_member(tmp_path, "b", [r for r in forecasts(days, 3.0) if not (r["run_date"] == HT[0] and r["hour"] == 1)])
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"), test=HT)
    counts = counts_of(report)
    assert counts.loc["e", "member_missing"] == 1
    assert list(counts.loc["e", ["planned", "produced", "missing"]]) == [4 * DAY, 4 * DAY - 1, 1]


@pytest.mark.parametrize("windows", [HAND, None], ids=["windows", "plain"])
def test_a_report_told_a_shorter_horizon_than_the_results_plans_and_reads_that_one(tmp_path, windows):
    """The members and the ensemble plan what the report is told, and read only that: the members' records of horizons 2 and 3
    are no forecasts of it."""
    days = windows.issue_days() if windows else days_of(*HT)
    write_member(tmp_path, "a", forecasts(days, 1.0, horizons=(1, 2, 3), windows=windows), windows, horizon=3)
    gap = lambda r: not (r["run_date"] == HT[0] and r["hour"] == 1)  # noqa: E731
    held = [r for r in forecasts(days, 3.0, horizons=(1, 2, 3), windows=windows) if gap(r)]
    write_member(tmp_path, "b", held, windows, horizon=3)
    report = report_of(
        tmp_path,
        AverageEnsemble(["a", "b"], name="e"),
        members=("a", "b"),
        horizon=1,
        **({} if windows else {"test": HT}),
    )
    counts = counts_of(report)
    assert counts.loc["e", "member_missing"] == 1
    assert list(counts.loc["e", ["planned", "produced"]]) == [4 * DAY, 4 * DAY - 1]
    assert list(counts.loc[["a", "b"], "produced"]) == [4 * DAY, 4 * DAY - 1]
    assert set(report.by_horizon()["horizon"]) == {1}


def test_a_key_every_member_lacks_is_missing_too(tmp_path):
    """Planned and produced add up with the reason, as for a model: 02:00 on a spring switch day, say."""
    days = HAND.issue_days()
    held = lambda r: not (r["run_date"] == HT[0] and r["hour"] == 2)  # noqa: E731
    for name in ("a", "b"):
        write_member(tmp_path, name, [r for r in forecasts(days, windows=HAND) if held(r)], HAND)
    counts = counts_of(report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b")))
    assert counts.loc["e", "member_missing"] == 1
    assert counts.loc["e", "planned"] == counts.loc["e", "produced"] + counts.loc["e", "member_missing"]


def test_a_key_missing_for_an_unchosen_member_is_no_gap_on_test_but_shortens_choosing(tmp_path):
    days = HAND.issue_days()
    write_member(tmp_path, "a", forecasts(days, 1.0, windows=HAND), HAND)
    held = lambda r: not (r["hour"] == 1 and r["run_date"] in (HV[0], "2024-05-02", HT[0]))  # noqa: E731
    write_member(tmp_path, "weak", [r for r in forecasts(days, 9.0, windows=HAND) if held(r)], HAND)
    report = report_of(tmp_path, AverageEnsemble(["a", "weak"], top=1, name="e"), members=("a", "weak"))
    assert choice_of(report)["chosen"] == ["a"]
    assert choice_of(report)["validation_forecasts"] == SERVED - 2, "the common keys are over all the members"
    counts = counts_of(report)
    assert counts.loc["e", "produced"] == 4 * DAY and "member_missing" not in counts
    assert keyed(records_of(report, "e")).keys() == keyed(records_of(report, "a")).keys()


def test_a_gap_in_a_chosen_member_is_missing_on_test(tmp_path):
    days = HAND.issue_days()
    write_member(tmp_path, "a", forecasts(days, 9.0, windows=HAND), HAND)
    write_member(tmp_path, "b", [r for r in forecasts(days, 1.0, windows=HAND) if r["run_date"] != HT[0]], HAND)
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], top=1, name="e"), members=("a", "b"))
    assert choice_of(report)["chosen"] == ["b"]
    counts = counts_of(report)
    assert counts.loc["e", "produced"] == 3 * DAY
    assert counts.loc["e", "member_missing"] == DAY
    assert HT[0] not in {r["run_date"] for r in records_of(report, "e")}


def test_a_non_finite_prediction_is_missing_for_that_key(tmp_path):
    days = HAND.issue_days()
    write_member(tmp_path, "a", forecasts(days, 1.0, windows=HAND), HAND)
    b = forecasts(days, 1.0, windows=HAND)
    broken = [r for r in b if r["run_date"] in (HV[0], HT[0]) and r["hour"] == 0]
    broken[0]["prediction"], broken[1]["prediction"] = float("nan"), float("inf")
    write_member(tmp_path, "b", b, HAND)
    report = report_of(
        tmp_path,
        AverageEnsemble(["a", "b"], name="plain"),
        AverageEnsemble(["a", "b"], top=2, name="top"),
        members=("a", "b"),
    )
    counts = counts_of(report)
    assert counts.loc["plain", "produced"] == counts.loc["top", "produced"] == 4 * DAY - 1
    assert counts.loc["plain", "member_missing"] == counts.loc["top", "member_missing"] == 1
    assert choice_of(report, "top")["validation_forecasts"] == SERVED - 1
    assert counts_of(report.validation).loc["plain", "produced"] == SERVED - 1
    assert all(np.isfinite(r["prediction"]) for r in stored(tmp_path / "plain.jsonl"))


def test_ensemble_ranking_ignores_keys_with_a_nan_actual(tmp_path):
    days = HAND.issue_days()
    for name, error in (("weak", 10.0), ("strong", 1.0)):
        records = forecasts(days, error, windows=HAND)
        for r in records:
            if (r["run_date"], r["hour"]) in ((HV[0], 0), (HT[0], 0)):
                r["actual"] = float("nan")
        write_member(tmp_path, name, records, HAND)
    report = report_of(
        tmp_path,
        AverageEnsemble(["weak", "strong"], top=1, name="e"),
        AverageEnsemble(["weak", "strong"], name="plain"),
        members=("weak", "strong"),
    )
    assert choice_of(report)["chosen"] == ["strong"]
    assert choice_of(report)["validation_forecasts"] == SERVED - 1
    assert choice_of(report)["validation_mae"] == 1.0
    assert report.choices().attrs["member_mae"]["e"][0] == {"weak": 10.0, "strong": 1.0}
    held = {(r["run_date"], r["hour"]): r for r in records_of(report, "e")}
    assert np.isnan(held[HT[0], 0]["actual"]) and np.isfinite(held[HT[0], 0]["prediction"])
    assert len(held) == 4 * DAY, "a NaN actual stays a forecast, as for a model"
    assert report.summary().set_index("model").loc["e", "forecasts"] == 4 * DAY
    assert report.validation.summary().set_index("model").loc["plain", "forecasts"] == SERVED


def test_the_ranking_is_on_the_common_keys_and_not_on_each_members_own(tmp_path):
    """`a` looks worse on its own keys because it alone forecast a very bad stretch. On the common keys it is better."""
    days = HAND.issue_days()
    write_member(tmp_path, "a", forecasts(days, lambda d, *_: 50.0 if d == HV[0] else 1.0, windows=HAND), HAND)
    write_member(tmp_path, "b", [r for r in forecasts(days, 2.0, windows=HAND) if r["run_date"] != HV[0]], HAND)
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], top=1, name="e"), members=("a", "b"))
    assert choice_of(report)["chosen"] == ["a"]
    assert report.choices().attrs["member_mae"]["e"][0] == {"a": 1.0, "b": 2.0}


def thin(tmp_path, a_days, b_days, windows, hours=(0, 1, 2)):
    """Results of `a` and `b` on those validation days, and on every test day."""
    tested = windows.periods[-1][1]
    days = [*a_days, *days_of(*tested)]
    write_member(tmp_path, "a", forecasts(days, 1.0, hours, windows=windows), windows)
    days = [*b_days, *days_of(*tested)]
    write_member(tmp_path, "b", forecasts(days, 2.0, hours, windows=windows), windows)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = report_of(tmp_path, AverageEnsemble(["a", "b"], top=1, name="e"), members=("a", "b"))
    return report, caught


GAPPED = RunWindows.fixed(("2024-01-01", "2024-03-10"), ("2024-03-20", "2024-03-22"))
ALL_DAYS = days_of(*GAPPED.validation)


def test_limited_by_lists_who_lacks_most_first(tmp_path):
    windows = GAPPED
    tested = days_of(*windows.test)
    records = lambda days, error: forecasts([*days, *tested], error, (0, 1, 2), windows=windows)  # noqa: E731
    write_member(tmp_path, "a", records(ALL_DAYS, 1.0), windows)
    write_member(tmp_path, "z", records(ALL_DAYS[2:], 2.0), windows)
    write_member(tmp_path, "b", records(ALL_DAYS[5:], 3.0), windows)
    write_member(tmp_path, "c", records(ALL_DAYS[:-2], 4.0), windows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = report_of(
            tmp_path, AverageEnsemble(["a", "z", "b", "c"], top=1, name="e"), members=("a", "z", "b", "c")
        )
    limited_by = report.choices().attrs["limited_by"]["e"][0]
    assert list(limited_by.items()) == [("b", 15), ("c", 6), ("z", 6)]


def test_a_full_validation_has_a_share_of_one_and_says_nothing(tmp_path):
    report, caught = thin(tmp_path, ALL_DAYS, ALL_DAYS, GAPPED)
    row = choice_of(report)
    assert row["validation_share"] == 1.0 and row["validation_forecasts"] == len(ALL_DAYS) * 3
    assert report.choices().attrs["limited_by"]["e"][0] == {}
    assert not warned(caught, "was made on")


def test_a_members_holes_show_in_the_share_with_who_made_them(tmp_path):
    report, caught = thin(tmp_path, ALL_DAYS, ALL_DAYS[: len(ALL_DAYS) // 2], GAPPED)
    row = choice_of(report)
    assert row["validation_forecasts"] == len(ALL_DAYS) // 2 * 3
    assert row["validation_share"] == row["validation_forecasts"] / (len(ALL_DAYS) * 3)
    assert report.choices().attrs["limited_by"]["e"][0] == {"b": (len(ALL_DAYS) - len(ALL_DAYS) // 2) * 3}
    shown = warned(caught, "of the")
    assert len(shown) == 1 and f"{row['validation_forecasts']} forecasts" in shown[0] and "'b'" in shown[0]


def test_the_share_is_of_the_fullest_member_and_not_of_all_that_any_member_holds(tmp_path):
    report, _ = thin(tmp_path, ALL_DAYS[:60], ALL_DAYS[10:], GAPPED)
    assert choice_of(report)["validation_forecasts"] == 50 * 3
    assert choice_of(report)["validation_share"] == 50 / 60
    assert report.choices().attrs["limited_by"]["e"][0] == {"a": 10 * 3, "b": 10 * 3}


@pytest.mark.parametrize("lost, warns", [(7, False), (8, True)])
def test_the_share_warns_below_ninety_percent(tmp_path, lost, warns):
    """70 days: 7 lost leave exactly 0.9."""
    _, caught = thin(tmp_path, ALL_DAYS, ALL_DAYS[lost:], GAPPED)
    assert len(ALL_DAYS) == 70 and MIN_VALIDATION_SHARE == 0.9
    assert bool(warned(caught, "of the")) is warns


@pytest.mark.parametrize("count, warns", [(28, False), (27, True)])
def test_the_target_days_warn_below_twenty_eight(tmp_path, count, warns):
    first = "2024-01-01"
    validation = (first, plus_days(first, count - 1))
    windows = RunWindows.fixed(validation, ("2024-03-20", "2024-03-22"))
    days = days_of(*validation)
    report, caught = thin(tmp_path, days, days, windows)
    assert MIN_VALIDATION_DAYS == 28
    found = warned(caught, "target days")
    assert bool(found) is warns
    if warns:
        assert f"{count} target days ({count * 3} forecasts)" in found[0]
    assert choice_of(report)["validation_share"] == 1.0
    # One forecast a day: as many issue days as target days, and the same rule.
    assert MIN_VALIDATION_ISSUES == 28
    issued = warned(caught, "issued on")
    assert bool(issued) is warns
    if warns:
        assert f"issued on {count} days ({count * 3} forecasts)" in issued[0]


def test_distinct_target_days_are_counted_not_forecasts(tmp_path):
    """Many horizons and periods on few days are still few days."""
    windows = RunWindows.fixed(("2024-01-01", "2024-01-10"), ("2024-03-20", "2024-03-22"))
    write_members(tmp_path, windows=windows, horizons=(1, 2, 3), a=1.0, b=2.0)
    with pytest.warns(UserWarning, match="12 target days"):
        report_of(tmp_path, AverageEnsemble(["a", "b"], top=1, name="e"), members=("a", "b"), horizon=3)


def test_an_unchosen_members_holes_flip_the_choice_and_now_say_so(tmp_path):
    """The intersection rule stays, so holes can decide the choice. What changed is that the table and a warning say it."""
    windows = RunWindows.fixed(("2024-01-01", "2024-03-31"), ("2024-04-01", "2024-04-30"))
    hard = lambda target_date: datetime.date.fromisoformat(target_date).day % 5 == 0  # noqa: E731
    days = windows.issue_days()
    write_member(tmp_path, "a", forecasts(days, 3.0, (0, 12), windows=windows), windows)
    b_error = lambda run_date, target_date, *_: 30.0 if hard(target_date) else 1.0  # noqa: E731
    write_member(tmp_path, "b", forecasts(days, b_error, (0, 12), windows=windows), windows)
    no_holes = forecasts(days, 40.0, (0, 12), windows=windows)
    write_member(tmp_path, "c", no_holes, windows)
    ensemble = lambda: AverageEnsemble(["a", "b", "c"], top=1, name="e")  # noqa: E731
    assert choice_of(report_of(tmp_path, ensemble()))["chosen"] == ["a"]
    held = [r for r in no_holes if r["window"] == "test" or not hard(r["target_date"])]
    write_member(tmp_path, "c", held, windows)
    with pytest.warns(UserWarning, match=r"the holes of \{'c'"):
        report = report_of(tmp_path, ensemble())
    assert choice_of(report)["chosen"] == ["b"]
    assert choice_of(report)["validation_share"] < 0.9


# ---------------------------------------------------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def no_fit(monkeypatch):
    """The names of the models that ran before a refusal: there must be none."""
    fits = []
    monkeypatch.setattr(BaseModel, "run", lambda model, *args, **kwargs: fits.append(model.name))
    return fits


@pytest.mark.parametrize("kwargs", [dict(test=T), dict(test_start=T[0], test_end=T[1]), dict(forecast_only=True)])
@pytest.mark.parametrize("top", [1, "validation"])
def test_top_without_validation_is_refused_before_any_fit(df, no_fit, kwargs, top):
    pipeline = built(ols("a"), AverageEnsemble(["a"], top=top, by="validation"))
    with pytest.raises(ValueError, match="no validation window"):
        pipeline.run(df, horizon=1, **kwargs)
    assert no_fit == []


def test_a_model_never_reads_an_ensembles_file_as_its_own_in_a_report(tmp_path):
    """Its row wouldn't move when the members do: an old file is not a forecast of anything."""
    write_members(tmp_path, a=1.0, b=3.0)
    report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))
    with pytest.raises(ManifestMismatchError, match="an ensemble's results"):
        report_of(tmp_path, members=("a", "b", "e"))


def test_a_top_ensemble_with_no_validation_fails_an_array_task_before_any_fit(tmp_path, experiment, no_fit):
    path = experiment(model_index=0)
    config = yaml.safe_load(path.read_text())
    del config["model_pipeline"]["validation"]
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="no validation window"):
        Workflow.load(path).run()
    assert no_fit == []
    assert not (tmp_path / "results").exists()


def test_a_validation_on_file_is_not_a_validation_in_this_run(df, tmp_path, no_fit):
    write_members(tmp_path, a=1.0)
    with pytest.raises(ValueError, match="no validation window"):
        built(ols("a"), AverageEnsemble(["a"], top=1, name="e")).run(df, test=T, horizon=1, save_dir=str(tmp_path))
    assert no_fit == []


def test_a_forecast_only_report_leaves_the_file_alone_and_may_read_what_chose(tmp_path):
    write_members(tmp_path, a=2.0, b=1.0)
    make = lambda: [AverageEnsemble(["a", "b"], name="e"), AverageEnsemble(["a", "b"], top=1, name="t")]  # noqa: E731
    report_of(tmp_path, *make(), members=("a", "b"))
    files = {
        name: (tmp_path / name).read_bytes() for name in ("e.jsonl", "t.jsonl", "e.manifest.json", "t.manifest.json")
    }
    report = reporting(*make(), members=("a", "b")).report(
        tmp_path, horizon=1, forecast_only=True, test_start=HT[0], test_end=HT[1]
    )
    assert {name: (tmp_path / name).read_bytes() for name in files} == files
    assert choice_of(report, "t")["chosen"] == ["b"]
    assert {r["run_date"] for r in records_of(report, "t")} == set(days_of(*HT))
    assert all(r["prediction"] == r["actual"] + 1.0 for r in records_of(report, "t"))
    assert all(r["prediction"] == r["actual"] + 1.5 for r in records_of(report, "e"))


def test_a_report_with_no_validation_on_file_refuses_top(tmp_path):
    write_member(tmp_path, "a", forecasts(days_of(*HT), 1.0))
    with pytest.raises(ValueError, match="no validation window"):
        report_of(tmp_path, AverageEnsemble(["a"], top=1, name="e"), members=("a",), test=HT)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(top=0),
        dict(top=4),
        dict(top=-1),
        dict(top=True),
        dict(top=1.5),
        dict(top="best"),
        dict(by="test"),
        dict(by=None),
        dict(name=""),
    ],
)
def test_parameters_that_cannot_work_are_refused(kwargs):
    with pytest.raises(ValueError):
        AverageEnsemble(["a", "b", "c"], **kwargs)


@pytest.mark.parametrize("members", [[], ["a", "a"], "abc", ["a", 1]])
def test_members_that_cannot_work_are_refused(members):
    with pytest.raises(ValueError):
        AverageEnsemble(members)


def test_a_numpy_top_is_a_number():
    assert AverageEnsemble(["a", "b"], top=np.int64(2)).top == 2


def test_top_out_of_range_is_refused_at_add_and_at_load(tmp_path):
    ensemble = AverageEnsemble(["a", "b"], top=2)
    ensemble.top = 3
    with pytest.raises(ValueError, match="from 1 to 2"):
        ModelPipeline().add_ensemble(ensemble)
    with pytest.raises(TypeError, match="AverageEnsemble"):
        ModelPipeline().add_ensemble("avg")
    path = tmp_path / "pipeline.yaml"
    params = {"members": ["a"], "top": 2, "name": "e"}
    path.write_text(yaml.safe_dump({"models": [], "ensembles": [{"class": "AverageEnsemble", "params": params}]}))
    with pytest.raises(ValueError, match="from 1 to 1"):
        ModelPipeline.load(path)


@pytest.mark.parametrize(
    "ensembles, message",
    [
        ([AverageEnsemble(["a", "b"], name="a")], "unique"),
        ([AverageEnsemble(["a", "b"], name="e"), AverageEnsemble(["a"], name="e")], "unique"),
        ([AverageEnsemble(["a", "b"], name="My E"), AverageEnsemble(["a"], name="my_e")], "unique"),
        ([AverageEnsemble(["a", "typo"], name="e")], "unknown member 'typo'"),
        ([AverageEnsemble(["a", "e"], name="e")], "members of each other"),
        ([AverageEnsemble(["f"], name="e"), AverageEnsemble(["e"], name="f")], "members of each other"),
        (
            [AverageEnsemble(["a", "b"], top=1, name="inner"), AverageEnsemble(["inner", "a"], top=1, name="outer")],
            "can't have the ensembles",
        ),
        (
            [AverageEnsemble(["a", "b"], name="inner"), AverageEnsemble(["inner", "a"], top=1, name="outer")],
            "can't have the ensembles",
        ),
    ],
)
def test_ensembles_that_cannot_work_are_refused_before_any_fit(df, no_fit, ensembles, message):
    pipeline = built(ols("a"), ols("b", SMALL), *ensembles)
    with pytest.raises(ValueError, match=message):
        pipeline.run(df, validation=V, test=T, horizon=1)
    assert no_fit == []


def test_a_benchmark_may_be_an_ensemble_and_a_mistyped_one_still_raises(df):
    good = built(ols("a"), ols("b", SMALL), AverageEnsemble(["a", "b"], name="e"))
    good.add_evaluator(DMEvaluator(benchmark="e", interval=None))
    assert "DM ratio" in good.run(df, test=T, horizon=1).summary()
    bad = built(ols("a"), AverageEnsemble(["a"], name="e")).add_evaluator(DMEvaluator(benchmark="typo"))
    with pytest.raises(ValueError, match="unknown benchmark"):
        bad.run(df, test=T, horizon=1)


def test_an_ensemble_never_overwrites_a_models_results_and_a_model_never_resumes_an_ensembles(df, tmp_path):
    write_member(tmp_path, "a", forecasts(days_of(*HT)))
    write_member(tmp_path, "x", forecasts(days_of(*HT)))
    before = (tmp_path / "x.jsonl").read_bytes()
    with pytest.raises(ManifestMismatchError, match="not an ensemble's"):
        report_of(tmp_path, AverageEnsemble(["a"], name="x"), members=("a",), test=HT)
    assert (tmp_path / "x.jsonl").read_bytes() == before

    other = tmp_path / "other"
    built(ols("a"), AverageEnsemble(["a"], name="e")).run(df, test=T, horizon=1, save_dir=str(other))
    with pytest.raises(ManifestMismatchError, match="an ensemble's"):
        built(ols("e")).run(df, test=T, horizon=1, save_dir=str(other))


ISSUED = {"issue_time": "11:45", "issue_timezone": "Europe/Warsaw", "unknown": "raise"}
RULE = {"kind": "published", "time": "13:00", "days_before": 1, "timezone": None}


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("target", "load", "target"),
        ("freq", "15min", "frequencies"),
        ("horizon", 2, "horizon"),
        ("data_timezone", "Etc/GMT-1", "data_timezone"),
        ("actuals", {"through": "2024-05-13", "hash": "other target"}, "a different target"),
        ("column_hashes", {"price": "one target", "load_d+1": "revised load"}, r"columns they share.*load_d\+1"),
        ("column_hashes", {"price": "other target", "wind_d+1": "wind"}, r"columns they share.*price"),
        ("information_set", {**ISSUED, "issue_time": "23:00"}, "information sets.*23:00"),
        ("information_set", {**ISSUED, "issue_timezone": "UTC"}, "information sets"),
        ("information_set", {**ISSUED, "rules": {"auction": RULE}}, "information sets"),
        ("information_set", None, "information sets.*none"),
    ],
)
def test_members_of_different_tasks_are_refused(tmp_path, field, value, message):
    days = HAND.issue_days()
    actuals = {"actuals": {"through": "2024-05-13", "hash": "one target"}, "information_set": ISSUED}
    write_member(tmp_path, "a", forecasts(days, 1.0, windows=HAND), HAND, **actuals)
    write_member(tmp_path, "b", forecasts(days, 2.0, windows=HAND), HAND, **{**actuals, field: value})
    with pytest.raises(ManifestMismatchError, match=message):
        report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))
    assert not (tmp_path / "e.jsonl").exists()


def test_older_manifests_are_not_compared_through_different_days(tmp_path):
    days = HAND.issue_days()
    write_member(tmp_path, "a", forecasts(days, 1.0, windows=HAND), HAND, column_hashes=None)
    other = dict(column_hashes=None, data_columns_hash="other data", test_end="2024-05-13")
    write_member(tmp_path, "b", forecasts(days, 2.0, windows=HAND), HAND, **other)
    with pytest.warns(UserWarning, match="have no `column_hashes`"):
        report = report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))
    assert counts_of(report).loc["e", "produced"] == 4 * DAY


def test_older_manifests_are_compared_on_their_data_hash(tmp_path):
    days = HAND.issue_days()
    write_member(tmp_path, "a", forecasts(days, 1.0, windows=HAND), HAND, column_hashes=None)
    write_member(
        tmp_path, "b", forecasts(days, 2.0, windows=HAND), HAND, column_hashes=None, data_columns_hash="other data"
    )
    with pytest.raises(ManifestMismatchError, match="data hashes"):
        report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))
    write_member(tmp_path, "b", forecasts(days, 2.0, windows=HAND), HAND, data_columns_hash="other data")
    with pytest.raises(ManifestMismatchError, match="data hashes"):
        report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))


@pytest.mark.parametrize(
    "a_fields, b_fields",
    [
        (dict(assumptions=["one", "two"]), dict(assumptions=["two", "one"])),
        (dict(rules={"x": {**RULE, "note": "from the docs"}}), dict(rules={"x": {**RULE, "note": "other words"}})),
        (dict(rules={"x": {**RULE, "note": "words"}}), dict(rules={"x": RULE})),
        (dict(unknown="raise"), dict(unknown="warn")),
        (dict(use_defaults=True), dict()),
        (dict(rules={"x": RULE, "y": RULE}), dict(rules={"y": RULE, "x": RULE})),
    ],
    ids=[
        "assumptions in another order",
        "note text",
        "a note against none",
        "unknown",
        "defaults spelled out",
        "rule order",
    ],
)
def test_what_doesnt_decide_availability_doesnt_refuse(tmp_path, a_fields, b_fields):
    days = HAND.issue_days()
    write_member(tmp_path, "a", forecasts(days, 1.0, windows=HAND), HAND, information_set={**ISSUED, **a_fields})
    write_member(tmp_path, "b", forecasts(days, 2.0, windows=HAND), HAND, information_set={**ISSUED, **b_fields})
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))
    assert counts_of(report).loc["e", "produced"] == 4 * DAY


@pytest.mark.parametrize(
    "a_fields, b_fields",
    [
        (dict(assumptions=["one"]), dict(assumptions=["one", "two"])),
        (dict(rules={"x": RULE}), dict(rules={"x": {**RULE, "time": "14:00"}})),
        (dict(rules={"x": RULE}), dict(rules={"y": RULE})),
        (dict(), dict(use_defaults=False)),
    ],
    ids=["another assumption", "another time in a rule", "a rule on another column", "no defaults"],
)
def test_what_decides_availability_refuses(tmp_path, a_fields, b_fields):
    days = HAND.issue_days()
    write_member(tmp_path, "a", forecasts(days, 1.0, windows=HAND), HAND, information_set={**ISSUED, **a_fields})
    write_member(tmp_path, "b", forecasts(days, 2.0, windows=HAND), HAND, information_set={**ISSUED, **b_fields})
    with pytest.raises(ManifestMismatchError, match="information sets"):
        report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))


def resumed_on_other_data(df, directory, **windows):
    """`a` on one frame and `b` on another, then `a` resumed on `b`'s with the override: no forecast of `a` is redone."""
    moved = with_noise(df, "load")
    shard(df, "a", SMALL, directory, **windows)
    shard(moved, "b", PREDICTORS, directory, **windows)
    with pytest.raises(ManifestMismatchError, match="columns they share"):
        averaged(directory, SMALL, PREDICTORS, **({"test": T} if "validation" not in windows else {}))
    before = (Path(directory) / "a.jsonl").read_bytes()
    built(ols("a", SMALL)).run(moved, horizon=1, save_dir=str(directory), allow_manifest_mismatch=True, **windows)
    assert (Path(directory) / "a.jsonl").read_bytes() == before
    return moved


@pytest.mark.parametrize("windows", [dict(test=T), dict(validation=V, test=T)], ids=["plain", "windows"])
def test_a_resume_over_other_data_says_so_for_good(df, tmp_path, windows):
    moved = resumed_on_other_data(df, tmp_path, **windows)
    manifest = json.loads(manifest_path(tmp_path / "a.jsonl").read_text())
    assert manifest["data_unverified"] is True
    assert "data_unverified" not in json.loads(manifest_path(tmp_path / "b.jsonl").read_text())
    kwargs = {"test": T} if "validation" not in windows else {}
    with pytest.warns(UserWarning, match=r"stored forecasts of \['a'\] may not come from the frame"):
        report = averaged(tmp_path, SMALL, PREDICTORS, **kwargs)
    assert counts_of(report).loc["e", "produced"] == 3 * DAY
    shard(
        moved, "a", SMALL, tmp_path, **windows
    )  # a later resume compares fine, and knows no more about the old forecasts
    assert json.loads(manifest_path(tmp_path / "a.jsonl").read_text())["data_unverified"] is True


def test_results_resumed_without_their_manifest_are_warned_about_by_name(df, tmp_path):
    moved = with_noise(df, "load")
    shard(df, "a", SMALL, tmp_path, test=T)
    shard(moved, "b", PREDICTORS, tmp_path, test=T)
    manifest_path(tmp_path / "a.jsonl").unlink()
    shard(moved, "a", SMALL, tmp_path, test=T)
    assert json.loads(manifest_path(tmp_path / "a.jsonl").read_text())["data_unverified"] is True
    with pytest.warns(UserWarning, match=r"stored forecasts of \['a'\]"):
        averaged(tmp_path, SMALL, PREDICTORS, test=T)


def test_a_nested_average_warns_about_its_leaf_model_too(df, tmp_path):
    resumed_on_other_data(df, tmp_path, test=T)
    pipeline = built(
        ols("a", SMALL),
        ols("b", PREDICTORS),
        AverageEnsemble(["a"], name="inner"),
        AverageEnsemble(["inner", "b"], name="outer"),
    )
    with pytest.warns(UserWarning, match=r"outer: the stored forecasts of \['a'\]"):
        pipeline.report(tmp_path, test=T, horizon=1)


@pytest.mark.parametrize(
    "steps",
    [
        [dict(test=T)],
        [dict(test=T), dict(test=T)],
        [dict(validation=V), dict(validation=V, test=T)],
        [dict(validation=V, test=T), dict(validation=V)],
    ],
    ids=["fresh", "resumed", "validation then test", "names less"],
)
def test_normal_runs_write_no_such_field(df, tmp_path, steps):
    for step in steps:
        shard(df, "a", SMALL, tmp_path, **step)
    assert "data_unverified" not in json.loads(manifest_path(tmp_path / "a.jsonl").read_text())


def test_an_override_on_the_same_frame_says_nothing_of_the_data(df, tmp_path):
    shard(df, "a", SMALL, tmp_path, test=T)
    longer = (T[0], "2023-04-07")
    built(ols("a", SMALL)).run(df, test=longer, horizon=1, save_dir=str(tmp_path), allow_manifest_mismatch=True)
    assert "data_unverified" not in json.loads(manifest_path(tmp_path / "a.jsonl").read_text())


def reads_wind(ctx):
    base = ["is_weekend_d+{horizon}", "is_weekday_d+{horizon}", "load_d+{horizon}"]
    return base if ctx.value("wind_d+1") > 100 else base[:2]


def test_a_column_two_members_only_read_through_ctx_is_compared(df, tmp_path):
    moved = df.copy()
    moved["wind_d+1"] = moved["wind_d+1"] * 0.01
    built(ols("g1", reads_wind)).run(df, test=T, horizon=1, save_dir=str(tmp_path))
    built(ols("g2", reads_wind)).run(moved, test=T, horizon=1, save_dir=str(tmp_path))
    held = json.loads(manifest_path(tmp_path / "g1.jsonl").read_text())
    assert "wind_d+1" in held["column_hashes"] and "wind_d+1" not in held["predictors"]
    pipeline = built(ols("g1", reads_wind), ols("g2", reads_wind), AverageEnsemble(["g1", "g2"], name="e"))
    with pytest.raises(ManifestMismatchError, match=r"columns they share.*wind_d\+1"):
        pipeline.report(tmp_path, test=T, horizon=1)


def test_a_column_only_read_stays_hashed_when_a_manifest_grows(df, tmp_path):
    built(ols("g", reads_wind)).run(df, validation=V, horizon=1, save_dir=str(tmp_path))
    built(ols("g", reads_wind)).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))
    assert "wind_d+1" in json.loads(manifest_path(tmp_path / "g.jsonl").read_text())["column_hashes"]


@pytest.mark.parametrize("predictors", [PREDICTORS, SMALL], ids=["other columns too", "the same columns"])
def test_equal_values_in_another_dtype_are_not_another_frame_to_a_column(df, tmp_path, predictors):
    other = df.copy()
    other["is_weekend_d+1"] = other["is_weekend_d+1"].astype("float32")
    shard(df, "a", SMALL, tmp_path, test=T)
    shard(other, "b", predictors, tmp_path, test=T)
    held = {name: json.loads(manifest_path(tmp_path / f"{name}.jsonl").read_text()) for name in ("a", "b")}
    assert held["a"]["column_hashes"]["is_weekend_d+1"] == held["b"]["column_hashes"]["is_weekend_d+1"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = averaged(tmp_path, SMALL, predictors, test=T)
    assert counts_of(report).loc["e", "produced"] == 3 * DAY
    assert warned(caught, "frames differ"), "the frame's own hash still sees the dtype, which is worth a word"


def test_each_hash_is_computed_once_per_run_with_the_same_bytes_as_alone(df, tmp_path, monkeypatch):
    calls = {"column": 0, "frame": 0}
    real_column, real_frame = result_manifest.hash_column, result_manifest.hash_frame

    def column(*args):
        calls["column"] += 1
        return real_column(*args)

    def frame(*args):
        calls["frame"] += 1
        return real_frame(*args)

    alone = tmp_path / "alone"
    for name, predictors in (("a", PREDICTORS), ("b", SMALL), ("c", WIND)):
        shard(df, name, predictors, alone, validation=V, test=T)

    monkeypatch.setattr(result_manifest, "hash_column", column)
    monkeypatch.setattr(result_manifest, "hash_frame", frame)
    together = tmp_path / "together"
    built(ols("a"), ols("b", SMALL), ols("c", WIND)).run(df, validation=V, test=T, horizon=1, save_dir=str(together))
    assert calls == {"column": 5, "frame": 1}, "price and the four distinct columns, whatever the number of models"
    for name in ("a", "b", "c"):
        left = without_timestamps(manifest_path(alone / f"{name}.jsonl"))
        right = without_timestamps(manifest_path(together / f"{name}.jsonl"))
        assert left == right, name


def test_a_frame_changed_between_two_runs_is_hashed_again(df, tmp_path):
    frame = df.copy()
    pipeline = built(ols("a", SMALL))
    pipeline.run(frame, test=T, horizon=1, save_dir=str(tmp_path / "first"))
    frame["load_d+1"] = frame["load_d+1"] + 1.0
    frame["unread"] = 1.0
    pipeline.run(frame, test=T, horizon=1, save_dir=str(tmp_path / "second"))
    first = json.loads(manifest_path(tmp_path / "first" / "a.jsonl").read_text())
    second = json.loads(manifest_path(tmp_path / "second" / "a.jsonl").read_text())
    assert first["column_hashes"]["load_d+1"] != second["column_hashes"]["load_d+1"]
    assert first["frame_hash"] != second["frame_hash"]
    assert first["column_hashes"]["price"] == second["column_hashes"]["price"]


def test_data_hashes_are_only_compared_on_the_same_columns(tmp_path):
    days = HAND.issue_days()
    write_member(tmp_path, "a", forecasts(days, 1.0, windows=HAND), HAND, predictors=["x"], data_columns_hash="of x")
    write_member(tmp_path, "b", forecasts(days, 2.0, windows=HAND), HAND, predictors=["y"], data_columns_hash="of y")
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))
    assert report.summary().set_index("model").loc["e", "forecasts"] == 4 * DAY


def test_two_frequencies_in_one_member_are_refused(tmp_path):
    days = days_of(*HT)
    write_member(tmp_path, "a", [*forecasts(days[:2], 1.0), *forecasts(days[2:], 1.0, freq="15min")])
    with pytest.raises(ManifestMismatchError, match="several frequencies"):
        report_of(tmp_path, AverageEnsemble(["a"], name="e"), members=("a",), test=HT)


def test_members_with_records_of_another_frequency_are_refused_also_without_a_manifest(tmp_path):
    write_member(tmp_path, "a", forecasts(days_of(*HT), 1.0))
    write_member(tmp_path, "b", forecasts(days_of(*HT), 2.0, freq="15min"))
    for name in ("a", "b"):
        manifest_path(tmp_path / f"{name}.jsonl").unlink()
    with pytest.raises(ManifestMismatchError, match="frequencies"):
        report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"), test=HT)


def test_members_on_different_data_are_refused_after_real_runs(df, tmp_path):
    moved = df.copy()
    moved["load_d+1"] = moved["load_d+1"] + 1.0
    built(ols("a"), ols("other", SMALL)).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))
    built(ols("b")).run(moved, validation=V, test=T, horizon=1, save_dir=str(tmp_path))
    members = ("a", "b", "other")
    with pytest.raises(ManifestMismatchError, match=r"columns they share.*load_d\+1"):
        report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=members)
    # `other` reads load too, so the column they share says it, though their columns differ.
    with pytest.raises(ManifestMismatchError, match=r"columns they share.*load_d\+1"):
        report_of(tmp_path, AverageEnsemble(["other", "b"], name="f"), members=members)


def test_members_on_a_different_target_are_refused_after_real_runs(df, tmp_path):
    moved = df.copy()
    moved["price"] = moved["price"] + 1.0
    built(ols("a")).run(df, test=T, horizon=1, save_dir=str(tmp_path))
    built(ols("b", SMALL)).run(moved, test=T, horizon=1, save_dir=str(tmp_path))
    with pytest.raises(ManifestMismatchError, match="columns they share.*price"):
        report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"), test=T)


def test_disagreeing_actuals_are_refused(tmp_path):
    days = HAND.issue_days()
    write_member(tmp_path, "a", forecasts(days, 1.0, windows=HAND), HAND)
    b = forecasts(days, 2.0, windows=HAND)
    b[3]["actual"] += 1e-9
    write_member(tmp_path, "b", b, HAND)
    with pytest.raises(ManifestMismatchError, match="disagree on the `actual` of 1 forecast"):
        report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))
    b[3]["actual"] = float("nan")
    write_member(tmp_path, "b", b, HAND)
    with pytest.raises(ManifestMismatchError, match="disagree on the `actual` of 1 forecast"):
        report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))


def test_a_key_held_twice_is_refused(tmp_path):
    days = HAND.issue_days()
    write_member(tmp_path, "a", forecasts(days, 1.0, windows=HAND), HAND)
    records = forecasts(days, 2.0, windows=HAND)
    write_member(tmp_path, "b", [*records, records[0]], HAND)
    with pytest.raises(ManifestMismatchError, match="twice"):
        report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))


def test_no_common_validation_key_is_refused(tmp_path):
    days = HAND.issue_days()
    write_member(tmp_path, "a", [r for r in forecasts(days, 1.0, windows=HAND) if r["run_date"] <= "2024-05-04"], HAND)
    write_member(tmp_path, "b", [r for r in forecasts(days, 2.0, windows=HAND) if r["run_date"] > "2024-05-04"], HAND)
    with pytest.raises(ValueError, match="nothing is ever chosen on nothing"):
        report_of(tmp_path, AverageEnsemble(["a", "b"], top=1, name="e"), members=("a", "b"))


def test_all_actuals_missing_is_no_common_key_either(tmp_path):
    records = [{**r, "actual": float("nan")} for r in forecasts(HAND.issue_days(), 1.0, windows=HAND)]
    write_member(tmp_path, "a", records, HAND)
    with pytest.raises(ValueError, match="nothing is ever chosen on nothing"):
        report_of(tmp_path, AverageEnsemble(["a"], top=1, name="e"), members=("a",))


def shard(frame, name, predictors, directory, **windows):
    built(ols(name, predictors)).run(frame, horizon=1, save_dir=str(directory), **windows)


def with_noise(frame, column, before=None):
    """`column` and its lags revised, as another version of the data would hold them, everywhere or before a day."""
    out = frame.copy()
    rng = np.random.default_rng(9)
    for name in out.columns:
        if name == column or name.startswith(f"{column}_d"):
            noise = rng.normal(0, 80.0, len(out))
            out[name] = out[name] + (noise if before is None else np.where(out.index < before, noise, 0.0))
    return out


def averaged(directory, a_predictors, b_predictors, **kwargs):
    pipeline = built(ols("a", a_predictors), ols("b", b_predictors), AverageEnsemble(["a", "b"], name="e"))
    return pipeline.report(directory, horizon=1, **kwargs)


def warned(caught, text):
    return [str(w.message) for w in caught if text in str(w.message)]


@pytest.mark.parametrize("windows", [dict(test=T), dict(validation=V, test=T)], ids=["plain", "windows"])
def test_a_column_two_members_read_differently_is_refused_after_real_runs(df, tmp_path, windows):
    shard(df, "a", PREDICTORS, tmp_path, **windows)
    shard(with_noise(df, "load"), "b", SMALL, tmp_path, **windows)
    with pytest.raises(ManifestMismatchError, match=r"columns they share.*load_d\+1"):
        averaged(tmp_path, PREDICTORS, SMALL, **({"test": T} if "validation" not in windows else {}))


def test_a_target_revised_before_anything_read_is_refused_after_real_runs(df, tmp_path):
    shard(df, "a", SMALL, tmp_path, test=T)
    shard(with_noise(df, "price", before=pd.Timestamp("2023-02-01", tz=df.index.tz)), "b", WIND, tmp_path, test=T)
    with pytest.raises(ManifestMismatchError, match="columns they share.*price"):
        averaged(tmp_path, SMALL, WIND, test=T)


@pytest.mark.parametrize("windows", [dict(test=T), dict(validation=V, test=T)], ids=["plain", "windows"])
def test_a_column_only_one_member_reads_can_only_be_warned_about_after_real_runs(df, tmp_path, windows):
    shard(df, "a", SMALL, tmp_path, **windows)
    shard(with_noise(df, "wind"), "b", WIND, tmp_path, **windows)
    with pytest.warns(UserWarning, match="their frames differ"):
        report = averaged(tmp_path, SMALL, WIND, **({"test": T} if "validation" not in windows else {}))
    assert counts_of(report).loc["e", "produced"] == 3 * DAY


def test_members_of_one_frame_on_other_columns_say_nothing(df, tmp_path):
    for name, predictors in (("a", SMALL), ("b", WIND)):
        shard(df, name, predictors, tmp_path, test=T)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        averaged(tmp_path, SMALL, WIND, test=T)
    assert (
        not warned(caught, "frames differ") and not warned(caught, "different days") and not warned(caught, "manifest")
    )


def test_validation_first_then_the_test_next_to_one_run_of_both_says_nothing(df, tmp_path):
    shard(df, "a", SMALL, tmp_path, validation=V)
    shard(df, "a", SMALL, tmp_path, validation=V, test=T)
    shard(df, "b", WIND, tmp_path, validation=V, test=T)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        averaged(tmp_path, SMALL, WIND)
    assert not warned(caught, "frames differ") and not warned(caught, "different days")


def test_the_column_hashes_cover_both_windows_after_a_manifest_grows(df, tmp_path):
    shard(df, "a", SMALL, tmp_path / "grown", validation=V)
    shard(df, "a", SMALL, tmp_path / "grown", validation=V, test=T)
    shard(df, "a", SMALL, tmp_path / "once", validation=V, test=T)
    shard(df, "a", SMALL, tmp_path / "narrower", validation=V, test=T)
    shard(df, "a", SMALL, tmp_path / "narrower", validation=V)  # names less than is on file, which stays
    once = json.loads(manifest_path(tmp_path / "once" / "a.jsonl").read_text())
    for folder in ("grown", "narrower"):
        held = json.loads(manifest_path(tmp_path / folder / "a.jsonl").read_text())
        assert held["test_end"] == T[1], folder
        assert held["column_hashes"] == once["column_hashes"] and held["frame_hash"] == once["frame_hash"], folder


def test_a_shard_with_another_issue_time_is_refused_after_real_runs(df, tmp_path):
    late = InformationSet("23:00", "Europe/Warsaw", rules=RULES)
    built(ols("a", SMALL), info=INFO).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))
    built(ols("b", WIND), info=late).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))
    with pytest.raises(ManifestMismatchError, match="information sets.*23:00"):
        averaged(tmp_path, SMALL, WIND)
    other = tmp_path / "other"
    built(ols("a", SMALL), info=INFO).run(df, validation=V, test=T, horizon=1, save_dir=str(other))
    built(ols("b", WIND)).run(df, validation=V, test=T, horizon=1, save_dir=str(other))
    with pytest.raises(ManifestMismatchError, match="information sets.*none"):
        averaged(other, SMALL, WIND)


@pytest.mark.parametrize(
    "fields, message",
    [
        (
            dict(column_hashes={"price": "revised", "load_d+1": "revised"}),
            "columns they share",
        ),
        (dict(information_set=ISSUED), "information sets"),
        (dict(target="load"), "target"),
    ],
)
def test_a_nested_average_is_checked_on_its_leaf_models(tmp_path, fields, message):
    days = HAND.issue_days()
    write_member(tmp_path, "a", forecasts(days, 1.0, windows=HAND), HAND)
    write_member(
        tmp_path,
        "c",
        forecasts(days, 3.0, windows=HAND),
        HAND,
        predictors=["wind_d+1"],
        column_hashes={"wind_d+1": "wind"},
    )
    write_member(tmp_path, "b", forecasts(days, 2.0, windows=HAND), HAND, **fields)
    with pytest.raises(ManifestMismatchError, match=message):
        report_of(tmp_path, AverageEnsemble(["a", "b"], name="direct"))
    outer = [AverageEnsemble(["a", "c"], name="inner"), AverageEnsemble(["inner", "b"], name="outer")]
    with pytest.raises(ManifestMismatchError, match=message):
        report_of(tmp_path, *outer)
    deeper = [
        AverageEnsemble(["a"], name="one"),
        AverageEnsemble(["one", "c"], name="two"),
        AverageEnsemble(["two"], name="three"),
        AverageEnsemble(["b", "three"], name="four"),
    ]
    with pytest.raises(ManifestMismatchError, match=message):
        report_of(tmp_path, *deeper)


def test_hashes_through_different_days_are_not_compared_and_say_so(tmp_path):
    days = HAND.issue_days()
    shifted = dict(
        test_end="2024-05-13",
        data_columns_hash="other data",
        column_hashes={"price": "other target", "load_d+1": "other load"},
        frame_hash="other frame",
        actuals={"through": "2024-05-14", "hash": "other target"},
    )
    write_member(
        tmp_path, "a", forecasts(days, 1.0, windows=HAND), HAND, actuals={"through": "2024-05-13", "hash": "one"}
    )
    write_member(tmp_path, "b", forecasts(days, 2.0, windows=HAND), HAND, **shifted)
    with pytest.warns(UserWarning, match="through different days"):
        report = report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))
    assert counts_of(report).loc["e", "produced"] == 4 * DAY


def test_manifests_older_than_the_column_hashes_are_warned_about(tmp_path):
    days = HAND.issue_days()
    for name in ("a", "b"):
        path = write_member(tmp_path, name, forecasts(days, 1.0, windows=HAND), HAND)
        held = json.loads(manifest_path(path).read_text())
        del held["column_hashes"], held["frame_hash"]
        manifest_path(path).write_text(json.dumps(held))
    with pytest.warns(UserWarning, match=r"manifests of \['a', 'b'\] have no `column_hashes`"):
        report = report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))
    assert counts_of(report).loc["e", "produced"] == 4 * DAY


@pytest.mark.parametrize("without", [("b",), ("a", "b")], ids=["one", "all"])
def test_members_without_a_manifest_are_warned_about(tmp_path, without):
    for name in ("a", "b"):
        write_member(tmp_path, name, forecasts(days_of(*HT), 1.0))
    for name in without:
        manifest_path(tmp_path / f"{name}.jsonl").unlink()
    with pytest.warns(UserWarning, match="no manifest for"):
        report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"), test=HT)


def test_results_held_in_memory_are_not_warned_about(df):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        built(ols("a"), ols("b", SMALL), AverageEnsemble(["a", "b"], name="e")).run(df, test=T, horizon=1)
    assert not warned(caught, "manifest")


def test_a_plain_average_of_ensembles_is_the_mean_of_their_means(df):
    report = built(
        *full_small_wind(),
        AverageEnsemble(["inner", "wind"], name="outer"),  # before what it is made of
        AverageEnsemble(["full", "small"], name="inner"),
    ).run(df, validation=V, test=T, horizon=1)
    assert list(report.refs)[3:] == ["outer", "inner"]
    for view in (report, report.validation):
        full, small, wind = (keyed(records_of(view, name)) for name in ("full", "small", "wind"))
        outer = keyed(records_of(view, "outer"))
        assert outer.keys() == full.keys()
        for key, record in outer.items():
            inner = (full[key]["prediction"] + small[key]["prediction"]) / 2
            assert record["prediction"] == (inner + wind[key]["prediction"]) / 2


def test_a_plain_average_with_a_top_member_has_test_forecasts_alone(df):
    report = built(
        *full_small_wind(),
        AverageEnsemble(["full", "small"], top=1, name="best"),
        AverageEnsemble(["best", "wind"], name="outer"),
    ).run(df, validation=V, test=T, horizon=1)
    assert set(report.validation.refs) == {"full", "small", "wind"}, "and report.validation still works"
    best, wind = keyed(records_of(report, "best")), keyed(records_of(report, "wind"))
    outer = keyed(records_of(report, "outer"))
    assert len(outer) == 3 * DAY
    for key, record in outer.items():
        assert record["prediction"] == (best[key]["prediction"] + wind[key]["prediction"]) / 2
    assert counts_of(report).loc["outer", "missing"] == 0


# ---------------------------------------------------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------------------------------------------------


def test_a_plain_average_has_forecasts_in_every_window_and_a_top_one_in_the_test_window_alone(df, tmp_path):
    report = built(
        *full_small_wind(),
        AverageEnsemble(["full", "small"], name="plain"),
        AverageEnsemble(["full", "small"], top=1, name="top"),
    ).run(df, validation=V, test=T, horizon=2, save_dir=str(tmp_path))
    plain, top = stored(tmp_path / "plain.jsonl"), stored(tmp_path / "top.jsonl")
    assert {r["window"] for r in plain} == {"validation", "test"}
    assert {r["window"] for r in top} == {"test"}
    assert set(report.refs) == {"full", "small", "wind", "plain", "top"}
    assert set(report.validation.refs) == {"full", "small", "wind", "plain"}
    assert report.validation.choices().equals(report.choices())
    full = keyed(stored(tmp_path / "full.jsonl"))
    assert {r["day_in_test"] for r in plain if r["window"] == "test"} == {0, 1, 2}
    assert all(r["day_in_test"] == full[key_of(r)]["day_in_test"] for r in plain)


def test_a_validation_report_leaves_out_what_a_models_does(tmp_path):
    """A plain average's forecasts for test days, issued on the last validation day, are counted as a model's are."""
    write_members(tmp_path, horizons=(1, 2), a=1.0, b=2.0)
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"), horizon=2)
    counts = counts_of(report.validation)
    assert list(counts.loc["e"]) == list(counts.loc["a"])
    assert counts.loc["e", "target_in_test"] == 3 * DAY, (
        "horizon 1 of the last validation day, horizon 2 of the last two"
    )
    assert "member_missing" not in counts
    assert all(r["target_date"] < HT[0] for r in records_of(report.validation, "e"))


def test_a_report_may_name_part_of_what_is_on_file(tmp_path):
    write_members(tmp_path, a=2.0, b=1.0)
    ensembles = lambda: [AverageEnsemble(["a", "b"], top=1, name="e"), AverageEnsemble(["a", "b"], name="plain")]  # noqa: E731
    whole = report_of(tmp_path, *ensembles(), members=("a", "b"))
    records, manifest = (tmp_path / "e.jsonl").read_bytes(), without_timestamps(tmp_path / "e.manifest.json")
    test_only = report_of(tmp_path, *ensembles(), members=("a", "b"), test=HT)
    assert test_only.choices().equals(whole.choices())
    assert forecast_lines(records_of(test_only, "e")) == forecast_lines(records_of(whole, "e"))
    assert test_only.window == "test"
    validation_only = report_of(tmp_path, *ensembles(), members=("a", "b"), validation=HV)
    assert validation_only.window == "validation" and "e" not in validation_only.refs
    assert forecast_lines(records_of(validation_only, "plain")) == forecast_lines(records_of(whole.validation, "plain"))
    assert validation_only.choices().equals(whole.choices())
    assert (tmp_path / "e.jsonl").read_bytes() == records, "the file holds what is on file, whatever a report names"
    assert without_timestamps(tmp_path / "e.manifest.json") == manifest


def test_a_reports_segments_are_what_the_rows_of_choices_count_in_whatever_it_names(tmp_path):
    windows = yearly_members(tmp_path, ("2024-01-05", "2025-12-26"), a_better_in={0})
    ensemble = lambda: AverageEnsemble(["a", "b"], top=1, name="e")  # noqa: E731
    for kwargs in (dict(), dict(test=windows.test)):
        report = report_of(tmp_path, ensemble(), members=("a", "b"), **kwargs)
        assert report.windows.segments == windows.segments and report.windows.mode == windows.mode
        assert list(report.choices()["segment"]) == [0, 1]
        for _, row in report.choices().iterrows():
            assert report.windows.segments[row["segment"]].test == (row["test_start"], row["test_end"])
    assert report.window == "test" and report.windows.validation is None, "it still holds the windows it names"


def test_a_run_that_names_fewer_windows_keeps_a_plain_average_on_all_of_them(df, tmp_path):
    models = lambda: [ols("a"), ols("b", SMALL)]  # noqa: E731
    built(*models()).run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))
    report = built(*models(), AverageEnsemble(["a", "b"], name="e")).run(df, test=T, horizon=1, save_dir=str(tmp_path))
    assert report.window == "test"
    assert {r["window"] for r in stored(tmp_path / "e.jsonl")} == {"validation", "test"}
    assert {r["run_date"] for r in records_of(report, "e")} == set(RunWindows(test=T).issue_days())
    assert read_windows(tmp_path / "e.jsonl") == read_windows(tmp_path / "a.jsonl") == RunWindows.fixed(V, T)


def test_forecast_only_averages(tmp_path, df):
    day = "2023-05-20"
    report = built(ols("a"), ols("b", SMALL), AverageEnsemble(["a", "b"], name="e")).run(
        df, test_start=day, test_end=day, horizon=1, forecast_only=True, save_dir=str(tmp_path)
    )
    a, b = keyed(records_of(report, "a")), keyed(records_of(report, "b"))
    for key, record in keyed(records_of(report, "e")).items():
        assert record["prediction"] == (a[key]["prediction"] + b[key]["prediction"]) / 2
    assert len(report.predictions().query("model == 'e'")) == DAY


# ---------------------------------------------------------------------------------------------------------------------
# An ordinary model everywhere
# ---------------------------------------------------------------------------------------------------------------------


LONG = RunWindows.fixed(("2024-01-01", "2024-01-20"), ("2024-01-21", "2024-02-29"))


@pytest.fixture
def long_report(tmp_path):
    """Forty test days with a chosen and a plain ensemble, one with a gap, on known errors."""
    days = LONG.issue_days()

    def write(name, level, skip=lambda r: False):
        def error(run_date, target_date, hour, horizon):
            return level + random.Random(f"{name} {run_date} {hour} {horizon}").gauss(0, 1)

        records = [r for r in forecasts(days, error, horizons=(1, 2), windows=LONG) if not skip(r)]
        write_member(tmp_path, name, records, LONG, horizon=2)

    write("a", 1.0)
    write("b", -1.5)
    write("c", 4.0, skip=lambda r: r["run_date"] == "2024-02-01" and r["hour"] == 5)
    pipeline = reporting(
        AverageEnsemble(["a", "b", "c"], top=2, name="top2"),
        AverageEnsemble(["a", "c"], name="gappy"),
    ).add_evaluator(
        MAEEvaluator(),
        RMSEEvaluator(),
        BiasEvaluator(),
        rMAEEvaluator("a"),
        DMEvaluator(benchmark="a", reps=20),
        MCSEvaluator(reps=20),
        EconomicEvaluator(),
    )
    return pipeline, tmp_path


def by_hand_mae(records):
    return float(np.mean([abs(r["prediction"] - r["actual"]) for r in records]))


def test_every_table_holds_an_ensemble_as_a_model_on_common_keys(long_report):
    pipeline, directory = long_report
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = pipeline.report(directory, horizon=2)
        top2 = records_of(report, "top2")
        gappy = records_of(report, "gappy")
        assert len(gappy) == len(top2) - 2

        summary = report.summary().set_index("model")
        assert summary.loc["top2", "MAE"] == pytest.approx(by_hand_mae(top2), abs=1e-12)
        assert summary.loc["gappy", "forecasts"] == len(gappy)
        assert summary.loc["top2", "forecasts"] == summary.loc["a", "forecasts"]
        assert summary.loc["gappy", "rMAE"] > 0 and np.isfinite(summary.loc["top2", "MCS p"])
        assert summary.loc["gappy", "DM n_forecasts"] == len(gappy)
        assert np.isfinite(summary.loc["top2", "Bias"]) and summary.loc["c", "Bias"] > summary.loc["gappy", "Bias"]
        assert np.isfinite(summary.loc["top2", "profit"])
        assert summary.loc["gappy", "profit days"] > 0

        tables = (
            report.by_hour(),
            report.by_horizon(),
            report.by_hour_horizon(),
            report.by_year(),
            report.by_year_horizon(),
            report.by_run_weekday_horizon(),
            report.by_target_weekday_horizon(),
            report.by_quarter(),
            report.by_day(),
            report.by_price_level(),
        )
        for table in tables:
            assert {"top2", "gappy"} <= set(table["model"])
            assert table[table["model"] == "gappy"]["MAE"].notna().all()

        tests = report.tests(benchmark="top2", reps=20).set_index(["model", "horizon"])
        assert tests.loc[("gappy", 1), "n_forecasts"] == len(gappy) // 2
        tests = report.tests(benchmark="gappy", reps=20).set_index(["model", "horizon"])
        assert tests.loc[("top2", 2), "n_forecasts"] == sum(r["horizon"] == 2 for r in gappy)

        mcs = report.mcs(reps=20, models=["top2", "gappy", "a"])
        assert set(mcs["model"]) == {"top2", "gappy", "a"}
        assert mcs.attrs["limited_by"] == {"gappy": 2}

        ranks = report.ranks(reps=20, models=["top2", "gappy", "a"])
        assert ranks.attrs["n_forecasts"] == len(gappy)

        compared = report.compare("gappy", "top2").iloc[0]
        assert compared["n_forecasts"] == len(gappy)
        common = [r for r in top2 if key_of(r) in keyed(gappy)]
        assert compared["mae_b"] == pytest.approx(by_hand_mae(common), abs=1e-12)

        board = report.leaderboard(benchmark="a").set_index("model")
        assert board.loc["gappy", "n_forecasts"] == board.loc["gappy", "n_common"] == len(gappy)
        assert board.loc["top2", "n_common"] == len(top2)

        # gappy lacks a period of 2024-02-02 and of 2024-02-03, so the daily max of those isn't taken over 23 hours.
        aggregated = report.aggregate(agg="max")
        counts = counts_of(aggregated)
        assert counts.loc["gappy", "incomplete_day"] == counts.loc["c", "incomplete_day"] == 2
        assert counts.loc["top2", "incomplete_day"] == 0
        assert aggregated.summary().set_index("model").loc["top2", "forecasts"] == 40 * 2
        assert aggregated.choices().equals(report.choices())
        assert report.validation.aggregate(agg="mean").summary().set_index("model").loc["gappy", "forecasts"] > 0

        assert set(report.validation.summary()["model"]) == {"a", "b", "c", "gappy"}


CLONED = RunWindows.fixed(("2023-01-01", "2023-01-31"), ("2023-02-01", "2023-12-31"))


def cloned_members(directory, seed=25, extra_n=-0.14):
    """A year of one period a day: `ben`, `m` and `n` share a daily shock, and `m` and `n` are a little better than `ben`."""
    rng = np.random.default_rng(seed)
    days = CLONED.issue_days()
    shock = {day: rng.normal(0, 1) for day in days}

    def member(name, extra):
        noise = {day: rng.normal(0, 1) for day in days}
        records = forecasts(days, lambda day, *_: 10 + shock[day] + noise[day] + extra, (0,), windows=CLONED)
        write_member(directory, name, records, CLONED)

    member("ben", 0.0)
    member("m", -0.14)
    member("n", extra_n)


@pytest.fixture
def clone_reports(tmp_path):
    """The same three members without and with a `top=1` ensemble, which is a clone of what it chose."""
    cloned_members(tmp_path)
    members = ("ben", "m", "n")
    evaluators = lambda: [DMEvaluator(benchmark="ben", interval=None), MCSEvaluator(reps=50)]  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        without = reporting(members=members).add_evaluator(*evaluators()).report(tmp_path, horizon=1)
        ensemble = AverageEnsemble(list(members), top=1, name="best")
        clone = reporting(ensemble, members=members).add_evaluator(*evaluators()).report(tmp_path, horizon=1)
    twin = choice_of(clone, "best")["chosen"][0]
    return without, clone, twin


def test_a_clone_counts_once_in_a_holm_family(clone_reports):
    without, clone, twin = clone_reports
    assert twin != "ben", "this is about a clone of a model tested, and the next test about one of the benchmark"
    before = without.tests(benchmark="ben", interval=None).set_index("model")
    after = clone.tests(benchmark="ben", interval=None).set_index("model")
    for column in ("p_value", "p_holm", "m", "verdict"):
        assert list(after.loc[list(before.index), column]) == list(before[column]), column
    assert (after["m"] == 2).all()
    assert after.loc["best", "p_holm"] == after.loc[twin, "p_holm"]
    assert after.loc["best", "verdict"] == after.loc[twin, "verdict"]
    assert after.attrs["clones"] == {1: [[twin, "best"]]} and before.attrs["clones"] == {1: []}
    per_hour = clone.tests(benchmark="ben", interval=None, by="hour")
    assert per_hour.attrs["clones"] == {1: [[twin, "best"]]} and "p_holm" not in per_hour


def test_a_clone_counts_once_for_the_dm_evaluator_and_leaderboard_too(clone_reports):
    without, clone, twin = clone_reports
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before, after = without.summary().set_index("model"), clone.summary().set_index("model")
        boards = (
            without.leaderboard(benchmark="ben").set_index("model"),
            clone.leaderboard(benchmark="ben").set_index("model"),
        )
    for column in ("DM p", "DM p_holm", "DM m", "DM verdict"):
        assert list(after.loc[["m", "n"], column]) == list(before.loc[["m", "n"], column]), column
    assert after.loc["best", "DM p_holm"] == after.loc[twin, "DM p_holm"]
    assert after.attrs["clones"] == {"DM": [[twin, "best"]], "MCS": [[twin, "best"]]}
    assert before.attrs["clones"] == {"DM": [], "MCS": []}
    assert list(boards[1].loc[["m", "n"], "p_holm"]) == list(boards[0].loc[["m", "n"], "p_holm"])
    assert boards[1].attrs["clones"] == [[twin, "best"]]


def test_a_clone_of_the_benchmark_is_a_sane_row_outside_the_family(tmp_path):
    cloned_members(tmp_path)
    members = ("ben", "m", "n")
    # `only` chooses from one model, so it is a clone of it whatever the numbers are.
    ensemble = AverageEnsemble(["ben"], top=1, name="only")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        without = report_of(tmp_path, members=members).tests(benchmark="ben", interval=None).set_index("model")
        report = report_of(tmp_path, ensemble, members=members)
        table = report.tests(benchmark="ben", interval=None).set_index("model")
        board = report.leaderboard(benchmark="ben").set_index("model")
    row = table.loc["only"]
    assert (row["ratio"], row["statistic"], row["p_value"], row["p_holm"]) == (1.0, 0.0, 1.0, 1.0)
    assert row["verdict"] == "no difference"
    assert (table["m"] == 2).all() and list(table.loc[["m", "n"], "p_holm"]) == list(without.loc[["m", "n"], "p_holm"])
    assert table.attrs["clones"] == {1: [["ben", "only"]]}
    assert list(board.loc[["m", "n"], "p_holm"]) == list(without.loc[["m", "n"], "p_holm"])
    compared = report.compare("only", "ben").iloc[0]
    assert (compared["ratio"], compared["p_hac"]) == (1.0, 1.0)


@pytest.mark.parametrize("statistic", ["R", "max"])
def test_a_confidence_set_collapses_clones_and_gives_each_twin_the_result(clone_reports, statistic):
    without, clone, twin = clone_reports
    before = without.mcs(reps=50, statistic=statistic).set_index("model")
    after = clone.mcs(reps=50, statistic=statistic).set_index("model")
    for column in ("mae", "p_value", "in_90", "in_75"):
        assert list(after.loc[list(before.index), column]) == list(before[column]), column
        assert after.loc["best", column] == after.loc[twin, column], column
    assert after.attrs["clones"] == [[twin, "best"]] and before.attrs["clones"] == []


def test_the_mcs_evaluator_collapses_clones_too(clone_reports):
    without, clone, twin = clone_reports
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, before, after in (
            ("summary", without.summary().set_index("model"), clone.summary().set_index("model")),
            ("by_horizon", without.by_horizon().set_index("model"), clone.by_horizon().set_index("model")),
        ):
            for column in ("MCS p", "in MCS"):
                assert list(after.loc[list(before.index), column]) == list(before[column]), (name, column)
                assert after.loc["best", column] == after.loc[twin, column], (name, column)
            assert np.isfinite(after["MCS p"]).all(), name


def test_a_ranking_collapses_clones_and_doesnt_split_their_chance_of_being_best(clone_reports):
    without, clone, twin = clone_reports
    before = without.ranks(reps=200).set_index("model")
    after = clone.ranks(reps=200).set_index("model")
    for column in ("rank", "p_best", "rank_low", "rank_high"):
        assert list(after.loc[list(before.index), column]) == list(before[column]), column
        assert after.loc["best", column] == after.loc[twin, column], column
    assert after.attrs["clones"] == [[twin, "best"]] and before.attrs["clones"] == []


def test_a_set_of_nothing_but_clones_is_too_small_for_a_confidence_set_and_ranks_as_one(tmp_path):
    write_members(tmp_path, windows=LONG, a=1.0)
    ensembles = [AverageEnsemble(["a"], name="one"), AverageEnsemble(["a"], top=1, name="two")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = report_of(tmp_path, *ensembles, members=("a",))
        ranks = report.ranks(reps=20).set_index("model")
    with pytest.warns(UserWarning, match="these are 1 once exact clones count once"):
        table = report.mcs(reps=20).set_index("model")
    assert table["p_value"].isna().all() and table["in_90"].isna().all()
    assert table.attrs["clones"] == [["a", "one", "two"]]
    assert (ranks["rank"] == 1).all() and (ranks["p_best"] == 1.0).all()


def test_two_models_and_a_twin_give_what_two_models_give(tmp_path):
    days = LONG.issue_days()
    error_of = lambda name: lambda day, target, hour, horizon: random.Random(f"{name} {day} {hour}").gauss(0, 3)  # noqa: E731
    twin = forecasts(days, error_of("a"), windows=LONG)
    write_member(tmp_path, "a", twin, LONG)
    write_member(tmp_path, "t", twin, LONG)
    write_member(tmp_path, "b", forecasts(days, error_of("b"), windows=LONG), LONG)
    write_member(tmp_path, "c", forecasts(days, error_of("c"), windows=LONG), LONG)
    report = report_of(tmp_path, members=("a", "b", "c", "t"))
    with pytest.warns(UserWarning, match="at least 3 models, got 2"):
        two = report.mcs(reps=20, models=["a", "b"])
    with pytest.warns(UserWarning, match="these are 2 once exact clones count once.*'a', 't'"):
        with_twin = report.mcs(reps=20, models=["a", "b", "t"])
    assert two["p_value"].isna().all() and with_twin["p_value"].isna().all()
    assert with_twin.attrs["clones"] == [["a", "t"]]
    assert report.mcs(reps=20, models=["a", "b", "c"])["p_value"].notna().all()

    evaluator = [MCSEvaluator(reps=20)]
    with pytest.warns(UserWarning, match="these are 2 once exact clones count once"):
        table = EvaluationReport({n: report.refs[n] for n in ("a", "b", "t")}, evaluator).summary()
    assert table["MCS p"].isna().all() and (table["MCS n_days"] > 0).all()
    assert table.attrs["clones"] == {"MCS": [["a", "t"]]}


def test_models_on_the_same_losses_with_other_predictions_are_no_clones(tmp_path):
    """One errs up and the other down by the same amount: the kernels can't tell them apart, and that stays loud."""
    write_members(tmp_path, windows=LONG, a=1.0, b=-1.0, c=2.0)
    report = report_of(tmp_path)
    with pytest.raises(ValueError, match="identical losses.*a, b, c"):
        report.mcs(reps=20)
    report.evaluators = [MCSEvaluator(reps=20)]
    with pytest.warns(UserWarning, match="No model confidence set.*a, b, c"):
        assert report.summary()["MCS p"].isna().all()
    with pytest.warns(UserWarning, match="same losses on every day"):
        ranks = report.ranks(reps=20)
    assert ranks.attrs["clones"] == []
    assert report.tests(benchmark="c", interval=None).attrs["clones"] == {1: []}
    assert sorted(ranks["rank"].tolist()) == [1, 2, 3]


def test_the_same_values_on_other_keys_are_no_clones():
    days = days_of(*HT)
    frame = lambda on: pd.DataFrame(  # noqa: E731
        [
            {
                "run_date": pd.Timestamp(day),
                "target_date": pd.Timestamp(plus_days(day, 1)),
                "hour": 0,
                "horizon": 1,
                "prediction": 5.0,
                "actual": 1.0,
            }
            for day in on
        ]
    )
    panel = LossPanel({"a": frame(days[:2]), "b": frame(days[2:]), "c": frame(days[:2])})
    assert panel.clone_groups(panel.rows(), ["a", "b", "c"]) == [["a", "c"]]
    other = key_frame(dict.fromkeys(days, 2.0))
    against = LossPanel({"a": frame(days[:2]), "b": frame(days[2:]), "c": frame(days[:2]), "ben": other})
    assert against.pair_clone_groups(against.rows(), ["a", "b", "c"], "ben") == [["a", "c"]]


def test_clones_are_read_on_the_keys_compared(tmp_path):
    """With other keys their tests differ, so they are two tests. On the keys of a whole set they are one model."""
    days = LONG.issue_days()
    error_of = lambda name: lambda day, target, hour, horizon: random.Random(f"{name} {day} {hour}").gauss(0, 3)  # noqa: E731
    records = forecasts(days, error_of("a"), windows=LONG)
    write_member(tmp_path, "a", records, LONG)
    write_member(tmp_path, "shorter", [r for r in records if r["run_date"] != "2024-02-01"], LONG)
    write_member(tmp_path, "c", forecasts(days, error_of("c"), windows=LONG), LONG)
    write_member(tmp_path, "d", forecasts(days, error_of("d"), windows=LONG), LONG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = report_of(tmp_path, members=("a", "shorter", "c", "d"))
        table = report.tests(benchmark="c", interval=None).set_index("model")
        confidence = report.mcs(reps=20)
    assert table.attrs["clones"] == {1: []} and (table["m"] == 3).all()
    assert table.loc["a", "p_value"] != table.loc["shorter", "p_value"]
    assert confidence.attrs["clones"] == [["a", "shorter"]]
    assert confidence.set_index("model").loc["a", "p_value"] == confidence.set_index("model").loc["shorter", "p_value"]


def key_frame(values):
    """The frame a panel takes, of one prediction per day: `values` is {day: prediction}."""
    rows = [
        {
            "run_date": pd.Timestamp(day),
            "target_date": pd.Timestamp(plus_days(day, 1)),
            "hour": 0,
            "horizon": 1,
            "prediction": prediction,
            "actual": 1.0,
        }
        for day, prediction in values.items()
    ]
    return pd.DataFrame(rows, columns=["run_date", "target_date", "hour", "horizon", "prediction", "actual"])


def test_one_ulp_on_one_key_is_no_clone():
    days = days_of(*HT)
    values = {day: 5.0 + i for i, day in enumerate(days)}
    nudged = {**values, days[1]: float(np.nextafter(values[days[1]], np.inf))}
    other = {day: 2.0 for day in days}
    panel = LossPanel({"a": key_frame(values), "b": key_frame(nudged), "c": key_frame(values), "ben": key_frame(other)})
    assert panel.clone_groups(panel.rows(), ["a", "b", "c", "ben"]) == [["a", "c"]]
    assert panel.pair_clone_groups(panel.rows(), ["a", "b", "c"], "ben") == [["a", "c"]]


def test_negative_zero_is_the_same_prediction_as_zero():
    days = days_of(*HT)
    panel = LossPanel(
        {
            "zero": key_frame(dict.fromkeys(days, 0.0)),
            "negative": key_frame(dict.fromkeys(days, -0.0)),
            "ben": key_frame(dict.fromkeys(days, 7.0)),
        }
    )
    assert panel.clone_groups(panel.rows(), ["zero", "negative", "ben"]) == [["zero", "negative"]]
    assert panel.pair_clone_groups(panel.rows(), ["zero", "negative"], "ben") == [["zero", "negative"]]
    repeating = LossPanel(
        {"negative": key_frame(dict.fromkeys(days, -0.0)), "ben": key_frame(dict.fromkeys(days, 0.0))}
    )
    assert repeating.pair_clone_groups(repeating.rows(), ["negative"], "ben") == [["ben", "negative"]]


def test_a_model_with_no_forecasts_is_nobodys_clone():
    days = days_of(*HT)
    held = key_frame(dict.fromkeys(days, 5.0))
    panel = LossPanel(
        {"a": held, "none": key_frame({}), "nothing": key_frame({}), "ben": key_frame(dict.fromkeys(days, 2.0))}
    )
    assert panel.clone_groups(panel.rows(), ["none", "nothing", "a"]) == []
    assert panel.pair_clone_groups(panel.rows(), ["none", "nothing", "a"], "ben") == []


def test_a_slice_is_keys_and_predictions_against_the_benchmark():
    days = days_of(*HT)
    values = {day: 5.0 + i for i, day in enumerate(days)}
    holed = {day: 2.0 for day in days[:-1]}  # the benchmark has no forecast on the last day
    panel = LossPanel(
        {
            "a": key_frame(values),
            "differs_where_nothing_is_compared": key_frame({**values, days[-1]: 99.0}),
            "lacks_a_day_that_is_compared": key_frame({day: value for day, value in values.items() if day != days[0]}),
            "other": key_frame({day: value + 1.0 for day, value in values.items()}),
            "ben": key_frame(holed),
        }
    )
    models = ["a", "differs_where_nothing_is_compared", "lacks_a_day_that_is_compared", "other"]
    assert panel.pair_clone_groups(panel.rows(), models, "ben") == [["a", "differs_where_nothing_is_compared"]]
    assert panel.clone_groups(panel.rows(), models) == []


def test_a_model_that_repeats_the_benchmark_on_fewer_keys_tests_nothing(tmp_path):
    """The benchmark less one day: ratio 1, p 1, and it must not cost `m` and `n` their verdict."""
    cloned_members(tmp_path)
    ben = stored(tmp_path / "ben.jsonl")
    members = ("ben", "m", "n")
    without = report_of(tmp_path, members=members).tests(benchmark="ben", interval=None).set_index("model")
    write_member(tmp_path, "bb", [r for r in ben if r["run_date"] != "2023-06-01"], CLONED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = report_of(tmp_path, members=(*members, "bb"))
        table = report.tests(benchmark="ben", interval=None).set_index("model")
    assert (table["m"] == 2).all() and table.attrs["clones"] == {1: [["ben", "bb"]]}
    assert list(table.loc[["m", "n"], "p_holm"]) == list(without.loc[["m", "n"], "p_holm"])
    assert list(table.loc[["m", "n"], "verdict"]) == list(without.loc[["m", "n"], "verdict"])
    assert (table.loc["bb", "ratio"], table.loc["bb", "p_value"], table.loc["bb", "p_holm"]) == (1.0, 1.0, 1.0)
    assert table.loc["bb", "verdict"] == "no difference"
    assert report.leaderboard(benchmark="ben").attrs["clones"] == [["ben", "bb"]]


def test_models_that_differ_only_where_the_benchmark_has_no_forecast_are_one_test(tmp_path):
    cloned_members(tmp_path)
    m = stored(tmp_path / "m.jsonl")
    hole = "2023-06-01"
    write_member(
        tmp_path,
        "t",
        [{**r, "prediction": r["prediction"] + (5.0 if r["run_date"] == hole else 0.0)} for r in m],
        CLONED,
    )
    write_member(tmp_path, "ben", [r for r in stored(tmp_path / "ben.jsonl") if r["run_date"] != hole], CLONED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = (
            report_of(tmp_path, members=("ben", "m", "n", "t")).tests(benchmark="ben", interval=None).set_index("model")
        )
    assert (table["m"] == 2).all() and table.attrs["clones"] == {1: [["m", "t"]]}
    assert (
        table.loc["t", "p_value"] == table.loc["m", "p_value"] and table.loc["t", "p_holm"] == table.loc["m", "p_holm"]
    )


def test_the_clones_of_tests_are_per_horizon_and_of_summary_per_evaluator(tmp_path):
    days = LONG.issue_days()

    def error_of(name):
        return lambda day, target, hour, horizon: random.Random(f"{name} {day} {hour} {horizon}").gauss(0, 3)

    write_member(tmp_path, "a", forecasts(days, error_of("a"), horizons=(1, 2), windows=LONG), LONG, horizon=2)

    def twin_at_one(day, target, hour, horizon):
        return error_of("a" if horizon == 1 else "t")(day, target, hour, horizon)

    write_member(tmp_path, "t", forecasts(days, twin_at_one, horizons=(1, 2), windows=LONG), LONG, horizon=2)
    write_member(tmp_path, "c", forecasts(days, error_of("c"), horizons=(1, 2), windows=LONG), LONG, horizon=2)
    write_member(tmp_path, "d", forecasts(days, error_of("d"), horizons=(1, 2), windows=LONG), LONG, horizon=2)
    pipeline = reporting(members=("a", "t", "c", "d"))
    pipeline.add_evaluator(DMEvaluator(benchmark="c", interval=None), MCSEvaluator(reps=20))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = pipeline.report(tmp_path, horizon=2)
        table = report.tests(benchmark="c", interval=None)
        summary = report.summary()
        at_one = report.mcs(reps=20, horizon=1)
    assert table.attrs["clones"] == {1: [["a", "t"]], 2: []}
    assert summary.attrs["clones"] == {"DM": [], "MCS": []}, "over all horizons they aren't twins"
    assert at_one.attrs["clones"] == [["a", "t"]]
    assert list(table[table["horizon"] == 1]["m"].unique()) == [2]
    assert list(table[table["horizon"] == 2]["m"].unique()) == [3]


def test_elimination_goes_through_every_twin():
    days = [(pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).date().isoformat() for i in range(60)]
    frames = {}
    for name in ("a", "b", "c"):
        rng = random.Random(name)
        frames[name] = key_frame({day: 1.0 + rng.random() * {"a": 1, "b": 4, "c": 9}[name] for day in days})
    frames["t"] = frames["b"]
    panel = LossPanel(frames)
    names = ["a", "b", "c", "t"]
    compared = compare_set(panel, names, panel.rows())
    result = confidence_set(compared, 0.10, "R", 7, 50, 0)
    assert compared.clones == (("b", "t"),)
    assert sorted(result.elimination_order) == [0, 1, 2, 3]
    order = list(result.elimination_order)
    assert abs(order.index(1) - order.index(3)) == 1, "twins fall together"
    assert result.p_values[1] == result.p_values[3]


def test_rmae_with_a_base_that_has_no_forecasts_here_says_so(tmp_path):
    write_members(tmp_path, a=1.0, b=2.0)
    pipeline = reporting(AverageEnsemble(["a", "b"], top=1, name="t"), members=("a", "b")).add_evaluator(
        rMAEEvaluator("t")
    )
    report = pipeline.report(tmp_path, horizon=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert report.summary().set_index("model").loc["t", "rMAE"] == 1.0
    assert not warned(caught, "rMAEEvaluator")
    with pytest.warns(UserWarning, match="rMAEEvaluator: the base model 't' has no forecasts here"):
        assert report.validation.summary()["rMAE"].isna().all()


def test_join_keeps_ensembles_and_choices_and_refuses_a_clash(long_report, tmp_path):
    pipeline, directory = long_report
    other = tmp_path / "other"
    write_member(other, "d", forecasts(LONG.issue_days(), 2.0, horizons=(1, 2), windows=LONG), LONG, horizon=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = pipeline.report(directory, horizon=2)
        extra = reporting(AverageEnsemble(["d"], top=1, name="only_d"), members=("d",)).report(other, horizon=2)
        joined = report.join(extra)
        assert {"top2", "gappy", "only_d", "d", "a"} <= set(joined.refs)
        assert list(joined.choices()["ensemble"]) == ["top2", "only_d"]
        assert set(joined.validation.refs) == {"a", "b", "c", "gappy", "d"}
        assert joined.summary().set_index("model").loc["only_d", "forecasts"] == 40 * DAY * 2
        with pytest.raises(ValueError, match="Both reports have a model named"):
            report.join(report)


def test_exporters_write_ensembles_and_the_terminal_shows_the_choices(long_report, tmp_path, capsys, monkeypatch):
    pipeline, directory = long_report
    monkeypatch.setenv("COLUMNS", "400")
    csv_path, excel_path = tmp_path / "out.csv", tmp_path / "out.xlsx"
    pipeline.add_exporter(CsvExporter(str(csv_path), summary_path=str(tmp_path / "summary.csv")))
    pipeline.add_exporter(ExcelExporter(str(excel_path), sheets=["summary", "horizon", "details"]))
    pipeline.add_exporter(TerminalExporter(show=["summary"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = pipeline.report(directory, horizon=2)
    written = pd.read_csv(csv_path)
    assert {"top2_prediction", "top2_error", "gappy_prediction"} <= set(written.columns)
    assert written["top2_prediction"].notna().sum() == len(records_of(report, "top2"))
    assert written["gappy_prediction"].notna().sum() == len(records_of(report, "gappy"))
    assert set(pd.read_csv(tmp_path / "summary.csv")["model"]) == {"a", "b", "c", "top2", "gappy"}
    sheets = pd.read_excel(excel_path, sheet_name=None)
    assert any("top2" in sheet.to_string() for sheet in sheets.values())
    shown = capsys.readouterr().out
    assert "Choices made on validation" in shown
    assert shown.index("Choices made on validation") < shown.index("Summary")
    assert "2024-01-21" in shown and "top2" in shown
    summary_row = next(line for line in shown[shown.index("Summary") :].splitlines() if "gappy" in line)
    assert str(len(records_of(report, "gappy"))) in summary_row, "its forecast count sits next to the models'"


def test_a_report_without_choices_prints_and_holds_none(df, capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    pipeline = built(ols("a"), ols("b", SMALL), AverageEnsemble(["a", "b"], name="e"))
    pipeline.add_exporter(TerminalExporter(show=["summary"]))
    report = pipeline.run(df, test=T, horizon=1)
    assert "Choices" not in capsys.readouterr().out
    assert report.choices().empty and "validation_mae" in report.choices()
    assert counts_of(report).loc["e", "produced"] == 3 * DAY


def test_records_passed_as_lists_have_no_choices():
    assert EvaluationReport({"m": forecasts(days_of(*HT))}, []).choices().empty


# ---------------------------------------------------------------------------------------------------------------------
# Stored results and SLURM
# ---------------------------------------------------------------------------------------------------------------------


def test_members_in_separate_pipelines_give_what_one_run_gives(df, tmp_path):
    ensembles = lambda: [  # noqa: E731
        AverageEnsemble(["full", "small", "wind"], top=2, name="top2"),
        AverageEnsemble(["full", "small"], name="plain"),
    ]
    one = tmp_path / "one"
    single = built(*full_small_wind(), *ensembles()).run(df, validation=V, test=T, horizon=2, save_dir=str(one))
    shards = tmp_path / "shards"
    for model in full_small_wind():
        built(model).run(df, validation=V, test=T, horizon=2, save_dir=str(shards))
    assert not (shards / "top2.jsonl").exists()
    merged = built(*full_small_wind(), *ensembles()).report(shards, horizon=2)
    for name in ("top2", "plain"):
        assert (shards / f"{name}.jsonl").read_bytes() == (one / f"{name}.jsonl").read_bytes()
        assert without_timestamps(shards / f"{name}.manifest.json") == without_timestamps(one / f"{name}.manifest.json")
    assert merged.choices().equals(single.choices())
    assert merged.summary().equals(single.summary())
    assert merged.forecast_counts().equals(single.forecast_counts())


def test_a_plain_run_reported_later_is_what_one_run_gives(df, tmp_path):
    one, shards = tmp_path / "one", tmp_path / "shards"
    ensemble = lambda: AverageEnsemble(["a", "b"], name="e")  # noqa: E731
    built(ols("a"), ols("b", SMALL), ensemble()).run(df, test=T, horizon=2, save_dir=str(one))
    for model in (ols("a"), ols("b", SMALL)):
        built(model).run(df, test=T, horizon=2, save_dir=str(shards))
    merged = built(ols("a"), ols("b", SMALL), ensemble()).report(shards, test=T, horizon=2)
    assert (shards / "e.jsonl").read_bytes() == (one / "e.jsonl").read_bytes()
    counts = counts_of(merged)
    assert counts.loc["e", "planned"] == counts.loc["a", "planned"] == 3 * DAY * 2
    assert counts.loc["e", "produced"] == 3 * DAY * 2


def test_in_memory_and_on_disk_are_the_same_numbers(df, tmp_path):
    make = lambda: built(*full_small_wind(), AverageEnsemble(["full", "small", "wind"], top="validation", name="e"))  # noqa: E731
    memory = make().run(df, validation=V, test=T, horizon=1)
    disk = make().run(df, validation=V, test=T, horizon=1, save_dir=str(tmp_path))
    assert forecast_lines(records_of(memory, "e")) == forecast_lines(records_of(disk, "e"))
    assert memory.choices().equals(disk.choices())
    assert memory.forecast_counts().equals(disk.forecast_counts())


def test_a_missing_member_raises_and_allow_missing_leaves_the_ensemble_out(tmp_path):
    write_members(tmp_path, a=1.0, b=2.0)
    make = lambda: reporting(  # noqa: E731
        AverageEnsemble(["a", "b"], name="fine"),  # reported first: what comes after it is no more known for that
        AverageEnsemble(["a", "b", "c"], name="uses_c"),
        AverageEnsemble(["a", "b", "c"], top=1, name="chooses_with_c"),
        AverageEnsemble(["uses_c", "a"], name="nested_on_c"),
    )
    with pytest.raises(FileNotFoundError, match="c"):
        make().report(tmp_path, horizon=1)
    with pytest.warns(UserWarning) as caught:
        report = make().report(tmp_path, horizon=1, allow_missing=True)
    shown = "\n".join(str(w.message) for w in caught)
    for name in ("uses_c", "chooses_with_c", "nested_on_c"):
        assert f"ensemble {name!r} is left out" in shown
        assert name not in report.refs
    assert "never averaged" in shown
    assert set(report.refs) == {"a", "b", "fine"}
    assert report.choices().empty
    assert not (tmp_path / "uses_c.jsonl").exists(), "and nothing was averaged over the members that happen to be there"
    assert all(r["prediction"] == r["actual"] + 1.5 for r in records_of(report, "fine"))


def test_a_rerun_or_extended_member_changes_the_ensemble_on_the_next_report(tmp_path):
    days = HAND.issue_days()
    write_members(tmp_path, a=1.0, b=3.0)
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))
    assert all(r["prediction"] == r["actual"] + 2.0 for r in records_of(report, "e"))

    write_member(tmp_path, "b", forecasts(days, 5.0, windows=HAND), HAND)
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))
    assert all(r["prediction"] == r["actual"] + 3.0 for r in records_of(report, "e"))

    write_member(tmp_path, "b", [r for r in forecasts(days, 5.0, windows=HAND) if r["run_date"] != HT[1]], HAND)
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))
    assert counts_of(report).loc["e", "produced"] == 3 * DAY
    write_member(tmp_path, "b", forecasts(days, 5.0, windows=HAND), HAND)
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))
    assert counts_of(report).loc["e", "produced"] == 4 * DAY


def test_an_extended_run_changes_the_ensemble_on_the_next_report(df, tmp_path):
    longer = (T[0], "2023-04-07")
    make = lambda: built(ols("a"), ols("b", SMALL), AverageEnsemble(["a", "b"], name="e"))  # noqa: E731
    built(ols("a"), ols("b", SMALL)).run(df, test=T, horizon=1, save_dir=str(tmp_path))
    assert counts_of(make().report(tmp_path, test=T, horizon=1)).loc["e", "produced"] == 3 * DAY
    built(ols("a"), ols("b", SMALL)).run(
        df, test=longer, horizon=1, save_dir=str(tmp_path), allow_manifest_mismatch=True
    )
    assert counts_of(make().report(tmp_path, test=longer, horizon=1)).loc["e", "produced"] == 5 * DAY
    # What it holds doesn't depend on the period a report names.
    assert counts_of(make().report(tmp_path, test=T, horizon=1)).loc["e", "produced"] == 3 * DAY
    assert len(stored(tmp_path / "e.jsonl")) == 5 * DAY

    windowed = tmp_path / "windowed"
    make().run(df, validation=V, horizon=1, save_dir=str(windowed))
    assert {r["window"] for r in stored(windowed / "e.jsonl")} == {"validation"}
    built(ols("a"), ols("b", SMALL)).run(df, validation=V, test=T, horizon=1, save_dir=str(windowed))
    assert {r["window"] for r in stored(windowed / "e.jsonl")} == {"validation"}, "until the next report"
    assert counts_of(make().report(windowed, horizon=1)).loc["e", "produced"] == 3 * DAY


def test_records_outside_the_windows_or_with_another_tag_are_read_by_nothing(tmp_path):
    """A resumed file can hold them. A report of the member leaves them out, so the ensemble must."""
    outside = forecasts([plus_days(HV[0], -3)], 1.0, windows=None)
    for r in outside:
        r["window"] = "validation"
    for name, off in (("a", 1.0), ("b", 3.0)):
        records = forecasts(HAND.issue_days(), off, windows=HAND)
        for r in records:
            if r["run_date"] == HT[0] and r["hour"] == 0:
                r["window"] = "validation"  # on a test day
            if r["run_date"] == HV[0] and r["hour"] == 0:
                r["window"] = "test"  # on a validation day
        write_member(tmp_path, name, [*outside, *records], HAND)
    report = report_of(tmp_path, AverageEnsemble(["a", "b"], name="e"), members=("a", "b"))
    assert keyed(records_of(report, "e")).keys() == keyed(records_of(report, "a")).keys()
    assert keyed(records_of(report.validation, "e")).keys() == keyed(records_of(report.validation, "a")).keys()
    written = stored(tmp_path / "e.jsonl")
    assert min(r["run_date"] for r in written) == HV[0]
    assert (HT[0], 0) not in {(r["run_date"], r["hour"]) for r in written}
    assert (HV[0], 0) not in {(r["run_date"], r["hour"]) for r in written}


def test_day_in_test_comes_from_the_first_member_that_holds_the_key(tmp_path):
    days = days_of(*HT)
    write_member(tmp_path, "a", forecasts(days, 1.0))
    later = [{**r, "day_in_test": r["day_in_test"] + 100} for r in forecasts(days, 3.0)]
    write_member(tmp_path, "b", later)
    report_of(tmp_path, AverageEnsemble(["a", "b"], name="first_a"), members=("a", "b"), test=HT)
    report_of(tmp_path, AverageEnsemble(["b", "a"], name="first_b"), members=("a", "b"), test=HT)
    assert {r["day_in_test"] for r in stored(tmp_path / "first_a.jsonl")} == {0, 1, 2, 3}
    assert {r["day_in_test"] for r in stored(tmp_path / "first_b.jsonl")} == {100, 101, 102, 103}


def failing_on(monkeypatch, call):
    """`os.replace` as a crash at its `call`-th use from now on."""
    real = os.replace
    calls = []

    def replace(source, destination):
        calls.append(destination)
        if len(calls) == call:
            raise OSError("the disk is gone")
        real(source, destination)

    monkeypatch.setattr("epftoolbox2.ensembles._compute.os.replace", replace)
    return calls


def test_a_crash_while_writing_leaves_the_old_file_whole_and_nothing_else(tmp_path, monkeypatch):
    write_members(tmp_path, a=1.0, b=3.0)
    ensemble = lambda: AverageEnsemble(["a", "b"], name="e")  # noqa: E731
    report_of(tmp_path, ensemble(), members=("a", "b"))
    before = (tmp_path / "e.jsonl").read_bytes()
    write_member(tmp_path, "b", forecasts(HAND.issue_days(), 5.0, windows=HAND), HAND)
    failing_on(monkeypatch, 2)  # the manifest, then the records
    with pytest.raises(OSError, match="the disk is gone"):
        report_of(tmp_path, ensemble(), members=("a", "b"))
    assert (tmp_path / "e.jsonl").read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("call", [1, 2], ids=["before the manifest", "between the two writes"])
def test_a_crash_on_the_first_write_doesnt_refuse_every_later_run(tmp_path, monkeypatch, call):
    write_members(tmp_path, a=1.0, b=3.0)
    ensemble = lambda: AverageEnsemble(["a", "b"], name="e")  # noqa: E731
    with monkeypatch.context() as patched:
        failing_on(patched, call)
        with pytest.raises(OSError, match="the disk is gone"):
            report_of(tmp_path, ensemble(), members=("a", "b"))
    assert not (tmp_path / "e.jsonl").exists()
    report = report_of(tmp_path, ensemble(), members=("a", "b"))
    assert counts_of(report).loc["e", "produced"] == 4 * DAY


def test_the_same_member_results_give_the_same_bytes_whatever_the_order_on_file(tmp_path):
    write_members(tmp_path, horizons=(1, 2), a=lambda d, t, hour, z: 0.1 + hour % 3, b=-0.7, c=1.3)
    make = lambda: AverageEnsemble(["a", "b", "c"], top="validation", name="e")  # noqa: E731
    report_of(tmp_path, make(), horizon=2)
    records, manifest = (tmp_path / "e.jsonl").read_bytes(), without_timestamps(tmp_path / "e.manifest.json")
    created = json.loads((tmp_path / "e.manifest.json").read_text())["created_at"]

    for name in ("a", "b", "c"):
        path = tmp_path / f"{name}.jsonl"
        lines = path.read_text().splitlines(keepends=True)
        random.Random(name).shuffle(lines)
        path.write_text("".join(lines))
        # A member re-run with nothing new only moves its manifest's timestamps.
        held = json.loads(manifest_path(path).read_text())
        manifest_path(path).write_text(json.dumps({**held, "updated_at": "later"}))

    report_of(tmp_path, make(), horizon=2)
    assert (tmp_path / "e.jsonl").read_bytes() == records
    assert without_timestamps(tmp_path / "e.manifest.json") == manifest
    assert json.loads((tmp_path / "e.manifest.json").read_text())["created_at"] == created
    assert list(tmp_path.glob("*.tmp")) == [], "written atomically, and nothing left behind"


def test_the_manifest_says_what_the_ensemble_was_made_of(tmp_path):
    write_members(tmp_path, a=2.0, b=1.0)
    report_of(tmp_path, AverageEnsemble(["a", "b"], top=1, name="e"), members=("a", "b"))
    manifest = json.loads((tmp_path / "e.manifest.json").read_text())
    assert manifest["kind"] == "ensemble"
    assert manifest["model_class"] == "AverageEnsemble"
    assert (manifest["members"], manifest["top"], manifest["by"]) == (["a", "b"], 1, "validation")
    assert manifest["choices"][0]["chosen"] == ["b"]
    assert manifest["choices"][0]["member_mae"] == {"a": 2.0, "b": 1.0}
    assert manifest["target"] == "price" and manifest["freq"] == "1h"
    assert RunWindows.from_dict(manifest["windows"]) == HAND == read_windows(tmp_path / "e.jsonl")
    before = manifest["member_results"]
    assert set(before) == {"a", "b"} and all(entry["manifest"] and entry["results"] for entry in before.values())
    write_member(tmp_path, "b", forecasts(HAND.issue_days(), 1.5, windows=HAND), HAND)
    report_of(tmp_path, AverageEnsemble(["a", "b"], top=1, name="e"), members=("a", "b"))
    after = json.loads((tmp_path / "e.manifest.json").read_text())["member_results"]
    assert after["a"] == before["a"] and after["b"]["results"] != before["b"]["results"]
    assert after["b"]["manifest"] == before["b"]["manifest"]


def test_the_object_keeps_nothing_of_what_it_chose(tmp_path):
    ensemble = AverageEnsemble(["a", "b"], top=1, name="e")
    state = dict(vars(ensemble))
    write_members(tmp_path, a=2.0, b=1.0)
    report_of(tmp_path, ensemble, members=("a", "b"))
    assert vars(ensemble) == state


# ---------------------------------------------------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------------------------------------------------


def test_yaml_round_trip(df, tmp_path):
    pipeline = built(
        *full_small_wind(),
        AverageEnsemble(["full", "small", "wind"], top=2, name="top2"),
        AverageEnsemble(["full", "small", "wind"], top="validation", name="topv"),
        AverageEnsemble(["full", "top2"], name="plain"),
    )
    path = tmp_path / "pipeline.yaml"
    pipeline.save(path)
    saved = yaml.safe_load(path.read_text())
    assert [c["class"] for c in saved["ensembles"]] == ["AverageEnsemble"] * 3
    assert saved["ensembles"][0]["params"] == {
        "members": ["full", "small", "wind"],
        "top": 2,
        "by": "validation",
        "name": "top2",
    }
    assert list(saved) == ["models", "ensembles", "evaluators", "exporters"]
    loaded = ModelPipeline.load(path)
    assert [vars(e) for e in loaded.ensembles] == [vars(e) for e in pipeline.ensembles]
    second = tmp_path / "second.yaml"
    loaded.save(second)
    assert second.read_text() == path.read_text()
    assert "AverageEnsemble" in COMPONENT_REGISTRY["ensembles"]
    left = pipeline.run(df, validation=V, test=T, horizon=1)
    right = loaded.run(df, validation=V, test=T, horizon=1)
    for name in ("top2", "topv", "plain"):
        assert forecast_lines(records_of(left, name)) == forecast_lines(records_of(right, name))


def test_a_pipeline_without_ensembles_saves_what_it_always_did(tmp_path):
    path = tmp_path / "pipeline.yaml"
    built(ols("a")).save(path)
    assert "ensembles" not in path.read_text()
    assert list(yaml.safe_load(path.read_text())) == ["models", "evaluators", "exporters"]


def test_old_yaml_without_ensembles_loads(tmp_path):
    path = tmp_path / "pipeline.yaml"
    model = {"class": "OLSModel", "params": {"predictors": ["x"], "name": "a"}}
    path.write_text(yaml.safe_dump({"models": [model], "evaluators": []}))
    assert ModelPipeline.load(path).ensembles == []


def test_an_unknown_ensemble_class_is_refused(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump({"models": [], "ensembles": [{"class": "WeightedEnsemble", "params": {}}]}))
    with pytest.raises(ValueError, match="Unknown ensemble: WeightedEnsemble"):
        ModelPipeline.load(path)


@pytest.fixture
def experiment(tmp_path, df):
    """A workflow YAML over a local CSV with three models and two ensembles, inline, and a validation and a test window."""
    full = ["load_d+1", "wind_d+1", "is_weekend_d+1", "is_weekday_d+1"]
    zone = df.index.tz
    frame = df[(df.index >= pd.Timestamp("2023-02-15", tz=zone)) & (df.index < pd.Timestamp("2023-04-10", tz=zone))]
    csv = frame[[*full, "price"]].tz_convert("UTC").reset_index(names="datetime")
    csv["datetime"] = csv["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    csv.to_csv(tmp_path / "frame.csv", index=False)

    def model(name, predictors):
        return {"class": "OLSModel", "params": {"predictors": predictors, "training_window": 20, "name": name}}

    def ensemble(**params):
        return {"class": "AverageEnsemble", "params": params}

    config = {
        "data_pipeline": {
            "start": "2023-02-15",
            "end": "2023-04-09",
            "sources": [{"class": "CsvSource", "params": {"file_path": str(tmp_path / "frame.csv")}}],
            "transformers": [{"class": "TimezoneTransformer", "params": {"target_tz": "Europe/Warsaw"}}],
        },
        "model_pipeline": {
            "validation": list(V),
            "test": list(T),
            "target": "price",
            "horizon": 1,
            "save_dir": "results",
            "models": [
                model("full", full),
                model("small", ["load_d+1", "is_weekend_d+1", "is_weekday_d+1"]),
                model("wind", ["wind_d+1", "is_weekend_d+1", "is_weekday_d+1"]),
            ],
            "ensembles": [
                ensemble(members=["full", "small", "wind"], top=2, name="top2"),
                ensemble(members=["full", "small"], name="plain"),
            ],
            "evaluators": [{"class": "MAEEvaluator", "params": {}}],
        },
    }
    path = tmp_path / "experiment.yaml"

    def write(**environment):
        path.write_text(yaml.safe_dump(dict(config, environment=environment) if environment else config))
        return path

    return write


def test_workflow_array_tasks_leave_the_ensembles_to_the_report_and_match_one_run(tmp_path, experiment):
    whole = Workflow.load(experiment()).run()
    results = tmp_path / "results"
    expected = {name: (results / f"{name}.jsonl").read_bytes() for name in ("top2", "plain")}
    expected_choices = whole.choices()
    for path in results.iterdir():
        path.unlink()

    for index in (0, 1, 2):
        Workflow.load(experiment(model_index=index)).run()
    assert sorted(p.name for p in results.glob("*.jsonl")) == ["full.jsonl", "small.jsonl", "wind.jsonl"]

    report = Workflow.load(experiment(model_index=0)).report()
    assert {name: (results / f"{name}.jsonl").read_bytes() for name in ("top2", "plain")} == expected
    assert report.choices().equals(expected_choices)
    assert set(report.summary()["model"]) == {"full", "small", "wind", "top2", "plain"}
    assert set(report.validation.summary()["model"]) == {"full", "small", "wind", "plain"}


def test_workflow_round_trip_keeps_the_ensembles_and_epf_report_prints_the_choices(tmp_path, experiment, capsys):
    path = experiment()
    workflow = Workflow.load(path)
    assert [e.name for e in workflow._build_model_pipeline().ensembles] == ["top2", "plain"]
    saved = tmp_path / "saved.yaml"
    workflow.save(saved)
    written = yaml.safe_load(saved.read_text())["model_pipeline"]["ensembles"]
    assert [c["params"]["name"] for c in written] == ["top2", "plain"]
    assert [vars(e) for e in Workflow.load(saved)._build_model_pipeline().ensembles] == [
        vars(e) for e in workflow._build_model_pipeline().ensembles
    ]
    Workflow.load(path).run()
    capsys.readouterr()
    cli.cmd_report(argparse.Namespace(yaml=str(path), allow_missing=False))
    shown = capsys.readouterr().out
    assert "top2" in shown and "validation_mae" in shown and "plain" in shown


def test_a_workflow_with_ensembles_in_a_pipeline_file_loads_them(tmp_path):
    path = tmp_path / "model_pipeline.yaml"
    reporting(AverageEnsemble(["a", "b"], top=1, name="e"), members=("a", "b")).save(path)
    workflow = Workflow(data_start="2023-01-01", data_end="2023-02-01", model_pipeline=str(path), model_index=0)
    assert [e.name for e in workflow._build_model_pipeline().ensembles] == ["e"]


# ---------------------------------------------------------------------------------------------------------------------
# DST
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("freq, per_hour", [("1h", 1), ("15min", 4)])
def test_days_of_23_and_25_hours_in_results_line_up_in_validation_and_in_test(tmp_path, freq, per_hour):
    """Whatever periods a day holds, 92 or 100 of them too, a key is a key: no period is dropped, moved or matched by position."""
    short, long = (HV[0], HT[0]), ("2024-05-03", HT[1])

    def periods_of(run_date):
        return 23 * per_hour if run_date in short else 25 * per_hour if run_date in long else 24 * per_hour

    def records_of_member(error):
        records = forecasts(HAND.issue_days(), error, tuple(range(25 * per_hour)), windows=HAND, freq=freq)
        return [r for r in records if r["hour"] < periods_of(r["run_date"])]

    write_member(tmp_path, "a", records_of_member(1.0), HAND, freq=freq)
    write_member(tmp_path, "b", records_of_member(3.0), HAND, freq=freq)
    report = report_of(
        tmp_path,
        AverageEnsemble(["a", "b"], name="plain"),
        AverageEnsemble(["a", "b"], top=1, name="e"),
        members=("a", "b"),
        freq=freq,
    )
    for view, name in ((report, "e"), (report, "plain"), (report.validation, "plain")):
        assert keyed(records_of(view, name)).keys() == keyed(records_of(view, "a")).keys()
    served = sum(periods_of(day) for day in days_of(*HV)[:-1])
    assert choice_of(report)["validation_forecasts"] == served
    held = pd.Series([r["run_date"] for r in records_of(report, "plain")]).value_counts()
    assert (held[HT[0]], held[HT[1]], held["2024-05-10"]) == (23 * per_hour, 25 * per_hour, 24 * per_hour)
    assert all(r["prediction"] == r["actual"] + 2.0 for r in records_of(report, "plain"))
    assert all(r["prediction"] == r["actual"] + 1.0 for r in records_of(report, "e"))


@pytest.mark.parametrize("freq, per_hour", [("1h", 1), ("15min", 4)])
def test_keys_align_over_a_switch_in_validation_and_in_test(freq, per_hour):
    """Spring in validation, autumn in test: whatever a run gives on those days, it lines up across the members."""
    frame = make_market_data(start="2023-02-20", days=260, freq=freq, horizon=2, noise=5.0, seed=4)
    validation = ("2023-03-25", "2023-03-27")  # the switch is on 2023-03-26: 23 hours
    test = ("2023-10-28", "2023-10-30")  # the switch is on 2023-10-29: 25 hours
    report = built(
        ols("a", window=21),
        ols("b", SMALL, window=21),
        AverageEnsemble(["a", "b"], name="plain"),
        AverageEnsemble(["a", "b"], top=1, name="top"),
    ).run(frame, validation=validation, test=test, horizon=1, freq=freq)

    a_validation, b_validation = (keyed(records_of(report.validation, name)) for name in ("a", "b"))
    a_test, b_test = keyed(records_of(report, "a")), keyed(records_of(report, "b"))
    assert keyed(records_of(report.validation, "plain")).keys() == a_validation.keys() == b_validation.keys()
    assert keyed(records_of(report, "plain")).keys() == a_test.keys() == b_test.keys()
    assert keyed(records_of(report, "top")).keys() == a_test.keys()

    # The spring issue day has no row for its missing hour: the switch's short day has 23 hours of forecasts.
    per_day = 24 * per_hour
    issued = pd.Series([key[0] for key in a_validation]).value_counts()
    assert issued["2023-03-26"] == per_day - per_hour
    # The repeated autumn hour is one forecast per period id, read from the second occurrence.
    assert pd.Series([key[0] for key in a_test]).value_counts()["2023-10-29"] == per_day

    counts = counts_of(report.validation)
    assert counts.loc["plain", "produced"] == counts.loc["a", "produced"] == len(a_validation)
    # What the clock doesn't have isn't planned, so it is no forecast a member lacks.
    assert counts.loc["plain", "planned"] == counts.loc["a", "planned"] == len(a_validation)
    assert "member_missing" not in counts.columns
    assert counts.loc["a", "missing"] == counts.loc["plain", "missing"] == 0

    for key, record in keyed(records_of(report, "plain")).items():
        assert record["prediction"] == (a_test[key]["prediction"] + b_test[key]["prediction"]) / 2
    for key, record in keyed(records_of(report.validation, "plain")).items():
        assert record["prediction"] == (a_validation[key]["prediction"] + b_validation[key]["prediction"]) / 2
        assert forecast_lines([{"x": record["actual"]}]) == forecast_lines([{"x": a_validation[key]["actual"]}])

    # What the choice was scored on: every validation key with a finite actual, on whatever day.
    finite = sum(np.isfinite(r["actual"]) for r in a_validation.values())
    assert choice_of(report, "top")["validation_forecasts"] == finite
    chosen = keyed(records_of(report, choice_of(report, "top")["chosen"][0]))
    assert all(r["prediction"] == chosen[key_of(r)]["prediction"] for r in records_of(report, "top"))
