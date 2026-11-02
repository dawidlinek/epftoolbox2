"""`Use(list, train_on="matching")`: a forecast trains only on the past days for which the function returned its list.

Everything here is checked against `Rows`, which reads the frame on its own: the rows of a period, chosen by date, the targets
by wall clock and the days a label function gives. So a test fails if the code and this reading part ways.
"""

import datetime
import hashlib
import json
import os
import re
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.exporters import CsvExporter, ExcelExporter, TerminalExporter
from epftoolbox2.information_set import (
    Availability,
    InformationSet,
    InformationSetError,
    leakage_probe,
    perturb_unknown,
)
from epftoolbox2.models import LassoCVModel, OLSModel, Use, _worker
from epftoolbox2.models._calendar import training_span
from epftoolbox2.models._worker import NOT_PLANNED, _worker_init, matching_rows
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.models import _contract
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_digest
from epftoolbox2.results.matching import training_day_rows
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

DOCS_PAGE = Path(__file__).resolve().parents[2] / "docs/src/content/docs/model-pipeline/predictor-functions.mdx"
RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "mixed": Availability.static(),
    "price": Availability.published("13:00", 1),
    "signal": Availability.published("10:00", 0),  # `signal_d-1` is known on the day before its row
    "early": Availability.published("10:15", 0),  # known on the row's day, before the issue time
    "late": Availability.published("13:00", 0),  # known on the row's day, after the issue time
}
INFO = InformationSet("11:45", "Europe/Warsaw", rules=RULES)
TEST = ("2023-04-05", "2023-04-14")
WINDOW = 45
TWO = ["load_d+{horizon}", "wind_d+{horizon}"]


class InProcessPool:
    """Stands in for the worker pool, whose start is most of the time these runs would take. The tests about resuming and
    the number of processes use the real one."""

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


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)


@pytest.fixture(scope="module")
def frame():
    data = make_market_data(noise=5.0, seed=7, horizon=3)
    rng = np.random.default_rng(11)
    data["signal"] = rng.normal(0, 1, len(data))
    data["signal_d-1"] = calendar_shift(data["signal"], -1)
    data["early"] = rng.normal(0, 1, len(data))
    data["late"] = rng.normal(0, 1, len(data))
    # 0 or 1 on a weekend and a number on other days: binary on the rows of a weekend list only, not in the window.
    weekend = data.index.dayofweek >= 5
    data["mixed"] = np.where(weekend, rng.integers(0, 2, len(data)).astype(float), rng.normal(5, 2, len(data)))
    for h in (1, 2, 3):
        data[f"mixed_d+{h}"] = calendar_shift(data["mixed"], h)
    return data


class Rows:
    """The frame read without the library: a period's rows in time order and what a forecast trains on of them."""

    def __init__(self, data, freq="1h", target="price"):
        self.data = data
        self.wall = data.index.tz_localize(None)
        self.dates = self.wall.normalize()
        self.periods = (self.wall.hour * 4 + self.wall.minute // 15) if freq == "15min" else self.wall.hour
        unique = pd.Series(data[target].to_numpy(), index=self.wall)
        unique = unique[~unique.index.duplicated(keep=False)]  # a target on the repeated autumn hour has no value
        self._targets = {h: unique.reindex(self.wall + pd.Timedelta(days=h)).to_numpy() for h in range(1, 8)}

    def target(self, horizon):
        return self._targets[horizon]

    def window(self, day, period, horizon, window=WINDOW):
        """The rows of the window of the forecast of (day, period, horizon), the last one first in line to forecast from."""
        day = pd.Timestamp(day)
        first = day - pd.Timedelta(days=window + horizon)
        return np.flatnonzero((self.periods == period) & (self.dates >= first) & (self.dates <= day))

    def matched(self, day, period, horizon, label, window=WINDOW):
        """The training rows with the label of `day`: those dated up to the target of the day before, before dropping missing
        values."""
        rows = self.window(day, period, horizon, window)
        rows = rows[self.dates[rows] <= pd.Timestamp(day) - pd.Timedelta(days=1 + horizon)]
        same = label(pd.Timestamp(day).date(), period, horizon)
        return rows[[label(self.dates[row].date(), period, horizon) == same for row in rows]]

    def fitted(self, rows, names, horizon):
        """Of those rows, the ones with no missing value in `names` and the target."""
        x = self.data[names].to_numpy(dtype=float)[rows]
        return rows[np.isfinite(x).all(axis=1) & np.isfinite(self.target(horizon)[rows])]

    def days(self, rows):
        return sorted({self.dates[row].date() for row in rows})

    def digest(self, rows):
        ordinals = np.array([day.toordinal() for day in self.days(rows)], dtype="<i8")
        return hashlib.sha256(ordinals.tobytes()).hexdigest()[:12]

    def ols(self, rows, test_row, names, horizon, scaled):
        """The forecast by hand: means and deviations of these rows alone, with `scaled` columns scaled, the others as they are."""
        x = self.data[names].to_numpy(dtype=float)[rows]
        y = self.target(horizon)[rows]
        test = self.data[names].to_numpy(dtype=float)[[test_row]]
        means = np.where(scaled, x.mean(axis=0), 0.0)
        deviations = np.where(scaled, x.std(axis=0, ddof=1), 1.0)
        deviations[deviations == 0] = 1.0
        fit = LinearRegression(fit_intercept=False).fit((x - means) / deviations, (y - y.mean()) / y.std(ddof=1))
        return float(fit.predict((test - means) / deviations)[0] * y.std(ddof=1) + y.mean()), fit.coef_


def real_pool(monkeypatch, processes):
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", ProcessPoolExecutor)
    monkeypatch.setenv("MAX_PROCESSES", str(processes))


def key(record):
    return record["run_date"], record["hour"], record["horizon"]


def by_key(records):
    return {key(r): r for r in records}


def expanded(names, horizon):
    return [name.replace("{horizon}", str(horizon)) for name in names]


def stored_lines(path):
    return sorted(line for line in Path(path).read_text().splitlines() if line)


# ---------------------------------------------------------------------------
# What a forecast trains on
# ---------------------------------------------------------------------------


def weekend_split(ctx):
    if ctx.target_day.weekday() >= 5:
        return Use(TWO, train_on="matching")
    return PREDICTORS


def test_train_on_matching_fits_only_on_days_with_the_same_list(frame):
    ref = OLSModel(weekend_split, training_window=60).run(frame, *TEST, horizon=1, information_set=INFO)
    assert ref.produced == 240
    # Fitted by hand: weekend targets only, with the usual cut (training targets end at D-1).
    run_day, hour = "2023-04-07", 10  # Friday -> Saturday target
    local = frame[frame.index.hour == hour]
    local = local[
        (local.index.date >= pd.Timestamp("2023-02-05").date()) & (local.index.date <= pd.Timestamp(run_day).date())
    ]
    train = local.iloc[:-2]
    train = train[[(d + pd.Timedelta(days=1)).weekday() >= 5 for d in train.index.normalize()]]
    x = train[["load_d+1", "wind_d+1"]].to_numpy()
    y = local["price"].shift(-1).iloc[:-2][train.index].to_numpy()
    mean_x, std_x = x.mean(0), x.std(0, ddof=1)
    mean_y, std_y = y.mean(), y.std(ddof=1)
    fit = LinearRegression(fit_intercept=False).fit((x - mean_x) / std_x, (y - mean_y) / std_y)
    test_x = (local[["load_d+1", "wind_d+1"]].to_numpy()[-1:] - mean_x) / std_x
    expected = fit.predict(test_x)[0] * std_y + mean_y
    got = by_key(ref._results)[(run_day, hour, 1)]["prediction"]
    assert got == pytest.approx(expected, rel=1e-9)


def by_target_weekend(day, period, horizon):
    return (day + datetime.timedelta(days=horizon)).weekday() >= 5


THREE = [*TWO, "mixed_d+{horizon}"]


def hand_lists(ctx):
    """`mixed_d+h` is binary on the rows of the weekend list and a number on the rows it isn't in the set of: its list's mask
    comes from the window, so it is scaled, and a mask per set of days would leave it as it is."""
    return Use(THREE if by_target_weekend(ctx.issue_day, ctx.period, ctx.horizon) else TWO, train_on="matching")


@pytest.mark.parametrize("horizon", [1, 3])
def test_a_matching_forecast_is_a_manual_ols_on_the_matching_days_scaler_included(frame, horizon):
    rows = Rows(frame)
    ref = OLSModel(hand_lists, training_window=WINDOW).run(frame, *TEST, horizon=horizon, information_set=INFO)
    records = by_key(ref._results)
    assert len(records) == ref.count == 24 * horizon * 10
    checked = 0
    for (day, period, hz), record in records.items():
        names = expanded(THREE if by_target_weekend(pd.Timestamp(day).date(), period, hz) else TWO, hz)
        matched = rows.matched(day, period, hz, by_target_weekend)
        fitted = rows.fitted(matched, names, hz)
        assert record["training_days"] == len(rows.days(fitted))
        assert record["matching_days"] == rows.digest(matched)
        if hz == horizon and period in (0, 10):
            prediction, coefficients = rows.ols(
                fitted, rows.window(day, period, hz)[-1], names, hz, np.ones(len(names), dtype=bool)
            )
            assert record["prediction"] == pytest.approx(prediction, abs=1e-9)
            assert record["coefficients"] == pytest.approx(coefficients.tolist(), abs=1e-9)
            checked += 1
    assert checked == 2 * 10
    # The point of `mixed`: binary on the set of the weekend list, a number in the window its mask is read from.
    day = {1: "2023-04-07", 3: "2023-04-05"}[horizon]
    assert by_target_weekend(pd.Timestamp(day).date(), 10, horizon)
    column = frame[f"mixed_d+{horizon}"].to_numpy()
    assert set(column[rows.matched(day, 10, horizon, by_target_weekend)].tolist()) <= {0.0, 1.0}
    assert not set(column[rows.window(day, 10, horizon)[: -(1 + horizon)]].tolist()) <= {0.0, 1.0}


def one_list(ctx):
    return Use(PREDICTORS, train_on="matching")


def one_list_on_all(ctx):
    return Use(PREDICTORS, train_on="all")


@pytest.mark.parametrize(
    ("switch", "freq"), [("spring", "1h"), ("autumn", "1h"), ("spring", "15min"), ("autumn", "15min")]
)
def test_one_list_on_every_day_is_bitwise_train_on_all_across_a_switch(switch, freq):
    start, day = {"spring": ("2023-02-15", "2023-03-26"), "autumn": ("2023-09-15", "2023-10-29")}[switch]
    data = make_market_data(start=start, days=60, freq=freq, horizon=3, noise=5.0, seed=3)
    days = (str(pd.Timestamp(day).date()), str((pd.Timestamp(day) + pd.Timedelta(days=4)).date()))
    run = dict(test_start=days[0], test_end=days[1], horizon=3, freq=freq, information_set=INFO)
    matching = OLSModel(one_list, training_window=30).run(data, **run)
    every_day = OLSModel(one_list_on_all, training_window=30).run(data, **run)
    strip = {"predictor_list", "training_days", "matching_days"}

    def compared(ref):
        return sorted(json.dumps({k: v for k, v in r.items() if k not in strip}, sort_keys=True) for r in ref._results)

    assert compared(matching) == compared(every_day)
    assert matching.skipped == every_day.skipped
    assert {r["predictor_list"] for r in matching._results}.isdisjoint(r["predictor_list"] for r in every_day._results)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def test_the_function_runs_once_per_training_day_period_and_horizon_as_of_that_day(frame):
    calls = []

    def recording(ctx):
        calls.append((ctx.issue_day, ctx.period, ctx.horizon))
        assert ctx.target_day == ctx.issue_day + datetime.timedelta(days=ctx.horizon)
        return Use(PREDICTORS, train_on="matching") if ctx.period == 5 else PREDICTORS

    model = OLSModel(recording, training_window=WINDOW)
    plan = model._plan_predictors(frame, "price", 3, "1h", INFO, *TEST, availability="raise")
    first = datetime.date.fromisoformat(TEST[0])
    last = datetime.date.fromisoformat(TEST[1])
    test_days = [first + datetime.timedelta(days=d) for d in range((last - first).days + 1)]
    expected = {(day, period, hz) for day in test_days for hz in (1, 2, 3) for period in range(24)}
    # Only a pair with a matching forecast has anything to read, from the start of its first window to the day before its
    # last forecast.
    for hz in (1, 2, 3):
        for d in range((last - first).days + WINDOW + hz):
            expected.add((first + datetime.timedelta(days=d - WINDOW - hz), 5, hz))
    assert sorted(calls) == sorted(expected)
    assert plan.calls == len(calls)

    calls.clear()
    model.run(frame, *TEST, horizon=3, information_set=INFO, plan=plan)
    assert calls == []  # before any fit, and only then


def test_a_day_with_no_row_at_a_period_is_not_asked():
    data = make_market_data(start="2023-02-01", days=60, horizon=1, noise=5.0, seed=3)
    calls = set()

    def recording(ctx):
        calls.add((ctx.issue_day, ctx.period))
        return one_list(ctx)

    OLSModel(recording, training_window=20)._plan_predictors(data, "price", 1, "1h", INFO, "2023-03-28", "2023-03-30")
    switch = datetime.date(2023, 3, 26)
    assert (switch, 2) not in calls and (switch, 3) in calls and (switch - datetime.timedelta(days=1), 2) in calls


def test_the_function_answers_on_a_training_day_for_that_day(frame):
    seen = {}

    def reading(ctx):
        seen[(ctx.issue_day, ctx.period, ctx.horizon)] = (ctx.value("signal_d-1"), ctx.published("early"))
        return one_list(ctx)

    data = frame.copy()
    data.loc["2023-03-20", "early"] = np.nan
    OLSModel(reading, training_window=WINDOW)._plan_predictors(data, "price", 1, "1h", INFO, *TEST)
    day = datetime.date(2023, 3, 20)
    expected = data.loc["2023-03-20 10:00", "signal_d-1"].item()
    assert seen[(day, 10, 1)] == (expected, False)
    assert seen[(day + datetime.timedelta(days=1), 10, 1)][1] is True


def test_a_day_answers_alike_as_an_issue_day_and_as_a_training_day():
    """In another zone, whose clocks move on other days: 04:20 in New York is 10:20 in Warsaw until 12 March, 09:20 until 26
    March and 10:20 after, and `early` is out at 10:15. So the answer flips with the day's own issue time, not the forecast's."""
    data = make_market_data(start="2023-01-15", days=90, noise=5.0, seed=3)
    data["early"] = 1.0
    info = InformationSet("04:20", "America/New_York", rules=RULES)

    def answers(first, last):
        seen = {}

        def recording(ctx):
            seen[(ctx.issue_day, ctx.period)] = ctx.published("early")
            return one_list(ctx)

        OLSModel(recording, training_window=20)._plan_predictors(data, "price", 1, "1h", info, first, last)
        return seen

    as_issue_days = answers("2023-03-05", "2023-04-02")
    as_training_days = answers("2023-04-03", "2023-04-03")
    days = [
        datetime.date(2023, 3, 15) + datetime.timedelta(days=d) for d in range(19)
    ]  # the window, up to before its end
    compared = {(day, period) for day in days for period in range(24)} & as_issue_days.keys() & as_training_days.keys()
    assert len(compared) > 400
    assert {key: as_training_days[key] for key in compared} == {key: as_issue_days[key] for key in compared}
    assert as_issue_days[(datetime.date(2023, 3, 10), 12)] and not as_issue_days[(datetime.date(2023, 3, 20), 12)]
    assert as_training_days[(datetime.date(2023, 3, 27), 12)] and not as_training_days[(datetime.date(2023, 3, 20), 12)]


def frozen_after_planning(ctx):
    if os.environ.get("PLAN_IS_DONE"):
        raise AssertionError("the function ran after planning")
    return Use(TWO if ctx.target_weekday >= 5 else PREDICTORS, train_on="matching")


def test_workers_get_what_they_need_without_calling_the_function(frame, monkeypatch):
    real_pool(monkeypatch, 2)
    data = frame.iloc[:2800]
    model = OLSModel(frozen_after_planning, training_window=WINDOW)
    plan = model._plan_predictors(data, "price", 1, "1h", INFO, *TEST)
    monkeypatch.setenv("PLAN_IS_DONE", "1")
    assert model.run(data, *TEST, horizon=1, information_set=INFO, plan=plan).produced == 240


def test_a_plan_for_another_window_is_refused_and_a_plan_with_no_matching_list_has_none_of_this(frame):
    model = OLSModel(weekend_split, training_window=WINDOW)
    plan = model._plan_predictors(frame, "price", 1, "1h", INFO, *TEST)
    assert plan.uses_matching and plan.training_window == WINDOW and plan.day_lists
    other = OLSModel(weekend_split, training_window=WINDOW + 1)
    with pytest.raises(ValueError, match="training window"):
        other.run(frame, *TEST, horizon=1, information_set=INFO, plan=plan)

    calls = []

    def plain(ctx):
        calls.append(ctx.issue_day)
        return PREDICTORS

    plan = OLSModel(plain, training_window=WINDOW)._plan_predictors(frame, "price", 1, "1h", INFO, *TEST)
    assert not plan.uses_matching and not plan.day_lists and plan.training_window is None
    assert plan.calls == len(calls) == 240 and min(calls) == datetime.date.fromisoformat(TEST[0])
    assert (
        OLSModel(plain, training_window=WINDOW).run(frame, *TEST, horizon=1, information_set=INFO).matching_lists
        is None
    )


def test_use_takes_the_two_rules_and_a_list_returned_only_on_training_days_is_still_validated(frame):
    assert Use(PREDICTORS, train_on="matching").train_on == "matching"
    with pytest.raises(ValueError, match="train_on must be"):
        Use(PREDICTORS, train_on="weekends")

    def repeats_a_name_on_a_training_day(ctx):
        if ctx.issue_day < datetime.date.fromisoformat(TEST[0]):
            return Use(["load_d+{horizon}", "load_d+{horizon}"], train_on="matching")
        return one_list(ctx)

    model = OLSModel(repeats_a_name_on_a_training_day, training_window=WINDOW)
    with pytest.raises(ValueError, match="repeated names"):
        model.run(frame, *TEST, horizon=1, information_set=INFO)


# ---------------------------------------------------------------------------
# No look-ahead
# ---------------------------------------------------------------------------

ISSUE_DAY = "2023-04-07"
EXTRA = "price_d-1"


def by_published_cells(ctx):
    """Three cells decide its list: one known, one read by value and one known only after each day's issue time."""
    names = list(TWO)
    if ctx.published("early"):
        names.append("is_weekend_d+{horizon}")
    if ctx.value("signal_d-1") > 0:
        names.append("is_weekday_d+{horizon}")
    if ctx.published("late"):  # never, on any day it is asked as of its own issue time
        names.append(EXTRA)
    return Use(names, train_on="matching")


def issued(data, day=ISSUE_DAY, horizon=3, function=by_published_cells, **run):
    ref = OLSModel(function, training_window=WINDOW).run(
        data, test_start=day, test_end=day, horizon=horizon, information_set=INFO, **run
    )
    return by_key(ref._results)


def sets_of(records):
    return {k: (r["matching_days"], r["training_days"], r["prediction"]) for k, r in records.items()}


@pytest.mark.filterwarnings("ignore:.*was never True")
def test_the_matching_set_does_not_move_when_values_unknown_on_the_issue_day_are_perturbed(frame):
    original = issued(frame)
    assert len(original) == 72
    perturbed, counts, unchecked = perturb_unknown(frame, ISSUE_DAY, INFO)
    assert counts["early"] and counts["late"] and counts["signal"] and not unchecked
    assert sets_of(issued(perturbed)) == sets_of(original)


@pytest.mark.filterwarnings("ignore:.*was never True")
def test_values_of_a_training_day_known_only_after_its_own_issue_time_do_not_move_the_set(frame):
    """`late` of an earlier day is known on the issue day, so `perturb_unknown` leaves it. As of that day's own issue time it
    wasn't, which is the time the function answers as of: taking it away or adding noise there can't move anything."""
    original = issued(frame)
    day = pd.Timestamp(ISSUE_DAY).tz_localize("Europe/Warsaw")
    earlier = frame.index < day
    gone = frame.copy()
    gone.loc[earlier, "late"] = np.nan
    noisy = frame.copy()
    noisy.loc[earlier, "late"] = 1e6 + np.arange(earlier.sum())
    assert sets_of(issued(gone)) == sets_of(original)
    assert sets_of(issued(noisy)) == sets_of(original)


@pytest.mark.filterwarnings("ignore:.*was never True")
def test_a_known_value_of_a_training_day_does_move_the_set_so_the_tests_above_can_fail(frame):
    original = sets_of(issued(frame))
    day = "2023-03-20"
    hours = frame.index.tz_localize(None).normalize() == pd.Timestamp(day)
    for column, value in (("early", np.nan), ("signal_d-1", -1e6)):
        changed = frame.copy()
        changed.loc[hours, column] = value
        moved = sets_of(issued(changed))
        assert {k for k in moved if moved[k][0] != original[k][0]}, column
    # And a cell the function reads on the issue day is the forecast's own.
    own = frame.copy()
    own.loc[own.index.tz_localize(None).normalize() == pd.Timestamp(ISSUE_DAY), "signal_d-1"] = -1e6
    assert sets_of(issued(own)) != original


@pytest.mark.parametrize("horizon", [1, 3])
def test_the_newest_training_target_stays_the_day_before_the_issue_day(frame, horizon):
    day = pd.Timestamp(ISSUE_DAY)
    period = 10
    original = issued(frame, horizon=horizon, function=one_list)[(ISSUE_DAY, period, horizon)]["prediction"]

    def with_target_of(target_day):
        changed = frame.copy()
        changed.loc[target_day.strftime("%Y-%m-%d") + " 10:00", "price"] = 1e7
        return issued(changed, horizon=horizon, function=one_list)[(ISSUE_DAY, period, horizon)]["prediction"]

    assert with_target_of(day - pd.Timedelta(days=1)) != original  # in the set, the last one
    assert with_target_of(day) == original
    assert with_target_of(day + pd.Timedelta(days=horizon)) == original


def test_leakage_probe_and_verify_pass_a_clean_function_and_the_control_reaches_it(frame):
    result = leakage_probe(
        lambda: OLSModel(by_published_cells, training_window=WINDOW),
        frame,
        ISSUE_DAY,
        INFO,
        horizon=3,
        availability="warn",
        check_target_boundary=True,
    )
    result.assert_ok()
    assert result.forecasts == 72 and result.perturbed["price"]

    def clean(ctx):
        names = [*PREDICTORS, "signal_d-1"] if ctx.published("early") else [*TWO, "signal_d-1"]
        return Use(names, train_on="matching")

    report = (
        ModelPipeline(information_set=INFO)
        .add_model(OLSModel(clean, training_window=WINDOW, name="clean"))
        .verify(frame, *TEST, horizon=3, days=2)
    )
    report.assert_ok()
    assert report.table().iloc[0]["control moved"] > 0


def leaks_the_target_of_the_issue_day(anchor, window, hz):
    """The span one day late: the newest training target is dated the issue day."""
    return training_span(anchor + 1, window, hz)


def test_verify_catches_a_matching_forecast_fitted_on_a_target_dated_on_the_issue_day(frame, monkeypatch):
    pipeline = ModelPipeline(information_set=INFO).add_model(OLSModel(one_list, training_window=WINDOW, name="m"))
    pipeline.verify(frame, *TEST, horizon=1, days=1, positive_control=False).assert_ok()

    monkeypatch.setattr(_worker, "training_span", leaks_the_target_of_the_issue_day)
    assert pipeline.verify(frame, *TEST, horizon=1, days=1, positive_control=False).status("m") == "leak"
    probe = leakage_probe(
        lambda: OLSModel(one_list, training_window=WINDOW), frame, ISSUE_DAY, INFO, check_target_boundary=True
    )
    assert not probe.ok


def test_a_probe_finds_nothing_in_a_set_with_or_without_the_date_of_the_doubled_autumn_row():
    """Rows are chosen by date before the filter, so a set can't reach past the span, whatever it holds."""
    data = make_market_data(start="2023-08-01", days=120, noise=5.0, seed=1)
    run_day = "2023-10-30"  # hour 2 of the day before has two rows

    def probe(function):
        return leakage_probe(
            lambda: OLSModel(function, training_window=40),
            data,
            run_day,
            InformationSet(rules=RULES),
            horizon=1,
            check_target_boundary=True,
        )

    def other_list_on_the_switch_day(ctx):
        return Use(TWO if ctx.issue_day == datetime.date(2023, 10, 29) else PREDICTORS, train_on="matching")

    assert probe(one_list).changed == []
    assert probe(other_list_on_the_switch_day).changed == []


def reads_late_on_training_days_only(ctx):
    if ctx.issue_day < datetime.date.fromisoformat(TEST[0]):
        ctx.value("late")
    return one_list(ctx)


def test_a_late_value_on_a_training_day_is_refused_or_listed_like_on_an_issue_day(frame):
    with pytest.raises(InformationSetError, match=r"ctx.value\('late'\)") as caught:
        OLSModel(reads_late_on_training_days_only, training_window=WINDOW).run(
            frame, *TEST, horizon=1, information_set=INFO
        )
    assert "for training day" in "".join(caught.value.__notes__)

    pipeline = ModelPipeline(information_set=INFO).add_model(
        OLSModel(reads_late_on_training_days_only, training_window=WINDOW, name="m")
    )
    with pytest.warns(UserWarning, match="not known"):
        report = pipeline.run(frame, test=TEST, horizon=1)
    row = report.availability().query("predictor == 'late'").iloc[0]
    assert (row["role"], row["status"], row["periods"], row["horizons"]) == ("ctx.value", "late", "0-23", "1")
    verify = pipeline.verify(frame, *TEST, horizon=1, days=1, positive_control=False)
    assert verify.status("m") == "leak"
    assert verify.checks[0].probe.late_reads == {"late": 24}


def test_verify_catches_a_set_chosen_on_training_days_by_a_leaked_value():
    """By its name `signal_d-1` is known on each day. A transformer built it from the next day, so the training day just
    before the issue day reads the issue day's value: the only thing the noise can reach, whatever the list does."""
    source = make_market_data(noise=5.0, seed=7)
    source["signal"] = 20.0 + np.random.default_rng(1).normal(0, 2, len(source))
    info = InformationSet("11:45", "Europe/Warsaw", rules={**RULES, "signal": Availability.realised("1h")})

    def leaking(data):
        out = data.copy()
        out["signal_d-1"] = calendar_shift(out["signal"], 1)
        return out

    first = datetime.date.fromisoformat(TEST[0])

    def on_training_days(ctx):
        if ctx.issue_day < first and ctx.value("signal_d-1") < 0:  # the noise is positive, so no list moves
            return PREDICTORS
        return Use(PREDICTORS, train_on="matching")

    pipeline = ModelPipeline(information_set=info).add_model(
        OLSModel(on_training_days, training_window=WINDOW, name="m")
    )
    report = pipeline.verify(source, TEST[0], TEST[0], horizon=1, days=1, positive_control=False, rebuild=leaking)
    assert report.status("m") == "leak"
    # The hours of the issue day that aren't out by then, as a built value of the training day before it.
    assert report.checks[0].probe.moved_reads == {"signal_d-1": 14}
    assert report.checks[0].probe.max_abs_change == 0.0


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def on_weekends(ctx):
    return Use(PREDICTORS if ctx.issue_day.weekday() >= 5 else TWO, train_on="matching")


def weekend_label(day, period, horizon):
    return day.weekday() >= 5


@pytest.mark.parametrize(
    ("switch", "freq"), [("spring", "1h"), ("autumn", "1h"), ("spring", "15min"), ("autumn", "15min")]
)
def test_a_switch_day_in_the_training_window_and_in_the_test(switch, freq):
    start, day, n_rows = {"spring": ("2023-02-01", "2023-03-26", 23), "autumn": ("2023-09-01", "2023-10-29", 25)}[
        switch
    ]
    per_hour = 4 if freq == "15min" else 1
    data = make_market_data(start=start, days=80, freq=freq, horizon=3, noise=5.0, seed=3)
    rows = Rows(data, freq)
    assert (rows.dates == pd.Timestamp(day)).sum() == n_rows * per_hour  # the day it is about, hours as they fall
    first, last = pd.Timestamp(day) - pd.Timedelta(days=2), pd.Timestamp(day) + pd.Timedelta(days=9)
    window = 20
    ref = OLSModel(on_weekends, training_window=window).run(
        data, str(first.date()), str(last.date()), horizon=3, freq=freq, information_set=INFO
    )
    # In spring the clock has no 02:00 to issue from on the switch day (3 horizons) or to score on it (issued 1 and 2 days before).
    not_on_the_clock = {"spring": 5, "autumn": 0}[switch] * per_hour
    assert ref.produced + sum(ref.skipped.values()) == ref.count == 12 * 24 * per_hour * 3 - not_on_the_clock
    switch_periods = range(2 * per_hour, 3 * per_hour)
    by_hand = 0
    for (issue_day, period, hz), record in by_key(ref._results).items():
        names = expanded(PREDICTORS if pd.Timestamp(issue_day).weekday() >= 5 else TWO, hz)
        matched = rows.matched(issue_day, period, hz, weekend_label, window)
        fitted = rows.fitted(matched, names, hz)
        assert record["training_days"] == len(rows.days(fitted)), (issue_day, period, hz)
        assert record["matching_days"] == rows.digest(matched), (issue_day, period, hz)
        if period in switch_periods and hz in (1, 3):
            scaled = np.array(["is_week" not in name for name in names])
            test_row = rows.window(issue_day, period, hz, window)[-1]
            prediction, _ = rows.ols(fitted, test_row, names, hz, scaled)
            assert record["prediction"] == pytest.approx(prediction, abs=1e-9), (issue_day, period, hz)
            by_hand += 1
    assert by_hand

    # What the warning and explain() say before any fit is what the records say afterwards.
    model = OLSModel(on_weekends, training_window=window)
    plan = model._plan_predictors(data, "price", 3, freq, INFO, str(first.date()), str(last.date()))
    model._load_periods(data, ["price", *sorted({n for names in plan.lists for n in names})], 3, "price")
    counts = model._matching_day_counts(plan, "price")
    for (issue_day, period, hz), record in by_key(ref._results).items():
        assert counts[(period, hz)][plan.position_of(issue_day)] == record["training_days"], (issue_day, period, hz)


def test_a_doubled_autumn_row_follows_its_date():
    data = make_market_data(start="2023-09-01", days=80, noise=5.0, seed=3)
    rows = Rows(data)
    ref = OLSModel(on_weekends, training_window=20).run(
        data, "2023-11-03", "2023-11-06", horizon=1, information_set=INFO
    )
    records = by_key(ref._results)

    # A Sunday trains on the day with two rows at 02:00, and counts it once.
    sunday = records[("2023-11-05", 2, 1)]
    matched = rows.matched("2023-11-05", 2, 1, weekend_label, 20)
    assert (rows.dates[matched] == pd.Timestamp("2023-10-29")).sum() == 2
    fitted = rows.fitted(matched, expanded(PREDICTORS, 1), 1)
    assert (rows.dates[fitted] == pd.Timestamp("2023-10-29")).sum() == 2
    assert sunday["training_days"] == len(rows.days(fitted)) == len(fitted) - 1

    # A Monday leaves both out, and hour 3 next to it is no different on that day.
    monday = rows.matched("2023-11-06", 2, 1, weekend_label, 20)
    assert (rows.dates[monday] == pd.Timestamp("2023-10-29")).sum() == 0
    assert records[("2023-11-06", 2, 1)]["training_days"] == records[("2023-11-06", 3, 1)]["training_days"]


# ---------------------------------------------------------------------------
# Too few days
# ---------------------------------------------------------------------------

STARTS = datetime.date(2023, 4, 10)


def a_list_starts_in_the_test(ctx):
    return Use(PREDICTORS if ctx.issue_day >= STARTS else TWO, train_on="matching")


def test_no_matching_days_are_counted_and_too_few_are_warned_about_in_the_terminal_too(
    frame, tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("COLUMNS", "200")
    days = ("2023-04-07", "2023-04-16")
    pipeline = (
        ModelPipeline(information_set=INFO)
        .add_model(OLSModel(a_list_starts_in_the_test, training_window=WINDOW, name="starts"))
        .add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="fixed"))
        .add_evaluator(MAEEvaluator())
        .add_exporter(TerminalExporter(show=["summary"]))
    )
    # The first day of the new list trains on a row at D-1-h: at horizon h it has none on h + 1 issue days.
    skipped = 24 * (2 + 3 + 4)
    # Then 1, 2, 3 days and so on for 4 predictors, as far as the test reaches. Four is thin, and at horizon 1 five isn't.
    thin = 24 * (4 + 4 + 3)
    with pytest.warns(UserWarning, match=rf"starts: {thin} forecast\(s\).*as few as 1 day\(s\) for 4 predictors"):
        report = pipeline.run(frame, test=days, horizon=3, save_dir=tmp_path)

    counts = report.forecast_counts().set_index("model")
    assert counts.loc["starts", "no_matching_days"] == skipped
    assert set(report.training_days().query("predictors == 4 and horizons == '1'")["largest_days"]) == {5}
    assert counts.loc["starts", "produced"] == counts.loc["starts", "planned"] - skipped
    assert counts.loc["fixed", "no_matching_days"] == 0

    table = report.training_days()
    assert table["model"].tolist() == ["starts"] * 6  # `{horizon}` makes one list per horizon of each
    thin_rows = table[table["thin_forecasts"] > 0]
    assert thin_rows["thin_forecasts"].sum() == thin
    assert thin_rows["smallest_days"].min() == 1
    assert (thin_rows["predictors"] == 4).all()

    out = capsys.readouterr().out
    assert "no more training days than predictors" in out
    assert thin_rows.iloc[0]["list"] in out and str(int(thin_rows.iloc[0]["thin_forecasts"])) in out

    # The merge after array jobs says so too, from the files alone.
    with pytest.warns(UserWarning, match=rf"starts: {thin} forecast\(s\).*as few as 1 day\(s\) for 4 predictors"):
        merged = pipeline.report(tmp_path, test=days, horizon=3)
    pd.testing.assert_frame_equal(merged.training_days(), table)
    pd.testing.assert_frame_equal(merged.forecast_counts(), report.forecast_counts())
    assert "no more training days than predictors" in capsys.readouterr().out

    # Nothing about it for a model that has no such list.
    other = ModelPipeline(information_set=INFO).add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="fixed"))
    alone = other.report(tmp_path, test=days, horizon=3)
    assert alone.training_days().empty and list(alone.training_days().columns) == list(table.columns)
    assert "no_matching_days" not in alone.forecast_counts().columns


# ---------------------------------------------------------------------------
# Lists on some days
# ---------------------------------------------------------------------------


def mixed_label(day, period, horizon):
    """Four things on four kinds of day: the same names under another rule don't match, and neither do other names."""
    if period < 6:
        return "morning"
    if day.weekday() == 0:
        return "same names, all"
    return "two" if day.weekday() >= 5 else "four"


def mixed_lists(ctx):
    label = mixed_label(ctx.issue_day, ctx.period, ctx.horizon)
    if label == "morning":
        return PREDICTORS
    if label == "same names, all":
        return Use(PREDICTORS, train_on="all")
    return Use(TWO if label == "two" else PREDICTORS, train_on="matching")


def test_each_forecast_of_a_mixed_function_trains_on_its_own_set(frame):
    rows = Rows(frame)
    days = ("2023-04-08", "2023-04-17")  # weekends and Mondays
    ref = OLSModel(mixed_lists, training_window=WINDOW).run(frame, *days, horizon=2, information_set=INFO)
    assert ref.produced == ref.count == 24 * 2 * 10
    labels = set()
    for (day, period, hz), record in by_key(ref._results).items():
        label = mixed_label(pd.Timestamp(day).date(), period, hz)
        labels.add(label)
        if label in ("morning", "same names, all"):
            assert "training_days" not in record and "matching_days" not in record
            continue
        names = expanded(TWO if label == "two" else PREDICTORS, hz)
        matched = rows.matched(day, period, hz, mixed_label)
        fitted = rows.fitted(matched, names, hz)
        assert record["training_days"] == len(rows.days(fitted)), (day, period, hz)
        assert record["matching_days"] == rows.digest(matched), (day, period, hz)
        scaled = np.array(["is_week" not in name for name in names])
        prediction, _ = rows.ols(fitted, rows.window(day, period, hz)[-1], names, hz, scaled)
        assert record["prediction"] == pytest.approx(prediction, abs=1e-9)
    assert labels == {"morning", "same names, all", "two", "four"}

    # Against plain counts too: 45 + h days from a window, less the h + 1 of the cut.
    tuesday = by_key(ref._results)[("2023-04-11", 12, 1)]
    window_days = pd.date_range("2023-02-24", "2023-04-09").date
    assert len(window_days) == 45
    assert tuesday["training_days"] == sum(day.weekday() in (1, 2, 3, 4) for day in window_days)


# ---------------------------------------------------------------------------
# Resuming and processes
# ---------------------------------------------------------------------------


def wait_for_a_record(path, timeout=60.0):
    """Wait until the results file at `path` holds a forecast. The pool collects days that finished together in any order,
    so a worker that stopped without waiting could have its failed day collected first, and the run would end with nothing
    saved: the resume would then be a fresh run."""
    deadline = time.monotonic() + timeout
    while not (os.path.exists(path) and os.path.getsize(path) > 0) and time.monotonic() < deadline:
        time.sleep(0.01)


class CrashesAfterSomeFits(OLSModel):
    """Stops a real pool halfway, as a machine might: a worker's fits count in that worker. It stops once a forecast is
    in the file `saved_to` names, so the run always leaves some."""

    fits = 0
    saved_to = None

    def _fit_predict(self, train_x, train_y, test_x):
        type(self).fits += 1
        limit = os.environ.get("CRASH_AFTER")
        if limit and type(self).fits > int(limit):
            wait_for_a_record(type(self).saved_to)
            raise RuntimeError("the machine went down")
        return super()._fit_predict(train_x, train_y, test_x)


def test_an_interrupted_run_resumed_through_the_real_pool_equals_a_fresh_one_bitwise(frame, tmp_path, monkeypatch):
    data = frame.iloc[:2800]
    fresh, path = tmp_path / "fresh.jsonl", tmp_path / "resumed.jsonl"

    def make():
        return CrashesAfterSomeFits(mixed_lists, training_window=WINDOW)

    real_pool(monkeypatch, 1)
    make().run(data, *TEST, horizon=2, information_set=INFO, save_to=str(fresh))

    real_pool(monkeypatch, 2)
    monkeypatch.setenv("CRASH_AFTER", "100")
    monkeypatch.setattr(CrashesAfterSomeFits, "saved_to", path)  # inherited by the forked workers
    with pytest.raises(RuntimeError, match="the machine went down"):
        make().run(data, *TEST, horizon=2, information_set=INFO, save_to=str(path))
    monkeypatch.delenv("CRASH_AFTER")
    written = stored_lines(path)
    assert 0 < len(written) < 480

    real_pool(monkeypatch, 3)
    assert make().run(data, *TEST, horizon=2, information_set=INFO, save_to=str(path)).produced == 480
    assert stored_lines(path) == stored_lines(fresh)
    assert (
        json.loads((tmp_path / "resumed.manifest.json").read_text())["predictor_lists"]
        == json.loads((tmp_path / "fresh.manifest.json").read_text())["predictor_lists"]
    )


def test_forecasts_are_bitwise_equal_at_one_and_three_processes(frame, tmp_path, monkeypatch):
    data = frame.iloc[:2800]
    one, three = tmp_path / "one.jsonl", tmp_path / "three.jsonl"
    real_pool(monkeypatch, 1)
    OLSModel(hand_lists, training_window=WINDOW).run(data, *TEST, horizon=2, information_set=INFO, save_to=str(one))
    real_pool(monkeypatch, 3)
    OLSModel(hand_lists, training_window=WINDOW).run(data, *TEST, horizon=2, information_set=INFO, save_to=str(three))
    assert stored_lines(three) == stored_lines(one)


LIST_CHANGES_ON = datetime.date(2023, 3, 20)


def reads_a_global(ctx):
    return Use(TWO if ctx.issue_day < LIST_CHANGES_ON else PREDICTORS, train_on="matching")


def test_a_changed_set_on_a_stored_key_raises_though_its_list_is_the_same(frame, tmp_path, monkeypatch):
    path = tmp_path / "m.jsonl"
    make = lambda: OLSModel(reads_a_global, training_window=WINDOW)  # noqa: E731
    make().run(frame, *TEST, horizon=1, information_set=INFO, save_to=str(path))
    make().run(frame, *TEST, horizon=1, information_set=INFO, save_to=str(path))  # nothing changed: nothing to say

    monkeypatch.setitem(globals(), "LIST_CHANGES_ON", datetime.date(2023, 3, 25))
    with pytest.raises(
        ManifestMismatchError, match=r"240 stored forecast\(s\) of a train_on='matching' list have other matching days"
    ):
        make().run(frame, *TEST, horizon=1, information_set=INFO, save_to=str(path))

    # An override resumes, as for a list, and says so.
    resumed = make().run(frame, *TEST, horizon=1, information_set=INFO, save_to=str(path), allow_manifest_mismatch=True)
    assert resumed.produced == 240


def by_value(ctx):
    return Use(TWO if ctx.value("signal_d-1") > 0 else PREDICTORS, train_on="matching")


@pytest.mark.parametrize("windows", [False, True], ids=["plain", "with windows"])
def test_a_data_change_that_moves_a_value_driven_choice_on_a_past_day_is_caught(frame, tmp_path, windows):
    """`signal_d-1` is only read, so `data_columns_hash` doesn't cover it. Nothing but the stored sets can tell."""
    run = dict(validation=("2023-04-01", "2023-04-03"), test=TEST) if windows else dict(test=TEST)
    pipeline = ModelPipeline(information_set=INFO).add_model(OLSModel(by_value, training_window=WINDOW, name="m"))
    pipeline.run(frame, horizon=1, save_dir=tmp_path, **run)
    pipeline.run(frame, horizon=1, save_dir=tmp_path, **run)

    changed = frame.copy()
    hours = changed.index.tz_localize(None).normalize() == pd.Timestamp("2023-03-20")
    changed.loc[hours, "signal_d-1"] = -np.sign(changed.loc[hours, "signal_d-1"]) * 1e3
    with pytest.raises(ManifestMismatchError, match="have other matching days"):
        pipeline.run(changed, horizon=1, save_dir=tmp_path, **run)


# ---------------------------------------------------------------------------
# With the rest of a run
# ---------------------------------------------------------------------------


def test_lasso_on_a_filtered_set_works_and_is_deterministic(frame):
    def run():
        return LassoCVModel(weekend_split, training_window=WINDOW, cv=3).run(
            frame, "2023-04-07", "2023-04-08", horizon=1, information_set=INFO
        )

    first, second = run(), run()
    assert first.produced == first.count == 48
    assert not first.skipped
    records = by_key(first._results)
    assert all(np.isfinite(r["prediction"]) for r in records.values())
    assert sorted(map(json.dumps, first._results), key=str) == sorted(map(json.dumps, second._results), key=str)
    # The set is what it is whatever fits it: a smaller one than the window's.
    assert 0 < records[("2023-04-07", 12, 1)]["training_days"] < WINDOW / 2


def without_tags(records):
    return sorted(
        json.dumps({k: v for k, v in r.items() if k not in ("window", "day_in_test")}, sort_keys=True) for r in records
    )


@pytest.mark.filterwarnings("ignore:.*was never True")
@pytest.mark.parametrize("function", [mixed_lists, by_published_cells], ids=["by the day", "by cells"])
def test_each_window_of_a_run_is_what_a_plain_run_over_it_gives(frame, function):
    validation, test = ("2023-04-01", "2023-04-03"), ("2023-04-10", "2023-04-12")
    pipeline = (
        ModelPipeline(information_set=INFO)
        .add_model(OLSModel(function, training_window=WINDOW, name="m"))
        .add_evaluator(MAEEvaluator())
    )
    both = pipeline.run(frame, horizon=2, validation=validation, test=test)
    for days, part in ((test, both), (validation, both.validation)):
        alone = pipeline.run(frame, horizon=2, test=days)
        assert without_tags(part.refs["m"].iter_records()) == without_tags(alone.refs["m"].iter_records())
        assert {r["window"] for r in part.refs["m"].iter_records()} == {"test" if days == test else "validation"}
        pd.testing.assert_frame_equal(part.training_days(), alone.training_days())
        pd.testing.assert_frame_equal(part.forecast_counts(), alone.forecast_counts())


def reads_late_in_the_validation_days(ctx):
    if datetime.date(2023, 4, 1) <= ctx.issue_day <= datetime.date(2023, 4, 3) and ctx.period in (5, 6):
        ctx.value("late")
    if ctx.issue_day == datetime.date(2023, 4, 6) and ctx.period == 7:  # between the windows: only a training day
        ctx.value("late")
    return one_list(ctx) if ctx.period in (5, 7) else PREDICTORS


def test_what_a_window_lists_is_what_a_plain_run_over_it_lists(frame):
    """Period 5 of the test window trains on the days the function read a late value, which a plain run over it evaluates as
    training days. Period 6 doesn't, so the test window has nothing to say about it, and the validation window does. Period 7
    reads one on a day in neither window, after the validation window, so only the test window lists it."""
    validation, test = ("2023-04-01", "2023-04-03"), ("2023-04-10", "2023-04-12")
    pipeline = ModelPipeline(information_set=INFO).add_model(
        OLSModel(reads_late_in_the_validation_days, training_window=WINDOW, name="m")
    )
    with pytest.warns(UserWarning, match="not known"):
        both = pipeline.run(frame, horizon=1, validation=validation, test=test)
    for days, part in ((test, both), (validation, both.validation)):
        with pytest.warns(UserWarning, match="not known"):
            alone = pipeline.run(frame, horizon=1, test=days)
        pd.testing.assert_frame_equal(part.availability(), alone.availability())
    assert both.availability()["periods"].tolist() == ["5, 7"]
    assert both.validation.availability()["periods"].tolist() == ["5-6"]


def test_previous_year_filters_each_forecast_on_its_own_window():
    data = make_market_data(start="2022-04-01", days=460, noise=5.0, seed=3)
    test = ("2023-05-01", "2023-05-02")
    pipeline = ModelPipeline(information_set=INFO).add_model(OLSModel(on_weekends, training_window=30, name="m"))
    report = pipeline.run(data, horizon=1, validation="previous_year", test=test)
    alone = pipeline.run(data, horizon=1, test=test)
    assert without_tags(report.refs["m"].iter_records()) == without_tags(alone.refs["m"].iter_records())
    rows = Rows(data)
    # The first day of the year before has as much of a window as any other, and its set is its own.
    first = next(r for r in report.validation.refs["m"].iter_records() if key(r) == ("2022-05-01", 12, 1))
    matched = rows.matched("2022-05-01", 12, 1, weekend_label, 30)
    assert first["training_days"] == len(rows.days(matched))
    assert first["matching_days"] == rows.digest(matched)


def test_validation_first_and_test_later_is_what_one_run_gives(frame, tmp_path):
    validation, test = ("2023-04-01", "2023-04-03"), ("2023-04-10", "2023-04-12")
    pipeline = ModelPipeline(information_set=INFO).add_model(OLSModel(mixed_lists, training_window=WINDOW, name="m"))
    pipeline.run(frame, horizon=2, validation=validation, save_dir=tmp_path / "later")
    pipeline.run(frame, horizon=2, validation=validation, test=test, save_dir=tmp_path / "later")
    pipeline.run(frame, horizon=2, validation=validation, test=test, save_dir=tmp_path / "once")
    assert stored_lines(tmp_path / "later" / "m.jsonl") == stored_lines(tmp_path / "once" / "m.jsonl")


def test_forecast_only_filters_like_a_backtest(frame):
    day = "2023-04-14"
    model = OLSModel(weekend_split, training_window=WINDOW)
    live = by_key(model.run(frame, day, day, horizon=1, information_set=INFO, forecast_only=True)._results)
    backtest = by_key(model.run(frame, day, day, horizon=1, information_set=INFO)._results)
    assert live == backtest and live[(day, 12, 1)]["training_days"]


def test_ensembles_over_matching_models(frame):
    validation, test = ("2023-04-01", "2023-04-04"), ("2023-04-10", "2023-04-14")
    pipeline = (
        ModelPipeline(information_set=INFO)
        .add_model(OLSModel(weekend_split, training_window=WINDOW, name="weekends"))
        .add_model(OLSModel(a_list_starts_in_the_test, training_window=WINDOW, name="starts"))
        .add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="fixed"))
        .add_ensemble(AverageEnsemble(["weekends", "starts", "fixed"], name="average"))
        .add_ensemble(AverageEnsemble(["weekends", "fixed"], top=1, by="validation", name="best"))
        .add_evaluator(MAEEvaluator())
    )
    with pytest.warns(UserWarning):
        report = pipeline.run(frame, horizon=1, validation=validation, test=test)
    counts = report.forecast_counts().set_index("model")
    assert counts.loc["starts", "no_matching_days"] == 48  # the first two days at horizon 1
    assert counts.loc["average", "member_missing"] == 48
    members = {name: by_key(report.refs[name].iter_records()) for name in ("weekends", "starts", "fixed")}
    average = by_key(report.refs["average"].iter_records())
    assert set(average) == set(members["starts"])
    forecast = ("2023-04-12", 10, 1)
    assert average[forecast]["prediction"] == np.mean([members[name][forecast]["prediction"] for name in members])
    assert report.choices().iloc[0]["chosen"][0] in ("weekends", "fixed")
    assert report.training_days()["model"].unique().tolist() == ["weekends", "starts"]


def test_tables_and_exporters_read_matching_records_and_a_merge_gives_the_same(frame, tmp_path, capsys):
    pipeline = (
        ModelPipeline(information_set=INFO)
        .add_model(OLSModel(weekend_split, training_window=WINDOW, name="m"))
        .add_evaluator(MAEEvaluator())
        .add_exporter(CsvExporter(str(tmp_path / "out.csv"), summary_path=str(tmp_path / "summary.csv")))
        .add_exporter(ExcelExporter(str(tmp_path / "out.xlsx")))
        .add_exporter(TerminalExporter())
    )
    report = pipeline.run(frame, test=TEST, horizon=2, save_dir=tmp_path / "runs")
    assert report.summary().iloc[0]["forecasts"] == 480
    assert report.training_days()["thin_forecasts"].sum() == 0 and not report.training_days().empty
    assert "training days" not in capsys.readouterr().out  # nothing to warn about, so no table
    assert len(pd.read_csv(tmp_path / "out.csv")) == 480
    assert (tmp_path / "out.xlsx").stat().st_size
    assert len(report.by_hour()) == 24
    merged = pipeline.report(tmp_path / "runs", test=TEST, horizon=2)
    pd.testing.assert_frame_equal(merged.summary(), report.summary())
    pd.testing.assert_frame_equal(merged.training_days(), report.training_days())
    assert (
        merged.refs["m"].matching_lists == report.refs["m"].matching_lists
    )  # the list with train_on="all" isn't there

    manifest = json.loads((tmp_path / "runs" / "m.manifest.json").read_text())
    assert {entry["train_on"] for entry in manifest["predictor_lists"]} == {"all", "matching"}
    stored = [json.loads(line) for line in (tmp_path / "runs" / "m.jsonl").read_text().splitlines()]
    matching = {entry["hash"] for entry in manifest["predictor_lists"] if entry["train_on"] == "matching"}
    for record in stored:
        assert ("training_days" in record) == ("matching_days" in record) == (record["predictor_list"] in matching)


def test_explain_shows_the_smallest_median_and_largest_days_of_a_matching_list(frame, capsys):
    rows = Rows(frame)
    pipeline = ModelPipeline(information_set=INFO).add_model(OLSModel(mixed_lists, training_window=WINDOW, name="m"))
    days = ("2023-04-08", "2023-04-17")
    tables = pipeline.explain(frame, model="m", test=days, horizon=1, hour=12)
    assert "smallest days" in capsys.readouterr().out
    table = tables["lists"]
    assert table.loc[table["train_on"] == "all", "smallest days"].eq("-").all()

    expected = {}
    for day in pd.date_range(*days).date:
        for period in range(24):
            label = mixed_label(day, period, 1)
            if label in ("two", "four"):
                names = expanded(TWO if label == "two" else PREDICTORS, 1)
                count = len(rows.days(rows.fitted(rows.matched(str(day), period, 1, mixed_label), names, 1)))
                expected.setdefault(label, []).append(count)
    found = {
        ("two" if row["predictors"] == 2 else "four"): (row["smallest days"], row["median days"], row["largest days"])
        for _, row in table[table["train_on"] == "matching"].iterrows()
    }
    assert found == {label: (min(counts), float(np.median(counts)), max(counts)) for label, counts in expected.items()}

    # A function with no such list has the table it always had.
    def plain(ctx):
        return PREDICTORS

    other = ModelPipeline(information_set=INFO).add_model(OLSModel(plain, training_window=WINDOW, name="p"))
    columns = other.explain(frame, model="p", test=days, horizon=1, hour=12, echo=False)["lists"].columns
    assert "smallest days" not in columns


def test_a_stored_list_this_run_does_not_have_stays_in_the_table_with_no_verdict():
    uses = [
        ("gone", 3, 1, 5),
        ("kept", 3, 2, 0),
        ("kept", 3, 2, 6),
        ("kept", 3, 2, 6),
        ("kept", 4, 2, 7),
        ("kept", 4, 2, 21),
    ]
    rows = training_day_rows("m", uses, {"kept": 6})
    # Six days for six predictors are thin, seven aren't, and a forecast with no day has no fit to be thin.
    assert [(r["list"], r["predictors"], r["thin_forecasts"]) for r in rows] == [("gone", None, 0), ("kept", 6, 2)]
    assert (rows[1]["smallest_days"], rows[1]["median_days"], rows[1]["largest_days"]) == (0, 6.0, 21)
    assert (rows[1]["periods"], rows[1]["horizons"], rows[1]["forecasts"]) == ("3-4", "2", 5)


def test_a_saved_function_with_matching_lists_round_trips(frame, tmp_path):
    pipeline = ModelPipeline(information_set=INFO).add_model(OLSModel(weekend_split, training_window=WINDOW, name="m"))
    path = tmp_path / "pipeline.yaml"
    pipeline.save(path)
    loaded = ModelPipeline.load(path)
    run = dict(test=("2023-04-07", "2023-04-08"), horizon=1)
    assert sorted(map(str, pipeline.run(frame, **run).refs["m"]._results)) == sorted(
        map(str, loaded.run(frame, **run).refs["m"]._results)
    )


def test_the_docs_example_trains_weekend_targets_on_about_a_hundred_days():
    blocks = re.findall(r"```python\n(.*?)```", DOCS_PAGE.read_text(), flags=re.DOTALL)
    namespace = {}
    exec(next(block for block in blocks if "def weekend_model" in block), namespace)
    data = make_market_data(start="2023-01-01", days=520, horizon=1, noise=5.0, seed=7)
    data["load_forecast_d+1"] = data["load_d+1"]
    data["wind_forecast_d+1"] = data["wind_d+1"]
    tables = namespace["pipeline"].explain(
        data, model="weekend", test=("2024-04-01", "2024-04-14"), horizon=1, hour=12, echo=False
    )
    weekend = tables["lists"].query("predictors == 2").iloc[0]
    assert weekend["train_on"] == "matching" and weekend["median days"] == 104
    assert tables["lists"].query("predictors == 3").iloc[0]["median days"] == 260


# ---------------------------------------------------------------------------
# What the plan asks and counts
# ---------------------------------------------------------------------------


def test_only_the_matching_forecasts_of_a_pair_are_asked_for_their_windows(frame):
    last = datetime.date(2023, 4, 10)
    calls = set()

    def matching_on_the_last_day(ctx):
        calls.add((ctx.issue_day, ctx.period))
        return one_list(ctx) if ctx.issue_day == last and ctx.period == 5 else PREDICTORS

    OLSModel(matching_on_the_last_day, training_window=WINDOW)._plan_predictors(
        frame, "price", 1, "1h", INFO, "2023-04-05", "2023-04-10"
    )
    first_test = datetime.date(2023, 4, 5)
    first_read = last - datetime.timedelta(days=WINDOW + 1)
    asked = {day for day, period in calls if day < first_test}
    assert asked == {first_read + datetime.timedelta(days=d) for d in range((first_test - first_read).days)}
    assert {period for day, period in calls if day < first_test} == {5}


def test_the_same_names_with_train_on_all_on_past_days_alone_do_not_match(frame):
    """The list with `train_on="all"` is on no forecast, so it has no id of its own in the plan."""
    rows = Rows(frame)
    first = datetime.date.fromisoformat(TEST[0])

    def label(day, period, horizon):
        return "all" if day < first and day.weekday() == 0 else "matching"

    def past_mondays_on_all(ctx):
        return Use(PREDICTORS, train_on=label(ctx.issue_day, ctx.period, ctx.horizon))

    ref = OLSModel(past_mondays_on_all, training_window=WINDOW).run(frame, *TEST, horizon=1, information_set=INFO)
    names = expanded(PREDICTORS, 1)
    assert {r["predictor_list"] for r in ref._results} == set(ref.matching_lists)
    for (day, period, hz), record in by_key(ref._results).items():
        matched = rows.matched(day, period, hz, label)
        assert record["training_days"] == len(rows.days(rows.fitted(matched, names, hz))), (day, period)
        assert record["matching_days"] == rows.digest(matched)
    assert by_key(ref._results)[("2023-04-05", 12, 1)]["training_days"] == WINDOW - 7  # seven Mondays left out


def test_what_the_plan_counts_is_what_the_records_say_in_a_frame_with_holes():
    data = make_market_data(start="2023-08-01", days=120, noise=5.0, seed=9, horizon=2)
    rng = np.random.default_rng(0)
    for column in ("load_d+1", "wind_d+1", "load_d+2", "wind_d+2", "price"):
        data.loc[rng.random(len(data)) < 0.03, column] = np.nan
    lost_rows = (data.index.date == datetime.date(2023, 10, 5)) | (
        (data.index.date == datetime.date(2023, 10, 3)) & (data.index.hour == 5)
    )
    data = data[~lost_rows]

    def by_target(ctx):
        return Use(TWO if ctx.target_weekday % 3 == 0 else [*TWO, "price_d-1"], train_on="matching")

    days = ("2023-10-02", "2023-10-20")
    model = OLSModel(by_target, training_window=30, name="m")
    plan = model._plan_predictors(data, "price", 2, "1h", None, *days, warn_unpublished=False)
    ref = model.run(data, *days, horizon=2, plan=plan)

    checker = OLSModel(by_target, training_window=30, name="m")
    checker._set_freq("1h")
    columns = sorted({name for names in plan.lists for name in names})
    checker._load_periods(data, ["price", *columns], 2, "price")
    absent = _contract.missing_test_rows(data.index, "1h", plan.issue_days)
    assert (5, pd.Timestamp("2023-10-03")) in absent
    counts = checker._matching_day_counts(plan, "price", {(p, day.toordinal()) for p, day in absent})
    assert len(ref._results) > 500
    for record in ref._results:
        count = counts[(record["hour"], record["horizon"])][plan.position_of(record["run_date"])]
        assert count == record["training_days"], (record["run_date"], record["hour"], record["horizon"])
    # A forecast with no row to forecast from isn't counted at all, and neither is a day with no rows.
    assert counts[(5, 1)][plan.position_of("2023-10-03")] == -1
    assert (counts[(5, 1)][plan.position_of("2023-10-05")], counts[(6, 2)][plan.position_of("2023-10-05")]) == (-1, -1)


def test_explain_leaves_out_a_forecast_with_no_row_to_forecast_from(frame):
    def at_five(ctx):
        return one_list(ctx) if ctx.period == 5 else PREDICTORS

    last = datetime.date(2023, 4, 14)
    data = frame[~((frame.index.date == last) & (frame.index.hour == 5))]
    pipeline = ModelPipeline(information_set=INFO).add_model(OLSModel(at_five, training_window=WINDOW, name="m"))
    table = pipeline.explain(data, model="m", test=("2023-04-10", str(last)), horizon=1, hour=5, echo=False)["lists"]
    row = table[table["train_on"] == "matching"].iloc[0]
    # Counted from the row before it, that forecast would come out a day short, and a -1 would be a count of its own.
    assert (row["smallest days"], row["median days"], row["largest days"]) == (WINDOW, float(WINDOW), WINDOW)


def test_a_row_the_plan_has_no_list_for_raises_rather_than_quietly_not_matching():
    with pytest.raises(RuntimeError, match="its plan has no list for"):
        matching_rows(np.array([3]), np.array([0, 0, 0, NOT_PLANNED]), 0, 0)


def only_on_training_days(ctx):
    if ctx.issue_day < datetime.date.fromisoformat(TEST[0]):
        ctx.value("signal_d-1")
        ctx.value("no_rule_value")
        ctx.published("no_rule_published")
    return one_list(ctx)


def test_what_the_function_only_reads_on_training_days_is_recorded_and_listed(frame, tmp_path):
    data = frame.assign(no_rule_value=1.0, no_rule_published=1.0)
    pipeline = ModelPipeline(information_set=INFO).add_model(
        OLSModel(only_on_training_days, training_window=WINDOW, name="m")
    )
    with pytest.warns(UserWarning, match="not known"):
        report = pipeline.run(data, test=TEST, horizon=1, save_dir=tmp_path)
    manifest = json.loads((tmp_path / "m.manifest.json").read_text())
    assert {"signal_d-1", "no_rule_value", "no_rule_published"} <= set(manifest["column_hashes"])
    listed = report.availability().set_index("predictor")
    assert (listed.loc["no_rule_value", "status"], listed.loc["no_rule_value", "role"]) == ("no rule", "ctx.value")
    assert (listed.loc["no_rule_published", "status"], listed.loc["no_rule_published", "role"]) == (
        "no rule",
        "ctx.published",
    )
    assert listed.loc["no_rule_value", "periods"] == "0-23"


# ---------------------------------------------------------------------------
# The order of a list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("train_on", ["all", "matching"])
def test_the_same_names_in_another_order_warn_whatever_the_rule(frame, train_on):
    def by_day(ctx):
        return Use(PREDICTORS if ctx.issue_day.day % 2 else PREDICTORS[::-1], train_on=train_on)

    with pytest.warns(UserWarning, match="another order.*Sort the names"):
        OLSModel(by_day, training_window=WINDOW)._plan_predictors(frame, "price", 1, "1h", INFO, *TEST)


def test_another_order_on_training_days_alone_warns_and_other_lists_do_not(frame, recwarn):
    first = datetime.date.fromisoformat(TEST[0])

    def past_days_reversed(ctx):
        return Use(PREDICTORS[::-1] if ctx.issue_day < first else PREDICTORS, train_on="matching")

    with pytest.warns(UserWarning, match="another order"):
        OLSModel(past_days_reversed, training_window=WINDOW)._plan_predictors(frame, "price", 1, "1h", INFO, *TEST)

    recwarn.clear()
    OLSModel(mixed_lists, training_window=WINDOW)._plan_predictors(frame, "price", 1, "1h", INFO, *TEST)
    OLSModel(weekend_split, training_window=WINDOW)._plan_predictors(frame, "price", 1, "1h", INFO, *TEST)
    assert not [w for w in recwarn if "another order" in str(w.message)]


# ---------------------------------------------------------------------------
# What a fit can't do on few rows
# ---------------------------------------------------------------------------


def test_lasso_below_its_cv_rows_is_counted_as_a_fit_error_and_the_warning_says_it_can_be(frame):
    model = LassoCVModel(a_list_starts_in_the_test, training_window=WINDOW, cv=5, name="lasso")
    with pytest.warns(UserWarning, match=r"lasso: 96 forecast\(s\).*a fit can fail.*fit_error"):
        ref = model.run(frame, "2023-04-12", "2023-04-16", horizon=1, information_set=INFO)
    assert ref.skipped == {"fit_error": 96}  # 1, 2, 3 and 4 days fail, and 5 days fit
    assert {r["training_days"] for r in ref._results} == {5}


# ---------------------------------------------------------------------------
# What a merged report says
# ---------------------------------------------------------------------------


def test_a_report_on_part_of_a_run_counts_the_forecasts_of_that_part(frame, tmp_path):
    pipeline = (
        ModelPipeline(information_set=INFO)
        .add_model(OLSModel(a_list_starts_in_the_test, training_window=WINDOW, name="starts"))
        .add_model(OLSModel(PREDICTORS, training_window=WINDOW, name="fixed"))
        .add_evaluator(MAEEvaluator())
    )
    with pytest.warns(UserWarning, match="no more days"):
        wide = pipeline.run(frame, test=("2023-04-07", "2023-04-16"), horizon=2, save_dir=tmp_path)
    assert "no_matching_days" in wide.forecast_counts().columns

    # At horizon 1 the new list has no forecast on 10 and 11 April, at horizon 2 none up to the 12th: 5 days of 6.
    part = ("2023-04-10", "2023-04-12")
    with pytest.warns(UserWarning, match=r"starts \(24 of 144, 120 missing\)"):
        narrower = pipeline.report(tmp_path, test=part, horizon=2)
    counts = narrower.forecast_counts().set_index("model")
    forecasts = narrower.summary().set_index("model")["forecasts"]
    assert counts["produced"].to_dict() == forecasts.to_dict() == {"starts": 24, "fixed": 144}
    assert counts["planned"].to_dict() == {"starts": 144, "fixed": 144}
    assert counts["missing"].to_dict() == {"starts": 120, "fixed": 0}
    # What the manifest says is about the run's period, so this report doesn't pretend it is about this part.
    assert "no_matching_days" not in counts.columns

    whole = pipeline.report(tmp_path, test=("2023-04-07", "2023-04-16"), horizon=2)
    pd.testing.assert_frame_equal(whole.forecast_counts(), wide.forecast_counts())


class FailsOnFewRows(OLSModel):
    """A ValueError from the fit, as scikit-learn raises on degenerate input, below a number of rows the environment gives."""

    def _fit_predict(self, train_x, train_y, test_x):
        if train_x.shape[0] < int(os.environ.get("FIT_ERROR_BELOW", 0)):
            raise ValueError("too few rows for this model")
        return super()._fit_predict(train_x, train_y, test_x)


def manifest_of(path):
    return json.loads((Path(path) / "m.manifest.json").read_text())


def test_skip_counts_go_in_the_manifest_per_window_and_a_run_replaces_its_own(frame, tmp_path, monkeypatch):
    validation, test = ("2023-03-31", "2023-04-03"), ("2023-04-10", "2023-04-14")
    pipeline = ModelPipeline(information_set=INFO).add_model(
        FailsOnFewRows(weekend_split, training_window=WINDOW, name="m")
    )

    monkeypatch.setenv("FIT_ERROR_BELOW", "20")  # the weekend list's forecasts have fewer rows than that
    report = pipeline.run(frame, horizon=1, validation=validation, test=test, save_dir=tmp_path)
    in_test = report.forecast_counts().iloc[0]["fit_error"]
    in_validation = report.validation.forecast_counts().iloc[0]["fit_error"]
    assert in_test == 24 and in_validation == 48  # the Friday in the test; the Friday and Saturday in validation
    assert manifest_of(tmp_path)["skipped"] == {"test": {"fit_error": 24}, "validation": {"fit_error": 48}}

    merged = pipeline.report(tmp_path, validation=validation, test=test, horizon=1)
    pd.testing.assert_frame_equal(merged.forecast_counts(), report.forecast_counts())
    pd.testing.assert_frame_equal(merged.validation.forecast_counts(), report.validation.forecast_counts())

    # A resume tries them again, and what it finds replaces the counts: here the same.
    resumed = pipeline.run(frame, horizon=1, validation=validation, test=test, save_dir=tmp_path)
    assert resumed.forecast_counts().iloc[0]["fit_error"] == 24
    assert manifest_of(tmp_path)["skipped"] == {"test": {"fit_error": 24}, "validation": {"fit_error": 48}}

    # A run naming one window replaces that window's alone.
    monkeypatch.delenv("FIT_ERROR_BELOW")
    pipeline.run(frame, horizon=1, test=test, save_dir=tmp_path)
    assert manifest_of(tmp_path)["skipped"] == {"validation": {"fit_error": 48}}
    merged = pipeline.report(tmp_path, validation=validation, test=test, horizon=1)
    assert "fit_error" not in merged.forecast_counts().columns
    assert merged.validation.forecast_counts().iloc[0]["fit_error"] == 48

    # And nothing skipped leaves no key at all.
    pipeline.run(frame, horizon=1, validation=validation, test=test, save_dir=tmp_path)
    assert "skipped" not in manifest_of(tmp_path)


def test_a_crash_records_no_skips_so_what_is_missing_is_not_explained_away(frame, tmp_path, monkeypatch):
    days = ("2023-04-07", "2023-04-14")
    data = frame.iloc[:2800]
    real_pool(monkeypatch, 1)

    def pipeline():
        return ModelPipeline(information_set=INFO).add_model(
            CrashesAfterSomeFits(a_list_starts_in_the_test, training_window=WINDOW, name="m")
        )

    monkeypatch.setenv("CRASH_AFTER", "100")
    monkeypatch.setattr(CrashesAfterSomeFits, "saved_to", tmp_path / "m.jsonl")
    with pytest.raises(RuntimeError, match="the machine went down"):
        pipeline().run(data, test=days, horizon=1, save_dir=tmp_path)
    monkeypatch.delenv("CRASH_AFTER")
    assert "skipped" not in manifest_of(tmp_path)
    with pytest.warns(UserWarning, match=r"m \(\d+ of 192, \d+ missing\)"):
        crashed = pipeline().report(tmp_path, test=days, horizon=1)
    assert "no_matching_days" not in crashed.forecast_counts().columns

    # The run finished this time: 2 days without a forecast, and the merge can name them.
    pipeline().run(data, test=days, horizon=1, save_dir=tmp_path)
    assert manifest_of(tmp_path)["skipped"] == {"test": {"no_matching_days": 48}}
    with pytest.warns(UserWarning, match=r"m \(144 of 192, 48 missing, no_matching_days 48\)"):
        merged = pipeline().report(tmp_path, test=days, horizon=1)
    assert merged.forecast_counts().iloc[0]["no_matching_days"] == 48


def test_skip_counts_say_nothing_about_a_stored_forecast():
    manifest = {"model_class": "OLSModel", "created_at": "now"}
    assert manifest_digest({**manifest, "skipped": {"test": {"fit_error": 3}}}) == manifest_digest(manifest)


def test_a_stored_record_without_its_matching_days_is_refused(frame, tmp_path):
    path = tmp_path / "m.jsonl"
    make = lambda: OLSModel(one_list, training_window=WINDOW)  # noqa: E731
    make().run(frame, *TEST, horizon=1, information_set=INFO, save_to=str(path))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    del records[0]["matching_days"]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ManifestMismatchError, match=r"1 stored forecast\(s\) .* have other matching days"):
        make().run(frame, *TEST, horizon=1, information_set=INFO, save_to=str(path))
