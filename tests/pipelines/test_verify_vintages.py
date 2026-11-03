"""`verify` and a revised series read on a day a `train_on="matching"` list trains on (decision 19 of the step 3 spec).

The frame holds such a value as revised later, not as it was out on that day, so a forecast's set of days can be one no live run
would have chosen, and no perturbation shows it: the value is known by its rule either way. A read of a column of class
`latest_revision` or `last_write_stamp` there makes the model `inconclusive`, naming the column. A `snapshot` or `row_stamp`, a
class not declared, a read on the issue day alone and a list that trains on every day leave it `ok`, and in each of those the
positive control moves the forecasts, so the `ok` isn't one the check couldn't help giving.
"""

from concurrent.futures import Future

import numpy as np
import pytest

from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import OLSModel, Use
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from tests.synthetic import calendar_shift, make_market_data

PL = "Europe/Warsaw"
TEST = ("2023-04-05", "2023-04-14")
TWO = ["load_d+{horizon}", "wind_d+{horizon}"]


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


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)


def info(vintage):
    rules = {
        "load": Availability.published("10:00", 7),
        "wind": Availability.published("10:00", 7),
        "is_weekend": Availability.static(),
        "is_weekday": Availability.static(),
        "price": Availability.published("13:00", 1),
        "signal": Availability.published("10:00", 0, vintage=vintage),  # `signal_d-1` is known the day before its row
    }
    return InformationSet("11:45", PL, rules=rules)


def matching_on_the_signal(ctx):
    names = [*TWO, "is_weekend_d+{horizon}"] if ctx.value("signal_d-1") > 0 else [*TWO, "is_weekday_d+{horizon}"]
    return Use(names, train_on="matching")


def all_on_the_signal(ctx):
    return Use(matching_on_the_signal(ctx).predictors, train_on="all")


def published_signal_matching(ctx):
    names = [*TWO, "is_weekend_d+{horizon}"] if ctx.published("signal_d-1") else list(TWO)
    return Use(names, train_on="matching")


def issue_day_only(ctx):
    """Reads the signal on the last day of the test alone, verified on its own: no day it trains on reads it."""
    if ctx.issue_day.isoformat() == TEST[1]:
        ctx.value("signal_d-1")
    return Use([*TWO, "is_weekend_d+{horizon}"], train_on="matching")


def verified(function, vintage):
    pipeline = ModelPipeline(information_set=info(vintage)).add_model(OLSModel(function, training_window=30, name="m"))
    test = (TEST[1], TEST[1]) if function is issue_day_only else TEST
    return pipeline.verify(frame_data(), *test, horizon=1, days=1)


_FRAME = {}


def frame_data():
    if "data" not in _FRAME:
        data = make_market_data(noise=5.0, seed=7, horizon=1)
        data["signal"] = np.random.default_rng(11).normal(0, 1, len(data))
        data["signal_d-1"] = calendar_shift(data["signal"], -1)
        _FRAME["data"] = data
    return _FRAME["data"]


@pytest.mark.parametrize("vintage", ["latest_revision", "last_write_stamp"])
@pytest.mark.parametrize("function", [matching_on_the_signal, published_signal_matching])
def test_a_revised_column_read_on_a_training_day_of_a_matching_list_is_inconclusive(function, vintage):
    report = verified(function, vintage)
    assert report.status("m") == "inconclusive"
    problems = [line for line in report.problems() if "signal_d-1" in line]
    assert len(problems) == 1 and f"'signal_d-1': '{vintage}'" in problems[0] and "train_on='matching'" in problems[0]
    assert report.checks[0].revised_reads == {"signal_d-1": vintage}
    assert report.table().iloc[0]["control moved"] > 0  # the only thing keeping it from ok


@pytest.mark.parametrize(
    "function, vintage",
    [
        (matching_on_the_signal, "snapshot"),
        (matching_on_the_signal, "row_stamp"),
        (matching_on_the_signal, None),  # a class not declared doesn't count
        (all_on_the_signal, "latest_revision"),  # no list trains on the days it chose by the value
        (issue_day_only, "latest_revision"),  # the issue day's read is the forecast's own
    ],
)
def test_a_snapshot_an_undeclared_class_and_a_list_on_every_day_are_ok_and_the_control_moves(function, vintage):
    report = verified(function, vintage)
    assert report.status("m") == "ok", report.problems()
    assert report.checks[0].revised_reads == {}
    assert report.table().iloc[0]["control moved"] > 0
    # The same function and frame with a revised class: inconclusive where a matching list reads it on a training day, so the
    # ok above is one the check could have withheld.
    revised = verified(function, "latest_revision").status("m")
    reads_on_training_days = function is matching_on_the_signal
    assert revised == ("inconclusive" if reads_on_training_days else "ok")


def test_a_read_outside_the_sample_counts_too():
    """The day sampled trains on every day, so it plans no training day; the last day of the period, not sampled, has a matching
    list. The whole period is planned too, and what the function reads on the days that list trains on counts as on a sampled
    day."""

    def late_in_the_period(ctx):
        chosen = matching_on_the_signal(ctx)  # the signal is read on every day asked
        return chosen if ctx.issue_day.isoformat() == TEST[1] else Use(chosen.predictors, train_on="all")

    pipeline = ModelPipeline(information_set=info("latest_revision"))
    pipeline.add_model(OLSModel(late_in_the_period, training_window=30, name="m"))
    report = pipeline.verify(frame_data(), *TEST, horizon=1, days=1, seed=0)
    assert TEST[1] not in report.issue_days
    assert report.status("m") == "inconclusive"
    assert report.checks[0].revised_reads == {"signal_d-1": "latest_revision"}
