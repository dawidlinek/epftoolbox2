"""Binary-column masks: the day a list's mask is read, and the re-read of a column with no value there.

Per list L, (period, horizon) and window (`BaseModel._plan_scalable_masks`, `BaseModel._scalable_mask`): the mask day m is the
window's first calendar day if L is the list of the window's first planned issue day, else the first planned issue day that uses
L; a key the clock skips has no say. A column with no value in the training rows of m is read on the first calendar day after m
whose rows have one, for fixed lists and functions alike, however often the list is used.
"""

import datetime
import json
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import OLSModel
from epftoolbox2.models._worker import _worker_init
from tests.synthetic import make_market_data

FRIDAY = 4
DRIVERS = ["load_d+{horizon}", "wind_d+{horizon}"]
SPRING_SWITCH = datetime.date(2023, 3, 26)  # Europe/Warsaw skips 02:00, period 2
RULES = {
    "load": Availability.published("10:00", 10),
    "wind": Availability.published("10:00", 10),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "late": Availability.static(),
    "regime": Availability.static(),
    "aux": Availability.static(),
    "price": Availability.published("13:00", 1),  # D's price is known at 11:45 on D: what "published" trains on
}
INFO = InformationSet("11:45", "Europe/Brussels", rules=RULES)


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


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)


def local_dates(data):
    return np.asarray(pd.DatetimeIndex(data.index.tz_localize(None)).normalize())


def by_key(records):
    """Records by forecast key, each as its JSON text, so that a NaN `actual` equals itself."""
    return {(r["run_date"], r["hour"], r["horizon"]): json.dumps(r, sort_keys=True) for r in records}


def fridays(keyed):
    return {key: record for key, record in keyed.items() if json.loads(record)["run_weekday"] == FRIDAY}


def fits(records):
    """Key -> (prediction, coefficients): what a forecast is, apart from its `actual`, which is read after D."""
    return {(r["run_date"], r["hour"], r["horizon"]): (r["prediction"], r["coefficients"]) for r in records}


# ---------------------------------------------------------------------------
# The known issue this fixes: whether a list was used on every issue day decided the re-read
# ---------------------------------------------------------------------------


def test_a_later_end_of_the_test_moves_no_forecast_through_a_mask_column_with_no_value():
    """`late` has no value before 1 March, the first issue day, so a mask read there can't tell whether it is binary. It used to
    stay unscaled for a list used on every issue day, as for a fixed list, and be read again on the first of the list's days
    with a value for any other list; whether the list was used on every day depended on the last issue day, so the forecasts of
    5 April moved when the test went on to 12 April, where another list starts (24 of 24 by up to 0.25 here, 22 of 24 by up to
    0.14 with LassoCV). Now it is read on the first calendar day with a value whatever the list's later use."""
    data = make_market_data(start="2023-01-01", days=120, horizon=1, noise=5.0, seed=3)
    dates = local_dates(data)
    values = dates >= np.datetime64("2023-03-01")
    late = np.full(len(data), np.nan)
    late[values] = np.random.default_rng(4).normal(size=int(values.sum()))
    data = data.assign(late=late)
    drivers = ["load_d+1", "wind_d+1"]

    def before_10_april(ctx):
        return drivers + ["late"] if ctx.issue_day < datetime.date(2023, 4, 10) else drivers

    def forecasts_of_5_april(test_end):
        records = OLSModel(before_10_april, training_window=40).run(data, "2023-03-01", test_end, horizon=1)._results
        return {r["hour"]: r["prediction"] for r in records if r["run_date"] == "2023-04-05"}

    shorter = forecasts_of_5_april("2023-04-07")
    assert len(shorter) == 24
    assert forecasts_of_5_april("2023-04-12") == shorter
