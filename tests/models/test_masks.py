"""Binary-column masks: the day a list's mask is read, and the re-read of a column with no value there.

Per list L, (period, horizon) and window (`BaseModel._plan_scalable_masks`, `BaseModel._scalable_mask`): the mask day m is the
window's first calendar day if L is the list of the window's first planned issue day, else the first planned issue day that uses
L; a key the clock skips has no say. A column with no value in the training rows of m is read on the first calendar day after m
whose rows have one, for fixed lists and functions alike, however often the list is used. Nothing after a forecast's issue day D
decides how it is scaled, so these tests change everything after D and hold D's forecasts bit for bit; a run of Fridays reads
the column where a run of every day does; an answer on a key the clock skips moves no mask; and a fixed list's column that starts
inside the test is scaled from the day it has a value, by hand.
"""

import datetime
import json
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import OLSModel, Use
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.models.base import BaseModel
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from tests.synthetic import PREDICTORS, make_market_data

FRIDAY = 4
DRIVERS = ["load_d+{horizon}", "wind_d+{horizon}"]
SPRING_SWITCH = datetime.date(2023, 3, 26)  # Europe/Warsaw skips 02:00, period 2
RULES = {
    "load": Availability.published("10:00", 10),
    "wind": Availability.published("10:00", 10),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "late": Availability.static(),
    "turns": Availability.static(),
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


# ---------------------------------------------------------------------------
# Nothing after D moves D's forecasts
# ---------------------------------------------------------------------------

START, END = "2023-03-20", "2023-04-14"  # a Monday, over the spring switch of 26 March
LATE_FROM = "2023-03-21"  # `late` starts inside the test: a list holding it has no value of it on the mask day
MASKED = PREDICTORS + ["late", "turns"]


def while_aux(ctx):
    return MASKED if ctx.published("aux") else PREDICTORS


def matching_while_aux(ctx):
    return Use(MASKED, train_on="matching") if ctx.published("aux") else PREDICTORS


LISTS = {
    "fixed": (MASKED, "d-1"),
    "function": (while_aux, "d-1"),
    "matching": (matching_while_aux, "d-1"),
    "published": (while_aux, "published"),
}
# Every level of every axis and every pair of levels (kind of list x issue days x horizon), in half of the cells.
CELLS = [
    ("fixed", None, 1),
    ("fixed", ["friday"], 3),
    ("function", None, 3),
    ("function", ["friday"], 1),
    ("matching", None, 1),
    ("matching", ["friday"], 3),
    ("published", None, 3),
    ("published", ["friday"], 1),
]


@pytest.fixture(scope="module")
def late_market():
    """`late`, continuous, and `turns`: no value before 21 March, then `turns` is 0 to 24 March, 1 to 28 March and continuous from
    29 March. A list holding them has no value of either on the window's first day. `late` is standardised once it is read, where
    the old rule kept it raw in a list used on every issue day; `turns` is a 0/1 column for the forecasts of 24 and 31 March, read
    on the right day, and not if read on a day after D or on rows a change after D reaches."""
    data = make_market_data(start="2023-01-15", days=95, horizon=3, noise=5.0, seed=3)
    dates = local_dates(data)
    on = dates >= np.datetime64(LATE_FROM)
    late = np.full(len(data), np.nan)
    late[on] = np.random.default_rng(4).normal(size=int(on.sum()))
    turns = np.where(on, 0.0, np.nan)
    turns[dates >= np.datetime64("2023-03-25")] = 1.0
    later = dates >= np.datetime64("2023-03-29")
    turns[later] = np.random.default_rng(5).normal(size=int(later.sum()))
    return data.assign(late=late, turns=turns, aux=1.0)


@pytest.mark.parametrize(("lists", "issue_days", "horizon"), CELLS)
def test_nothing_after_the_issue_day_moves_its_forecasts(late_market, lists, issue_days, horizon):
    """For D = 24 and 31 March: every finite value dated after D changed (missing values kept), `aux` missing after D (which
    changes a function's list after D), and the test ending on D. D's predictions and coefficients stay bit for bit, and so do
    its keys. `late` and `turns` have no value on the window's first day, and their first on 21 March, so at h=3 the forecasts of
    24 March have no row with them and those of 31 March read their mask from a later day than the first."""
    fn, newest = LISTS[lists]
    days = {} if issue_days is None else {"issue_days": issue_days}

    def run(frame, end=END):
        model = OLSModel(fn, training_window=30, newest_target=newest)
        return fits(model.run(frame, START, end, horizon=horizon, information_set=INFO, **days)._results)

    base = run(late_market)
    dates = local_dates(late_market)
    for day in ("2023-03-24", "2023-03-31"):
        after = dates > np.datetime64(day)
        changed = late_market.copy()
        columns = [c for c in late_market.columns if late_market[c].dtype.kind == "f"]
        values = changed[columns].to_numpy()
        shifted = after[:, None] & np.isfinite(values)
        values[shifted] += np.random.default_rng(1).normal(0, 3, size=int(shifted.sum()))
        changed[columns] = values
        blank = late_market.copy()
        blank.loc[after, "aux"] = np.nan
        on_d = {key: fit for key, fit in base.items() if key[0] == day}
        assert len(on_d) >= 24  # at h=3 some forecasts of 24 March with `late` have no row yet
        for what, frame, end in (("values", changed, END), ("aux", blank, END), ("end", late_market, day)):
            other = run(frame, end)
            assert {key: fit for key, fit in other.items() if key[0] == day} == on_d, (day, what)
            if what == "values":  # the change is seen after D
                assert any(other.get(key) != fit for key, fit in base.items() if key[0] > day)


# ---------------------------------------------------------------------------
# A run of Fridays reads the column where a run of every day does
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def turning_market():
    """`late`: no value before 21 March, 0 on 21 March, continuous from 22 March. At h=1 the rows of 23 March (to 21 March) hold
    only the 0, a binary column, and those of 24 March (to 22 March) a continuous value: read on the first calendar day with a
    value, Thursday 23 March, it stays raw; read on the first Friday it would be standardised. `aux` is missing on 21, 22 and
    29 March, none of them a Friday."""
    data = make_market_data(start="2023-01-15", days=95, horizon=3, noise=5.0, seed=3)
    dates = local_dates(data)
    late = np.full(len(data), np.nan)
    late[dates == np.datetime64("2023-03-21")] = 0.0
    later = dates >= np.datetime64("2023-03-22")
    late[later] = np.random.default_rng(4).normal(size=int(later.sum()))
    aux = np.ones(len(data))
    aux[np.isin(dates, np.array(["2023-03-21", "2023-03-22", "2023-03-29"], dtype="datetime64[ns]"))] = np.nan
    return data.assign(late=late, aux=aux)


def late_while_aux(ctx):
    return DRIVERS + ["late"] if ctx.published("aux") else DRIVERS


@pytest.mark.parametrize("lists", ["fixed", "function"])
def test_a_friday_run_reads_a_column_with_no_value_where_a_run_of_every_day_does(turning_market, lists):
    """The drivers alone and `late`: with the two weekday dummies, which a model leaves raw and which sum to 1, OLS spans a
    constant and gives a raw and a standardised `late` the same forecasts, so no mask could be seen in them."""
    fn = DRIVERS + ["late"] if lists == "fixed" else late_while_aux

    def run(start=START, **days):
        model = OLSModel(fn, training_window=30)
        return by_key(model.run(turning_market, start, END, horizon=3, information_set=INFO, **days)._results)

    every_day = run()
    on_friday = run(issue_days=["friday"])
    assert on_friday == fridays(every_day)
    assert {key[0] for key in on_friday} == {"2023-03-24", "2023-03-31", "2023-04-07", "2023-04-14"}

    # What makes this test see the day: a test from the first Friday reads the mask there, with the continuous value of 22
    # March in its rows at h=1, and its forecasts move at h=1 alone.
    from_friday = run(start="2023-03-24", issue_days=["friday"])
    moved = {
        key[2]
        for key, record in from_friday.items()
        if json.loads(record)["prediction"] != json.loads(on_friday[key])["prediction"]
    }
    assert moved == {1}


# ---------------------------------------------------------------------------
# A key the clock skips has no say
# ---------------------------------------------------------------------------


def skipped_by_the_clock(ctx):
    """02:00 on 26 March: the issue day or the target day of the key has no such hour."""
    return ctx.period == 2 and SPRING_SWITCH in (ctx.issue_day, ctx.target_day)


def masked_with_regime(ctx):
    return DRIVERS + ["regime"]


def other_where_skipped_regime(ctx):
    return DRIVERS if skipped_by_the_clock(ctx) else DRIVERS + ["regime"]


def masked_with_late(ctx):
    return DRIVERS + ["late"]


def other_where_skipped_late(ctx):
    return DRIVERS if skipped_by_the_clock(ctx) else DRIVERS + ["late"]


@pytest.fixture(scope="module")
def skip_market():
    """`regime`: the weekend flag before 22 March and continuous from then on, so a mask read on 24 March at h=2 (rows to 21
    March) leaves it raw and one read on 25 March (rows to 22 March) standardises it. `late`: no value before 21 March."""
    data = make_market_data(start="2023-01-15", days=95, horizon=3, noise=5.0, seed=3)
    dates = local_dates(data)
    regime = data["is_weekend"].to_numpy(dtype=float).copy()
    later = dates >= np.datetime64("2023-03-22")
    regime[later] += np.linspace(0.1, 0.9, int(later.sum()))
    on = dates >= np.datetime64(LATE_FROM)
    late = np.full(len(data), np.nan)
    late[on] = np.random.default_rng(4).normal(size=int(on.sum()))
    return data.assign(regime=regime, late=late)


SKIPS = {
    # The window's first issue day, 24 March, aims at the skipped hour at h=2: another list there used to move the mask day of
    # the list of every planned key from the window's first day to 25 March.
    "first issue day": (masked_with_regime, other_where_skipped_regime, "2023-03-24", "2023-03-31", 2, ["regime"]),
    # In the middle of a run of every day: another list on the skipped keys used to make the list "not used on every day", and so
    # read `late` again where a list used on every day kept it raw.
    "inside the test": (masked_with_late, other_where_skipped_late, START, END, 1, ["late"]),
}


@pytest.mark.parametrize("case", list(SKIPS))
def test_an_answer_on_a_key_the_clock_skips_moves_no_mask(skip_market, tmp_path, case):
    same, other, start, end, horizon, column = SKIPS[case]

    def run(fn, name):
        save_to = tmp_path / f"{name}.jsonl"
        records = OLSModel(fn, training_window=30).run(skip_market, start, end, horizon=horizon, save_to=str(save_to))
        lists = json.loads(manifest_path(save_to).read_text())["predictor_lists"]
        masks = {entry["hash"]: entry["binary_masks"] for entry in lists if entry["forecasts"]}
        return by_key(records.iter_records()), masks

    everywhere, masks = run(same, "same")
    skipped, other_masks = run(other, "other")
    assert skipped == everywhere
    assert other_masks == masks
    fixed = OLSModel(DRIVERS + column, training_window=30).run(skip_market, start, end, horizon=horizon)._results
    assert by_key(fixed) == {
        key: json.dumps({k: v for k, v in json.loads(r).items() if k != "predictor_list"}, sort_keys=True)
        for key, r in everywhere.items()
    }


# ---------------------------------------------------------------------------
# A fixed list's column that starts on the third day, by hand
# ---------------------------------------------------------------------------

THIRD = ("2023-03-01", "2023-03-12")  # no clock change
WINDOW = 40


def third_day_market():
    """`late`: its first value on 1 March, the first day of the test, continuous from then on."""
    data = make_market_data(start="2023-01-01", days=80, horizon=1, noise=5.0, seed=3)
    on = local_dates(data) >= np.datetime64(THIRD[0])
    late = np.full(len(data), np.nan)
    late[on] = np.random.default_rng(4).normal(size=int(on.sum()))
    return data.assign(late=late)


def hand_forecast(data, period, day, names, scaled):
    """OLS on the standardised target without an intercept, from the rows of `period` dated D-W-1 to D-2 (h=1) with every value,
    the columns `scaled` standardised on those rows and the others as they are."""
    wall = data.index.tz_localize(None)
    dates = wall.normalize()
    here = np.asarray(wall.hour == period)
    D = pd.Timestamp(day)
    rows = np.flatnonzero(
        here & np.asarray((dates >= D - pd.Timedelta(days=WINDOW + 1)) & (dates <= D - pd.Timedelta(days=2)))
    )
    prices = pd.Series(data["price"].to_numpy(), index=wall)
    target = prices.reindex(wall[rows] + pd.Timedelta(days=1)).to_numpy()
    x = data[names].to_numpy(dtype=float)
    usable = np.isfinite(target) & np.isfinite(x[rows]).all(axis=1)
    y, train = target[usable], x[rows][usable]
    means = np.where(scaled, train.mean(axis=0), 0.0)
    deviations = np.where(scaled, train.std(axis=0, ddof=1), 1.0)
    coefficients, *_ = np.linalg.lstsq((train - means) / deviations, (y - y.mean()) / y.std(ddof=1), rcond=None)
    test_row = np.flatnonzero(here & np.asarray(dates == D))[-1]
    return float(((x[test_row] - means) / deviations) @ coefficients * y.std(ddof=1) + y.mean())


def test_a_fixed_list_s_column_that_starts_on_the_third_day_is_scaled_from_there():
    """`late` has its first value on 1 March, the first day of the test, so at h=1 the rows of 1 and 2 March (dated up to 27 and
    28 February) have none, and those of 3 March (up to 1 March) have one. Its mask is read there and standardises it, where a
    fixed list used to keep it raw for good. Forecasts of 1 and 2 March have no finite row and are skipped."""
    data = third_day_market()
    names = ["load_d+1", "wind_d+1", "late"]

    ref = OLSModel(names, training_window=WINDOW).run(data, *THIRD, horizon=1)
    records = by_key(ref._results)
    assert {key[0] for key in records} == {f"2023-03-{day:02d}" for day in range(3, 13)}
    assert ref.skipped == {"no_finite_training_rows": 48}
    for day in ("2023-03-07", "2023-03-12"):
        for period in (0, 9, 18):
            prediction = json.loads(records[(day, period, 1)])["prediction"]
            assert prediction == pytest.approx(hand_forecast(data, period, day, names, [True] * 3), abs=1e-9)
            raw = hand_forecast(data, period, day, names, [True, True, False])
            assert abs(prediction - raw) > 1e-6


# ---------------------------------------------------------------------------
# Results scaled by the old rule aren't resumed silently
# ---------------------------------------------------------------------------


def stored(path):
    with open(path) as lines:
        return by_key(json.loads(line) for line in lines if line.strip())


def cut_short(path, before="2023-03-08"):
    """What a run stopped before `before` leaves in the file."""
    with open(path) as lines:
        kept = [line for line in lines if line.strip() and json.loads(line)["run_date"] < before]
    with open(path, "w") as out:
        out.writelines(kept)
    return len(kept)


@pytest.mark.parametrize(
    "names", [["load_d+1", "wind_d+1", "late"], ["load_d+1", "wind_d+1"]], ids=["late", "complete"]
)
def test_plain_results_of_a_fixed_list_scaled_by_the_old_rule_are_refused_on_resume(tmp_path, monkeypatch, names):
    """Plain results of fixed lists have no mask digest to compare, so a run made before a column with no value on the first day
    was read again, cut short and resumed by this code, would have mixed the two scalings without a word. Such a run now records
    its digests, and results whose manifest has none are refused; with every column there from the first day nothing was read
    again, the manifest is as it was and the resume gives a fresh run's records."""
    data = third_day_market()
    original = BaseModel._precompute_scalable_masks

    def without_reread(self, horizons, offset=None, last=None):
        offset = self._offset if offset is None else offset
        return original(self, horizons, offset, offset)

    def run(path, **kwargs):
        return OLSModel(names, training_window=WINDOW).run(data, *THIRD, horizon=1, save_to=str(path), **kwargs)

    old = tmp_path / "old" / "m.jsonl"
    with monkeypatch.context() as patch:
        patch.setattr(BaseModel, "_precompute_scalable_masks", without_reread)
        run(old)
    assert "binary_masks" not in json.loads(manifest_path(old).read_text())
    assert cut_short(old) > 0
    fresh = by_key(OLSModel(names, training_window=WINDOW).run(data, *THIRD, horizon=1)._results)

    # This code's own run, cut short, resumes whatever the columns.
    new = tmp_path / "new" / "m.jsonl"
    run(new)
    digests = json.loads(manifest_path(new).read_text()).get("binary_masks")
    cut_short(new)
    run(new)
    assert stored(new) == fresh

    if "late" in names:
        assert sorted(digests) == sorted(f"{period}-1" for period in range(24))
        with pytest.raises(ManifestMismatchError, match="the scaling rule changed"):
            run(old)
        run(old, allow_manifest_mismatch=True)  # an override resumes, and its manifest has the digests from then on
        assert json.loads(manifest_path(old).read_text())["binary_masks"] == digests
        run(old)
    else:
        assert digests is None
        run(old)
        assert stored(old) == fresh
