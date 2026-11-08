"""Block predict (step 6d): a model whose class defines `_predict_block` predicts each refit unit's block in one call.

The call holds the issue rows of every day of the unit's block that the frame has with finite inputs, whichever of them the run
forecasts, in date order, padded to the rule's nominal length (7 for `"7D"`, 31 for `"1MS"`) with copies of the first row; each
forecast is taken from its own row. So a call has one shape and the same rows alone, in the full block, from any test start, on
Fridays only and after a resume, and only the frame's end or a missing input changes which days are in it. No target and no row
after the block enters a call.

`KernelModel` stands in for an in-context model (6e): its fit keeps the training rows (the context) and a forecast is the mean of
the training targets weighted by a Gaussian kernel of the distance to its row, each row on its own. `by_length` and `by_call` make
it see its call (its length, the mean of a column over its rows) by a tiny amount, so a test can tell whether two calls were the
same: what a real model's last bits would show.
"""

import datetime
import json
from concurrent.futures import Future
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.models import LassoCVModel, LEARModel, NaiveModel, OLSModel, RidgeModel
from epftoolbox2.models import _worker as worker
from epftoolbox2.models._calendar import (
    FitUnit,
    block_days,
    nominal_block_length,
    refit_anchor,
    refit_block_end,
    unit_seed,
)
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.models.base import BaseModel
from tests.synthetic import PREDICTORS, make_market_data

WINDOW = 28
FIRST, LAST = "2023-03-13", "2023-04-02"  # three "7D" blocks, Monday to Sunday
WEEK = ("2023-03-20", "2023-03-26")  # the spring switch is on its Sunday (Europe/Warsaw): no 02:00 on 26 March
SEES_CALL = {"by_length": 1e-12, "by_call": 1e-6}


class InProcessPool:
    """Stands in for the worker pool, as in tests/models/test_refit.py: the fits run in this process, so the spies see them."""

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
    KernelModel.calls.clear()
    yield
    KernelModel.calls.clear()


class KernelModel(BaseModel):
    """A numpy in-context regressor: the fit is the training rows, a forecast the kernel-weighted mean of their targets, each
    row on its own (`_kernel_mean`), so one at a time and in a block give the same bits. Every block call's standardised rows
    are kept in `calls` (this process's runs, in order). `by_length` adds that times the call's row count to each of its
    predictions and `by_call` that times the mean of its second column (wind)."""

    calls: List[np.ndarray] = []

    def __init__(
        self,
        predictors=PREDICTORS,
        training_window=WINDOW,
        *,
        bandwidth=2.0,
        by_length=0.0,
        by_call=0.0,
        block_predict=True,
        **kwargs,
    ):
        super().__init__(predictors, training_window, **kwargs)
        self.bandwidth = bandwidth
        self.by_length = by_length
        self.by_call = by_call
        self.block_predict = block_predict

    @property
    def _model_kwargs(self):
        # The worker rebuilds the model from these: `block_predict` must be one of them.
        return {
            "bandwidth": self.bandwidth,
            "by_length": self.by_length,
            "by_call": self.by_call,
            "block_predict": self.block_predict,
        }

    def _fit_predict(self, train_x, train_y, test_x):
        return self._predict_fitted(self._fit_model(train_x, train_y), test_x)

    def _fit_model(self, train_x, train_y):
        return train_x.copy(), train_y.copy()

    def _kernel_mean(self, fitted, row):
        context_x, context_y = fitted
        distance = ((context_x - row) ** 2).sum(axis=1)
        weights = np.exp(-(distance - distance.min()) / (2 * self.bandwidth**2))
        return float(weights @ context_y / weights.sum())

    def _predict_fitted(self, fitted, test_x):
        return self._kernel_mean(fitted, test_x[0]), []

    def _predict_block(self, fitted, rows):
        type(self).calls.append(rows.copy())
        seen = 0.0  # each term only when asked for, so that the plain model reads no other row (0 * NaN is NaN)
        if self.by_length:
            seen += self.by_length * len(rows)
        if self.by_call:
            seen += self.by_call * float(rows[:, 1].mean())
        return np.array([self._kernel_mean(fitted, row) + seen for row in rows]), [[] for _ in rows]


def kernel(refit="7D", **kwargs):
    return KernelModel(refit=refit, **kwargs)


@pytest.fixture(scope="module")
def market():
    """1 January to 20 April 2023 00:00 in Warsaw (109 days of elapsed time), over the spring switch."""
    return make_market_data(start="2023-01-01", days=109, horizon=2, noise=5.0, seed=3)


@pytest.fixture
def block_spy(monkeypatch):
    """(unit, the dates of its call's rows, the rows, the period's day ids) of each block call, as `block_rows` gives them."""
    seen = []
    original = worker.block_rows

    def spy(cfg, unit, days, x, col_idx):
        rows = original(cfg, unit, days, x, col_idx)
        first = cfg["refit"]["first_ordinal"]
        seen.append((unit, [datetime.date.fromordinal(int(days[row]) + first) for row in rows], rows, days))
        return rows

    monkeypatch.setattr(worker, "block_rows", spy)
    return seen


def local_dates(data):
    return np.asarray(pd.DatetimeIndex(data.index.tz_localize(None)).normalize())


def day(text):
    return datetime.date.fromisoformat(text)


def keyed(records, *drop):
    """Key -> the record as JSON text without `drop`, so a NaN `actual` equals itself."""
    return {
        (r["run_date"], r["hour"], r["horizon"]): json.dumps(
            {k: v for k, v in r.items() if k not in drop}, sort_keys=True
        )
        for r in records
    }


def predictions(records):
    """Key -> (prediction, coefficients, refit_anchor): what the fit made of it."""
    return {
        (r["run_date"], r["hour"], r["horizon"]): (r["prediction"], r["coefficients"], r.get("refit_anchor"))
        for r in records
    }


def lines(path):
    return sorted(line for line in Path(path).read_text().splitlines() if line.strip())


def moved(data, rows, columns, seed=1):
    """A copy of `data` with every finite value of `columns` on `rows` moved by noise."""
    out = data.copy()
    values = out[columns].to_numpy(dtype=float, copy=True)
    where = rows[:, None] & np.isfinite(values)
    values[where] += np.random.default_rng(seed).normal(0, 50, size=int(where.sum()))
    out[columns] = values
    return out


def inputs(horizons=(1, 2)):
    """Every column a forecast at these horizons reads."""
    return [name.format(horizon=h) for h in horizons for name in PREDICTORS]


def finite_days(data, period, horizon, days):
    """The days of `days` on which `data` has a row at wall-clock hour `period` with every input of `horizon` finite."""
    wall = data.index.tz_localize(None)
    names = [name.format(horizon=horizon) for name in PREDICTORS]
    kept = []
    for date in days:
        rows = np.asarray((wall.normalize() == pd.Timestamp(date)) & (wall.hour == period))
        if rows.any() and np.isfinite(data.loc[rows, names].to_numpy(dtype=float)[-1]).all():
            kept.append(date)
    return kept


# ---------------------------------------------------------------------------
# The calendar helpers and the hooks
# ---------------------------------------------------------------------------


def test_block_days_and_their_nominal_length():
    week = block_days("2023-03-22", "7D")
    assert week == [day("2023-03-20") + datetime.timedelta(days=i) for i in range(7)]
    assert block_days("2023-03-20", "7D") == week == block_days(day("2023-03-26"), "7D")
    assert block_days("2023-03-22", "1D") == [day("2023-03-22")]
    assert block_days("2024-02-14", "1MS")[0] == day("2024-02-01") and len(block_days("2024-02-14", "1MS")) == 29
    assert len(block_days("2023-05-31", "3MS")) == 91  # April to June
    for rule in ("1D", "2D", "7D", "1MS", "3MS", "12MS"):
        for offset in range(0, 400, 17):
            issue = day("2023-01-01") + datetime.timedelta(days=offset)
            days = block_days(issue, rule)
            assert days[0] == refit_anchor(issue, rule) and days[-1] == refit_block_end(issue, rule)
            assert issue in days and len(days) <= nominal_block_length(rule)
    assert [nominal_block_length(rule) for rule in ("1D", "7D", "14D", "1MS", "3MS", "12MS")] == [1, 7, 14, 31, 93, 372]
    for bad in (None, "auto", "7d", "0D", 7):
        with pytest.raises(ValueError, match="nominal length"):
            nominal_block_length(bad)


def test_unit_seed_reads_the_unit_and_the_base_seed_alone():
    """A fixed hash of (random_state, period, horizon, anchor, list, member, seed index): pinned, so it can't drift between
    versions, a non-negative int below 2**63, different whenever one part is, and the same for equal parts of other types."""
    unit = FitUnit(5, 2, day("2024-01-01"), None, 0)
    assert unit_seed(0, unit) == 6902156273302525642
    assert unit_seed(0, unit, 1) == 2000807216878674471
    assert unit_seed(7, FitUnit(5, 2, day("2024-01-01"), 3, 1), 2) == 2734003731399182304
    others = [
        unit_seed(1, unit),
        unit_seed(0, unit._replace(period=6)),
        unit_seed(0, unit._replace(horizon=3)),
        unit_seed(0, unit._replace(anchor=day("2024-01-08"))),
        unit_seed(0, unit._replace(list_id=0)),
        unit_seed(0, unit._replace(member=1)),
        unit_seed(0, unit, 1),
        unit_seed(0, unit._replace(period=None)),
    ]
    assert len({unit_seed(0, unit), *others}) == 1 + len(others)
    assert all(0 <= seed < 2**63 for seed in others)
    same = FitUnit(np.int64(5), np.int32(2), pd.Timestamp("2024-01-01"), None, np.int64(0))
    assert unit_seed(np.int64(0), same) == unit_seed(0, unit)
    for bad in (0.0, None, True):
        with pytest.raises(TypeError):
            unit_seed(bad, unit)
    with pytest.raises(TypeError):
        unit_seed(0, unit, 1.0)


def test_a_class_with_predict_block_gets_the_hooks_and_no_core_model_has_them():
    """`block_predict` True, `_predicts_together` and `_co_batch_items` (the other days of the block of the coarsest member's
    rule) come with `_predict_block`. Core models have none of them, so `verify` checks no co-batch of theirs by default."""
    model = kernel()
    assert model.block_predict is True and model._predicts_together is True
    assert model._co_batch_items("2023-03-22") == [d for d in block_days("2023-03-22", "7D") if d != day("2023-03-22")]
    assert model._co_batch_items(pd.Timestamp("2023-03-20")) == block_days("2023-03-20", "7D")[1:]
    assert len(kernel("1MS")._co_batch_items("2023-04-12")) == 29
    assert kernel("1D")._co_batch_items("2023-04-12") == []
    # "auto": the block of the coarsest member's rule holds the call of every member.
    week = [d for d in block_days("2023-04-10", "7D") if d != day("2023-04-12")]
    assert kernel("auto", training_window=[28, 91])._co_batch_items("2023-04-12") == week
    for alone in (kernel(None), kernel(block_predict=False)):
        assert alone._predicts_together is False and alone._co_batch_items("2023-03-22") == []

    class Defines(KernelModel):
        _predicts_together = False

        def _co_batch_items(self, run_day):
            return ["2023-01-01"]

    assert Defines(refit="7D")._predicts_together is False and Defines()._co_batch_items("x") == ["2023-01-01"]
    for core in (OLSModel, RidgeModel, LassoCVModel, LEARModel, NaiveModel, BaseModel):
        for name in ("_predict_block", "block_predict", "_predicts_together", "_co_batch_items"):
            assert not hasattr(core, name), (core.__name__, name)


# ---------------------------------------------------------------------------
# What a call holds
# ---------------------------------------------------------------------------


def test_a_call_holds_every_day_of_the_block_the_frame_has_padded_with_its_first_row(market, block_spy):
    """One call per unit, of the nominal length: the issue rows of the block's days with a row and finite inputs, in date
    order, whatever days the run forecasts (here Wednesday alone), then copies of the first. A day with a missing input (period
    5 on Wednesday, horizon 1) is left out and its forecast skipped; the spring day has no 02:00, and the day h before it
    reads `load_d+{h}` there, so period 2's call has five days. A frame that ends on Friday leaves the weekend out, and the
    call keeps its length."""
    data = market.copy()
    wednesday = np.asarray((data.index.tz_localize(None).normalize() == "2023-03-22") & (data.index.hour == 5))
    data.loc[wednesday, "wind_d+1"] = np.nan
    ref = kernel().run(data, "2023-03-22", "2023-03-22", horizon=2)
    assert len(ref._results) == 47 and (5, 1) not in {(r["hour"], r["horizon"]) for r in ref._results}
    assert ref.skipped == {"missing_test_predictors": 1}

    week = block_days(WEEK[0], "7D")
    assert len(block_spy) == len(KernelModel.calls) == 24 * 2 - 1  # period 5 at horizon 1 has no forecast to call for
    for (unit, dates, rows, days), call in zip(block_spy, KernelModel.calls):
        assert unit.anchor == day(WEEK[0])
        assert dates == finite_days(data, unit.period, unit.horizon, week)
        assert call.shape == (7, len(PREDICTORS))
        assert len(np.unique(call, axis=0)) == len(dates)  # one row per day, and the pads are copies of one of them
        assert (call[len(dates) :] == call[0]).all()
        assert [rows[i] < rows[i + 1] for i in range(len(rows) - 1)] == [True] * (len(rows) - 1)
        # 02:00 on 26 March: no row, and the day h before it reads its inputs there (`load_d+{h}`); Wednesday's wind at 05:00.
        spring = day(WEEK[1])
        left_out = {spring, spring - datetime.timedelta(days=unit.horizon)} if unit.period == 2 else set()
        if (unit.period, unit.horizon) == (5, 1):
            left_out = {day("2023-03-22")}
        assert set(week) - set(dates) == left_out, unit

    block_spy.clear()
    KernelModel.calls.clear()
    friday = data[local_dates(data) <= np.datetime64("2023-03-24")]
    kernel().run(friday, "2023-03-21", "2023-03-21", horizon=1)
    assert {len(dates) for _, dates, _, _ in block_spy} == {5, 4}  # Monday to Friday; period 5 without Wednesday
    assert {call.shape for call in KernelModel.calls} == {(7, len(PREDICTORS))}


def test_a_monthly_unit_calls_31_rows_and_each_member_of_auto_its_own_rule(market, block_spy):
    """`"1MS"`: the month's days the frame has with finite inputs (1 to 18 April, at 00:00 to 19 April: `load_d+1` reads 20
    April), padded to 31. `"auto"` with windows [28, 91]: the 28-day member refits daily, a call of its issue day alone
    (length 1), and the 91-day member weekly, a call of the week's days (7)."""
    kernel("1MS").run(market, "2023-04-12", "2023-04-12", horizon=1)
    april = block_days("2023-04-12", "1MS")
    for unit, dates, _, _ in block_spy:
        assert dates == finite_days(market, unit.period, unit.horizon, april)
    assert {len(dates) for _, dates, _, _ in block_spy} == {18, 19}
    assert {call.shape[0] for call in KernelModel.calls} == {31}

    block_spy.clear()
    KernelModel.calls.clear()
    kernel("auto", training_window=[28, 91]).run(market, "2023-04-12", "2023-04-12", horizon=1)
    by_member = {}
    for (unit, dates, _, _), call in zip(block_spy, KernelModel.calls):
        by_member.setdefault(unit.member, set()).add((unit.anchor, tuple(dates), call.shape[0]))
    assert by_member[0] == {(day("2023-04-12"), (day("2023-04-12"),), 1)}
    assert by_member[1] == {(day("2023-04-10"), tuple(block_days("2023-04-10", "7D")), 7)}


def test_the_repeated_autumn_hour_enters_a_call_by_its_second_row(block_spy):
    """As a forecast is issued from it (`issue_row`): on 29 October 2023 Warsaw has 02:00 twice."""
    autumn = make_market_data(start="2023-09-01", days=70, horizon=1, noise=5.0, seed=4)
    kernel().run(autumn, "2023-10-25", "2023-10-25", horizon=1)
    ((unit, dates, rows, days),) = [entry for entry in block_spy if entry[0].period == 2]
    assert dates == block_days("2023-10-23", "7D")
    sunday = rows[-1]
    assert days[sunday - 1] == days[sunday] and (sunday + 1 == len(days) or days[sunday + 1] != days[sunday])


# ---------------------------------------------------------------------------
# A forecast doesn't depend on the run around it
# ---------------------------------------------------------------------------


def test_a_forecast_is_the_same_alone_in_the_full_block_from_any_test_start_and_on_fridays(market):
    """With a model that sees its call's length and rows, so the calls themselves are compared: a day alone, a test start and
    end inside a block, a run of Fridays and a run over three blocks give each forecast the same bits."""
    full = keyed(kernel(**SEES_CALL).run(market, FIRST, LAST, horizon=2)._results, "day_in_test")
    assert len(full) == 21 * 48 - 4  # 02:00 on 26 March is no issue hour, nor the target hour of the two days before
    runs = [
        kernel(**SEES_CALL).run(market, "2023-03-22", "2023-03-22", horizon=2),
        kernel(**SEES_CALL).run(market, "2023-03-26", "2023-03-26", horizon=2),
        kernel(**SEES_CALL).run(market, "2023-03-15", "2023-03-29", horizon=2),
        kernel(**SEES_CALL).run(market, FIRST, LAST, horizon=2, issue_days=["friday"]),
    ]
    for ref in runs:
        part = keyed(ref._results, "day_in_test")
        assert part and part.keys() <= full.keys()
        assert part == {key: full[key] for key in part}


def test_the_data_end_inside_a_block_leaves_the_shape_of_its_call(market):
    """A frame that ends on Friday 24 March: Monday to Wednesday's forecasts (targets to Friday) are those of the long frame,
    since the call is still 7 rows long, with two more copies of Monday's row. A model that sees the call's length would move
    if the call shrank to the days the frame has."""
    friday = market[local_dates(market) <= np.datetime64("2023-03-24")]
    model = {"by_length": 1e-12}
    short = predictions(kernel(**model).run(friday, "2023-03-20", "2023-03-22", horizon=2)._results)
    long = predictions(kernel(**model).run(market, FIRST, LAST, horizon=2)._results)
    assert len(short) == 3 * 48
    assert short == {key: long[key] for key in short}


def test_a_run_cut_short_resumes_to_a_fresh_one(market, tmp_path):
    """A resume runs the pending forecasts of a unit alone: its call still holds the whole block, so the file is a fresh run's,
    line for line."""
    fresh, cut = tmp_path / "fresh" / "m.jsonl", tmp_path / "cut" / "m.jsonl"
    kernel(**SEES_CALL).run(market, FIRST, LAST, horizon=2, save_to=str(fresh))
    kernel(**SEES_CALL).run(market, FIRST, LAST, horizon=2, save_to=str(cut))
    stored = lines(cut)
    kept = [line for line, keep in zip(stored, np.random.default_rng(2).random(len(stored)) < 0.6) if keep]
    cut.write_text("\n".join(kept) + "\n")
    units = {}
    for line in stored:
        record = json.loads(line)
        units.setdefault((record["hour"], record["horizon"], record["refit_anchor"]), []).append(line in kept)
    assert sum(any(v) and not all(v) for v in units.values()) > 50  # many units are stored in part
    kernel(**SEES_CALL).run(market, FIRST, LAST, horizon=2, save_to=str(cut))
    assert lines(cut) == lines(fresh)


def test_block_predict_false_gives_the_same_forecasts_one_row_at_a_time(market):
    """The fake has no cross-row dependence, so its block call and its rows one at a time (`_predict_fitted`) agree to the bit;
    with `block_predict=False` no call is made."""
    together = kernel().run(market, FIRST, LAST, horizon=2)._results
    assert KernelModel.calls
    KernelModel.calls.clear()
    alone = kernel(block_predict=False).run(market, FIRST, LAST, horizon=2)._results
    assert not KernelModel.calls
    assert keyed(alone) == keyed(together)
    assert keyed(kernel(None).run(market, *WEEK, horizon=1)._results, "refit_anchor")  # refit=None fits each on its own


# ---------------------------------------------------------------------------
# No look-ahead through a call
# ---------------------------------------------------------------------------


def test_no_row_after_the_block_and_no_target_reaches_a_call(market):
    """With Wednesday's inputs missing the call is padded. Moving every input after the block's last day, and the target from
    its first day on, leaves every forecast of the block (a pad of the next day's row would move them); moving Thursday's inputs
    moves Monday's forecasts of the model that sees its call, and not those of the plain one, so the perturbation reaches the
    call and the plain model doesn't read its other rows."""
    data = market.copy()
    dates = local_dates(data)
    data.loc[dates == np.datetime64("2023-03-22"), inputs()] = np.nan
    after = dates > np.datetime64(WEEK[1])
    later = moved(moved(data, after, inputs()), dates >= np.datetime64(WEEK[0]), ["price"], seed=2)
    base = predictions(kernel(**SEES_CALL).run(data, *WEEK, horizon=2)._results)
    assert len(base) == 6 * 48 - 4  # no Wednesday; 02:00 on 26 March is no issue or target hour
    assert predictions(kernel(**SEES_CALL).run(later, *WEEK, horizon=2)._results) == base

    thursday = moved(data, dates == np.datetime64("2023-03-23"), inputs(), seed=3)
    monday = {key: value for key, value in base.items() if key[0] == WEEK[0]}
    seen = predictions(kernel(**SEES_CALL).run(thursday, WEEK[0], WEEK[0], horizon=2)._results)
    assert all(seen[key] != value for key, value in monday.items())
    plain = predictions(kernel().run(data, WEEK[0], WEEK[0], horizon=2)._results)
    assert predictions(kernel().run(thursday, WEEK[0], WEEK[0], horizon=2)._results) == plain


# ---------------------------------------------------------------------------
# A call that goes wrong
# ---------------------------------------------------------------------------


class Short(KernelModel):
    def _predict_block(self, fitted, rows):
        predicted, coefficients = super()._predict_block(fitted, rows)
        return predicted[:-1], coefficients


class Refuses(KernelModel):
    def _fit_model(self, train_x, train_y):
        raise ValueError("degenerate")

    def _fit_predict(self, train_x, train_y, test_x):
        return self._predict_fitted(self._fit_model(train_x, train_y), test_x)

    def _predict_fitted(self, fitted, test_x):
        return super()._predict_fitted(fitted, test_x)


def test_a_call_that_returns_another_shape_raises_and_a_refused_fit_is_every_forecast_s_fit_error(market):
    with pytest.raises(RuntimeError, match="one of each per row"):
        Short(refit="7D").run(market, *WEEK, horizon=1)
    with pytest.raises(RuntimeError, match=r"no forecasts produced \(166 planned\). Skipped: fit_error x166"):
        Refuses(refit="7D").run(market, *WEEK, horizon=1)
