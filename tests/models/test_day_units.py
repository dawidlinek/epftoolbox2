"""Day and pooled units (step 6f): one fit of every period of a day at once, on the refit calendar.

A day unit's rows are local dates, its predictors day-level columns (one value per date) and its targets the P periods of the
target day, a slot with no target masked; a pooled unit's rows are the (date, period) rows of every period, each predictor read
on its period's own row, and `period` a feature when listed. Both predict every issue day of their block that the frame has in
one call of the block's nominal length, so a forecast is the same alone, in its block, after a resume and whatever the test
start or the data's end.

The models are numpy stand-ins for 6l's and 6k's: `DayRidge` fits one ridge per output on that output's valid slots (so P
per-period ridges on the same rows and inputs), `PooledOLS` one least-squares fit over the stacked rows. `by_length` and
`by_call` make a model see its call (its row count, the mean of a column over its rows) by a tiny amount, so a test can tell
whether two calls were the same. The references are built from the frame with pandas and numpy, not with the library's rows.
"""

import datetime
import json
import math
import warnings
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import DailyProfileTransformer
from epftoolbox2.information_set import Availability, InformationSet, cuts_at_issue_day
from epftoolbox2.models import OLSModel
from epftoolbox2.models._calendar import IssueCalendar, block_days, refit_anchor
from epftoolbox2.models._units import PERIOD_FEATURE, check_day_level, profile_names
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.models.base import BaseModel
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from epftoolbox2.scalers import AsinhScaler, StandardScaler
from epftoolbox2.scalers.target import MultiTargetAsinh
from tests.models import adapter_suite as suite
from tests.synthetic import calendar_shift, make_market_data

WINDOW = 28
HOURS = (0, 8, 12, 18)


def slot(hour, freq="1h"):
    """A `DailyProfileTransformer` column's slot of `hour`: `h08` hourly, `s32` at 15 minutes."""
    return f"h{hour:02d}" if freq == "1h" else f"s{hour * 4:02d}"


def profile_columns(freq="1h"):
    """A day unit's predictors: D-1's prices and the target day's load at `HOURS`, and the target day's weekend flag."""
    price = [f"price_{slot(hour, freq)}_d-1" for hour in HOURS]
    load = [f"load_{slot(hour, freq)}_d+{{horizon}}" for hour in HOURS]
    return [*price, *load, "is_weekend_d+{horizon}"]


DAY_PREDICTORS = profile_columns()
PRICE = DAY_PREDICTORS[: len(HOURS)]
POOLED_PREDICTORS = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "price_d-1", PERIOD_FEATURE]
FIRST, LAST = "2023-03-13", "2023-04-02"  # three "7D" blocks, Monday to Sunday
WEEK = ("2023-03-20", "2023-03-26")  # the spring switch is on its Sunday (Europe/Warsaw): no 02:00 on 26 March
RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
}


class InProcessPool:
    """Stands in for the worker pool, as in tests/models/test_block_predict.py: the fits run in this process."""

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
    DayRidge.calls.clear()
    PooledOLS.calls.clear()
    yield
    DayRidge.calls.clear()
    PooledOLS.calls.clear()


def _seen(model, rows):
    """What a model that sees its call adds to each of its predictions: nothing unless asked (0 * NaN is NaN)."""
    seen = 0.0
    if model.by_length:
        seen += model.by_length * len(rows)
    if model.by_call:
        seen += model.by_call * float(rows[:, 1].mean())
    if model.by_rows:
        seen += model.by_rows * len(np.unique(rows, axis=0))
    return seen


class DayRidge(BaseModel):
    """A ridge per output on its valid slots, with an intercept, on the scaled inputs and the targets through each period's
    Invariant: P per-period ridges with the same penalty, rows and inputs. Each call's rows are kept in `calls`."""

    _unit_kind = "day"
    calls: list = []

    def __init__(
        self,
        predictors=DAY_PREDICTORS,
        training_window=WINDOW,
        *,
        alpha=1.0,
        by_length=0.0,
        by_call=0.0,
        by_rows=0.0,
        **kw,
    ):
        kw.setdefault("refit", "7D")
        super().__init__(predictors, training_window, **kw)
        self.alpha = alpha
        self.by_length = by_length
        self.by_call = by_call
        self.by_rows = by_rows

    @property
    def _model_kwargs(self):
        return {"alpha": self.alpha, "by_length": self.by_length, "by_call": self.by_call, "by_rows": self.by_rows}

    def _fit_predict(self, train_x, train_y, test_x):
        raise NotImplementedError("a day unit fits through _fit_days")

    def _fit_days(self, x, y, y_mask):
        assert np.isnan(y[~y_mask]).all() and np.isfinite(y[y_mask]).all()
        coefs = np.zeros((x.shape[1], y.shape[1]))
        intercepts = np.zeros(y.shape[1])
        for period in range(y.shape[1]):
            rows = y_mask[:, period]
            if not rows.any():
                continue
            coefs[:, period], intercepts[period] = ridge(x[rows], y[rows, period], self.alpha)
        return coefs, intercepts

    def _predict_days(self, fitted, rows):
        type(self).calls.append(rows.copy())
        coefs, intercepts = fitted
        seen = _seen(self, rows)
        return np.array([row @ coefs + intercepts + seen for row in rows])


class PooledOLS(BaseModel):
    """Least squares with an intercept over a pooled unit's stacked rows; a forecast is its own row's, with the fit's
    coefficients. Each call's rows are kept in `calls`."""

    _unit_kind = "pooled"
    calls: list = []

    def __init__(
        self,
        predictors=POOLED_PREDICTORS,
        training_window=WINDOW,
        *,
        target_transform="per_period",
        by_length=0.0,
        by_call=0.0,
        by_rows=0.0,
        **kw,
    ):
        kw.setdefault("refit", "7D")
        super().__init__(predictors, training_window, **kw)
        self.target_transform = target_transform
        self.by_length = by_length
        self.by_call = by_call
        self.by_rows = by_rows

    @property
    def _model_kwargs(self):
        return {
            "target_transform": self.target_transform,
            "by_length": self.by_length,
            "by_call": self.by_call,
            "by_rows": self.by_rows,
        }

    def _fit_predict(self, train_x, train_y, test_x):
        raise NotImplementedError("a pooled unit fits through _fit_model and _predict_block")

    def _fit_model(self, train_x, train_y):
        return least_squares(train_x, train_y)

    def _predict_block(self, fitted, rows):
        type(self).calls.append(rows.copy())
        seen = _seen(self, rows)
        predictions = np.array([linear(fitted, row) + seen for row in rows])
        return predictions, [[float(value) for value in fitted] for _ in rows]


def ridge(x, y, alpha):
    """(coefficients, intercept) of a ridge with an unpenalised intercept, by the normal equations of the centred rows."""
    mean_x, mean_y = x.mean(axis=0), y.mean()
    centred = x - mean_x
    beta = np.linalg.solve(centred.T @ centred + alpha * np.eye(x.shape[1]), centred.T @ (y - mean_y))
    return beta, mean_y - mean_x @ beta


def least_squares(x, y):
    design = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(design, y, rcond=None)[0]


def linear(fitted, row):
    """intercept + row . coefficients, exactly rounded (`math.fsum`), so the same whatever the row's memory layout: a period
    unit's test rows can be a column-major array, whose row a BLAS dot sums in another order."""
    return float(math.fsum(np.asarray(row, dtype=np.float64) * fitted[1:]) + fitted[0])


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def with_profiles(data, horizon=2, freq="1h"):
    """`make_market_data` with the daily profiles of price and load at `HOURS`, price lagged a day and load led by each
    horizon, by calendar days (tests/synthetic.py's own shift)."""
    profiles = DailyProfileTransformer(columns=["price", "load"], freq=freq).transform(data[["price", "load"]])
    out = data.copy()
    for hour in HOURS:
        out[f"price_{slot(hour, freq)}_d-1"] = calendar_shift(profiles[f"price_{slot(hour, freq)}"], -1)
        for h in range(1, horizon + 1):
            out[f"load_{slot(hour, freq)}_d+{h}"] = calendar_shift(profiles[f"load_{slot(hour, freq)}"], h)
    return out


def frame(start="2023-01-01", days=109, tz="Europe/Warsaw", freq="1h", horizon=2, seed=3):
    data = make_market_data(start=start, days=days, tz=tz, freq=freq, horizon=horizon, noise=5.0, seed=seed)
    with warnings.catch_warnings():  # the profile of a switch day has a NaN or a first occurrence: what the tests want
        warnings.filterwarnings("ignore", "DailyProfileTransformer: some local days", UserWarning)
        return with_profiles(data, horizon, freq)


@pytest.fixture(scope="module")
def market():
    """1 January to 20 April 2023 00:00 in Warsaw (109 days), over the spring switch."""
    return frame()


@pytest.fixture(scope="module")
def plain():
    """The same dates on a clock without switches (UTC+1): every date has 24 hours."""
    return frame(tz="Etc/GMT-1")


@pytest.fixture(scope="module")
def autumn():
    """1 September to 9 November 2023 in Warsaw, over the autumn switch (29 October)."""
    return frame(start="2023-09-01", days=70, seed=4)


def day(text):
    return datetime.date.fromisoformat(text)


def keyed(records, *drop):
    return {
        (r["run_date"], r["hour"], r["horizon"]): json.dumps(
            {k: v for k, v in r.items() if k not in drop}, sort_keys=True
        )
        for r in records
    }


def predictions(records):
    return {(r["run_date"], r["hour"], r["horizon"]): (r["prediction"], r["coefficients"]) for r in records}


def lines(path):
    return sorted(line for line in Path(path).read_text().splitlines() if line.strip())


def dates_of(data):
    return np.asarray(pd.DatetimeIndex(data.index.tz_localize(None)).normalize())


# ---------------------------------------------------------------------------
# The kinds, their hooks and what they refuse
# ---------------------------------------------------------------------------


def test_a_day_or_pooled_class_gets_the_block_hooks_and_the_engine_takes_neither():
    """Both predict their blocks in one call, so `verify` checks the co-batch and the cut frame by default; neither is
    engine-capable; a core model's kind is "period" and it has none of the hooks."""
    for model in (DayRidge(), PooledOLS()):
        assert model._predicts_together is True and cuts_at_issue_day(model)
        assert model._co_batch_items("2023-03-22") == [
            d for d in block_days("2023-03-22", "7D") if d != day("2023-03-22")
        ]
        assert model._engine_capable() is False
    assert not hasattr(DayRidge, "block_predict")  # a day unit has no one-row-at-a-time path
    assert DayRidge(refit="1D")._co_batch_items("2023-03-22") == []
    assert OLSModel(["x"])._unit_kind == "period" and BaseModel._unit_kind == "period"
    for name in ("_predict_block", "block_predict", "_predicts_together", "_co_batch_items"):
        assert not hasattr(OLSModel, name)


def test_a_unit_needs_refit_and_one_class_defining_its_hooks(market):
    with pytest.raises(ValueError, match="a day unit is a fit on the refit calendar"):
        DayRidge(refit=None).run(market, *WEEK, horizon=1)
    with pytest.raises(ValueError, match="a pooled unit is a fit on the refit calendar"):
        PooledOLS(refit=None).run(market, *WEEK, horizon=1)

    class Half(DayRidge):
        def _fit_days(self, x, y, y_mask):
            return super()._fit_days(x, y, y_mask)

    with pytest.raises(ValueError, match="fits day units, from _fit_days and _predict_days"):
        Half()

    class NoBlock(BaseModel):
        _unit_kind = "pooled"

        def _fit_predict(self, train_x, train_y, test_x):
            return 0.0, []

        def _fit_model(self, train_x, train_y):
            return None

    with pytest.raises(ValueError, match="fits pooled units, from _fit_model and _predict_block"):
        NoBlock(["x"], refit="7D")


def test_a_list_may_change_with_the_horizon_and_the_issue_day_but_not_the_period(market):
    """One fit reads one list: a `(horizon, hour)` function, or a function of `ctx` that answers two periods of one issue day
    with two lists, raises before any fit; a function that changes its list with the issue day gives each list a unit."""

    def by_hour(horizon, hour):
        return DAY_PREDICTORS[:2] if hour < 12 else DAY_PREDICTORS[:3]

    with pytest.raises(ValueError, match=r"at horizon 1 period 0 has .* and period 12"):
        DayRidge(by_hour).run(market, *WEEK, horizon=1)

    def by_period(ctx):
        return PRICE if ctx.period != 5 else PRICE[:2]

    with pytest.raises(ValueError, match=r"returned two lists on 2023-03-20 at horizon 1: period 0 .* period 5"):
        DayRidge(by_period).run(market, *WEEK, horizon=1)
    with pytest.raises(ValueError, match="two lists on 2023-03-20"):
        PooledOLS(lambda ctx: POOLED_PREDICTORS[:2] if ctx.hour else POOLED_PREDICTORS[1:3]).run(
            market, *WEEK, horizon=1
        )

    def by_day(ctx):  # Thursday on, one input fewer
        return [
            name.format(horizon=ctx.horizon) for name in DAY_PREDICTORS[: (8 if ctx.issue_day.weekday() < 3 else 7)]
        ]

    seen = []
    real = DayRidge._fit_days

    def spy(self, x, y, y_mask):
        seen.append(x.shape[1])
        return real(self, x, y, y_mask)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DayRidge, "_fit_days", spy)
        ref = DayRidge(by_day).run(market, *WEEK, horizon=1)
    assert sorted(seen) == [7, 8] and len(ref._results) == 7 * 24 - 2  # 02:00 of 26 March is no issue or target hour
    assert len({r["predictor_list"] for r in ref._results}) == 2


def test_what_a_unit_refuses_in_this_step(market):
    from epftoolbox2.models import Use
    from epftoolbox2.models.pooled import check_pooled_model

    with pytest.raises(ValueError, match="train_on='matching' filters the rows of one period"):
        DayRidge(lambda ctx: Use(PRICE, train_on="matching")).run(market, *WEEK, horizon=1)
    with pytest.raises(ValueError, match="the virtual 'period' feature is for fixed lists"):
        PooledOLS(lambda ctx: POOLED_PREDICTORS).run(market, *WEEK, horizon=1)
    with pytest.raises(ValueError, match="the frame has a column named 'period'"):
        PooledOLS().run(market.assign(period=1.0), *WEEK, horizon=1)
    PooledOLS(POOLED_PREDICTORS[:-1]).run(
        market.assign(period=1.0), WEEK[0], WEEK[0], horizon=1
    )  # not listed: a column
    for model in (DayRidge(), PooledOLS()):
        with pytest.raises(ValueError, match="units .* can't be pooled over zones"):
            check_pooled_model(model)


# ---------------------------------------------------------------------------
# The day-level check
# ---------------------------------------------------------------------------


def test_the_day_level_check_refuses_a_column_that_changes_with_the_period_and_names_its_profile(market):
    with pytest.raises(
        ValueError, match=r"'price' holds values from .* Give its daily profile instead: price_h00 … price_h23"
    ):
        DayRidge(["price", *PRICE]).run(market, *WEEK, horizon=1)
    with pytest.raises(
        ValueError, match=r"'price_d-1' holds .*: price_h00_d-1 … price_h23_d-1 \(DailyProfileTransformer"
    ):
        DayRidge([*PRICE, "price_d-1"]).run(market, *WEEK, horizon=1)
    assert profile_names("load_d+2", "15min").startswith("load_s00_d+2 … load_s95_d+2")
    assert profile_names("wind", "1h").startswith("wind_h00 … wind_h23")


def test_the_day_level_check_accepts_profiles_calendar_flags_and_a_date_with_some_rows_missing(market):
    """`price_h05_d-1`, `is_monday_d+{horizon}` and a date with a value on some rows and NaN on others (the day after a spring
    switch has no `_d-1` at 02:00; here a training date loses half its rows too) are day-level; the date's value is the one its
    rows hold."""
    data = market.copy()
    monday = (data.index.dayofweek == 0).astype(float)
    for h in (1, 2):
        data[f"is_monday_d+{h}"] = calendar_shift(pd.Series(monday, index=data.index), h)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "DailyProfileTransformer: some local days", UserWarning)
        profiles = DailyProfileTransformer(columns=["price"]).transform(data[["price"]])
    data["price_h05_d-1"] = calendar_shift(profiles["price_h05"], -1)
    rows = (dates_of(data) == np.datetime64("2023-03-08")) & (data.index.hour >= 12)
    data.loc[rows, PRICE] = np.nan
    assert data["price_h00_d-1"][dates_of(data) == np.datetime64("2023-03-27")].isna().sum() == 1
    predictors = ["price_h05_d-1", "is_monday_d+{horizon}", *PRICE]
    full = DayRidge(predictors).run(data, *WEEK, horizon=2)
    assert len(full._results) == 7 * 48 - 4
    check_day_level(data, ["price_h05_d-1", "is_monday_d+1", *PRICE], "1h", "m")
    with pytest.raises(ValueError, match="'load_d\\+1' holds"):
        check_day_level(data, ["load_d+1"], "1h", "m")
    # Only the dates the fits of the run train on: a column that changes on a later date isn't read there as a training value.
    later = data.copy()
    later.loc[(dates_of(data) == np.datetime64("2023-03-21")) & (data.index.hour == 3), "price_h00_d-1"] += 1.0
    DayRidge(predictors).run(later, *WEEK, horizon=2)
    with pytest.raises(ValueError, match="'price_h00_d-1' holds .* on 2023-03-21"):
        DayRidge(predictors).run(later, "2023-03-27", "2023-03-27", horizon=2)
    with pytest.raises(ValueError, match="on 2023-03-21"):
        check_day_level(later, ["price_h00_d-1"], "1h", "m")


# ---------------------------------------------------------------------------
# Day units against per-period fits by hand
# ---------------------------------------------------------------------------


def day_tables(data, names, horizon, target="price"):
    """By hand from the frame: each local date's value of `names` (the last non-NaN one of its rows) and, per wall-clock
    hour, the target `horizon` calendar days later where that hour exists once on the target day, NaN where it doesn't or
    where the issue date has no row at that hour; and which (date, hour) has a row."""
    local = data.index.tz_localize(None)
    dates, hours = local.normalize(), local.hour
    x = data[names].groupby(dates).last()
    count = data[target].groupby([dates, hours]).count().unstack()
    price = data[target].groupby([dates, hours]).last().unstack().where(count == 1)
    every = pd.date_range(x.index.min(), x.index.max() + pd.Timedelta(days=horizon), freq="D")
    price = price.reindex(index=every, columns=range(24))
    exists = count.reindex(index=x.index, columns=range(24)).notna()
    y = pd.DataFrame(price.shift(-horizon).reindex(x.index).to_numpy(), index=x.index).where(exists.to_numpy())
    return x, y, exists


def invariant(values):
    scaler = AsinhScaler(apply_to="target", standardise=False)
    _, z, _ = scaler.fit_transform(np.zeros((len(values), 0)), values, np.zeros((1, 0)))
    return z, scaler


def day_reference(data, issue, horizon, predictors=DAY_PREDICTORS, alpha=1.0, window=WINDOW, refit="7D", newest=None):
    """period -> the forecast of (period, horizon, issue) by hand: the dates of the span of its anchor, the inputs scaled over
    the dates with every input and a target of some period, binary columns (read on every date of the span) kept, and for each
    period its own ridge on the dates where its target is, on its Invariant (sklearn's Ridge, not the model's). `newest`
    (period -> k) ends the span at the anchor + max k - h, and a period's slots after the anchor + its k - h are no target."""
    from sklearn.linear_model import Ridge

    names = [name.format(horizon=horizon) for name in predictors]
    x, y, _ = day_tables(data, names, horizon)
    anchor = pd.Timestamp(refit_anchor(issue, refit))
    k = {period: -1 for period in range(24)} if newest is None else {p: min(v, horizon - 1) for p, v in newest.items()}
    last = anchor + pd.Timedelta(days=min(max(k.values()) - horizon, -1))
    span = (x.index >= last - pd.Timedelta(days=window - 1)) & (x.index <= last)
    for period, newest_k in k.items():  # a slot whose target isn't known on the anchor
        y.loc[y.index > anchor + pd.Timedelta(days=newest_k - horizon), period] = np.nan
    xs, ys = x[span].to_numpy(), y[span].to_numpy()
    mask = StandardScaler.get_scalable_mask(xs)
    keep = np.isfinite(xs).all(axis=1) & np.isfinite(ys).any(axis=1)
    test = x.loc[[pd.Timestamp(issue)]].to_numpy()
    fit_x, _, test_x = StandardScaler().fit_transform(xs[keep], np.zeros(int(keep.sum())), test, scalable_mask=mask)
    out = {}
    for period in range(24):
        rows = np.isfinite(ys[keep, period])
        if not rows.any():
            continue
        z, scaler = invariant(ys[keep][rows, period])
        fitted = Ridge(alpha=alpha).fit(fit_x[rows], z)
        out[period] = scaler.inverse(float(fitted.predict(test_x)[0]))
    return out


def by_key(records):
    return {(r["run_date"], r["hour"], r["horizon"]): r for r in records}


@pytest.mark.parametrize("horizon", [1, 2])
def test_a_day_ridge_is_24_per_period_ridges_on_a_clock_without_switches(plain, horizon):
    """UTC+1: every date has its 24 hours, so a day unit's fit is the 24 per-period ridges of the same penalty on the same
    dates and inputs, to 1e-10, on each day of two blocks."""
    records = by_key(DayRidge(alpha=2.0).run(plain, "2023-03-15", "2023-03-28", horizon=horizon)._results)
    assert len(records) == 14 * 24 * horizon
    worst = 0.0
    for issue in pd.date_range("2023-03-15", "2023-03-28").date:
        for hz in range(1, horizon + 1):
            reference = day_reference(plain, issue.isoformat(), hz, alpha=2.0)
            assert sorted(reference) == list(range(24))
            for period, value in reference.items():
                worst = max(worst, abs(records[(issue.isoformat(), period, hz)]["prediction"] - value))
    assert worst < 1e-10, worst


@pytest.mark.parametrize(
    "frame_name, issues",
    [("market", ("2023-03-24", "2023-03-26", "2023-03-29")), ("autumn", ("2023-10-27", "2023-10-29", "2023-11-01"))],
)
def test_on_switch_days_a_masked_slot_drops_that_slot_alone(frame_name, issues, request):
    """Over a spring and an autumn switch: the slot with no target (02:00 of the spring day, the day before it has its target
    there; the repeated 02:00 of the autumn day) drops out of its period's fit and no other, every other period of that date
    trains, and the forecasts are the hand-fitted ridges', to 1e-10. The calendar plans no key the clock doesn't have."""
    data = request.getfixturevalue(frame_name)
    seen = []
    real = DayRidge._fit_days

    def spy(self, x, y, y_mask):
        seen.append(y_mask.copy())
        return real(self, x, y, y_mask)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DayRidge, "_fit_days", spy)
        records = by_key(DayRidge().run(data, issues[0], issues[-1], horizon=2)._results)
    assert any((~mask).sum() and (~mask).any(axis=1).sum() == (~mask).sum() for mask in seen)  # lone masked slots
    calendar = IssueCalendar("1h", data.index.tz, (1, 2))
    first = day(issues[0])
    planned = {
        ((first + datetime.timedelta(days=d)).isoformat(), p, h)
        for p, h, d in calendar.tasks(first, range((day(issues[-1]) - first).days + 1))
    }
    assert set(records) == planned
    worst = 0.0
    for issue in issues:
        for hz in (1, 2):
            for period, value in day_reference(data, issue, hz).items():
                key = (issue, period, hz)
                if key in records:
                    worst = max(worst, abs(records[key]["prediction"] - value))
    assert worst < 1e-10, worst
    if frame_name == "autumn":
        assert np.isnan(records[("2023-10-27", 2, 2)]["actual"])  # the repeated hour's target


def test_the_issue_calendar_s_counts_at_an_hour_and_a_quarter_hour_for_both_kinds():
    """Over both switches, each kind's records are the keys `IssueCalendar` plans: 23/24 forecasts of a switch's target day
    hourly, 92/96 at 15 minutes (100 quarter-hours on the autumn day, its repeated four with a NaN actual), none missing."""
    for freq, periods in (("1h", 24), ("15min", 96)):
        for start, switch in (("2023-02-14", "2023-03-26"), ("2023-09-17", "2023-10-29")):
            data = frame(start=start, days=46, freq=freq, horizon=1)
            issue = (day(switch) - datetime.timedelta(days=1)).isoformat()
            names = profile_columns(freq)
            for model in (DayRidge(names), PooledOLS()):
                ref = model.run(data, issue, switch, horizon=1, freq=freq)
                calendar = IssueCalendar(freq, data.index.tz, (1,))
                planned = {
                    ((day(issue) + datetime.timedelta(days=d)).isoformat(), p, 1)
                    for p, _, d in calendar.tasks(day(issue), [0, 1])
                }
                assert set(by_key(ref._results)) == planned and ref.skipped == {}
                on_target = [r for r in ref._results if r["run_date"] == issue]
                expected = {"2023-03-26": periods - periods // 24, "2023-10-29": periods}[switch]
                assert len(on_target) == expected
                if switch == "2023-10-29":
                    assert sum(np.isnan(r["actual"]) for r in on_target) == periods // 24


# ---------------------------------------------------------------------------
# Pooled units against fits by hand, and the period unit at one period a day
# ---------------------------------------------------------------------------

INVARIANT = {"name": "asinh", "params": {"apply_to": "target", "standardise": False}}


class PeriodOLS(BaseModel):
    """`PooledOLS`'s fit as a period unit: one period's rows, one forecast per row through `_predict_fitted`."""

    def _fit_predict(self, train_x, train_y, test_x):
        return self._predict_fitted(self._fit_model(train_x, train_y), test_x)

    def _fit_model(self, train_x, train_y):
        return least_squares(train_x, train_y)

    def _predict_fitted(self, fitted, test_x):
        return linear(fitted, test_x[0]), [float(value) for value in fitted]


def test_multi_target_asinh_is_the_invariant_of_each_column_to_the_bit():
    """Each column is `AsinhScaler(apply_to="target", standardise=False)` fitted on its valid values, there and back,
    including a column whose MAD is 0 (its standard deviation), a constant one (scale 1.0) and one with no value at all."""
    rng = np.random.default_rng(0)
    y = rng.normal(50, 20, size=(40, 5))
    y[:25, 1] = 7.0  # MAD 0
    y[:, 2] = 3.0 + 1e-15 * rng.normal(size=40)  # constant up to rounding noise
    y[:, 4] = np.nan
    mask = rng.random(size=y.shape) > 0.2
    target = MultiTargetAsinh().fit(y, mask)
    z = target.transform(y)
    probe = rng.normal(0, 1.5, size=(9, 5))
    back = target.inverse(probe)
    for column in range(4):
        values = y[mask[:, column] & np.isfinite(y[:, column]), column]
        expected, scaler = invariant(values)
        assert np.array_equal(z[mask[:, column], column], expected)
        assert all(back[i, column] == scaler.inverse(float(probe[i, column])) for i in range(9))
    assert (target.center[4], target.scale[4]) == (0.0, 1.0) and target.scale[2] == 1.0
    groups = np.repeat(np.arange(5), 8)
    stacked = MultiTargetAsinh().fit_groups(y[:8].T.reshape(-1), groups, 5)
    for group in range(5):
        values = y[:8, group][np.isfinite(y[:8, group])]
        if len(values):
            assert np.array_equal(stacked.transform(values, np.full(len(values), group)), invariant(values)[0])


def test_a_pooled_unit_of_one_period_is_its_period_unit_to_the_bit(market, tmp_path):
    """At `freq="1D"` a day has one period, so a pooled unit's rows are its period unit's and the per-period Invariant is the
    scaler's: the records of the two (with raw inputs, `AsinhScaler(apply_to="target", standardise=False)`) are the same text."""
    daily = market.at_time("12:00")
    predictors = POOLED_PREDICTORS[:-1]
    pooled = PooledOLS(predictors, scaler=INVARIANT).run(daily, FIRST, LAST, horizon=2, freq="1D")
    alone = PeriodOLS(predictors, WINDOW, scaler=INVARIANT, refit="7D").run(daily, FIRST, LAST, horizon=2, freq="1D")
    assert len(pooled._results) == 21 * 2 and keyed(pooled._results) == keyed(alone._results)


def pooled_reference(
    data, issue, horizon, predictors=POOLED_PREDICTORS, transform="per_period", window=WINDOW, newest=None
):
    """period -> the forecast of (period, horizon, issue) by hand: every row of every hour dated in the span of its anchor, by
    date then hour (both rows of a repeated hour), its own row's inputs and the hour as `period`, the inputs scaled over the
    rows with every input and a target (binary columns, read on every row of the span, kept), the targets through each hour's
    Invariant (or one over every row), one least-squares fit, and each hour's forecast from its last row of the issue date."""
    names = [name.format(horizon=horizon) for name in predictors]
    local = data.index.tz_localize(None)
    dates, hours = local.normalize(), np.asarray(local.hour)
    count = data["price"].groupby([dates, hours]).count().unstack()
    table = data["price"].groupby([dates, hours]).last().unstack().where(count == 1)
    at = table.index.get_indexer(dates + pd.Timedelta(days=horizon))
    target = np.where(at >= 0, table.to_numpy()[np.maximum(at, 0), hours], np.nan)
    x = np.column_stack([hours if name == PERIOD_FEATURE else data[name].to_numpy(float) for name in names])
    anchor = pd.Timestamp(refit_anchor(issue, "7D"))
    # Each hour's own span: up to the anchor + k - h with k its newest target (-1 for "d-1"), W dates.
    k = np.array([-1 if newest is None else min(newest[hour], horizon - 1) for hour in range(24)])[hours]
    last = np.asarray(anchor + pd.to_timedelta(np.minimum(k - horizon, -1), unit="D"))
    days_back = np.asarray(last - np.asarray(dates), dtype="timedelta64[D]").astype(np.int64)
    span = (days_back >= 0) & (days_back < window)
    order = np.argsort(np.asarray(dates[span]).astype(np.int64) // 10**9 * 24 + hours[span], kind="stable")
    xs, ys, groups = x[span][order], target[span][order], hours[span][order]
    mask = StandardScaler.get_scalable_mask(xs)
    keep = np.isfinite(xs).all(axis=1) & np.isfinite(ys)
    issued = [
        np.flatnonzero((dates == pd.Timestamp(issue)) & (hours == p))[-1]
        for p in range(24)
        if ((dates == pd.Timestamp(issue)) & (hours == p)).any()
    ]
    periods = hours[issued]
    fit_x, _, test_x = StandardScaler().fit_transform(
        xs[keep], np.zeros(int(keep.sum())), x[issued], scalable_mask=mask
    )
    z = np.empty(int(keep.sum()))
    scalers = {}
    for group in range(24) if transform == "per_period" else [None]:
        rows = groups[keep] == group if group is not None else np.ones(len(z), dtype=bool)
        z[rows], scalers[group] = invariant(ys[keep][rows])
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(fit_x)), fit_x]), z, rcond=None)[0]
    return {
        int(p): scalers[int(p) if transform == "per_period" else None].inverse(float(row @ beta[1:] + beta[0]))
        for p, row in zip(periods, test_x)
    }


@pytest.mark.parametrize("transform", ["per_period", "pooled"])
def test_a_pooled_unit_is_one_fit_over_every_period_s_rows_by_hand(market, transform):
    """Each predictor on its period's own row (`price_d-1` that hour's price), `period` the hour, every period's rows of the
    span stacked, one fit; each forecast mapped back with its hour's Invariant or the pooled one: to 1e-10 of the hand fit,
    across the spring switch."""
    records = by_key(PooledOLS(target_transform=transform).run(market, *WEEK, horizon=2)._results)
    worst = 0.0
    for issue in pd.date_range(*WEEK).date:
        for hz in (1, 2):
            for period, value in pooled_reference(market, issue.isoformat(), hz, transform=transform).items():
                if (issue.isoformat(), period, hz) in records:
                    worst = max(worst, abs(records[(issue.isoformat(), period, hz)]["prediction"] - value))
    assert worst < 1e-10, worst
    other = by_key(
        PooledOLS(target_transform="pooled" if transform == "per_period" else "per_period")
        .run(market, *WEEK, horizon=2)
        ._results
    )
    assert all(other[key]["prediction"] != records[key]["prediction"] for key in records)


# ---------------------------------------------------------------------------
# A call holds the block, whatever the run holds
# ---------------------------------------------------------------------------

SEES_CALL = {"by_length": 1e-12, "by_call": 1e-6}
KINDS = {"day": DayRidge, "pooled": PooledOLS}


@pytest.mark.parametrize("kind", KINDS)
def test_a_call_holds_every_day_of_the_block_the_frame_has_padded_with_its_first(market, kind):
    """A run of Wednesday alone: one call per unit (horizon), of the nominal length (7 days, times 24 periods for a pooled
    unit), its rows the block's days in date order (and periods), copies of the first row after them. A pooled call has no
    02:00 of 26 March, nor of the day before it, whose `load_d+1` (and the day before that's `load_d+2`) reads it. With
    Wednesday's inputs missing its forecasts are skipped and the call leaves it out; a frame that ends on Friday leaves the
    weekend out, and the call keeps its length."""
    model = KINDS[kind]
    data = market.copy()
    ref = model().run(data, "2023-03-22", "2023-03-22", horizon=2)
    assert len(ref._results) == 48 and len(model.calls) == 2
    length = 7 if kind == "day" else 7 * 24
    for call in model.calls:
        assert call.shape[0] == length
        real = 7 if kind == "day" else 7 * 24 - 2
        assert len(np.unique(call, axis=0)) == real and (call[real:] == call[0]).all()

    model.calls.clear()
    names = PRICE if kind == "day" else ["price_d-1"]
    data.loc[dates_of(data) == np.datetime64("2023-03-22"), names] = np.nan
    ref = model().run(data, "2023-03-21", "2023-03-22", horizon=1)
    assert ref.skipped == {"missing_test_predictors": 24}
    expected = 6 if kind == "day" else 6 * 24 - 2
    assert [len(np.unique(call, axis=0)) for call in model.calls] == [expected]

    model.calls.clear()
    friday = market[dates_of(market) <= np.datetime64("2023-03-24")]
    model().run(friday, "2023-03-21", "2023-03-21", horizon=1)
    ((call,),) = [model.calls]
    assert call.shape[0] == length and len(np.unique(call, axis=0)) == (5 if kind == "day" else 5 * 24)


@pytest.mark.parametrize("kind", KINDS)
def test_a_forecast_is_the_same_alone_in_its_block_from_any_test_start_and_on_fridays(market, kind):
    """With a model that sees its call (its length and the mean of a column over its rows): a day alone, a test start and end
    inside a block, a run of Fridays, give each forecast the bits of a run over three blocks."""
    model = KINDS[kind]
    full = keyed(model(**SEES_CALL).run(market, FIRST, LAST, horizon=2)._results, "day_in_test")
    # 02:00 of 26 March is no issue or target hour; a pooled unit's `price_d-1` of 02:00 on 27 March has no value.
    assert len(full) == 21 * 48 - 4 - (2 if kind == "pooled" else 0)
    runs = [
        model(**SEES_CALL).run(market, "2023-03-22", "2023-03-22", horizon=2),
        model(**SEES_CALL).run(market, "2023-03-26", "2023-03-26", horizon=2),
        model(**SEES_CALL).run(market, "2023-03-15", "2023-03-29", horizon=2),
        model(**SEES_CALL).run(market, FIRST, LAST, horizon=2, issue_days=["friday"]),
    ]
    for ref in runs:
        part = keyed(ref._results, "day_in_test")
        assert part and part.keys() <= full.keys()
        assert part == {key: full[key] for key in part}


@pytest.mark.parametrize("kind", KINDS)
def test_the_data_end_inside_a_block_leaves_the_shape_of_its_call(market, kind):
    """A frame that ends on Friday 24 March: Monday to Wednesday's forecasts are the long frame's, its call still the block's
    nominal length (a model that sees the length would move if the call shrank)."""
    model = KINDS[kind]
    friday = market[dates_of(market) <= np.datetime64("2023-03-24")]
    short = predictions(model(by_length=1e-12).run(friday, "2023-03-20", "2023-03-22", horizon=2)._results)
    long = predictions(model(by_length=1e-12).run(market, FIRST, LAST, horizon=2)._results)
    assert len(short) == 3 * 48 and short == {key: long[key] for key in short}


@pytest.mark.parametrize("kind", KINDS)
def test_a_run_cut_short_resumes_to_a_fresh_one(market, tmp_path, kind):
    """A resume runs a unit's pending forecasts: its call still holds the whole block, so the file is a fresh run's, line for
    line, with many units stored in part."""
    model = KINDS[kind]
    fresh, cut = tmp_path / "fresh" / "m.jsonl", tmp_path / "cut" / "m.jsonl"
    model(**SEES_CALL).run(market, FIRST, LAST, horizon=2, save_to=str(fresh))
    model(**SEES_CALL).run(market, FIRST, LAST, horizon=2, save_to=str(cut))
    stored = lines(cut)
    kept = [line for line, keep in zip(stored, np.random.default_rng(2).random(len(stored)) < 0.6) if keep]
    cut.write_text("\n".join(kept) + "\n")
    units = {}
    for line in stored:
        record = json.loads(line)
        units.setdefault((record["horizon"], record["refit_anchor"]), []).append(line in kept)
    assert sum(any(v) and not all(v) for v in units.values()) == len(units)  # every unit is stored in part
    model(**SEES_CALL).run(market, FIRST, LAST, horizon=2, save_to=str(cut))
    assert lines(cut) == lines(fresh)


# ---------------------------------------------------------------------------
# Nothing after D moves D's forecasts
# ---------------------------------------------------------------------------

LOAD_REALISED = InformationSet("11:45", "Europe/Brussels", rules={**RULES, "price": Availability.realised("1h")})


def by_day(ctx):
    """A list that changes on Thursdays and with the horizon."""
    names = DAY_PREDICTORS if ctx.issue_day.weekday() != 3 else DAY_PREDICTORS[1:]
    return [name.format(horizon=ctx.horizon) for name in names]


def pooled_by_day(ctx):
    names = POOLED_PREDICTORS[:-1] if ctx.issue_day.weekday() != 3 else POOLED_PREDICTORS[1:-1]
    return [name.format(horizon=ctx.horizon) for name in names]


# Every level of each axis (kind x list x issue days x horizon) and every pair of levels, as in test_binary_mask.py's grid.
AFTER_D_CELLS = [
    ("day", "fixed", None, 1),
    ("day", "function", ["friday"], 2),
    ("day", "published", None, 2),
    ("pooled", "fixed", ["friday"], 2),
    ("pooled", "function", None, 1),
    ("pooled", "published", ["friday"], 1),
]


@pytest.mark.parametrize(("kind", "lists", "issue_days", "horizon"), AFTER_D_CELLS)
def test_nothing_after_the_issue_day_moves_its_forecasts(market, kind, lists, issue_days, horizon):
    """For D = Friday 24 March and Friday 31 March (inside a block, on the "7D" calendar): every finite value dated after D
    changed (by one draw per date and column, so that a day-level column stays one: the later forecasts train on those dates,
    and the day-level check holds them to it), and the test ending on D. D's predictions and coefficients stay bit for bit
    (the block's later days are in D's call, so a model that read their rows would move: that is the co-batch check's); the
    change is seen after D. A target transform fitted on every date of the frame would move them."""
    model = KINDS[kind]
    predictors = {"fixed": None, "function": by_day if kind == "day" else pooled_by_day, "published": None}[lists]
    settings = {} if predictors is None else {"predictors": predictors}
    if lists == "published":
        settings["newest_target"] = "published"
    days = {} if issue_days is None else {"issue_days": issue_days}
    first, last = "2023-03-20", "2023-04-09"

    def run(frame, end=last):
        ref = model(**settings).run(frame, first, end, horizon=horizon, information_set=LOAD_REALISED, **days)
        return predictions(ref._results)

    base = run(market)
    dates = dates_of(market)
    columns = [c for c in market.columns if market[c].dtype.kind == "f"]
    for issue in ("2023-03-24", "2023-03-31"):
        on_d = {key: fit for key, fit in base.items() if key[0] == issue}
        assert len(on_d) == 24 * horizon - (1 if issue == "2023-03-24" and horizon == 2 else 0)  # no 02:00 on 26 March
        after = dates > np.datetime64(issue)
        moved = market.copy()
        values = moved[columns].to_numpy(dtype=float, copy=True)
        codes = pd.factorize(dates)[0]
        noise = np.random.default_rng(1).normal(0, 3, size=(codes.max() + 1, len(columns)))[codes]
        where = after[:, None] & np.isfinite(values)
        values[where] += noise[where]
        moved[columns] = values
        for what, frame_, end in (("values", moved, last), ("end", market, issue)):
            other = run(frame_, end)
            assert {key: fit for key, fit in other.items() if key[0] == issue} == on_d, (issue, what)
            if what == "values":
                assert any(other.get(key) != base[key] for key in base if key[0] > issue)


# ---------------------------------------------------------------------------
# newest_target="published": the newest date's slots its rule doesn't know yet are masked
# ---------------------------------------------------------------------------

PUBLISHED = {
    # Actuals 1 h after their hour, at 11:45: the anchor's periods 0-9 (to 10:00) are known, 10-23 aren't.
    "load realised 1h": (Availability.realised("1h"), list(range(10))),
    # Day-ahead prices published at 13:00 the day before: the anchor's 24 are known at 11:45.
    "price published 13:00 D-1": (Availability.published("13:00", 1), list(range(24))),
}


@pytest.mark.parametrize("horizon", [1, 2])
@pytest.mark.parametrize("case", PUBLISHED)
def test_published_masks_the_slots_its_rule_doesn_t_know_on_the_anchor(market, case, horizon):
    """A day unit with `newest_target="published"` on the anchor Monday 13 March: its span is the W dates up to the newest one
    any period's target is known on (the anchor + 0 - h, one date later than "d-1"), and on that date the slots of the periods
    whose target isn't known yet are masked, counted by hand from the rule; every other slot trains. Each record's
    `newest_target_day` is its period's, and the forecasts are the hand fit's on those slots, to 1e-10."""
    rule, known = PUBLISHED[case]
    info = InformationSet("11:45", "Europe/Brussels", rules={**RULES, "price": rule})
    masks = []
    real = DayRidge._fit_days

    def spy(self, x, y, y_mask):
        masks.append(y_mask.copy())
        return real(self, x, y, y_mask)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DayRidge, "_fit_days", spy)
        ref = DayRidge(newest_target="published").run(
            market, "2023-03-13", "2023-03-15", horizon=horizon, information_set=info
        )
    by_horizon = dict(zip((1, 2)[:horizon], masks))
    mask = by_horizon[horizon]
    assert mask.shape == (WINDOW, 24)
    assert mask[:-1].all() and np.flatnonzero(mask[-1]).tolist() == known
    records = by_key(ref._results)
    anchor = day("2023-03-13")
    for (_, period, hz), record in records.items():
        k = 0 if period in known else -1
        assert record["newest_target_day"] == (anchor + datetime.timedelta(days=min(k, hz - 1))).isoformat()
    newest = {period: (0 if period in known else -1) for period in range(24)}
    for issue in ("2023-03-13", "2023-03-15"):
        for period, value in day_reference(market, issue, horizon, newest=newest).items():
            assert abs(records[(issue, period, horizon)]["prediction"] - value) < 1e-10


@pytest.mark.parametrize("case", PUBLISHED)
def test_a_published_pooled_unit_reads_each_period_s_own_span(market, case):
    """Each period's rows are its period unit's: the W dates up to the anchor + k(p) - h, so the stacked rows of the newest
    date are the known periods' alone; the forecasts are the hand fit's on those rows, to 1e-10."""
    rule, known = PUBLISHED[case]
    info = InformationSet("11:45", "Europe/Brussels", rules={**RULES, "price": rule})
    sizes = []
    real = PooledOLS._fit_model

    def spy(self, train_x, train_y):
        sizes.append(len(train_x))
        return real(self, train_x, train_y)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PooledOLS, "_fit_model", spy)
        PooledOLS(newest_target="published").run(market, "2023-03-13", "2023-03-13", horizon=1, information_set=info)
        PooledOLS().run(market, "2023-03-13", "2023-03-13", horizon=1)
    published, plain = sizes
    assert published == plain == WINDOW * 24  # W dates of each period, the known ones a date later
    newest = {period: (0 if period in known else -1) for period in range(24)}
    found = by_key(
        PooledOLS(newest_target="published")
        .run(market, "2023-03-13", "2023-03-15", horizon=1, information_set=info)
        ._results
    )
    for issue in ("2023-03-13", "2023-03-15"):
        for period, value in pooled_reference(market, issue, 1, newest=newest).items():
            assert abs(found[(issue, period, 1)]["prediction"] - value) < 1e-10


# ---------------------------------------------------------------------------
# Predictor functions: lists from the plan only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_a_function_returning_the_fixed_list_is_the_fixed_list(market, kind):
    """A unit reads its list from the plan, the only place a worker takes it from (nn_api's day model got none from the
    model's own empty list): a function of `ctx` returning the fixed list gives its records, apart from `predictor_list`."""
    model = KINDS[kind]
    names = DAY_PREDICTORS if kind == "day" else POOLED_PREDICTORS[:-1]
    fixed = model(names).run(market, *WEEK, horizon=2)._results
    planned = model(lambda ctx: [name.format(horizon=ctx.horizon) for name in names]).run(market, *WEEK, horizon=2)
    assert {r["predictor_list"] for r in planned._results} == {r["list_hash"] for r in fixed}
    assert keyed(planned._results, "predictor_list") == keyed(fixed)


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def test_the_manifest_records_the_unit_kind_and_target_transform_and_a_resume_compares_them(market, tmp_path):
    for model, kind, transform in (
        (DayRidge(), "day", "per_period"),
        (PooledOLS(), "pooled", "per_period"),
        (PooledOLS(target_transform="pooled"), "pooled", "pooled"),
    ):
        save = tmp_path / f"{kind}-{transform}" / "m.jsonl"
        model.run(market, WEEK[0], WEEK[0], horizon=1, save_to=str(save))
        manifest = json.loads(manifest_path(save).read_text())
        assert (manifest["unit_kind"], manifest["target_transform"]) == (kind, transform)
        assert "block_predict" not in manifest["model_params"]
    save = tmp_path / "ols" / "m.jsonl"
    OLSModel(POOLED_PREDICTORS[:-1], WINDOW).run(market, WEEK[0], WEEK[0], horizon=1, save_to=str(save))
    assert not {"unit_kind", "target_transform"} & set(json.loads(manifest_path(save).read_text()))

    save = tmp_path / "day-per_period" / "m.jsonl"
    stored = json.loads(manifest_path(save).read_text())
    for field, other in (("unit_kind", "pooled"), ("target_transform", "pooled")):
        manifest_path(save).write_text(json.dumps({**stored, field: other}))
        with pytest.raises(ManifestMismatchError, match=f"{field}: '{other}' \\(existing results\\)"):
            DayRidge().run(market, WEEK[0], WEEK[0], horizon=1, save_to=str(save))
    manifest_path(save).write_text(json.dumps(stored))
    DayRidge().run(market, WEEK[0], WEEK[0], horizon=1, save_to=str(save))  # the same: resumes


# ---------------------------------------------------------------------------
# verify and the adapter suite (step 6c)
# ---------------------------------------------------------------------------

VERIFY_INFO = InformationSet(rules=RULES)


@pytest.mark.filterwarnings("ignore:overflow encountered in sinh")  # the control's 1e6 noise, mapped back by sinh
@pytest.mark.parametrize("kind", KINDS)
def test_verify_is_ok_with_its_control_rerun_co_batch_and_cut_frame(market, kind):
    """`adapter_suite.check_verify` on a block's first, middle and last (spring) day: no forecast moves under the probe, every
    one under the control, two runs agree, the six other days of the block named and noise in them moves nothing, and the
    frame cut after the day gives the same bits. A model that reads the prices of its call's other days is a leak, and one
    that counts its call's distinct rows fails on the cut frame."""

    def make(name, **params):
        return KINDS[kind](name=name, **params)

    suite.check_verify(make, market, VERIFY_INFO, [WEEK[0], "2023-03-22", WEEK[1]], items=6)
    with pytest.raises(AssertionError, match="differ when the frame ends on 2023-03-20"):
        suite.check_verify(lambda name: make(name, by_rows=1e-3), market, VERIFY_INFO, [WEEK[0]])
    if kind == "day":  # the call's second column is D-1's 08:00 price, unknown on the block's rows from D+2 on
        with pytest.raises(AssertionError):
            suite.check_verify(lambda name: make(name, by_call=1e-3), market, VERIFY_INFO, [WEEK[0]])


@pytest.fixture
def registered(monkeypatch):
    from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY

    for cls in KINDS.values():
        monkeypatch.setitem(COMPONENT_REGISTRY["models"], cls.__name__, __name__)


SUITE_TEST = ("2023-03-15", "2023-03-28")


@pytest.mark.parametrize("kind", KINDS)
def test_the_adapter_suite(market, tmp_path, monkeypatch, registered, kind):
    """YAML round trip, a loop and its shards, a resume after deleting 40 % of the records, `engine="on"` (which takes neither
    kind), no card to pre-flight or show."""

    def make(name, **params):
        return KINDS[kind](name=name, **params)

    suite.check_yaml_round_trip(make, market, tmp_path, SUITE_TEST, horizon=2)
    variants = ({}, {"training_window": 21}, {"refit": "1D"})
    suite.check_loop_and_shards(make, market, tmp_path / "loop", SUITE_TEST, horizon=2, variants=variants)
    suite.check_resume(make, market, tmp_path / "resume", SUITE_TEST, horizon=2)
    suite.check_engine(make, market, SUITE_TEST, horizon=2)
    suite.check_preflight_without_extra(make, monkeypatch, market, SUITE_TEST)
    suite.check_provenance(make, market, tmp_path / "provenance", SUITE_TEST)


@pytest.mark.parametrize("kind", KINDS)
def test_the_adapter_suite_s_calendar_counts(kind):
    """23 forecasts for the spring target day, 24 for the autumn one with the repeated hour's actual NaN; 96 quarter-hours a
    day at 15 minutes."""
    both = pd.concat([frame(start="2023-02-14", days=46, horizon=1), frame(start="2023-09-17", days=46, horizon=1)])
    suite.check_dst_counts(lambda name, **p: KINDS[kind](name=name, **p), both, "2023-03-26", "2023-10-29")
    quarter = frame(start="2023-01-01", days=60, freq="15min", horizon=1)
    names = profile_columns("15min") if kind == "day" else POOLED_PREDICTORS

    def make(name, **params):
        return KINDS[kind](names, name=name, **params)

    suite.check_quarter_hours(make, quarter, ("2023-02-20", "2023-02-21"))
