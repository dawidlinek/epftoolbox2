"""`ShapeModel` (step 7f): a stored hourly forecast plus a learner's forecast of each quarter-hour's deviation from its hour's mean.

Pinned: the hand fit (per-slot OLS of y - ȳ over native training days, plus the base on the issue row; an intercept-only learner
the per-slot mean deviation, B24_static); that the model is its learner run on the deviation, plus the base, for every kind of
learner (refit, a list of windows, pooled units); the native-day filter on a frame that turns 15-minute mid-history (the
2025-10-01 pattern); the keys of both switches; that the training rows and the issue row read the same chained base; `verify`
with its control; the planted leak of the actual hour mean as base; the newest targets under `newest_target="published"`; YAML,
the manifest and the refusals.
"""

import datetime
import json
import math
import warnings
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.sources import ForecastSource
from epftoolbox2.data.transformers import LagTransformer, PeriodStatsTransformer, ResampleTransformer
from epftoolbox2.data.transformers.period_stats import hour_starts
from epftoolbox2.information_set import Availability, InformationSet, InformationSetError
from epftoolbox2.models import (
    BaseModel,
    HistGBModel,
    LEARModel,
    NaiveModel,
    OLSModel,
    PooledModel,
    ResidualModel,
    RidgeModel,
    ShapeModel,
    _worker,
)
from epftoolbox2.models._calendar import NewestTargets
from epftoolbox2.models._worker import _worker_init, current_fit, member_rows
from epftoolbox2.models.shape import distinct_dates, hour_mean_column, hour_means, native_column
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

ZONE = "Europe/Warsaw"
SWITCH = "2023-03-10"  # the first natively 15-minute date: before it each hour's four quarter-hours hold one value
REGRESSORS = ["load_hour_dev_d+{horizon}"]
BASE = "base_fc_d+{horizon}"
RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published("13:00", 1),
}
INFO = InformationSet("11:45", ZONE, rules=RULES)
WIDE = (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
NOT_CHAINED = "chained column"


class InProcessPool:
    """Stands in for the worker pool: in this process, so the fits can be watched, and fast."""

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


def local_dates(index):
    wall = index.tz_localize(None) if index.tz is not None else index
    return np.asarray(wall.normalize())


def step_filled(df, switch):
    """`df` with the price of every date before `switch` replaced by its hour's mean on each quarter-hour: hourly data stepped
    over the quarter-hours, as `EntsoeSource(prefer_15min=True)` gives the history before the 2025-10-01 switch."""
    out = df.copy()
    hourly = out["price"].groupby(hour_starts(out.index)).transform("mean")
    before = local_dates(out.index) < np.datetime64(switch)
    out.loc[before, "price"] = hourly[before]
    out["price_d-1"] = calendar_shift(out["price"], -1)
    return out


def with_shape_columns(df, seed=5):
    """The shape inputs: the load's within-hour deviation, a stand-in hourly forecast `base_fc` (the hour's mean price plus
    noise, one value per hour) and the actual hour mean, each at D+1 and D+2."""
    out = df.copy()
    stats = PeriodStatsTransformer(columns=["price", "load"]).transform(out[["price", "load"]])
    out["price_hour_mean"] = stats["price_hour_mean"]
    out["load_hour_dev"] = stats["load_hour_dev"]
    keys = hour_starts(out.index)
    _, inverse = np.unique(keys, return_inverse=True)
    out["base_fc"] = out["price_hour_mean"] + np.random.default_rng(seed).normal(0, 4, inverse.max() + 1)[inverse]
    for h in (1, 2):
        for name in ("load_hour_dev", "base_fc", "price_hour_mean"):
            out[f"{name}_d+{h}"] = calendar_shift(out[name], h)
    return out


@pytest.fixture(scope="module")
def frame():
    """15-minute data from 2023-02-01 for 100 days, hourly (step-filled) before 2023-03-10, over the spring switch."""
    return with_shape_columns(
        step_filled(make_market_data("2023-02-01", 100, freq="15min", noise=3.0, seed=11), SWITCH)
    )


@pytest.fixture(scope="module")
def autumn():
    """15-minute data from 2023-10-01 for 40 days, native throughout, over the autumn switch."""
    return with_shape_columns(make_market_data("2023-10-01", 40, freq="15min", noise=3.0, seed=12))


SHORT = "reaches before the first native"  # the warning of fits short of native days


def quiet_run(model, data, *args, **kwargs):
    """`model.run` without the warning that a stand-in base isn't chained, nor that of fits short of native days (a test that
    trains across the switch on purpose sets `min_native_days`)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*{NOT_CHAINED}", category=UserWarning)
        warnings.filterwarnings("ignore", message=f".*{SHORT}", category=UserWarning)
        return model.run(data, *args, freq="15min", **kwargs)


class Oracle:
    """Lookups of one frame built without the model, each once: a column by (local date, quarter-hour of the day), the issue
    row's (the second occurrence) on a repeated hour, and the native dates (no repeated hour in the frames they're used on)."""

    def __init__(self, frame):
        self.frame = frame
        self._slots = {}
        self._native = None

    def slot(self, column):
        if column not in self._slots:
            series = self.frame[column]
            wall = series.index.tz_localize(None)
            periods = wall.hour * 4 + wall.minute // 15
            self._slots[column] = {(d, int(p)): float(v) for d, p, v in zip(wall.date, periods, series.to_numpy())}
        return self._slots[column]

    @property
    def native(self):
        if self._native is None:
            wall = self.frame.index.tz_localize(None)
            table = pd.DataFrame({"date": wall.date, "hour": wall.hour, "value": self.frame["price"].to_numpy()})
            hours = table.groupby(["date", "hour"])["value"].agg(["count", "min", "max"])
            distinct = (hours["count"] == 4) & (hours["max"] != hours["min"])
            self._native = {day for (day, _), on in distinct.items() if on}
        return self._native


@pytest.fixture(scope="module")
def oracle(frame):
    return Oracle(frame)


def by_key(records):
    return {(r["hour"], r["horizon"], r["run_date"]): r for r in records}


# -- the hand fit --------------------------------------------------------------------------------------------------------


def training_rows(oracle, issue, horizon, period, window):
    """(deviation, regressor) of each training row of (period, horizon) for `issue`, from the frame alone: rows dated D-1-h-W+1 to
    D-1-h, a row's target the quarter-hour's price on its target date less its hour's mean there, native target dates only."""
    price = oracle.slot("price")
    regressor = oracle.slot(f"load_hour_dev_d+{horizon}")
    native = oracle.native
    rows = []
    for back in range(window):
        day = issue - datetime.timedelta(days=1 + horizon + back)
        target = day + datetime.timedelta(days=horizon)
        hour = [price.get((target, 4 * (period // 4) + k), math.nan) for k in range(4)]
        if target not in native or (day, period) not in regressor or not np.isfinite(hour).all():
            continue
        y = price[(target, period)] - float(np.mean(hour))
        if np.isfinite(regressor[(day, period)]):
            rows.append((y, regressor[(day, period)]))
    return np.array(rows)


@pytest.mark.parametrize(
    "issue", ["2023-03-20", "2023-03-28"], ids=["across the native switch", "after the spring switch"]
)
def test_the_forecast_is_a_per_slot_ols_of_the_deviation_plus_the_base(frame, oracle, issue):
    window = 21  # across the switch: its native days alone, which min_native_days=1 lets train
    model = ShapeModel(BASE, OLSModel(REGRESSORS, training_window=window, fit_intercept=True), min_native_days=1)
    found = by_key(quiet_run(model, frame, issue, issue, horizon=2)._results)
    day = datetime.date.fromisoformat(issue)
    checked = 0
    for horizon in (1, 2):
        base = oracle.slot(f"base_fc_d+{horizon}")
        regressor = oracle.slot(f"load_hour_dev_d+{horizon}")
        for period in range(96):
            rows = training_rows(oracle, day, horizon, period, window)
            if (day, period) not in base:
                assert (period, horizon, issue) not in found
                continue
            design = np.column_stack([np.ones(len(rows)), rows[:, 1]])
            beta = np.linalg.lstsq(design, rows[:, 0], rcond=None)[0]
            expected = base[(day, period)] + beta[0] + beta[1] * regressor[(day, period)]
            record = found[(period, horizon, issue)]
            assert record["prediction"] == pytest.approx(expected, abs=1e-9), (period, horizon)
            assert len(record["coefficients"]) == 1  # the learner's
            checked += 1
            if issue == "2023-03-20":
                assert len(rows) < window  # the step-filled dates before the switch dropped
    assert checked >= 2 * 92


def test_an_intercept_only_learner_forecasts_the_per_slot_mean_deviation(frame, oracle):
    """B24_static: a learner with no predictor and an intercept (`LEARModel([], scaler="standard")`: LEAR forecasts the training
    mean of a fit of no column) forecasts the base plus the mean deviation of its slot over the native days of its window."""
    issue, window = "2023-03-22", 28
    model = ShapeModel(BASE, LEARModel([], training_window=window, scaler="standard"), min_native_days=1)
    found = by_key(quiet_run(model, frame, issue, issue, horizon=1)._results)
    day = datetime.date.fromisoformat(issue)
    base = oracle.slot("base_fc_d+1")
    for period in range(96):
        rows = training_rows(oracle, day, 1, period, window)
        assert 0 < len(rows) < window
        expected = base[(day, period)] + float(np.mean(rows[:, 0]))
        assert found[(period, 1, issue)]["prediction"] == pytest.approx(expected, abs=1e-9), period


# -- the model is its learner on the deviation plus the base ------------------------------------------------------------

# (learner, its issue days): the slow cells over fewer days, the spring switch (2023-03-26) in every cell's training rows, and the
# step-filled dates before 2023-03-10 in every cell's windows.
DAYS, SHORT_DAYS = ("2023-03-23", "2023-03-28"), ("2023-03-27", "2023-03-28")
LEARNERS = {
    "ols": (lambda: OLSModel(REGRESSORS, training_window=21, fit_intercept=True), DAYS),
    "ols refit": (lambda: OLSModel(REGRESSORS, training_window=21, refit="7D", fit_intercept=True), DAYS),
    "ols windows": (lambda: OLSModel(REGRESSORS, training_window=[14, 21], fit_intercept=True), SHORT_DAYS),
    "ols mask fit": (
        lambda: OLSModel([*REGRESSORS, "is_weekend_d+{horizon}"], training_window=21, binary_mask="fit"),
        DAYS,
    ),
    "ridge": (lambda: RidgeModel(REGRESSORS, training_window=21), SHORT_DAYS),
    "lear refit": (
        lambda: LEARModel([*REGRESSORS, "load_d+{horizon}"], training_window=21, refit="7D", scaler="standard"),
        DAYS,
    ),
    "hgb pooled": (lambda: HistGBModel([*REGRESSORS, "period"], training_window=21, refit="7D", max_iter=15), DAYS),
}


def deviation_frame(frame, oracle):
    """`frame` with `dev`, the price less its hour's mean on native dates and NaN on the others, built without the model."""
    out = frame.copy()
    native = np.isin(np.asarray([d for d in out.index.tz_localize(None).date]), list(oracle.native))
    out["dev"] = np.where(native, (out["price"] - out["price_hour_mean"]).to_numpy(), np.nan)
    return out


@pytest.mark.parametrize("name", list(LEARNERS))
def test_the_model_is_its_learner_run_on_the_deviation_plus_the_base(frame, oracle, name):
    """The learner run with the target `dev` (price less hour mean, NaN off the native dates) gives each forecast's learner part
    to the bit: the ShapeModel's forecast is the issue row's base plus it (each window's of a list, then their mean), on the
    same keys. `min_native_days=1` lets the fits train on the native days of windows across the switch."""
    build, (start, end) = LEARNERS[name]
    shape = quiet_run(ShapeModel(BASE, build(), min_native_days=1), frame, start, end, horizon=2)
    plain = build().run(deviation_frame(frame, oracle), start, end, target="dev", horizon=2, freq="15min")
    found, learned = by_key(shape._results), by_key(plain._results)
    days = (datetime.date.fromisoformat(end) - datetime.date.fromisoformat(start)).days + 1
    assert found.keys() == learned.keys() and len(found) >= days * 2 * 92
    for key, record in learned.items():
        day = datetime.date.fromisoformat(key[2])
        base = oracle.slot(f"base_fc_d+{key[1]}")[(day, key[0])]
        if "member_predictions" in record:
            members = [base + member for member in record["member_predictions"]]
            assert found[key]["member_predictions"] == members, key
            assert found[key]["prediction"] == math.fsum(members) / len(members), key
        else:
            assert found[key]["prediction"] == base + record["prediction"], key
        assert found[key]["coefficients"] == record["coefficients"], key
        assert found[key]["actual"] == oracle.slot("price")[(day + datetime.timedelta(days=key[1]), key[0])], key


# -- native days -----------------------------------------------------------------------------------------------------------


def test_distinct_dates_need_an_hour_of_four_values_not_all_equal():
    index = pd.date_range("2023-05-01", periods=3 * 96, freq="15min", tz=ZONE)
    values = np.repeat(np.arange(3 * 24, dtype=float), 4)  # every hour step-filled
    values[96 + 40] += 1.0  # 2023-05-02: one quarter-hour off its hour's value, so one hour of distinct values
    values[2 * 96 + 41] = np.nan  # 2023-05-03: the same hour with a quarter-hour missing, so none
    values[2 * 96 + 40] += 1.0
    flags = distinct_dates(pd.Series(values, index=index), 4)
    assert [bool(flags[day * 96]) for day in range(3)] == [False, True, False]
    assert (flags.reshape(3, 96) == flags.reshape(3, 96)[:, :1]).all()  # one flag per date


@pytest.mark.parametrize("native", ["distinct", SWITCH, datetime.date(2023, 3, 10)])
def test_the_native_filter_drops_exactly_the_step_filled_dates(frame, native, monkeypatch):
    """Each fit trains on the native dates of its span and on no other: the training rows' target dates (the fit's own day ids,
    `current_fit().row_days`, plus the horizon) are the span's dates from the switch on, for both rules (`min_native_days=0`
    lets every fit across the switch train)."""
    seen = []
    original = OLSModel._fit_predict

    def spy(self, train_x, train_y, test_x):
        fit = current_fit()
        seen.append((fit.unit, fit.row_days.copy()))
        return original(self, train_x, train_y, test_x)

    monkeypatch.setattr(OLSModel, "_fit_predict", spy)
    window = 14
    learner = OLSModel(REGRESSORS, training_window=window, fit_intercept=True)
    quiet_run(ShapeModel(BASE, learner, native=native, min_native_days=0), frame, "2023-03-14", "2023-03-17", horizon=2)
    first = frame.index[0].date()
    switch = datetime.date.fromisoformat(SWITCH)
    assert len(seen) == 4 * 2 * 96
    for unit, row_days in seen:
        targets = {first + datetime.timedelta(days=int(day) + unit.horizon) for day in row_days}
        span = {unit.anchor - datetime.timedelta(days=back) for back in range(1, window + 1)}
        assert targets == {day for day in span if day >= switch}, unit


def test_both_native_rules_give_the_same_forecasts_here(frame):
    days = ("2023-03-14", "2023-03-15")
    learner = OLSModel(REGRESSORS, 14, fit_intercept=True)
    distinct = quiet_run(ShapeModel(BASE, learner, min_native_days=0), frame, *days, horizon=2)
    dated = quiet_run(ShapeModel(BASE, learner, native=SWITCH, min_native_days=0), frame, *days, horizon=2)
    assert by_key(distinct._results) == by_key(dated._results) and distinct.produced == 2 * 2 * 96


def native_days_of(issue, horizon, window):
    """The native target dates of a window of `window` days for a forecast issued on `issue` at `horizon` (targets D-W to D-1)."""
    switch = datetime.date.fromisoformat(SWITCH)
    targets = [issue - datetime.timedelta(days=back) for back in range(1, window + 1)]
    return sum(day >= switch for day in targets)


@pytest.mark.parametrize("native", ["distinct", SWITCH])
def test_a_fit_short_of_native_days_is_skipped_by_default_and_warned_about(frame, native):
    """Issue days within a window of the switch: by default (`min_native_days=None`, each window's length) every fit whose
    window reaches before the first native date is skipped as `too_few_native_days` before any fit, and one warning names the
    issue days. Before, they trained on 1 to 13 rows of their 14 and some forecast prices far off. The window of 2023-03-24
    (targets 2023-03-10 to 03-23) is the first that is native throughout."""
    learner = OLSModel(REGRESSORS, training_window=14, fit_intercept=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*{NOT_CHAINED}", category=UserWarning)
        warnings.filterwarnings(
            "ignore", message=".*produced", category=UserWarning
        )  # the count's own shortfall warning
        with pytest.warns(
            UserWarning, match=r"on 13 issue days, from 2023-03-11 to 2023-03-23, a fit's training window"
        ) as w:
            ref = ShapeModel(BASE, learner, native=native).run(
                frame, "2023-03-11", "2023-03-25", horizon=1, freq="15min"
            )
    message = str([x for x in w if SHORT in str(x.message)][0].message)
    assert "first native 15-minute date (2023-03-10)" in message and "skipped as too_few_native_days" in message
    assert ref.skipped == {"too_few_native_days": 13 * 96} and ref.produced == 96 + 92  # 2023-03-26's target day has 92


@pytest.mark.parametrize("refit", [None, "7D"])
def test_min_native_days_lets_fits_with_enough_native_days_train(frame, tmp_path, refit):
    """`min_native_days=3`: a fit with 3 native target dates in its window trains, one with fewer is skipped, its skip record
    saying how many it had; the warning says the others train on fewer days than their window."""
    learner = OLSModel(REGRESSORS, training_window=14, refit=refit, fit_intercept=True)
    model = ShapeModel(BASE, learner, min_native_days=3, name="short")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*{NOT_CHAINED}", category=UserWarning)
        with pytest.warns(UserWarning, match=SHORT):
            ref = model.run(
                frame, "2023-03-11", "2023-03-16", horizon=1, freq="15min", save_to=str(tmp_path / "s.jsonl")
            )
    records = [json.loads(line) for line in (tmp_path / "s.jsonl").read_text().splitlines()]
    for day in ("2023-03-11", "2023-03-12", "2023-03-13", "2023-03-14", "2023-03-15", "2023-03-16"):
        issue = datetime.date.fromisoformat(day)
        anchor = (
            issue if refit is None else issue - datetime.timedelta(days=issue.weekday())
        )  # Mondays, the 7D calendar
        native = native_days_of(anchor, 1, 14)
        on_day = [r for r in records if r["run_date"] == day]
        assert len(on_day) == 96, day
        if native >= 3:
            assert all(r["status"] == "ok" for r in on_day), day
        else:
            assert {r["status"] for r in on_day} == {"too_few_native_days"}, day
            assert {r["detail"] for r in on_day} == {f"window 14: {native} of 3 native days"}, day
    assert ref.produced + ref.skipped["too_few_native_days"] == 6 * 96 and ref.produced > 0


def test_native_days_are_counted_from_the_frame_s_first_row(autumn, tmp_path):
    """The autumn frame is native from its first date, 2023-10-01, so its rows are dated from then and a target date of the
    window before 2023-10-01 + h has no row. Issued 2023-10-04 at horizon 2, a 14-day window's targets 2023-09-20 to 10-03 hold one
    row's, of 2023-10-03: skipped with `min_native_days=3`, though three of the window's dates are native."""
    learner = OLSModel(REGRESSORS, training_window=14, fit_intercept=True)
    model = ShapeModel(BASE, learner, min_native_days=3, name="floor")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*({NOT_CHAINED}|{SHORT}|produced)", category=UserWarning)
        model.run(autumn, "2023-10-04", "2023-10-06", horizon=2, freq="15min", save_to=str(tmp_path / "f.jsonl"))
    records = [json.loads(line) for line in (tmp_path / "f.jsonl").read_text().splitlines()]
    by_day = {}
    for record in records:
        by_day.setdefault((record["run_date"], record["horizon"]), set()).add((record["status"], record.get("detail")))
    assert by_day[("2023-10-04", 2)] == {("too_few_native_days", "window 14: 1 of 3 native days")}
    assert by_day[("2023-10-04", 1)] == {("too_few_native_days", "window 14: 2 of 3 native days")}
    assert by_day[("2023-10-05", 2)] == {("too_few_native_days", "window 14: 2 of 3 native days")}
    assert by_day[("2023-10-05", 1)] == by_day[("2023-10-06", 2)] == {("ok", None)}


def test_an_expanding_learner_has_no_native_minimum_unless_given(frame):
    """An `"expanding"` window has no length to fill: it trains on the native days it has, without a warning, unless given
    `min_native_days`."""
    learner = OLSModel(REGRESSORS, training_window="expanding", fit_intercept=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*{NOT_CHAINED}", category=UserWarning)
        warnings.filterwarnings("error", message=f".*{SHORT}")
        ref = ShapeModel(BASE, learner).run(frame, "2023-03-12", "2023-03-12", horizon=1, freq="15min")
    assert ref.produced == 96
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*produced", category=UserWarning)
        ref = quiet_run(ShapeModel(BASE, learner, min_native_days=5), frame, "2023-03-12", "2023-03-15", horizon=1)
    assert ref.skipped == {"too_few_native_days": 3 * 96} and ref.produced == 96  # 2, 3, 4 and 5 native days


def test_a_day_before_the_switch_with_any_distinct_hour_counts_as_native(frame):
    """`native="distinct"` reads every hour of the date: a step-filled date with one hour of distinct quarter-hours is native,
    where a date rule isn't."""
    data = frame.copy()
    wall = data.index.tz_localize(None)
    on = (wall.normalize() == pd.Timestamp("2023-03-05")) & (wall.hour == 17) & (wall.minute == 30)
    data.loc[on, "price"] += 2.0
    model = ShapeModel(BASE, OLSModel(REGRESSORS, 14, fit_intercept=True))
    model._set_freq("15min")
    native = model._preprocess(data[["price", "base_fc_d+1"]], 1, "price")[native_column("price", 1)]
    on_row = native.index.tz_localize(None).normalize() == pd.Timestamp("2023-03-04")  # its target date is 2023-03-05
    assert (native[on_row] == 1.0).all()
    dated = ShapeModel(BASE, OLSModel(REGRESSORS, 14, fit_intercept=True), native=SWITCH)
    dated._set_freq("15min")
    assert (
        dated._preprocess(data[["price", "base_fc_d+1"]], 1, "price")[native_column("price", 1)][on_row] == 0.0
    ).all()


def test_the_hour_mean_is_period_stats_mean_at_the_target_date(frame):
    model = ShapeModel(BASE, OLSModel(REGRESSORS, 14, fit_intercept=True))
    model._set_freq("15min")
    processed = model._preprocess(frame[["price", "base_fc_d+1", "base_fc_d+2"]], 2, "price")
    for h in (1, 2):
        expected = frame[f"price_hour_mean_d+{h}"].to_numpy()
        found = processed[hour_mean_column("price", h)].to_numpy()
        assert np.array_equal(found, expected, equal_nan=True), h
    assert np.array_equal(hour_means(frame["price"], 4), frame["price_hour_mean"].to_numpy(), equal_nan=True)
    # An hour with a quarter-hour missing has no mean, on any of its rows: its other three don't train.
    holed = frame["price"].copy()
    holed.iloc[4 * 24 * 40 + 4 * 10 + 2] = np.nan  # 10:30 on the 41st date
    means = hour_means(holed, 4)
    assert np.isnan(means[4 * 24 * 40 + 4 * 10 : 4 * 24 * 40 + 4 * 11]).all()
    assert np.isfinite(means[4 * 24 * 40 + 4 * 9 : 4 * 24 * 40 + 4 * 10]).all()


# -- both switches ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("which", "start", "end"), [("spring", "2023-03-24", "2023-03-27"), ("autumn", "2023-10-27", "2023-10-30")]
)
def test_the_keys_of_both_switches_are_a_96_model_run_s(frame, autumn, which, start, end, monkeypatch):
    """The ShapeModel's keys and actuals are those of a plain 15-minute model on the same days: 92 on the spring day, 96 on an
    ordinary one and on the autumn day, whose repeated hour's 4 keys have no actual (its two occurrences share a wall-clock
    period). No training row reads the skipped hour (there is none) or the repeated hour's target (there is none)."""
    data = frame if which == "spring" else autumn
    seen = []
    original = OLSModel._fit_predict

    def spy(self, train_x, train_y, test_x):
        fit = current_fit()
        seen.append((fit.unit, fit.row_days.copy()))
        return original(self, train_x, train_y, test_x)

    monkeypatch.setattr(OLSModel, "_fit_predict", spy)
    shape = quiet_run(ShapeModel(BASE, OLSModel(REGRESSORS, 14, fit_intercept=True)), data, start, end, horizon=2)
    monkeypatch.setattr(OLSModel, "_fit_predict", original)
    plain = OLSModel([*REGRESSORS, "base_fc_d+{horizon}"], 14, fit_intercept=True).run(
        data, start, end, horizon=2, freq="15min"
    )
    found, expected = by_key(shape._results), by_key(plain._results)
    assert found.keys() == expected.keys()
    assert {k: r["actual"] for k, r in found.items()} == pytest.approx(
        {k: r["actual"] for k, r in expected.items()}, nan_ok=True
    )
    per_day = pd.Series([record["target_date"] for record in found.values()]).value_counts()
    switch = "2023-03-26" if which == "spring" else "2023-10-29"
    assert per_day[switch] == (2 * 92 if which == "spring" else 2 * 96)  # horizons 1 and 2
    if which == "spring":  # the switch day as an issue day has no row in the skipped hour either
        assert per_day["2023-03-27"] == per_day["2023-03-28"] == 96 + 92
    if which == "autumn":
        missing = [k for k, r in found.items() if r["target_date"] == switch and not np.isfinite(r["actual"])]
        assert sorted({k[0] for k in missing}) == [8, 9, 10, 11]
    first = data.index[0].date()
    switch_day = datetime.date.fromisoformat(switch)
    for unit, row_days in seen:
        if unit.period in (8, 9, 10, 11):  # 02:00-02:59, the hour the switch skips or repeats
            targets = {first + datetime.timedelta(days=int(day) + unit.horizon) for day in row_days}
            assert switch_day not in targets, unit


# -- the chained base --------------------------------------------------------------------------------------------------------


def hourly_of(frame15):
    """The hourly frame the upstream runs on: the hour means of the 15-minute columns, with their `_d+1` and `_d+2`."""
    columns = ["price", "load", "wind", "is_weekend", "is_weekday"]
    hourly = frame15[columns].tz_convert("UTC").resample("1h").mean().tz_convert(ZONE)
    for h in (1, 2):
        for name in ("load", "wind", "is_weekend", "is_weekday"):
            hourly[f"{name}_d+{h}"] = calendar_shift(hourly[name], h)
    return hourly


def chained_frame(frame15, directory):
    """`frame15` with the upstream's hourly forecast stepped over the quarter-hours (`price_fc`), its `_d+1` and the ramp's
    hour lag `price_fc_d+1_h-1`, as a data pipeline of ForecastSource, ResampleTransformer and LagTransformer makes them."""
    source = ForecastSource(str(directory), model="price_up", column="price_fc")
    stepped = ResampleTransformer(freq="15min").transform(source.fetch(*WIDE)[["price_fc"]])
    data = frame15.copy()
    data["price_fc"] = stepped["price_fc"].reindex(frame15.index.tz_convert("UTC")).to_numpy()
    data["price_fc_d+1"] = LagTransformer(["price_fc"], [-1], "day").transform(data[["price_fc"]])["price_fc_d+1"]
    ramp = LagTransformer(["price_fc_d+1"], [1], "hour").transform(data[["price_fc_d+1"]])
    data["price_fc_d+1_h-1"] = ramp["price_fc_d+1_h-1"]
    return source, data


@pytest.fixture(scope="module")
def chain(frame, tmp_path_factory):
    """An hourly OLS upstream issued at 09:30, stored from 2023-03-01 (every row the downstream's fits read), chained in."""
    root = tmp_path_factory.mktemp("upstream")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MAX_PROCESSES", "1")
        patch.setenv("THREADS_PER_PROCESS", "1")
        patch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
        pipeline = ModelPipeline(information_set=InformationSet("09:30", ZONE, rules=RULES), availability="warn")
        pipeline.add_model(OLSModel(PREDICTORS, training_window=28, name="price_up"))
        pipeline.run(hourly_of(frame), "2023-03-01", "2023-04-20", target="price", horizon=1, save_dir=str(root))
    source, data = chained_frame(frame, root)
    records = {}
    for line in (root / "price_up.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("status", "ok") == "ok":
            records[(record["run_date"], record["hour"])] = record["prediction"]
    return {"source": source, "data": data, "records": records, "info": INFO.with_source_rules([source])}


CHAINED = ["price_fc_d+{horizon}", "price_fc_d+{horizon}_h-1", *REGRESSORS]
TEST = ("2023-04-10", "2023-04-12")


def chained_model(**kwargs):
    learner = OLSModel(CHAINED, training_window=21, fit_intercept=True)
    return ShapeModel("price_fc_d+{horizon}", learner, name="b24_dyn", **kwargs)


def test_train_and_serve_read_the_same_chained_base(chain, monkeypatch):
    """Every training row of every fit holds, in the base and in the ramp's hour lag, the upstream's stored forecast for that
    row (or nothing where it stored none), and every issue row's base is the stored forecast its forecast adds the learner's
    to."""
    records, seen = chain["records"], []
    original = _worker._fit_member

    def spy(arrays, cfg, hour, hz, day, member, newest, test, list_id, predictors, *rest, **kwargs):
        rows = member_rows(arrays["days"], day, member, hz, newest)
        index = arrays["col_index"]
        at = pd.DatetimeIndex(arrays["timestamps"][rows]).tz_localize("UTC")
        issue = pd.DatetimeIndex(arrays["timestamps"][[test]]).tz_localize("UTC")
        seen.append(
            (
                at,
                arrays["x"][rows, index["price_fc_d+1"]],
                arrays["x"][rows, index["price_fc_d+1_h-1"]],
                issue,
                arrays["x"][test, index["price_fc_d+1"]],
            )
        )
        return original(arrays, cfg, hour, hz, day, member, newest, test, list_id, predictors, *rest, **kwargs)

    monkeypatch.setattr(_worker, "_fit_member", spy)
    ref = chained_model().run(chain["data"], *TEST, horizon=1, freq="15min", information_set=chain["info"])
    assert ref.produced == 3 * 96 and len(seen) == 3 * 96

    def stored(instant):
        wall = instant.tz_convert(ZONE).tz_localize(None)
        return records.get((wall.date().isoformat(), wall.hour))

    compared = 0
    for at, base, ramp, issue, issue_base in seen:
        for instant, value, lagged in zip(at, base, ramp):
            for found, expected in ((value, stored(instant)), (lagged, stored(instant - pd.Timedelta(hours=1)))):
                if expected is None:
                    assert np.isnan(found), instant
                else:
                    assert found == expected, instant
                    compared += 1
        assert issue_base == stored(issue[0])
    assert compared > 3 * 96 * 2 * 19


def test_a_base_that_isn_t_chained_warns(chain, frame):
    learner = OLSModel(REGRESSORS, training_window=21, fit_intercept=True)
    info = InformationSet("11:45", ZONE, rules={**RULES, "base_fc": Availability.published("10:00", 1)})
    with pytest.warns(UserWarning, match=r"base \(base_fc_d\+1\) is no chained column"):
        ShapeModel(BASE, learner).run(frame, *TEST, horizon=1, freq="15min", information_set=info)
    with pytest.warns(UserWarning, match="can't be told to be a chained column without an information set"):
        chained_model().run(chain["data"], *TEST, horizon=1, freq="15min")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=f".*{NOT_CHAINED}")
        chained_model().run(chain["data"], *TEST, horizon=1, freq="15min", information_set=chain["info"])


def test_the_manifest_records_the_chained_base(chain, tmp_path):
    """The base's upstream is in the manifest's `chained_inputs`, as the learner's chained inputs are."""
    learner = OLSModel(REGRESSORS, training_window=21, fit_intercept=True)  # no chained regressor: the base alone
    model = ShapeModel("price_fc_d+{horizon}", learner, name="static")
    model.run(
        chain["data"], *TEST, horizon=1, freq="15min", information_set=chain["info"], save_to=str(tmp_path / "s.jsonl")
    )
    manifest = json.loads((tmp_path / "s.manifest.json").read_text())
    assert manifest["chained_inputs"]["price_fc"]["model"] == "price_up"
    assert "price_fc_d+1" in manifest["predictors"]
    assert manifest["model_params"]["base"] == "price_fc_d+{horizon}"


def test_a_base_from_an_upstream_that_read_something_late_is_late_upstream(frame, tmp_path):
    """The base alone reads the upstream (the learner reads no chained column): an upstream that read wind on its own day,
    after its 09:30 issue, makes the base `late upstream`, refused at raise and listed at warn."""
    pipeline = ModelPipeline(
        information_set=InformationSet("09:30", ZONE, rules={**RULES, "wind": Availability.published("10:00", 0)}),
        availability="warn",
    )
    pipeline.add_model(OLSModel(PREDICTORS, training_window=28, name="price_up"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the upstream's own listing
        pipeline.run(hourly_of(frame), "2023-04-01", "2023-04-20", target="price", horizon=1, save_dir=str(tmp_path))
    source, data = chained_frame(frame, tmp_path)
    info = INFO.with_source_rules([source])
    model = ShapeModel("price_fc_d+{horizon}", OLSModel(REGRESSORS, training_window=7, fit_intercept=True), name="late")
    with pytest.raises(InformationSetError, match="price_fc"):
        model.run(data, *TEST, horizon=1, freq="15min", information_set=info)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ref = model.run(data, *TEST, horizon=1, freq="15min", information_set=info, availability="warn")
    rows = [row for row in ref.availability_listing if row["predictor"] == "price_fc_d+1"]
    assert rows and all(row["status"] == "late upstream" for row in rows)


def test_a_missing_base_on_the_issue_row_skips_its_forecast(frame):
    data = frame.copy()
    wall = data.index.tz_localize(None)
    on = (wall.normalize() == pd.Timestamp("2023-03-28")) & (wall.hour == 10)
    data.loc[on, "base_fc_d+1"] = np.nan
    model = ShapeModel(BASE, OLSModel(REGRESSORS, 14, fit_intercept=True))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*produced", category=UserWarning)  # the count's shortfall warning
        ref = quiet_run(model, data, "2023-03-28", "2023-03-28", horizon=1)
    assert ref.produced == 92 and ref.skipped == {"missing_test_predictors": 4}
    assert all(np.isfinite(record["prediction"]) for record in ref._results)


def test_verify_is_ok_with_its_control_on_the_chained_base(chain):
    """`verify` with its control, and without a warning about the base: its probe runs get the rules to tell a chained column
    by (`_cut_rules_in_probes`)."""
    pipeline = ModelPipeline(information_set=chain["info"], availability="raise").add_model(chained_model())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = pipeline.verify(chain["data"], *TEST, horizon=1, freq="15min", days=2)
    assert not [w for w in caught if NOT_CHAINED in str(w.message)]
    row = report.table().set_index("model").loc["b24_dyn"]
    assert row["status"] == "ok", report.problems()
    assert row["control moved"] > 0


def test_the_actual_hour_mean_of_a_later_date_is_refused_as_base_by_name(frame):
    """`{target}_hour_mean_d+k`, the learner's own training offset, refused by name, as `{target}_d+k` is as a predictor."""
    leak = ShapeModel("price_hour_mean_d+{horizon}", OLSModel(REGRESSORS, training_window=21, fit_intercept=True))
    with pytest.raises(ValueError, match=r"price_hour_mean_d\+1.*look-ahead"):
        quiet_run(leak, frame, *TEST, horizon=1)


def with_hour_mean_ahead(frame):
    """`frame` with `price_hour_mean_h+24`, the actual hour mean 24 elapsed hours ahead: the next day's outside a switch."""
    data = frame.copy()
    data["price_hour_mean_h+24"] = LagTransformer(["price_hour_mean"], [-24], "hour").transform(
        data[["price_hour_mean"]]
    )["price_hour_mean_h+24"]
    return data


def test_the_actual_hour_mean_as_base_is_refused_by_the_availability_check(frame):
    """A planted leak the name doesn't give away: the base `price_hour_mean_h+24`, the actual hour mean 24 hours ahead, whose
    prices are out at 13:00 on D, after the 11:45 issue. Refused at raise, listed at warn, and `verify` finds it."""
    learner = OLSModel(REGRESSORS, training_window=21, fit_intercept=True)
    leak = ShapeModel("price_hour_mean_h+24", learner, name="leak")
    data = with_hour_mean_ahead(frame)
    with pytest.raises(InformationSetError, match=r"price_hour_mean_h\+24"):
        quiet_run(leak, data, *TEST, horizon=1, information_set=INFO)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ref = leak.run(data, *TEST, horizon=1, freq="15min", information_set=INFO, availability="warn")
    assert "price_hour_mean_h+24" in {row["predictor"] for row in ref.availability_listing}
    pipeline = ModelPipeline(information_set=INFO, availability="warn").add_model(leak)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        report = pipeline.verify(data, *TEST, horizon=1, freq="15min", days=1)
    assert report.table().set_index("model").loc["leak", "status"] == "leak"


def test_explain_lists_the_base_judged_on_the_issue_row(chain, frame):
    """`explain()` shows the learner's inputs and, apart, the base: the chained one with its upstream and known at the issue,
    the hour mean ahead late."""
    pipeline = ModelPipeline(information_set=chain["info"]).add_model(chained_model())
    out = pipeline.explain(chain["data"], model="b24_dyn", test=TEST, horizon=1, hour=40, freq="15min", echo=False)
    assert list(out["predictors"]["predictor"]) == ["price_fc_d+1", "price_fc_d+1_h-1", "load_hour_dev_d+1"]
    base = out["base"].set_index("predictor").loc["price_fc_d+1"]
    assert base["role"] == "base" and base["status"] == "ok" and "price_up" in base["rule"]
    leak = ShapeModel("price_hour_mean_h+24", OLSModel(REGRESSORS, 21, fit_intercept=True), name="leak")
    out = (
        ModelPipeline(information_set=INFO)
        .add_model(leak)
        .explain(with_hour_mean_ahead(frame), model="leak", test=TEST, horizon=1, hour=40, freq="15min", echo=False)
    )
    assert out["base"].set_index("predictor").loc["price_hour_mean_h+24", "status"] != "ok"


# -- newest targets under "published" ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("native", ["distinct", SWITCH])
def test_published_newest_targets_are_the_oldest_of_the_hour_or_of_the_date(native, monkeypatch):
    """A quarter-hour's deviation reads every quarter-hour of its hour, and the `distinct` flag every hour of its date: each
    period's newest target is the oldest of its hour's periods, or of the date's."""
    days = [datetime.date(2023, 4, 3), datetime.date(2023, 4, 4)]
    k = (np.arange(96 * 2).reshape(96, 2) * 7 % 5 - 3).astype(np.int64)
    monkeypatch.setattr(BaseModel, "_newest_targets", lambda self, *args: NewestTargets(days, k))
    model = ShapeModel(BASE, OLSModel(REGRESSORS, 14, newest_target="published"), native=native)
    found = model._newest_targets(None, "price", 1, "15min", None, days)
    for period in range(96):
        group = range(96) if native == "distinct" else range(4 * (period // 4), 4 * (period // 4) + 4)
        assert list(found.k[period]) == list(k[list(group)].min(axis=0)), period
    assert found.days == days


def test_with_published_targets_a_quarter_hour_trains_to_its_hour_s_oldest_known_target(frame, monkeypatch):
    """A price known 75 minutes after its quarter-hour ends: at 11:45 on D, D's 10:00 and 10:15 are out, 10:30 and 10:45 not.
    A plain model trains period 40 (10:00) on D's target; a ShapeModel, whose deviation reads the whole hour, stops at D-1 for
    every period of that hour, while the 09:00 hour, all out, trains on D. Checked on the fits' own rows."""
    info = InformationSet(
        "11:45",
        ZONE,
        rules={**RULES, "price": Availability.realised("75min"), "base_fc": Availability.published("10:00", 1)},
    )
    learner = OLSModel(REGRESSORS, training_window=14, newest_target="published", fit_intercept=True)
    issue = datetime.date(2023, 4, 12)
    plain = learner._newest_targets(frame, "price", 1, "15min", info, [issue]).k[:, 0]
    assert list(plain[36:44]) == [0, 0, 0, 0, 0, 0, -1, -1]  # the rule differs within the 10:00 hour
    seen = {}
    original = OLSModel._fit_predict

    def spy(self, train_x, train_y, test_x):
        fit = current_fit()
        seen[fit.unit.period] = int(fit.row_days.max()) + fit.unit.horizon
        return original(self, train_x, train_y, test_x)

    monkeypatch.setattr(OLSModel, "_fit_predict", spy)
    model = ShapeModel(BASE, learner, native=SWITCH)
    quiet_run(model, frame, issue.isoformat(), issue.isoformat(), horizon=1, information_set=info)
    first = frame.index[0].date().toordinal()
    newest = {period: datetime.date.fromordinal(first + day) for period, day in seen.items()}
    assert [newest[p] for p in (36, 39)] == [issue, issue]  # the 09:00 hour: all four out on D
    assert [newest[p] for p in (40, 41, 42, 43)] == [issue - datetime.timedelta(days=1)] * 4


# -- coverage of chained inputs on native rows -----------------------------------------------------------------------------


@pytest.mark.parametrize("native", ["distinct", SWITCH])
def test_coverage_counts_only_rows_with_a_native_target_date(frame, tmp_path, native):
    """An upstream from 2023-03-09, whose forecasts cover every row with a native target date (from 2023-03-10): a fit across
    the switch reads earlier rows without a forecast, which the native filter drops anyway, so the coverage check is silent at
    raise; an upstream from 2023-03-12 leaves native rows uncovered, refused."""
    learner = OLSModel(["price_fc_d+{horizon}", *REGRESSORS], training_window=14, fit_intercept=True)
    for start, covered in (("2023-03-09", True), ("2023-03-12", False)):
        pipeline = ModelPipeline(information_set=InformationSet("09:30", ZONE, rules=RULES), availability="warn")
        pipeline.add_model(OLSModel(PREDICTORS, training_window=28, name="price_up"))
        pipeline.run(hourly_of(frame), start, "2023-03-20", target="price", horizon=1, save_dir=str(tmp_path / start))
        source, data = chained_frame(frame, tmp_path / start)
        model = ShapeModel("price_fc_d+{horizon}", learner, native=native, min_native_days=0, name="cov")
        run = lambda: model.run(  # noqa: E731
            data, "2023-03-14", "2023-03-15", horizon=1, freq="15min", information_set=INFO.with_source_rules([source])
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=f".*{SHORT}", category=UserWarning)
            if covered:
                warnings.filterwarnings("error", message=".*have no value of")
                assert run().produced == 2 * 96
            else:
                with pytest.raises(InformationSetError, match=r"price_fc_d\+1 on .* from 2023-03-09"):
                    run()


# -- YAML, the manifest, the refusals ---------------------------------------------------------------------------------------


def test_yaml_round_trip_nests_the_learner(tmp_path):
    learner = OLSModel(REGRESSORS, training_window=[14, 21], refit="7D", fit_intercept=True, name="inner")
    pipeline = ModelPipeline().add_model(ShapeModel(BASE, learner, native=datetime.date(2023, 3, 10), name="b24"))
    path = tmp_path / "pipeline.yaml"
    pipeline.save(path)
    loaded = ModelPipeline.load(path)
    assert loaded.to_dict() == pipeline.to_dict()
    model = loaded.models[0]
    assert isinstance(model, ShapeModel) and isinstance(model.learner, OLSModel)
    assert (model.base, model.native, model.period, model.name) == (BASE, SWITCH, "1h", "b24")
    assert (model.training_window, model.refit, model.learner.fit_intercept) == ([14, 21], "7D", True)
    saved = pipeline.to_dict()["models"][0]["params"]
    assert set(saved) == {"base", "learner", "period", "native", "name"}


def test_a_resume_with_another_native_rule_or_learner_refuses(frame, tmp_path):
    save_to = str(tmp_path / "shape.jsonl")
    quiet_run(
        ShapeModel(BASE, OLSModel(REGRESSORS, 14, fit_intercept=True)),
        frame,
        "2023-03-28",
        "2023-03-28",
        horizon=1,
        save_to=save_to,
    )
    with pytest.raises(ManifestMismatchError, match="native"):
        quiet_run(
            ShapeModel(BASE, OLSModel(REGRESSORS, 14, fit_intercept=True), native=SWITCH),
            frame,
            "2023-03-28",
            "2023-03-29",
            horizon=1,
            save_to=save_to,
        )
    with pytest.raises(ManifestMismatchError, match="RidgeModel"):
        quiet_run(
            ShapeModel(BASE, RidgeModel(REGRESSORS, 14)), frame, "2023-03-28", "2023-03-29", horizon=1, save_to=save_to
        )


def test_the_period_is_kept_as_the_hour_however_it_is_written():
    assert ShapeModel(BASE, OLSModel(REGRESSORS), period="60min").period == "1h"


def test_the_learner_s_settings_are_the_model_s_and_set_on_the_learner():
    learner = OLSModel(REGRESSORS, training_window=28, refit="7D", newest_target="published", binary_mask=None)
    model = ShapeModel(BASE, learner)
    assert (model.predictors, model.training_window, model.refit, model.newest_target) == (
        learner.predictors,
        28,
        "7D",
        "published",
    )
    learner.training_window = 56
    assert model.training_window == 56 and model._members == (56,)
    with pytest.raises(ValueError, match="set it on the learner"):
        model.training_window = 14
    with pytest.raises(ValueError, match="set it on the learner"):
        model.scaler = "asinh"
    assert not model._engine_capable() and model._worker_model()[0] is OLSModel


class DayLearner(OLSModel):
    _unit_kind = "day"


class TorchLike(OLSModel):
    _residual_learner_ok = False


@pytest.mark.parametrize(
    ("build", "error", "match"),
    [
        (lambda: ShapeModel(BASE, NaiveModel(["base_fc_d+1"])), ValueError, "fits nothing"),
        (lambda: ShapeModel(BASE, ResidualModel(OLSModel(PREDICTORS), OLSModel(PREDICTORS))), TypeError, "learner"),
        (lambda: ShapeModel(BASE, ShapeModel(BASE, OLSModel(REGRESSORS))), TypeError, "learner"),
        (lambda: ShapeModel(BASE, OLSModel(REGRESSORS, normalise_by="load")), ValueError, "per-row target"),
        (lambda: ShapeModel(BASE, OLSModel(lambda h: REGRESSORS)), ValueError, "list of names"),
        (lambda: ShapeModel(BASE, DayLearner(REGRESSORS)), ValueError, "day units"),
        (lambda: ShapeModel(BASE, TorchLike(REGRESSORS)), ValueError, "torch"),
        (lambda: ShapeModel(None, OLSModel(REGRESSORS)), ValueError, "base names"),
        (lambda: ShapeModel("base_{hour}", OLSModel(REGRESSORS)), ValueError, "placeholder"),
        (lambda: ShapeModel(BASE, OLSModel(REGRESSORS), native="sometimes"), ValueError, "native"),
        (lambda: ShapeModel(BASE, OLSModel(REGRESSORS), period="30min"), ValueError, "period"),
        (lambda: PooledModel(ShapeModel(BASE, OLSModel(REGRESSORS))), ValueError, "can't be pooled"),
        (lambda: ResidualModel(OLSModel(PREDICTORS), ShapeModel(BASE, OLSModel(REGRESSORS))), ValueError, "ShapeModel"),
        (lambda: ShapeModel(BASE, OLSModel(REGRESSORS), min_native_days=-1), ValueError, "min_native_days"),
        (lambda: ShapeModel(BASE, OLSModel(REGRESSORS), min_native_days=True), ValueError, "min_native_days"),
    ],
)
def test_what_a_shape_model_refuses(build, error, match):
    with pytest.raises(error, match=match):
        build()


def test_a_per_row_target_and_another_freq_are_refused(frame):
    model = ShapeModel(BASE, OLSModel(REGRESSORS, 14, fit_intercept=True))
    for key, value in (("bounds", (0, None)), ("normalise_by", "load"), ("zero_when", "is_weekend")):
        with pytest.raises(ValueError, match="per-row target"):
            setattr(model, key, value)
    hourly = make_market_data("2023-02-01", 60, noise=3.0)
    with pytest.raises(ValueError, match="15-minute run"):
        model.run(hourly, "2023-03-20", "2023-03-20", horizon=1, freq="1h")
    model.learner.normalise_by = "load"  # changed after the model was built: refused at the run
    with pytest.raises(ValueError, match="per-row target"):
        quiet_run(model, frame, "2023-03-20", "2023-03-20", horizon=1)


def test_other_models_frames_are_unchanged(frame):
    """Only a ShapeModel's period frames hold the auxiliary columns."""
    model = OLSModel(REGRESSORS, 14)
    model._set_freq("15min")
    processed = model._preprocess(frame[["price", "load_hour_dev_d+1"]], 2, "price")
    assert list(processed.columns) == ["price", "load_hour_dev_d+1", "price_d+1", "price_d+2", "day", "hour"]
