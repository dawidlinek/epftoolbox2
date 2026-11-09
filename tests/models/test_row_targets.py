"""Per-row targets (step 7c): `normalise_by`, `bounds` and `zero_when` (`epftoolbox2.models._row_target`).

A record equals a fit by hand with numpy and scikit-learn alone: OLS and Ridge on y / c, c (`cap_d+{horizon}`, a capacity of the
target instant) varying by row, the forecast times the issue row's c; with and without `refit="7D"`, with one window and with a
list, at 1 h, 15 minutes and one row a day, over both clock changes (six cells covering every pair of levels of those five
axes). `bounds` clips after the forecast is scaled back, in multiples of the scale with `normalise_by`; `zero_when` makes a
forecast 0 with no fit and leaves every other record of the model bit for bit; a pooled unit, a block call (`HistGBModel`) and
a period one equal a run on the frame whose target is y / c, times c. The columns are judged as predictors (late at the issue,
late on a training row under a rule that changes over time, through a predictor function too), `verify` passes with its control
and finds a leak planted through either column, and the engine runs such a model on the model path, bit for bit. With all
three None nothing changes: the step 2-5 equivalence grids say so for every existing model, outside the fast suite.
"""

import inspect
import json
import math
import warnings
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import RidgeCV

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
    Use,
)
from epftoolbox2.models._calendar import refit_anchor
from epftoolbox2.models._history import HistorySpec
from epftoolbox2.models._worker import _worker_init, current_fit
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

SWITCH_DAYS = {"spring": "2023-03-26", "autumn": "2023-10-29"}  # Sundays; Warsaw's clocks change at 02:00
PER_HOUR = {"1h": 1, "15min": 4}
GRID = np.logspace(-3, 4, 36)  # RidgeModel's default penalties, computed here
HAND = 1e-9
SCALE = "cap_d+{horizon}"
RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "cap": Availability.published(
        "10:00", 2
    ),  # a capacity of day X out at 10:00 on X-2: known at the issue for X = D+1
    # A flag of day X out at 10:00 on X: known at the issue and fresh, so `verify`'s control perturbs it. (With a static rule,
    # a forecast `zero_when` makes 0 reads nothing the control moves, and `verify` calls it inconclusive.)
    "flag": Availability.published("10:00", 0),
    "night": Availability.static(),
}
INFO = InformationSet(rules=RULES)


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
        return None


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)


def day_of(day, offset=0):
    return (pd.Timestamp(day) + pd.Timedelta(days=offset)).date().isoformat()


def with_columns(data, seed=4):
    """`data` with `cap` (a positive value that changes from row to row), `flag` (0 or 1 by row), `night` (1 before 05:00) and
    their `_d+1`, `_d+2` (the value at the target instant, by local calendar days)."""
    rng = np.random.default_rng(seed)
    out = data.copy()
    out["cap"] = 30.0 + 40.0 * rng.random(len(out))
    out["flag"] = rng.integers(0, 2, len(out)).astype(float)
    out["night"] = (out.index.hour < 5).astype(float)
    for h in (1, 2):
        for name in ("cap", "flag", "night"):
            out[f"{name}_d+{h}"] = calendar_shift(out[name], h)
    return out


def market(switch="spring", freq="1h", seed=3):
    """A frame from 70 days before the switch to 30 after, in Warsaw time; `"1D"` its rows at noon."""
    hourly = make_market_data(
        start=day_of(SWITCH_DAYS[switch], -70),
        days=100,
        freq="1h" if freq == "1D" else freq,
        horizon=2,
        noise=5.0,
        seed=seed,
    )
    return with_columns(hourly.at_time("12:00") if freq == "1D" else hourly, seed=seed + 1)


@pytest.fixture(scope="module")
def spring():
    return market("spring")


def by_key(records):
    return {(r["run_date"], r["hour"], r["horizon"]): r for r in records}


def run(model, data, start, end, horizon=2, freq="1h", **kwargs):
    return model.run(data, test_start=start, test_end=end, horizon=horizon, freq=freq, **kwargs)._results


# ---------------------------------------------------------------------------
# The oracle: OLS or Ridge on y / c of the rows chosen by date, by hand
# ---------------------------------------------------------------------------


def hand(data, freq, period, horizon, day, estimator, window, anchor=None, scale=SCALE):
    """The forecast issued on `day` from the frame with numpy and scikit-learn alone: the rows of `period` dated from
    anchor-W-h to anchor-1-h (the anchor is the day itself without a refit), each one's target at the same wall-clock time h dates
    later (none for a repeated hour) divided by the row's own `scale` value (a row without a positive one drops), the 0/1 columns
    as they are and the others and the target standardised on those rows, least squares (`"ols"`) or `RidgeCV` (`"ridge"`), and
    the issue row (the last of `day`) predicted, scaled back and multiplied by its own `scale` value."""
    names = [name.replace("{horizon}", str(horizon)) for name in PREDICTORS]
    wall = data.index.tz_localize(None)
    dates = wall.normalize()
    periods = (
        np.zeros(len(wall), dtype=int)
        if freq == "1D"
        else wall.hour * PER_HOUR[freq] + wall.minute // (60 // PER_HOUR[freq])
    )
    prices = pd.Series(data["price"].to_numpy(), index=wall)
    once = prices[~prices.index.duplicated(keep=False)]
    x = data[names].to_numpy(dtype=float)
    c = data[scale.replace("{horizon}", str(horizon))].to_numpy(dtype=float)
    newest = pd.Timestamp(anchor or day) - pd.Timedelta(days=1 + horizon)
    oldest = newest - pd.Timedelta(days=window - 1)
    here = np.asarray(periods == period)
    rows = np.flatnonzero(here & np.asarray((dates >= oldest) & (dates <= newest)))
    y = once.reindex(wall[rows] + pd.Timedelta(days=horizon)).to_numpy()
    test = np.flatnonzero(here & np.asarray(dates == pd.Timestamp(day)))[-1]
    usable = np.isfinite(y) & np.isfinite(x[rows]).all(axis=1) & np.isfinite(c[rows]) & (c[rows] > 0)
    used = rows[usable]
    target, train = y[usable] / c[used], x[used]
    binary = np.array(["is_week" in name for name in names])
    means = np.where(binary, 0.0, train.mean(axis=0))
    deviations = np.where(binary, 1.0, train.std(axis=0, ddof=1))
    centre, spread = target.mean(), target.std(ddof=1)
    scaled_x, scaled_y = (train - means) / deviations, (target - centre) / spread
    row = ((x[test] - means) / deviations)[None, :]
    if estimator == "ols":
        coefficients, *_ = np.linalg.lstsq(scaled_x, scaled_y, rcond=None)
        z = float((row @ coefficients)[0])
    else:
        z = float(RidgeCV(alphas=GRID, fit_intercept=True).fit(scaled_x, scaled_y).predict(row)[0])
    return float(c[test] * (z * spread + centre))


def checked_periods(freq):
    if freq == "1D":
        return [0]
    per_hour = PER_HOUR[freq]
    return [*range(2 * per_hour, 3 * per_hour), 12 * per_hour]  # the hour the clocks change, and noon


# Six cells: every level of each axis (estimator, frequency, switch, refit, windows) and every pair of levels.
CELLS = [
    ("ols", "1h", "spring", None, 30),
    ("ridge", "1h", "autumn", "7D", [20, 30]),
    ("ols", "15min", "autumn", "7D", 30),
    ("ridge", "15min", "spring", None, [20, 30]),
    ("ols", "1D", "spring", "7D", [20, 30]),
    ("ridge", "1D", "autumn", None, 30),
]


@pytest.mark.parametrize(("estimator", "freq", "switch", "refit", "window"), CELLS)
def test_a_forecast_is_a_fit_by_hand_on_y_over_c_times_the_issue_row_s_c(estimator, freq, switch, refit, window):
    data = market(switch, freq)
    S = SWITCH_DAYS[switch]
    cls = OLSModel if estimator == "ols" else RidgeModel
    model = cls(PREDICTORS, training_window=window, refit=refit, normalise_by=SCALE)
    records = run(model, data, day_of(S, -1), day_of(S, 2), freq=freq)
    windows = window if isinstance(window, list) else [window]
    checked = 0
    for (run_date, period, horizon), record in by_key(records).items():
        if period not in checked_periods(freq):
            continue
        anchor = None if refit is None else refit_anchor(run_date, refit).isoformat()
        members = [hand(data, freq, period, horizon, run_date, estimator, w, anchor) for w in windows]
        if len(members) > 1:
            assert record["member_predictions"] == pytest.approx(members, abs=HAND, rel=0), (run_date, period, horizon)
        assert record["prediction"] == pytest.approx(math.fsum(members) / len(members), abs=HAND, rel=0), (
            run_date,
            period,
            horizon,
        )
        assert "detail" not in record and record["status"] == "ok"
        checked += 1
    periods = len(checked_periods(freq))
    # 4 issue days, 2 horizons; in spring the switch hour has no row on S, and no target on S from S-1 and S-2.
    assert checked >= (4 * 2 * periods) - (3 * 2 * PER_HOUR.get(freq, 0) if switch == "spring" else 0)
    assert checked > 0


def test_the_forecast_moves_with_the_issue_row_s_scale_alone(spring):
    """Doubling c on the issue rows of one day doubles that day's forecasts and moves no other record: the scale is the issue
    row's, and nothing else of a forecast reads it there."""
    D = "2023-04-12"
    model = OLSModel(PREDICTORS, training_window=30, normalise_by=SCALE)
    base = by_key(run(model, spring, D, day_of(D, 1)))
    moved = spring.copy()
    on_D = moved.index.strftime("%Y-%m-%d") == D
    for h in (1, 2):
        moved.loc[on_D, f"cap_d+{h}"] *= 2.0
    after = by_key(run(OLSModel(PREDICTORS, training_window=30, normalise_by=SCALE), moved, D, day_of(D, 1)))
    for key, record in base.items():
        if key[0] == D:
            assert after[key]["prediction"] == 2.0 * record["prediction"], key
            assert after[key]["coefficients"] == record["coefficients"], key
        else:
            assert after[key] == record, key


UNITS = {
    "period": lambda **kw: OLSModel(PREDICTORS, training_window=30, **kw),
    "period unit": lambda **kw: OLSModel(PREDICTORS, training_window=30, refit="7D", **kw),
    "pooled unit": lambda **kw: HistGBModel(PREDICTORS, training_window=30, periods="pooled", max_iter=20, **kw),
    "block call": lambda **kw: HistGBModel(PREDICTORS, training_window=30, periods="each", max_iter=20, **kw),
}


@pytest.mark.parametrize("unit", list(UNITS))
def test_a_training_row_without_a_positive_scale_drops_and_an_issue_row_without_one_is_skipped(spring, unit):
    """Training rows of (7, 1, D) with c -3, 0 and missing drop out: the forecasts are the run on y / c without those rows' targets
    (and a hand fit, for the period model). The issue row of (9, 1, D) with c 0 is `non_positive_scale`, (10, 1, D) with c missing
    `missing_test_predictors`."""
    D = "2023-04-12"  # a Wednesday: a "7D" unit is fitted on Monday 10 April, on rows up to the 8th
    data = spring.copy()
    stamps = data.index.strftime("%Y-%m-%d %H")
    planted = ["2023-04-05 07", "2023-04-06 07", "2023-04-07 07"]
    for stamp, value in zip(planted, (-3.0, 0.0, np.nan)):
        data.loc[stamps == stamp, "cap_d+1"] = value  # training rows of (7, 1, D)
    data.loc[stamps == f"{D} 09", "cap_d+1"] = 0.0  # the issue row of (9, 1, D)
    data.loc[stamps == f"{D} 10", "cap_d+1"] = np.nan
    ref = UNITS[unit](normalise_by=SCALE).run(data, D, D, horizon=1)
    found = by_key(ref._results)
    assert ref.skipped == {"non_positive_scale": 1, "missing_test_predictors": 1}
    # The oracle: the target of each planted training row (at its instant one day later) missing.
    ratio = divided(data)
    targets = (pd.to_datetime(planted) + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H")
    ratio.loc[stamps.isin(list(targets)), "price"] = np.nan
    on_ratio = by_key(run(UNITS[unit](), ratio, D, D, horizon=1))
    expected = times_scale(data, on_ratio)
    assert found.keys() == on_ratio.keys() - {(D, 9, 1), (D, 10, 1)}
    for key, record in found.items():
        assert record["prediction"] == expected[key], key
    if unit == "period":
        assert found[(D, 7, 1)]["prediction"] == pytest.approx(hand(data, "1h", 7, 1, D, "ols", 30), abs=HAND, rel=0)


# ---------------------------------------------------------------------------
# bounds
# ---------------------------------------------------------------------------


def issue_values(data, records, column):
    """`column` on the issue row of each record (the last row of its run date at its hour), by key."""
    wall = data.index.tz_localize(None)
    out = {}
    for key in records:
        run_date, period, horizon = key
        rows = np.flatnonzero((wall.normalize() == pd.Timestamp(run_date)) & (wall.hour == period))
        out[key] = float(data[column.replace("{horizon}", str(horizon))].to_numpy()[rows[-1]])
    return out


@pytest.mark.parametrize("refit", [None, "7D"])
def test_bounds_clip_after_scaling_back_in_multiples_of_the_scale(spring, refit):
    """The clipped forecast is min(max(c * y_hat, low * c), high * c), to the bit, with the coefficients of the fit unclipped:
    clipping before the scaler's inverse, or in target units, gives other numbers."""
    D = "2023-04-10"
    free = by_key(
        run(OLSModel(PREDICTORS, training_window=30, refit=refit, normalise_by=SCALE), spring, D, day_of(D, 4))
    )
    scales = issue_values(spring, free, SCALE)
    ratios = np.array([record["prediction"] / scales[key] for key, record in free.items()])
    low, high = (float(value) for value in np.quantile(ratios, [0.3, 0.7]))
    model = OLSModel(PREDICTORS, training_window=30, refit=refit, normalise_by=SCALE, bounds=(low, high))
    clipped = by_key(run(model, spring, D, day_of(D, 4)))
    assert clipped.keys() == free.keys()
    bound = {"low": 0, "high": 0}
    for key, record in free.items():
        c = scales[key]
        expected = min(max(record["prediction"], low * c), high * c)
        assert clipped[key]["prediction"] == expected, key
        assert clipped[key]["coefficients"] == record["coefficients"], key
        bound["low"] += expected == low * c
        bound["high"] += expected == high * c
    assert bound["low"] > 20 and bound["high"] > 20


def test_bounds_without_normalise_by_are_in_target_units_and_either_side_may_be_open(spring):
    D = "2023-04-10"
    free = by_key(run(OLSModel(PREDICTORS, training_window=30), spring, D, day_of(D, 2)))
    values = np.array([record["prediction"] for record in free.values()])
    low, high = (float(value) for value in np.quantile(values, [0.25, 0.75]))
    for bounds in ((low, high), (low, None), (None, high)):
        clipped = by_key(run(OLSModel(PREDICTORS, training_window=30, bounds=bounds), spring, D, day_of(D, 2)))
        for key, record in free.items():
            expected = record["prediction"]
            if bounds[0] is not None:
                expected = max(expected, bounds[0])
            if bounds[1] is not None:
                expected = min(expected, bounds[1])
            assert clipped[key]["prediction"] == expected, (bounds, key)


def test_each_training_window_of_a_list_is_clipped_on_its_own(spring):
    """A member's forecast is that of the model with its window alone, bounds and all, and the forecast their mean."""
    D = "2023-04-10"
    common = dict(normalise_by=SCALE, bounds=(1.5, 2.2))
    together = by_key(run(OLSModel(PREDICTORS, training_window=[20, 30], **common), spring, D, D))
    alone = [by_key(run(OLSModel(PREDICTORS, training_window=w, **common), spring, D, D)) for w in (20, 30)]
    for key, record in together.items():
        members = [member[key]["prediction"] for member in alone]
        assert record["member_predictions"] == members, key
        assert record["prediction"] == math.fsum(members) / 2, key


def test_a_naive_model_clips_its_value_and_takes_no_other_per_row_target(spring):
    D = "2023-04-10"
    plain = by_key(run(NaiveModel(["price_d-1"]), spring, D, day_of(D, 1), horizon=1))
    clipped = by_key(run(NaiveModel(["price_d-1"], bounds=(None, 100.0)), spring, D, day_of(D, 1), horizon=1))
    assert any(r["prediction"] > 100.0 for r in plain.values())
    for key, record in plain.items():
        assert clipped[key]["prediction"] == min(record["prediction"], 100.0), key
    naive = NaiveModel(["price_d-1"])
    for key in ("normalise_by", "zero_when"):
        with pytest.raises(ValueError, match="NaiveModel fits nothing"):
            setattr(naive, key, "cap")
    with pytest.raises(TypeError):
        NaiveModel(["price_d-1"], normalise_by="cap")


# ---------------------------------------------------------------------------
# zero_when
# ---------------------------------------------------------------------------


class CountsFits(OLSModel):
    """OLS that keeps the unit of every fit it makes (refit=None and refit alike)."""

    fits: list = []

    def _fit_predict(self, train_x, train_y, test_x):
        type(self).fits.append(current_fit().unit)
        return super()._fit_predict(train_x, train_y, test_x)

    def _fit_model(self, train_x, train_y):
        type(self).fits.append(current_fit().unit)
        return super()._fit_model(train_x, train_y)

    def _predict_fitted(self, fitted, test_x):
        return super()._predict_fitted(fitted, test_x)


@pytest.mark.parametrize("window", [30, [20, 30]])
def test_zero_when_gives_0_without_a_fit_and_leaves_every_other_record_as_it_was(spring, window):
    """`flag` is 0 or 1 by row: where it is 1 on the issue row the forecast is 0, with no coefficients, `detail` "zero_when" and
    status ok (with a list of windows, members of 0), and no fit was made for it; every other record is the record of the model
    without `zero_when`, to the bit (its training rows are untouched). The issue row decides, not the row of the target day."""
    D = "2023-04-10"
    plain = by_key(run(OLSModel(PREDICTORS, training_window=window), spring, D, day_of(D, 1)))
    CountsFits.fits.clear()
    zeroed = by_key(run(CountsFits(PREDICTORS, training_window=window, zero_when="flag"), spring, D, day_of(D, 1)))
    assert zeroed.keys() == plain.keys()
    flags = issue_values(spring, plain, "flag")
    targets = issue_values(spring, plain, "flag_d+{horizon}")
    members = 2 if isinstance(window, list) else 1
    # refit=None: a fit's anchor is its issue day, one fit per member
    fitted = {(unit.anchor.isoformat(), unit.period, unit.horizon) for unit in CountsFits.fits}
    assert len(CountsFits.fits) == members * len(fitted)
    zeros = 0
    for key, record in plain.items():
        if flags[key] == 1.0:
            assert zeroed[key]["prediction"] == 0.0 and zeroed[key]["coefficients"] == []
            assert zeroed[key]["detail"] == "zero_when" and zeroed[key]["status"] == "ok"
            assert zeroed[key]["actual"] == record["actual"]
            if members > 1:
                assert zeroed[key]["member_predictions"] == [0.0, 0.0]
            fields = ("prediction", "coefficients", "member_predictions")
            assert {k: v for k, v in zeroed[key].items() if k not in (*fields, "detail")} == {
                k: v for k, v in record.items() if k not in fields
            }
            assert key not in fitted
            zeros += 1
        else:
            assert zeroed[key] == record, key
            assert key in fitted
    assert 0 < zeros < len(plain)
    assert any(flags[key] != targets[key] for key in plain)  # the target row would say otherwise somewhere


def test_zero_when_comes_after_the_input_skips_and_before_the_scale_and_the_fit(spring):
    """A forecast with a missing input is skipped whatever `zero_when` says, and a missing `zero_when` on the issue row is a
    `missing_test_predictors` skip; a flag of 1 is a forecast of 0 even where the issue row's scale is 0 or missing (a clear-sky
    night) or the fit would fail (`fit_error`): the two places a model's keys differ from the model's without `zero_when`."""
    D = "2023-04-10"
    data = spring.copy()
    stamps = data.index.strftime("%Y-%m-%d %H")
    data.loc[stamps == f"{D} 03", "load_d+1"] = np.nan
    data.loc[stamps == f"{D} 03", "flag"] = 1.0
    data.loc[stamps == f"{D} 04", "flag"] = np.nan
    for hour, cap in (("05", 0.0), ("06", np.nan)):
        data.loc[stamps == f"{D} {hour}", "flag"] = 1.0
        data.loc[stamps == f"{D} {hour}", "cap_d+1"] = cap
    ref = OLSModel(PREDICTORS, training_window=30, zero_when="flag", normalise_by=SCALE).run(data, D, D, horizon=1)
    found = by_key(ref._results)
    assert (D, 3, 1) not in found and (D, 4, 1) not in found
    assert ref.skipped == {"missing_test_predictors": 2}
    assert found[(D, 5, 1)]["detail"] == found[(D, 6, 1)]["detail"] == "zero_when"
    plain = OLSModel(PREDICTORS, training_window=30, normalise_by=SCALE).run(data, D, D, horizon=1)
    assert plain.skipped == {"missing_test_predictors": 2, "non_positive_scale": 1}  # 3 and 6 missing, 5 at 0
    # Two training rows: RidgeCV's leave-one-out can't fit, so every forecast of the plain model is a fit_error.
    flags = issue_values(data, {(D, hour, 1): None for hour in range(24)}, "flag")
    tiny = RidgeModel(PREDICTORS, training_window=2, zero_when="flag").run(data, D, D, horizon=1)
    assert set(by_key(tiny._results)) == {key for key, f in flags.items() if f == 1.0} - {
        (D, 3, 1)
    }  # 3: a missing input
    assert set(tiny.skipped) == {"fit_error", "missing_test_predictors"}


@pytest.mark.parametrize("window", [30, [20, 30]])
def test_a_refit_unit_whose_forecasts_are_all_0_makes_no_fit(spring, window):
    """`night_d+{horizon}` is 1 at the target hours before 05:00: those periods' units fit nothing and forecast 0, the others
    fit once per unit (and member) and give the records of the model without it."""
    D = "2023-04-10"
    plain = by_key(run(OLSModel(PREDICTORS, training_window=window, refit="7D"), spring, D, day_of(D, 6)))
    CountsFits.fits.clear()
    model = CountsFits(PREDICTORS, training_window=window, refit="7D", zero_when="night_d+{horizon}")
    zeroed = by_key(run(model, spring, D, day_of(D, 6)))
    assert zeroed.keys() == plain.keys()
    for key, record in plain.items():
        if key[1] < 5:
            assert (zeroed[key]["prediction"], zeroed[key]["coefficients"], zeroed[key]["detail"]) == (
                0.0,
                [],
                "zero_when",
            )
            if isinstance(window, list):
                assert zeroed[key]["member_predictions"] == [0.0, 0.0]
        else:
            assert zeroed[key] == record, key
    members = 2 if isinstance(window, list) else 1
    assert {unit.period for unit in CountsFits.fits} == set(range(5, 24))
    assert len(CountsFits.fits) == 19 * 2 * members  # one week's unit per (period, horizon, member)


# ---------------------------------------------------------------------------
# Other unit kinds: on the frame whose target is y / c
# ---------------------------------------------------------------------------


def divided(data, horizon=1):
    """`data` with `price` divided by the capacity of its own instant: at horizon h the target of a row is y / c, c the row's
    `cap_d+h`, the per-row target a model with `normalise_by` trains on."""
    return data.assign(price=data["price"] / data["cap"])


def times_scale(data, divided_records, horizon_scale=SCALE):
    """The records of a run on `divided` mapped back: each prediction times its issue row's scale."""
    scales = issue_values(data, divided_records, horizon_scale)
    return {key: scales[key] * record["prediction"] for key, record in divided_records.items()}


@pytest.mark.parametrize(
    "make",
    [
        lambda **kw: OLSModel(PREDICTORS, training_window=30, **kw),
        lambda **kw: OLSModel(PREDICTORS, training_window=30, refit="7D", **kw),
        lambda **kw: HistGBModel(PREDICTORS, training_window=30, periods="pooled", max_iter=20, **kw),
        lambda **kw: HistGBModel(PREDICTORS, training_window=30, periods="each", max_iter=20, **kw),
    ],
    ids=["period", "period unit", "pooled unit", "block call"],
)
def test_a_forecast_is_the_run_on_y_over_c_times_the_issue_row_s_c(spring, make):
    D = "2023-04-10"
    scaled = by_key(run(make(normalise_by=SCALE), spring, D, day_of(D, 2), horizon=1))
    on_ratio = by_key(run(make(), divided(spring), D, day_of(D, 2), horizon=1))
    expected = times_scale(spring, on_ratio)
    assert scaled.keys() == on_ratio.keys()
    for key, record in scaled.items():
        assert record["prediction"] == expected[key], key
        assert record["coefficients"] == on_ratio[key]["coefficients"], key


def test_a_pooled_unit_reads_each_forecast_s_zero_when_and_scale_on_its_own_row(spring):
    D = "2023-04-10"

    def make(**kw):
        return HistGBModel(PREDICTORS, training_window=30, max_iter=20, **kw)

    on_ratio = by_key(run(make(), divided(spring), D, day_of(D, 2), horizon=1))
    expected = times_scale(spring, on_ratio)
    bounds = (1.5, 2.2)
    both = by_key(run(make(normalise_by=SCALE, bounds=bounds, zero_when="flag"), spring, D, day_of(D, 2), horizon=1))
    flags = issue_values(spring, on_ratio, "flag")
    scales = issue_values(spring, on_ratio, SCALE)
    assert both.keys() == on_ratio.keys()
    for key, record in both.items():
        if flags[key] == 1.0:
            assert (record["prediction"], record["coefficients"], record["detail"]) == (0.0, [], "zero_when")
        else:
            c = scales[key]
            assert record["prediction"] == min(max(expected[key], bounds[0] * c), bounds[1] * c), key


# ---------------------------------------------------------------------------
# The columns are inputs: checked, hashed and perturbed as predictors are
# ---------------------------------------------------------------------------


def listed(ctx):
    return PREDICTORS


LATE = {
    # D+1's capacity out at 23:00 on D
    "normalise_by": (SCALE, {"cap": Availability.published("23:00", 1)}, "cap_d+1"),
    # D's flag out at 23:00 on D
    "zero_when": ("flag", {"flag": Availability.published("23:00", 0)}, "flag"),
}


@pytest.mark.parametrize("predictors", [PREDICTORS, listed], ids=["fixed", "function"])
@pytest.mark.parametrize("key", list(LATE))
def test_a_late_column_is_refused_at_raise_and_listed_at_warn(spring, predictors, key):
    column, rules, name = LATE[key]
    late = InformationSet(rules={**RULES, **rules})
    model = OLSModel(predictors, training_window=30, **{key: column})
    D = "2023-04-10"
    with pytest.raises(InformationSetError, match=name.replace("+", r"\+")):
        model.run(spring, D, D, horizon=1, information_set=late, availability="raise")
    ref = model.run(spring, D, D, horizon=1, information_set=late, availability="warn", warn_listing=False)
    assert {row["predictor"] for row in ref.availability_listing} == {name}
    # Known at the issue, nothing is listed.
    assert model.run(spring, D, D, horizon=1, information_set=INFO, availability="warn").availability_listing == []


@pytest.mark.parametrize("predictors", [PREDICTORS, listed], ids=["fixed", "function"])
def test_a_scale_known_on_the_issue_row_but_late_on_a_training_row_is_caught_over_time(spring, predictors):
    """`cap` has no rule for 3 to 5 April (a value no rule holds for is never known): the issue row of 10 April reads 11 April's
    capacity, known, and a training row of (h=1) reads one of the gap, never known, so the over-time check refuses and lists
    it. A training window that ends before the gap passes."""
    gap = [
        Availability.static().during(valid_until="2023-04-02"),
        Availability.static().during(valid_from="2023-04-06"),
    ]
    info = InformationSet(rules={**RULES, "cap": gap})
    D = "2023-04-10"
    model = OLSModel(predictors, training_window=30, normalise_by=SCALE)
    with pytest.raises(InformationSetError, match=r"cap_d\+1"):
        model.run(spring, D, D, horizon=1, information_set=info, availability="raise")
    ref = model.run(spring, D, D, horizon=1, information_set=info, availability="warn", warn_listing=False)
    assert {row["predictor"] for row in ref.availability_listing} == {"cap_d+1"}
    later = "2023-04-20"  # a 10-day window trains on rows of 9 to 18 April, whose capacities are of 10 to 19 April
    short = OLSModel(predictors, training_window=10, normalise_by=SCALE)
    assert short.run(spring, later, later, horizon=1, information_set=info, availability="raise").produced == 24


def verified(model, data, information_set=INFO, **kwargs):
    pipeline = ModelPipeline(information_set=information_set).add_model(model)
    return pipeline.verify(data, "2023-04-12", "2023-04-12", horizon=1, days=1, **kwargs)


@pytest.mark.parametrize("refit", [None, "7D"])
def test_verify_is_ok_with_its_control_and_the_control_moves_through_the_scale(spring, refit):
    """`ok`: no forecast moves under the probe, and the control moves every one, through each series perturbed on its own (a
    series whose own control moved nothing would make it inconclusive), the scale and `zero_when`'s flag among them: a forecast
    of 0 becomes a fitted one when the flag isn't 1. (Not with `bounds`: a control's noise can be clipped onto the forecast it
    perturbed, and a forecast that doesn't move is one the probe can't vouch for.)"""
    model = OLSModel(PREDICTORS, training_window=30, refit=refit, normalise_by=SCALE, zero_when="flag")
    report = verified(model, spring)
    report.assert_ok()
    row = report.table().iloc[0]
    assert row["changed"] == 0 and row["control moved"] > 0


@pytest.mark.parametrize(
    ("key", "rules"),
    [
        ("normalise_by", {"cap": Availability.published("23:00", 1)}),
        ("zero_when", {"flag": Availability.published("23:00", 0)}),
    ],
    ids=["through the scale", "through zero_when"],
)
def test_a_leak_planted_through_a_column_is_a_leak(spring, key, rules):
    """The column's rule puts the issue row's value after the issue time (the run lists it at "warn" and goes on), and the
    model reads it there: perturbing it moves the forecasts."""
    column = SCALE if key == "normalise_by" else "flag"
    model = OLSModel(PREDICTORS, training_window=30, **{key: column})
    report = verified(model, spring, InformationSet(rules={**RULES, **rules}), positive_control=False, rerun=False)
    assert report.table().iloc[0]["status"] == "leak", report.table().to_dict("records")


def test_the_columns_are_read_inputs_not_regressors(spring, tmp_path):
    """Loaded, hashed and listed in the manifest's predictors, recorded in its model_params, compared on resume; the
    regressors' layout and the coefficients are the list's."""
    D = "2023-04-10"
    model = OLSModel(PREDICTORS, training_window=30, normalise_by=SCALE, bounds=[0, 2.5], zero_when="flag")
    path = tmp_path / "ols.jsonl"
    model.run(spring, D, D, horizon=1, save_to=str(path))
    manifest = json.loads(manifest_path(path).read_text())
    assert {"cap_d+1", "flag"} <= set(manifest["predictors"])
    assert {"cap_d+1", "flag"} <= set(manifest["column_hashes"])
    assert manifest["model_params"]["normalise_by"] == SCALE
    assert manifest["model_params"]["bounds"] == [0.0, 2.5]
    assert manifest["model_params"]["zero_when"] == "flag"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert {len(r["coefficients"]) for r in records if r.get("detail") is None} == {len(PREDICTORS)}
    plain = OLSModel(PREDICTORS, training_window=30).run(spring, D, D, horizon=1, save_to=str(tmp_path / "plain.jsonl"))
    plain_manifest = json.loads(manifest_path(plain.path).read_text())
    assert manifest["predictor_layout_hash"] == plain_manifest["predictor_layout_hash"]
    assert not {"normalise_by", "bounds", "zero_when"} & set(plain_manifest["model_params"])
    for changed in (dict(bounds=(0, 2.6)), dict(bounds=(0, 2.5)), dict(normalise_by=None, bounds=(0, 2.5))):
        other = OLSModel(PREDICTORS, training_window=30, **{"normalise_by": SCALE, "zero_when": "flag", **changed})
        if changed == dict(bounds=(0, 2.5)):
            assert other.run(spring, D, D, horizon=1, save_to=str(path)).produced == 24  # the same: it resumes
            continue
        with pytest.raises(ManifestMismatchError, match="model_params"):
            other.run(spring, D, D, horizon=1, save_to=str(path))


def test_a_moved_scale_refuses_a_resume(spring, tmp_path):
    D = "2023-04-10"
    path = tmp_path / "ols.jsonl"
    OLSModel(PREDICTORS, training_window=30, normalise_by=SCALE).run(spring, D, D, horizon=1, save_to=str(path))
    moved = spring.copy()
    moved["cap_d+1"] = moved["cap_d+1"] * 1.01
    with pytest.raises(ManifestMismatchError):
        OLSModel(PREDICTORS, training_window=30, normalise_by=SCALE).run(moved, D, D, horizon=1, save_to=str(path))


def test_the_engine_runs_such_a_model_on_the_model_path_bit_for_bit(spring, tmp_path):
    for key, value in (("normalise_by", SCALE), ("bounds", (0, None)), ("zero_when", "flag")):
        assert not OLSModel(PREDICTORS, **{key: value})._engine_capable(), key
    assert OLSModel(PREDICTORS)._engine_capable()
    D = "2023-04-10"
    files = {}
    for engine in ("off", "on"):
        pipeline = ModelPipeline(engine=engine)
        pipeline.add_model(
            RidgeModel(PREDICTORS, training_window=30, refit="7D", normalise_by=SCALE, zero_when="flag", name="rt")
        )
        pipeline.add_model(RidgeModel(PREDICTORS, training_window=30, refit="7D", name="plain"))
        pipeline.run(spring, D, day_of(D, 3), horizon=2, save_dir=str(tmp_path / engine))
        files[engine] = sorted((tmp_path / engine / "rt.jsonl").read_text().splitlines())
        manifest = json.loads(manifest_path(tmp_path / engine / "rt.jsonl").read_text())
        assert "fit_path" not in manifest
    assert files["on"] == files["off"]
    plain = json.loads(manifest_path(tmp_path / "on" / "plain.jsonl").read_text())
    assert plain.get("fit_path") == "engine-2"  # the plain model beside it did run in an engine group


def test_yaml_writes_the_parameters_only_when_set_and_reads_them_back(tmp_path):
    pipeline = ModelPipeline()
    pipeline.add_model(OLSModel(PREDICTORS, name="plain"))
    pipeline.add_model(
        LEARModel(PREDICTORS, name="lear", normalise_by=SCALE, bounds=(0, 1.05), zero_when="night_d+{horizon}")
    )
    pipeline.add_model(HistGBModel(PREDICTORS, name="hgb", bounds=(None, 5.0)))
    pipeline.add_model(NaiveModel(["price_d-1"], name="naive", bounds=(0, None)))
    path = tmp_path / "pipeline.yaml"
    pipeline.save(path)
    text = path.read_text()
    params = {m["params"]["name"]: m["params"] for m in __import__("yaml").safe_load(text)["models"]}
    assert not {"normalise_by", "bounds", "zero_when"} & set(params["plain"])
    assert params["lear"]["bounds"] == [0.0, 1.05] and params["lear"]["normalise_by"] == SCALE
    assert params["hgb"]["bounds"] == [None, 5.0] and "normalise_by" not in params["hgb"]
    loaded = {model.name: model for model in ModelPipeline.load(path).models}
    assert loaded["lear"].bounds == (0.0, 1.05) and loaded["lear"].zero_when == "night_d+{horizon}"
    assert loaded["hgb"].bounds == (None, 5.0) and loaded["naive"].bounds == (0.0, None)
    assert loaded["plain"].normalise_by is None and loaded["plain"].bounds is None


def test_explain_lists_the_columns_with_their_role(spring, capsys):
    pipeline = ModelPipeline(information_set=INFO)
    pipeline.add_model(
        OLSModel(PREDICTORS, training_window=30, normalise_by=SCALE, bounds=(0, 1.05), zero_when="flag", name="rt")
    )
    pipeline.add_model(OLSModel(PREDICTORS, training_window=30, name="plain"))
    tables = pipeline.explain(spring, model="rt", test=("2023-04-10", "2023-04-12"), horizon=2, hour=12)
    table = tables["row_target"]
    assert table[["role", "predictor", "status"]].values.tolist() == [
        ["normalise_by", "cap_d+2", "ok"],
        ["zero_when", "flag", "ok"],
    ]
    assert "cap_d+2" not in set(tables["predictors"]["predictor"])
    printed = capsys.readouterr().out
    assert "per-row target:" in printed and "times the scale" in printed
    assert "row_target" not in pipeline.explain(
        spring, model="plain", test=("2023-04-10", "2023-04-12"), horizon=2, hour=12
    )


# ---------------------------------------------------------------------------
# What takes them, what refuses them, and their values
# ---------------------------------------------------------------------------


class DayUnits(BaseModel):
    """A model of day units that would fail if it ever fitted."""

    _unit_kind = "day"

    def __init__(self, predictors=("load_d+{horizon}",), **kw):
        kw.setdefault("refit", "7D")
        super().__init__(list(predictors), 30, **kw)

    def _fit_predict(self, train_x, train_y, test_x):
        raise AssertionError("no fit")

    def _fit_days(self, x, y, y_mask):
        raise AssertionError("no fit")

    def _predict_days(self, fitted, rows):
        raise AssertionError("no fit")


class ReadsHistory(OLSModel):
    """A model that reads series (`_history_spec`), which a per-row target can't reach."""

    def _history_spec(self, target):
        return HistorySpec(7, (), (), target, True, self.newest_target)


@pytest.mark.parametrize("key, value", [("normalise_by", SCALE), ("bounds", (0, 1)), ("zero_when", "flag")])
def test_day_units_and_history_models_refuse_them_before_any_fit(spring, key, value):
    with pytest.raises(ValueError, match="day units"):
        DayUnits(**{key: value}).run(spring, "2023-04-10", "2023-04-10", horizon=1)
    with pytest.raises(ValueError, match="reads history"):
        ReadsHistory(PREDICTORS, training_window=30, **{key: value}).run(spring, "2023-04-10", "2023-04-10", horizon=1)


def test_a_residual_model_and_a_pooled_model_refuse_them(spring, tmp_path):
    learner = OLSModel(PREDICTORS, bounds=(0, None))
    with pytest.raises(ValueError, match="per-row target"):
        ResidualModel(OLSModel(PREDICTORS), learner)
    learner = OLSModel(PREDICTORS)
    residual = ResidualModel(OLSModel(PREDICTORS, bounds=(0, None)), learner)  # the base's own run applies its bounds
    with pytest.raises(ValueError, match="per-row target"):
        residual.bounds = (0, None)
    learner.zero_when = "flag"  # after the learner was given: refused at the run, before the base runs
    with pytest.raises(ValueError, match="per-row target"):
        residual.run(spring, "2023-04-10", "2023-04-10", horizon=1, save_to=str(tmp_path / "residual.jsonl"))
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="per-row target"):
        PooledModel(OLSModel(PREDICTORS, normalise_by=SCALE))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        (dict(normalise_by=3), TypeError),
        (dict(zero_when=" flag"), ValueError),
        (dict(normalise_by="cap_d+{h}"), ValueError),
        (dict(bounds=(1, 0)), ValueError),
        (dict(bounds=(0, float("inf"))), ValueError),
        (dict(bounds=(0, 1, 2)), ValueError),
        (dict(bounds="0,1"), ValueError),
        (dict(bounds=(True, 1)), ValueError),
    ],
)
def test_values_that_are_refused(kwargs, error):
    with pytest.raises(error):
        OLSModel(PREDICTORS, **kwargs)


def test_values_as_kept():
    model = OLSModel(PREDICTORS, bounds=[0, np.float64(1.05)])
    assert model.bounds == (0.0, 1.05) and all(type(side) is float for side in model.bounds)
    assert OLSModel(PREDICTORS, bounds=(None, None)).bounds is None
    assert OLSModel(PREDICTORS, bounds=(2, 2)).bounds == (2.0, 2.0)
    assert OLSModel(PREDICTORS, zero_when="night_d+{horizon}").zero_when == "night_d+{horizon}"


def test_with_all_three_none_a_record_and_a_manifest_are_unchanged(spring, tmp_path):
    """Spot check of what the step 2-5 equivalence grids show for every existing model, outside the fast suite: no key in a
    record, the manifest's model_params and predictors, or the worker's config."""
    D = "2023-04-10"
    model = OLSModel(PREDICTORS, training_window=30, refit="7D")
    model.run(spring, D, D, horizon=1, save_to=str(tmp_path / "ols.jsonl"))
    manifest = json.loads(manifest_path(tmp_path / "ols.jsonl").read_text())
    assert manifest["model_params"] == {"scaler": "standard", **model._model_kwargs}
    assert not {"cap_d+1", "flag"} & set(manifest["predictors"])
    record = json.loads(Path(tmp_path / "ols.jsonl").read_text().splitlines()[0])
    assert "detail" not in record and list(record)[-1] == "revision"


def test_a_torch_day_model_refuses_them_and_a_period_one_is_its_run_on_y_over_c(spring):
    """`TorchModel(output="day")` refuses them when it is built, and a period one turned into a day one (no torch needed); with
    the [torch] extra, `MLPHourlyModel` (period units) with `normalise_by` is its run on the frame whose target is y / c, times c,
    to the bit."""
    from epftoolbox2.models import MLPHourlyModel, TorchModel

    with pytest.raises(ValueError, match="takes no per-row target"):
        TorchModel("epftoolbox2.nets:MLP", predictors=["load_d+{horizon}"], output="day", bounds=(0, None))
    period = TorchModel("epftoolbox2.nets:MLP", predictors=["load_d+{horizon}"], output="period", zero_when="flag")
    with pytest.raises(ValueError, match="takes no per-row target"):
        period.output = "day"
    assert period.output == "period"
    pytest.importorskip("torch")
    D = "2023-04-10"

    def make(**kw):
        fast = dict(training_window=28, seeds=2, max_epochs=3, patience=2, lr=1e-2, batch_size=16, device="cpu")
        return MLPHourlyModel(PREDICTORS, hidden=[4], **fast, **kw)

    scaled = by_key(run(make(normalise_by=SCALE), spring, D, day_of(D, 1), horizon=1))
    on_ratio = by_key(run(make(), divided(spring), D, day_of(D, 1), horizon=1))
    expected = times_scale(spring, on_ratio)
    assert scaled.keys() == on_ratio.keys() and scaled
    for key, record in scaled.items():
        assert record["prediction"] == expected[key], key


def test_both_stores_read_back_a_forecast_of_0_with_its_detail(spring, tmp_path):
    from epftoolbox2.results.store import read_records

    records = {}
    for fmt in ("jsonl", "parquet"):
        pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=30, zero_when="flag", name="rt"))
        pipeline.run(spring, "2023-04-10", "2023-04-10", horizon=1, save_dir=str(tmp_path / fmt), save_format=fmt)
        records[fmt] = sorted(json.dumps(r, sort_keys=True) for r in read_records(tmp_path / fmt / f"rt.{fmt}"))
    assert records["jsonl"] == records["parquet"] and len(records["jsonl"]) == 24
    assert any('"detail": "zero_when"' in line for line in records["jsonl"])


# ---------------------------------------------------------------------------
# After the reviews: verify's flip probe, the issue-row judging of zero_when, the listing, the refusals
# ---------------------------------------------------------------------------


def test_a_late_flag_that_is_0_on_the_issue_day_is_a_leak():
    """The flag of day X is out at 23:00 on X, after the issue: a model reading it on the issue row leaks, whatever it holds.
    Noise only ever makes a flag "not 1", so a flag of 0 on D moved nothing under it; flipped (0 to 1 and back), it moves the
    forecasts."""
    data = market("spring")
    on_d = data.index.strftime("%Y-%m-%d") == "2023-04-12"
    data["flag"] = np.where(on_d, 0.0, data["flag"])
    late = InformationSet(rules={**RULES, "flag": Availability.published("23:00", 0)})
    report = verified(OLSModel(PREDICTORS, training_window=30, zero_when="flag"), data, late, rerun=False)
    row = report.table().iloc[0]
    assert row["status"] == "leak" and row["changed"] == 24, row.to_dict()


def test_verify_is_ok_with_a_static_zero_when_and_with_tight_bounds_and_finds_a_leak_through_them():
    """A forecast of 0 by a static `night_d+{horizon}` reads its flag alone, which no control can perturb: it needn't move. A
    model with tight `bounds` is probed without them, so a control's noise clipped back onto the forecast is no blind spot.
    A late predictor is still a leak with `bounds` set."""
    data = market("spring")
    for model in (
        OLSModel(PREDICTORS, training_window=30, zero_when="night_d+{horizon}"),
        OLSModel(PREDICTORS, training_window=30, bounds=(None, 60.0)),
    ):
        report = verified(model, data)
        report.assert_ok()
        assert report.table().iloc[0]["control moved"] > 0
    frame = data.assign(load_actual=data["load"])
    rules = InformationSet(rules={**RULES, "load_actual": Availability.realised("1h")})
    model = OLSModel([*PREDICTORS, "load_actual"], training_window=30, bounds=(None, 60.0))
    assert verified(model, frame, rules, positive_control=False, rerun=False).table().iloc[0]["status"] == "leak"


@pytest.mark.parametrize("predictors", [PREDICTORS, listed], ids=["fixed", "function"])
def test_zero_when_is_judged_on_the_issue_row_alone(spring, predictors):
    """`flag` has no rule for 3 to 5 April, inside the training window of a forecast on 10 April, which no fit reads: the run goes
    on at "raise". The same gap in `normalise_by`'s column is refused (`test_a_scale_known_on_the_issue_row_..._over_time`)."""
    gap = [
        Availability.static().during(valid_until="2023-04-02"),
        Availability.static().during(valid_from="2023-04-06"),
    ]
    info = InformationSet(rules={**RULES, "flag": gap})
    D = "2023-04-10"
    model = OLSModel(predictors, training_window=30, zero_when="flag")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # nor a warning that the issue row alone was judged
        ref = model.run(spring, D, D, horizon=1, information_set=info, availability="raise")
    assert ref.produced == 24
    # On the issue day itself the gap refuses it.
    with pytest.raises(InformationSetError, match="flag"):
        model.run(spring, "2023-04-04", "2023-04-04", horizon=1, information_set=info, availability="raise")


@pytest.mark.parametrize(
    "predictors", [[*PREDICTORS, SCALE], lambda ctx: [*PREDICTORS, SCALE]], ids=["fixed", "function"]
)
def test_a_column_both_a_predictor_and_the_scale_is_listed_once(spring, predictors):
    late = InformationSet(rules={**RULES, "cap": Availability.published("23:00", 1)})
    model = OLSModel(predictors, training_window=30, normalise_by=SCALE)
    D = "2023-04-10"
    ref = model.run(spring, D, D, horizon=1, information_set=late, availability="warn", warn_listing=False)
    assert [row["predictor"] for row in ref.availability_listing] == ["cap_d+1"]


def test_a_model_that_fits_nothing_takes_bounds_alone(spring):
    class Echo(OLSModel):
        _fits_nothing = True

    D = "2023-04-10"
    for key, value in (("normalise_by", SCALE), ("zero_when", "flag")):
        with pytest.raises(ValueError, match="fits nothing"):
            Echo(["price_d-1"], training_window=5, **{key: value}).run(spring, D, D, horizon=1)


def test_a_zero_when_value_other_than_0_or_1_on_an_issue_row_is_refused(spring):
    D = "2023-04-10"
    data = spring.copy()
    stamps = data.index.strftime("%Y-%m-%d %H")
    data.loc[stamps == "2023-04-05 03", "flag"] = 0.5  # a training row's: never read, never refused
    data.loc[stamps == f"{D} 04", "flag"] = np.nan  # missing: no forecast
    ref = OLSModel(PREDICTORS, training_window=30, zero_when="flag").run(data, D, D, horizon=1)
    assert ref.skipped == {"missing_test_predictors": 1}
    data.loc[stamps == f"{D} 05", "flag"] = 2.0
    with pytest.raises(ValueError, match=r"zero_when='flag' holds 2 on an issue row"):
        OLSModel(PREDICTORS, training_window=30, zero_when="flag").run(data, D, D, horizon=1)
    # A day of the run that isn't an issue day (Monday the 10th, issued on Tuesdays) isn't read.
    model = OLSModel(PREDICTORS, training_window=30, zero_when="flag")
    assert model.run(data, D, day_of(D, 1), horizon=1, issue_days=["tuesday"]).produced > 0


def weekend_matching(ctx):
    if ctx.target_day.weekday() >= 5:
        return Use(PREDICTORS, train_on="matching")
    return PREDICTORS


def test_a_matching_list_with_normalise_by_is_its_run_on_y_over_c(spring):
    D = "2023-04-13"  # a Thursday: horizon 2 reaches Saturday, a matching list
    scaled = OLSModel(weekend_matching, training_window=40, normalise_by=SCALE).run(spring, D, day_of(D, 1), horizon=2)
    on_ratio = OLSModel(weekend_matching, training_window=40).run(divided(spring), D, day_of(D, 1), horizon=2)
    expected = times_scale(spring, by_key(on_ratio._results))
    found = by_key(scaled._results)
    assert found.keys() == expected.keys() and any("matching_days" in r for r in found.values())
    for key, record in found.items():
        assert record["prediction"] == expected[key], key
        assert record.get("matching_days") == by_key(on_ratio._results)[key].get("matching_days"), key


# The registry's models that don't take all three, and why. A model added to the registry takes `normalise_by`, `bounds` and
# `zero_when` in its signature and its TUI schema, or is listed here with its reason.
ROW_TARGET_EXCEPTIONS = {
    "NaiveModel": ({"bounds"}, "it fits nothing: no target to divide, no fit to replace"),
    "MLPDailyModel": (set(), "day units: one fit of every period of a date"),
    "ResidualModel": (set(), "its forecast is the learner's plus the base's, made after the learner's fit"),
    "PooledModel": (set(), "a pooled fit reads each zone's rows itself"),
}


def accepted_parameters(cls):
    """The keyword parameters `cls(...)` takes: its `__init__`'s, and through `**kwargs` those of the next class's that names
    them, as far as one without `**kwargs`."""
    names = set()
    for klass in cls.__mro__:
        if "__init__" not in vars(klass) or klass is object:
            continue
        parameters = inspect.signature(klass.__init__).parameters
        names |= {name for name, p in parameters.items() if p.kind is not inspect.Parameter.VAR_KEYWORD}
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            break
    return names


def test_every_registered_model_takes_the_row_target_or_says_why_not():
    from epftoolbox2 import tui
    from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY

    three = {"normalise_by", "bounds", "zero_when"}
    for name, module in COMPONENT_REGISTRY["models"].items():
        cls = getattr(__import__(module, fromlist=[name]), name)
        expected, why = ROW_TARGET_EXCEPTIONS.get(name, (three, None))
        assert accepted_parameters(cls) & three == expected, (name, why)
        if name in tui.PARAM_SCHEMAS:
            assert {f["name"] for f in tui.PARAM_SCHEMAS[name]} & three == expected, (name, "TUI", why)
        else:
            assert name in ROW_TARGET_EXCEPTIONS, f"{name} isn't offered in the TUI: list it with its reason"
