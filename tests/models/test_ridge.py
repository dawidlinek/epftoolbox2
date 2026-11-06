"""`RidgeModel` (step 4c): Ridge regression whose penalty is chosen on every fit by leave-one-out (`RidgeCV(cv=None)`).

A record equals scikit-learn's `RidgeCV` fitted by hand on the training rows chosen by date and scaled by the library's rule,
written out here with numpy alone, at 1 h, 15 minutes and one row a day, over both clock changes, with and without an intercept
and with the binary columns scaled too, also from a refit's anchor over a leap day. The penalty itself is checked against a
leave-one-out loop of `Ridge` refits, so a K-fold or GCV choice can't pass for it. The same fit through `_fit_predict` and through
`_fit_model` + `_predict_fitted` is the same bit for bit, a run resumes and a pool of 1, 2 or 3 processes gives the same records,
`verify` passes with a positive control that can fail and calls a planted target-day input a leak.
"""

import datetime
import json
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import Ridge, RidgeCV
from threadpoolctl import threadpool_limits

import epftoolbox2.models as models_package
from epftoolbox2 import tui
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import BaseModel, LassoCVModel, OLSModel, RidgeModel
from epftoolbox2.models import ridge as ridge_module
from epftoolbox2.models._calendar import refit_anchor
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.models.ridge import DEFAULT_ALPHAS
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY
from epftoolbox2.results.manifest import ManifestMismatchError, manifest_path
from epftoolbox2.scalers import scaler_spec
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

GRID = np.logspace(-3, 4, 36)  # the grid `alphas=None` means, written out here and not imported
WINDOW = 40
ZONE = "Europe/Warsaw"
SWITCH_DAYS = {"spring": "2023-03-26", "autumn": "2023-10-29"}  # the clocks change at 02:00 Warsaw time
PER_HOUR = {"1h": 1, "15min": 4}
SCALE_BINARY = {"name": "standard", "params": {"scale_binary": True}}
HAND = 1e-12  # the hand fits' tolerance on a forecast in target units (prices of about 100)


class InProcessPool:
    """Stands in for the worker pool, whose start is most of the time these runs would take. The tests about the pool use the
    real one."""

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


def ridge(predictors=PREDICTORS, window=WINDOW, **kwargs):
    return RidgeModel(predictors, training_window=window, **kwargs)


def day_of(day, offset=0):
    return (pd.Timestamp(day) + pd.Timedelta(days=offset)).date().isoformat()


def market(switch, freq="1h", horizon=2, seed=3):
    """Seventy days before the switch and thirty after, hourly or every 15 minutes, with a price of noise on a linear model."""
    return make_market_data(
        start=day_of(SWITCH_DAYS[switch], -70), days=100, tz=ZONE, freq=freq, horizon=horizon, noise=5.0, seed=seed
    )


def daily_market(days=330, seed=5):
    """One row per local date at noon from 2023-01-02 on, over both clock changes: `demand` is linear in the drivers with a
    level of 900 and a flag that is not a calendar, and `_d+h` columns are built by date."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=days, freq="D")
    index = (dates + pd.Timedelta(hours=12)).tz_localize(ZONE)
    load = 1000 + 100 * np.sin(np.arange(days) / 9) + rng.normal(0, 20, days)
    wind = rng.gamma(2.0, 50.0, days)
    weekend = (dates.dayofweek >= 5).astype(float)
    flag = (rng.random(days) > 0.6).astype(float)
    frame = pd.DataFrame({"load": load, "wind": wind, "is_weekend": weekend, "flag": flag}, index=index)
    frame["demand"] = 900 + 0.5 * load - 0.8 * wind - 60 * weekend + 25 * flag + rng.normal(0, 15, days)
    for h in (1, 2, 3):
        for name in ("load", "wind", "is_weekend"):
            frame[f"{name}_d+{h}"] = frame[name].shift(-h)
    return frame


DAILY_PREDICTORS = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "flag"]


def by_key(records):
    return {(r["run_date"], r["hour"], r["horizon"]): r for r in records}


def keyed(records, *drop):
    """Key -> the record as JSON text without `drop`: equal text is equal to the bit, a NaN `actual` included."""
    return {
        (r["run_date"], r["hour"], r["horizon"]): json.dumps(
            {k: v for k, v in r.items() if k not in drop}, sort_keys=True
        )
        for r in records
    }


def lines(path):
    """A result file's records as sorted text: the order a pool finishes them in isn't fixed."""
    return sorted(line for line in Path(path).read_text().splitlines() if line.strip())


# ---------------------------------------------------------------------------
# The oracle: RidgeCV on the rows chosen by date, scaled by hand
# ---------------------------------------------------------------------------


def hand_forecast(
    data,
    freq,
    period,
    horizon,
    day,
    names,
    target="price",
    window=WINDOW,
    anchor=None,
    alphas=GRID,
    fit_intercept=True,
    scale_binary=False,
    estimator=None,
):
    """The forecast issued on `day` from the frame with numpy and scikit-learn alone: the rows of `period` dated from
    anchor-W-h to anchor-1-h (the anchor is the day itself without a refit; both rows of a repeated autumn hour train), the
    target of each at the same wall-clock time `horizon` dates later (none for a repeated hour), the columns that are 0/1 on
    those rows as they are (unless `scale_binary`) and every other column and the target to mean 0 and sample standard deviation
    1, `RidgeCV` on that, and the row of `day` predicted and scaled back. Returns the forecast, the coefficients (in the scaled
    space, as a record holds them) and the target's scale, which turns a difference in price units into one in scaled units."""
    wall = data.index.tz_localize(None) if data.index.tz is not None else data.index
    dates = wall.normalize()
    if freq == "1D":
        periods = np.zeros(len(wall), dtype=int)
    else:
        periods = wall.hour * PER_HOUR[freq] + wall.minute // (60 // PER_HOUR[freq])
    fit_day = pd.Timestamp(anchor or day)
    newest = fit_day - pd.Timedelta(days=1 + horizon)
    oldest = newest - pd.Timedelta(days=window - 1)
    here = np.asarray(periods == period)
    rows = np.flatnonzero(here & np.asarray((dates >= oldest) & (dates <= newest)))
    prices = pd.Series(data[target].to_numpy(), index=wall)
    once = prices[~prices.index.duplicated(keep=False)]
    y_all = once.reindex(wall[rows] + pd.Timedelta(days=horizon)).to_numpy()
    x = data[names].to_numpy(dtype=float)
    test_row = np.flatnonzero(here & np.asarray(dates == pd.Timestamp(day)))[-1]
    usable = np.isfinite(y_all) & np.isfinite(x[rows]).all(axis=1)
    y, train = y_all[usable], x[rows][usable]
    binary = np.array([set(np.unique(column)) <= {0.0, 1.0} for column in train.T]) & (not scale_binary)
    # Column by column, as a one-dimensional reduction sums: a sum over the rows of a matrix adds in another order and moves the
    # scaled values in their last bits, which a 1e-12 comparison of a fit made on them would see.
    means = np.array([0.0 if b else np.mean(column) for b, column in zip(binary, train.T)])
    deviations = np.array([1.0 if b else np.std(column, ddof=1) for b, column in zip(binary, train.T)])
    scaled_x, scaled_y = (train - means) / deviations, (y - np.mean(y)) / np.std(y, ddof=1)
    fit = (estimator or RidgeCV(alphas=alphas, fit_intercept=fit_intercept)).fit(scaled_x, scaled_y)
    row = ((x[test_row] - means) / deviations)[None, :]
    return float(fit.predict(row)[0] * np.std(y, ddof=1) + np.mean(y)), fit.coef_, float(np.std(y, ddof=1))


def run(model, data, start, end, horizon=2, freq="1h", target="price", save_to=None, **kwargs):
    ref = model.run(data, start, end, target=target, horizon=horizon, freq=freq, save_to=save_to, **kwargs)
    return ref._results if save_to is None else [json.loads(line) for line in open(save_to)]


def check_against_hand(records, data, freq, names, periods, horizons, target="price", **hand):
    checked = 0
    found = by_key(records)
    for (run_date, period, horizon), record in found.items():
        if period not in periods or horizon not in horizons:
            continue
        expanded = [name.replace("{horizon}", str(horizon)) for name in names]
        forecast, coefficients, _ = hand_forecast(
            data, freq, period, horizon, run_date, expanded, target=target, **hand
        )
        assert record["prediction"] == pytest.approx(forecast, abs=HAND, rel=0), (run_date, period, horizon)
        assert np.allclose(record["coefficients"], coefficients, atol=HAND, rtol=0), (run_date, period, horizon)
        checked += 1
    return checked


# ---------------------------------------------------------------------------
# The parameters
# ---------------------------------------------------------------------------


def test_the_default_grid_is_the_quality_study_s_and_is_what_a_worker_gets():
    model = RidgeModel(PREDICTORS)
    assert model.alphas is None and model.fit_intercept is True and model.training_window == 365
    assert model._model_kwargs == {"alphas": np.logspace(-3, 4, 36).tolist(), "fit_intercept": True}
    assert len(DEFAULT_ALPHAS) == 36 and DEFAULT_ALPHAS[0] == 1e-3 and DEFAULT_ALPHAS[-1] == 1e4
    # a worker rebuilds the model from `_model_kwargs`, so the grid it fits with is that one
    worker_copy = RidgeModel(predictors=[], **model._model_kwargs)
    assert worker_copy._estimator().alphas == model._model_kwargs["alphas"] and worker_copy._estimator().cv is None
    # the resolved grid is a copy: changing it can't change the default
    model._model_kwargs["alphas"].append(1.0)
    assert len(RidgeModel(PREDICTORS)._model_kwargs["alphas"]) == 36


def test_a_set_grid_and_intercept_reach_the_estimator_as_they_are():
    model = RidgeModel(PREDICTORS, 100, [0.5, 2, 8], fit_intercept=False)
    assert model.alphas == [0.5, 2.0, 8.0] and model.training_window == 100
    estimator = model._estimator()
    assert estimator.alphas == [0.5, 2.0, 8.0] and estimator.fit_intercept is False and estimator.cv is None
    for grid in (np.array([1, 10]), (2, 3), range(1, 4), np.logspace(-1, 1, 3)):
        stored = RidgeModel(PREDICTORS, alphas=grid).alphas
        assert type(stored) is list and all(type(a) is float for a in stored)
        assert stored == [float(a) for a in grid]
    model.alphas = (1, 2)  # checked and normalised on assignment too: the worker reads it through `_model_kwargs`
    assert model.alphas == [1.0, 2.0] and model._model_kwargs["alphas"] == [1.0, 2.0]
    with pytest.raises(TypeError):
        RidgeModel(PREDICTORS, 365, None, False)  # fit_intercept is keyword-only


BAD_GRIDS = {
    "empty_list": lambda: [],
    "empty_tuple": lambda: (),
    "empty_array": lambda: np.array([]),
    "zero": lambda: [0.0],
    "negative": lambda: [-1.0],
    "one_negative": lambda: [1.0, -1.0],
    "nan": lambda: [np.nan],
    "inf": lambda: [np.inf],
    "int": lambda: 3,
    "float": lambda: 3.0,
    "text": lambda: "abc",
    "number_as_text": lambda: "1.0",
    "bytes": lambda: b"1",
    "bool": lambda: [True],
    "one_bool": lambda: [1.0, False],
    "two_dimensional": lambda: [[1.0, 2.0]],
    "letter": lambda: [1.0, "a"],
    "dict": lambda: {1: 2},
    "iterator": lambda: iter([1.0]),
}


@pytest.mark.parametrize("make_bad", list(BAD_GRIDS.values()), ids=list(BAD_GRIDS))
def test_a_grid_that_is_not_positive_finite_penalties_raises_at_once(make_bad):
    with pytest.raises(ValueError, match="alphas must be None"):
        RidgeModel(PREDICTORS, alphas=make_bad())
    model = RidgeModel(PREDICTORS, alphas=[1.0])
    with pytest.raises(ValueError, match="alphas must be None"):
        model.alphas = make_bad()
    assert model.alphas == [1.0]


@pytest.mark.parametrize(
    "grid", [[1e-7], [1.0, 1e-9], (1e-12,), np.array([5e-7, 3.0])], ids=["1e-7", "mixed", "1e-12", "array"]
)
def test_a_penalty_below_1e_6_is_refused_and_points_to_ols(grid):
    """scikit-learn's leave-one-out route loses digits as 1/alpha (0.17 in a coefficient at 1e-12 on 247 correlated columns), so a
    grid with such a penalty raises at once; 1e-6 itself and the default grid's 1e-3 are accepted."""
    with pytest.raises(ValueError, match=r"alphas below 1e-06 are refused.*Use OLSModel for an unpenalised fit"):
        RidgeModel(PREDICTORS, alphas=grid)
    model = RidgeModel(PREDICTORS, alphas=[1e-6, 1e-3])
    with pytest.raises(ValueError, match="Use OLSModel"):
        model.alphas = grid
    assert model.alphas == [1e-6, 1e-3]
    assert RidgeModel(PREDICTORS, alphas=[1e-6]).alphas == [1e-6] and min(DEFAULT_ALPHAS) == 1e-3


def test_the_registry_the_package_and_the_tui_know_it():
    assert COMPONENT_REGISTRY["models"]["RidgeModel"] == "epftoolbox2.models.ridge"
    assert ridge_module.RidgeModel is RidgeModel
    assert "RidgeModel" in models_package.__all__
    assert "RidgeModel" in tui._MODELS and {f["name"] for f in tui.PARAM_SCHEMAS["RidgeModel"]} >= {
        "name",
        "predictors",
        "training_window",
        "alphas",
        "scaler",
        "newest_target",
        "fit_intercept",
        "refit",
    }
    made = tui._instantiate("RidgeModel", {"predictors": ["load"], "alphas": [0.1, 1.0], "fit_intercept": False})
    assert isinstance(made, RidgeModel) and made.alphas == [0.1, 1.0] and made.fit_intercept is False


def test_one_class_defines_the_three_seam_methods_so_refit_works():
    assert RidgeModel._can_refit()
    assert {RidgeModel._owner_of(name) for name in ("_fit_predict", "_fit_model", "_predict_fitted")} == {RidgeModel}
    assert RidgeModel(PREDICTORS, refit="7D").refit == "7D"

    class OtherPenalty(RidgeModel):
        def _fit_predict(self, train_x, train_y, test_x):
            return 0.0, []

    with pytest.raises(ValueError, match="can't refit"):
        OtherPenalty(PREDICTORS, refit="7D")


# ---------------------------------------------------------------------------
# YAML, manifest
# ---------------------------------------------------------------------------


def test_a_saved_pipeline_writes_alphas_and_the_intercept_only_when_set(tmp_path):
    def saved(*models):
        pipeline = ModelPipeline()
        for model in models:
            pipeline.add_model(model)
        return pipeline.to_dict()["models"]

    plain, explicit = (
        saved(RidgeModel(PREDICTORS, name="r")),
        saved(RidgeModel(PREDICTORS, name="r", alphas=None, fit_intercept=True, scaler="standard")),
    )
    assert plain == explicit
    assert plain[0]["class"] == "RidgeModel"
    assert "alphas" not in plain[0]["params"] and "fit_intercept" not in plain[0]["params"]

    path = tmp_path / "pipeline.yaml"
    pipeline = ModelPipeline().add_model(RidgeModel(PREDICTORS, name="plain"))
    pipeline.add_model(
        RidgeModel(
            PREDICTORS, 200, [0.1, 1e-05, 1234.5678901234567], fit_intercept=False, scaler=SCALE_BINARY, name="set"
        )
    )
    pipeline.add_model(RidgeModel(PREDICTORS, name="full", alphas=DEFAULT_ALPHAS, refit="7D", newest_target="d-1"))
    pipeline.save(path)
    text = yaml.safe_load(path.read_text())["models"]
    assert "alphas" not in text[0]["params"]
    assert (
        text[1]["params"]["alphas"] == [0.1, 1e-05, 1234.5678901234567] and text[1]["params"]["fit_intercept"] is False
    )
    assert text[2]["params"]["alphas"] == DEFAULT_ALPHAS and text[2]["params"]["refit"] == "7D"

    plain_back, set_back, full_back = ModelPipeline.load(path).models
    assert isinstance(plain_back, RidgeModel) and plain_back.alphas is None and plain_back.fit_intercept is True
    assert plain_back._model_kwargs == RidgeModel(PREDICTORS)._model_kwargs
    assert set_back.alphas == [0.1, 1e-05, 1234.5678901234567] and set_back.fit_intercept is False
    assert set_back.training_window == 200 and set_back.scaler == SCALE_BINARY and set_back.name == "set"
    assert set_back._model_kwargs == {"alphas": [0.1, 1e-05, 1234.5678901234567], "fit_intercept": False}
    assert full_back.alphas == DEFAULT_ALPHAS and full_back.refit == "7D"
    assert full_back._model_kwargs == plain_back._model_kwargs  # the grid written out is the default one
    # and a second save of what was loaded is the same file
    again = tmp_path / "again.yaml"
    ModelPipeline.load(path).save(again)
    assert again.read_text() == path.read_text()


def test_the_manifest_records_the_resolved_grid_and_a_resume_with_another_one_is_refused(tmp_path):
    data = market("spring")
    path = tmp_path / "ridge.jsonl"
    run(ridge(), data, "2023-03-25", "2023-03-26", save_to=str(path))
    stored = lines(path)
    assert len(stored) == 2 * 24 * 2 - 3  # the skipped hour of 26 March, as an issue hour (twice) and as a target hour
    manifest = json.loads(manifest_path(path).read_text())
    assert manifest["model_params"] == {
        "scaler": "standard",
        "alphas": np.logspace(-3, 4, 36).tolist(),
        "fit_intercept": True,
    }
    # the grid written out, and the model with the default, are one configuration: the run resumes
    run(ridge(alphas=list(GRID)), data, "2023-03-25", "2023-03-26", save_to=str(path))
    assert lines(path) == stored
    for other in (
        {"alphas": GRID[:-1].tolist()},
        {"alphas": np.logspace(-3, 4, 37).tolist()},
        {"fit_intercept": False},
    ):
        with pytest.raises(ManifestMismatchError, match="model_params"):
            run(ridge(**other), data, "2023-03-25", "2023-03-26", save_to=str(path))


def test_ridge_leaves_the_other_models_manifests_and_yaml_alone(tmp_path):
    """A new class changes nothing of OLS and LassoCV: their manifests have no `alphas` and their YAML no new key."""
    data = market("spring")
    for model in (OLSModel(PREDICTORS, WINDOW), LassoCVModel(PREDICTORS, WINDOW, cv=3)):
        path = tmp_path / f"{type(model).__name__}.jsonl"
        run(model, data, "2023-03-25", "2023-03-25", horizon=1, save_to=str(path))
        assert "alphas" not in json.loads(manifest_path(path).read_text())["model_params"]
    assert "alphas" not in ModelPipeline().add_model(OLSModel(PREDICTORS)).to_dict()["models"][0]["params"]


# ---------------------------------------------------------------------------
# What a forecast is
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("switch", list(SWITCH_DAYS))
@pytest.mark.parametrize("freq", list(PER_HOUR))
def test_forecasts_equal_ridgecv_on_hand_scaled_rows_over_a_clock_change(switch, freq):
    """Rows chosen by date and scaled by hand, `RidgeCV` fitted on them: every period of the hourly run and the switch hour
    and a few others of the 15-minute one, on the switch day and the three after it. In autumn both rows of the repeated hour
    train, and the target of a repeated hour is missing."""
    horizon = 2
    data = market(switch, freq=freq, horizon=horizon)
    start = SWITCH_DAYS[switch]
    records = run(ridge(PREDICTORS + ["price_d-1"]), data, start, day_of(start, 3), horizon, freq)
    per_hour = PER_HOUR[freq]
    periods = set(range(24 * per_hour)) if freq == "1h" else {0, 1, 7, 8, 9, 10, 11, 33, 95}
    checked = check_against_hand(records, data, freq, PREDICTORS + ["price_d-1"], periods, range(1, horizon + 1))
    assert (
        checked >= (3 if switch == "spring" else 4) * horizon * len(periods) - 3 * horizon
    )  # nothing silently unchecked
    assert len(records) == len(by_key(records))


@pytest.mark.parametrize("fit_intercept", [True, False], ids=["intercept", "no_intercept"])
@pytest.mark.parametrize("binary_too", [False, True], ids=["mask", "scale_binary"])
@pytest.mark.parametrize("alphas", [None, [0.5, 5.0, 50.0, 500.0]], ids=["default_grid", "set_grid"])
def test_forecasts_on_one_row_a_day_equal_ridgecv_with_every_option(alphas, binary_too, fit_intercept):
    """The options through the worker's rebuilt model: a flag that is a 0/1 column the mask leaves alone (or `scale_binary`
    scales), a set grid and no intercept. The level of 900 is carried by the target's centring alone, so an intercept the
    penalty shrinks (a mutant that drops `fit_intercept`) moves these."""
    data = daily_market()
    model = ridge(
        DAILY_PREDICTORS,
        60,
        alphas=alphas,
        fit_intercept=fit_intercept,
        scaler=SCALE_BINARY if binary_too else "standard",
    )
    for start, end in (("2023-03-24", "2023-03-29"), ("2023-10-26", "2023-10-31")):  # both clock changes
        records = run(model, data, start, end, horizon=3, freq="1D", target="demand")
        assert len(records) == 6 * 3
        checked = check_against_hand(
            records, data, "1D", DAILY_PREDICTORS, {0}, {1, 2, 3}, target="demand", window=60,
            alphas=GRID if alphas is None else alphas, fit_intercept=fit_intercept, scale_binary=binary_too,
        )  # fmt: skip
        assert checked == 6 * 3


def test_each_option_moves_the_forecasts():
    """The hand fits above can't pass by luck: a dropped grid, intercept or scaling option changes the numbers."""
    data = daily_market()
    base = by_key(run(ridge(DAILY_PREDICTORS, 60), data, "2023-04-10", "2023-04-16", 3, "1D", "demand"))
    for options in ({"alphas": [1e4]}, {"fit_intercept": False}, {"scaler": SCALE_BINARY}):
        other = by_key(run(ridge(DAILY_PREDICTORS, 60, **options), data, "2023-04-10", "2023-04-16", 3, "1D", "demand"))
        moves = [abs(other[key]["prediction"] - base[key]["prediction"]) for key in base]
        assert max(moves) > 0.05, options


@pytest.mark.parametrize("fit_intercept", [True, False], ids=["intercept", "no_intercept"])
@pytest.mark.parametrize(("alpha", "tolerance"), [(1e-3, 1e-9), (3.7, 1e-10), (1e4, 1e-10)], ids=["1e-3", "3.7", "1e4"])
def test_a_one_element_grid_is_ridge_with_that_penalty(alpha, tolerance, fit_intercept):
    """With one penalty there is nothing to choose: the forecast is `Ridge(alpha)` on the scaled rows. The design has no exactly
    collinear pair (the weekday dummy would make one with the weekend's): with one, the two solvers part by 5e-11 in a
    coefficient at 1e-3, and the hand fits above, which compare with `RidgeCV` itself, hold to 1e-12. At the grid's smallest
    penalty scikit-learn's leave-one-out route is less exact than `Ridge` (it loses digits as 1/alpha: 1.5e-10 in these forecasts
    in standardised units; against an exact SVD ridge on 247 correlated columns 1.7e-10 at 1e-3, 1.7e-7 at 1e-6), so that one gets a
    tolerance of 1e-9."""
    data = market("spring", horizon=1)
    names = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "price_d-1"]
    model = ridge(names, alphas=[alpha], fit_intercept=fit_intercept)
    records = run(model, data, "2023-03-24", "2023-03-27", horizon=1)
    estimator = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    for (run_date, period, horizon), record in by_key(records).items():
        expanded = [name.replace("{horizon}", str(horizon)) for name in names]
        forecast, coefficients, scale = hand_forecast(
            data, "1h", period, horizon, run_date, expanded, estimator=estimator
        )
        # in the standardised space: the forecast before it is scaled back to price units, as the coefficients are
        assert abs(record["prediction"] - forecast) <= tolerance * scale, (run_date, period)
        assert np.allclose(record["coefficients"], coefficients, atol=tolerance, rtol=0)
    # 4 days of 24 hours less 3: the skipped 02:00 of 26 March as an issue hour and as a target hour, and the 02:00 of the 27th,
    # whose `price_d-1` is that skipped hour's
    assert len(records) == 4 * 24 - 3


def leave_one_out(x, y, alpha, fit_intercept):
    """The leave-one-out mean squared error of `Ridge(alpha)`: n refits on n-1 rows, each predicting the row left out."""
    errors = []
    for left_out in range(len(y)):
        keep = np.arange(len(y)) != left_out
        fit = Ridge(alpha=alpha, fit_intercept=fit_intercept).fit(x[keep], y[keep])
        errors.append((y[left_out] - fit.predict(x[left_out : left_out + 1])[0]) ** 2)
    return float(np.mean(errors))


@pytest.mark.parametrize("fit_intercept", [True, False], ids=["intercept", "no_intercept"])
@pytest.mark.parametrize(("rows", "columns"), [(40, 5), (25, 30)], ids=["more_rows", "more_columns"])
def test_the_penalty_is_the_leave_one_out_minimiser_of_the_grid(rows, columns, fit_intercept):
    """The chosen penalty is the grid's first minimiser of a leave-one-out loop of `Ridge` refits written here, not a K-fold or
    generalised cross-validation choice."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(rows, columns))
    x = (x - x.mean(0)) / x.std(0, ddof=1)
    y = x[:, :3] @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 1, rows)
    y = (y - y.mean()) / y.std(ddof=1)
    errors = np.array([leave_one_out(x, y, alpha, fit_intercept) for alpha in GRID])
    fitted = ridge(fit_intercept=fit_intercept)._fit_model(x, y)
    assert fitted.alpha_ == GRID[int(np.argmin(errors))]
    assert fitted.cv is None


@pytest.mark.parametrize("fit_intercept", [True, False], ids=["intercept", "no_intercept"])
@pytest.mark.parametrize(
    ("rows", "columns"), [(30, 4), (4, 9), (3, 3)], ids=["more_rows", "more_columns", "three_rows"]
)
def test_fit_predict_is_predict_fitted_of_fit_model_to_the_bit(rows, columns, fit_intercept):
    rng = np.random.default_rng(rows * 100 + columns)
    x, y, row = rng.normal(size=(rows, columns)), rng.normal(size=rows), rng.normal(size=(1, columns))
    model = ridge(alphas=[0.3, 3.0, 30.0], fit_intercept=fit_intercept)
    one_step = model._fit_predict(x, y, row)
    two_steps = model._predict_fitted(model._fit_model(x, y), row)
    assert one_step == two_steps  # prediction and coefficients, equal as floats: the bit
    assert np.float64(one_step[0]).tobytes() == np.float64(two_steps[0]).tobytes()
    assert isinstance(one_step[1], list) and len(one_step[1]) == columns
    again = model._predict_fitted(model._fit_model(x, y), row)
    assert again == one_step  # and a fit repeated is the same fit


@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0")  # the scaler of a one-row window
def test_fewer_than_three_training_rows_is_a_counted_skip_not_a_forecast():
    """With one or two rows every leave-one-out error is 0/0: scikit-learn warns, may keep its first penalty and returns a
    forecast like any other. Here the fit raises, which the worker counts as `fit_error`; with three rows it fits."""
    for rows in (0, 1, 2):
        with pytest.raises(ValueError, match="at least 3 training rows"):
            ridge()._fit_model(np.ones((rows, 3)), np.ones(rows))
    rng = np.random.default_rng(0)
    assert ridge()._fit_model(rng.normal(size=(3, 3)), rng.normal(size=3)).alpha_ in GRID
    data = market("spring", horizon=1)
    with pytest.raises(
        RuntimeError, match=r"no forecasts produced \(48 planned\)\. Skipped: fit_error x48 .*at least 3 training rows"
    ):
        ridge(window=1).run(data, "2023-03-20", "2023-03-21", target="price", horizon=1)  # one row a period
    with pytest.raises(RuntimeError, match="fit_error x48"):
        ridge(window=2).run(data, "2023-03-20", "2023-03-21", target="price", horizon=1)
    three = ridge(window=3).run(data, "2023-03-20", "2023-03-21", target="price", horizon=1)
    assert len(three._results) == 2 * 24 and not three.skipped


# ---------------------------------------------------------------------------
# With refit=
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("refit", "issues", "anchor"),
    [
        ("7D", ["2024-02-28", "2024-02-29", "2024-03-01", "2024-03-03"], "2024-02-26"),
        ("1MS", ["2024-02-05", "2024-02-28", "2024-02-29"], "2024-02-01"),
        ("1MS", ["2024-03-02", "2024-03-04"], "2024-03-01"),
    ],
    ids=["7D-over-the-leap-day", "1MS-to-the-leap-day", "1MS-after-it"],
)
def test_a_refit_forecast_is_the_hand_fit_at_its_anchor(refit, issues, anchor):
    """The forecasts of a block are one `RidgeCV` fit on the anchor's rows, each predicted from its own row. The runs start after
    the anchor, so nothing but the anchor can put the cut there."""
    data = make_market_data(start="2023-12-01", days=105, horizon=2, noise=5.0, seed=11)
    names = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]
    records = run(ridge(names, refit=refit), data, issues[0], issues[-1])
    found = by_key(records)
    assert len({key[0] for key in found}) == (pd.Timestamp(issues[-1]) - pd.Timestamp(issues[0])).days + 1
    for issue in issues:
        for period in (0, 7, 18):
            for horizon in (1, 2):
                record = found[(issue, period, horizon)]
                assert record["refit_anchor"] == anchor
                assert refit_anchor(datetime.date.fromisoformat(issue), refit).isoformat() == anchor
                expanded = [name.replace("{horizon}", str(horizon)) for name in names]
                forecast, coefficients, _ = hand_forecast(data, "1h", period, horizon, issue, expanded, anchor=anchor)
                assert record["prediction"] == pytest.approx(forecast, abs=1e-9, rel=0)
                assert np.allclose(record["coefficients"], coefficients, atol=1e-9, rtol=0)


def test_refitting_every_day_is_not_refitting_and_a_block_shares_one_fit():
    data = make_market_data(start="2023-01-01", days=90, horizon=2, noise=5.0, seed=3)
    plain = by_key(run(ridge(), data, "2023-03-20", "2023-03-31"))
    daily = by_key(run(ridge(refit="1D"), data, "2023-03-20", "2023-03-31"))
    assert {k: r["prediction"] for k, r in plain.items()} == {k: r["prediction"] for k, r in daily.items()}
    assert {k: r["coefficients"] for k, r in plain.items()} == {k: r["coefficients"] for k, r in daily.items()}
    weekly = by_key(run(ridge(refit="7D"), data, "2023-03-20", "2023-03-26"))
    for horizon in (1, 2):
        # one fit, so one coefficient vector for the seven forecasts of a period and horizon
        assert len({json.dumps(weekly[(f"2023-03-{d}", 5, horizon)]["coefficients"]) for d in range(20, 27)}) == 1


# ---------------------------------------------------------------------------
# Determinism, resume, the pool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("refit", [None, "7D"])
def test_one_two_or_three_processes_and_any_order_give_the_same_records(refit, monkeypatch):
    data = market("spring", horizon=2)

    def records():
        return keyed(run(ridge(refit=refit), data, "2023-03-22", "2023-03-28"))

    expected = records()
    assert len(expected) == 7 * 24 * 2 - 4  # the skipped hour of 26 March, as an issue hour and as a target hour
    original = BaseModel._run_tasks

    def reversed_tasks(self, tasks, *args, **kwargs):
        return original(self, list(reversed(tasks)), *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(BaseModel, "_run_tasks", reversed_tasks)
        assert records() == expected
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", ProcessPoolExecutor)
    for processes in ("1", "2", "3"):
        monkeypatch.setenv("MAX_PROCESSES", processes)
        assert records() == expected, processes
    monkeypatch.setenv("MAX_PROCESSES", "2")
    monkeypatch.setenv("THREADS_PER_PROCESS", "3")
    assert records() == expected


@pytest.mark.parametrize("refit", [None, "7D"])
def test_a_run_cut_short_resumes_to_a_fresh_one(refit, tmp_path):
    data = market("autumn", horizon=2)
    fresh, cut = tmp_path / "fresh" / "m.jsonl", tmp_path / "cut" / "m.jsonl"
    run(ridge(refit=refit), data, "2023-10-26", "2023-11-01", save_to=str(fresh))
    run(ridge(refit=refit), data, "2023-10-26", "2023-11-01", save_to=str(cut))
    stored = lines(cut)
    keep = np.random.default_rng(2).random(len(stored)) < 0.6
    cut.write_text("\n".join(line for line, kept in zip(stored, keep) if kept) + "\n")
    assert 0 < keep.sum() < len(stored)
    calls = []
    original = BaseModel._run_tasks

    def counted(self, tasks, *args, **kwargs):
        calls.append(len(tasks))
        return original(self, tasks, *args, **kwargs)

    BaseModel._run_tasks = counted
    try:
        run(ridge(refit=refit), data, "2023-10-26", "2023-11-01", save_to=str(cut))
    finally:
        BaseModel._run_tasks = original
    assert calls == [len(stored) - int(keep.sum())]
    assert lines(cut) == lines(fresh)


# ---------------------------------------------------------------------------
# verify: the leakage boundary
# ---------------------------------------------------------------------------

RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
}
CLEAN = PREDICTORS + ["price_d-1"]
ISSUE_DAY = "2023-04-12"
TEST_PERIOD = ("2023-04-03", "2023-04-28")


@pytest.fixture(scope="module")
def verify_data():
    df = make_market_data(days=120, noise=5.0, seed=7, horizon=3)
    df["price_d-2"] = calendar_shift(df["price"], -2)
    return df


@pytest.fixture
def one_thread():
    with threadpool_limits(limits=1):
        yield


def verified(model, data, **kwargs):
    pipeline = ModelPipeline(information_set=InformationSet(rules=RULES))
    return pipeline.add_model(model).verify(data, *TEST_PERIOD, days=1, **kwargs)


def row_of(report):
    return report.table().iloc[0]


def test_verify_is_ok_with_a_control_that_moved(verify_data, one_thread):
    for refit in (None, "7D"):
        report = verified(ridge(CLEAN, 40, refit=refit, name="Ridge"), verify_data)
        assert report.ok
        report.assert_ok()
        verdict = row_of(report)
        assert (verdict["forecasts"], verdict["changed"], verdict["control moved"]) == (24, 0, 24)
        assert verdict["status"] == "ok" and report.problems() == []
        assert report.checks[0].control.max_abs_change > 1.0


class IgnoresItsInputs(RidgeModel):
    """Forecasts the mean of its training targets, whatever the forecast's own row holds (one class: `refit` stays None)."""

    def _fit_predict(self, train_x, train_y, test_x):
        return float(train_y.mean()), []


class ReadsIssueDayTarget(RidgeModel):
    """Its `load_d+1` column is really the price of the row's own day: the name is legal, the frame is untouched."""

    def _preprocess(self, data, horizon, target):
        out = super()._preprocess(data, horizon, target)
        out["load_d+1"] = out[target]
        return out


class TrainsOnIssueDayTarget(RidgeModel):
    """Its newest training row has the price of the issue day as its target, one day past the model's boundary."""

    def _preprocess(self, data, horizon, target):
        out = super()._preprocess(data, horizon, target)
        out[f"{target}_d+1"] = calendar_shift(out[target], 2)
        return out


def test_the_positive_control_can_fail_for_a_ridge_that_ignores_its_inputs(verify_data, one_thread):
    report = verified(IgnoresItsInputs(CLEAN, training_window=40, name="Blind"), verify_data)
    assert row_of(report)["changed"] == 0 and row_of(report)["control moved"] == 0
    assert row_of(report)["status"] == "inconclusive" and not report.ok
    # Without the control the same model is called clean, which is why the control exists.
    assert verified(IgnoresItsInputs(CLEAN, training_window=40), verify_data, positive_control=False).ok


@pytest.mark.parametrize("model_class", [ReadsIssueDayTarget, TrainsOnIssueDayTarget])
def test_a_planted_target_day_input_is_a_leak(model_class, verify_data, one_thread):
    report = verified(model_class(CLEAN, training_window=40, name="Sneaky"), verify_data, positive_control=False)
    assert not report.ok
    assert row_of(report)["status"] == "leak" and row_of(report)["changed"] == 24


def test_a_late_predictor_is_a_leak_for_the_periods_it_is_late(verify_data, one_thread):
    frame = verify_data.assign(load_actual=verify_data["load"])
    rules = {**RULES, "load_actual": Availability.realised("1h")}
    pipeline = ModelPipeline(information_set=InformationSet(rules=rules))
    pipeline.add_model(ridge(PREDICTORS + ["load_actual"], 40, name="Late"))
    report = pipeline.verify(frame, ISSUE_DAY, ISSUE_DAY, days=1)
    assert row_of(report)["status"] == "leak"
    assert sorted({period for period, _ in report.checks[0].probe.changed}) == list(range(10, 24))


# ---------------------------------------------------------------------------
# Next to the other models
# ---------------------------------------------------------------------------


def test_ridge_with_a_tiny_penalty_approaches_the_ols_it_regularises():
    """On a well-posed design (no collinear pair) a penalty of 1e-6 on standardised columns is OLS up to the shrinkage it still
    has: a sanity check on scaling and rows against the library's own OLS, with an intercept. (Not smaller: scikit-learn's
    leave-one-out route loses digits in proportion to 1/alpha, which is why `RidgeModel` refuses penalties below it.)"""
    data = daily_market()
    names = ["load_d+{horizon}", "wind_d+{horizon}", "flag"]
    small = by_key(run(ridge(names, 60, alphas=[1e-6]), data, "2023-04-10", "2023-04-16", 3, "1D", "demand"))
    ols = by_key(run(OLSModel(names, 60, fit_intercept=True), data, "2023-04-10", "2023-04-16", 3, "1D", "demand"))
    assert small.keys() == ols.keys() and len(small) == 7 * 3
    for key, record in small.items():
        assert record["prediction"] == pytest.approx(ols[key]["prediction"], abs=1e-4, rel=0), key


def test_the_default_scaler_is_the_standard_one():
    assert RidgeModel(PREDICTORS).scaler == "standard" and scaler_spec(RidgeModel(PREDICTORS).scaler) == "standard"
