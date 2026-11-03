"""The estimator options of step 3i: `LassoCVModel(fit_intercept=)`, `OLSModel(fit_intercept=)` and `StandardScaler(scale_binary=)`.

Each is keyword-only and defaults to what the model did before, and a default leaves `_model_kwargs`, the scaler's spec, the
manifest and a saved file as they were, so no old result or file moves. A set option reaches the worker (which rebuilds the model
from `_model_kwargs`), and a forecast equals scikit-learn's fit on the rows chosen by date, scaled by the library's rule, written out
here by hand: with `refit=None` and on a refit's anchor.
"""

import datetime
import json
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LassoCV, LinearRegression

from epftoolbox2.models import LassoCVModel, OLSModel
from epftoolbox2.models._calendar import refit_anchor
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import manifest_path
from epftoolbox2.scalers import StandardScaler, scaler_spec
from epftoolbox2.scalers.base import mean_std

ZONE = "Europe/Warsaw"
WINDOW = 60
# A level the dummies can't carry: a weekend dummy without its complement, and a 0/1 flag that isn't a calendar.
PREDICTORS = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "flag"]
SCALE_BINARY = {"name": "standard", "params": {"scale_binary": True}}
PROVENANCE = ("package_version", "git_sha", "created_at", "updated_at")


class InProcessPool:
    """Stands in for the worker pool. It still rebuilds the model from `_model_kwargs` (`_worker_init`), as a worker does."""

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


def daily_frame(days=160, seed=3):
    """One row per local date at noon: the target a linear function of the drivers with a level of 900, and `_d+h` columns
    built by date."""
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


def run(model, frame, start="2023-04-10", end="2023-04-23", horizon=3, save_to=None):
    ref = model.run(frame, start, end, target="demand", horizon=horizon, freq="1D", save_to=save_to)
    return ref._results if save_to is None else [json.loads(line) for line in open(save_to)]


def by_key(records):
    return {(r["run_date"], r["horizon"]): r for r in records}


def rows_of(frame, issue_day, horizon, window=WINDOW):
    """The rows a forecast issued on `issue_day` at `horizon` trains on, by date ([D - W - h, D - 1 - h]), their targets (the
    target of each row's date + h), and the issue row, from the frame alone."""
    local = frame.index.tz_localize(None)
    dates = local.normalize()
    day = pd.Timestamp(issue_day)
    names = [name.replace("{horizon}", str(horizon)) for name in PREDICTORS]
    train = (dates >= day - pd.Timedelta(days=window + horizon)) & (dates <= day - pd.Timedelta(days=1 + horizon))
    # By the wall clock: noon h dates later, across the spring switch too.
    target = pd.Series(frame["demand"].to_numpy(), index=local).reindex(local + pd.Timedelta(days=horizon)).to_numpy()
    x, y = frame.loc[train, names].to_numpy(), target[train]
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
    return x[keep], y[keep], frame.loc[dates == day, names].to_numpy()


def scaled(x, y, test, binary_too):
    """The library's standardisation, written out: each non-binary column (every column with `binary_too`) and the target to
    mean 0 and sample standard deviation 1 over the training rows."""
    x, test = x.copy(), test.copy()
    for i in range(x.shape[1]):
        binary = set(np.unique(x[:, i])) <= {0.0, 1.0}
        if binary and not binary_too:
            continue
        mean, std, _ = mean_std(x[:, i])
        x[:, i], test[:, i] = (x[:, i] - mean) / std, (test[:, i] - mean) / std
    y_mean, y_std, _ = mean_std(y)
    return x, (y - y_mean) / y_std, test, (y_mean, y_std)


def lasso_by_hand(frame, issue_day, horizon, fit_intercept, binary_too, cv=5, max_iter=10000):
    x, y, test = rows_of(frame, issue_day, horizon)
    x, y, test, (mean, std) = scaled(x, y, test, binary_too)
    fit = LassoCV(cv=cv, max_iter=max_iter, n_jobs=1, fit_intercept=fit_intercept).fit(x, y)
    return fit.predict(test)[0] * std + mean


# --- defaults move nothing ---------------------------------------------------------------------------------------------------


def test_the_defaults_leave_the_worker_kwargs_and_the_scaler_spec_as_they_were():
    assert LassoCVModel(PREDICTORS)._model_kwargs == {"cv": 5, "max_iter": 10000}
    assert LassoCVModel(PREDICTORS, fit_intercept=True)._model_kwargs == {"cv": 5, "max_iter": 10000}
    assert OLSModel(PREDICTORS)._model_kwargs == {}
    assert OLSModel(PREDICTORS, fit_intercept=False)._model_kwargs == {}
    assert StandardScaler().get_params() == {} and StandardScaler(scale_binary=False).get_params() == {}
    for spec in ("standard", StandardScaler(), StandardScaler(scale_binary=False), {"name": "standard", "params": {}}):
        assert scaler_spec(spec) == "standard"
    assert scaler_spec({"name": "standard", "params": {"scale_binary": False}}) == "standard"
    assert OLSModel(PREDICTORS, scaler=StandardScaler(scale_binary=False)).scaler == "standard"

    assert LassoCVModel(PREDICTORS, fit_intercept=False)._model_kwargs == {
        "cv": 5,
        "max_iter": 10000,
        "fit_intercept": False,
    }
    assert OLSModel(PREDICTORS, fit_intercept=True)._model_kwargs == {"fit_intercept": True}
    assert scaler_spec(StandardScaler(scale_binary=True)) == SCALE_BINARY == scaler_spec(SCALE_BINARY)


def test_the_options_are_keyword_only():
    with pytest.raises(TypeError):
        LassoCVModel(PREDICTORS, 365, 5, 10000, False)
    with pytest.raises(TypeError):
        OLSModel(PREDICTORS, 365, True)
    with pytest.raises(TypeError):
        StandardScaler(True)


def test_a_saved_pipeline_writes_an_option_only_when_set(tmp_path):
    def saved(*models):
        pipeline = ModelPipeline()
        for model in models:
            pipeline.add_model(model)
        return pipeline.to_dict()["models"]

    plain = saved(OLSModel(PREDICTORS, name="o"), LassoCVModel(PREDICTORS, name="l"))
    explicit = saved(
        OLSModel(PREDICTORS, name="o", fit_intercept=False, scaler=StandardScaler()),
        LassoCVModel(PREDICTORS, name="l", fit_intercept=True, scaler={"name": "standard", "params": {}}),
    )
    assert plain == explicit
    assert all("fit_intercept" not in model["params"] and model["params"]["scaler"] == "standard" for model in plain)

    path = tmp_path / "p.yaml"
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, name="o", fit_intercept=True))
    pipeline.add_model(LassoCVModel(PREDICTORS, name="l", cv=7, fit_intercept=False, scaler=SCALE_BINARY))
    pipeline.save(path)
    ols, lasso = ModelPipeline.load(path).models
    assert ols.fit_intercept is True and lasso.fit_intercept is False and lasso.scaler == SCALE_BINARY
    assert ols._model_kwargs == {"fit_intercept": True}
    assert lasso._model_kwargs == {"cv": 7, "max_iter": 10000, "fit_intercept": False}


def test_explicit_defaults_give_the_records_and_manifest_of_omitted_ones(tmp_path):
    frame = daily_frame()
    outputs = []
    for name, models in (
        ("omitted", (OLSModel(PREDICTORS, WINDOW, name="o"), LassoCVModel(PREDICTORS, WINDOW, name="l"))),
        (
            "explicit",
            (
                OLSModel(PREDICTORS, WINDOW, name="o", fit_intercept=False, scaler=StandardScaler(scale_binary=False)),
                LassoCVModel(PREDICTORS, WINDOW, name="l", fit_intercept=True, scaler=StandardScaler()),
            ),
        ),
    ):
        files = {}
        for model in models:
            path = tmp_path / name / f"{model.name}.jsonl"
            path.parent.mkdir(exist_ok=True)
            run(model, frame, save_to=str(path))
            manifest = json.loads(manifest_path(path).read_text())
            files[model.name] = (
                sorted(path.read_text().splitlines()),
                {k: v for k, v in manifest.items() if k not in PROVENANCE},
            )
        outputs.append(files)
    assert outputs[0] == outputs[1]
    assert outputs[0]["l"][1]["model_params"] == {"scaler": "standard", "cv": 5, "max_iter": 10000}
    assert outputs[0]["o"][1]["model_params"] == {"scaler": "standard"}


# --- what a set option does --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("binary_too", [False, True], ids=["mask", "scale_binary"])
@pytest.mark.parametrize("fit_intercept", [True, False], ids=["intercept", "no_intercept"])
def test_lasso_forecasts_are_scikit_learn_s_on_the_rows_by_date(fit_intercept, binary_too):
    """Each combination, through the worker's rebuilt model, equals LassoCV on the training rows scaled by hand."""
    frame = daily_frame()
    model = LassoCVModel(
        PREDICTORS, WINDOW, cv=4, fit_intercept=fit_intercept, scaler=SCALE_BINARY if binary_too else "standard"
    )
    found = by_key(run(model, frame))
    assert len(found) == 14 * 3
    for (run_date, horizon), record in found.items():
        expected = lasso_by_hand(frame, run_date, horizon, fit_intercept, binary_too, cv=4)
        assert record["prediction"] == pytest.approx(expected, abs=1e-9), (run_date, horizon)


def test_each_lasso_option_moves_the_forecasts():
    """The options change numbers, so a mutant that drops one (from `_model_kwargs`, the estimator or the scaler) can't pass the
    hand fits above by luck."""
    frame = daily_frame()
    plain = by_key(run(LassoCVModel(PREDICTORS, WINDOW, cv=4), frame))
    for options in ({"fit_intercept": False}, {"scaler": SCALE_BINARY}):
        other = by_key(run(LassoCVModel(PREDICTORS, WINDOW, cv=4, **options), frame))
        moves = [abs(other[key]["prediction"] - plain[key]["prediction"]) for key in plain]
        assert max(moves) > 0.01, options


def test_ols_with_an_intercept_is_scikit_learn_s_on_the_raw_rows():
    """An OLS with an intercept is invariant to the scaling of its columns, so it equals LinearRegression on the raw values (up to
    rounding); without one, a weekend dummy without its complement can't carry the level and the forecasts differ."""
    frame = daily_frame()
    with_intercept = by_key(run(OLSModel(PREDICTORS, WINDOW, fit_intercept=True), frame))
    without = by_key(run(OLSModel(PREDICTORS, WINDOW), frame))
    moves = []
    for (run_date, horizon), record in with_intercept.items():
        x, y, test = rows_of(frame, run_date, horizon)
        expected = LinearRegression().fit(x, y).predict(test)[0]
        assert record["prediction"] == pytest.approx(expected, abs=1e-7), (run_date, horizon)
        moves.append(abs(record["prediction"] - without[(run_date, horizon)]["prediction"]))
    assert max(moves) > 1.0


def test_scale_binary_standardises_the_binary_columns_and_ignores_the_mask():
    rng = np.random.default_rng(0)
    train = np.column_stack([rng.normal(5, 2, 40), (rng.random(40) > 0.5).astype(float)])
    test = np.array([[5.0, 1.0]])
    y = rng.normal(10, 3, 40)
    mask = StandardScaler.get_scalable_mask(train)
    assert mask.tolist() == [True, False]

    kept, _, kept_test = StandardScaler().fit_transform(train, y, test, scalable_mask=mask)
    assert np.array_equal(kept[:, 1], train[:, 1]) and kept_test[0, 1] == 1.0

    out, _, out_test = StandardScaler(scale_binary=True).fit_transform(train, y, test, scalable_mask=mask)
    mean, std, _ = mean_std(train[:, 1])
    assert np.allclose(out[:, 1], (train[:, 1] - mean) / std) and out_test[0, 1] == pytest.approx((1.0 - mean) / std)
    assert np.array_equal(out[:, 0], kept[:, 0])  # a non-binary column as before
    assert abs(out[:, 1].mean()) < 1e-12 and out[:, 1].std(ddof=1) == pytest.approx(1.0)


# --- with refit= -------------------------------------------------------------------------------------------------------------


def test_with_refit_a_forecast_is_the_options_fit_on_its_anchor():
    """`refit="7D"`: each forecast is the estimator with its options, fitted on the rows of a forecast issued on the anchor
    (scaled by its own rows, every column with scale_binary) and predicting the issue row."""
    frame = daily_frame()
    lasso = LassoCVModel(PREDICTORS, WINDOW, cv=4, fit_intercept=False, scaler=SCALE_BINARY, refit="7D")
    ols = OLSModel(PREDICTORS, WINDOW, fit_intercept=True, refit="7D")
    lasso_found, ols_found = by_key(run(lasso, frame)), by_key(run(ols, frame))
    assert {r["refit_anchor"] for r in lasso_found.values()} == {"2023-04-10", "2023-04-17"}
    for (run_date, horizon), record in lasso_found.items():
        anchor = refit_anchor(datetime.date.fromisoformat(run_date), "7D").isoformat()
        assert record["refit_anchor"] == anchor
        x, y, _ = rows_of(frame, anchor, horizon)
        _, _, test = rows_of(frame, run_date, horizon)
        sx, sy, stest, (mean, std) = scaled(x, y, test, binary_too=True)
        expected = LassoCV(cv=4, max_iter=10000, n_jobs=1, fit_intercept=False).fit(sx, sy).predict(stest)[0]
        assert record["prediction"] == pytest.approx(expected * std + mean, abs=1e-9), (run_date, horizon)

        expected = LinearRegression().fit(x, y).predict(test)[0]
        assert ols_found[(run_date, horizon)]["prediction"] == pytest.approx(expected, abs=1e-7), (run_date, horizon)


def test_refit_every_day_with_the_options_is_refit_none():
    frame = daily_frame()
    for make in (
        lambda **kw: LassoCVModel(PREDICTORS, WINDOW, cv=4, fit_intercept=False, scaler=SCALE_BINARY, **kw),
        lambda **kw: OLSModel(PREDICTORS, WINDOW, fit_intercept=True, scaler=SCALE_BINARY, **kw),
    ):
        none, daily = by_key(run(make(), frame)), by_key(run(make(refit="1D"), frame))
        assert {key: r["prediction"] for key, r in none.items()} == {key: r["prediction"] for key, r in daily.items()}
