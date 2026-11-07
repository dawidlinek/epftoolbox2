"""`LEARModel` (step 4d) on synthetic data: the API, `lago()`'s settings, YAML and the manifest, the fit through `_fit_stats`
and the kernels, hand fits with and without `refit`, a near-constant column, the probes, the pool and a resume.

The reproduction of the original epftoolbox on Lago's data is `test_lear_lago.py`; the kernels themselves are
`test_lars_kernels.py`; the members of a window list are held to their single-window models in `test_window_lists.py`.
"""

import json
import math
import warnings
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_limits

import epftoolbox2.models as models_package
from epftoolbox2 import tui
from epftoolbox2._extras import has_extra
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import BaseModel, LEARModel, OLSModel, RidgeModel
from epftoolbox2.models import _lars
from epftoolbox2.models import lear as lear_module
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.models.lear import LAGO_WINDOWS
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY
from epftoolbox2.presets import lear_predictors
from epftoolbox2.results.manifest import manifest_path
from epftoolbox2.scalers import make_scaler
from tests.models.test_lars_kernels import _lars_old_ic
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data

# `is_weekday` left out: with an intercept it is minus `is_weekend` after centring, an exact twin, at whose tie LARS takes 0.24.2's
# degenerate-entrant turn on a last bit, where the numpy and numba kernels can part (see `_lars`). Two inputs the price doesn't
# depend on, for the Lasso to drop.
BASE = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}"]
NAMES = BASE + ["price_d-1", "noise_a", "noise_b"]
WINDOW = 40
HAND = 1e-12  # a record against the same kernels composed by hand, in target units (prices of about 100)
PROVENANCE = ("package_version", "git_sha", "created_at", "updated_at", "kernels")


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


def with_noise(df, seed=11):
    rng = np.random.default_rng(seed)
    return df.assign(noise_a=rng.normal(0, 1, len(df)), noise_b=rng.gamma(2.0, 3.0, len(df)))


@pytest.fixture(scope="module")
def data():
    return with_noise(make_market_data(days=120, noise=5.0, seed=3, horizon=2))


def lear(names=NAMES, window=WINDOW, **kwargs):
    return LEARModel(names, training_window=window, **kwargs)


def run(model, frame, start, end, horizon=2, save_to=None, **kwargs):
    ref = model.run(frame, start, end, horizon=horizon, save_to=save_to, **kwargs)
    return ref._results if save_to is None else [json.loads(line) for line in open(save_to)]


def by_key(records):
    return {(r["run_date"], r["hour"], r["horizon"]): r for r in records}


def keyed(records, *drop):
    return {
        (r["run_date"], r["hour"], r["horizon"]): json.dumps(
            {k: v for k, v in r.items() if k not in drop}, sort_keys=True
        )
        for r in records
    }


def lines(path):
    return sorted(line for line in Path(path).read_text().splitlines() if line.strip())


def rows_of(frame, period, horizon, day, names, window=WINDOW, newest=-1, anchor=None, target="price"):
    """Training rows chosen by date as `training_span` does, written out: the rows of `period` dated from fit-day - W - h ... -
    1 - h with "d-1" (`newest` = -1), or fit-day - W ... - 1 with the newest target on the issue day (`newest` = 0, h = 1);
    their targets the price `horizon` dates later on the same wall clock; the row of `day` to forecast from."""
    names = [name.replace("{horizon}", str(horizon)) for name in names]
    wall = frame.index.tz_localize(None)
    dates = wall.normalize()
    here = np.asarray(wall.hour == period)
    fit_day = pd.Timestamp(anchor or day)
    last = fit_day - pd.Timedelta(days=1 + horizon) if newest == -1 else fit_day - pd.Timedelta(days=1)
    first = last - pd.Timedelta(days=window - 1)
    rows = np.flatnonzero(here & np.asarray((dates >= first) & (dates <= last)))
    prices = pd.Series(frame[target].to_numpy(), index=wall)
    once = prices[~prices.index.duplicated(keep=False)]
    y = once.reindex(wall[rows] + pd.Timedelta(days=horizon)).to_numpy()
    x = frame[names].to_numpy(dtype=float)
    keep = np.isfinite(y) & np.isfinite(x[rows]).all(axis=1)
    test = np.flatnonzero(here & np.asarray(dates == pd.Timestamp(day)))[-1]
    return x[rows][keep], y[keep], x[test : test + 1]


def hand_forecast(frame, period, horizon, day, names=NAMES, scaler="asinh", criterion="bic", solver="lars", **rows):
    """The forecast by hand: the rows above, the library's scaler fitted on them, `lear_fit` of their centred sums with the
    numpy kernels, `ybar + (x - xbar) @ w` on the scaled test row, scaled back."""
    train_x, train_y, test_x = rows_of(frame, period, horizon, day, names, **rows)
    scale = make_scaler(scaler)
    sx, sy, st = scale.fit_transform(train_x, train_y, test_x)
    fit = _lars.lear_fit(_lars.gram_stats(sx, sy), criterion, solver)
    return scale.inverse(float(fit.ybar + (st[0] - fit.xbar) @ fit.w)), fit, (sx, sy, st, scale)


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------


def test_the_defaults_are_d2_s_recommended_single_model():
    model = LEARModel(NAMES)
    assert (model.criterion, model.solver, model.scaler, model.name) == ("bic", "lars", "asinh", "LEAR")
    assert model.training_window == [56, 84, 728, "expanding"] and model._members == (56, 84, 728, "expanding")
    assert (model.newest_target, model.refit) == ("d-1", None)
    assert model._model_kwargs == {"criterion": "bic", "solver": "lars"}
    assert LEARModel(NAMES, 364, "aic", "cd")._model_kwargs == {"criterion": "aic", "solver": "cd"}


@pytest.mark.parametrize(
    "field, bad", [("criterion", "cv"), ("criterion", "AIC"), ("solver", "exact"), ("solver", None)]
)
def test_an_unknown_criterion_or_solver_raises_at_once(field, bad):
    with pytest.raises(ValueError, match=field):
        LEARModel(NAMES, **{field: bad})
    model = LEARModel(NAMES)
    with pytest.raises(ValueError, match=field):
        setattr(model, field, bad)


def test_lago_is_a_plain_lear_model_with_the_paper_s_settings():
    model = LEARModel.lago(["load_forecast_amprion", "wind_solar_forecast"])
    assert type(model) is LEARModel and model.name == "LEAR"
    assert model.predictors == lear_predictors(["load_forecast_amprion", "wind_solar_forecast"], order="lago")
    assert model.training_window == [49, 77, 1085, 1449] == list(LAGO_WINDOWS)
    assert (model.criterion, model.solver, model.newest_target, model.refit) == ("aic", "cd", "published", None)
    assert model.scaler == {"name": "asinh", "params": {"apply_to": "all", "standardise": False}}
    other = LEARModel.lago(["x"], price="p", calendar_prefix="cal_", name="Lago")
    assert (
        other.name == "Lago" and other.predictors[0] == "p_h00" and other.predictors[-1] == "cal_is_sunday_d+{horizon}"
    )
    assert len(model.predictors) == 247 and len(other.predictors) == 96 + 72 + 7


def test_the_registry_the_package_and_the_tui_know_it():
    assert COMPONENT_REGISTRY["models"]["LEARModel"] == "epftoolbox2.models.lear"
    assert lear_module.LEARModel is LEARModel and "LEARModel" in models_package.__all__
    assert "LEARModel" in tui._MODELS
    fields = {f["name"]: f for f in tui.PARAM_SCHEMAS["LEARModel"]}
    assert set(fields) == {
        "name",
        "predictors",
        "training_window",
        "criterion",
        "solver",
        "scaler",
        "newest_target",
        "refit",
    }
    assert fields["criterion"]["choices"] == ["bic", "aic"] and fields["solver"]["choices"] == ["lars", "cd"]
    made = tui._instantiate("LEARModel", {"predictors": ["load"], "criterion": "aic", "solver": "cd"})
    assert isinstance(made, LEARModel) and (made.criterion, made.solver) == ("aic", "cd")


def test_one_class_defines_the_three_seam_methods_so_refit_works():
    assert LEARModel._can_refit()
    assert {LEARModel._owner_of(name) for name in ("_fit_predict", "_fit_model", "_predict_fitted")} == {LEARModel}
    assert lear(refit="7D").refit == "7D"


# ---------------------------------------------------------------------------
# YAML, manifest
# ---------------------------------------------------------------------------


def test_lago_saves_as_its_settings_and_loads_back_equal(tmp_path):
    exogenous = ["generation_forecast_fr", "load_forecast_fr"]
    path = tmp_path / "pipeline.yaml"
    pipeline = ModelPipeline(
        information_set=InformationSet(
            "11:45", "Europe/Brussels", rules={"price": Availability.published("13:00", 1, timezone="Etc/GMT-1")}
        )
    )
    pipeline.add_model(LEARModel.lago(exogenous)).add_model(LEARModel(NAMES, name="Default"))
    pipeline.save(path)
    text = path.read_text()
    assert "lago" not in text.lower()
    saved = {m["params"]["name"]: m["params"] for m in pipeline.to_dict()["models"]}
    assert saved["LEAR"] == {
        "predictors": lear_predictors(exogenous, order="lago"),
        "training_window": [49, 77, 1085, 1449],
        "name": "LEAR",
        "scaler": {"name": "asinh", "params": {"apply_to": "all", "standardise": False}},
        "newest_target": "published",
        "criterion": "aic",
        "solver": "cd",
    }
    assert saved["Default"] == {
        "predictors": NAMES,
        "training_window": [56, 84, 728, "expanding"],
        "name": "Default",
        "scaler": "asinh",
        "criterion": "bic",
        "solver": "lars",
    }
    loaded = ModelPipeline.load(path)
    for before, after in zip(pipeline.models, loaded.models):
        assert type(after) is LEARModel
        assert {k: v for k, v in vars(after).items() if not k.startswith("_")} == {
            k: v for k, v in vars(before).items() if not k.startswith("_")
        }
    again = tmp_path / "again.yaml"
    loaded.save(again)
    assert again.read_text() == text


def without_provenance(path):
    return {k: v for k, v in json.loads(Path(path).read_text()).items() if k not in PROVENANCE}


def test_the_manifest_records_the_criterion_the_solver_and_the_kernels(tmp_path, data):
    save = tmp_path / "lear.jsonl"
    run(lear(criterion="aic", solver="cd"), data, "2023-03-20", "2023-03-21", save_to=str(save))
    manifest = json.loads(manifest_path(save).read_text())
    assert manifest["model_params"] == {"scaler": "asinh", "criterion": "aic", "solver": "cd"}
    assert manifest["kernels"] == _lars._kernels().BACKEND
    # a model without the kernels has no such key, so its manifest is what it was
    other = tmp_path / "ols.jsonl"
    run(OLSModel(PREDICTORS, training_window=WINDOW), data, "2023-03-20", "2023-03-21", save_to=str(other))
    assert "kernels" not in json.loads(manifest_path(other).read_text())


def test_a_resume_on_the_other_kernels_warns_and_resumes(tmp_path, data):
    save = tmp_path / "lear.jsonl"
    run(lear(), data, "2023-03-20", "2023-03-21", save_to=str(save))
    stored = lines(save)
    save.write_text("\n".join(stored[: len(stored) // 2]) + "\n")
    path = manifest_path(save)
    manifest = json.loads(path.read_text())
    other = "numba" if manifest["kernels"] == "numpy" else "numpy"
    manifest["kernels"] = other
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.warns(UserWarning, match=f"fitted with the {other} kernels"):
        run(lear(), data, "2023-03-20", "2023-03-21", save_to=str(save))
    assert len(lines(save)) == len(stored) and json.loads(path.read_text())["kernels"] == _lars._kernels().BACKEND
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        run(lear(), data, "2023-03-20", "2023-03-21", save_to=str(save))  # the same backend: quiet


def test_the_kernels_are_ready_before_the_pool_starts(data, monkeypatch):
    order = []
    original = BaseModel._run_tasks
    monkeypatch.setattr(_lars, "_kernels", lambda real=_lars._kernels: order.append("kernels") or real())
    monkeypatch.setattr(BaseModel, "_run_tasks", lambda self, *a, **k: order.append("pool") or original(self, *a, **k))
    run(lear(), data, "2023-03-20", "2023-03-20", horizon=1)
    assert order[:2] == ["kernels", "pool"]


# ---------------------------------------------------------------------------
# The fit: hand fits, the seam, a near-constant column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "criterion, solver, scaler",
    [
        ("bic", "lars", "asinh"),
        ("aic", "cd", {"name": "asinh", "params": {"standardise": False}}),
        ("aic", "lars", "standard"),
    ],
)
def test_forecasts_are_the_kernels_on_the_rows_chosen_by_date_over_a_clock_change(data, criterion, solver, scaler):
    model = lear(criterion=criterion, solver=solver, scaler=scaler)
    records = by_key(run(model, data, "2023-03-24", "2023-03-28"))
    checked = 0
    for (day, period, horizon), record in records.items():
        if period not in (0, 2, 7, 18):
            continue
        expected, fit, _ = hand_forecast(data, period, horizon, day, scaler=scaler, criterion=criterion, solver=solver)
        assert record["prediction"] == pytest.approx(expected, abs=HAND), (day, period, horizon)
        np.testing.assert_allclose(record["coefficients"], fit.w, rtol=0, atol=1e-12)
        checked += 1
    # 02:00 of 26 March: no slot to issue on or aim at (4 keys), and no price_d-1 on 27 March (2)
    assert checked == 5 * 4 * 2 - 6


def test_the_exact_refit_is_the_lasso_at_old_lasso_lars_ic_s_alpha(data):
    """`solver="lars"` against scikit-learn alone: old LassoLarsIC's α* on the normalised path, then `Lasso(α*)` to 1e-12 on
    the centred columns as they are, within 1e-6 in target units; α* itself the in-test copy's."""
    for criterion in ("aic", "bic"):
        for period, day in ((3, "2023-03-21"), (12, "2023-04-02"), (20, "2023-04-10")):
            expected, fit, (sx, sy, st, scale) = hand_forecast(data, period, 1, day, criterion=criterion)
            alpha, _, _, _ = _lars_old_ic(sx, sy, criterion)
            assert fit.alpha == pytest.approx(alpha, rel=1e-9)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                lasso = Lasso(alpha=alpha, tol=1e-12, max_iter=10**6).fit(sx, sy)
            reference = scale.inverse(float(lasso.predict(st)[0]))
            record = by_key(run(lear(criterion=criterion), data, day, day, horizon=1))[(day, period, 1)]
            assert record["prediction"] == pytest.approx(reference, abs=1e-6)
            assert record["prediction"] == pytest.approx(expected, abs=HAND)


def test_fit_predict_is_predict_fitted_of_fit_model_and_fit_model_is_fit_stats_of_the_sums(data):
    train_x, train_y, test_x = rows_of(data, 9, 1, "2023-03-22", NAMES)
    scale = make_scaler("asinh")
    sx, sy, st = scale.fit_transform(train_x, train_y, test_x)
    for criterion, solver in (("bic", "lars"), ("aic", "cd")):
        model = lear(criterion=criterion, solver=solver)
        once = model._fit_predict(sx, sy, st)
        fitted = model._fit_model(sx, sy)
        assert once == model._predict_fitted(fitted, st)
        assert [x.hex() for x in once[1]] == [x.hex() for x in fitted.w.tolist()]
        via_stats = model._fit_stats(_lars.gram_stats(sx, sy))
        assert np.array_equal(via_stats.w, fitted.w) and via_stats.alpha == fitted.alpha
        assert isinstance(once[0], float) and isinstance(once[1], list)
    with pytest.raises(ValueError, match="one row"):
        lear()._predict_fitted(fitted, np.vstack([st, st]))


@pytest.mark.parametrize("refit, issues, anchor", [("7D", ["2023-03-20", "2023-03-23", "2023-03-26"], "2023-03-20")])
def test_a_refit_forecast_is_the_hand_fit_at_its_anchor(data, refit, issues, anchor):
    records = by_key(run(lear(refit=refit), data, issues[0], issues[-1]))
    for day in issues:
        for period, horizon in ((1, 1), (5, 2), (22, 2)):
            expected, fit, _ = hand_forecast(data, period, horizon, day, anchor=anchor)
            record = records[(day, period, horizon)]
            assert record["refit_anchor"] == anchor
            assert record["prediction"] == pytest.approx(expected, abs=1e-9), (day, period, horizon)
            np.testing.assert_allclose(record["coefficients"], fit.w, rtol=0, atol=1e-9)


def test_a_published_newest_target_trains_on_the_issue_day_s_price(data):
    info = InformationSet("11:45", "Europe/Brussels", rules={"price": Availability.published("13:00", 1)})
    records = by_key(run(lear(newest_target="published"), data, "2023-03-21", "2023-03-22", horizon=1, cut_rules=info))
    for (day, period, horizon), record in records.items():
        if period in (4, 16):
            expected, _, _ = hand_forecast(data, period, horizon, day, newest=0)
            assert record["prediction"] == pytest.approx(expected, abs=HAND)
            assert record["newest_target_day"] == day


def test_a_near_constant_column_gets_coefficient_zero_and_moves_nothing(data):
    """0.3 with 1e-15 of noise is a constant: the scaler trains on exact zeros, LARS never lets it in and CD never moves it."""
    rng = np.random.default_rng(5)
    flat = data.assign(flat=0.3 + rng.normal(0, 1e-15, len(data)))
    assert flat["flat"].std() > 0
    for solver in ("lars", "cd"):
        with_it = by_key(run(lear(NAMES + ["flat"], solver=solver), flat, "2023-03-21", "2023-03-22"))
        without = by_key(run(lear(NAMES, solver=solver), flat, "2023-03-21", "2023-03-22"))
        assert with_it.keys() == without.keys()
        for key, record in with_it.items():
            assert record["coefficients"][-1] == 0.0
            assert record["prediction"] == pytest.approx(without[key]["prediction"], abs=1e-12)


# ---------------------------------------------------------------------------
# Determinism, resume, the pool, backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("refit", [None, "7D"])
def test_one_two_or_three_processes_and_any_order_give_the_same_records(refit, data, monkeypatch):
    def records():
        return keyed(run(lear(window=[20, 40], refit=refit), data, "2023-03-22", "2023-03-28"))

    expected = records()
    # 02:00 of 26 March: no slot to issue on (2) or aim at (2), and no price_d-1 on 27 March (2)
    assert len(expected) == 7 * 24 * 2 - 6
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


@pytest.mark.parametrize("refit", [None, "7D"])
def test_a_run_cut_short_resumes_to_a_fresh_one(refit, tmp_path):
    frame = with_noise(make_market_data(start="2023-08-20", days=80, noise=5.0, seed=4, horizon=2))
    fresh, cut = tmp_path / "fresh" / "m.jsonl", tmp_path / "cut" / "m.jsonl"

    def model():
        return lear(window=[20, 40], refit=refit, solver="cd", criterion="aic")

    run(model(), frame, "2023-10-26", "2023-11-01", save_to=str(fresh))
    run(model(), frame, "2023-10-26", "2023-11-01", save_to=str(cut))
    stored = lines(cut)
    keep = np.random.default_rng(2).random(len(stored)) < 0.6
    cut.write_text("\n".join(line for line, kept in zip(stored, keep) if kept) + "\n")
    assert 0 < keep.sum() < len(stored)
    run(model(), frame, "2023-10-26", "2023-11-01", save_to=str(cut))
    assert lines(cut) == lines(fresh)
    assert without_provenance(manifest_path(cut)) == without_provenance(manifest_path(fresh))


@pytest.mark.skipif(not has_extra("fast"), reason="needs the [fast] extra (numba)")
def test_the_backends_agree_within_1e_9_with_the_same_kinks(data, monkeypatch):
    def run_on(backend):
        monkeypatch.setenv("EPFTOOLBOX2_KERNELS", backend)
        return by_key(run(lear(window=[20, 40], criterion="aic", solver="cd"), data, "2023-03-24", "2023-03-27"))

    numpy, numba = run_on("numpy"), run_on("numba")
    assert numpy.keys() == numba.keys()
    for key, record in numpy.items():
        assert numba[key]["prediction"] == pytest.approx(record["prediction"], abs=1e-9)
        for a, b in zip(record["member_predictions"], numba[key]["member_predictions"]):
            assert a == pytest.approx(b, abs=1e-9)
    train_x, train_y, _ = rows_of(data, 9, 1, "2023-03-22", NAMES)
    sx, sy, _ = make_scaler("asinh").fit_transform(train_x, train_y, train_x[:1])
    stats = _lars.gram_stats(sx, sy)
    for criterion in ("aic", "bic"):
        for solver in ("lars", "cd"):
            a = _lars.lear_fit(stats, criterion, solver)
            b = _lars._numba_kernels().lear_fit(stats, criterion, solver)
            assert (a.kink, a.converged) == (b.kink, b.converged)
            np.testing.assert_allclose(a.w, b.w, rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# verify: the leakage boundary
# ---------------------------------------------------------------------------

RULES = {
    "load": Availability.published("10:00", 7),
    "wind": Availability.published("10:00", 7),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "noise_a": Availability.static(),
    "noise_b": Availability.static(),
    "price": Availability.published("13:00", 1),
}
LEAR_SHAPED = BASE + ["price_d-1", "price_d-2", "noise_a"]
TEST_PERIOD = ("2023-04-03", "2023-04-28")


@pytest.fixture(scope="module")
def verify_data():
    df = with_noise(make_market_data(days=120, noise=5.0, seed=7, horizon=3))
    df["price_d-2"] = calendar_shift(df["price"], -2)
    return df


@pytest.fixture
def one_thread():
    with threadpool_limits(limits=1):
        yield


def verified(model, frame, **kwargs):
    pipeline = ModelPipeline(information_set=InformationSet("11:45", "Europe/Brussels", rules=RULES))
    return pipeline.add_model(model).verify(frame, *TEST_PERIOD, days=1, **kwargs)


@pytest.mark.parametrize("newest_target", ["d-1", "published"])
@pytest.mark.parametrize("refit", [None, "7D"])
def test_verify_is_ok_with_a_control_that_moved(verify_data, one_thread, newest_target, refit):
    model = lear(LEAR_SHAPED, [20, 40], newest_target=newest_target, refit=refit, name="LEAR")
    report = verified(model, verify_data)
    verdict = report.table().iloc[0]
    assert report.ok and verdict["status"] == "ok" and report.problems() == []
    assert (verdict["forecasts"], verdict["changed"], verdict["control moved"]) == (24, 0, 24)


class ReadsTheTargetDay(LEARModel):
    """Its `noise_a` column is really the price of the target day: the name is legal, the frame is untouched."""

    def _preprocess(self, data, horizon, target):
        out = super()._preprocess(data, horizon, target)
        out["noise_a"] = out[f"{target}_d+1"].fillna(0.0)
        return out


def test_a_model_that_reads_the_target_day_is_a_leak(verify_data, one_thread):
    report = verified(
        ReadsTheTargetDay(LEAR_SHAPED, training_window=40, name="Sneaky"), verify_data, positive_control=False
    )
    assert not report.ok and report.table().iloc[0]["status"] == "leak"


# ---------------------------------------------------------------------------
# Next to the other models
# ---------------------------------------------------------------------------


def test_lear_leaves_the_other_models_manifests_and_yaml_alone(tmp_path, data):
    """No `kernels` key, no new parameter: OLS and Ridge write what they wrote before LEAR existed."""
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, 30, name="OLS")).add_model(RidgeModel(PREDICTORS, 30))
    assert pipeline.to_dict()["models"] == [
        {
            "class": "OLSModel",
            "params": {"predictors": PREDICTORS, "training_window": 30, "name": "OLS", "scaler": "standard"},
        },
        {
            "class": "RidgeModel",
            "params": {"predictors": PREDICTORS, "training_window": 30, "name": "Model", "scaler": "standard"},
        },
    ]
    for model in pipeline.models:
        save = tmp_path / f"{model.name}.jsonl"
        run(model, data, "2023-03-20", "2023-03-20", horizon=1, save_to=str(save))
        assert set(json.loads(manifest_path(save).read_text())) & {"kernels"} == set()


def test_a_lasso_with_a_vanishing_penalty_choice_still_forecasts_every_key(data):
    """Few rows and more inputs than rows (n <= p, as Lago's 49-row window over 247 inputs): every key is forecast, finite."""
    wide = data.assign(**{f"noise_{i}": np.random.default_rng(i).normal(0, 1, len(data)) for i in range(30)})
    names = NAMES + [f"noise_{i}" for i in range(30)]
    records = run(lear(names, window=12, criterion="aic", solver="cd"), wide, "2023-03-21", "2023-03-21")
    assert len(records) == 24 * 2 and all(math.isfinite(r["prediction"]) for r in records)
