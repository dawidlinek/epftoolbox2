"""`LEARModel(criterion="cv")` (step 4f): old `LassoLarsCV(cv=KFold(5), normalize=True, max_iter=2500)` from the blocks' sums.

scikit-learn 0.24.2 doesn't install next to the locked version, so `_lasso_lars_cv` stands in for it here: its `LarsCV.fit` and
`_lars_path_residues` written out on the rows, with `KFold(5)`, today's `lars_path` without its rounding line (0.24.2's solver,
see `test_lars_kernels.py`) on the Gram, residues of the held-out rows and scipy's `interp1d`. The kernel works from block sums
instead, so the two share only `lars_path`. `test_lear_lago.py` holds the kernel to 0.24.2 itself on Lago's data (the
fixture's `cv_{M}.csv`), X-form paths of the short windows included.
"""

import math
import warnings

import numpy as np
import pytest
from scipy import interpolate
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lars_path as sklearn_lars_path
from sklearn.model_selection import KFold

from epftoolbox2 import tui
from epftoolbox2._extras import has_extra
from epftoolbox2.information_set import InformationSet
from epftoolbox2.models import LEARModel
from epftoolbox2.models import _lars
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.scalers import make_scaler
from tests.models.test_lars_kernels import EPS, _without_rounding
from threadpoolctl import threadpool_limits

from tests.models.test_lear import (
    LEAR_SHAPED,
    NAMES,
    RULES,
    TEST_PERIOD,
    InProcessPool,
    by_key,
    rows_of,
    run,
    with_noise,
)
from tests.synthetic import calendar_shift, make_market_data

TIGHT = 1e-9  # the kernel against the in-test 0.24.2 on the same rows


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)


@pytest.fixture(scope="module")
def data():
    """test_lear.py's frame."""
    return with_noise(make_market_data(days=120, noise=5.0, seed=3, horizon=2))


@pytest.fixture(scope="module")
def verify_data():
    """test_lear.py's frame for `verify`."""
    df = with_noise(make_market_data(days=120, noise=5.0, seed=7, horizon=3))
    df["price_d-2"] = calendar_shift(df["price"], -2)
    return df


# ---------------------------------------------------------------------------
# The in-test 0.24.2
# ---------------------------------------------------------------------------


def _path(Xn, yc, alpha_min=0.0):
    """0.24.2's `lars_path(method="lasso", max_iter=2500)` on the Gram: today's without its rounding line."""
    with warnings.catch_warnings(), _without_rounding():
        warnings.simplefilter("ignore", ConvergenceWarning)
        alphas, _, coefs = sklearn_lars_path(
            Xn, yc, Gram=Xn.T @ Xn, method="lasso", alpha_min=np.float64(alpha_min), max_iter=2500, eps=EPS
        )
    return alphas, coefs


def _lasso_lars_cv(X, y, max_n_alphas=1000):
    """(alpha_, coef_, intercept_) of 0.24.2's `LassoLarsCV(cv=KFold(5), normalize=True, max_iter=2500, max_n_alphas)`, its
    `fit` and `_lars_path_residues` line by line on the rows (the Gram for every path, as the kernel)."""
    cv_paths = []
    for train, test in KFold(5).split(X):
        X_train, y_train, X_test, y_test = X[train].copy(), y[train].copy(), X[test].copy(), y[test].copy()
        X_mean = X_train.mean(axis=0)
        X_train -= X_mean
        X_test -= X_mean
        y_mean = y_train.mean(axis=0)
        y_train -= y_mean
        y_test -= y_mean
        norms = np.sqrt(np.sum(X_train**2, axis=0))
        nonzeros = np.flatnonzero(norms)
        X_train[:, nonzeros] /= norms[nonzeros]
        alphas, coefs = _path(X_train, y_train)
        coefs[nonzeros] /= norms[nonzeros][:, np.newaxis]
        residues = np.dot(X_test, coefs) - y_test[:, np.newaxis]
        cv_paths.append((alphas, residues.T))
    all_alphas = np.unique(np.concatenate([alphas for alphas, _ in cv_paths]))
    stride = int(max(1, int(len(all_alphas) / float(max_n_alphas))))
    all_alphas = all_alphas[::stride]
    mse_path = np.empty((len(all_alphas), len(cv_paths)))
    for index, (alphas, residues) in enumerate(cv_paths):
        alphas = alphas[::-1]
        residues = residues[::-1]
        if alphas[0] != 0:
            alphas = np.r_[0, alphas]
            residues = np.r_[residues[0, np.newaxis], residues]
        if alphas[-1] != all_alphas[-1]:
            alphas = np.r_[alphas, all_alphas[-1]]
            residues = np.r_[residues, residues[-1, np.newaxis]]
        with np.errstate(divide="ignore", invalid="ignore"):
            this_residues = interpolate.interp1d(alphas, residues, axis=0)(all_alphas)
        this_residues **= 2
        mse_path[:, index] = np.mean(this_residues, axis=-1)
    mask = np.all(np.isfinite(mse_path), axis=-1)
    all_alphas = all_alphas[mask]
    best_alpha = all_alphas[np.argmin(mse_path[mask].mean(axis=-1))]
    X_offset = X.mean(axis=0)
    Xc = X - X_offset
    X_scale = np.sqrt(np.sum(Xc**2, axis=0))
    X_scale[X_scale == 0] = 1.0
    y_offset = y.mean()
    _, coef_path = _path(Xc / X_scale, y - y_offset, best_alpha)
    coef = coef_path[:, -1] / X_scale
    return float(best_alpha), coef, float(y_offset - X_offset @ coef)


def _design(kind, seed):
    rng = np.random.default_rng(seed)
    n, p = {"n>p": (200, 30), "n<p": (49, 60), "square": (60, 60), "wide": (77, 247)}[kind]
    X = rng.standard_normal((n, p)) * rng.uniform(0.5, 3.0, p) + rng.normal(0, 2, p)
    beta = np.zeros(p)
    beta[: p // 4] = rng.standard_normal(p // 4)
    return X, X @ beta + 2.0 * rng.standard_normal(n)


def _intercept(fit):
    return float(fit.ybar - fit.xbar @ fit.w)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_n_alphas", [1000, 7], ids=["grid", "strided"])
@pytest.mark.parametrize("kind", ["n>p", "n<p", "square", "wide"])
def test_the_kernel_is_old_lasso_lars_cv_on_the_rows(kind, max_n_alphas):
    """α*, the coefficients and the intercept of the kernel, from the blocks' sums, against 0.24.2's LassoLarsCV on the rows;
    with max_n_alphas=7 the grid is strided."""
    for seed in range(3):
        X, y = _design(kind, seed)
        alpha, coef, intercept = _lasso_lars_cv(X, y, max_n_alphas)
        fit = _lars.lear_fit_cv(_lars.cv_stats(X, y), max_n_alphas)
        assert fit.alpha == pytest.approx(alpha, rel=TIGHT), (kind, seed)
        np.testing.assert_allclose(fit.w, coef, rtol=0, atol=TIGHT * max(1.0, np.abs(coef).max()))
        assert _intercept(fit) == pytest.approx(intercept, abs=TIGHT * max(1.0, abs(intercept)))
        assert fit.converged


def test_the_blocks_are_kfold_s_contiguous_test_blocks_in_row_order():
    X, y = _design("n<p", 0)
    stats = _lars.cv_stats(X, y)
    tests = [test for _, test in KFold(5).split(X)]
    assert [len(t) for t in tests] == [10, 10, 10, 10, 9] == [b.n for b in stats.blocks]
    for block, test in zip(stats.blocks, tests):
        assert np.array_equal(test, np.arange(test[0], test[-1] + 1))
        expected = _lars.gram_stats(X[test], y[test])
        for got, want in zip(block, expected):
            np.testing.assert_array_equal(got, want)
    whole = _lars.pooled(stats.blocks)
    direct = _lars.gram_stats(X, y)
    assert whole.n == direct.n
    for got, want in zip(whole[1:], direct[1:]):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


def test_each_fold_is_its_training_rows_normalised_and_its_held_out_rows_about_their_means():
    X, y = _design("n>p", 1)
    folds = _lars.cv_folds(_lars.cv_stats(X, y))
    for fold, (train, test) in zip(folds, KFold(5).split(X)):
        xbar, ybar = X[train].mean(axis=0), y[train].mean()
        xc = X[train] - xbar
        norm = np.sqrt((xc**2).sum(axis=0))
        xn = xc / norm
        assert (fold.n, fold.n_test) == (len(train), len(test))
        np.testing.assert_allclose(fold.norm, norm, rtol=1e-13)
        np.testing.assert_allclose(fold.R, xn.T @ xn, rtol=0, atol=1e-13)
        np.testing.assert_allclose(fold.qn, xn.T @ (y[train] - ybar), rtol=0, atol=1e-12)
        held_x, held_y = X[test] - xbar, y[test] - ybar
        np.testing.assert_allclose(fold.A, held_x.T @ held_x, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(fold.b, held_x.T @ held_y, rtol=1e-12, atol=1e-10)
        assert fold.c == pytest.approx(held_y @ held_y, rel=1e-12)


def test_fewer_rows_than_folds_raise_as_kfold_did_and_five_fit():
    X, y = _design("n>p", 2)
    with pytest.raises(ValueError, match="at least 5 training rows"):
        _lars.cv_stats(X[:4], y[:4])
    fit = _lars.lear_fit_cv(_lars.cv_stats(X[:5], y[:5]))
    assert np.isfinite(fit.w).all() and math.isfinite(fit.ybar)


def test_a_constant_target_fits_the_mean():
    """Every fold's path is the single kink α = 0 (0.24.2's interp1d raised on it): w = 0 and the forecast is the mean."""
    X, _ = _design("n>p", 3)
    fit = _lars.lear_fit_cv(_lars.cv_stats(X, np.full(X.shape[0], 2.5)))
    assert fit.alpha == 0.0 and not fit.w.any() and fit.ybar == pytest.approx(2.5, abs=1e-15)


def test_a_column_constant_on_a_fold_s_training_rows_is_dropped_in_that_fold_only():
    """A dummy that is 0.1 (with its rounding) on the first four blocks and varies in the last: in the fold that holds out the
    last block it is constant, and its centred noise would become a unit column there; it is dropped in that fold, as the
    scaler drops a column constant over the window, and nowhere else."""
    X, y = _design("n>p", 4)
    X[:, 7] = 0.1
    X[160:, 7] = np.random.default_rng(0).normal(0.1, 1.0, 40)
    stats = _lars.cv_stats(X, y)
    raw = _lars.pooled(stats.blocks[:4]).G[7, 7]
    assert raw > 0.0  # rounding noise, not an exact 0
    folds = _lars.cv_folds(stats)
    assert not folds[4].R[7].any() and not folds[4].R[:, 7].any() and folds[4].qn[7] == 0.0
    for fold in folds[:4]:
        assert fold.R[7, 7] == pytest.approx(1.0)
    fit = _lars.lear_fit_cv(stats)
    assert np.isfinite(fit.w).all() and np.abs(fit.w).max() < 100


def test_a_twin_on_a_fold_s_training_rows_is_dropped_in_that_fold_only():
    """Column 9 is a copy of column 2 on the first four blocks only. The fold that holds out the last block sees an exact twin,
    which LARS can't take, and drops the later copy as `lear_fit` would on that window; the other folds and the full window
    see two columns."""
    X, y = _design("n>p", 5)
    X[:160, 9] = X[:160, 2]
    folds = _lars.cv_folds(_lars.cv_stats(X, y))
    assert not folds[4].R[9].any() and folds[4].R[2, 2] == pytest.approx(1.0)
    for fold in folds[:4]:
        assert fold.R[9, 9] == pytest.approx(1.0) and abs(fold.R[2, 9]) < 1.0 - 1e-6
    _, R, _, _ = _lars.cv_window(_lars.cv_stats(X, y))
    assert R[9, 9] == pytest.approx(1.0)


def test_the_cd_solver_and_the_wrong_statistics_are_refused():
    X, y = _design("n>p", 6)
    with pytest.raises(ValueError, match="solver='lars'"):
        _lars.lear_fit(_lars.cv_stats(X, y), "cv", "cd")
    with pytest.raises(TypeError, match="CvStats"):
        _lars.lear_fit(_lars.gram_stats(X, y), "cv", "lars")
    with pytest.raises(TypeError, match="GramStats"):
        _lars.lear_fit(_lars.cv_stats(X, y), "bic", "lars")
    fit = _lars.lear_fit(_lars.cv_stats(X, y), "cv", "lars")
    again = _lars.lear_fit_cv(_lars.cv_stats(X, y))
    assert np.array_equal(fit.w, again.w) and fit.alpha == again.alpha


@pytest.mark.skipif(not has_extra("fast"), reason="needs the [fast] extra (numba)")
def test_numba_agrees_with_numpy_with_the_same_alpha():
    numba = _lars._numba_kernels()
    for seed in range(40):
        kind = ("n>p", "n<p", "square", "wide")[seed % 4]
        X, y = _design(kind, 100 + seed)
        stats = _lars.cv_stats(X, y)
        for cap in (1000, 7):
            a, b = _lars.lear_fit_cv(stats, cap), numba.lear_fit_cv(stats, cap)
            assert (a.kink, a.converged) == (b.kink, b.converged), (kind, seed, cap)
            assert a.alpha == pytest.approx(b.alpha, rel=1e-12)
            np.testing.assert_allclose(a.w, b.w, rtol=0, atol=1e-10 * max(1.0, np.abs(a.w).max()))
    with pytest.raises(ValueError, match="solver='lars'"):
        numba.lear_fit(stats, "cv", "cd")


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def hand_cv_forecast(frame, period, horizon, day, names=NAMES, scaler="asinh", **rows):
    """The forecast by hand: the rows by date, the library's scaler fitted on them, `lear_fit_cv` of their blocks' sums with the
    numpy kernels, `ybar + (x - xbar) @ w` on the scaled test row, scaled back."""
    train_x, train_y, test_x = rows_of(frame, period, horizon, day, names, **rows)
    scale = make_scaler(scaler)
    sx, sy, st = scale.fit_transform(train_x, train_y, test_x)
    fit = _lars.lear_fit_cv(_lars.cv_stats(sx, sy))
    return scale.inverse(float(fit.ybar + (st[0] - fit.xbar) @ fit.w)), fit, (sx, sy, st, scale)


def test_cv_takes_the_lars_solver_and_cd_raises_whichever_is_set_first():
    model = LEARModel(NAMES, criterion="cv")
    assert (model.criterion, model.solver) == ("cv", "lars")
    assert model._model_kwargs == {"criterion": "cv", "solver": "lars"}
    with pytest.raises(ValueError, match="solver='lars'"):
        LEARModel(NAMES, criterion="cv", solver="cd")
    with pytest.raises(ValueError, match="solver='lars'"):
        model.solver = "cd"
    assert model.solver == "lars"
    lago = LEARModel.lago(["load_forecast_amprion", "wind_solar_forecast"])
    with pytest.raises(ValueError, match="solver='lars'"):
        lago.criterion = "cv"
    lago.solver = "lars"
    lago.criterion = "cv"
    assert (lago.criterion, lago.solver) == ("cv", "lars")


def test_the_tui_offers_cv():
    fields = {f["name"]: f for f in tui.PARAM_SCHEMAS["LEARModel"]}
    assert fields["criterion"]["choices"] == ["bic", "aic", "cv"]
    made = tui._instantiate("LEARModel", {"predictors": ["load"], "criterion": "cv"})
    assert (made.criterion, made.solver) == ("cv", "lars")


def test_a_cv_model_saves_and_loads_back_equal(tmp_path):
    path = tmp_path / "pipeline.yaml"
    pipeline = ModelPipeline().add_model(LEARModel(NAMES, [20, 40], criterion="cv", name="CV"))
    pipeline.save(path)
    assert pipeline.to_dict()["models"][0]["params"] == {
        "predictors": NAMES,
        "training_window": [20, 40],
        "name": "CV",
        "scaler": "asinh",
        "criterion": "cv",
        "solver": "lars",
    }
    loaded = ModelPipeline.load(path).models[0]
    assert type(loaded) is LEARModel and (loaded.criterion, loaded.solver) == ("cv", "lars")


def test_forecasts_are_the_cv_kernel_on_the_rows_chosen_by_date_over_a_clock_change(data):
    records = by_key(run(LEARModel(NAMES, 40, "cv"), data, "2023-03-24", "2023-03-28"))
    checked = 0
    for (day, period, horizon), record in records.items():
        if period not in (0, 2, 7, 18):
            continue
        expected, fit, _ = hand_cv_forecast(data, period, horizon, day)
        assert record["prediction"] == pytest.approx(expected, abs=1e-12), (day, period, horizon)
        np.testing.assert_allclose(record["coefficients"], fit.w, rtol=0, atol=1e-12)
        checked += 1
    assert checked == 5 * 4 * 2 - 6  # as for the criteria (test_lear.py): 02:00 of 26 March and price_d-1 on the 27th


def test_the_cv_fit_is_old_lasso_lars_cv_on_the_scaled_rows(data):
    for period, day in ((3, "2023-03-21"), (12, "2023-04-02"), (20, "2023-04-10")):
        _, fit, (sx, sy, _, _) = hand_cv_forecast(data, period, 1, day)
        alpha, coef, intercept = _lasso_lars_cv(sx, sy)
        assert fit.alpha == pytest.approx(alpha, rel=TIGHT)
        np.testing.assert_allclose(fit.w, coef, rtol=0, atol=TIGHT)
        assert _intercept(fit) == pytest.approx(intercept, abs=TIGHT)


def test_fit_predict_is_predict_fitted_of_fit_model_and_fit_model_is_fit_stats_of_the_blocks(data):
    train_x, train_y, test_x = rows_of(data, 9, 1, "2023-03-22", NAMES)
    sx, sy, st = make_scaler("asinh").fit_transform(train_x, train_y, test_x)
    model = LEARModel(NAMES, 40, "cv")
    once = model._fit_predict(sx, sy, st)
    fitted = model._fit_model(sx, sy)
    assert once == model._predict_fitted(fitted, st)
    via_stats = model._fit_stats(_lars.cv_stats(sx, sy))
    assert np.array_equal(via_stats.w, fitted.w) and via_stats.alpha == fitted.alpha


def test_a_refit_forecast_is_the_hand_cv_fit_at_its_anchor(data):
    records = by_key(run(LEARModel(NAMES, 40, "cv", refit="7D"), data, "2023-03-20", "2023-03-26"))
    for day in ("2023-03-20", "2023-03-23", "2023-03-26"):
        for period, horizon in ((1, 1), (5, 2), (22, 2)):
            expected, fit, _ = hand_cv_forecast(data, period, horizon, day, anchor="2023-03-20")
            record = records[(day, period, horizon)]
            assert record["refit_anchor"] == "2023-03-20"
            assert record["prediction"] == pytest.approx(expected, abs=1e-9), (day, period, horizon)
            np.testing.assert_allclose(record["coefficients"], fit.w, rtol=0, atol=1e-9)


def test_a_window_of_fewer_rows_than_folds_skips_every_key_as_a_fit_error(data):
    with pytest.raises(RuntimeError, match=r"fit_error x24 .*at least 5 training rows for its 5 folds, got 4"):
        LEARModel(NAMES, 4, "cv").run(data, "2023-03-21", "2023-03-21", horizon=1)


def test_cv_is_not_bic_and_costs_more_paths(data):
    """The two criteria pick different penalties on these rows (else the comparison tests would prove nothing)."""
    _, cv, (sx, sy, _, _) = hand_cv_forecast(data, 12, 1, "2023-04-02")
    bic = _lars.lear_fit(_lars.gram_stats(sx, sy), "bic", "lars")
    assert cv.alpha != pytest.approx(bic.alpha, rel=1e-3)


@pytest.mark.skipif(not has_extra("fast"), reason="needs the [fast] extra (numba)")
def test_the_backends_agree_on_cv_records(data, monkeypatch):
    def run_on(backend):
        monkeypatch.setenv("EPFTOOLBOX2_KERNELS", backend)
        return by_key(run(LEARModel(NAMES, [20, 40], "cv"), data, "2023-03-24", "2023-03-26"))

    numpy, numba = run_on("numpy"), run_on("numba")
    assert numpy.keys() == numba.keys()
    for key, record in numpy.items():
        assert numba[key]["prediction"] == pytest.approx(record["prediction"], abs=1e-9)


# ---------------------------------------------------------------------------
# verify: the leakage boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("newest_target", ["d-1", "published"])
def test_verify_is_ok_with_a_control_that_moved(verify_data, newest_target):
    model = LEARModel(LEAR_SHAPED, [20, 40], "cv", newest_target=newest_target, name="LEAR_cv")
    pipeline = ModelPipeline(information_set=InformationSet("11:45", "Europe/Brussels", rules=RULES))
    with threadpool_limits(limits=1):
        report = pipeline.add_model(model).verify(verify_data, *TEST_PERIOD, days=1)
    verdict = report.table().iloc[0]
    assert report.ok and verdict["status"] == "ok" and report.problems() == []
    assert (verdict["forecasts"], verdict["changed"], verdict["control moved"]) == (24, 0, 24)
