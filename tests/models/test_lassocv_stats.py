"""`models/_cv_stats.py`: KFold's CV blocks of a fit's rows from exact span sums, `LassoCVModel._fit_stats` (scikit-learn's own
coordinate descent on each fold's Gram) and `LEARModel(criterion="cv")._fit_stats` from them, against the model path on the
same rows (no pipeline).

The model path of a fit is `StandardScaler.fit_transform` on its rows, then `LassoCV(cv=5, max_iter, n_jobs=1,
fit_intercept)` (or `_fit_model`), `_predict_fitted` on the scaled issue row and `inverse`. The engine's is `SpanSums(span)`,
`over(ρ)` and `over(block)` for each of `block_rows(ρ)`, `standard_stats`, `block_stats`, `_fit_stats`, `_predict_fitted` on
`StandardParams.transform(row)` and `StandardParams.inverse`. Tolerances in target units (targets of about 50): LassoCV 1e-8
with the same `alpha_`, LEAR `"cv"` 1e-9 with the same penalty position, as the step-5 spec's table. A penalty flip is allowed
only as a tie of the cross-validated error and at most once per 100 designs. Larger sweeps, timings and mutants are in
`epftoolbox2-research/library_check_step5/lassocv_stats/`.
"""

import functools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, enet_path
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from epftoolbox2.models import LassoCVModel, LEARModel
from epftoolbox2.models import _cv_stats, _lars, _stats
from epftoolbox2.models._sums import SpanSums
from epftoolbox2.scalers.standard import StandardScaler

DE = Path(__file__).resolve().parents[1] / "fixtures" / "lago" / "data_DE.parquet"
LASSO_TOL = 1e-8
LEAR_TOL = 1e-9
MAX_ITER = 10000  # LassoCVModel's default


@pytest.fixture(autouse=True)
def one_blas_thread():
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # both paths run the same descent and warn alike
        yield


# --- the two paths -----------------------------------------------------------------------------------------------------


def scaled_rows(x, y, x0, mask, scale_binary=False):
    scaler = StandardScaler(scale_binary=scale_binary)
    mask = None if mask is None else np.asarray(mask, dtype=bool)
    Z, ys, z0 = scaler.fit_transform(x.copy(), y.copy(), x0.copy(), scalable_mask=mask)
    return Z, ys, z0, scaler


def model_lassocv(x, y, x0, mask, fit_intercept=True, scale_binary=False):
    """The model path's forecast (target units) and its fitted `LassoCV`."""
    Z, ys, z0, scaler = scaled_rows(x, y, x0, mask, scale_binary)
    fitted = LassoCV(cv=5, max_iter=MAX_ITER, n_jobs=1, fit_intercept=fit_intercept).fit(Z, ys)
    return scaler.inverse(float(fitted.predict(z0)[0])), fitted


def model_lear(model, x, y, x0, mask):
    Z, ys, z0, scaler = scaled_rows(x, y, x0, mask)
    fitted = model._fit_model(Z, ys)
    return scaler.inverse(float(model._predict_fitted(fitted, z0)[0])), fitted


def engine_stats(xs, ys, rows, mask, scale_binary=False, cols=None, folds=5):
    """`ScaledStats` with their CV blocks, and `StandardParams`, of the span (`xs`, `ys`) over `rows` and `cols`."""
    cols = np.arange(xs.shape[1]) if cols is None else np.asarray(cols)
    rows = np.asarray(rows)
    span = SpanSums(xs, ys)
    raw = span.over(rows, cols)
    values = (xs[rows][:, cols], ys[rows])
    stats, params = _stats.standard_stats(raw, values, mask, scale_binary)
    blocks = [span.over(block, cols) for block in _cv_stats.block_rows(rows, folds)]
    return stats._replace(blocks=_cv_stats.block_stats(raw, blocks, params, values)), params


def engine_path(model, stats, params, x0):
    """The engine's forecast (target units) and fit, or (None, None) where `_fit_stats` hands it back."""
    fitted = model._fit_stats(stats)
    if fitted is None:
        return None, None
    pred, _ = model._predict_fitted(fitted, params.transform(x0))
    return params.inverse(float(pred)), fitted


def position(alphas, alpha):
    return int(np.argmin(np.abs(np.asarray(alphas) - alpha)))


def compare_lassocv(x, y, x0, mask, fit_intercept, rows=None, xs=None, ys=None, scale_binary=False):
    """(|Δ forecast|, flip) of the engine against `LassoCV` on one fit, flip = None or (positions, CV errors) when the two
    choose different penalties. `xs`, `ys` the span when it holds more rows than the fit's (`rows` into it)."""
    pm, fm = model_lassocv(x, y, x0, mask, fit_intercept, scale_binary)
    if xs is None:
        xs, ys, rows = x, y, np.arange(len(y))
    stats, params = engine_stats(xs, ys, rows, mask, scale_binary)
    pe, fe = engine_path(LassoCVModel([], fit_intercept=fit_intercept), stats, params, x0)
    assert pe is not None, "handed back"
    i, j = position(fm.alphas_, fm.alpha_), position(fm.alphas_, fe.alpha_)
    assert abs(fe.alpha_ / fm.alphas_[j] - 1.0) <= 1e-12  # the same grid, to rounding
    if i != j:
        mean = fm.mse_path_.mean(axis=1)
        return abs(pe - pm), ((i, j), (float(mean[i]), float(mean[j])))
    return abs(pe - pm), None


# --- data ----------------------------------------------------------------------------------------------------------------


def realistic_columns(rng, n):
    """Price with spikes (3,000 and -500 against ~50), a load in MW, night-time solar mostly 0, a 0/1 dummy, prices with two
    decimals, a weather-like column: (n, 6), and which are 0/1."""
    t = np.arange(n)
    price = 50 + 15 * np.sin(t / 7) + rng.normal(size=n) * 8
    spikes = rng.random(n) < 0.01
    price[spikes] = rng.choice([3000.0, -500.0], size=spikes.sum())
    load = np.round(21000 + 4000 * np.sin(t / 3.5) + rng.normal(size=n) * 800, 1)
    solar = np.zeros(n)
    on = rng.random(n) < 0.05
    solar[on] = rng.uniform(0, 300, size=on.sum())
    dummy = (rng.random(n) < 0.3).astype(float)
    cents = np.round(rng.normal(40, 25, size=n), 2)
    weather = rng.normal(size=n) * 3 + 12
    return np.column_stack([price, load, solar, dummy, cents, weather]), np.array([0, 0, 0, 1, 0, 0], dtype=bool)


def harsh_design(rng, n, p, cond):
    """Columns with singular values log-spaced from 1 to 1/cond, rotated, in raw units with offsets; a sparse signal."""
    Q, _ = np.linalg.qr(rng.normal(size=(n, p)))
    W, _ = np.linalg.qr(rng.normal(size=(p, p)))
    s = np.logspace(0, -np.log10(cond), p) if p > 1 else np.ones(1)
    Z = Q @ np.diag(s) @ W.T * np.sqrt(n)
    scale = 10 ** rng.uniform(-1, 4, size=p)
    x = Z * scale + rng.uniform(-1, 1, size=p) * scale * rng.uniform(0, 5, size=p)
    beta = rng.normal(size=p) * (rng.random(p) < 0.5)
    return x, 50 + x @ (beta / scale) * 5 + rng.normal(size=n) * 5


def mixed_design(rng, n, p):
    """Realistic columns (spikes, MW loads, solar, dummies) and Gaussian ones of any scale; a sparse signal, or none."""
    cols = [
        realistic_columns(rng, n)[0][:, j // 3 % 6]
        if j % 3 == 0
        else rng.normal(size=n) * 10 ** rng.uniform(-2, 4) + rng.uniform(-100, 100)
        for j in range(p)
    ]
    x = np.column_stack(cols)
    beta = rng.normal(size=p) * (rng.random(p) < 0.4) * (rng.random() < 0.9)  # one in ten: a target unrelated to x
    return x, 50 + x @ (beta / np.maximum(np.std(x, axis=0), 1e-12)) * 3 + rng.normal(size=n) * 5


def random_design(rng):
    """n > p with every fold's training part longer than p; the issue row last."""
    p = int(np.exp(rng.uniform(0, np.log(30))))
    n = int(rng.integers(int(1.3 * p) + 6, 4 * p + 150))
    x, y = harsh_design(rng, n + 1, p, 10 ** rng.uniform(0, 3)) if rng.random() < 0.5 else mixed_design(rng, n + 1, p)
    return x[:-1], y[:-1], x[-1:]


@functools.lru_cache(maxsize=None)
def lear_design(hour):
    """A Lago-shaped design from the DE fixture: rows are target days t, 247 columns (the prices of t-1, t-2, t-3 and t-7; both
    exogenous series on t, t-1 and t-7; seven weekday dummies of t), the target the price of t at `hour`."""
    frame = pd.read_parquet(DE)
    values = frame.to_numpy(dtype=float)
    days = values.shape[0] // 24
    P, L, WS = (values[: days * 24, k].reshape(days, 24) for k in range(3))
    weekday = pd.DatetimeIndex(frame.index[::24][:days]).dayofweek.to_numpy()
    t = np.arange(7, days)
    dummies = np.zeros((t.size, 7))
    dummies[np.arange(t.size), weekday[t]] = 1.0
    x = np.hstack(
        [P[t - 1], P[t - 2], P[t - 3], P[t - 7], L[t], L[t - 1], L[t - 7], WS[t], WS[t - 1], WS[t - 7], dummies]
    )
    return x, P[t, hour]


def lear_rows(hour, n):
    x, y = lear_design(hour)
    return x[-n - 1 : -1], y[-n - 1 : -1], x[-1:]


def rows_stats(Z, y, folds=5):
    """`ScaledStats` (with blocks) straight from rows that no scaler centred: z̄ and ȳ away from 0, as a VST without
    standardisation leaves them (step 5j). The model path is then `LassoCV` on these rows as they are."""
    g = _lars.gram_stats(Z, y)
    blocks = tuple(
        _stats.ScaledStats(b.n, b.xbar, b.ybar, b.G, b.q, b.yy, yscale=1.0) for b in _lars.cv_stats(Z, y, folds).blocks
    )
    return _stats.ScaledStats(g.n, g.xbar, g.ybar, g.G, g.q, g.yy, blocks=blocks, yscale=1.0)


# --- blocks ----------------------------------------------------------------------------------------------------------------


def test_block_rows_are_kfolds_test_blocks_of_rho():
    rng = np.random.default_rng(1)
    for n_span in (5, 7, 11, 364, 1449):
        keep = rng.random(n_span) < 0.8
        keep[: min(n_span, 5)] = True
        rows = np.flatnonzero(keep)
        blocks = _cv_stats.block_rows(rows)
        expected = [rows[test] for _, test in KFold(5).split(rows)]
        assert [b.tolist() for b in blocks] == [e.tolist() for e in expected]
        assert [b.tolist() for b in _cv_stats.block_rows(keep)] == [e.tolist() for e in expected]
    with pytest.raises(ValueError, match="at least 5 rows"):
        _cv_stats.block_rows(np.arange(4))
    with pytest.raises(ValueError, match="ascending"):
        _cv_stats.block_rows(np.array([0, 2, 1, 3, 4, 5]))


@pytest.mark.parametrize("scale_binary", [False, True])
def test_block_stats_are_the_blocks_of_the_scaled_rows(scale_binary):
    """Each block's n, z̄, ȳ, G, q, yy against `cv_stats` of `fit_transform`'s rows (the model path of LEAR's `"cv"`), relative to
    each element's scale; ρ a strict subset of the span, so the blocks are ρ's, not the span's."""
    rng = np.random.default_rng(2)
    for n in (60, 365, 1449):
        x, binary = realistic_columns(rng, n)
        x = np.column_stack([x, (np.arange(n) >= n - n // 6).astype(float)])  # a dummy that is 1 in the last block only
        binary = np.r_[binary, True]
        y = 50 + 0.3 * x[:, 0] + rng.normal(size=n) * 5
        ys = y.copy()
        ys[rng.random(n) < 0.05] = np.nan
        rows = np.flatnonzero(np.isfinite(ys))
        stats, params = engine_stats(x, ys, rows, ~binary, scale_binary)
        Z, yz, _, _ = scaled_rows(x[rows], y[rows], x[rows][:1], ~binary, scale_binary)
        positions = _cv_stats.block_rows(np.arange(rows.size))
        for got, want, at in zip(stats.blocks, _lars.cv_stats(Z, yz).blocks, positions):
            assert got.n == want.n
            zs = np.sqrt(np.diag(want.G) / want.n + want.xbar**2) + 1.0
            assert np.abs(got.zbar - want.xbar).max() <= 1e-13 * zs.max()
            assert abs(got.ybar - want.ybar) <= 1e-13 * (1 + abs(want.ybar))
            scale = np.sqrt(np.outer(np.diag(want.G), np.diag(want.G))) + 1.0
            assert (np.abs(got.G - want.G) <= 1e-12 * scale).all()
            assert (np.abs(got.q - want.q) <= 1e-12 * (np.sqrt(np.diag(want.G) * want.yy) + 1.0)).all()
            assert abs(got.yy - want.yy) <= 1e-12 * want.yy
            # exactly zero where the column is constant on the block (the dummy before the last block, solar's zeros)
            flat = np.ptp(Z[at], axis=0) == 0
            assert (got.G[flat] == 0).all() and (got.G[:, flat] == 0).all() and (got.q[flat] == 0).all()


def test_blocks_must_be_kfolds_blocks_of_rho_over_the_same_span():
    rng = np.random.default_rng(3)
    x, binary = realistic_columns(rng, 120)
    y = 50 + rng.normal(size=120)
    ys = y.copy()
    ys[:8] = np.nan  # ρ leaves out rows of the first span block
    rows = np.flatnonzero(np.isfinite(ys))
    span = SpanSums(x, ys)
    raw = span.over(rows, np.arange(6))
    values = (x[rows], ys[rows])
    _, params = _stats.standard_stats(raw, values, ~binary)
    span_blocks = [np.intersect1d(rows, b) for _, b in KFold(5).split(np.arange(120))]  # cut on the span's rows
    with pytest.raises(ValueError, match="KFold"):
        _cv_stats.block_stats(raw, [span.over(b, np.arange(6)) for b in span_blocks], params, values)
    other = SpanSums(x + 1.0, ys)  # another span: other shifts
    with pytest.raises(ValueError, match="span"):
        _cv_stats.block_stats(raw, [other.over(b, np.arange(6)) for b in _cv_stats.block_rows(rows)], params, values)


# --- LassoCV against scikit-learn --------------------------------------------------------------------------------------


@pytest.mark.parametrize("part", range(5))
def test_lassocv_same_alpha_within_1e_8_on_random_designs(part):
    """500 designs in all (100 per part), with an intercept, and every fifth design without one too."""
    rng = np.random.default_rng(500 + part)
    worst, flips = 0.0, []
    for i in range(100):
        x, y, x0 = random_design(rng)
        for fit_intercept in (True, False) if i % 5 == 0 else (True,):
            delta, flip = compare_lassocv(x, y, x0, None, fit_intercept)
            if flip is None:
                worst = max(worst, delta)
            else:
                (a, b), (ma, mb) = flip
                assert abs(ma - mb) <= 1e-10 * ma, (i, fit_intercept, flip)  # a tie of the CV error, nothing else
                flips.append((i, fit_intercept, flip))
    assert worst <= LASSO_TOL, worst
    assert len(flips) <= 1, flips


@pytest.mark.parametrize("hour, n", [(0, 364), (19, 364), (8, 728)])
def test_lassocv_on_lear_shaped_rows(hour, n):
    """The DE fixture's 247 Lago columns (prices, the two exogenous series, weekday dummies)."""
    x, y, x0 = lear_rows(hour, n)
    delta, flip = compare_lassocv(x, y, x0, StandardScaler.get_scalable_mask(x), True)
    assert flip is None and delta <= LASSO_TOL, (delta, flip)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("scale_binary", [False, True])
def test_lassocv_on_realistic_data_with_rows_left_out(fit_intercept, scale_binary):
    """Spiky prices, MW loads, night-time solar mostly 0, dummies (one 1 only in ρ's last block, so constant on a whole training
    part), ρ a strict subset of the span (missing targets and a gappy weather column)."""
    rng = np.random.default_rng(4 + 2 * fit_intercept + scale_binary)
    for _ in range(6):
        n = int(rng.integers(80, 700))
        x, binary = realistic_columns(rng, n + 1)
        late = (np.arange(n + 1) >= n - n // 7).astype(float)
        x = np.column_stack([x, late, rng.normal(size=(n + 1, 4)) * [1, 10, 100, 1000]])
        binary = np.r_[binary, True, np.zeros(4, dtype=bool)]
        y = 50 + 0.2 * x[:, 0] + 2 * x[:, 3] + 5 * late + x[:, 7:] @ (rng.normal(size=4) / np.array([1, 10, 100, 1000]))
        y = y + rng.normal(size=n + 1) * 5
        xs, ys, x0 = x[:-1].copy(), y[:-1].copy(), x[-1:]
        ys[rng.random(n) < 0.04] = np.nan
        xs[rng.random(n) < 0.03, 5] = np.nan
        rows = np.flatnonzero(np.isfinite(ys) & np.isfinite(xs).all(axis=1))
        delta, flip = compare_lassocv(
            xs[rows], ys[rows], x0, ~binary, fit_intercept, rows, xs, ys, scale_binary=scale_binary
        )
        assert flip is None and delta <= LASSO_TOL, (delta, flip)


def test_lassocv_from_rows_no_scaler_centred():
    """`ScaledStats` whose z̄ and ȳ are far from 0 (what a VST without standardisation hands in): `LassoCV` on the rows as they
    are, with and without an intercept. The grid is the window's centred q with an intercept and Z'y without."""
    rng = np.random.default_rng(5)
    for i in range(12):
        n, p = int(rng.integers(60, 300)), int(rng.integers(2, 12))
        Z = rng.normal(size=(n + 1, p)) * rng.uniform(0.5, 3, size=p) + rng.uniform(-3, 3, size=p)
        y = 2.0 + Z @ (rng.normal(size=p) * (rng.random(p) < 0.6)) + rng.normal(size=n + 1)
        stats = rows_stats(Z[:-1], y[:-1])
        for fit_intercept in (True, False):
            fm = LassoCV(cv=5, max_iter=MAX_ITER, n_jobs=1, fit_intercept=fit_intercept).fit(Z[:-1], y[:-1])
            fe = _cv_stats.lassocv_fit(stats, fit_intercept, MAX_ITER)
            assert position(fm.alphas_, fe.alpha_) == position(fm.alphas_, fm.alpha_), (i, fit_intercept)
            assert abs(fe.predict(Z[-1:])[0] - fm.predict(Z[-1:])[0]) <= 1e-10, (i, fit_intercept)


def test_ties_of_the_cv_error_go_to_the_first_penalty(monkeypatch):
    """`LinearModelCV.fit` takes `np.argmin` of the folds' mean error: of tied penalties the first (largest). Sums that pool to
    the window give no exact tie (with equal blocks the training parts' covariances average 1 - B/16S of the window's, B its
    between-block part, above the grid's second point 0.933 wherever the within-block part doesn't oppose B), so the curve is
    set: a tie at positions 10 and 20, and a flat curve."""
    rng = np.random.default_rng(13)
    x, y, x0 = random_design(rng)
    stats, _ = engine_stats(x, y, np.arange(len(y)), None)
    grid = _cv_stats.alpha_grid(stats.q, stats.n)
    curve = np.linspace(2.0, 1.0, 100)
    curve[[10, 20]] = 0.5
    for mse, first in ((np.tile(curve, (5, 1)), 10), (np.ones((5, 100)), 0)):
        monkeypatch.setattr(_cv_stats, "cv_mse", lambda *args, mse=mse, **kwargs: mse)
        assert LassoCVModel([])._fit_stats(stats).alpha_ == grid[first]


def test_the_grid_is_lassocvs():
    rng = np.random.default_rng(6)
    for fit_intercept in (True, False):
        x, y, x0 = random_design(rng)
        Z, ys, _, _ = scaled_rows(x, y, x0, None)
        stats, _ = engine_stats(x, y, np.arange(len(y)), None)
        b = stats.q if fit_intercept else stats.uncentred()[1]
        grid = _cv_stats.alpha_grid(b, stats.n)
        fitted = LassoCV(cv=5, max_iter=MAX_ITER, n_jobs=1, fit_intercept=fit_intercept).fit(Z, ys)
        assert grid.shape == fitted.alphas_.shape == (100,)
        assert np.abs(grid / fitted.alphas_ - 1.0).max() <= 1e-13
    assert (_cv_stats.alpha_grid(np.zeros(3), 10) == np.finfo(np.float64).resolution).all()


def test_a_shape_only_x_gives_the_same_path():
    """With a Gram and `check_input=False`, `enet_path` reads X only for its shape and y only for y·y: the training rows, a
    NaN-filled array and the module's broadcast view give bitwise the same coefficients, and so does the one-value y."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 9)) * [1, 2, 3, 1, 1, 5, 1, 1, 1]
    y = X @ rng.normal(size=9) + rng.normal(size=200)
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    G, q = np.ascontiguousarray(Xc.T @ Xc), np.ascontiguousarray(Xc.T @ yc)
    grid = _cv_stats.alpha_grid(q, 200)
    kwargs = dict(l1_ratio=1.0, alphas=grid, precompute=G, Xy=q, check_input=False, max_iter=MAX_ITER, tol=1e-4)
    want = enet_path(np.asfortranarray(Xc), yc, **kwargs)[1]
    assert enet_path(np.full((200, 9), np.nan), yc, **kwargs)[1].tobytes() == want.tobytes()
    assert enet_path(np.broadcast_to(np.nan, (200, 9)), yc, **kwargs)[1].tobytes() == want.tobytes()
    one = _cv_stats._norm_vector(float(yc @ yc))
    if float(one @ one) == float(yc @ yc):
        assert enet_path(np.full((200, 9), np.nan), one, **kwargs)[1].tobytes() == want.tobytes()
    assert _cv_stats._path(G, q, float(yc @ yc), 200, grid, MAX_ITER, 1e-4).tobytes() == (
        enet_path(np.full((200, 9), np.nan), _cv_stats._norm_vector(float(yc @ yc)), **kwargs)[1].tobytes()
    )


def test_hands_back_n_at_most_p_and_short_training_parts():
    rng = np.random.default_rng(8)
    model = LassoCVModel([])
    for n, p, fits in (
        (40, 40, False),
        (40, 50, False),
        (40, 32, False),
        (40, 31, True),
        (41, 32, False),
        (42, 32, True),
    ):
        x = rng.normal(size=(n, p))
        y = x[:, 0] + rng.normal(size=n)
        stats, _ = engine_stats(x, y, np.arange(n), np.ones(p, dtype=bool))
        # 40 rows: blocks of 8, so every training part has 32 rows; 41: the first block 9, 32 again; 42: 33
        assert (model._fit_stats(stats) is not None) == fits, (n, p)
    x = rng.normal(size=(60, 3))
    y = x[:, 0] + rng.normal(size=60)
    stats, _ = engine_stats(x, y, np.arange(60), np.ones(3, dtype=bool))
    assert model._fit_stats(stats._replace(blocks=None)) is None
    assert LassoCVModel([], cv=3)._fit_stats(stats) is None  # five blocks, three folds asked for
    assert LassoCVModel([], cv=KFold(5))._fit_stats(stats) is None
    assert LassoCVModel([], cv=None)._fit_stats(stats) is not None
    bad = stats._replace(blocks=(stats.blocks[0]._replace(yy=np.nan), *stats.blocks[1:]))
    assert model._fit_stats(bad) is None
    stats3, _ = engine_stats(x, y, np.arange(60), np.ones(3, dtype=bool), folds=3)
    delta = abs(
        LassoCVModel([], cv=3)._fit_stats(stats3).alpha_
        - LassoCV(cv=3, max_iter=MAX_ITER, n_jobs=1).fit(*scaled_rows(x, y, x[:1], np.ones(3, dtype=bool))[:2]).alpha_
    )
    assert delta <= 1e-12


class FineGridLassoCV(LassoCVModel):
    """Overrides only the estimator: a finer grid the statistics don't honour."""

    def _estimator(self):
        return LassoCV(cv=self.cv, max_iter=self.max_iter, n_jobs=1, fit_intercept=self.fit_intercept, eps=1e-5)


class ShortLassoCV(LassoCVModel):
    """Overrides only the estimator, with a setting the statistics honour: they read it from the estimator."""

    def _estimator(self):
        return LassoCV(cv=self.cv, max_iter=20, n_jobs=1, fit_intercept=self.fit_intercept)


def test_the_fit_reads_the_estimators_settings_and_hands_back_what_it_cant_honour():
    """`_fit_stats` takes cv, max_iter and fit_intercept from `_estimator()` and hands back (None) where that isn't exactly
    `LassoCV(cv, max_iter, n_jobs=1, fit_intercept)` with valid parameters: the model path raises there (a `fit_error`
    skip) or fits other settings."""
    rng = np.random.default_rng(14)
    x = rng.normal(size=(301, 6)) * 3 + 10
    y = 50 + x @ rng.normal(size=6) + rng.normal(size=301)
    xs, ys, x0 = x[:-1], y[:-1], x[-1:]
    mask = np.ones(6, dtype=bool)
    stats, params = engine_stats(xs, ys, np.arange(300), mask)
    Z, yz, z0, scaler = scaled_rows(xs, ys, x0, mask)
    assert LassoCVModel([])._fit_stats(stats) is not None
    for kwargs in (dict(max_iter=1e4), dict(max_iter=0), dict(fit_intercept=1)):
        model = LassoCVModel([], **kwargs)
        assert model._fit_stats(stats) is None, kwargs
        with pytest.raises(ValueError):  # scikit-learn's InvalidParameterError
            model._fit_model(Z, yz)
    assert FineGridLassoCV([])._fit_stats(stats) is None
    short = ShortLassoCV([])
    fitted = short._fit_model(Z, yz)
    assert position(fitted.alphas_, short._fit_stats(stats).alpha_) == position(fitted.alphas_, fitted.alpha_)
    pe, _ = engine_path(short, stats, params, x0)
    pm = scaler.inverse(float(fitted.predict(z0)[0]))
    assert abs(pe - pm) <= LASSO_TOL


def test_the_held_out_errors_and_the_fit_are_the_same_at_any_blas_thread_count():
    """The held-out errors are einsum loops, not a threaded GEMM, whose bits move with the thread count and reach the
    choice; scikit-learn's kernel on p-vectors doesn't thread. p 247, as LEAR's design."""
    rng = np.random.default_rng(15)
    n, p = 728, 247
    x = rng.normal(size=(n, p)) @ (np.eye(p) + 0.3 * rng.normal(size=(p, p)) / np.sqrt(p)) * 5 + 20
    y = 50 + x @ (rng.normal(size=p) * (rng.random(p) < 0.1)) * 0.3 + rng.normal(size=n) * 5
    stats, _ = engine_stats(x, y, np.arange(n), np.ones(p, dtype=bool))
    grid = _cv_stats.alpha_grid(stats.q, n)
    out = []
    for threads in (1, 4):
        with threadpool_limits(threads):
            out.append((_cv_stats.cv_mse(stats, grid, True, MAX_ITER), _cv_stats.lassocv_fit(stats)))
    assert out[0][0].tobytes() == out[1][0].tobytes()
    assert out[0][1].alpha_ == out[1][1].alpha_ and out[0][1].coef_.tobytes() == out[1][1].coef_.tobytes()
    assert out[0][1].intercept_ == out[1][1].intercept_


def test_cancellation_in_a_fold_and_a_constant_target_hand_back():
    """A column that barely varies on a training part far from the span's shift, and a target constant on ρ (which
    `standard_stats` lets through with the scale 1.0), raise `StatsFallback("cancellation")`: the engine sends them back."""
    rng = np.random.default_rng(9)
    n = 200
    x = rng.normal(size=(n, 3))
    x[:, 1] = 1e4 + 1e-3 * rng.normal(size=n)
    x[-40:, 1] = rng.normal(size=40) * 50  # the last block (and the span's shift) far from the rest
    y = 50 + x[:, 0] + rng.normal(size=n)
    with pytest.raises(_stats.StatsFallback, match="cancellation"):
        engine_stats(x, y, np.arange(n), np.ones(3, dtype=bool))
    ys = np.r_[np.full(n - 10, 42.0), 50 + rng.normal(size=10)]
    rows = np.arange(n - 10)  # the target is constant on ρ, not on the span
    with pytest.raises(_stats.StatsFallback, match="target"):
        engine_stats(rng.normal(size=(n, 3)), ys, rows, np.ones(3, dtype=bool))


# --- LEAR "cv" ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [364, 728, 1449])
def test_lear_cv_from_blocks_same_position_within_1e_9(n):
    for hour in (0, 19):
        x, y, x0 = lear_rows(hour, n)
        mask = StandardScaler.get_scalable_mask(x)
        model = LEARModel([], criterion="cv", scaler="standard")
        pm, fm = model_lear(model, x, y, x0, mask)
        stats, params = engine_stats(x, y, np.arange(n), mask)
        pe, fe = engine_path(model, stats, params, x0)
        assert fe.kink == fm.kink, (hour, n)
        assert abs(pe - pm) <= LEAR_TOL, (hour, n, pe, pm)


def test_lear_cv_on_realistic_data_with_columns_constant_on_a_training_part():
    """A dummy that is 1 only in ρ's last block, night-time solar that is 0 outside one block, a copy and a complement: the
    folds drop the same constant columns and twins as the model path's (the blocks' exact zeros), same position, 1e-9."""
    rng = np.random.default_rng(10)
    for i in range(10):
        n = int(rng.integers(60, 700))
        x, binary = realistic_columns(rng, n + 1)
        t = np.arange(n + 1)
        late = (t >= n - n // 6).astype(float)
        dawn = np.where((t > n // 5) & (t < 2 * n // 5), rng.uniform(0, 20, size=n + 1), 0.0)
        x = np.column_stack([x, late, dawn, x[:, 0].copy(), 1.0 - x[:, 3], rng.normal(size=(n + 1, 5))])
        y = 50 + 0.3 * x[:, 0] + 4 * late + 0.5 * dawn + x[:, -5:] @ rng.normal(size=5) + rng.normal(size=n + 1) * 5
        xs, ys, x0 = x[:-1].copy(), y[:-1].copy(), x[-1:]
        ys[rng.random(n) < 0.03] = np.nan
        rows = np.flatnonzero(np.isfinite(ys))
        mask = StandardScaler.get_scalable_mask(xs[rows])
        model = LEARModel([], criterion="cv", scaler="standard")
        pm, fm = model_lear(model, xs[rows], ys[rows], x0, mask)
        stats, params = engine_stats(xs, ys, rows, mask)
        pe, fe = engine_path(model, stats, params, x0)
        assert fe.kink == fm.kink and abs(pe - pm) <= LEAR_TOL, (i, fe.kink, fm.kink, pe, pm)


def test_lear_cv_needs_blocks_and_says_so():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(50, 4))
    y = x @ [1.0, 0.5, 0.0, 0.0] + rng.normal(size=50) * 0.1
    stats, _ = engine_stats(x, y, np.arange(50), np.ones(4, dtype=bool))
    model = LEARModel([], criterion="cv", scaler="standard")
    with pytest.raises(TypeError, match="blocks"):
        model._fit_stats(stats._replace(blocks=None))
    a = model._fit_stats(stats)
    b = model._fit_stats(_cv_stats.cv_stats_of(stats))
    assert a.w.tobytes() == b.w.tobytes() and a.kink == b.kink
    assert LEARModel._ENGINE_NEEDS == frozenset() and LEARModel([])._ENGINE_NEEDS == frozenset()
    assert model._ENGINE_NEEDS == frozenset({"blocks"}) and LassoCVModel._ENGINE_NEEDS == frozenset({"blocks"})
    model.criterion = "bic"
    assert model._ENGINE_NEEDS == frozenset()
    assert "_ENGINE_NEEDS" not in LEARModel([], criterion="cv")._model_kwargs


# --- union independence ----------------------------------------------------------------------------------------------------


def test_every_statistic_choice_and_hand_back_is_the_fits_own():
    """The same fit from a span that holds other columns too, in another order: its blocks, penalty, coefficients and forecast
    bitwise the same, and a fit that hands back hands back with the same reason."""
    rng = np.random.default_rng(12)
    n = 400
    x, binary = realistic_columns(rng, n + 1)
    x = np.column_stack([x, (np.arange(n + 1) >= 340).astype(float)])
    binary = np.r_[binary, True]
    y = 50 + 0.2 * x[:, 0] + 3 * x[:, 3] + 2 * x[:, 6] + rng.normal(size=n + 1) * 5
    xs, ys, x0 = x[:-1].copy(), y[:-1].copy(), x[-1:]
    ys[rng.random(n) < 0.05] = np.nan
    rows = np.flatnonzero(np.isfinite(ys))
    own, own_params = engine_stats(xs, ys, rows, ~binary)
    union = np.column_stack([rng.normal(size=(n, 9)) * 1e3, xs, rng.lognormal(size=(n, 11)), np.zeros((n, 2))])
    perm = rng.permutation(union.shape[1])
    union = union[:, perm]
    where = np.argsort(perm)[9 : 9 + xs.shape[1]]
    other, other_params = engine_stats(union, ys, rows, ~binary, cols=where)
    for a, b in zip(own.blocks, other.blocks):
        for field in ("zbar", "G", "q"):
            assert getattr(a, field).tobytes() == getattr(b, field).tobytes()
        assert (a.n, a.ybar, a.yy) == (b.n, b.ybar, b.yy)
    for model in (
        LassoCVModel([]),
        LassoCVModel([], fit_intercept=False),
        LEARModel([], criterion="cv", scaler="standard"),
    ):
        pa, fa = engine_path(model, own, own_params, x0)
        pb, fb = engine_path(model, other, other_params, x0)
        assert pa == pb
        if isinstance(model, LassoCVModel):
            assert fa.alpha_ == fb.alpha_ and fa.coef_.tobytes() == fb.coef_.tobytes()
        else:
            assert fa.kink == fb.kink and fa.w.tobytes() == fb.w.tobytes()
    # A fit that hands back does so from its own data whatever the span holds besides.
    near = xs.copy()
    near[:, 4] = 1e4 + 1e-3 * rng.normal(size=n)
    near[rows[-60:], 4] = rng.normal(size=60) * 50
    reasons = []
    for span, cols in ((near, np.arange(7)), (np.column_stack([union, near]), union.shape[1] + np.arange(7))):
        with pytest.raises(_stats.StatsFallback) as raised:
            engine_stats(span, ys, rows, ~binary, cols=cols)
        reasons.append(str(raised.value))
    assert reasons[0] == reasons[1]
