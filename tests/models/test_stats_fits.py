"""`models/_stats.py`: `StandardScaler`'s map from exact span sums, and the OLS, Ridge and LEAR fits that read it, against the
model path's own scaler and estimators on the same rows (no pipeline).

The model path of a fit is `StandardScaler.fit_transform` on its rows, `_fit_model`, `_predict_fitted` on the scaled issue row
and `inverse`; the engine's is `SpanSums(span).over(ρ)`, `standard_stats`, `_fit_stats`, `_predict_fitted` on
`StandardParams.transform(row)` and `StandardParams.inverse`. Tolerances are in target units (targets of about 50): OLS 1e-8,
Ridge and LEAR 1e-9, as the step-5 spec's table. Larger sweeps, the timings and the mutants are in
`epftoolbox2-research/library_check_step5/stats_fits/`.
"""

import functools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import RidgeCV
from threadpoolctl import threadpool_limits

from epftoolbox2.models import LEARModel, OLSModel, RidgeModel
from epftoolbox2.models import _lars, _stats
from epftoolbox2.models._sums import SpanSums
from epftoolbox2.models.ridge import DEFAULT_ALPHAS
from epftoolbox2.scalers.base import mean_std
from epftoolbox2.scalers.standard import StandardScaler

DE = Path(__file__).resolve().parents[1] / "fixtures" / "lago" / "data_DE.parquet"
OLS_TOL = 1e-8
RIDGE_TOL = 1e-9
LEAR_TOL = 1e-9


@pytest.fixture(autouse=True)
def one_blas_thread():
    with threadpool_limits(1):
        yield


# --- the two paths -----------------------------------------------------------------------------------------------------


def model_path(model, x, y, x0, mask, scale_binary=False):
    """The model path's forecast (target units), its fit and its scaled rows."""
    scaler = StandardScaler(scale_binary=scale_binary)
    mask = None if mask is None else np.asarray(mask, dtype=bool)
    Z, ys, z0 = scaler.fit_transform(x.copy(), y.copy(), x0.copy(), scalable_mask=mask)
    fitted = model._fit_model(Z, ys)
    pred, _ = model._predict_fitted(fitted, z0)
    return scaler.inverse(float(pred)), fitted, (Z, ys, z0, scaler)


def stats_of(xs, ys, rows, mask, scale_binary=False, cols=None):
    """`standard_stats` of the span (`xs`, `ys`) over `rows` (indices) and `cols`."""
    cols = np.arange(xs.shape[1]) if cols is None else np.asarray(cols)
    raw = SpanSums(xs, ys).over(rows, cols)
    return _stats.standard_stats(raw, (xs[rows][:, cols], ys[rows]), mask, scale_binary)


def engine_path(model, stats, params, x0):
    """The engine's forecast (target units) and fit, or (None, None) where `_fit_stats` hands it back."""
    fitted = model._fit_stats(stats)
    if fitted is None:
        return None, None
    pred, _ = model._predict_fitted(fitted, params.transform(x0))
    return params.inverse(float(pred)), fitted


def ulps(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.where(a == b, 0.0, np.abs(a - b) / np.spacing(np.maximum(np.abs(a), np.abs(b))))


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
    """Scaled-like rows with singular values log-spaced from 1 to 1/cond over every direction, randomly rotated, then put in
    raw units with offsets; the target about 50."""
    Q, _ = np.linalg.qr(rng.normal(size=(n, p)))
    W, _ = np.linalg.qr(rng.normal(size=(p, p)))
    s = np.logspace(0, -np.log10(cond), p) if p > 1 else np.ones(1)
    Z = Q @ np.diag(s) @ W.T * np.sqrt(n)
    scale = 10 ** rng.uniform(-1, 4, size=p)
    x = Z * scale + rng.uniform(-1, 1, size=p) * scale * rng.uniform(0, 5, size=p)
    y = 50 + x @ (rng.normal(size=p) / scale) * 5 + rng.normal(size=n) * 5
    return x, y


def mixed_design(rng, n, p, cond):
    """Realistic columns (spikes, MW loads, solar, dummies) and Gaussian ones, near-duplicates at relative noise 1/cond."""
    cols = []
    for j in range(p):
        kind = j % 4
        if kind == 0:
            col, _ = realistic_columns(rng, n)
            cols.append(col[:, j // 4 % 6])
        else:
            cols.append(rng.normal(size=n) * 10 ** rng.uniform(-2, 4) + rng.uniform(-100, 100))
    x = np.column_stack(cols)
    if p > 2 and cond > 10:
        j = int(rng.integers(1, p))
        src = int(rng.integers(0, j))
        if np.std(x[:, src]) > 0 and not np.array_equal(np.unique(x[:, src]), [0.0, 1.0]):
            x[:, j] = x[:, src] + np.std(x[:, src]) * rng.normal(size=n) / cond
    y = 50 + x @ (rng.normal(size=p) / np.maximum(np.std(x, axis=0), 1e-12)) * 3 + rng.normal(size=n) * 5
    return x, y


def random_design(rng, max_cond=1e4):
    p = int(np.exp(rng.uniform(np.log(1), np.log(60))))
    n = int(rng.integers(p + 2, 4 * p + 300))
    cond = 10 ** rng.uniform(0, np.log10(max_cond))
    make = harsh_design if rng.random() < 0.5 else mixed_design
    x, y = make(rng, n + 1, p, cond)
    return x[:-1], y[:-1], x[-1:]


def designs(part):
    """The 100 random designs of one part of the OLS and Ridge tests (the same designs for both)."""
    rng = np.random.default_rng(1000 + part)
    return [random_design(rng) for _ in range(100)]


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


def lear_rows(hour, n, drop_sunday=False):
    x, y = lear_design(hour)
    if drop_sunday:
        x = x[:, :-1]
    return x[-n - 1 : -1], y[-n - 1 : -1], x[-1:]


# --- standard_stats against StandardScaler + gram_stats ------------------------------------------------------------------


@pytest.mark.parametrize("n", [365, 1449, 9000, 40_000])
@pytest.mark.parametrize("scale_binary", [False, True])
def test_means_and_scales_within_2_ulp_of_the_scalers(n, scale_binary):
    rng = np.random.default_rng(n + scale_binary)
    for _ in range(3):
        x, binary = realistic_columns(rng, n)
        y = 50 + 0.3 * x[:, 0] + rng.normal(size=n) * 5
        y_span = y.copy()
        gone = rng.random(n) < 0.03  # rows with a missing target: ρ is a strict subset of the span
        y_span[gone] = np.nan
        rows = np.flatnonzero(~gone)
        stats, params = stats_of(x, y_span, rows, ~binary, scale_binary)
        xr, yr = x[rows], y[rows]
        scaled = np.ones(x.shape[1], dtype=bool) if scale_binary else ~binary
        assert (params.scaled == scaled).all()
        for j in range(x.shape[1]):
            if scaled[j]:
                mean, std, constant = mean_std(xr[:, j])
                assert ulps(params.mean[j], mean) <= 2 and ulps(params.scale[j], std) <= 2, j
                assert params.constant[j] == constant
            else:
                assert params.mean[j] == 0.0 and params.scale[j] == 1.0
                assert stats.zbar[j] == pytest.approx(xr[:, j].mean(), rel=1e-15, abs=0)
        ymean, ystd, _ = mean_std(yr)
        assert ulps(params.ymean, ymean) <= 2 and ulps(params.yscale, ystd) <= 2


@pytest.mark.parametrize("scale_binary", [False, True])
def test_the_scaled_sums_are_gram_stats_of_the_scaled_rows(scale_binary):
    """G, q, yy, z̄ against `gram_stats` of `fit_transform`'s rows, relative to the scale of each element; the issue row's
    transform and the inverse against the scaler's."""
    rng = np.random.default_rng(11)
    for n in (365, 2000):
        x, binary = realistic_columns(rng, n)
        y = 50 + 0.3 * x[:, 0] + rng.normal(size=n) * 5
        x0 = x[-1:]
        x, y = x[:-1], y[:-1]
        stats, params = stats_of(x, y, np.arange(n - 1), ~binary, scale_binary)
        scaler = StandardScaler(scale_binary=scale_binary)
        Z, ys, z0 = scaler.fit_transform(x.copy(), y.copy(), x0.copy(), scalable_mask=~binary)
        g = _lars.gram_stats(Z, ys)
        d = np.sqrt(np.diag(g.G))
        scale = np.outer(d, d)
        assert np.abs(stats.G - g.G).max() <= 1e-13 * scale.max()
        assert (np.abs(stats.G - g.G) <= 1e-13 * scale).all()
        assert (np.abs(stats.q - g.q) <= 1e-13 * d * np.sqrt(g.yy)).all()
        assert stats.yy == pytest.approx(g.yy, rel=1e-13)
        assert np.abs(stats.zbar - g.xbar).max() <= 1e-14
        assert abs(stats.ybar - g.ybar) <= 1e-14
        # The issue row: the scaler's formula on the fit's parameters, so within what their last-bit differences carry.
        model_mean = np.array([mean_std(x[:, k])[0] if params.scaled[k] else 0.0 for k in range(x.shape[1])])
        model_scale = np.array([mean_std(x[:, k])[1] if params.scaled[k] else 1.0 for k in range(x.shape[1])])
        z = params.transform(x0)
        carried = (np.abs(params.mean - model_mean) + np.abs(z) * np.abs(params.scale - model_scale)) / model_scale
        assert (np.abs(z - z0) <= carried + 2 * np.spacing(np.abs(z0))).all()
        # The inverse is the scaler's formula on the fit's parameters: bitwise with the same ones, and against the model path's
        # within 1 ulp beyond what the parameters' own last-bit differences carry.
        same = StandardScaler()
        same._target_mean, same._target_std = params.ymean, params.yscale
        v = rng.normal(size=500) * 3
        assert np.array_equal(params.inverse(v), same.inverse(v))
        carried = np.abs(v) * abs(params.yscale - scaler._target_std) + abs(params.ymean - scaler._target_mean)
        gap = np.abs(params.inverse(v) - scaler.inverse(v))
        assert (gap <= carried + np.spacing(np.abs(v) * params.yscale + abs(params.ymean))).all()


def test_values_as_a_callable_give_the_same_bits_as_arrays():
    rng = np.random.default_rng(3)
    x, binary = realistic_columns(rng, 500)
    x[:, 4] = 0.3  # a constant column, so a candidate reads its values
    y = 50 + rng.normal(size=500)
    raw = SpanSums(x, y).over(np.arange(500), np.arange(6))
    a, pa = _stats.standard_stats(raw, (x, y), ~binary)
    b, pb = _stats.standard_stats(raw, lambda j: y if j is None else x[:, j], ~binary)
    for f in ("zbar", "G", "q"):
        assert getattr(a, f).tobytes() == getattr(b, f).tobytes()
    assert (a.yy, a.ybar, a.n) == (b.yy, b.ybar, b.n)
    for f in pa._fields:
        assert np.asarray(getattr(pa, f)).tobytes() == np.asarray(getattr(pb, f)).tobytes()
    for za, zb in zip(a.rows(), b.rows()):
        assert za.tobytes() == zb.tobytes()


def test_the_statistics_are_the_same_whatever_else_the_span_holds():
    """A fit's statistics are a function of its own columns and rows: the same bits from a span of 4 columns or of 40, in any
    order (the exact sums' property, carried through the map)."""
    rng = np.random.default_rng(5)
    x, binary = realistic_columns(rng, 800)
    y = 50 + rng.normal(size=800) * 5
    rows = np.flatnonzero(rng.random(800) < 0.9)
    own = stats_of(x[:, :4], y, rows, ~binary[:4])
    union = np.column_stack([rng.normal(size=(800, 20)), x[:, :4], rng.normal(size=(800, 16)) * 1e3])
    perm = rng.permutation(40)
    union = union[:, perm]
    where = np.argsort(perm)[20:24]
    other = stats_of(union, y, rows, ~binary[:4], cols=where)
    for f in ("zbar", "G", "q"):
        assert getattr(own[0], f).tobytes() == getattr(other[0], f).tobytes()
    assert own[0].yy == other[0].yy
    fit = OLSModel([], fit_intercept=True)._fit_stats(own[0])
    assert fit.coef_.tobytes() == OLSModel([], fit_intercept=True)._fit_stats(other[0]).coef_.tobytes()


# --- constant columns: the model path's flag, from the sums plus `mean_std` on the candidates ----------------------------


def _constant_case(rng, n, kind):
    """(x span, y span, ρ, the column's index, whether the column is 0/1): one column constant on ρ, beside two ordinary."""
    base = np.column_stack([rng.normal(size=n) * 8 + 50, np.round(21000 + rng.normal(size=n) * 800, 1)])
    y = 50 + 0.5 * base[:, 0] + rng.normal(size=n) * 5
    out = rng.random(n) < 0.3  # the rows outside ρ: a missing target
    if kind == "noise":
        col = 0.3 + rng.normal(size=n) * 1e-15
        out[:] = False
    elif kind == "dummy_zero_on_rho":
        col = out.astype(float)  # 1 exactly where the target is missing: 0 on ρ, span mean about 0.3
    elif kind == "dummy_one_on_rho":
        col = (~out).astype(float)
    elif kind == "point3_on_rho":
        col = np.where(out, 0.7, 0.3)
    elif kind == "five_on_rho":
        col = np.where(out, 7.0, 5.0)
    else:
        raise ValueError(kind)
    y_span = y.copy()
    y_span[out] = np.nan
    x = np.column_stack([base[:, 0], col, base[:, 1]])
    return x, y_span, np.flatnonzero(~out), 1, kind.startswith("dummy")


CONSTANT_KINDS = ["noise", "dummy_zero_on_rho", "dummy_one_on_rho", "point3_on_rho", "five_on_rho"]


@pytest.mark.parametrize("kind", CONSTANT_KINDS)
@pytest.mark.parametrize("scale_binary", [False, True])
def test_constant_flags_are_the_model_paths(kind, scale_binary):
    """Constant on ρ, at n 365 to 40,000: the flag is `mean_std`'s, the scaled issue row stays O(1) where the column differs
    from ρ's value (z ≈ 1e8 there if the sums' noise were taken for a spread), and OLS with or without an intercept and LEAR
    forecast as the model path does."""
    rng = np.random.default_rng(CONSTANT_KINDS.index(kind) * 2 + scale_binary)
    for n in (365, 1000, 4000, 9000, 40_000):
        x, y_span, rows, j, binary = _constant_case(rng, n, kind)
        mask = np.array([True, not binary, True])
        stats, params = stats_of(x, y_span, rows, mask, scale_binary)
        xr, yr = x[rows], y_span[rows]
        assert params.constant[j], (n, kind)
        if params.scaled[j]:
            mean, std, constant = mean_std(xr[:, j])
            assert constant and params.mean[j] == mean and params.scale[j] == 1.0
        assert (stats.G[j] == 0).all() and (stats.G[:, j] == 0).all() and stats.q[j] == 0
        x0 = x[:1].copy()
        x0[0, j] = 1.0 if binary else 0.7
        assert np.abs(params.transform(x0)).max() < 1e3
        for model in (
            OLSModel([], fit_intercept=True),
            OLSModel([], fit_intercept=False),
            LEARModel([], scaler="standard"),
        ):
            pm, fm, _ = model_path(model, xr, yr, x0, mask, scale_binary)
            pe, fe = engine_path(model, stats, params, x0)
            assert pe is not None, (n, kind, model)
            assert abs(pe - pm) <= OLS_TOL, (n, kind, model, pe, pm)


@pytest.mark.parametrize("scale_binary", [False, True])
def test_np_full_and_noise_columns_are_constant(scale_binary):
    """`np.full(30, 0.3)` (numpy's std 5.6e-17) and 0.3 + 1e-15 noise: constant, as `mean_std` says, at n 30."""
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.normal(size=30) * 3 + 10, np.full(30, 0.3), 0.3 + rng.normal(size=30) * 1e-15])
    y = 50 + x[:, 0] + rng.normal(size=30)
    stats, params = stats_of(x, y, np.arange(30), np.ones(3, dtype=bool), scale_binary)
    assert params.constant.tolist() == [mean_std(x[:, k])[2] for k in range(3)] == [False, True, True]
    x0 = np.array([[10.0, 0.31, 0.31]])
    pm, _, _ = model_path(OLSModel([]), x, y, x0, np.ones(3, dtype=bool), scale_binary)
    pe, _ = engine_path(OLSModel([]), stats, params, x0)
    assert abs(pe - pm) <= OLS_TOL


def test_the_sums_noise_on_a_dummy_constant_on_rho_is_what_the_candidate_rule_catches():
    """The 5d amendment's case: a 0/1 dummy that is 0 on ρ with span mean about 0.3. Its variance from the sums is rounding
    noise, some of it negative and some above 1e-9 of the column's scale (so a threshold of 1e-9 would take it for a spread),
    all of it far below `CANDIDATE_RTOL`; the rule makes every such column a candidate."""
    rng = np.random.default_rng(2)
    above, negative = 0, 0
    for n in (365, 500, 1000, 2000, 4000, 9000, 20_000, 40_000):
        for _ in range(4):
            x, y_span, rows, j, _ = _constant_case(rng, n, "dummy_zero_on_rho")
            raw = SpanSums(x, y_span).over(rows, np.arange(3))
            r = raw.n
            var = (raw.S[j, j] - raw.u[j] ** 2 / r) / (r - 1)
            rms = np.sqrt(raw.S[j, j] / r)
            above += bool(var > 0 and np.sqrt(var) > 1e-9 * rms)
            negative += bool(var < 0)
            assert not var > 0 or np.sqrt(var) <= _stats.CANDIDATE_RTOL * rms
            assert _stats._candidates(
                np.array([var]), np.array([raw.shift[j] + raw.u[j] / r]), np.array([raw.S[j, j]]), r
            )[0]
    assert above > 0 and negative > 0


# --- precision: columns the sums can't scale ---------------------------------------------------------------------------


def test_a_column_far_from_its_span_mean_relative_to_its_spread_goes_back():
    """A column 0.3 ± 1e-5 on ρ and 0.7 elsewhere: not constant, but S - u²/r cancels (κ about 1e8, a relative error of about
    eps·κ in its row of G); `StatsFallback`. With a spread of 1e-2 (κ about 1e2) it is scaled from the sums and OLS is within
    tolerance. (Measured, a column that carries signal: OLS off by 4.7e-11 at κ up to 3e4, 1e-9 at 3e5, 8.5e-7 at 4e8.)"""
    rng = np.random.default_rng(4)
    n = 2000
    base = rng.normal(size=n) * 8 + 50
    out = rng.random(n) < 0.3
    y = 50 + 0.5 * base + rng.normal(size=n) * 5
    y[out] = np.nan
    rows = np.flatnonzero(~out)
    for spread, falls_back in ((1e-5, True), (1e-2, False)):
        col = np.where(out, 0.7, 0.3 + rng.normal(size=n) * spread)
        x = np.column_stack([base, col])
        if falls_back:
            with pytest.raises(_stats.StatsFallback) as info:
                stats_of(x, y, rows, np.ones(2, dtype=bool))
            assert info.value.reason == "cancellation"
            continue
        stats, params = stats_of(x, y, rows, np.ones(2, dtype=bool))
        pm, _, _ = model_path(OLSModel([]), x[rows], y[rows], x[:1], np.ones(2, dtype=bool))
        pe, _ = engine_path(OLSModel([]), stats, params, x[:1])
        assert abs(pe - pm) <= OLS_TOL


def test_fewer_than_three_rows_raise():
    x = np.array([[1.0, 2.0], [3.0, 5.0], [4.0, 4.0]])
    raw = SpanSums(x, np.array([1.0, 2.0, 4.0])).over(np.arange(2), np.arange(2))
    with pytest.raises(ValueError, match="at least 3 rows"):
        _stats.standard_stats(raw, (x[:2], np.array([1.0, 2.0])), None)


# --- OLS -----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("part", range(5))
def test_ols_within_1e_8_of_linear_regression_on_random_designs(part):
    """500 random designs (100 per part), n > p + 1, cond(Z) up to 1e4, with and without an intercept: every fit the engine
    keeps is within 1e-8 of `LinearRegression`; it hands back only fits with cond(Z) above √GUARD (the pivot ratio is at most
    cond(Z)²), and keeps most."""
    kept = 0
    for x, y, x0 in designs(part):
        mask = StandardScaler.get_scalable_mask(x)
        stats, params = stats_of(x, y, np.arange(x.shape[0]), mask)
        for fit_intercept in (False, True):
            model = OLSModel([], fit_intercept=fit_intercept)
            pm, fm, (Z, *_) = model_path(model, x, y, x0, mask)
            pe, fe = engine_path(model, stats, params, x0)
            if pe is None:
                Zc = Z - Z.mean(axis=0) if fit_intercept else Z
                live = np.abs(Zc).max(axis=0) > 0
                assert np.linalg.cond(Zc[:, live]) > 0.99 * np.sqrt(_stats.GUARD)
                continue
            kept += 1
            assert abs(pe - pm) <= OLS_TOL, (x.shape, fit_intercept, pe, pm)
            assert fe.coef_.shape == fm.coef_.shape
    assert kept >= 140


@pytest.mark.parametrize("n", [364, 728, 1449])
def test_ols_on_lear_shaped_rows(n):
    """p 247 of the DE fixture (cond(Z) 1.7e3 to 6.2e3): within 1e-8, without an intercept and with one beside six weekday
    dummies; with one beside all seven (a one-hot that sums to the intercept) the fit goes back."""
    for hour in (0, 12, 19):
        for fit_intercept, drop in ((False, False), (True, True)):
            x, y, x0 = lear_rows(hour, n, drop_sunday=drop)
            mask = StandardScaler.get_scalable_mask(x)
            model = OLSModel([], fit_intercept=fit_intercept)
            pm, _, _ = model_path(model, x, y, x0, mask)
            stats, params = stats_of(x, y, np.arange(n), mask)
            pe, _ = engine_path(model, stats, params, x0)
            assert pe is not None and abs(pe - pm) <= OLS_TOL, (hour, n, fit_intercept, pe, pm)
        x, y, x0 = lear_rows(hour, n)
        stats, _ = stats_of(x, y, np.arange(n), StandardScaler.get_scalable_mask(x))
        assert OLSModel([], fit_intercept=True)._fit_stats(stats) is None


def test_ols_hands_back_rank_problems_and_short_windows():
    """None for n <= p + 1, an exact duplicate and a 1e-6 near-duplicate (engine report `check_rank.py`'s cases), on the
    LEAR rows and on a small design; a 1e-1 near-duplicate is kept and within tolerance (a 1e-3 one, cond(Z) about 1e3, can
    go either way at `GUARD` 1e5)."""
    rng = np.random.default_rng(6)
    x, y, x0 = lear_rows(12, 728)
    src = x[:, 12]  # price(t-1, 12)
    for extra, kept in ((src.copy(), False), (src + 1e-6 * np.std(src) * rng.normal(size=src.size), False)):
        xe = np.column_stack([x, extra])
        mask = StandardScaler.get_scalable_mask(xe)
        stats, _ = stats_of(xe, y, np.arange(xe.shape[0]), mask)
        for fit_intercept in (False, True):
            assert OLSModel([], fit_intercept=fit_intercept)._fit_stats(stats) is None
    small = rng.normal(size=(40, 5)) * [1, 10, 100, 1e3, 1e4] + 50
    ys = small @ [1, 0.1, 0.01, 1e-3, 1e-4] + rng.normal(size=40)
    for extra, kept in (
        (small[:, 1].copy(), False),
        (small[:, 1] + 1e-6 * np.std(small[:, 1]) * rng.normal(size=40), False),
        (small[:, 1] + 1e-1 * np.std(small[:, 1]) * rng.normal(size=40), True),
    ):
        xe = np.column_stack([small, extra])
        mask = np.ones(6, dtype=bool)
        stats, params = stats_of(xe, ys, np.arange(40), mask)
        for fit_intercept in (False, True):
            model = OLSModel([], fit_intercept=fit_intercept)
            pe, _ = engine_path(model, stats, params, xe[:1])
            if not kept:
                assert pe is None
                continue
            pm, _, _ = model_path(model, xe, ys, xe[:1], mask)
            assert abs(pe - pm) <= OLS_TOL
    for n in (4, 6, 7):  # p = 5: n <= p + 1 goes back, n = 7 is kept
        stats, _ = stats_of(small[:n], ys[:n], np.arange(n), np.ones(5, dtype=bool))
        assert (OLSModel([])._fit_stats(stats) is None) == (n <= 6)


def test_conditioning_sweep_every_fit_the_guard_lets_through_is_within_1e_8():
    """cond(Z) from 1e2 to 1e7 (the research sweep has 4,500 more and the LEAR rows): every fit kept within 1e-8; the guard
    hands back every fit whose (largest/smallest pivot)² exceeds `GUARD`, and without it some fit is off by more."""
    rng = np.random.default_rng(77)
    worst_kept, worst_unguarded = 0.0, 0.0
    for i in range(160):
        p = int(rng.integers(2, 40))
        n = int(rng.integers(p + 2, 4 * p + 200))
        cond = 10 ** rng.uniform(2, 7)
        x, y = (harsh_design if i % 2 else mixed_design)(rng, n + 1, p, cond)
        x, y, x0 = x[:-1], y[:-1], x[-1:]
        mask = StandardScaler.get_scalable_mask(x)
        stats, params = stats_of(x, y, np.arange(n), mask)
        for fit_intercept in (False, True):
            pm, _, _ = model_path(OLSModel([], fit_intercept=fit_intercept), x, y, x0, mask)
            fit = _stats.ols_fit(stats, fit_intercept)
            free = _stats.ols_fit(stats, fit_intercept, guard=np.inf)
            if free is not None:
                worst_unguarded = max(
                    worst_unguarded, abs(params.inverse(float(free.predict(params.transform(x0))[0])) - pm)
                )
            if fit is not None:
                worst_kept = max(worst_kept, abs(params.inverse(float(fit.predict(params.transform(x0))[0])) - pm))
    assert worst_kept <= OLS_TOL
    assert worst_unguarded > OLS_TOL


# --- Ridge ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("part", range(5))
def test_ridge_same_penalty_and_within_1e_9_of_ridgecv(part):
    """The OLS test's 500 random designs, the default 36-value grid and one-value grids, with and without an intercept: the
    same penalty as `RidgeCV(alphas, fit_intercept, cv=None)` in every fit, and the forecast within 1e-9 in every fit the
    engine keeps. It hands back the fits where scikit-learn's own rounding at the chosen penalty, `ridgecv_rounding / α` in
    target units, exceeds `RIDGE_ROUNDING_MAX`: there RidgeCV is up to about 4e-8/α off an exact ridge (research sweep), the
    engine is not."""
    rng = np.random.default_rng(2000 + part)
    kept = 0
    for i, (x, y, x0) in enumerate(designs(part)):
        mask = StandardScaler.get_scalable_mask(x)
        stats, params = stats_of(x, y, np.arange(x.shape[0]), mask)
        alphas = None if i % 2 else [float(DEFAULT_ALPHAS[int(rng.integers(0, 36))])]
        fit_intercept = bool(i % 4 < 2)
        model = RidgeModel([], alphas=alphas, fit_intercept=fit_intercept)
        pm, fm, _ = model_path(model, x, y, x0, mask)
        free = _stats.ridge_fit(stats, model._model_kwargs["alphas"], fit_intercept, rounding_max=np.inf)
        assert free.alpha_ == fm.alpha_
        pe, fe = engine_path(model, stats, params, x0)
        Z, z = stats.rows()
        rounding = (
            _stats.ridgecv_rounding(Z - stats.zbar, z - stats.ybar) if fit_intercept else _stats.ridgecv_rounding(Z, z)
        )
        if rounding * params.yscale / fm.alpha_ > _stats.RIDGE_ROUNDING_MAX:
            assert fe is None
            continue
        kept += 1
        assert fe.alpha_ == fm.alpha_
        assert abs(pe - pm) <= RIDGE_TOL, (x.shape, alphas, fit_intercept, pe, pm, fm.alpha_)
    assert kept >= 40


@pytest.mark.parametrize("n", [364, 728, 1449])
def test_ridge_on_lear_shaped_rows(n):
    for hour in (0, 12, 19):
        x, y, x0 = lear_rows(hour, n)
        mask = StandardScaler.get_scalable_mask(x)
        stats, params = stats_of(x, y, np.arange(n), mask)
        for fit_intercept in (True, False):
            model = RidgeModel([], fit_intercept=fit_intercept)
            pm, fm, _ = model_path(model, x, y, x0, mask)
            pe, fe = engine_path(model, stats, params, x0)
            assert fe is not None and fe.alpha_ == fm.alpha_
            assert abs(pe - pm) <= RIDGE_TOL


def test_ridge_hands_back_n_at_most_p_and_needs_rows():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(12, 10)) * 5 + 20
    y = 50 + x @ rng.normal(size=10) + rng.normal(size=12)
    model = RidgeModel([], alphas=[10.0, 100.0])
    for n in (3, 10, 11):
        stats, _ = stats_of(x[:n], y[:n], np.arange(n), np.ones(10, dtype=bool))
        assert (model._fit_stats(stats) is None) == (n <= 10)
    assert model._fit_stats(stats._replace(rows=None)) is None
    assert model._fit_stats(stats) is not None
    # a one-value grid is RidgeCV's too
    pm, fm, _ = model_path(model, x, y, x[:1], np.ones(10, dtype=bool))
    assert RidgeCV(alphas=[10.0, 100.0], cv=None).fit(*fm_rows(x, y)).alpha_ == fm.alpha_


def test_ridge_ties_go_to_the_first_penalty_as_in_ridgecv():
    """Every column constant on the rows: no penalty changes the fit, the leave-one-out errors tie exactly, and `RidgeCV`
    keeps the first penalty of the grid (a strict improvement is needed to move on); so must the engine."""
    rng = np.random.default_rng(13)
    x = np.column_stack([np.full(40, 3.0), np.full(40, 0.25 + 1e-16)])
    y = 50 + rng.normal(size=40)
    stats, params = stats_of(x, y, np.arange(40), np.ones(2, dtype=bool))
    for fit_intercept in (True, False):
        model = RidgeModel([], alphas=[100.0, 10.0, 1000.0], fit_intercept=fit_intercept)
        pm, fm, _ = model_path(model, x, y, x[:1], np.ones(2, dtype=bool))
        pe, fe = engine_path(model, stats, params, x[:1])
        assert fm.alpha_ == fe.alpha_ == 100.0
        assert abs(pe - pm) <= RIDGE_TOL


def fm_rows(x, y):
    scaler = StandardScaler()
    Z, ys, _ = scaler.fit_transform(x.copy(), y.copy(), x[:1].copy(), scalable_mask=np.ones(x.shape[1], dtype=bool))
    return Z, ys


# --- LEAR ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [364, 728, 1449])
def test_lear_from_scaled_stats_same_kink_within_1e_9(n):
    """`LEARModel._fit_stats(ScaledStats)` (through `.gram()`) against `_fit_model` on `fit_transform`'s rows: the same kink,
    the same exact twins dropped, within 1e-9, BIC and AIC, `"lars"` and `"cd"` (seven hours in the research tree's
    `lear_check.py`)."""
    for hour in (0, 19):
        x, y, x0 = lear_rows(hour, n)
        mask = StandardScaler.get_scalable_mask(x)
        stats, params = stats_of(x, y, np.arange(n), mask)
        _, _, (Z, ys, _, _) = model_path(OLSModel([]), x, y, x0, mask)
        g = _lars.gram_stats(Z, ys)
        assert _lars.without_twins(g.G, g.q)[2] == _lars.without_twins(stats.G, stats.q)[2]
        for criterion in ("bic", "aic"):
            for solver in ("lars", "cd"):
                model = LEARModel([], criterion=criterion, solver=solver, scaler="standard")
                pm, fm, _ = model_path(model, x, y, x0, mask)
                pe, fe = engine_path(model, stats, params, x0)
                assert fe.kink == fm.kink, (hour, n, criterion, solver)
                assert abs(pe - pm) <= LEAR_TOL, (hour, n, criterion, solver, pe, pm)


def test_lear_on_realistic_designs_with_constant_and_twin_columns():
    rng = np.random.default_rng(9)
    for i in range(12):
        n = int(rng.integers(60, 700))
        x, binary = realistic_columns(rng, n + 1)
        extra = [
            x[:, 0].copy(),
            1.0 - x[:, 3],
            np.where(rng.random(n + 1) < 0.5, 0.0, 0.0),
        ]  # a copy, a complement, zeros
        x = np.column_stack([x, *extra, rng.normal(size=(n + 1, 8))])
        y = 50 + 0.3 * x[:, 0] + x[:, 6:] @ rng.normal(size=x.shape[1] - 6) + rng.normal(size=n + 1) * 5
        x, y, x0 = x[:-1], y[:-1], x[-1:]
        mask = StandardScaler.get_scalable_mask(x)
        stats, params = stats_of(x, y, np.arange(n), mask)
        for criterion, solver in (("bic", "lars"), ("aic", "cd")):
            model = LEARModel([], criterion=criterion, solver=solver, scaler="standard")
            pm, fm, _ = model_path(model, x, y, x0, mask)
            pe, fe = engine_path(model, stats, params, x0)
            assert fe.kink == fm.kink and abs(pe - pm) <= LEAR_TOL, (i, criterion, solver, pe, pm)


def test_lear_cv_needs_blocks_and_gram_stats_still_work():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(50, 4))
    y = x @ [1.0, 0.5, 0.0, 0.0] + rng.normal(size=50) * 0.1
    stats, _ = stats_of(x, y, np.arange(50), np.ones(4, dtype=bool))
    with pytest.raises(TypeError, match="blocks"):
        LEARModel([], criterion="cv", scaler="standard")._fit_stats(stats)
    model = LEARModel([], scaler="standard")
    a = model._fit_stats(stats)
    b = model._fit_stats(stats.gram())
    assert a.w.tobytes() == b.w.tobytes() and a.kink == b.kink


# --- the seam ------------------------------------------------------------------------------------------------------------


def test_engine_needs_and_the_fits_predict_as_the_model_path_reads_them():
    assert OLSModel._ENGINE_NEEDS == frozenset() and LEARModel._ENGINE_NEEDS == frozenset()
    assert RidgeModel._ENGINE_NEEDS == frozenset({"rows"})
    rng = np.random.default_rng(12)
    x = rng.normal(size=(30, 3)) * 4 + 7
    y = 50 + x @ [1.0, 2.0, -1.0] + rng.normal(size=30)
    stats, params = stats_of(x, y, np.arange(30), np.ones(3, dtype=bool))
    for model in (OLSModel([], fit_intercept=True), RidgeModel([], alphas=[10.0])):
        fit = model._fit_stats(stats)
        assert isinstance(fit, _stats.LinearFit)
        pred, coefs = model._predict_fitted(fit, params.transform(x[:1]))
        assert isinstance(coefs, list) and len(coefs) == 3
        assert pred == (params.transform(x[:1]) @ fit.coef_ + fit.intercept_)[0]
    A, b, yy = stats.uncentred()
    assert np.array_equal(A, stats.G) and np.array_equal(b, stats.q) and yy == stats.yy  # every column scaled: z̄ = 0
