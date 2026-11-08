"""`models/_cv_stats.py`: KFold's CV blocks of a fit's rows from exact span sums, and `LEARModel(criterion="cv")._fit_stats` from
them, against the model path on the same rows (no pipeline).

The model path of a fit is `StandardScaler.fit_transform` on its rows, `_fit_model` (`cv_stats` of the scaled rows),
`_predict_fitted` on the scaled issue row and `inverse`. The engine's is `SpanSums(span)`, `over(ρ)` and `over(block)` for each
of `block_rows(ρ)`, `standard_stats`, `block_stats`, `_fit_stats`, `_predict_fitted` on `StandardParams.transform(row)` and
`StandardParams.inverse`. LEAR `"cv"` within 1e-9 in target units (targets of about 50) at the same penalty position, as the
step-5 spec's table. Larger sweeps and mutants are in `epftoolbox2-research/library_check_step5/lassocv_stats/`.
"""

import functools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from epftoolbox2.models import LEARModel
from epftoolbox2.models import _cv_stats, _lars, _stats
from epftoolbox2.models._sums import SpanSums
from epftoolbox2.scalers.standard import StandardScaler

DE = Path(__file__).resolve().parents[1] / "fixtures" / "lago" / "data_DE.parquet"
LEAR_TOL = 1e-9


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
    assert model._ENGINE_NEEDS == frozenset({"blocks"})
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
    model = LEARModel([], criterion="cv", scaler="standard")
    pa, fa = engine_path(model, own, own_params, x0)
    pb, fb = engine_path(model, other, other_params, x0)
    assert pa == pb and fa.kink == fb.kink and fa.w.tobytes() == fb.w.tobytes()
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
