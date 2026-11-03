"""LEAR's kernels: `models/_lars.py` (the numpy reference) against scikit-learn, and `models/_lars_numba.py` against it.

scikit-learn 0.24.2, which Lago et al. ran, doesn't install next to the locked version, so these stand in for it:
- LARS: today's `lars_path`. Its solver differs from 0.24.2's by one line, a rounding of the equiangular correlations to 15
  decimals; `_without_rounding()` takes that line out, which leaves 0.24.2's solver. Both are compared.
- Old `LassoLarsIC`: `_lars_old_ic`, a copy of the research port's (`epfq.py:115-140`), returning its criterion too.
- Coordinate descent: `_cd_0242`, 0.24.2's `enet_coordinate_descent_gram` (`_cd_fast.pyx`) line by line in Python. Today's
  version screens features and updates `H` once per coordinate, so it stops elsewhere.

With an exact or a 1e-8 near duplicate, whether the twin is tried as an entrant turns on the last bit of a BLAS sum (OpenBLAS's
gemv rounds by the buffer's alignment), in 0.24.2 as here. Those designs are compared up to the kink where the twin's sibling
enters, and the degenerate-entrant rule itself on Gram designs of dyadic numbers, whose ties are exact.
"""

import contextlib
import math
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
import sklearn.linear_model._least_angle as least_angle
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, lars_path_gram
from sklearn.linear_model import lars_path as sklearn_lars_path

from epftoolbox2._extras import MissingExtraError
from epftoolbox2.models import _lars

EPS = np.finfo(float).eps
TWIN = 3  # the duplicated column of the duplicate designs; the twin is the last column


def _design(kind, seed):
    """X and y of one synthetic design, with 8 true coefficients and unit noise."""
    rng = np.random.default_rng(seed)
    if kind == "n>p":
        X = rng.standard_normal((200, 30))
    elif kind == "n<p":
        X = rng.standard_normal((49, 247))
    elif kind == "weekday dummies":  # the seventh is collinear with the other six after centring
        X = np.c_[rng.standard_normal((120, 8)), np.eye(7)[rng.integers(0, 7, 120)]]
    elif kind == "duplicate":
        X = rng.standard_normal((100, 12))
        X = np.c_[X, X[:, TWIN]]
    elif kind == "near duplicate":
        X = rng.standard_normal((100, 12))
        X = np.c_[X, X[:, TWIN] + 1e-8 * rng.standard_normal(100)]
    elif kind == "zero column":
        X = rng.standard_normal((100, 12))
        X[:, 5] = 0.0
    else:
        raise ValueError(kind)
    beta = np.zeros(X.shape[1])
    beta[:8] = rng.standard_normal(8)
    return X, X @ beta + rng.standard_normal(X.shape[0])


def _normalised(X, y):
    """Centred unit-norm columns and centred y, as 0.24.2's `_preprocess_data(normalize=True)` made them."""
    Xc = X - X.mean(axis=0)
    norms = np.sqrt((Xc**2).sum(axis=0))
    norms[norms == 0] = 1.0
    return Xc / norms, y - y.mean()


class _NumpyWithoutAround:
    """numpy, but `around` does nothing: today's `_lars_path_solver` calls it only to round `corr_eq_dir`."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def around(a, decimals=0, out=None):
        return a if out is None else out


@contextlib.contextmanager
def _without_rounding():
    original = least_angle.np
    least_angle.np = _NumpyWithoutAround()
    try:
        yield
    finally:
        least_angle.np = original


def _sklearn_path(Xn, yc, alpha_min=0.0, rounding=True):
    """(alphas, coefs (kinks, p), whether the degenerate-entrant rule fired) of `lars_path` on the Gram, as old LassoLarsIC
    called it. alpha_min goes in as a float64: under numpy 2 a Python float plus scikit-learn's float32 tolerance is float32."""
    with warnings.catch_warnings(record=True) as caught, contextlib.nullcontext() if rounding else _without_rounding():
        warnings.simplefilter("always", ConvergenceWarning)
        alphas, _, coefs = sklearn_lars_path(
            Xn, yc, Gram=Xn.T @ Xn, method="lasso", alpha_min=np.float64(alpha_min), max_iter=2500, eps=EPS
        )
    return alphas, coefs.T, any("degenerate" in str(w.message) for w in caught)


def _assert_same_path(path, alphas, coefs, tol=1e-10):
    """Same kinks and active sets; alphas within tol of the first, coefficients within tol of the largest."""
    assert len(path.alphas) == len(alphas)
    scale = max(np.max(np.abs(coefs)), 1e-300)
    np.testing.assert_array_equal(np.abs(path.coefs) > 1e-10 * scale, np.abs(coefs) > 1e-10 * scale)
    assert np.max(np.abs(path.alphas - alphas)) <= tol * alphas[0]
    assert np.max(np.abs(path.coefs - coefs)) <= tol * scale


# ------------------------------------------------------------------------------------------------------------- lars_path


@pytest.mark.parametrize("rounding", [True, False], ids=["today", "0.24.2"])
@pytest.mark.parametrize("kind", ["n>p", "n<p", "weekday dummies", "zero column"])
def test_lars_path_equals_scikit_learn(kind, rounding):
    for seed in range(4):
        X, y = _design(kind, seed)
        Xn, yc = _normalised(X, y)
        alphas, coefs, _ = _sklearn_path(Xn, yc, rounding=rounding)
        path = _lars.lars_path(Xn.T @ Xn, Xn.T @ yc, len(y))
        _assert_same_path(path, alphas, coefs)
        assert path.stop == "alpha_min"
        np.testing.assert_allclose(
            path.corr, Xn.T @ yc - path.coefs @ (Xn.T @ Xn), rtol=0, atol=1e-12 * np.abs(Xn.T @ yc).max()
        )


@pytest.mark.parametrize("rounding", [True, False], ids=["today", "0.24.2"])
@pytest.mark.parametrize("kind", ["duplicate", "near duplicate"])
def test_lars_path_with_a_twin_equals_scikit_learn_until_the_tie(kind, rounding):
    """Up to the kink at which one of the twins enters, the path is scikit-learn's. From the next one, which of the two tied
    columns enters, and whether the other is then tried as a degenerate entrant, is a matter of the last bit. Twins are never
    both in the model."""
    twin = 12
    for seed in range(8):
        X, y = _design(kind, seed)
        Xn, yc = _normalised(X, y)
        alphas, coefs, _ = _sklearn_path(Xn, yc, rounding=rounding)
        path = _lars.lars_path(Xn.T @ Xn, Xn.T @ yc, len(y))
        entered = np.flatnonzero((coefs[:, TWIN] != 0) | (coefs[:, twin] != 0))
        end = int(entered[0]) if entered.size else len(alphas)  # the kink before a twin's coefficient leaves 0
        before = path._replace(alphas=path.alphas[:end], coefs=path.coefs[:end])
        _assert_same_path(before, alphas[:end], coefs[:end])
        assert not np.any((path.coefs[:, TWIN] != 0) & (path.coefs[:, twin] != 0))


# Gram designs of dyadic numbers: every step before the degenerate entrant is exact, so its tie is too.
_NEAR = 2.0**-50
DYADIC = {
    # the twin of column 0 is first in line when it ties: 0.24.2 zeroes its correlation
    "twin first in line": ([[1, 1, 0.5], [1, 1, 0.5], [0.5, 0.5, 1]], [1, 1, 0.75]),
    # the twin is behind another inactive column: 0.24.2 swaps the correlations back but not the Gram rows
    "twin behind": ([[1, 0, 1, 0.5], [0, 1, 0, 0], [1, 0, 1, 0.5], [0.5, 0, 0.5, 1]], [1, 0.125, 1, 0.75]),
    # its pivot is sqrt(2^-49) = 4.2e-8, below 0.24.2's 1e-7: it never enters
    "near twin": ([[1, 1 - _NEAR, 0.5], [1 - _NEAR, 1, 0.5], [0.5, 0.5, 1]], [1, 1, 0.75]),
}


@pytest.mark.parametrize("name", list(DYADIC))
def test_the_degenerate_entrant_rule_is_0_24_2_s(name):
    G, q = (np.array(v, dtype=float) for v in DYADIC[name])
    for n in (1, 10):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            alphas, _, coefs = lars_path_gram(q, G, n_samples=n, method="lasso", alpha_min=0.0, max_iter=2500, eps=EPS)
        assert any("degenerate" in str(w.message) for w in caught)
        _assert_same_path(_lars.lars_path(G, q, n), alphas, coefs.T, tol=1e-15)


def test_a_degenerate_entrant_can_end_the_path_early_as_in_0_24_2():
    """0.24.2 restarts the zeroed correlation at 0; it grows back as the path goes on, and when it reaches the active ones the
    twin is tried again, zeroed again, and with nothing left the path stops at "alpha 0" with the coefficients of α = 0.25.
    The least-squares fit there would be (5/6, 0, 1/3). Kept, so this kernel's path is the one Lago's LassoLarsIC walked."""
    G, q = (np.array(v, dtype=float) for v in DYADIC["twin first in line"])
    path = _lars.lars_path(G, q, 1)
    np.testing.assert_allclose(path.alphas, [1.0, 0.5, 0.0], rtol=0, atol=1e-15)
    np.testing.assert_allclose(path.coefs[-1], [2 / 3, 0.0, 1 / 6], rtol=0, atol=1e-15)


@pytest.mark.parametrize("kind", ["n>p", "n<p", "weekday dummies"])
def test_a_path_stopped_at_alpha_min_ends_where_scikit_learn_s_does(kind):
    """The early stop at alpha_min + float32 eps, and the interpolation back to alpha_min when a step overshoots it."""
    for seed in range(3):
        X, y = _design(kind, seed)
        Xn, yc = _normalised(X, y)
        full = _lars.lars_path(Xn.T @ Xn, Xn.T @ yc, len(y))
        for k in (1, len(full.alphas) // 3, len(full.alphas) // 2):
            alpha_min = 0.7 * full.alphas[k] + 0.3 * full.alphas[k + 1]
            for rounding in (True, False):
                alphas, coefs, _ = _sklearn_path(Xn, yc, alpha_min, rounding=rounding)
                path = _lars.lars_path(Xn.T @ Xn, Xn.T @ yc, len(y), alpha_min)
                _assert_same_path(path, alphas, coefs)
                assert path.alphas[-1] == alpha_min and path.stop == "alpha_min"


def test_max_steps_caps_the_path_as_max_iter_does():
    X, y = _design("n>p", 0)
    Xn, yc = _normalised(X, y)
    alphas, _, coefs = sklearn_lars_path(Xn, yc, Gram=Xn.T @ Xn, method="lasso", max_iter=7, eps=EPS)
    path = _lars.lars_path(Xn.T @ Xn, Xn.T @ yc, len(y), max_steps=7)
    _assert_same_path(path, alphas, coefs.T)
    assert path.stop == "max_steps" and len(path.alphas) == 8


# ------------------------------------------------------------------------------------------------------------- path_at


def _exact_lasso(Xc, yc, alpha):
    model = Lasso(alpha=alpha, fit_intercept=False, precompute=Xc.T @ Xc, tol=1e-12, max_iter=10**6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit(Xc, yc).coef_


@pytest.mark.parametrize("kind", ["n>p", "n<p", "weekday dummies", "zero column", "duplicate"])
def test_path_at_is_the_lasso_between_kinks(kind):
    """On the centred, unnormalised columns (stage 2's problem). With a duplicate only the twins' sum is identified."""
    X, y = _design(kind, 1)
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    path = _lars.lars_path(Xc.T @ Xc, Xc.T @ yc, len(y))
    last = (
        len(path.alphas) - 2 if kind != "n<p" else len(path.alphas) // 2
    )  # n < p: the path ends interpolating the data
    for k in np.linspace(0, last, 5).round().astype(int):
        alpha = 0.6 * path.alphas[k] + 0.4 * path.alphas[k + 1]
        ours, exact = _lars.path_at(path, alpha), _exact_lasso(Xc, yc, alpha)
        if kind == "duplicate":
            ours, exact = ours.copy(), exact.copy()
            ours[TWIN] += ours[-1]
            exact[TWIN] += exact[-1]
            ours, exact = ours[:-1], exact[:-1]
        np.testing.assert_allclose(ours, exact, rtol=0, atol=1e-8 * max(1.0, np.max(np.abs(exact))))
    np.testing.assert_array_equal(_lars.path_at(path, path.alphas[0]), 0.0)
    np.testing.assert_array_equal(_lars.path_at(path, path.alphas[-1] / 2), path.coefs[-1])


# ------------------------------------------------------------------------------------------------------------- ic_select


def _lars_old_ic(X, y, criterion="aic", max_iter=2500):
    """LassoLarsIC as in scikit-learn <= 1.1 defaults (normalize=True, sigma^2 = var(y)).

    Returns (alpha_in_normalized_space, coef_on_original_scale_at_IC_optimum, intercept) and, added here, the criterion."""
    from sklearn.linear_model import lars_path

    n = X.shape[0]
    xm = X.mean(axis=0)
    ym = y.mean()
    Xc = X - xm
    yc = y - ym
    norms = np.sqrt((Xc**2).sum(axis=0))
    norms[norms == 0] = 1.0
    Xn = Xc / norms
    Gram = Xn.T @ Xn if n > X.shape[1] else None
    alphas, _, coefs = lars_path(
        Xn,
        yc,
        Gram=Gram,
        copy_X=True,
        copy_Gram=True,
        alpha_min=0.0,
        method="lasso",
        max_iter=max_iter,
        eps=np.finfo(float).eps,
    )
    K = 2.0 if criterion == "aic" else np.log(n)
    R = yc[:, None] - Xn @ coefs
    mse = np.mean(R**2, axis=0)
    sigma2 = np.var(yc)
    df = (np.abs(coefs) > np.finfo(coefs.dtype).eps).sum(axis=0)
    crit = n * mse / (sigma2 + np.finfo(float).eps) + K * df
    best = int(np.argmin(crit))
    coef = coefs[:, best] / norms
    return float(alphas[best]), coef, ym - xm @ coef, crit


def _random_design(rng):
    """A design for the 200- and 500-design checks: Gaussian columns, some correlated, sometimes weekday dummies or a zero
    column; n from 12 to 300, at times below p. No (near) duplicates: their ties make the path depend on the last bit."""
    n = int(rng.integers(12, 301))
    p = int(rng.integers(1, 41))
    X = rng.standard_normal((n, p))
    if p > 3 and rng.random() < 0.5:
        X[:, : p // 2] += rng.standard_normal((n, 1)) * rng.uniform(0.5, 2)
    if rng.random() < 0.3:
        X = np.c_[X, np.eye(7)[rng.integers(0, 7, n)]]
    if rng.random() < 0.2:
        X[:, int(rng.integers(0, X.shape[1]))] = 0.0
    beta = np.where(rng.random(X.shape[1]) < 0.3, rng.standard_normal(X.shape[1]), 0.0)
    return X, X @ beta + rng.standard_normal(n) * rng.uniform(0.2, 3)


@pytest.mark.parametrize("criterion", ["aic", "bic"])
def test_ic_select_chooses_old_lasso_lars_ic_s_alpha(criterion):
    rng = np.random.default_rng(20261103)
    for _ in range(200):
        X, y = _random_design(rng)
        alpha, _, _, crit = _lars_old_ic(X, y, criterion)
        stats = _lars.gram_stats(X, y)
        R, qn = _lars._normalised(stats.G, stats.q)
        choice = _lars.ic_select(_lars.lars_path(R, qn, stats.n), qn, stats.yy, stats.n, criterion)
        assert choice.alpha == pytest.approx(alpha, rel=1e-9, abs=1e-15)
        if X.shape[0] > X.shape[1]:  # with n <= p old LassoLarsIC walked the path in X form: α* alone is compared
            assert len(choice.criterion) == len(crit)
            np.testing.assert_allclose(choice.criterion, crit, rtol=0, atol=1e-9 * max(1.0, np.max(np.abs(crit))))


@pytest.mark.parametrize("criterion", ["aic", "bic"])
def test_ic_select_on_n_below_p_matches_the_x_form_path(criterion):
    """n <= p (Lago's 56-day window: 49 rows, 247 columns): old LassoLarsIC walked the path in X form; the Gram form picks the
    same α* (decision 6)."""
    for seed in range(4):
        X, y = _design("n<p", seed)
        alpha, *_ = _lars_old_ic(X, y, criterion)
        fit = _lars.lear_fit(_lars.gram_stats(X, y), criterion, "lars")
        assert fit.alpha == pytest.approx(alpha, rel=1e-9)


# ------------------------------------------------------------------------------------------------------------- cd_lasso


def _fsign(f):
    return 1.0 if f > 0 else (-1.0 if f < 0 else 0.0)


def _cd_0242(w, alpha, beta, Q, q, y, max_iter, tol):
    """scikit-learn 0.24.2's `enet_coordinate_descent_gram` (random=0, positive=0) in Python, line by line, with its BLAS calls
    as loops. Returns (w, gap, tol, n_iter + 1)."""
    n_features = Q.shape[0]
    Q = Q.tolist()
    q = list(q)
    w = list(w)
    H = [sum(Q[i][j] * w[j] for j in range(n_features)) for i in range(n_features)]
    gap = tol + 1.0
    d_w_tol = tol
    y_norm2 = float(np.dot(y, y))
    tol = tol * y_norm2
    n_iter = 0
    for n_iter in range(max_iter):
        w_max = 0.0
        d_w_max = 0.0
        for f_iter in range(n_features):
            ii = f_iter
            if Q[ii][ii] == 0.0:
                continue
            w_ii = w[ii]
            if w_ii != 0.0:
                for k in range(n_features):  # H -= w_ii * Q[ii]
                    H[k] = H[k] + -w_ii * Q[ii][k]
            tmp = q[ii] - H[ii]
            w[ii] = _fsign(tmp) * max(abs(tmp) - alpha, 0.0) / (Q[ii][ii] + beta)
            if w[ii] != 0.0:
                for k in range(n_features):  # H += w[ii] * Q[ii]
                    H[k] = H[k] + w[ii] * Q[ii][k]
            d_w_ii = abs(w[ii] - w_ii)
            if d_w_ii > d_w_max:
                d_w_max = d_w_ii
            if abs(w[ii]) > w_max:
                w_max = abs(w[ii])
        if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_iter == max_iter - 1:
            q_dot_w = sum(w[i] * q[i] for i in range(n_features))
            XtA = [q[ii] - H[ii] - beta * w[ii] for ii in range(n_features)]
            dual_norm_XtA = max(abs(v) for v in XtA)
            tmp = 0.0
            for ii in range(n_features):
                tmp += w[ii] * H[ii]
            R_norm2 = y_norm2 + tmp - 2.0 * q_dot_w
            w_norm2 = sum(v * v for v in w)
            if dual_norm_XtA > alpha:
                const = alpha / dual_norm_XtA
                A_norm2 = R_norm2 * (const**2)
                gap = 0.5 * (R_norm2 + A_norm2)
            else:
                const = 1.0
                gap = R_norm2
            gap += (
                alpha * sum(abs(v) for v in w)
                - const * y_norm2
                + const * q_dot_w
                + 0.5 * beta * (1 + const**2) * w_norm2
            )
            if gap < tol:
                break
    return np.array(w), gap, tol, n_iter + 1


@pytest.mark.parametrize("kind", ["n>p", "weekday dummies", "zero column", "few rows"])
def test_cd_lasso_is_0_24_2_s_coordinate_descent(kind):
    """Lago's refit, `Lasso(alpha, max_iter=2500)` with its tol 1e-4, often stops before converging: where it stops is part of
    the forecast. α at three points of the path, one of them the AIC choice."""
    if kind == "few rows":
        rng = np.random.default_rng(5)
        X = rng.standard_normal((20, 30))
        y = X[:, :4] @ rng.standard_normal(4) + rng.standard_normal(20)
    else:
        X, y = _design(kind, 2)
        X = X[:, :30]
    stats = _lars.gram_stats(X, y)
    yc = y - y.mean()
    fit = _lars.lear_fit(stats, "aic", "cd")
    path = _lars.lars_path(stats.G, stats.q, stats.n)
    stopped = set()
    for alpha in (fit.alpha, path.alphas[2], path.alphas[len(path.alphas) // 2]):
        ours = _lars.cd_lasso(stats.G, stats.q, stats.yy, stats.n, alpha)
        theirs, gap, gap_tol, sweeps = _cd_0242(
            np.zeros(X.shape[1]), alpha * stats.n, 0.0, stats.G, stats.q, yc, 2500, 1e-4
        )
        np.testing.assert_allclose(ours.w, theirs, rtol=0, atol=1e-12 * max(1.0, np.max(np.abs(theirs))))
        assert ours.sweeps == sweeps and ours.converged == (gap < gap_tol)
        stopped.add(sweeps)
    assert len(stopped) > 1


def test_cd_lasso_stopped_by_max_iter_is_0_24_2_s():
    """Strongly correlated columns at a small α don't converge in 60 sweeps: the gap is checked on the last sweep, found too
    large, and the descent stops there, unconverged, as Lago's refits at 2500 sweeps sometimes did."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((150, 1)) + 0.05 * rng.standard_normal((150, 25))
    y = X[:, :5].sum(axis=1) + rng.standard_normal(150)
    stats = _lars.gram_stats(X, y)
    path = _lars.lars_path(stats.G, stats.q, stats.n)
    for alpha in (path.alphas[len(path.alphas) // 2], path.alphas[-3]):
        ours = _lars.cd_lasso(stats.G, stats.q, stats.yy, stats.n, alpha, max_iter=60)
        theirs, gap, gap_tol, sweeps = _cd_0242(
            np.zeros(25), alpha * stats.n, 0.0, stats.G, stats.q, y - y.mean(), 60, 1e-4
        )
        assert (ours.sweeps, ours.converged, sweeps, gap < gap_tol) == (60, False, 60, False)
        np.testing.assert_allclose(ours.w, theirs, rtol=0, atol=1e-12 * max(1.0, np.max(np.abs(theirs))))


@pytest.mark.parametrize("kind", ["n>p", "weekday dummies"])
def test_cd_lasso_converged_is_the_exact_lasso(kind):
    X, y = _design(kind, 3)
    stats = _lars.gram_stats(X, y)
    path = _lars.lars_path(stats.G, stats.q, stats.n)
    for k in (1, len(path.alphas) // 2, len(path.alphas) - 3):
        alpha = 0.5 * (path.alphas[k] + path.alphas[k + 1])
        result = _lars.cd_lasso(stats.G, stats.q, stats.yy, stats.n, alpha, max_iter=10**5, tol=1e-12)
        assert result.converged
        np.testing.assert_allclose(result.w, _lars.path_at(path, alpha), rtol=0, atol=1e-8)


# ------------------------------------------------------------------------------------------------------------- lear_fit


@pytest.mark.parametrize("criterion", ["aic", "bic"])
@pytest.mark.parametrize("kind", ["n>p", "n<p", "weekday dummies"])
def test_lear_fit_is_alpha_star_then_the_lasso_refit(kind, criterion):
    """Stage 1's α* is old LassoLarsIC's; "lars" is the Lasso at α* on the unnormalised columns within 1e-8 (Lasso tol 1e-12),
    and "cd" is 0.24.2's coordinate descent there, to the bit."""
    X, y = _design(kind, 4)
    stats = _lars.gram_stats(X, y)
    alpha, *_ = _lars_old_ic(X, y, criterion)
    exact = _lars.lear_fit(stats, criterion, "lars")
    assert exact.alpha == pytest.approx(alpha, rel=1e-9) and exact.converged
    reference = _exact_lasso(X - stats.xbar, y - stats.ybar, exact.alpha)
    np.testing.assert_allclose(exact.w, reference, rtol=0, atol=1e-8 * max(1.0, np.max(np.abs(reference))))
    lago = _lars.lear_fit(stats, criterion, "cd")
    assert lago.alpha == exact.alpha and lago.kink == exact.kink
    np.testing.assert_array_equal(lago.w, _lars.cd_lasso(stats.G, stats.q, stats.yy, stats.n, exact.alpha).w)
    np.testing.assert_array_equal(exact.xbar, X.mean(axis=0))
    assert exact.ybar == y.mean()


# ------------------------------------------------------------------------------------------------------------- edge cases


def test_a_zero_column_gets_exactly_zero_and_changes_nothing():
    X, y = _design("zero column", 0)
    others = np.delete(np.arange(X.shape[1]), 5)
    with_it, without = _lars.gram_stats(X, y), _lars.gram_stats(X[:, others], y)
    a, b = _lars.lars_path(with_it.G, with_it.q, with_it.n), _lars.lars_path(without.G, without.q, without.n)
    assert len(a.alphas) == len(b.alphas)
    np.testing.assert_array_equal(a.coefs[:, 5], 0.0)
    np.testing.assert_allclose(a.alphas, b.alphas, rtol=0, atol=1e-12 * b.alphas[0])
    np.testing.assert_allclose(a.coefs[:, others], b.coefs, rtol=0, atol=1e-12 * np.max(np.abs(b.coefs)))
    for criterion in ("aic", "bic"):
        for solver in ("lars", "cd"):
            fit, reference = _lars.lear_fit(with_it, criterion, solver), _lars.lear_fit(without, criterion, solver)
            assert fit.w[5] == 0.0 and fit.kink == reference.kink
            np.testing.assert_allclose(
                fit.w[others], reference.w, rtol=0, atol=1e-12 * max(1.0, np.max(np.abs(reference.w)))
            )


def test_a_column_with_a_zero_diagonal_never_enters_even_with_a_stray_correlation():
    """Shared statistics may leave rounding in a zeroed column's cross terms or correlation: the column is cut out whole."""
    X, y = _design("n>p", 0)
    stats = _lars.gram_stats(X[:, :6], y)
    G, q = np.zeros((7, 7)), np.zeros(7)
    G[:6, :6], q[:6] = stats.G, stats.q
    G[6, :6] = G[:6, 6] = 1e-13
    q[6] = 1e3
    path = _lars.lars_path(G, q, stats.n)
    np.testing.assert_array_equal(path.coefs[:, 6], 0.0)
    reference = _lars.lars_path(stats.G, stats.q, stats.n)
    _assert_same_path(path._replace(coefs=path.coefs[:, :6]), reference.alphas, reference.coefs, tol=1e-12)


@pytest.mark.parametrize("solver", ["lars", "cd"])
def test_a_constant_target_gives_zero_coefficients(solver):
    X, _ = _design("n>p", 0)
    fit = _lars.lear_fit(_lars.gram_stats(X[:50, :5], np.full(50, 3.0)), "bic", solver)
    np.testing.assert_array_equal(fit.w, 0.0)
    assert fit.alpha == 0.0 and fit.kink == 0 and fit.ybar == 3.0


def test_one_column_and_two_rows():
    x = np.array([[1.0], [2.0], [4.0], [3.0]])
    y = np.array([1.0, 3.0, 2.0, 5.0])
    stats = _lars.gram_stats(x, y)
    path = _lars.lars_path(stats.G, stats.q, stats.n)
    assert path.alphas[0] == pytest.approx(abs(stats.q[0]) / 4)
    assert path.coefs[-1, 0] == pytest.approx(stats.q[0] / stats.G[0, 0])
    two = _lars.gram_stats(np.array([[1.0, 5.0, 2.0], [3.0, 4.0, 2.5]]), np.array([2.0, 7.0]))
    path = _lars.lars_path(two.G, two.q, two.n)  # one direction of variation: one step fits both rows
    rss = two.yy - two.q @ path.coefs[-1] - path.corr[-1] @ path.coefs[-1]
    assert len(path.alphas) == 2 and abs(rss) <= 1e-12 * two.yy
    for criterion in ("aic", "bic"):
        for solver in ("lars", "cd"):
            fit = _lars.lear_fit(two, criterion, solver)
            assert np.all(np.isfinite(fit.w)) and fit.kink in (0, 1)


def test_bad_inputs_raise():
    stats = _lars.gram_stats(np.eye(3), np.arange(3.0))
    with pytest.raises(ValueError, match="criterion"):
        _lars.lear_fit(stats, "hqic", "lars")
    with pytest.raises(ValueError, match="solver"):
        _lars.lear_fit(stats, "bic", "newton")
    with pytest.raises(ValueError, match="finite"):
        _lars.lars_path(np.full((2, 2), np.nan), np.zeros(2), 3)
    with pytest.raises(ValueError, match="must be"):
        _lars.lars_path(np.eye(3), np.zeros(2), 3)
    with pytest.raises(ValueError, match="gram_stats"):
        _lars.gram_stats(np.zeros((3, 2)), np.zeros(4))


# ------------------------------------------------------------------------------------------------------------- backends


def _lago_size_design(n, seed):
    """247 columns as LEAR has them: 240 correlated continuous inputs and seven weekday dummies."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 60)) @ rng.standard_normal((60, 240)) * 0.3 + rng.standard_normal((n, 240))
    X = np.c_[X, np.eye(7)[np.arange(n) % 7]]
    beta = np.zeros(247)
    beta[rng.choice(247, 40, replace=False)] = rng.standard_normal(40)
    return X, X @ beta + 2 * rng.standard_normal(n)


def _assert_backends_agree(numba_kernels, X, y, solvers=("lars", "cd")):
    """Same kinks, the same kink chosen, coefficients within 1e-10. A dropped coefficient keeps a rounding residue at its drop
    kink (0.24.2's `prev + gamma * ls`), and whether it is above float64 eps, so counted in df, depends on the summation order:
    there the criteria may differ by K per residue, and nowhere else."""
    stats = _lars.gram_stats(X, y)
    R, qn = _lars._normalised(stats.G, stats.q)
    reference, twin = _lars.lars_path(R, qn, stats.n), numba_kernels.lars_path(R, qn, stats.n)
    _assert_same_path(twin, reference.alphas, reference.coefs)
    assert twin.stop == reference.stop
    np.testing.assert_allclose(twin.corr, reference.corr, rtol=0, atol=1e-10 * max(1.0, np.max(np.abs(reference.corr))))
    counted = np.abs(reference.coefs) > EPS
    residues = counted != (np.abs(twin.coefs) > EPS)
    assert np.all(np.abs(reference.coefs[residues]) < 1e3 * EPS) and np.all(np.abs(twin.coefs[residues]) < 1e3 * EPS)
    ddf = counted.sum(axis=1) - (np.abs(twin.coefs) > EPS).sum(axis=1)
    for criterion in ("aic", "bic"):
        a = _lars.ic_select(reference, qn, stats.yy, stats.n, criterion)
        b = numba_kernels.ic_select(twin, qn, stats.yy, stats.n, criterion)
        assert a.kink == b.kink
        K = 2.0 if criterion == "aic" else math.log(stats.n)
        scale = max(1.0, np.max(np.abs(a.criterion)))
        np.testing.assert_allclose(b.criterion + K * ddf, a.criterion, rtol=0, atol=1e-9 * scale)
        for solver in solvers:
            a, b = _lars.lear_fit(stats, criterion, solver), numba_kernels.lear_fit(stats, criterion, solver)
            assert (a.kink, a.converged) == (b.kink, b.converged)
            assert abs(b.alpha - a.alpha) <= 1e-10 * reference.alphas[0]
            np.testing.assert_allclose(b.w, a.w, rtol=0, atol=1e-10 * max(1.0, np.max(np.abs(a.w))))


@pytest.fixture
def numba_kernels(monkeypatch):
    pytest.importorskip("numba")
    monkeypatch.setenv("EPFTOOLBOX2_KERNELS", "numba")
    return _lars._kernels()


def test_numba_twins_agree_with_numpy_on_496_random_designs(numba_kernels):
    """Both stages with "lars" on every design, with "cd" on every fourth (numpy's coordinate descent is a Python loop)."""
    rng = np.random.default_rng(496)
    for i in range(496):
        _assert_backends_agree(numba_kernels, *_random_design(rng), solvers=("lars", "cd") if i % 4 == 0 else ("lars",))


@pytest.mark.parametrize("n", [49, 77, 1085, 1449])
def test_numba_twins_agree_with_numpy_at_lago_s_sizes(numba_kernels, n):
    """p 247 and Lago's training rows for windows 56, 84, 1092 and 1456: with the 496 random ones, 500 designs."""
    _assert_backends_agree(numba_kernels, *_lago_size_design(n, n), solvers=("lars", "cd") if n < 1000 else ("lars",))


def test_the_kernel_backend_follows_epftoolbox2_kernels(monkeypatch):
    monkeypatch.setenv("EPFTOOLBOX2_KERNELS", "numpy")
    assert _lars._kernels() is _lars and _lars._kernels().BACKEND == "numpy"
    monkeypatch.setenv("EPFTOOLBOX2_KERNELS", "fortran")
    with pytest.raises(ValueError, match="EPFTOOLBOX2_KERNELS"):
        _lars._kernels()
    monkeypatch.setattr(_lars, "has_extra", lambda name: False)  # as without the [fast] extra
    monkeypatch.setenv("EPFTOOLBOX2_KERNELS", "auto")
    assert _lars._kernels() is _lars
    monkeypatch.delenv("EPFTOOLBOX2_KERNELS")
    assert _lars._kernels() is _lars
    monkeypatch.setenv("EPFTOOLBOX2_KERNELS", "numba")
    with pytest.raises(MissingExtraError, match=r"pip install 'epftoolbox2\[fast\]'"):
        _lars._kernels()


def test_numba_is_the_auto_backend_when_installed(monkeypatch):
    pytest.importorskip("numba")
    monkeypatch.delenv("EPFTOOLBOX2_KERNELS", raising=False)
    kernels = _lars._kernels()
    assert kernels.BACKEND == "numba" and kernels.__name__ == "epftoolbox2.models._lars_numba"


def test_importing_the_package_and_the_numpy_kernels_imports_no_numba():
    code = (
        "import os, sys\n"
        "os.environ['EPFTOOLBOX2_KERNELS'] = 'numpy'\n"
        "import epftoolbox2, epftoolbox2.models\n"
        "from epftoolbox2.models import _lars\n"
        "assert _lars._kernels() is _lars\n"
        "assert 'numba' not in sys.modules, 'numba was imported'\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "EPFTOOLBOX2_KERNELS"}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
