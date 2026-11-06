"""LARS-lasso and coordinate-descent kernels on a Gram matrix, for LEAR: the numpy reference.

A LEAR unit (Lago et al., 2021) is fitted in two stages from the centred sums of its training rows alone (`GramStats`, the seam
through which step 5's engine will hand in statistics it shares between models):

1. α* as old `LassoLarsIC(normalize=True)` chose it: the LARS-lasso path on centred unit-norm columns, and at every kink of it
   `n·MSE/(var(y) + eps) + K·df` (K = 2 for AIC, log n for BIC), the first argmin (`lars_path`, `ic_select`).
2. The Lasso at α* on the centred, unnormalised columns, either exactly, by a second LARS path run just past α* (`lasso_at`,
   `solver="lars"`), or as Lago's `Lasso(max_iter=2500)` ran it, by coordinate descent that may stop before convergence
   (`solver="cd"`).

`lars_path` transcribes scikit-learn 0.24.2's `_least_angle._lars_path_solver` (the version Lago et al. ran) for
`method="lasso"`, a given Gram and `positive=False`; `cd_lasso` transcribes its `_cd_fast.enet_coordinate_descent_gram` with no
L2 term. Where that code has a quirk, these keep it and say so (the degenerate entrant in `lars_path`). Two departures, both
where 0.24.2 would go wrong: a column whose Gram diagonal is 0 has its row, column and correlation set to exactly 0 first, so
rounding left in shared statistics can't make it enter; and coefficients that reach 0 at exactly the same step all leave
together (0.24.2 counts one of them out, after which its arrays no longer match in length and it raises).

`_lars_numba.py` holds numba twins of the same functions. The two agree within 1e-9 with the same kinks, not to the bit: this
module sums with BLAS, the twins in loops. `_kernels()` returns the module of the backend `EPFTOOLBOX2_KERNELS` asks for.
"""

import functools
import math
import os
import sys
import warnings
from typing import List, NamedTuple, Tuple

import numpy as np
from scipy.linalg import solve_triangular
from scipy.linalg.lapack import dpotrs

from .._extras import MissingExtraError, has_extra

BACKEND = "numpy"

EPS = float(np.finfo(np.float64).eps)
# 0.24.2's `equality_tolerance` and `tiny32`. It added them to float64 values under numpy 1, so in float64; under numpy 2 a
# Python float plus a float32 scalar is float32 (NEP 50), which today's scikit-learn gets for a float `alpha_min`.
EQUALITY_TOLERANCE = float(np.finfo(np.float32).eps)
TINY32 = float(np.finfo(np.float32).tiny)
DBL_MAX = float(np.finfo(np.float64).max)
# A Cholesky pivot below this means the entering column lies in the span of the active ones (0.24.2: `diag < 1e-7`).
DEGENERATE = 1e-7

# Why a path ended. The numba twin returns the position in this tuple.
STOPS = ("alpha_min", "max_steps", "all_active", "alpha_increasing")
CRITERIA = ("aic", "bic")
SOLVERS = ("lars", "cd")


class GramStats(NamedTuple):
    """Centred sums of a unit's training rows: everything a LEAR fit reads."""

    n: int  # rows
    xbar: np.ndarray  # column means, (p,)
    ybar: float  # target mean
    G: np.ndarray  # (X - xbar)'(X - xbar), (p, p)
    q: np.ndarray  # (X - xbar)'(y - ybar), (p,)
    yy: float  # |y - ybar|²


class LarsPath(NamedTuple):
    """A LARS-lasso path, one row per kink: alphas[k] = C/n and the exact lasso solution there."""

    alphas: np.ndarray  # (m,)
    coefs: np.ndarray  # (m, p)
    corr: np.ndarray  # (m, p): q - G w at each kink
    stop: str  # one of STOPS


class IcChoice(NamedTuple):
    alpha: float
    kink: int
    criterion: np.ndarray  # (m,): the criterion at each kink


class CdResult(NamedTuple):
    w: np.ndarray
    sweeps: int  # 0.24.2's n_iter: sweeps run, the last included
    converged: bool  # the duality gap fell below tol * yy


class LearFit(NamedTuple):
    """A fitted LEAR unit: predict a row x as ybar + (x - xbar) @ w."""

    w: np.ndarray
    xbar: np.ndarray
    ybar: float
    alpha: float  # α*, on the normalised scale of stage 1
    kink: int  # the stage-1 kink of α*
    converged: bool  # "lars": stage 2's path passed α*; "cd": the duality gap fell below tol


def gram_stats(x, y) -> GramStats:
    """The centred `X'X`, `X'y` and `y'y` of training rows `x` (n, p) and targets `y` (n,)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0] or x.shape[0] < 1:
        raise ValueError(
            f"gram_stats needs x of shape (n, p) and y of shape (n,) with n >= 1, got {x.shape} and {y.shape}"
        )
    xbar = x.mean(axis=0)
    ybar = float(y.mean())
    xc = x - xbar
    yc = y - ybar
    return GramStats(x.shape[0], xbar, ybar, xc.T @ xc, xc.T @ yc, float(yc @ yc))


def _checked(G, q, n) -> Tuple[np.ndarray, np.ndarray, float]:
    """Float64 copies of G (C order) and q, validated."""
    G = np.array(G, dtype=np.float64, order="C", copy=True)
    q = np.array(q, dtype=np.float64, copy=True)
    if q.ndim != 1 or G.shape != (q.size, q.size):
        raise ValueError(f"G must be (p, p) and q (p,), got {G.shape} and {q.shape}")
    if not (np.isfinite(G).all() and np.isfinite(q).all()):
        raise ValueError("G and q must be finite")
    n = float(n)
    if not n > 0:
        raise ValueError(f"n must be positive, got {n}")
    return G, q, n


def _min_pos(x: np.ndarray) -> float:
    """0.24.2's `arrayfuncs.min_pos`: the smallest value in (0, DBL_MAX), or DBL_MAX when there is none."""
    x = x[(x > 0.0) & (x < DBL_MAX)]
    return float(x.min()) if x.size else DBL_MAX


def _inv_sqrt(s: float) -> float:
    """1 / sqrt(s), with infinity standing for numpy's inf or nan when s <= 0 (both only tested for finiteness)."""
    return 1.0 / math.sqrt(s) if s > 0.0 else math.inf


def _swap(cov: np.ndarray, indices: np.ndarray, G: np.ndarray, i: int, j: int) -> None:
    """Swap positions i and j of the permuted problem: correlation, column index, Gram row and column."""
    cov[[i, j]] = cov[[j, i]]
    indices[[i, j]] = indices[[j, i]]
    G[[i, j]] = G[[j, i]]
    G[:, [i, j]] = G[:, [j, i]]


def _cholesky_delete(L: np.ndarray, n: int, go_out: int) -> None:
    """0.24.2's `arrayfuncs.cholesky_delete` on L[:n, :n]: remove row and column `go_out`, then restore the triangle with
    Givens rotations. The new factor is L[:n - 1, :n - 1]; row n - 1 is left stale, as there."""
    for i in range(go_out, n - 1):
        L[i, : i + 2] = L[i + 1, : i + 2]
    for i in range(go_out, n - 1):
        a, b = L[i, i], L[i, i + 1]
        r = math.hypot(a, b)  # BLAS drotg, with the sign flip that keeps the diagonal positive
        c, s = (a / r, b / r) if r > 0.0 else (1.0, 0.0)
        L[i, i] = r
        L[i, i + 1] = 0.0
        x = L[i + 1 : n - 1, i].copy()
        y = L[i + 1 : n - 1, i + 1].copy()
        L[i + 1 : n - 1, i] = c * x + s * y
        L[i + 1 : n - 1, i + 1] = c * y - s * x


def lars_path(G, q, n, alpha_min: float = 0.0, max_steps: int = 2500) -> LarsPath:
    """The LARS-lasso path of min 1/(2n)|y - Xw|² + α|w|₁ from G = X'X and q = X'y, from α = max|q|/n down to `alpha_min`.

    scikit-learn 0.24.2's `_lars_path_solver(method="lasso", Gram=G, positive=False)`, step by step: the permuted Gram, the
    inactive correlations `Cov`, the entrant by the first largest |Cov| in permuted order, the drop of a coefficient that
    changes sign, the "degenerate" entrant, the early stop at `alpha <= alpha_min + float32 eps` (interpolated to `alpha_min`
    when it overshoots by more than that), the bail-out when alpha increases, and at most `max_steps` steps.
    """
    G, q, n = _checked(G, q, n)
    alpha_min = float(alpha_min)
    p = q.size
    dead = ~(np.diag(G) > 0.0)  # a zero column (the scaler's near-constant rule) must never enter
    G[dead, :] = 0.0
    G[:, dead] = 0.0
    q[dead] = 0.0
    gram0 = G.copy()  # 0.24.2's Gram_copy and Cov_copy, in the original order
    q0 = q.copy()
    cov = q.copy()  # cov[n_active:] is 0.24.2's Cov, aligned with the permuted Gram
    indices = np.arange(p)  # indices[i]: the original column at position i of the permuted Gram
    max_features = min(max_steps, p)
    alphas = np.zeros(max_features + 1)
    coefs = np.zeros((max_features + 1, p))
    L = np.zeros((max_features, max_features))  # lower Cholesky factor of the active block, in active order
    sign_active = np.zeros(max_features)
    active: List[int] = []  # original column indices, in active order
    n_iter = n_active = 0
    drop = False
    idx = np.zeros(0, dtype=np.int64)

    while True:
        if n_active < p:
            c_idx = n_active + int(np.argmax(np.abs(cov[n_active:])))
            c_signed = cov[c_idx]
            C = abs(c_signed)
        else:
            C = 0.0
        alpha = C / n
        alphas[n_iter] = alpha
        if alpha <= alpha_min + EQUALITY_TOLERANCE:  # early stop
            if abs(alpha - alpha_min) > EQUALITY_TOLERANCE:
                if n_iter > 0:  # interpolate back to alpha_min (0 <= ss < 1)
                    ss = (alphas[n_iter - 1] - alpha_min) / (alphas[n_iter - 1] - alpha)
                    coefs[n_iter] = coefs[n_iter - 1] + ss * (coefs[n_iter] - coefs[n_iter - 1])
                alphas[n_iter] = alpha_min
            stop = "alpha_min"
            break
        if n_iter >= max_steps or n_active >= p:
            stop = "max_steps" if n_iter >= max_steps else "all_active"
            break
        if not drop:
            m = n_active
            sign_active[m] = np.sign(c_signed)
            _swap(cov, indices, G, m, c_idx)
            c = G[m, m]
            row = G[m, :m]
            if m:
                row = solve_triangular(L[:m, :m], row, lower=True, check_finite=False)
            L[m, :m] = row
            v = float(row @ row)
            diag = max(math.sqrt(abs(c - v)), EPS)
            L[m, m] = diag
            if diag < DEGENERATE:
                # 0.24.2's "drop for good": the entrant's correlation is zeroed and swapped back to where it came from, but
                # the Gram and the indices stay swapped. Unless the entrant was first in line (c_idx == m), position m now
                # pairs the old occupant's correlation with the entrant's Gram row, and c_idx the reverse. Kept as is.
                cov[m] = 0.0
                cov[[m, c_idx]] = cov[[c_idx, m]]
                continue
            active.append(int(indices[m]))
            n_active += 1

        if n_iter > 0 and alphas[n_iter - 1] < alpha:  # rounding has taken over: 0.24.2 stops here
            stop = "alpha_increasing"
            break

        signs = sign_active[:n_active]
        least_squares, _ = dpotrs(L[:n_active, :n_active], signs, lower=1)
        if least_squares.size == 1 and least_squares[0] == 0.0:  # a zero sign
            least_squares[...] = 1.0
            AA = 1.0
        else:
            AA = _inv_sqrt(float(np.sum(least_squares * signs)))
            if not math.isfinite(AA):  # L too ill-conditioned: regularise its diagonal until it isn't
                i = 0
                L_ = L[:n_active, :n_active].copy()
                while not math.isfinite(AA):
                    L_.flat[:: n_active + 1] += (2**i) * EPS
                    least_squares, _ = dpotrs(L_, signs, lower=1)
                    AA = 1.0 / math.sqrt(max(float(np.sum(least_squares * signs)), EPS))
                    i += 1
            least_squares *= AA

        corr_eq_dir = G[:n_active, n_active:].T @ least_squares
        inactive = cov[n_active:]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            g1 = _min_pos((C - inactive) / (AA - corr_eq_dir + TINY32))
            g2 = _min_pos((C + inactive) / (AA + corr_eq_dir + TINY32))
            gamma = min(g1, g2, C / AA)
            z = -coefs[n_iter, active] / (least_squares + TINY32)
        drop = False
        z_pos = _min_pos(z)
        if z_pos < gamma:  # an active coefficient reaches 0 first: the lasso step stops there and drops it
            idx = np.where(z == z_pos)[0][::-1]
            sign_active[idx] = -sign_active[idx]
            gamma = z_pos
            drop = True

        n_iter += 1
        if n_iter >= coefs.shape[0]:
            grow = 2 * max(1, max_features - n_active)
            coefs = np.concatenate([coefs, np.zeros((grow, p))])
            alphas = np.concatenate([alphas, np.zeros(grow)])
        coefs[n_iter, active] = coefs[n_iter - 1, active] + gamma * least_squares
        cov[n_active:] -= gamma * corr_eq_dir

        if drop:
            for ii in idx:  # descending
                _cholesky_delete(L, n_active, int(ii))
            old_active = n_active
            n_active -= len(idx)
            dropped = [active.pop(int(ii)) for ii in idx]
            for t, ii in enumerate(idx):  # each dropped column moves to the front of the inactive positions
                end = old_active - 1 - t
                order = np.r_[ii + 1 : end + 1, ii]
                indices[ii : end + 1] = indices[order]
                G[ii : end + 1] = G[order]
                G[:, ii : end + 1] = G[:, order]
                # its correlation afresh, as 0.24.2: Cov_copy[drop_idx] - Gram_copy[drop_idx] @ coef (a matrix-vector product)
                cov[end] = (q0[[dropped[t]]] - gram0[[dropped[t]]] @ coefs[n_iter])[0]
            sign_active = np.concatenate([np.delete(sign_active, idx), np.zeros(len(idx))])

    m = n_iter + 1
    coefs = coefs[:m].copy()
    return LarsPath(alphas[:m].copy(), coefs, q0 - coefs @ gram0, stop)


def path_at(path: LarsPath, alpha: float) -> np.ndarray:
    """The lasso solution at `alpha`, linear between the kinks around it: 0 at or above the first kink, the last kink's
    coefficients below the last."""
    alphas, coefs = path.alphas, path.coefs
    if alpha >= alphas[0]:
        return np.zeros(coefs.shape[1])
    below = np.flatnonzero(alphas[1:] <= alpha)
    if not below.size:
        return coefs[-1].copy()
    i = int(below[0]) + 1
    t = (alphas[i - 1] - alpha) / (alphas[i - 1] - alphas[i])
    return coefs[i - 1] + t * (coefs[i] - coefs[i - 1])


def lasso_at(G, q, n, alpha: float, max_steps: int = 2500) -> Tuple[np.ndarray, bool]:
    """The exact lasso at `alpha` from a path that passes it, and whether it did (stage 2 of `solver="lars"`).

    0.24.2's early stop ends a path at the first kink within float32 eps (1.19e-7, absolute) above `alpha_min` and doesn't
    interpolate it, so a path stopped at `alpha` itself can end on a kink above it: on Lago's DE inputs at W 1456 that was 6 fits
    of 144, off the exact refit by up to 2.3e-4 EUR/MWh. Stopped at `alpha - 2 * float32 eps`, the path ends below `alpha`, and
    `path_at` interpolates the segment that holds it, on which the lasso is linear."""
    path = lars_path(G, q, n, max(float(alpha) - 2.0 * EQUALITY_TOLERANCE, 0.0), max_steps)
    reached = bool(alpha >= path.alphas[0] or np.any(path.alphas <= alpha))
    return path_at(path, alpha), reached


def _penalty(criterion: str, n: float) -> float:
    if criterion == "aic":
        return 2.0
    if criterion == "bic":
        return math.log(n)
    raise ValueError(f"criterion must be one of {CRITERIA}, not {criterion!r}")


def ic_select(path: LarsPath, q, yy: float, n, criterion: str) -> IcChoice:
    """Old `LassoLarsIC`'s choice on a path (scikit-learn <= 1.1): at every kink `n·MSE/(var(y) + eps) + K·df`, with
    MSE = rss/n, var(y) = yy/n (ddof 0), K = 2 (AIC) or log n (BIC), df the count of |w| > float64 eps; the first argmin.
    rss = yy - q·w - c·w from the path's correlations c = q - Gw."""
    n = float(n)
    K = _penalty(criterion, n)
    W = path.coefs
    rss = yy - W @ np.asarray(q, dtype=np.float64) - np.sum(path.corr * W, axis=1)
    df = np.sum(np.abs(W) > EPS, axis=1)
    criterion_values = n * (rss / n) / (yy / n + EPS) + K * df
    kink = int(np.argmin(criterion_values))
    return IcChoice(float(path.alphas[kink]), kink, criterion_values)


def cd_lasso(G, q, yy: float, n, alpha: float, max_iter: int = 2500, tol: float = 1e-4) -> CdResult:
    """Lasso at `alpha` by cyclic coordinate descent from w = 0: scikit-learn 0.24.2's `enet_coordinate_descent_gram` with
    beta = 0, as `Lasso(alpha, max_iter, tol, precompute=G)` ran it on centred data.

    l1 = alpha·n. Per coordinate in column order (a column with G[j, j] = 0 is skipped): H -= w_j·G[j], then
    w_j = soft(q_j - H_j, l1) / G[j, j], then H += w_j·G[j]. After a sweep whose largest |Δw| is below tol·max|w| (or with w = 0,
    or on the last sweep), the duality gap is checked against tol·yy, and below it the descent stops.
    """
    Q, q, n = _checked(G, q, n)
    p = q.size
    w = np.zeros(p)
    H = np.zeros(p)  # Q w
    l1 = float(alpha) * n
    y_norm2 = float(yy)
    gap_tol = tol * y_norm2
    rows = list(Q)
    diag = np.diag(Q).tolist()
    q_list = q.tolist()
    converged = False
    sweep = 0
    for sweep in range(max_iter):
        w_max = 0.0
        d_w_max = 0.0
        for j in range(p):
            Qjj = diag[j]
            if Qjj == 0.0:
                continue
            w_j = float(w[j])
            if w_j != 0.0:
                H -= w_j * rows[j]
            tmp = q_list[j] - float(H[j])
            new = math.copysign(1.0, tmp) * max(abs(tmp) - l1, 0.0) / Qjj if tmp != 0.0 else 0.0
            w[j] = new
            if new != 0.0:
                H += new * rows[j]
            d_w = abs(new - w_j)
            if d_w > d_w_max:
                d_w_max = d_w
            if abs(new) > w_max:
                w_max = abs(new)
        if w_max == 0.0 or d_w_max / w_max < tol or sweep == max_iter - 1:
            if _duality_gap(w, H, q, y_norm2, l1) < gap_tol:
                converged = True
                break
    return CdResult(w, sweep + 1, converged)


def _duality_gap(w: np.ndarray, H: np.ndarray, q: np.ndarray, y_norm2: float, l1: float) -> float:
    """0.24.2's gap with beta = 0: R² = yy + w'H - 2q·w; the dual point scaled to |X'R|∞ <= l1."""
    q_dot_w = float(w @ q)
    dual_norm = float(np.max(np.abs(q - H))) if q.size else 0.0
    r_norm2 = y_norm2 + float(w @ H) - 2.0 * q_dot_w
    if dual_norm > l1:
        const = l1 / dual_norm
        gap = 0.5 * (r_norm2 + r_norm2 * const**2)
    else:
        const = 1.0
        gap = r_norm2
    return gap + (l1 * float(np.sum(np.abs(w))) - const * y_norm2 + const * q_dot_w)


def _normalised(G: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """G and q of unit-norm columns, as `_preprocess_data(normalize=True)` scaled them: by sqrt(G[j, j]), 0 -> 1."""
    d = np.diag(G)
    norm = np.where(d > 0.0, np.sqrt(np.where(d > 0.0, d, 1.0)), 1.0)
    return G / np.outer(norm, norm), q / norm


def _check_choices(criterion: str, solver: str) -> None:
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, not {criterion!r}")
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, not {solver!r}")


def lear_fit(stats: GramStats, criterion: str = "bic", solver: str = "lars") -> LearFit:
    """A LEAR unit from its centred sums: α* by `ic_select` on the normalised path (2500 steps at most, as Lago's
    `LassoLarsIC(max_iter=2500)`), then the Lasso at α* on the centred, unnormalised Gram, exactly (`"lars"`: `lasso_at`, a
    second path run just past α*) or by Lago's coordinate descent (`"cd"`: `max_iter=2500`, `tol=1e-4`)."""
    _check_choices(criterion, solver)
    G = np.asarray(stats.G, dtype=np.float64)
    q = np.asarray(stats.q, dtype=np.float64)
    R, qn = _normalised(G, q)
    choice = ic_select(lars_path(R, qn, stats.n), qn, stats.yy, stats.n, criterion)
    if solver == "lars":
        w, converged = lasso_at(G, q, stats.n, choice.alpha)
    else:
        w, _, converged = cd_lasso(G, q, stats.yy, stats.n, choice.alpha)
    return LearFit(w, stats.xbar, stats.ybar, choice.alpha, choice.kink, bool(converged))


KERNELS = ("auto", "numpy", "numba")


def _kernels():
    """The kernel module for `EPFTOOLBOX2_KERNELS` (auto, numpy or numba; default auto: numba when it imports).

    Both modules have the same functions; `BACKEND` names the one returned. The first numba call compiles (or loads from
    numba's cache) every kernel on a 3 x 2 problem, so call this in the parent before a process pool starts.
    """
    choice = os.environ.get("EPFTOOLBOX2_KERNELS", "auto").strip().lower() or "auto"
    if choice not in KERNELS:
        raise ValueError(f"EPFTOOLBOX2_KERNELS must be one of {KERNELS}, not {choice!r}")
    this = sys.modules[__name__]
    if choice == "numpy":
        return this
    if not has_extra("fast"):
        if choice == "numba":
            raise MissingExtraError("EPFTOOLBOX2_KERNELS=numba", "fast")
        return this
    try:
        return _numba_kernels()
    except ImportError as error:
        if choice == "numba":
            raise MissingExtraError("EPFTOOLBOX2_KERNELS=numba", "fast") from error
        warnings.warn(f"numba is installed but doesn't import ({error}); using the numpy kernels", RuntimeWarning)
        return this


@functools.lru_cache(maxsize=None)
def _numba_kernels():
    from . import _lars_numba

    _lars_numba.warm_up()
    return _lars_numba
