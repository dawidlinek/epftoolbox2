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

`lear_fit` adds a third, before both stages: of columns that are exact twins on the training rows (copies or exact negatives,
|G_jk| >= (1 - 1e-12) sqrt(G_jj G_kk)), one is kept, the first in column order unless a later one is longer, and the others' rows,
columns and correlations are set to 0 (`without_twins`). LARS can't take a second twin: its entry meets the degenerate-entrant rule on a tie that the last bit decides,
the path then carries coefficients of about 1e14 that cancel between the twins, and its result is far from the lasso (scikit-learn's
own `lars_path` does the same). A dropped twin's coefficient is exactly 0, and since the lasso splits a twin's weight any way it
likes, the fit is a lasso solution, one whose forecast stays the same wherever the twins stay equal.

`criterion="cv"` is old `LassoLarsCV(cv=KFold(5), normalize=True, max_iter=2500)` instead (`lear_fit_cv`, from `CvStats`: the
centred sums of five contiguous blocks of the rows in date order): α* by blocked cross-validation on per-fold normalised paths,
the coefficients read from the full window's normalised path at α*, with no refit. Its departures are in `cv_folds`.

`_lars_numba.py` holds numba twins of the same functions. On designs without exact twins (in `lear_fit`, every design) the two agree
within 1e-9 with the same kinks, not to the bit: this module sums with BLAS, the twins in loops. `_kernels()` returns the module of
the backend `EPFTOOLBOX2_KERNELS` asks for.
"""

import functools
import math
import os
import sys
import warnings
from typing import List, NamedTuple, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import solve_triangular
from scipy.linalg.lapack import dpotrs

from .._extras import MissingExtraError, has_extra
from ..scalers.base import CONSTANT_STD_RTOL

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
IC_CRITERIA = ("aic", "bic")
CRITERIA = IC_CRITERIA + ("cv",)
SOLVERS = ("lars", "cd")

# Old `LassoLarsCV`'s settings as Lago's code would have set them: `cv=5` (KFold(5), no shuffling), `max_iter=2500`, and its
# `max_n_alphas=1000`.
CV_FOLDS = 5
CV_MAX_STEPS = 2500
MAX_N_ALPHAS = 1000


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
    kink: (
        int  # the stage-1 kink of α*; with "cv" its position in the grid of penalties with a finite error in every fold
    )
    converged: bool  # "lars": stage 2's path passed α*; "cd": the duality gap fell below tol; "cv": the path reached α*


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
    raise ValueError(f"an information criterion is one of {IC_CRITERIA}, not {criterion!r}")


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


# Two columns are twins when the cosine of their centred training values is 1 or -1 to this relative tolerance: one is the other
# times a constant (a copy, an exact negative, a daily value repeated on every slot), up to rounding.
TWIN_RTOL = 1e-12


def without_twins(G, q) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    """G and q with all but one column of each group of exact twins dropped: its row, column and correlation set to 0, so it
    never enters LARS or moves in coordinate descent and gets the coefficient 0. Twins are columns j < k with G[j, j], G[k, k] > 0
    and |G[j, k]| >= (1 - TWIN_RTOL) sqrt(G[j, j] G[k, k]); a group is a column and its later twins. The one kept is the first in
    column order of those whose norm is the group's largest to TWIN_RTOL: the lasso, on the unnormalised columns, puts a group's
    weight on its longest column (the cheapest in |w|), and is indifferent between columns of one norm, which is every group a
    scaler leaves (copies and exact negatives scale alike). Returns (G, q, the dropped columns); the inputs themselves when there
    are none."""
    G = np.asarray(G, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    d = np.diag(G)
    live = d > 0.0
    if q.size < 2 or live.sum() < 2:
        return G, q, ()
    scale = np.sqrt(np.where(live, d, 0.0))
    twin = (np.abs(G) >= (1.0 - TWIN_RTOL) * np.outer(scale, scale)) & np.outer(live, live)
    np.fill_diagonal(twin, False)
    if not twin.any():
        return G, q, ()
    dropped = np.zeros(q.size, dtype=bool)
    for j in np.flatnonzero(live):
        if dropped[j]:
            continue
        group = np.r_[j, np.flatnonzero(twin[j, j + 1 :] & ~dropped[j + 1 :]) + j + 1]
        if group.size == 1:
            continue
        norms = scale[group]
        kept = group[np.flatnonzero(norms >= (1.0 - TWIN_RTOL) * norms.max())[0]]
        dropped[group[group != kept]] = True
    G = G.copy()
    q = q.copy()
    G[dropped, :] = 0.0
    G[:, dropped] = 0.0
    q[dropped] = 0.0
    return G, q, tuple(int(j) for j in np.flatnonzero(dropped))


def _norms(G: np.ndarray) -> np.ndarray:
    """The column norms `_preprocess_data(normalize=True)` divided by: sqrt(G[j, j]), 0 -> 1."""
    d = np.diag(G)
    return np.where(d > 0.0, np.sqrt(np.where(d > 0.0, d, 1.0)), 1.0)


def _normalised(G: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """G and q of unit-norm columns, as `_preprocess_data(normalize=True)` scaled them: by sqrt(G[j, j]), 0 -> 1."""
    norm = _norms(G)
    return G / np.outer(norm, norm), q / norm


CV_SOLVER_ERROR = (
    "criterion='cv' reads the coefficients from the LARS path at its penalty, as old LassoLarsCV did, with no refit, so it "
    "takes solver='lars', not 'cd'"
)


def _check_choices(criterion: str, solver: str) -> None:
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, not {criterion!r}")
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, not {solver!r}")
    if criterion == "cv" and solver == "cd":
        raise ValueError(CV_SOLVER_ERROR)


def lear_fit(stats: Union[GramStats, "CvStats"], criterion: str = "bic", solver: str = "lars") -> LearFit:
    """A LEAR unit from its centred sums, exact twins dropped first (`without_twins`): α* by `ic_select` on the normalised path (2500 steps at most, as Lago's
    `LassoLarsIC(max_iter=2500)`), then the Lasso at α* on the centred, unnormalised Gram, exactly (`"lars"`: `lasso_at`, a
    second path run just past α*) or by Lago's coordinate descent (`"cd"`: `max_iter=2500`, `tol=1e-4`).

    `criterion="cv"` takes the blocks' sums (`CvStats`) and is `lear_fit_cv`."""
    _check_choices(criterion, solver)
    if criterion == "cv":
        return lear_fit_cv(_cv_stats_of(stats))
    if not isinstance(stats, GramStats):
        raise TypeError(f"criterion={criterion!r} fits from GramStats, got {type(stats).__name__}")
    G, q, _ = without_twins(stats.G, stats.q)
    R, qn = _normalised(G, q)
    choice = ic_select(lars_path(R, qn, stats.n), qn, stats.yy, stats.n, criterion)
    if solver == "lars":
        w, converged = lasso_at(G, q, stats.n, choice.alpha)
    else:
        w, _, converged = cd_lasso(G, q, stats.yy, stats.n, choice.alpha)
    return LearFit(w, stats.xbar, stats.ybar, choice.alpha, choice.kink, bool(converged))


# ---------------------------------------------------------------------------
# criterion="cv": old LassoLarsCV(cv=KFold(5), normalize=True, max_iter=2500)
# ---------------------------------------------------------------------------


class CvStats(NamedTuple):
    """A unit's training rows cut into `CV_FOLDS` contiguous blocks in their order (date order), as KFold(5) without shuffling
    cuts them, the first n % 5 blocks one row longer: each block's centred sums. Fold k trains on the other blocks and is scored
    on block k; the full window is all of them (`pooled`). Everything `lear_fit_cv` reads, as `GramStats` is for the criteria."""

    blocks: Tuple[GramStats, ...]


class CvFold(NamedTuple):
    """One fold of `lear_fit_cv`: the normalised Gram of its training part, and the held-out block's sums about the training
    part's means, from which the block's squared residuals at any coefficients w are `w'Aw - 2 b'w + c`."""

    n: int  # training rows
    R: np.ndarray  # the training part's G / (norm ⊗ norm), after its constant and twin columns are dropped
    qn: np.ndarray  # q / norm
    norm: np.ndarray  # the training part's column norms, 0 -> 1
    A: np.ndarray  # Σ (x - xbar_train)(x - xbar_train)' over the held-out rows
    b: np.ndarray  # Σ (x - xbar_train)(y - ybar_train)
    c: float  # Σ (y - ybar_train)²
    n_test: int  # held-out rows


def cv_stats(x, y, folds: int = CV_FOLDS) -> CvStats:
    """The centred sums of each of `folds` contiguous blocks of the rows `x` (n, p), `y` (n,), in row order: KFold's sizes, n //
    folds and one more for the first n % folds. Fewer rows than folds raises ValueError, as KFold did."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(f"cv_stats needs x of shape (n, p) and y of shape (n,), got {x.shape} and {y.shape}")
    n = x.shape[0]
    if n < folds:
        raise ValueError(f"criterion='cv' needs at least {folds} training rows for its {folds} folds, got {n}")
    sizes = np.full(folds, n // folds)
    sizes[: n % folds] += 1
    edges = np.r_[0, np.cumsum(sizes)]
    return CvStats(tuple(gram_stats(x[a:b], y[a:b]) for a, b in zip(edges[:-1], edges[1:])))


def _cv_stats_of(stats) -> CvStats:
    if not isinstance(stats, CvStats):
        raise TypeError(f"criterion='cv' fits from CvStats (cv_stats(x, y)), got {type(stats).__name__}")
    return stats


def pooled(blocks: Sequence[GramStats]) -> GramStats:
    """The centred sums of the union of `blocks`' rows, from theirs: G = Σ G_b + n_b (xbar_b - xbar)(xbar_b - xbar)', and so on."""
    n = sum(int(b.n) for b in blocks)
    xbar = sum(b.n * b.xbar for b in blocks) / n
    ybar = float(sum(b.n * b.ybar for b in blocks) / n)
    p = xbar.size
    G = np.zeros((p, p))
    q = np.zeros(p)
    yy = 0.0
    for b in blocks:
        dx = b.xbar - xbar
        dy = b.ybar - ybar
        G += b.G + b.n * np.outer(dx, dx)
        q += b.q + (b.n * dy) * dx
        yy += b.yy + b.n * dy * dy
    return GramStats(n, xbar, ybar, G, q, float(yy))


def _constant_dropped(stats: GramStats) -> Tuple[np.ndarray, np.ndarray]:
    """G and q with every column that is constant on these rows up to rounding dropped (row, column and q set to 0): its
    standard deviation at or below `CONSTANT_STD_RTOL` of its root mean square, the scalers' rule (`scalers.base`) on the sums.
    The scaler zeroes such a column over the whole window; a fold's training part can hold one the window doesn't (a dummy or
    a night-time solar column that varies only in the held-out block), and normalising its rounding noise to a unit column
    would make it a regressor that the held-out block multiplies by about 1e15. The inputs themselves when there is none."""
    d = np.diag(stats.G)
    constant = d <= CONSTANT_STD_RTOL**2 * (d + stats.n * stats.xbar**2)
    constant &= d != 0.0  # exact zeros never enter already
    if not constant.any():
        return stats.G, stats.q
    G = stats.G.copy()
    q = stats.q.copy()
    G[constant, :] = 0.0
    G[:, constant] = 0.0
    q[constant] = 0.0
    return G, q


def _design(stats: GramStats) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The normalised Gram, correlations and norms of rows' sums, after their constant columns and exact twins are dropped."""
    G, q = _constant_dropped(stats)
    G, q, _ = without_twins(G, q)
    norm = _norms(G)
    return G / np.outer(norm, norm), q / norm, norm


def cv_folds(stats: CvStats) -> Tuple[CvFold, ...]:
    """The folds of `lear_fit_cv`, numbered by their held-out block. Each training part is centred on its own means and
    normalised by its own norms, as `_lars_path_residues` did, after the rules of `lear_fit` applied to it as if it were the window:
    exact twins on those rows are dropped (`without_twins`), so a column can be a twin in one fold only. One rule more than
    0.24.2: a column constant on the training part up to rounding is dropped there (`_constant_dropped`)."""
    blocks = stats.blocks
    folds = []
    for k, test in enumerate(blocks):
        train = pooled([b for i, b in enumerate(blocks) if i != k])
        R, qn, norm = _design(train)
        dx = test.xbar - train.xbar
        dy = test.ybar - train.ybar
        A = test.G + test.n * np.outer(dx, dx)
        b = test.q + (test.n * dy) * dx
        c = float(test.yy + test.n * dy * dy)
        folds.append(CvFold(int(train.n), R, qn, norm, A, b, c, int(test.n)))
    return tuple(folds)


def cv_grid(fold_alphas: Sequence[np.ndarray], max_n_alphas: int = MAX_N_ALPHAS) -> np.ndarray:
    """`LarsCV.fit`'s penalties: the unique kinks of every fold's path, ascending, every `stride`-th of them kept, stride =
    int(len / max_n_alphas) or 1."""
    grid = np.unique(np.concatenate([np.asarray(a, dtype=np.float64) for a in fold_alphas]))
    stride = int(max(1, int(len(grid) / float(max_n_alphas))))
    return grid[::stride]


def _held_out_points(alphas: np.ndarray, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The points `LarsCV.fit` interpolates a fold's residues between, ascending: (α, the kink whose coefficients sit there).
    The kinks in ascending α, the last kink at α = 0 unless the path ends there, the first (all zero) at grid[-1] unless the
    path's largest α is it, then sorted stably, as `interp1d` sorts with mergesort; the appended point can land inside."""
    m = alphas.size
    x = alphas[::-1]
    kink = np.arange(m)[::-1]
    if x[0] != 0:
        x = np.r_[0.0, x]
        kink = np.r_[m - 1, kink]
    if x[-1] != grid[-1]:
        x = np.r_[x, grid[-1]]
        kink = np.r_[kink, 0]
    order = np.argsort(x, kind="mergesort")
    return x[order], kink[order]


def held_out_mse(alphas, coefs, fold: CvFold, grid) -> np.ndarray:
    """The held-out block's mean squared residual at each penalty of `grid`, as `LarsCV.fit` reads it off a fold's path
    (`alphas`, `coefs` of the normalised training part): scipy's linear `interp1d` of the residues between `_held_out_points`
    (`searchsorted` on the left, clipped to [1, len - 1]; a segment of zero width gives nan, which the caller drops), squared
    and averaged over the block's rows. The residues are linear in the coefficients, so this interpolates the coefficients,
    divided by the training part's norms, and takes the squares from the block's sums (`CvFold`)."""
    grid = np.asarray(grid, dtype=np.float64)
    x, kink = _held_out_points(np.asarray(alphas), grid)
    V = np.asarray(coefs)[kink] / fold.norm
    VA = V @ fold.A
    own = np.einsum("ij,ij->i", VA, V)
    beta = V @ fold.b
    if x.size == 1:  # every fold's path is the single kink α = 0, and so is the grid (0.24.2's interp1d raised there)
        return np.full(grid.size, (own[0] - 2.0 * beta[0] + fold.c) / fold.n_test)
    cross = np.einsum("ij,ij->i", VA[:-1], V[1:])
    hi = np.clip(np.searchsorted(x, grid), 1, x.size - 1)
    lo = hi - 1
    width = x[hi] - x[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (grid - x[lo]) / width
    s = 1.0 - t
    sse = s * s * own[lo] + 2.0 * (t * s) * cross[lo] + t * t * own[hi] - 2.0 * (s * beta[lo] + t * beta[hi]) + fold.c
    sse[width == 0.0] = np.nan
    return sse / fold.n_test


def cv_choice(grid: np.ndarray, mse: np.ndarray) -> Tuple[float, int]:
    """α* and its position: of the penalties whose held-out error is finite in every fold, the first argmin of the mean over the
    folds (each fold's mean squared residual counts once, whatever its block's length)."""
    finite = np.all(np.isfinite(mse), axis=-1)
    if not finite.any():
        raise ValueError("criterion='cv': no penalty has a finite held-out error in every fold")
    kept = np.asarray(grid)[finite]
    index = int(np.argmin(mse[finite].mean(axis=-1)))
    return float(kept[index]), index


def cv_window(stats: CvStats) -> Tuple[GramStats, np.ndarray, np.ndarray, np.ndarray]:
    """The full window's sums, and its normalised Gram, correlations and norms after `lear_fit`'s rules."""
    full = pooled(stats.blocks)
    R, qn, norm = _design(full)
    return full, R, qn, norm


def lear_fit_cv(stats: CvStats, max_n_alphas: int = MAX_N_ALPHAS) -> LearFit:
    """A LEAR unit by old `LassoLarsCV(cv=KFold(5), normalize=True, max_iter=2500)` (scikit-learn 0.24.2's `LarsCV.fit` and
    `_lars_path_residues`), from the blocks' sums.

    Each fold's normalised path runs to α = 0 (`cv_folds`, `lars_path`); the penalties are `cv_grid`'s; each fold's held-out
    error is read off its path at every one (`held_out_mse`); α* is `cv_choice`'s. The coefficients are the full window's
    normalised path stopped at α* (`lars_path(R, qn, n, α*)`, its last row), divided by the norms, which is what 0.24.2's
    `_fit(alpha=α*)` returned: its early stop takes a kink within float32 eps (1.19e-7) of α* as it is instead of the lasso at
    α* exactly. No refit. `kink` is α*'s position in the grid of finite errors, `converged` whether the path reached α*."""
    stats = _cv_stats_of(stats)
    folds = cv_folds(stats)
    paths = [lars_path(f.R, f.qn, f.n, 0.0, CV_MAX_STEPS) for f in folds]
    grid = cv_grid([path.alphas for path in paths], max_n_alphas)
    mse = np.column_stack([held_out_mse(path.alphas, path.coefs, f, grid) for path, f in zip(paths, folds)])
    alpha, index = cv_choice(grid, mse)
    full, R, qn, norm = cv_window(stats)
    path = lars_path(R, qn, full.n, alpha, CV_MAX_STEPS)
    return LearFit(path.coefs[-1] / norm, full.xbar, full.ybar, alpha, index, path.stop == "alpha_min")


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
