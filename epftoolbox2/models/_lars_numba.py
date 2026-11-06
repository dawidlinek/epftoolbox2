"""numba twins of `_lars`'s kernels (the `[fast]` extra). Import this only through `_lars._kernels()`.

Every function here has the signature, the rules and the return types of its namesake in `_lars`, which is the reference and
documents them. The loops follow the reference step by step; their sums run in loop order where the reference's run in BLAS
order, so the two agree within 1e-9 with the same kinks, not to the bit. `gram_stats` is the reference's own: it is one BLAS
product, which numba wouldn't speed up.

The kernels are compiled with `error_model="numpy"`, so a division by zero gives inf or nan as in numpy (the step-length rules
rely on it) instead of raising.
"""

import math

import numpy as np
from numba import njit

from ._lars import (
    CRITERIA,
    DBL_MAX,
    DEGENERATE,
    EPS,
    EQUALITY_TOLERANCE,
    STOPS,
    TINY32,
    CdResult,
    GramStats,
    IcChoice,
    LarsPath,
    LearFit,
    _check_choices,
    _checked,
    _penalty,
    gram_stats,
)

BACKEND = "numba"

__all__ = ["BACKEND", "GramStats", "LarsPath", "IcChoice", "CdResult", "LearFit", "gram_stats", "lars_path", "path_at"]
__all__ += ["lasso_at", "ic_select", "cd_lasso", "lear_fit", "warm_up"]

_jit = njit(cache=True, nogil=True, fastmath=False, error_model="numpy")


@_jit
def _swap(cov, indices, G, i, j):
    t = cov[i]
    cov[i] = cov[j]
    cov[j] = t
    k = indices[i]
    indices[i] = indices[j]
    indices[j] = k
    p = G.shape[0]
    for c in range(p):
        t = G[i, c]
        G[i, c] = G[j, c]
        G[j, c] = t
    for r in range(p):
        t = G[r, i]
        G[r, i] = G[r, j]
        G[r, j] = t


@_jit
def _move_to(G, indices, ii, end):
    """Position ii of the permuted problem moves to `end` (> ii or equal); those between move up one."""
    p = G.shape[0]
    k = indices[ii]
    for i in range(ii, end):
        indices[i] = indices[i + 1]
    indices[end] = k
    for c in range(p):
        t = G[ii, c]
        for i in range(ii, end):
            G[i, c] = G[i + 1, c]
        G[end, c] = t
    for r in range(p):
        t = G[r, ii]
        for i in range(ii, end):
            G[r, i] = G[r, i + 1]
        G[r, end] = t


@_jit
def _chol_solve(L, k, b, out):
    """out = (L L')^-1 b for the lower factor L[:k, :k] (LAPACK potrs). Both solves read L by rows."""
    for i in range(k):
        t = b[i]
        for j in range(i):
            t -= L[i, j] * out[j]
        out[i] = t / L[i, i]
    for i in range(k - 1, -1, -1):
        out[i] /= L[i, i]
        x = out[i]
        for j in range(i):
            out[j] -= L[i, j] * x


@_jit
def _cholesky_delete(L, n, go_out):
    for i in range(go_out, n - 1):
        for j in range(i + 2):
            L[i, j] = L[i + 1, j]
    for i in range(go_out, n - 1):
        a = L[i, i]
        b = L[i, i + 1]
        r = math.hypot(a, b)
        if r > 0.0:
            c = a / r
            s = b / r
        else:
            c = 1.0
            s = 0.0
        L[i, i] = r
        L[i, i + 1] = 0.0
        for row in range(i + 1, n - 1):
            x = L[row, i]
            y = L[row, i + 1]
            L[row, i] = c * x + s * y
            L[row, i + 1] = c * y - s * x


@_jit
def _lars_kernel(G_in, q_in, n, alpha_min, max_steps):
    p = q_in.shape[0]
    G = G_in.copy()
    q = q_in.copy()
    for j in range(p):
        if not G[j, j] > 0.0:
            for k in range(p):
                G[j, k] = 0.0
                G[k, j] = 0.0
            q[j] = 0.0
    gram0 = G.copy()
    q0 = q.copy()
    cov = q.copy()
    indices = np.arange(p)
    max_features = min(max_steps, p)
    size = max(max_features, 1)
    alphas = np.zeros(max_features + 1)
    coefs = np.zeros((max_features + 1, p))
    L = np.zeros((size, size))
    sign_active = np.zeros(size)
    active = np.zeros(size, np.int64)
    least_squares = np.zeros(size)
    corr_eq_dir = np.zeros(p)
    idx = np.zeros(size, np.int64)
    dropped = np.zeros(size, np.int64)
    n_drop = 0
    n_iter = 0
    n_active = 0
    drop = False
    stop = 0
    c_idx = 0
    c_signed = 0.0

    while True:
        if n_active < p:
            c_idx = n_active
            C = abs(cov[n_active])
            for j in range(n_active + 1, p):
                a = abs(cov[j])
                if a > C:
                    C = a
                    c_idx = j
            c_signed = cov[c_idx]
        else:
            C = 0.0
        alpha = C / n
        alphas[n_iter] = alpha
        if alpha <= alpha_min + EQUALITY_TOLERANCE:
            if abs(alpha - alpha_min) > EQUALITY_TOLERANCE:
                if n_iter > 0:
                    ss = (alphas[n_iter - 1] - alpha_min) / (alphas[n_iter - 1] - alpha)
                    for j in range(p):
                        coefs[n_iter, j] = coefs[n_iter - 1, j] + ss * (coefs[n_iter, j] - coefs[n_iter - 1, j])
                alphas[n_iter] = alpha_min
            stop = 0
            break
        if n_iter >= max_steps or n_active >= p:
            stop = 1 if n_iter >= max_steps else 2
            break
        if not drop:
            m = n_active
            sign_active[m] = 1.0 if c_signed > 0.0 else (-1.0 if c_signed < 0.0 else 0.0)
            _swap(cov, indices, G, m, c_idx)
            c = G[m, m]
            for i in range(m):
                t = G[m, i]
                for k in range(i):
                    t -= L[i, k] * L[m, k]
                L[m, i] = t / L[i, i]
            v = 0.0
            for k in range(m):
                v += L[m, k] * L[m, k]
            diag = max(math.sqrt(abs(c - v)), EPS)
            L[m, m] = diag
            if diag < DEGENERATE:  # 0.24.2's "drop for good", with its swap (see _lars.lars_path)
                cov[m] = 0.0
                t = cov[m]
                cov[m] = cov[c_idx]
                cov[c_idx] = t
                continue
            active[n_active] = indices[m]
            n_active += 1

        if n_iter > 0 and alphas[n_iter - 1] < alpha:
            stop = 3
            break

        _chol_solve(L, n_active, sign_active, least_squares)
        if n_active == 1 and least_squares[0] == 0.0:
            least_squares[0] = 1.0
            AA = 1.0
        else:
            s = 0.0
            for i in range(n_active):
                s += least_squares[i] * sign_active[i]
            AA = 1.0 / math.sqrt(s) if s > 0.0 else math.inf
            if not math.isfinite(AA):
                L_ = L[:n_active, :n_active].copy()
                step = 0
                while not math.isfinite(AA):
                    for i in range(n_active):
                        L_[i, i] += (2.0**step) * EPS
                    _chol_solve(L_, n_active, sign_active, least_squares)
                    s = 0.0
                    for i in range(n_active):
                        s += least_squares[i] * sign_active[i]
                    AA = 1.0 / math.sqrt(max(s, EPS))
                    step += 1
            for i in range(n_active):
                least_squares[i] *= AA

        for j in range(n_active, p):
            corr_eq_dir[j] = 0.0
        for i in range(n_active):
            x = least_squares[i]
            for j in range(n_active, p):
                corr_eq_dir[j] += G[i, j] * x
        g1 = DBL_MAX
        g2 = DBL_MAX
        for j in range(n_active, p):
            t = corr_eq_dir[j]
            r = (C - cov[j]) / (AA - t + TINY32)
            if 0.0 < r < g1:
                g1 = r
            r = (C + cov[j]) / (AA + t + TINY32)
            if 0.0 < r < g2:
                g2 = r
        gamma = min(g1, g2, C / AA)
        drop = False
        z_pos = DBL_MAX
        for i in range(n_active):
            z = -coefs[n_iter, active[i]] / (least_squares[i] + TINY32)
            if 0.0 < z < z_pos:
                z_pos = z
        if z_pos < gamma:
            n_drop = 0
            for i in range(n_active - 1, -1, -1):
                if -coefs[n_iter, active[i]] / (least_squares[i] + TINY32) == z_pos:
                    idx[n_drop] = i
                    n_drop += 1
            for t_ in range(n_drop):
                sign_active[idx[t_]] = -sign_active[idx[t_]]
            gamma = z_pos
            drop = True

        n_iter += 1
        if n_iter >= alphas.shape[0]:
            grow = 2 * max(1, max_features - n_active)
            rows = alphas.shape[0] + grow
            new_alphas = np.zeros(rows)
            new_alphas[: alphas.shape[0]] = alphas
            alphas = new_alphas
            new_coefs = np.zeros((rows, p))
            new_coefs[: coefs.shape[0]] = coefs
            coefs = new_coefs
        for i in range(n_active):
            j = active[i]
            coefs[n_iter, j] = coefs[n_iter - 1, j] + gamma * least_squares[i]
        for j in range(n_active, p):
            cov[j] -= gamma * corr_eq_dir[j]

        if drop:
            for t_ in range(n_drop):
                _cholesky_delete(L, n_active, idx[t_])
            old_active = n_active
            n_active -= n_drop
            for t_ in range(n_drop):  # pop, highest position first
                ii = idx[t_]
                dropped[t_] = active[ii]
                for i in range(ii, old_active - 1 - t_):
                    active[i] = active[i + 1]
            for t_ in range(n_drop):
                end = old_active - 1 - t_
                _move_to(G, indices, idx[t_], end)
                d = dropped[t_]
                s = 0.0
                for k in range(p):
                    s += gram0[d, k] * coefs[n_iter, k]
                cov[end] = q0[d] - s
            for t_ in range(n_drop):
                ii = idx[t_]
                for i in range(ii, size - 1):
                    sign_active[i] = sign_active[i + 1]
                sign_active[size - 1] = 0.0

    m = n_iter + 1
    path_coefs = coefs[:m].copy()
    return alphas[:m].copy(), path_coefs, q0 - np.dot(path_coefs, gram0), stop


def lars_path(G, q, n, alpha_min: float = 0.0, max_steps: int = 2500) -> LarsPath:
    G, q, n = _checked(G, q, n)
    alphas, coefs, corr, stop = _lars_kernel(G, q, n, float(alpha_min), int(max_steps))
    return LarsPath(alphas, coefs, corr, STOPS[stop])


@_jit
def _path_at_kernel(alphas, coefs, alpha):
    m, p = coefs.shape
    out = np.zeros(p)
    if alpha >= alphas[0]:
        return out
    for i in range(1, m):
        if alphas[i] <= alpha:
            t = (alphas[i - 1] - alpha) / (alphas[i - 1] - alphas[i])
            for j in range(p):
                out[j] = coefs[i - 1, j] + t * (coefs[i, j] - coefs[i - 1, j])
            return out
    for j in range(p):
        out[j] = coefs[m - 1, j]
    return out


def path_at(path: LarsPath, alpha: float) -> np.ndarray:
    return _path_at_kernel(path.alphas, path.coefs, float(alpha))


@_jit
def _lasso_at_kernel(G, q, n, alpha, max_steps):
    alphas, coefs, _, _ = _lars_kernel(G, q, n, max(alpha - 2.0 * EQUALITY_TOLERANCE, 0.0), max_steps)
    reached = alpha >= alphas[0]
    for a in alphas:
        if a <= alpha:
            reached = True
    return _path_at_kernel(alphas, coefs, alpha), reached


def lasso_at(G, q, n, alpha: float, max_steps: int = 2500):
    G, q, n = _checked(G, q, n)
    w, reached = _lasso_at_kernel(G, q, n, float(alpha), int(max_steps))
    return w, bool(reached)


@_jit
def _ic_kernel(coefs, corr, q, yy, n, K):
    m, p = coefs.shape
    values = np.empty(m)
    best = 0
    for k in range(m):
        qw = 0.0
        cw = 0.0
        df = 0
        for j in range(p):
            w = coefs[k, j]
            qw += q[j] * w
            cw += corr[k, j] * w
            if abs(w) > EPS:
                df += 1
        rss = yy - qw - cw
        values[k] = n * (rss / n) / (yy / n + EPS) + K * df
        if values[k] < values[best]:
            best = k
    return values, best


def ic_select(path: LarsPath, q, yy: float, n, criterion: str) -> IcChoice:
    n = float(n)
    q = np.ascontiguousarray(q, dtype=np.float64)
    values, kink = _ic_kernel(path.coefs, path.corr, q, float(yy), n, _penalty(criterion, n))
    return IcChoice(float(path.alphas[kink]), int(kink), values)


@_jit
def _cd_kernel(Q, q, yy, n, alpha, max_iter, tol):
    p = q.shape[0]
    w = np.zeros(p)
    H = np.zeros(p)
    l1 = alpha * n
    gap_tol = tol * yy
    converged = False
    sweep = 0
    for sweep in range(max_iter):
        w_max = 0.0
        d_w_max = 0.0
        for j in range(p):
            Qjj = Q[j, j]
            if Qjj == 0.0:
                continue
            w_j = w[j]
            if w_j != 0.0:
                for k in range(p):
                    H[k] -= w_j * Q[j, k]
            tmp = q[j] - H[j]
            if tmp > 0.0:
                new = max(tmp - l1, 0.0) / Qjj
            elif tmp < 0.0:
                new = -max(-tmp - l1, 0.0) / Qjj
            else:
                new = 0.0
            w[j] = new
            if new != 0.0:
                for k in range(p):
                    H[k] += new * Q[j, k]
            d_w = abs(new - w_j)
            if d_w > d_w_max:
                d_w_max = d_w
            if abs(new) > w_max:
                w_max = abs(new)
        if w_max == 0.0 or d_w_max / w_max < tol or sweep == max_iter - 1:
            q_dot_w = 0.0
            w_dot_h = 0.0
            l1_norm = 0.0
            dual_norm = 0.0
            for j in range(p):
                q_dot_w += w[j] * q[j]
                w_dot_h += w[j] * H[j]
                l1_norm += abs(w[j])
                r = abs(q[j] - H[j])
                if r > dual_norm:
                    dual_norm = r
            r_norm2 = yy + w_dot_h - 2.0 * q_dot_w
            if dual_norm > l1:
                const = l1 / dual_norm
                gap = 0.5 * (r_norm2 + r_norm2 * const**2)
            else:
                const = 1.0
                gap = r_norm2
            gap += l1 * l1_norm - const * yy + const * q_dot_w
            if gap < gap_tol:
                converged = True
                break
    return w, sweep + 1, converged


def cd_lasso(G, q, yy: float, n, alpha: float, max_iter: int = 2500, tol: float = 1e-4) -> CdResult:
    Q, q, n = _checked(G, q, n)
    w, sweeps, converged = _cd_kernel(Q, q, float(yy), n, float(alpha), int(max_iter), float(tol))
    return CdResult(w, int(sweeps), bool(converged))


@_jit
def _lear_kernel(G, q, yy, n, K, use_cd):
    p = q.shape[0]
    norm = np.empty(p)
    for j in range(p):
        norm[j] = math.sqrt(G[j, j]) if G[j, j] > 0.0 else 1.0
    R = np.empty((p, p))
    qn = np.empty(p)
    for i in range(p):
        qn[i] = q[i] / norm[i]
        for j in range(p):
            R[i, j] = G[i, j] / (norm[i] * norm[j])
    alphas, coefs, corr, _ = _lars_kernel(R, qn, n, 0.0, 2500)
    _, kink = _ic_kernel(coefs, corr, qn, yy, n, K)
    alpha = alphas[kink]
    if use_cd:
        w, _, converged = _cd_kernel(G, q, yy, n, alpha, 2500, 1e-4)
    else:
        w, converged = _lasso_at_kernel(G, q, n, alpha, 2500)
    return w, alpha, kink, converged


def lear_fit(stats: GramStats, criterion: str = "bic", solver: str = "lars") -> LearFit:
    _check_choices(criterion, solver)
    G, q, n = _checked(stats.G, stats.q, stats.n)
    w, alpha, kink, converged = _lear_kernel(G, q, float(stats.yy), n, _penalty(criterion, n), solver == "cd")
    return LearFit(w, stats.xbar, stats.ybar, float(alpha), int(kink), bool(converged))


def warm_up() -> None:
    """Compile (or load from numba's cache) every kernel on a 3 x 2 problem."""
    stats = gram_stats(np.array([[1.0, 0.5], [0.0, 2.0], [3.0, 1.0]]), np.array([1.0, 2.0, 0.5]))
    for criterion in CRITERIA:
        for solver in ("lars", "cd"):
            lear_fit(stats, criterion, solver)
    path = lars_path(stats.G, stats.q, stats.n)
    ic_select(path, stats.q, stats.yy, stats.n, "bic")
    path_at(path, float(path.alphas[-1]))
    cd_lasso(stats.G, stats.q, stats.yy, stats.n, float(path.alphas[0]) / 2.0)
