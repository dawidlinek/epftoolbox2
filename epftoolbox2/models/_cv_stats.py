"""Cross-validation blocks for the step-5 engine, and `LassoCV` fitted from them.

A fit with an inner cross-validation (`LassoCVModel`, `LEARModel(criterion="cv")`) is scored on KFold's blocks of its rows ρ:
`folds` (5) contiguous blocks in row (date) order, the first n % folds of them one row longer, no shuffling. Its statistics
are those of `_stats.standard_stats` for the whole of ρ plus the same map for each block:

- `block_rows(rows)` cuts ρ (span row indices, ascending) into the blocks; `SpanSums.over(block, cols)` gives each block's
  `RawSums`, about the span's shifts like ρ's own.
- `block_stats(raw, blocks, params, values)` maps them with the fit's scaler parameters (`StandardParams`, from ρ, never
  from a block): z = ((x - c) - o)/σ, so a block's mean of a scaled column is (u_b/r_b - o)/σ, of an unscaled one c + u_b/r_b,
  of the target (Σ_b(y - c_y)/r_b - Σ(y - c_y)/r)/σ_y; its centred sums are G_b = (S_b - u_b u_b'/r_b)/(σσ') and so on, as
  `standard_stats` centres ρ's. A column the scaler zeroes (constant on ρ) has zero sums in every block.
- `cv_stats_of(stats)` reads them as LEAR's `CvStats`; `lassocv_fit(stats)` is `LassoCV` from them.

**A column constant on a block.** About the span's shift, S_b - u_b²/r_b of a column constant on the block cancels to noise of
about eps·r_b·(x - c)², not to 0, while the model path's own centring of those rows gives 0 up to eps². The noise is harmless
inside a training part where the column varies (it is below the model path's own rounding there), but where the column is
constant on a whole training part it is all there is: LEAR's fold rule (`_lars._constant_dropped`, std at most 1e-12 of the
rms) would no longer drop the column and would normalise the noise into a regressor. So each block's columns that the sums
can't call varying (`_stats`'s candidate rule on the block) are read on the block's rows: equal values make the column
exactly constant there (its row and column of G_b and q_b are 0, its mean the scaled value itself); unequal values with a
variance from the sums that is not positive raise `StatsFallback("cancellation")`.

**Cancellation, per fold.** Every non-constant block's centring errs by about eps·S_b,jj/σ_j² in absolute terms. A fold reads
the sum of its training blocks, G_tr = Σ G_b + Σ r_b (z̄_b - z̄_tr)(z̄_b - z̄_tr)' (`_lars.pooled`), and the held-out block's sums
about the training means, A = G_k + r_k (z̄_k - z̄_tr)(z̄_k - z̄_tr)'. Where Σ S_b,jj/σ_j² over the part's non-constant blocks
exceeds `CANCELLATION_MAX` (1e4) times G_tr,jj (or A_jj), and likewise for the target, that part's statistics can't be formed
to the model path's digits: `StatsFallback("cancellation")`. A column constant on a whole part has no noise there. This is
the rule `standard_stats` applies to ρ, at the level the folds read; it also catches a target constant on ρ, which
`standard_stats` lets through with the scale 1.0.

`LassoCV(cv=folds, max_iter, n_jobs=1, fit_intercept)` from the statistics (`lassocv_fit`, decision 15 of the step-5 spec), as
scikit-learn 1.8.0's `LinearModelCV.fit` runs it (`_coordinate_descent.py:1617-1936`):

1. The grid: `_alpha_grid`'s 100 penalties from α_max = max_j |Xy_j| / n down to 1e-3 α_max (`np.geomspace`, lines 96-213),
   with Xy the window's centred q (or Z'y without an intercept).
2. Per fold, `_path_residuals` (lines 1399-1538) with its `_pre_fit` Gram (`_base.py:781-877`: precompute "auto" is the Gram
   when the training part has more rows than columns): scikit-learn's own coordinate descent through the public
   `enet_path(X, y, l1_ratio=1, alphas=grid, precompute=G_tr, Xy=q_tr, check_input=False)`, the path warm-started from zeros
   along the grid. Its Gram kernel (`_cd_fast.pyx:959-1142`) reads X only for its shape (`enet_path`, lines 604 and 672:
   `l1_reg = alpha * n_samples`) and y only for y·y (line 1008: the stopping tolerance tol·y·y and the duality gap), so X is a
   shape-only array and y has the training part's n_tr values, √yy_tr and zeros (a scikit-learn that read y's length would
   still see n_tr). y·y is then yy_tr exactly where a float's square is (about half the values) and 1 ulp off elsewhere, below
   the model path's own rounding of y·y.
3. The held-out mean squared error at every penalty, with `_path_residuals`' intercept ȳ_tr - z̄_tr·w (none without an
   intercept): (w'Aw - 2b'w + c) / r_k from the block's sums about the training part's means. Then the mean over the folds, each
   fold counted once, and its first argmin (`np.argmin`).
4. The fit at that penalty on the window's Gram, from zeros; the intercept ȳ - z̄·w.

Products outside scikit-learn's kernel (the held-out errors, the intercept) are `np.einsum` loops, not BLAS: a threaded GEMM
sums in another order at another thread count (at p 247, 1 and 8 OpenBLAS threads gave held-out errors 4.4e-16 apart), and
those bits reach the choice of penalty. scikit-learn's kernel calls level-1 BLAS on p-vectors only, which OpenBLAS
threads on long vectors alone: its paths were bitwise the same at 1, 4 and 8 threads at p 247 (the tests check 1 and 4).

What differs from the model path: in the folds only rounding (the Gram from exact sums, not BLAS on the scaled rows; the same
kernel); in the final fit also the kernel's form, since `LassoCV` refits on the rows (`precompute="auto"` becomes False
there, line 1922). The two forms make the same coordinate updates in exact arithmetic except for their gap-safe screening:
the row form never takes back a feature it screened out (`_cd_fast.pyx:368-369`), the Gram form tests every feature again
(`screen_features_enet_gram`, lines 917-956). A penalty or a stopping sweep can therefore differ only where two candidates tie
to the last bit (the step-5 spec's choice flips; counted by the tests and the research sweeps).

Hand-backs (None: the model path fits it), each a function of the fit's own statistics: no blocks, or another number of them
than `folds`; a fold whose training part has no more rows than columns (the model path's descent runs on the rows there,
another rounding and stopping sequence; this includes n <= p); sums, a held-out error or coefficients that aren't finite; a
yy that isn't positive; and α_max at or below float64's resolution (scikit-learn's degenerate grid).
"""

import math
from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import enet_path

from ._lars import CV_FOLDS, CvStats, GramStats, pooled
from ._stats import (
    CANCELLATION_MAX,
    LinearFit,
    MIN_ROWS,
    ScaledStats,
    StandardParams,
    StatsFallback,
    Values,
    _candidates,
    _finite,
    _reader,
)
from ._sums import RawSums

GRID_EPS = 1e-3
"""`LassoCV`'s `eps`: the grid's smallest penalty over its largest."""

N_ALPHAS = 100
"""`LassoCV`'s number of penalties (its `alphas=100`, the default)."""

TOL = 1e-4
"""`LassoCV`'s `tol`: the coordinate descent stops when its duality gap is at most tol·y·y."""

RESOLUTION = float(np.finfo(np.float64).resolution)


def fold_sizes(n: int, folds: int = CV_FOLDS) -> np.ndarray:
    """KFold's block sizes for `n` rows: n // folds each, one more for the first n % folds."""
    sizes = np.full(folds, n // folds, dtype=np.int64)
    sizes[: n % folds] += 1
    return sizes


def _edges(n: int, folds: int) -> np.ndarray:
    return np.r_[0, np.cumsum(fold_sizes(n, folds))]


def block_rows(rows, folds: int = CV_FOLDS) -> Tuple[np.ndarray, ...]:
    """The fit's rows ρ (span row indices in ascending order, or a boolean mask over the span) cut into `folds` contiguous
    blocks of ρ in that order, KFold's sizes: the rows each block's `SpanSums.over` sums. Fewer rows than folds raise
    ValueError, as KFold does."""
    rows = np.asarray(rows)
    if rows.dtype == bool:
        rows = np.flatnonzero(rows)
    elif rows.size and not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("rows must be a boolean mask or integer indices")
    rows = rows.astype(np.int64).reshape(-1)
    if folds < 2:
        raise ValueError(f"cross-validation needs at least 2 folds, got {folds}")
    if rows.size < folds:
        raise ValueError(f"{folds} CV blocks need at least {folds} rows, got {rows.size}")
    if rows.size > 1 and not (np.diff(rows) > 0).all():
        raise ValueError("rows must be distinct and ascending (ρ in row order)")
    edges = _edges(rows.size, folds)
    return tuple(rows[a:b] for a, b in zip(edges[:-1], edges[1:]))


def block_stats(
    raw: RawSums, blocks: Sequence[RawSums], params: StandardParams, values: Values
) -> Tuple[ScaledStats, ...]:
    """The CV blocks of a fit as `ScaledStats` (see the module): `raw` the fit's sums over ρ, `blocks` those over each of
    `block_rows(ρ)` (same columns, same span), `params` the fit's scaler parameters from `standard_stats(raw, ...)`, and
    `values` its rows on ρ as `standard_stats` takes them (the pair (x, y), or `values(j)` and `values(None)`), read only for
    the columns a block's sums can't call varying.

    Raises `StatsFallback` where the blocks can't be formed to the model path's digits (a sum that isn't finite, a
    cancellation in a block or a fold), and ValueError where `blocks` aren't KFold's blocks of `raw`'s rows (a caller's bug)."""
    r = int(raw.n)
    shift = np.asarray(raw.shift, dtype=np.float64)
    p = shift.size
    folds = len(blocks)
    sizes = [int(b.n) for b in blocks]
    if folds < 2 or sizes != fold_sizes(r, folds).tolist():
        raise ValueError(f"blocks of {sizes} rows are not KFold's {folds} blocks of the fit's {r} rows")
    for b in blocks:
        if not (np.array_equal(np.asarray(b.shift), shift) and b.yshift == raw.yshift):
            raise ValueError("every block's sums must be over the fit's span and columns (the same shifts)")
    column, _ = _reader(values, r, p)
    edges = _edges(r, folds)
    sigma = np.asarray(params.scale, dtype=np.float64)
    ystd = float(params.yscale)
    zeroed = np.asarray(params.scaled, dtype=bool) & np.asarray(
        params.constant, dtype=bool
    )  # 0.0 in every training row
    forced = ~np.asarray(params.scaled, dtype=bool) & np.asarray(params.constant, dtype=bool)  # its value in every row
    yoffset = float(raw.sy) / r

    out = []
    noise = np.zeros(
        (folds, p)
    )  # Σ (x - c)² / σ² of the block where it isn't constant: the scale of its centring error
    ynoise = np.zeros(folds)
    for k, blk in enumerate(blocks):
        rb = sizes[k]
        u = np.asarray(blk.u, dtype=np.float64)
        S = np.asarray(blk.S, dtype=np.float64)
        v = np.asarray(blk.v, dtype=np.float64)
        sy, syy = float(blk.sy), float(blk.syy)
        if not (
            np.isfinite(u).all() and np.isfinite(S).all() and np.isfinite(v).all() and np.isfinite([sy, syy]).all()
        ):
            raise StatsFallback("overflow", f"a sum of CV block {k} is not finite")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            C = S - np.outer(u, u) / rb
            cq = v - u * (sy / rb)
            cyy = syy - sy * sy / rb
            dbar = u / rb
            var = np.diag(C) / (rb - 1) if rb > 1 else np.full(p, np.nan)
        square = np.diag(S).copy()
        zbar = np.where(params.scaled, (dbar - params.offset) / sigma, shift + dbar)
        constant = zeroed.copy()
        candidates = (_candidates(var, shift + dbar, square, rb) | forced) & ~zeroed
        for j in np.flatnonzero(candidates).tolist():
            col = column(j)[edges[k] : edges[k + 1]]
            if col.min() == col.max():
                constant[j] = True
                zbar[j] = ((col[0] - params.shift[j]) - params.offset[j]) / params.scale[j]
            elif not var[j] > 0:
                raise StatsFallback(
                    "cancellation",
                    f"CV block {k}, column {j}: variance {var[j]!r} from the sums, not constant on its rows",
                )
        zbar[zeroed] = 0.0
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            G = C / np.outer(sigma, sigma)
            q = cq / (sigma * ystd)
            yy = 0.0 if rb == 1 else cyy / (ystd * ystd)
            ybar = (sy / rb - yoffset) / ystd
            noise[k] = np.where(constant, 0.0, square / (sigma * sigma))
            ynoise[k] = 0.0 if rb == 1 else syy / (ystd * ystd)
        G[constant, :] = 0.0
        G[:, constant] = 0.0
        q[constant] = 0.0
        if not (
            np.isfinite(G).all() and np.isfinite(q).all() and np.isfinite(zbar).all() and np.isfinite([yy, ybar]).all()
        ):
            raise StatsFallback("overflow", f"a scaled sum or mean of CV block {k} is not finite")
        out.append(ScaledStats(n=rb, zbar=zbar, ybar=float(ybar), G=G, q=q, yy=float(yy), yscale=params.yscale))
    _check_folds(out, noise, ynoise)
    return tuple(out)


def _check_folds(blocks: Sequence[ScaledStats], noise: np.ndarray, ynoise: np.ndarray) -> None:
    """`StatsFallback("cancellation")` where a fold's training part or held-out block can't be centred to the model path's
    digits: its blocks' centring noise above `CANCELLATION_MAX` times the centred sum the fold reads (see the module)."""
    n = np.array([b.n for b in blocks], dtype=np.float64)
    zbar = np.array([b.zbar for b in blocks])
    ybar = np.array([b.ybar for b in blocks])
    gdiag = np.array([np.diag(b.G) for b in blocks])
    yy = np.array([b.yy for b in blocks])
    for k in range(len(blocks)):
        train = np.arange(len(blocks)) != k
        nt = n[train].sum()
        zt = (n[train, None] * zbar[train]).sum(axis=0) / nt
        yt = float((n[train] * ybar[train]).sum() / nt)
        x_train = gdiag[train].sum(axis=0) + (n[train, None] * (zbar[train] - zt) ** 2).sum(axis=0)
        x_test = gdiag[k] + n[k] * (zbar[k] - zt) ** 2
        y_train = yy[train].sum() + float((n[train] * (ybar[train] - yt) ** 2).sum())
        y_test = yy[k] + n[k] * (ybar[k] - yt) ** 2
        parts = (
            ("training part", noise[train].sum(axis=0), x_train, ynoise[train].sum(), y_train),
            ("held-out block", noise[k], x_test, ynoise[k], y_test),
        )
        for what, err, centred, yerr, ycentred in parts:
            with np.errstate(invalid="ignore", over="ignore"):
                bad = np.flatnonzero((err > 0) & ~(err <= CANCELLATION_MAX * centred))
            if bad.size:
                j = int(bad[0])
                raise StatsFallback(
                    "cancellation",
                    f"fold {k}'s {what}, column {j}: centring noise {err[j]:.3g} against a centred sum of {centred[j]:.3g}",
                )
            if yerr > 0 and not yerr <= CANCELLATION_MAX * ycentred:
                raise StatsFallback(
                    "cancellation",
                    f"fold {k}'s {what}, target: centring noise {yerr:.3g} against a centred sum of {ycentred:.3g}",
                )


def cv_stats_of(stats: ScaledStats) -> CvStats:
    """LEAR's `CvStats` from `ScaledStats` with blocks (`block_stats`): each block's sums under `GramStats`' names."""
    if stats.blocks is None:
        raise TypeError("criterion='cv' fits from the CV blocks, and these ScaledStats carry none (see block_stats)")
    return CvStats(tuple(block.gram() for block in stats.blocks))


# --- LassoCV ------------------------------------------------------------------------------------------------------------


def alpha_grid(Xy, n: int, eps: float = GRID_EPS, n_alphas: int = N_ALPHAS) -> np.ndarray:
    """scikit-learn's `_alpha_grid` for the lasso from a given Xy (`l1_ratio=1`, no sample weights): `np.geomspace` from
    α_max = sqrt(max_j Xy_j²) / n down to eps·α_max, or `n_alphas` times float64's resolution where α_max is at most it."""
    Xyw = np.asarray(Xy, dtype=np.float64).reshape(-1)[:, np.newaxis]
    alpha_max = np.sqrt(np.max(np.abs(np.sum(Xyw**2, axis=1)))) / (n * 1)
    if alpha_max <= RESOLUTION:
        return np.full(n_alphas, RESOLUTION)
    return np.geomspace(alpha_max, alpha_max * eps, num=n_alphas)


def _norm_vector(yy: float, n: int) -> Optional[np.ndarray]:
    """A y of `n` values with y·y as close to yy as one float's square comes (the coordinate descent reads y only for y·y):
    √yy, or a neighbour whose square is yy exactly, then zeros. A neighbour hits yy for about half the values (51 % of
    200,000 random ones); elsewhere y·y is 1 ulp off, below the model path's own rounding of y·y. None where yy isn't
    positive and finite."""
    if not (math.isfinite(yy) and yy > 0.0):
        return None
    s = math.sqrt(yy)
    for t in (s, float(np.nextafter(s, 0.0)), float(np.nextafter(s, np.inf))):
        if t * t == yy:
            s = t
            break
    y = np.zeros(max(int(n), 1))
    y[0] = s
    return y


def _path(Q: np.ndarray, Xy: np.ndarray, yy: float, n: int, alphas: np.ndarray, max_iter: int, tol: float):
    """`enet_path`'s lasso coefficients (p, len(alphas)) on the Gram `Q`, `Xy` and y·y = `yy` of `n` rows, or None where yy
    isn't positive. X is a shape-only array: with `check_input=False` and a Gram the path reads its shape alone."""
    y = _norm_vector(yy, n)
    if y is None:
        return None
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    Xy = np.ascontiguousarray(Xy, dtype=np.float64)
    X = np.broadcast_to(np.float64(np.nan), (int(n), Q.shape[0]))
    _, coefs, _ = enet_path(
        X,
        y,
        l1_ratio=1.0,
        alphas=alphas,
        precompute=Q,
        Xy=Xy,
        copy_X=False,
        check_input=False,
        max_iter=int(max_iter),
        tol=tol,
        selection="cyclic",
    )
    return coefs


def _part(part: GramStats, fit_intercept: bool) -> Tuple[np.ndarray, np.ndarray, float]:
    """The Gram, Xy and y·y a path reads from rows' centred sums: as they are with an intercept (`_pre_fit` centres the
    rows), uncentred (Z'Z, Z'y, y'y) without."""
    if fit_intercept:
        return part.G, part.q, float(part.yy)
    n = float(part.n)
    return (
        part.G + n * np.outer(part.xbar, part.xbar),
        part.q + (n * part.ybar) * part.xbar,
        float(part.yy + n * part.ybar * part.ybar),
    )


def cv_mse(stats: ScaledStats, grid: np.ndarray, fit_intercept: bool, max_iter: int, tol: float = TOL):
    """Each fold's held-out mean squared error at every penalty of `grid`, (folds, len(grid)), as `_path_residuals` computes
    it, from the blocks' sums; None where a fold's path can't run (a training yy that isn't positive)."""
    blocks = stats.blocks
    folds = len(blocks)
    p = np.asarray(stats.q).size
    mse = np.empty((folds, grid.size))
    for k in range(folds):
        train = pooled([b.gram() for i, b in enumerate(blocks) if i != k])
        test = blocks[k]
        Q, Xy, yy = _part(train, fit_intercept)
        coefs = _path(Q, Xy, yy, train.n, grid, max_iter, tol)
        if coefs is None:
            return None
        xoff, yoff = (train.xbar, train.ybar) if fit_intercept else (np.zeros(p), 0.0)
        dx = test.zbar - xoff
        dy = test.ybar - yoff
        A = test.G + test.n * np.outer(dx, dx)
        b = test.q + (test.n * dy) * dx
        c = float(test.yy + test.n * dy * dy)
        own = np.einsum(
            "ji,ji->i", coefs, np.einsum("jk,ki->ji", A, coefs)
        )  # w'Aw per penalty, no BLAS (see the module)
        sse = own - 2.0 * np.einsum("j,ji->i", b, coefs) + c
        mse[k] = sse / test.n
    return mse


def lassocv_fit(
    stats: ScaledStats,
    fit_intercept: bool = True,
    max_iter: int = 10000,
    folds: int = CV_FOLDS,
    tol: float = TOL,
    eps: float = GRID_EPS,
    n_alphas: int = N_ALPHAS,
) -> Optional[LinearFit]:
    """`LassoCV(cv=folds, max_iter=max_iter, n_jobs=1, fit_intercept=fit_intercept)` from a fit's scaled sums and its CV
    blocks (see the module), or None where the model path must fit it. `alpha_` of the fit is the chosen penalty."""
    blocks = stats.blocks
    n = int(stats.n)
    p = np.asarray(stats.q).size
    if blocks is None or len(blocks) != folds or n < max(MIN_ROWS, folds):
        return None
    sizes = np.array([int(b.n) for b in blocks])
    if int(sizes.sum()) != n or n - int(sizes.max()) <= p:
        return None
    if not (_finite(stats) and all(_finite(b) for b in blocks)):
        return None
    Q, Xy, yy = _part(stats.gram(), fit_intercept)
    grid = alpha_grid(Xy, n, eps, n_alphas)
    if grid[0] <= RESOLUTION:
        return None
    mse = cv_mse(stats, grid, fit_intercept, max_iter, tol)
    if mse is None or not np.isfinite(mse).all():
        return None
    index = int(np.argmin(mse.mean(axis=0)))
    alpha = float(grid[index])
    coefs = _path(Q, Xy, yy, n, np.array([alpha]), max_iter, tol)
    if coefs is None or not np.isfinite(coefs).all():
        return None
    w = np.ascontiguousarray(coefs[:, 0])
    intercept = float(stats.ybar - np.einsum("j,j->", stats.zbar, w)) if fit_intercept else 0.0
    return LinearFit(w, intercept, alpha)


__all__ = [
    "GRID_EPS",
    "N_ALPHAS",
    "TOL",
    "alpha_grid",
    "block_rows",
    "block_stats",
    "cv_mse",
    "cv_stats_of",
    "fold_sizes",
    "lassocv_fit",
]
