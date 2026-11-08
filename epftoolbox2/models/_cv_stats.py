"""Cross-validation blocks for the step-5 engine: KFold's blocks of a fit's rows from exact span sums.

A fit with an inner cross-validation (`LassoCVModel`, `LEARModel(criterion="cv")`) is scored on KFold's blocks of its rows ρ:
`folds` (5) contiguous blocks in row (date) order, the first n % folds of them one row longer, no shuffling. Its statistics
are those of `_stats.standard_stats` for the whole of ρ plus the same map for each block:

- `block_rows(rows)` cuts ρ (span row indices, ascending) into the blocks; `SpanSums.over(block, cols)` gives each block's
  `RawSums`, about the span's shifts like ρ's own.
- `block_stats(raw, blocks, params, values)` maps them with the fit's scaler parameters (`StandardParams`, from ρ, never
  from a block): z = ((x - c) - o)/σ, so a block's mean of a scaled column is (u_b/r_b - o)/σ, of an unscaled one c + u_b/r_b,
  of the target (Σ_b(y - c_y)/r_b - Σ(y - c_y)/r)/σ_y; its centred sums are G_b = (S_b - u_b u_b'/r_b)/(σσ') and so on, as
  `standard_stats` centres ρ's. A column the scaler zeroes (constant on ρ) has zero sums in every block.
- `cv_stats_of(stats)` reads them as LEAR's `CvStats`.

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
"""

from typing import Sequence, Tuple

import numpy as np

from ._lars import CV_FOLDS, CvStats
from ._stats import (
    CANCELLATION_MAX,
    ScaledStats,
    StandardParams,
    StatsFallback,
    Values,
    _candidates,
    _reader,
)
from ._sums import RawSums


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


__all__ = ["block_rows", "block_stats", "cv_stats_of", "fold_sizes"]
