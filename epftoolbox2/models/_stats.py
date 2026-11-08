"""Scaled statistics for the step-5 engine: `StandardScaler`'s map from a fit's exact sums, and the linear fits that read it.

`standard_stats(raw, values, mask, scale_binary)` turns `RawSums` over a fit's rows ρ (`_sums.SpanSums.over`: sums about the
span's shifts) into `ScaledStats`, the centred sums of the rows as `StandardScaler.fit_transform` would scale them, and
`StandardParams`, the scaler's parameters (to scale the issue row and map a forecast back). `OLSModel`, `RidgeModel` and
`LEARModel` fit from `ScaledStats` through `_fit_stats`; nothing in the run path calls them yet (step 5i will).

The map, per fit (r rows; c the span's shift of a column, d = x - c, u = Σd, S = Σdd', v = Σd(y - c_y)):

- **Mean and scale** of each column and of the target from the sums: μ = c + u/r, σ² = (S_jj - u_j²/r)/(r - 1) (ddof 1, as
  `mean_std`). The centred Gram of the scaled columns is G_jk = (S_jk - u_j u_k/r)/(σ_j σ_k), q_j = (v_j - u_j Σ(y - c_y)/r)/(σ_j
  σ_y) and yy = (Σ(y - c_y)² - (Σ(y - c_y))²/r)/σ_y², all shift-invariant. A scaled column has mean z̄ = 0 exactly (so
  `uncentred()`'s A = G + r z̄z̄' adds nothing to it), an unscaled one (a 0/1 column the mask leaves alone, centred on 0 and
  divided by 1) its mean on ρ; the target's scaled mean ȳ is 0.
- **Constant columns, decided as the model path decides them.** `mean_std` calls a column constant when its standard deviation is
  at most 1e-12 (`CONSTANT_STD_RTOL`) of its largest |x|; the scaler then zeroes its training values and divides its issue value
  by 1.0. The sums can't make that call themselves: about a shift far from the column's values on ρ (a 0/1 dummy that is 0 on ρ
  where the span's mean is 0.3), S_jj - u_j²/r cancels to noise of about √eps·|x - c|, some of it negative. So every column whose
  variance from the sums is not positive, or whose standard deviation is at most `CANDIDATE_RTOL` (1e-6) of max(|μ|, √(S_jj/r)),
  is a candidate, and every candidate is decided by `mean_std` on its own values in ρ (a column the scaler leaves alone by
  whether those values are all equal). √(S_jj/r) is the root mean square of x - c on ρ, the scale of that noise; |μ| stands
  for the largest |x| on ρ, which it equals to 1e-12·√(2r) wherever the rule could call the column constant. A constant
  column gets a zero row and column in G, q_j = 0, z̄_j = 0 (scaled) or its value (unscaled), and the scaler's own mean from
  `mean_std` (so its issue value is centred exactly as on the model path). The target is decided the same way and, as on the
  model path, isn't zeroed.
- **Precision.** For any other column the cancellation in S_jj - u_j²/r costs a relative error of about eps·κ_j in its
  variance and in its row of G, κ_j = S_jj / (S_jj - u_j²/r) = 1 + ((μ_j - c_j)/σ_j)²·(r - 1)/r: small wherever ρ's mean of the
  column is within a few of its standard deviations of the span's. A column (or the target) with κ above `CANCELLATION_MAX`
  (1e4), or a candidate that `mean_std` finds not constant while its variance from the sums is not positive, can't be scaled
  from the sums to the model path's digits: `standard_stats` raises `StatsFallback`, and that fit goes to the model path.
  Each is decided from the fit's own rows and span, so it is the same in any group.

The fits (`LinearFit` for OLS and Ridge; LEAR's `LearFit` from `ScaledStats.gram()`):

- **OLS**, n > p + 1: the Cholesky factor of G (with an intercept) or A (without), its zero columns (constant ones) left out
  with the coefficient 0, as the model path's minimum-norm `lstsq` gives an exact zero column. The fit is handed back (None) when
  the factorisation fails or (largest/smallest pivot)² exceeds `GUARD`: the normal equations err as eps·cond(Z)², gelsd as
  eps·cond(Z), and gelsd cuts its rank at max(n, p)·eps·σ_max, which a Gram can't resolve. `GUARD` is 1e5, set by a
  conditioning sweep (`epftoolbox2-research/library_check_step5/stats_fits/guard_sweep.py`; 9,588 fits of four design
  families, cond(Z) 1e2 to 1e8): with no guard the engine was up to 4.9e-10 from `LinearRegression` at cond(Z) 1e2 to 1e3,
  2.6e-7 at 1e3 to 1e4 and 8.5e-6 at 1e4 to 1e5; of the fits a pivot ratio of 1e5 let through (cond(Z) up to 1.5e4) none was
  over 7.9e-10, at 1e6 three were over 1e-8. The pivot ratio is at most cond(Z)², so every fit with cond(Z) below 316 is kept;
  the DE fixture's Lago-shaped rows (p 247, cond(Z) 1.7e3 to 6.2e3) have ratios of 2.3e3 to 1.2e4 and are kept.
- **Ridge**, n > p (scikit-learn's `svd` mode): `RidgeCV(alphas, fit_intercept, cv=None)`'s leave-one-out error, computed from
  the eigendecomposition G = VΛV' (or of A) and the scaled rows (`ScaledStats.rows`, which Ridge alone needs): with U = Z_c V,
  the fit at α is U(V'q/(λ + α)), each row's leverage Σ_k U_ik²/(λ_k + α) plus 1/n for the unpenalised intercept, and its
  error the residual over (1 - leverage). The first α of least mean squared error wins, as in `_RidgeGCV.fit`; coefficients
  V(V'q/(λ + α*)), the intercept ȳ - z̄·w. This is exact where scikit-learn's dual route is not: it forms c = U(w∘U'y) + y/α
  from two terms of about |y|/α that cancel, and errs by about `ridgecv_rounding`/α (research sweep `ridge_sweep.py`, 33,264
  fits of four families at every penalty of the default grid: RidgeCV up to 3.9e-7 EUR/MWh from a long-double ridge at
  α = 1e-3 where the engine is 8.5e-10 from it; their gap up to 4e-8/α on spiky targets). A fit whose `ridgecv_rounding`/α*,
  in target units, exceeds `RIDGE_ROUNDING_MAX` (1e-10) is handed back: of the fits kept none was more than 7.1e-10 from
  `RidgeCV` (the gap at most 8.7 times the measure wherever it was over 1e-10), of the DE fixture's Lago rows at α ≥ 10 (the
  penalties they choose) 374 of 384 are kept, and a fixed floor on α would have had to be 63.
- **LEAR**: `lear_fit` on `ScaledStats.gram()`, for every n.
"""

from typing import Callable, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg.lapack import dpotrf, dpotrs

from ..scalers.base import mean_std
from ._lars import GramStats
from ._sums import RawSums

EPS = float(np.finfo(np.float64).eps)

CANDIDATE_RTOL = 1e-6
"""A column whose standard deviation from the sums is at most this fraction of max(|mean|, rms of x - shift) on the fit's rows
(or whose variance from the sums is not positive) is decided constant or not by `mean_std` on its values (see the module)."""

CANCELLATION_MAX = 1e4
"""The largest S_jj / (S_jj - u_j²/r) (a column's cancellation factor) scaled from the sums; above it, `StatsFallback`."""

GUARD = 1e5
"""OLS hands a fit back when (largest/smallest Cholesky pivot)² of its normal equations exceeds this: the largest power of ten
at which every fit of the conditioning sweep that the guard let through was within 1e-8 of `LinearRegression` (see the
module). Engine2's 1e10 let fits through that were 1.8e-4 off."""

RIDGE_ROUNDING_MAX = 1e-10
"""Ridge hands a fit back when scikit-learn's own rounding at the chosen penalty, `ridgecv_rounding`/α in target units, exceeds
this: the gap to `RidgeCV` was at most 8.7 times that measure where it mattered, so no fit kept is more than about 9e-10 from
the model path, however exact the engine (see the module)."""

MIN_ROWS = 3
"""Fewer rows in ρ go to the model path for every model (the engine's fallback table)."""


class StatsFallback(Exception):
    """The fit's statistics can't be formed from the sums to the model path's digits; `reason` names why (for the counts)."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class StandardParams(NamedTuple):
    """`StandardScaler`'s parameters for one fit: what scales its issue row and maps its forecast back.

    `mean`/`scale` are the centre and divisor of each column (0.0 and 1.0 where the column isn't scaled, the scaler's mean and
    1.0 where it is constant); `scaled` the columns the scaler scales (its mask, or every column with `scale_binary`);
    `constant` the columns constant on the fit's rows (their centred training values are 0); `ymean`/`yscale` the target's.
    `shift`/`offset` split the centre as the sums hold it, the span's shift c and u/r (mean = fl(c + u/r)): a row is centred
    as (x - c) - u/r, which is how G and q are centred, not on the rounded mean, whose half ulp a coefficient would multiply
    (on a column of 50,000 ± 5e-5 that alone moved a forecast by 1.2e-8). A constant column has its `mean_std` mean as the
    shift and offset 0, an unscaled one 0 and 0.
    """

    mean: np.ndarray
    scale: np.ndarray
    scaled: np.ndarray
    constant: np.ndarray
    ymean: float
    yscale: float
    shift: np.ndarray
    offset: np.ndarray

    def transform(self, x) -> np.ndarray:
        """Rows `x` (k, p) as the scaler transforms its test rows: ((x - shift) - offset) / scale per column, the centring of
        G and q (an unscaled column as it is, a constant one as `(x - mean) / 1.0`, bitwise the scaler's)."""
        x = np.asarray(x, dtype=np.float64)
        return ((x - self.shift) - self.offset) / self.scale

    def transform_training(self, x) -> np.ndarray:
        """Training rows `x` (r, p) as `fit_transform` returns them: a constant scaled column exactly 0.0 (NaN stays NaN)."""
        z = self.transform(x)
        zeroed = self.scaled & self.constant
        if zeroed.any():
            z[:, zeroed] = np.where(np.isnan(np.asarray(x, dtype=np.float64)[:, zeroed]), np.nan, 0.0)
        return z

    def transform_target(self, y) -> np.ndarray:
        return (np.asarray(y, dtype=np.float64) - self.ymean) / self.yscale

    def inverse(self, value: float) -> float:
        """A forecast in scaled units back in target units, as `StandardScaler.inverse`."""
        return value * self.yscale + self.ymean


class ScaledStats(NamedTuple):
    """The centred sums of a fit's training rows in scaled space: everything `_fit_stats` reads.

    `zbar`, `ybar`: means of the scaled columns and target (0 for a scaled column and the target); `G`, `q`, `yy`: centred
    Σ(z - z̄)(z - z̄)', Σ(z - z̄)(y - ȳ), Σ(y - ȳ)²; `blocks`: the five CV blocks (step 5e), else None; `rows`: a callable giving
    the scaled (Z, y) of the fit's rows, for the models that need them (`_ENGINE_NEEDS`), else None; `yscale`: what a scaled
    target unit is in target units (the scaler's divisor), for a fit that bounds an error in target units (Ridge).
    """

    n: int
    zbar: np.ndarray
    ybar: float
    G: np.ndarray
    q: np.ndarray
    yy: float
    blocks: Optional[Tuple["ScaledStats", ...]] = None
    rows: Optional[Callable[[], Tuple[np.ndarray, np.ndarray]]] = None
    yscale: float = 1.0

    def gram(self) -> GramStats:
        """LEAR's seam (`_lars.GramStats`): the same sums under its names."""
        return GramStats(int(self.n), self.zbar, float(self.ybar), self.G, self.q, float(self.yy))

    def uncentred(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """(Z'Z, Z'y, y'y): A = G + n z̄z̄', b = q + n z̄ȳ, yy + n ȳ², for a fit without an intercept."""
        n = float(self.n)
        A = self.G + n * np.outer(self.zbar, self.zbar)
        b = self.q + (n * self.ybar) * self.zbar
        return A, b, float(self.yy + n * self.ybar * self.ybar)


Values = Union[Callable[[Optional[int]], np.ndarray], Tuple[np.ndarray, np.ndarray]]


def _reader(values: Values, p: int):
    """`values` as (column j's values, or the target's for None; the (x, y) rows), the rows built only when asked."""
    if callable(values):

        def rows():
            x = np.column_stack([np.asarray(values(j), dtype=np.float64) for j in range(p)]) if p else None
            return x, np.asarray(values(None), dtype=np.float64)

        return lambda j: np.asarray(values(j), dtype=np.float64), rows
    x, y = values
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != p or y.shape != (x.shape[0],):
        raise ValueError(
            f"values must be the fit's rows, x of shape (r, {p}) and y of shape (r,); got {x.shape}, {y.shape}"
        )
    return (lambda j: y if j is None else np.ascontiguousarray(x[:, j])), (lambda: (x, y))


def _candidates(var: np.ndarray, mean: np.ndarray, square: np.ndarray, r: int) -> np.ndarray:
    """Columns the sums can't call varying: variance not positive (or not finite), or std <= CANDIDATE_RTOL of max(|mean|, rms)."""
    with np.errstate(invalid="ignore"):
        std = np.sqrt(np.where(var > 0, var, 0.0))
        rms = np.sqrt(np.maximum(square, 0.0) / r)
        return ~(var > 0) | ~np.isfinite(var) | (std <= CANDIDATE_RTOL * np.maximum(np.abs(mean), rms))


def _cancellation(square: np.ndarray, centred: np.ndarray) -> np.ndarray:
    """S / (S - u²/r) per column, inf where the centred sum is not positive."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(centred > 0, square / np.where(centred > 0, centred, 1.0), np.inf)


def standard_stats(
    raw: RawSums, values: Values, mask: Optional[Sequence[bool]], scale_binary: bool = False
) -> Tuple[ScaledStats, StandardParams]:
    """`StandardScaler(scale_binary=scale_binary)`'s map of a fit's rows, from their exact sums `raw` (see the module).

    `values` gives the fit's raw rows on ρ, for the columns the sums can't decide: either a pair (x, y) of arrays (r rows,
    the columns in `raw`'s order) or a callable, `values(j)` column j's values and `values(None)` the target's. `mask` is the
    scalable mask the model path passes the scaler (True: scaled; a 0/1 column False), ignored with `scale_binary`; None reads
    it from the rows, as the scaler does. Raises `StatsFallback` where the sums can't give the model path's digits, and
    ValueError for fewer than `MIN_ROWS` rows (the engine sends those to the model path first).
    """
    r = int(raw.n)
    shift = np.asarray(raw.shift, dtype=np.float64)
    p = shift.size
    if r < MIN_ROWS:
        raise ValueError(f"standard_stats needs at least {MIN_ROWS} rows, got {r}")
    column, rows = _reader(values, p)
    if scale_binary:
        scaled = np.ones(p, dtype=bool)
    elif mask is None:
        from ..scalers.standard import StandardScaler

        x, _ = rows()
        scaled = StandardScaler.get_scalable_mask(x) if p else np.ones(0, dtype=bool)
    else:
        scaled = np.asarray(mask, dtype=bool).reshape(-1).copy()
        if scaled.size != p:
            raise ValueError(f"mask has {scaled.size} entries for {p} columns")

    u = np.asarray(raw.u, dtype=np.float64)
    S = np.asarray(raw.S, dtype=np.float64)
    v = np.asarray(raw.v, dtype=np.float64)
    sy, syy = float(raw.sy), float(raw.syy)

    # Centred sums of d = x - c (shift-invariant) and the means and variances they give.
    C = S - np.outer(u, u) / r  # exactly symmetric: u_j u_k = u_k u_j in IEEE arithmetic
    cq = v - u * (sy / r)
    cyy = syy - sy * sy / r
    square = np.diag(S).copy()
    centred = np.diag(C).copy()
    offset = u / r
    mean = shift + offset
    var = centred / (r - 1)

    constant = np.zeros(p, dtype=bool)
    candidates = np.flatnonzero(_candidates(var, mean, square, r))
    for j in candidates.tolist():
        col = column(j)
        if scaled[j]:
            m, _, is_constant = mean_std(col)
        else:
            m, is_constant = float(np.mean(col)), bool(col.size and col.min() == col.max())
        if is_constant:
            constant[j] = True
            mean[j] = m
        elif not var[j] > 0:
            raise StatsFallback(
                "cancellation", f"column {j}: variance {var[j]!r} from the sums, not constant on its rows"
            )
    kappa = _cancellation(square, centred)
    bad = np.flatnonzero(~constant & (kappa > CANCELLATION_MAX))
    if bad.size:
        j = int(bad[0])
        raise StatsFallback("cancellation", f"column {j}: S/(S - u²/r) = {kappa[j]:.3g} > {CANCELLATION_MAX:g}")

    # The target: centred and divided by its standard deviation, or by 1.0 where `mean_std` calls it constant (not zeroed).
    ymean = float(raw.yshift + sy / r)
    yvar = cyy / (r - 1)
    ysquare = np.array([syy])
    if _candidates(np.array([yvar]), np.array([ymean]), ysquare, r)[0]:
        ymean, ystd, y_constant = mean_std(column(None))
        ymean, ystd = float(ymean), float(ystd)
        if not y_constant and not yvar > 0:
            raise StatsFallback("cancellation", f"target: variance {yvar!r} from the sums, not constant on its rows")
    else:
        y_constant = False
        ystd = float(np.sqrt(yvar))
    if not y_constant and _cancellation(ysquare, np.array([cyy]))[0] > CANCELLATION_MAX:
        raise StatsFallback("cancellation", f"target: S/(S - u²/r) above {CANCELLATION_MAX:g}")

    sigma = np.where(scaled & ~constant, np.sqrt(np.where(var > 0, var, 1.0)), 1.0)
    varying = scaled & ~constant
    params = StandardParams(
        mean=np.where(scaled, mean, 0.0),
        scale=sigma,
        scaled=scaled,
        constant=constant,
        ymean=ymean,
        yscale=ystd,
        shift=np.where(varying, shift, np.where(scaled, mean, 0.0)),
        offset=np.where(varying, offset, 0.0),
    )

    # Scaled space: a scaled column has mean 0 exactly, an unscaled one its mean on ρ (a constant one the value itself).
    G = C / np.outer(sigma, sigma)
    q = cq / (sigma * ystd)
    zbar = np.where(scaled, 0.0, mean)
    if constant.any():
        G[constant, :] = 0.0
        G[:, constant] = 0.0
        q[constant] = 0.0
    yy = cyy / (ystd * ystd)

    def scaled_rows():
        x, y = rows()
        z = params.transform_training(x) if p else np.zeros((r, 0))
        return z, params.transform_target(y)

    stats = ScaledStats(n=r, zbar=zbar, ybar=0.0, G=G, q=q, yy=float(yy), blocks=None, rows=scaled_rows, yscale=ystd)
    return stats, params


# --- fits -------------------------------------------------------------------------------------------------------------


class LinearFit:
    """A linear fit from statistics: `predict(X) = X @ coef_ + intercept_`, as the scikit-learn estimators it stands for.

    `alpha_` is the chosen penalty of a Ridge fit (None for OLS)."""

    __slots__ = ("coef_", "intercept_", "alpha_")

    def __init__(self, coef: np.ndarray, intercept: float = 0.0, alpha: Optional[float] = None):
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = float(intercept)
        self.alpha_ = alpha

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_

    def __repr__(self) -> str:
        return f"LinearFit(p={self.coef_.size}, intercept={self.intercept_!r}, alpha={self.alpha_!r})"


def _live(M: np.ndarray) -> np.ndarray:
    """Columns of a Gram that aren't exactly zero (a constant column's row and column are 0.0 from `standard_stats`)."""
    if M.size == 0:
        return np.zeros(0, dtype=bool)
    return ~((M == 0.0).all(axis=0))


def cholesky_solve(M: np.ndarray, b: np.ndarray, guard: float = GUARD) -> Optional[np.ndarray]:
    """The solution w of M w = b over M's live columns (a zero column gets 0), or None when the Cholesky factorisation fails or
    (largest/smallest pivot)² exceeds `guard`."""
    p = b.size
    w = np.zeros(p)
    live = np.flatnonzero(_live(M))
    if live.size == 0:
        return w
    L, info = dpotrf(np.ascontiguousarray(M[np.ix_(live, live)]), lower=1, clean=1)
    if info != 0:
        return None
    pivots = np.diag(L)
    small = pivots.min()
    if not small > 0.0 or not (pivots.max() / small) ** 2 <= guard:
        return None
    x, info = dpotrs(L, np.ascontiguousarray(b[live]), lower=1)
    if info != 0 or not np.isfinite(x).all():
        return None
    w[live] = x
    return w


def ols_fit(stats: ScaledStats, fit_intercept: bool, guard: float = GUARD) -> Optional[LinearFit]:
    """`LinearRegression(fit_intercept)` from scaled sums by the normal equations, or None where the model path must fit it:
    n <= p + 1, a failed factorisation, or the guard."""
    p = np.asarray(stats.q).size
    if stats.n < MIN_ROWS or stats.n <= p + 1:
        return None
    if fit_intercept:
        M, b = stats.G, stats.q
    else:
        M, b, _ = stats.uncentred()
    w = cholesky_solve(M, b, guard)
    if w is None:
        return None
    intercept = float(stats.ybar - stats.zbar @ w) if fit_intercept else 0.0
    return LinearFit(w, intercept)


def ridgecv_rounding(Zc: np.ndarray, yc: np.ndarray) -> float:
    """The scale of `RidgeCV`'s own rounding at penalty α, times α, in scaled target units: eps · max_i Σ_j |Zc_ij| Σ_k |Zc_kj|
    |yc_k| on the rows it fits (centred with an intercept). Its dual route forms c = U(w∘U'y) + y/α, two terms of about |y|/α
    that cancel, and coef = Zc'c, so a forecast from a row like the training rows errs by about this over α."""
    if Zc.size == 0:
        return 0.0
    A = np.abs(Zc)
    return float(EPS * (A @ (A.T @ np.abs(yc))).max())


def ridge_fit(
    stats: ScaledStats, alphas: Sequence[float], fit_intercept: bool, rounding_max: float = RIDGE_ROUNDING_MAX
) -> Optional[LinearFit]:
    """`RidgeCV(alphas, fit_intercept, cv=None)` from scaled sums and rows (see the module), or None where the model path must
    fit it: n <= p, fewer than `MIN_ROWS` rows, no rows, or scikit-learn's own rounding at the chosen penalty,
    `ridgecv_rounding / α` times `stats.yscale` (target units), above `rounding_max`."""
    n = int(stats.n)
    p = np.asarray(stats.q).size
    if n < MIN_ROWS or n <= p or stats.rows is None:
        return None
    alphas = [float(a) for a in np.atleast_1d(np.asarray(alphas, dtype=np.float64))]
    Z, y = stats.rows()
    if fit_intercept:
        M, b = stats.G, stats.q
        Zc = Z - stats.zbar
        yc = y - stats.ybar
        base = 1.0 / n
    else:
        M, b, _ = stats.uncentred()
        Zc, yc, base = Z, y, 0.0
    live = np.flatnonzero(_live(M))
    w = np.zeros(p)
    if live.size:
        lam, V = np.linalg.eigh(M[np.ix_(live, live)])
        lam = np.maximum(lam, 0.0)
        U = Zc[:, live] @ V
        U2 = U * U
        Vb = V.T @ b[live]
    best_alpha, best_score = None, None
    for alpha in alphas:
        if live.size:
            inv = 1.0 / (lam + alpha)
            residual = yc - U @ (Vb * inv)
            leverage = U2 @ inv + base
        else:
            residual = yc
            leverage = np.full(n, base)
        score = float(np.mean((residual / (1.0 - leverage)) ** 2))
        if best_score is None or score < best_score:
            best_alpha, best_score = alpha, score
    if ridgecv_rounding(Zc, yc) * stats.yscale / best_alpha > rounding_max:
        return None
    if live.size:
        w[live] = V @ (Vb / (lam + best_alpha))
    intercept = float(stats.ybar - stats.zbar @ w) if fit_intercept else 0.0
    return LinearFit(w, intercept, best_alpha)


__all__ = [
    "CANDIDATE_RTOL",
    "CANCELLATION_MAX",
    "GUARD",
    "LinearFit",
    "RIDGE_ROUNDING_MAX",
    "ScaledStats",
    "StandardParams",
    "StatsFallback",
    "cholesky_solve",
    "ols_fit",
    "ridge_fit",
    "ridgecv_rounding",
    "standard_stats",
]
