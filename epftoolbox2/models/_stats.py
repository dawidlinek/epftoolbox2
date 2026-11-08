"""Scaled statistics for the step-5 engine: `StandardScaler`'s map from a fit's exact sums, and the linear fits that read it.

`standard_stats(raw, values, mask, scale_binary)` turns `RawSums` over a fit's rows ρ (`_sums.SpanSums.over`: sums about the
span's shifts) into `ScaledStats`, the centred sums of the rows as `StandardScaler.fit_transform` would scale them, and
`StandardParams`, the scaler's parameters (to scale the issue row and map a forecast back). `OLSModel`, `RidgeModel` and
`LEARModel` fit from `ScaledStats` through `_fit_stats`, as engine groups do (`_engine_worker.EngineFits`, step 5i).

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
  the factorisation fails, (largest/smallest pivot)² exceeds `GUARD` (1e5) or `dpocon`'s 1-norm condition estimate exceeds
  `CONDITION_MAX` (1e9): the normal equations err as eps·cond(Z)², gelsd as eps·cond(Z), and gelsd cuts its rank at
  max(n, p)·eps·σ_max, which a Gram can't resolve. The pivot ratio is only a lower bound on the condition number (1e3 times
  too small on the DE fixture's Lago rows, 1e5 times where a column is the mean of many others: such a fit, 1.2e-8 off,
  passed it alone), hence the estimate too. Both are set by the conditioning sweep
  (`epftoolbox2-research/library_check_step5/stats_fits/guard_sweep2.py`: 13,361 fits, p up to 248, four synthetic families
  and the Lago rows with a daily-mean column at four positions; no guard: up to 1.1e-10 from `LinearRegression` at cond(Z) 1e2
  to 1e3 on Gaussian designs, 3e-8 at 1e3 to 1e4, 2.3e-6 at 1e4 to 1e5). Of the 4,360 fits both guards keep, the worst is
  5.9e-9 where the issue row lies within 10 training standard deviations of the mean, 8.7e-9 where it lies 76,600 out (an
  extrapolation multiplies any coefficient's last bit; no property of the fit can see it); the 30 fits with a condition
  estimate between 1e9 and 1e10 reach 8.0e-9 and those between 1e10 and 1e11 1.2e-7. Every fit with cond(Z) below 316 is kept
  (the pivot ratio is at most cond(Z)², the estimate at most p·cond(Z)²), and so is every plain Lago row (ratios up to 1.2e4,
  estimates up to 1.2e8).
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

import math
from typing import Callable, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg.lapack import dpocon, dpotrf, dpotrs

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
"""OLS hands a fit back when (largest/smallest Cholesky pivot)² of its normal equations exceeds this (see the module; at 1e6
five fits the condition estimate let through were over 1e-8). Engine2's 1e10 let fits through that were 1.8e-4 off."""

CONDITION_MAX = 1e9
"""OLS also hands a fit back when LAPACK's estimate of the 1-norm condition number of its normal equations (`dpocon` on the
Cholesky factor, O(p²)) exceeds this: the pivot ratio alone is a lower bound on it, 1e3 to 1e5 times too small where a column
is the mean of many others. A decade below the largest power of ten at which no fit of the sweep was over 1e-8 (see the
module)."""

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
    target unit is in target units (the scaler's divisor), for a fit that bounds an error in target units: Ridge hands back
    a fit whose `yscale` is None (a builder that didn't set it), so a forgotten one can't loosen its budget.
    """

    n: int
    zbar: np.ndarray
    ybar: float
    G: np.ndarray
    q: np.ndarray
    yy: float
    blocks: Optional[Tuple["ScaledStats", ...]] = None
    rows: Optional[Callable[[], Tuple[np.ndarray, np.ndarray]]] = None
    yscale: Optional[float] = None

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


def _reader(values: Values, r: int, p: int):
    """`values` as (column j's values, or the target's for None; the (x, y) rows), the rows built only when asked. Rows that
    aren't the sums' r rows raise ValueError: a caller's bug, never a fit to hand back."""
    if callable(values):

        def column(j):
            col = np.asarray(values(j), dtype=np.float64)
            if col.shape != (r,):
                what = "the target" if j is None else f"column {j}"
                raise ValueError(f"values({j!r}) gives {col.shape} values for {what}; the sums are over {r} rows")
            return col

        def rows():
            x = np.column_stack([column(j) for j in range(p)]) if p else np.zeros((r, 0))
            return x, column(None)

        column(None)  # the target now, the columns as they are read
        return column, rows
    x, y = values
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != (r, p) or y.shape != (r,):
        raise ValueError(
            f"values must be the fit's rows, x of shape ({r}, {p}) and y of shape ({r},); got {x.shape}, {y.shape}"
        )
    return (lambda j: y if j is None else np.ascontiguousarray(x[:, j])), (lambda: (x, y))


def _candidates(var: np.ndarray, mean: np.ndarray, square: np.ndarray, r: int) -> np.ndarray:
    """Columns the sums can't call varying: variance not positive (or not finite), or std <= CANDIDATE_RTOL of max(|mean|, rms)."""
    with np.errstate(invalid="ignore"):
        std = np.sqrt(np.where(var > 0, var, 0.0))
        rms = np.sqrt(np.maximum(square, 0.0) / r)
        return ~(var > 0) | ~np.isfinite(var) | (std <= CANDIDATE_RTOL * np.maximum(np.abs(mean), rms))


def _scalar_candidate(var: float, mean: float, square: float, r: int) -> bool:
    """`_candidates` of one value (the target), in Python floats: the same decision, without arrays of one element."""
    if not var > 0 or not math.isfinite(var):
        return True
    return math.sqrt(var) <= CANDIDATE_RTOL * max(abs(mean), math.sqrt(max(square, 0.0) / r))


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
    column, rows = _reader(values, r, p)
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
    if not (np.isfinite(u).all() and np.isfinite(S).all() and np.isfinite(v).all() and np.isfinite([sy, syy]).all()):
        raise StatsFallback("overflow", "a sum of the fit's rows is not finite (squares past float64's range)")

    # Centred sums of d = x - c (shift-invariant) and the means and variances they give.
    with np.errstate(over="ignore", invalid="ignore"):
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
    if _scalar_candidate(yvar, ymean, syy, r):
        ymean, ystd, y_constant = mean_std(column(None))
        ymean, ystd = float(ymean), float(ystd)
        if not y_constant and not yvar > 0:
            raise StatsFallback("cancellation", f"target: variance {yvar!r} from the sums, not constant on its rows")
    else:
        y_constant = False
        ystd = math.sqrt(yvar)
    if not y_constant and (syy / cyy if cyy > 0 else math.inf) > CANCELLATION_MAX:
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
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        G = C / np.outer(sigma, sigma)
        q = cq / (sigma * ystd)
        yy = cyy / (ystd * ystd)
    zbar = np.where(scaled, 0.0, mean)
    if constant.any():
        G[constant, :] = 0.0
        G[:, constant] = 0.0
        q[constant] = 0.0
    if not (
        np.isfinite(G).all()
        and np.isfinite(q).all()
        and np.isfinite(zbar).all()
        and np.isfinite(params.mean).all()
        and np.isfinite(sigma).all()
        and np.isfinite([yy, ymean, ystd]).all()
        and ystd > 0
    ):
        raise StatsFallback("overflow", "a scaled sum, mean or scale of the fit is not finite")

    def scaled_rows():
        x, y = rows()
        z = params.transform_training(x) if p else np.zeros((r, 0))
        return z, params.transform_target(y)

    stats = ScaledStats(n=r, zbar=zbar, ybar=0.0, G=G, q=q, yy=float(yy), blocks=None, rows=scaled_rows, yscale=ystd)
    return stats, params


# --- fits -------------------------------------------------------------------------------------------------------------


def estimator_params(estimator, cls, honoured: Sequence[str], fixed: Optional[dict] = None) -> Optional[dict]:
    """The parameters of a model's `estimator` (its `_estimator()`, what the model path fits) where a fit from statistics may
    stand in for it, else None (the model path fits it, and raises where scikit-learn's `fit` would).

    It may where the estimator is exactly a `cls` (not a subclass), its parameters pass the installed scikit-learn's own
    validation (`_validate_params`, the check `fit` makes first: a float `max_iter` or a `fit_intercept` of 1 fails it), and
    every parameter equals that of `cls(**honoured, **fixed)`, the `honoured` ones (what the fit from statistics reads) taken
    from the estimator itself. So an option the statistics don't honour (a subclass's `eps=1e-5`, another `gcv_mode`) sends
    the fit back.

    The answer is a function of the estimator's class and attribute values alone, so it is kept per those values when every
    one is a plain Python scalar, string, None or list of them (`_plain`): an engine asks once per fit, and scikit-learn's
    validation costs more than a small fit from statistics."""
    if type(estimator) is not cls:
        return None
    key = _check_key(estimator, cls, honoured, fixed)
    if key is not None:
        known = _ESTIMATOR_CHECKS.get(key, _UNKNOWN)
        if known is not _UNKNOWN:
            return None if known is None else dict(known)
    params = _estimator_params(estimator, cls, honoured, fixed)
    if key is not None:
        if len(_ESTIMATOR_CHECKS) >= _ESTIMATOR_CHECKS_MAX:
            _ESTIMATOR_CHECKS.clear()
        _ESTIMATOR_CHECKS[key] = None if params is None else dict(params)
    return params


# `estimator_params`' answers by (class, honoured, fixed, the estimator's attribute values), each value with its type.
_ESTIMATOR_CHECKS: dict = {}
_ESTIMATOR_CHECKS_MAX = 256
_UNKNOWN = object()


def _plain(value):
    """A hashable key of a value that is None, a bool, int, float or str, or a list or tuple of such values, with the type of
    each (so 1, 1.0 and True differ, as `_same` has them); None for anything else, which isn't kept."""
    kind = type(value)
    if value is None or kind in (bool, int, float, str):
        return (kind, value)
    if kind in (list, tuple):
        parts = tuple(_plain(item) for item in value)
        return None if any(part is None for part in parts) else (kind, parts)
    return None


def _check_key(estimator, cls, honoured: Sequence[str], fixed: Optional[dict]):
    """The key `estimator_params` keeps its answer under, or None where a value isn't plain (`_plain`)."""
    values = []
    for name, value in sorted({**vars(estimator), **{f"\0fixed:{k}": v for k, v in (fixed or {}).items()}}.items()):
        plain = _plain(value)
        if plain is None:
            return None
        values.append((name, plain))
    return (cls, tuple(honoured), tuple(values))


def _estimator_params(estimator, cls, honoured: Sequence[str], fixed: Optional[dict] = None) -> Optional[dict]:
    """`estimator_params` itself, uncached."""
    validate = getattr(estimator, "_validate_params", None)
    if validate is None:
        return None  # a scikit-learn without the check: the model path decides
    try:
        validate()
    except (TypeError, ValueError):  # InvalidParameterError is both
        return None
    params = estimator.get_params(deep=False)
    reference = cls(**{name: params[name] for name in honoured}, **(fixed or {})).get_params(deep=False)
    if params.keys() != reference.keys() or not all(_same(params[k], reference[k]) for k in params):
        return None
    return params


def _same(a, b) -> bool:
    """Whether two parameter values are the same: identical, or equal and of one type (False where `==` isn't a truth)."""
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


class LinearFit:
    """A linear fit from statistics: `predict(X) = X @ coef_ + intercept_`, as the scikit-learn estimators it stands for.

    `alpha_` is the chosen penalty of a Ridge fit (None for OLS). `rounding` (Ridge) holds per column the scale of the model
    path's own rounding at α* in target units per unit of |z - z̄| of a row (`issue_rounding_ok`), None for OLS. `centred`:
    whether the fit has an intercept, so that a row is measured from the training means (`issue_metric`), not from 0."""

    __slots__ = ("coef_", "intercept_", "alpha_", "rounding", "centred")

    def __init__(
        self,
        coef: np.ndarray,
        intercept: float = 0.0,
        alpha: Optional[float] = None,
        rounding: Optional[np.ndarray] = None,
        centred: bool = True,
    ):
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = float(intercept)
        self.alpha_ = alpha
        self.rounding = rounding
        self.centred = bool(centred)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_

    def __repr__(self) -> str:
        return f"LinearFit(p={self.coef_.size}, intercept={self.intercept_!r}, alpha={self.alpha_!r})"


def _finite(stats: ScaledStats) -> bool:
    """Whether the sums a fit reads are finite (`standard_stats` raises before; other builders are checked here)."""
    return bool(
        np.isfinite(stats.G).all()
        and np.isfinite(stats.q).all()
        and np.isfinite(stats.zbar).all()
        and np.isfinite([stats.yy, stats.ybar]).all()
    )


def _live(M: np.ndarray) -> np.ndarray:
    """Columns of a Gram that aren't exactly zero (a constant column's row and column are 0.0 from `standard_stats`)."""
    if M.size == 0:
        return np.zeros(0, dtype=bool)
    return ~((M == 0.0).all(axis=0))


def cholesky_solve(
    M: np.ndarray, b: np.ndarray, guard: float = GUARD, condition_max: float = CONDITION_MAX
) -> Optional[np.ndarray]:
    """The solution w of M w = b over M's live columns (a zero column gets 0), or None when the Cholesky factorisation fails,
    (largest/smallest pivot)² exceeds `guard` or `dpocon`'s 1-norm condition estimate exceeds `condition_max`."""
    p = b.size
    w = np.zeros(p)
    live = np.flatnonzero(_live(M))
    if live.size == 0:
        return w
    Ml = np.ascontiguousarray(M if live.size == p else M[np.ix_(live, live)])
    L, info = dpotrf(Ml, lower=1, clean=1)
    if info != 0:
        return None
    pivots = np.diag(L)
    small = pivots.min()
    if not small > 0.0 or not (pivots.max() / small) ** 2 <= guard:
        return None
    rcond, info = dpocon(L, float(np.abs(Ml).sum(axis=0).max()), uplo="L")
    if info != 0 or not rcond * condition_max >= 1.0:
        return None
    x, info = dpotrs(L, np.ascontiguousarray(b if live.size == p else b[live]), lower=1)
    if info != 0 or not np.isfinite(x).all():
        return None
    w[live] = x
    return w


def ols_fit(
    stats: ScaledStats, fit_intercept: bool, guard: float = GUARD, condition_max: float = CONDITION_MAX
) -> Optional[LinearFit]:
    """`LinearRegression(fit_intercept)` from scaled sums by the normal equations, or None where the model path must fit it:
    n <= p + 1, a failed factorisation, or either guard."""
    p = np.asarray(stats.q).size
    if p == 0 or stats.n < MIN_ROWS or stats.n <= p + 1 or not _finite(stats):
        return None  # p 0: scikit-learn refuses a fit of no feature, which the model path counts as a fit_error
    if fit_intercept:
        M, b = stats.G, stats.q
    else:
        M, b, _ = stats.uncentred()
    w = cholesky_solve(M, b, guard, condition_max)
    if w is None:
        return None
    intercept = float(stats.ybar - stats.zbar @ w) if fit_intercept else 0.0
    return LinearFit(w, intercept, centred=fit_intercept)


class IssueMetric(NamedTuple):
    """The metric of a fit an issue row's distance from its training rows is measured in (`issue_metric`): the fit's active
    columns (a coefficient other than 0), the centre of a row on them (z̄ with an intercept, 0 without), sqrt((n - 1) / M_jj)
    of each (M the Gram of the active columns, centred with an intercept, uncentred without), the eigenvectors of M,
    sqrt((n - 1) / λ_k) of each (inf for an eigenvalue not above λ_max·eps·p, a direction the training rows don't spread
    along), and λ_max / λ_min (inf where singular, 1.0 with fewer than two columns)."""

    columns: np.ndarray
    centre: np.ndarray
    column_scale: np.ndarray
    vectors: np.ndarray
    scale: np.ndarray
    condition: float


def issue_metric(stats: ScaledStats, fitted) -> IssueMetric:
    """The fit's `IssueMetric`, from its scaled sums and its coefficients (`coef_` of a `LinearFit`, `w` of LEAR's fit): the
    directions the forecast reads, and how far the training rows spread along each. A function of the fit alone."""
    coef = np.asarray(fitted.coef_ if hasattr(fitted, "coef_") else fitted.w, dtype=np.float64).reshape(-1)
    columns = np.flatnonzero(coef != 0.0)
    centred = getattr(fitted, "centred", True)
    if columns.size == 0:
        empty = np.zeros(0)
        return IssueMetric(columns, empty, empty, np.zeros((0, 0)), empty, 1.0)
    M = stats.G if centred else stats.uncentred()[0]
    Ma = np.ascontiguousarray(M[np.ix_(columns, columns)])
    diagonal = np.diag(Ma)
    column_scale = np.where(diagonal > 0.0, np.sqrt((stats.n - 1) / np.where(diagonal > 0.0, diagonal, 1.0)), np.inf)
    lam, vectors = np.linalg.eigh(Ma)
    top = float(lam.max())
    spread = lam > top * EPS * columns.size
    scale = np.where(spread, np.sqrt((stats.n - 1) / np.where(spread, lam, 1.0)), np.inf)
    condition = (top / float(lam.min()) if spread.all() else np.inf) if columns.size > 1 else 1.0
    centre = stats.zbar[columns] if centred else np.zeros(columns.size)
    return IssueMetric(columns, centre, column_scale, vectors, scale, condition)


def issue_distance(metric: IssueMetric, z: np.ndarray) -> float:
    """How far an issue row (`z`, scaled as the fit's rows, one row) is from the fit's training rows in the fit's own metric
    (`issue_metric`), in training standard deviations: the larger of the row's largest coordinate (from its centre) along an
    active column, |z_j - z̄_j| / sd_j, and along an eigenvector of the active columns' Gram, |v_kᵀ(z - z̄)| sqrt((n - 1) /
    λ_k). The first is each column's own distance (an eigenvector of nearly equal eigenvalues can spread a column over
    several); the second finds a row out along the difference of two near-collinear columns, by that difference's own
    spread, however close to their training values each column is. A function of the fit and the row alone; 0.0 with no
    active column."""
    if metric.columns.size == 0:
        return 0.0
    row = np.asarray(z, dtype=np.float64).reshape(-1)[metric.columns] - metric.centre
    along = np.abs(metric.vectors.T @ row)
    own = np.abs(row)
    return float(
        max(
            np.where(own > 0.0, own * metric.column_scale, 0.0).max(),
            np.where(along > 0.0, along * metric.scale, 0.0).max(),
        )
    )


def issue_rounding_ok(fitted, stats: ScaledStats, z: np.ndarray, rounding_max: float = RIDGE_ROUNDING_MAX) -> bool:
    """Whether the model path's own rounding at an issue row stays within the budget it has at the training rows: a fit
    that carries `rounding` (Ridge's `LinearFit`) keeps `rounding @ |z - z̄|` at most `rounding_max` (target units); any
    other fit, True."""
    rounding = getattr(fitted, "rounding", None)
    if rounding is None:
        return True
    row = np.asarray(z, dtype=np.float64).reshape(-1)
    return bool(float(rounding @ np.abs(row - stats.zbar)) <= rounding_max)


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
    if p == 0 or n < MIN_ROWS or n <= p or stats.rows is None or stats.yscale is None or not _finite(stats):
        return None  # p 0: the model path's RidgeCV refuses it (a fit_error)
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
        if not np.isfinite(score):
            return None  # a NaN would lose every comparison and leave the first penalty chosen
        if best_score is None or score < best_score:
            best_alpha, best_score = alpha, score
    if not ridgecv_rounding(Zc, yc) * stats.yscale / best_alpha <= rounding_max:  # NaN hands back too
        return None
    if live.size:
        w[live] = V @ (Vb / (lam + best_alpha))
    intercept = float(stats.ybar - stats.zbar @ w) if fit_intercept else 0.0
    # The same measure at any row x: eps · Σ_j |x_j - z̄_j| Σ_k |Zc_kj| |yc_k| · yscale / α* (at a training row, its own term of
    # `ridgecv_rounding`), for an issue row far outside the training rows (`issue_rounding_ok`).
    rounding = EPS * (np.abs(Zc).T @ np.abs(yc)) * (stats.yscale / best_alpha)
    return LinearFit(w, intercept, best_alpha, rounding, centred=fit_intercept)


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
    "estimator_params",
    "IssueMetric",
    "issue_distance",
    "issue_metric",
    "issue_rounding_ok",
    "ols_fit",
    "ridge_fit",
    "ridgecv_rounding",
    "standard_stats",
]
