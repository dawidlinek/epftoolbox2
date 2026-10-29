"""Statistical tests and bootstraps for comparing point forecasts.

Pure numpy/scipy kernels behind the comparison evaluators. They take aligned, finite loss series and know nothing
about forecast keys: putting the compared models on their common keys and building one loss per day (the mean
absolute error over the periods of a day) is the caller's job.

Conventions shared by every function:

- A loss is "lower is better". ``loss_model`` and ``loss_benchmark`` are 1-D arrays of the same length (one value per
  day); ``losses`` is an ``n x K`` array with one column per model.
- A test statistic is positive when the model is better than the benchmark (lower mean loss).
- Non-finite inputs raise ``ValueError``; nothing is dropped silently.
- Every bootstrap is driven by an explicit seed (or ``numpy.random.Generator``), so the same data and arguments give
  the same numbers. Functions that resample also accept precomputed ``indices`` so that several comparisons can share
  one set of bootstrap samples.

The kernels reproduce the evaluation-protocol study that fixed the defaults (``evalproto``, see
``tests/test_stats_evalproto.py``): the Diebold–Mariano statistic, the Holm adjustment and the block-bootstrap
indices give that study's numbers, and the model confidence set follows ``arch.bootstrap.MCS``.

References
----------
Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics
13(3), 253-263.
Newey, W. K. & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation
consistent covariance matrix. Econometrica 55(3), 703-708.
Newey, W. K. & West, K. D. (1994). Automatic lag selection in covariance matrix estimation. Review of Economic
Studies 61(4), 631-653.
Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6(2),
65-70.
Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. Annals of Statistics
17(3), 1217-1241.
Politis, D. N. & Romano, J. P. (1992). A circular block-resampling procedure for stationary data. In R. LePage &
L. Billard (eds.), Exploring the Limits of Bootstrap, Wiley, 263-270.
Politis, D. N. & Romano, J. P. (1994). The stationary bootstrap. Journal of the American Statistical Association
89(428), 1303-1313.
Hansen, P. R., Lunde, A. & Nason, J. M. (2011). The model confidence set. Econometrica 79(2), 453-497.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy import stats as _sstats

__all__ = [
    "DMResult",
    "MCSResult",
    "RankResult",
    "RatioInterval",
    "block_bootstrap_indices",
    "default_lag",
    "dm_test",
    "holm",
    "minimum_detectable_effect",
    "model_confidence_set",
    "newey_west_variance",
    "rank_bootstrap",
    "ratio_interval",
]

SeedLike = Union[int, np.random.Generator, None]

# Upper bound on the number of float64 elements one vectorised bootstrap step holds at once (32 MB).
_CHUNK_ELEMENTS = 1 << 22


# --------------------------------------------------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DMResult:
    """Result of :func:`dm_test`.

    Attributes:
        statistic: Diebold–Mariano statistic, positive when the model has the lower mean loss.
        p_value: Two-sided p-value from the standard normal distribution.
        n: Number of loss pairs (days) used.
        lag: Newey–West truncation lag (0 for the plain variance).
        variance: ``"hac"`` or ``"plain"``.
    """

    statistic: float
    p_value: float
    n: int
    lag: int
    variance: str


@dataclass(frozen=True)
class RatioInterval:
    """Result of :func:`ratio_interval`: ``mean(loss_model) / mean(loss_benchmark)`` and its percentile interval.

    Attributes:
        ratio: Point estimate of the loss ratio (below 1: the model is better).
        low: Lower bound of the bootstrap percentile interval.
        high: Upper bound of the bootstrap percentile interval.
        level: Coverage of the interval, e.g. 0.95.
        n: Number of days.
        reps: Number of bootstrap repetitions.
    """

    ratio: float
    low: float
    high: float
    level: float
    n: int
    reps: int


@dataclass(frozen=True, eq=False)
class MCSResult:
    """Result of :func:`model_confidence_set`. Arrays are in the column order of ``losses`` and read-only.

    Attributes:
        p_values: MCS p-value of every model: the smallest size at which the model is excluded from the set. The
            last surviving model has p-value 1.
        elimination_order: Column indices in the order they were eliminated; the last entry survived every test.
        alpha: Size of the test; :meth:`included` uses it by default.
        statistic: ``"R"`` (T_R, the default) or ``"max"`` (T_max).
        n: Number of days.
        reps: Number of bootstrap repetitions.
    """

    p_values: np.ndarray
    elimination_order: Tuple[int, ...]
    alpha: float
    statistic: str
    n: int
    reps: int

    def included(self, alpha: Optional[float] = None) -> np.ndarray:
        """Boolean mask of the models in the confidence set at size ``alpha`` (default: the result's alpha).

        A model is included when its MCS p-value is larger than ``alpha`` (``arch.bootstrap.MCS.included``); the set
        then contains the best model with probability at least ``1 - alpha`` asymptotically.
        """
        a = self.alpha if alpha is None else float(alpha)
        return self.p_values > a


@dataclass(frozen=True, eq=False)
class RankResult:
    """Result of :func:`rank_bootstrap`. Arrays are in the column order of ``losses`` and read-only.

    Attributes:
        loss: Mean loss of every model over all days.
        rank: Rank of the mean loss (1 = lowest).
        p_best: Share of bootstrap samples in which the model has the lowest mean loss.
        rank_low: Lower end of the rank interval (integer, rounded outwards).
        rank_high: Upper end of the rank interval (integer, rounded outwards).
        level: Coverage of the rank interval, e.g. 0.90.
        n: Number of days.
        reps: Number of bootstrap repetitions.
    """

    loss: np.ndarray
    rank: np.ndarray
    p_best: np.ndarray
    rank_low: np.ndarray
    rank_high: np.ndarray
    level: float
    n: int
    reps: int


# --------------------------------------------------------------------------------------------------------------------
# Input checks
# --------------------------------------------------------------------------------------------------------------------


def _as_loss(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D (one loss per day), got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values; compare models on their common, finite keys")
    return a


def _as_pair(loss_model, loss_benchmark) -> Tuple[np.ndarray, np.ndarray]:
    lm = _as_loss(loss_model, "loss_model")
    lb = _as_loss(loss_benchmark, "loss_benchmark")
    if lm.shape != lb.shape:
        raise ValueError(f"loss_model and loss_benchmark differ in length: {lm.size} vs {lb.size}")
    if lm.size < 2:
        raise ValueError(f"at least 2 days are needed, got {lm.size}")
    return lm, lb


def _as_losses(losses, min_models: int) -> np.ndarray:
    a = np.asarray(losses, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"losses must be 2-D (days x models), got shape {a.shape}")
    if a.shape[0] < 2:
        raise ValueError(f"at least 2 days are needed, got {a.shape[0]}")
    if a.shape[1] < min_models:
        raise ValueError(f"at least {min_models} models (columns) are needed, got {a.shape[1]}")
    if not np.isfinite(a).all():
        raise ValueError("losses contain non-finite values; compare models on their common, finite keys")
    return a


def _check_level(level: float, name: str = "level") -> float:
    level = float(level)
    if not 0.0 < level < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {level}")
    return level


def _check_indices(indices, n: int) -> np.ndarray:
    idx = np.asarray(indices)
    if idx.ndim != 2 or idx.shape[1] != n or idx.shape[0] < 1:
        raise ValueError(f"indices must have shape (reps, {n}), got {idx.shape}")
    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("indices must be integers")
    if idx.min() < 0 or idx.max() >= n:
        raise ValueError(f"indices must lie in [0, {n})")
    return idx


def _resample_indices(n: int, block: float, reps: int, seed: SeedLike, kind: str, indices) -> np.ndarray:
    if indices is not None:
        return _check_indices(indices, n)
    return block_bootstrap_indices(n, block, reps, np.random.default_rng(seed), kind=kind)


def _bootstrap_means(losses: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Mean loss of every model in every bootstrap sample: ``losses[idx].mean(axis=1)``, shape (reps, K)."""
    reps, n = idx.shape
    k = losses.shape[1]
    out = np.empty((reps, k))
    step = max(1, _CHUNK_ELEMENTS // (n * k))
    for s in range(0, reps, step):
        out[s : s + step] = losses[idx[s : s + step]].mean(axis=1)
    return out


def _frozen(a: np.ndarray) -> np.ndarray:
    a = np.array(a)
    a.setflags(write=False)
    return a


# --------------------------------------------------------------------------------------------------------------------
# Diebold–Mariano test
# --------------------------------------------------------------------------------------------------------------------


def default_lag(n: int) -> int:
    """Newey–West rule-of-thumb truncation lag ``floor(4 * (n / 100) ** (2 / 9))``.

    It gives 4 for 182 days, 5 for 365 days and 6 for 730 days (Newey & West 1994, the Bartlett-kernel rule of thumb).

    Args:
        n: Number of observations (days).
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    return int(math.floor(4 * (n / 100) ** (2 / 9)))


def newey_west_variance(d, lag: int) -> float:
    """Newey–West (Bartlett kernel) estimate of the variance of the sample mean of ``d``.

    With ``x = d - mean(d)`` and the biased autocovariances ``g_k = sum_{t=k}^{n-1} x_t x_{t-k} / n``::

        newey_west_variance(d, lag) = (g_0 + 2 * sum_{k=1}^{lag} (1 - k / (lag + 1)) * g_k) / n

    i.e. the long-run variance of ``d`` divided by ``n`` (Newey & West 1987). The Bartlett weights keep it
    non-negative. ``lag=0`` gives ``var(d, ddof=0) / n``, the variance of epftoolbox's Diebold–Mariano test.

    Args:
        d: 1-D series, e.g. a daily loss differential.
        lag: Truncation lag, ``0 <= lag < len(d)``; :func:`default_lag` gives the usual choice.
    """
    d = _as_loss(d, "d")
    n = d.size
    lag = int(lag)
    if n < 1:
        raise ValueError("d is empty")
    if not 0 <= lag < n:
        raise ValueError(f"lag must be in [0, {n - 1}] for {n} observations, got {lag}")
    x = d - d.mean()
    s = x @ x / n
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / n
    return float(s / n)


def dm_test(loss_model, loss_benchmark, variance: str = "hac", lag: Optional[int] = None) -> DMResult:
    """Two-sided Diebold–Mariano test of equal predictive accuracy on paired daily losses.

    The loss differential is ``d_t = loss_benchmark_t - loss_model_t`` and the statistic is
    ``mean(d) / sqrt(V)``, compared with the standard normal distribution (Diebold & Mariano 1995). It is positive
    when the model has the lower mean loss. The Harvey–Leybourne–Newbold (1997) small-sample correction and the
    Student-t reference distribution are not applied.

    With daily losses that are each the mean absolute error over the periods of one day, this is the "multivariate"
    DM test of epftoolbox (Lago et al. 2021).

    Args:
        loss_model: Daily losses of the model, 1-D.
        loss_benchmark: Daily losses of the benchmark on the same days, 1-D.
        variance: ``"hac"`` (default): ``V`` is the Newey–West variance of the mean with ``lag`` (default
            :func:`default_lag`, 5 at 365 days), which allows for autocorrelated loss differentials.
            ``"plain"``: ``V = var(d, ddof=0) / n``, exactly epftoolbox's DM test (whose one-sided p-value is
            ``norm.sf(statistic)``).
        lag: Newey–West lag for ``variance="hac"``. Must be omitted (or 0) for ``"plain"``.

    Returns:
        :class:`DMResult`. If ``d`` is constant, the statistic is 0 (p-value 1) when ``d`` is zero and ±inf
        (p-value 0) otherwise.
    """
    lm, lb = _as_pair(loss_model, loss_benchmark)
    n = lm.size
    if variance == "plain":
        if lag not in (None, 0):
            raise ValueError("lag applies only to variance='hac'")
        lag = 0
    elif variance == "hac":
        lag = default_lag(n) if lag is None else int(lag)
    else:
        raise ValueError(f"variance must be 'hac' or 'plain', got {variance!r}")
    d = lb - lm
    v = newey_west_variance(d, lag)
    mean = float(d.mean())
    if v > 0:
        stat = mean / math.sqrt(v)
    elif mean == 0:
        stat = 0.0
    else:
        stat = math.copysign(math.inf, mean)
    p = float(2 * _sstats.norm.sf(abs(stat)))
    return DMResult(statistic=float(stat), p_value=p, n=n, lag=lag, variance=variance)


def holm(p_values) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values, controlling the family-wise error rate.

    With the ``m`` p-values sorted ascending, ``p_(1) <= ... <= p_(m)``, the adjusted values are
    ``min(1, max_{j <= i} (m - j + 1) * p_(j))``, returned in the input order. Reject hypothesis ``i`` at level
    ``alpha`` when its adjusted p-value is below ``alpha``.

    NaN entries are hypotheses that were not tested: they stay NaN and do not count in ``m``.

    Args:
        p_values: 1-D array-like of p-values in [0, 1].
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"p_values must be 1-D, got shape {p.shape}")
    out = np.full(p.shape, np.nan)
    tested = ~np.isnan(p)
    q = p[tested]
    if ((q < 0) | (q > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    m = q.size
    if m == 0:
        return out
    order = np.argsort(q, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * q[order]))
    res = np.empty(m)
    res[order] = adjusted
    out[tested] = res
    return out


# --------------------------------------------------------------------------------------------------------------------
# Block bootstraps
# --------------------------------------------------------------------------------------------------------------------


def block_bootstrap_indices(
    n: int, block: float, reps: int, rng: SeedLike = None, kind: str = "circular"
) -> np.ndarray:
    """Indices of ``reps`` block-bootstrap samples of a series of length ``n``, shape ``(reps, n)``.

    ``kind="circular"``: circular block bootstrap (Künsch 1989; Politis & Romano 1992). Each sample joins
    ``ceil(n / block)`` blocks of ``block`` consecutive indices with uniform random starts, wrapping around the end,
    and is cut to ``n``.

    ``kind="stationary"``: stationary bootstrap (Politis & Romano 1994). The first index is uniform; each further
    index starts a new block at a uniform position with probability ``1 / block`` and otherwise continues the
    previous one (wrapping around), so block lengths are geometric with mean ``block``.

    The random numbers are drawn in a fixed order (circular: one ``integers`` call for all starts; stationary:
    first indices, then the block-start uniforms, then the jump positions), so a given generator state always gives
    the same indices; this order reproduces the evaluation-protocol study.

    Args:
        n: Length of the series.
        block: Block length (circular, a positive integer) or mean block length (stationary, >= 1).
        reps: Number of bootstrap samples.
        rng: ``numpy.random.Generator`` (advanced in place) or an integer seed.
        kind: ``"circular"`` or ``"stationary"``.
    """
    n, reps = int(n), int(reps)
    if n < 1 or reps < 1:
        raise ValueError(f"n and reps must be positive, got n={n}, reps={reps}")
    rng = np.random.default_rng(rng)
    if kind == "circular":
        if int(block) != block or block < 1:
            raise ValueError(f"block must be a positive integer for the circular bootstrap, got {block}")
        block = int(block)
        nb = int(math.ceil(n / block))
        starts = rng.integers(0, n, size=(reps, nb))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
        return np.ascontiguousarray(idx.reshape(reps, -1)[:, :n])
    if kind == "stationary":
        if not block >= 1:
            raise ValueError(f"the mean block length must be >= 1, got {block}")
        first = rng.integers(0, n, reps)
        new_block = rng.random((reps, n)) < 1 / block
        jump = rng.integers(0, n, (reps, n))
        jump[:, 0] = first
        new_block[:, 0] = True
        t = np.arange(n)
        start = np.maximum.accumulate(np.where(new_block, t, 0), axis=1)
        return (np.take_along_axis(jump, start, axis=1) + (t - start)) % n
    raise ValueError(f"kind must be 'circular' or 'stationary', got {kind!r}")


def ratio_interval(
    loss_model,
    loss_benchmark,
    level: float = 0.95,
    block: float = 7,
    reps: int = 4000,
    seed: SeedLike = 0,
    kind: str = "circular",
    *,
    indices=None,
) -> RatioInterval:
    """Loss ratio ``mean(loss_model) / mean(loss_benchmark)`` with a block-bootstrap percentile interval.

    Both loss series are resampled with the same block-bootstrap indices (weekly blocks by default, which keeps the
    weekly pattern and the serial dependence of daily losses), the ratio of their sums is taken per sample, and the
    interval is given by the ``(1 - level) / 2`` and ``(1 + level) / 2`` quantiles (linear interpolation). An
    interval that excludes 1 says the MAE gap is larger than resampling noise.

    Args:
        loss_model: Daily losses of the model, 1-D.
        loss_benchmark: Daily losses of the benchmark on the same days, 1-D.
        level: Coverage of the interval.
        block: Block length (mean block length for the stationary bootstrap).
        reps: Number of bootstrap samples.
        seed: Seed or ``numpy.random.Generator`` of the bootstrap.
        kind: ``"circular"`` or ``"stationary"``, see :func:`block_bootstrap_indices`.
        indices: Precomputed bootstrap indices of shape ``(reps, n)``; when given, ``block``, ``reps``, ``seed``
            and ``kind`` are ignored.
    """
    lm, lb = _as_pair(loss_model, loss_benchmark)
    level = _check_level(level)
    idx = _resample_indices(lm.size, block, reps, seed, kind, indices)
    r = lm[idx].sum(axis=1) / lb[idx].sum(axis=1)
    tail = (1 - level) / 2
    low, high = np.quantile(r, [tail, 1 - tail])
    return RatioInterval(
        ratio=float(lm.mean() / lb.mean()),
        low=float(low),
        high=float(high),
        level=level,
        n=lm.size,
        reps=idx.shape[0],
    )


# --------------------------------------------------------------------------------------------------------------------
# Model confidence set
# --------------------------------------------------------------------------------------------------------------------


def _mcs_r(losses: np.ndarray, idx: np.ndarray):
    """Elimination with T_R = max_{i,j} t_ij, as arch.bootstrap.MCS(method="R")."""
    k = losses.shape[1]
    reps = idx.shape[0]
    mean = losses.mean(0)[:, None]
    diffs = mean - mean.T  # diffs[i, j] = L_i - L_j; large means i is worse than j
    boot = _bootstrap_means(losses, idx)
    step = max(1, _CHUNK_ELEMENTS // (k * k))

    sq = np.zeros((k, k))
    for s in range(0, reps, step):
        b = boot[s : s + step]
        dev = (b[:, :, None] - b[:, None, :]) - diffs
        sq += (dev**2).sum(0)
    var = sq / reps
    off = ~np.eye(k, dtype=bool)
    if (var[off] <= 0).any():
        i, j = np.argwhere((var <= 0) & off)[0]
        raise ValueError(f"models {i} and {j} have identical losses on every day; drop one before the MCS")
    var += np.eye(k)  # the diagonal is 0 / 1 = 0, as in arch
    sd = np.sqrt(var)
    t = diffs / sd

    # The elimination order depends only on the observed t, so it is fixed before the bootstrap distribution is read.
    included = np.ones(k, dtype=bool)
    order, observed = [], []
    while included.sum() > 1:
        inc = np.flatnonzero(included)
        t_inc = t[np.ix_(inc, inc)]
        observed.append(t_inc.max())
        worst = int(inc[np.unravel_index(np.argmax(t_inc), t_inc.shape)[0]])
        order.append(worst)
        included[worst] = False
    perm = np.array(order + [int(i) for i in np.flatnonzero(included)])

    # In elimination order, step s keeps the models at positions >= s, so pair (i, j) takes part up to step
    # min(i, j). Per sample: the maximum over the pairs that leave at step d, then a running maximum from the end.
    pos = np.arange(k)
    upper = pos[None, :] >= pos[:, None]
    d_p, sd_p = diffs[np.ix_(perm, perm)], sd[np.ix_(perm, perm)]
    simulated = np.empty((reps, k))
    for s in range(0, reps, step):
        b = boot[s : s + step][:, perm]
        z = ((b[:, :, None] - b[:, None, :]) - d_p) / sd_p
        leaving_row = np.where(upper, z, -np.inf).max(axis=2)  # pairs (d, j >= d)
        leaving_col = np.where(upper.T, z, -np.inf).max(axis=1)  # pairs (i >= d, d)
        simulated[s : s + step] = np.maximum(leaving_row, leaving_col)
    simulated = np.maximum.accumulate(simulated[:, ::-1], axis=1)[:, ::-1]
    pvals = [float((stat < simulated[:, step_]).mean()) for step_, stat in enumerate(observed)]
    return order, pvals, included


def _mcs_max(losses: np.ndarray, idx: np.ndarray):
    """Elimination with T_max = max_i t_i (deviation of model i from the average loss), as arch method="max"."""
    k = losses.shape[1]
    errors = losses - losses.mean(0)
    boot = _bootstrap_means(errors, idx)
    boot -= boot.mean(1)[:, None]

    included = np.ones(k, dtype=bool)
    order, pvals = [], []
    while included.sum() > 1:
        inc = np.flatnonzero(included)
        b = boot[:, inc]
        b -= b.mean(1)[:, None]
        sd = np.sqrt((b**2).mean(0))
        if (sd <= 0).any():
            raise ValueError(
                f"model {inc[np.argmax(sd <= 0)]} has a zero bootstrap standard deviation (identical losses?); "
                "the MCS is not valid for this set"
            )
        simulated = (b / sd).max(axis=1)
        dev = losses[:, inc].mean(0)
        dev -= dev.mean()
        t = dev / sd
        stat = t.max()
        pvals_step = float((stat < simulated).mean())
        worst = inc[t == stat]
        order.extend(int(w) for w in worst)
        pvals.extend([pvals_step] * worst.size)
        included[worst] = False
    return order, pvals, included


def model_confidence_set(
    losses,
    alpha: float = 0.10,
    statistic: str = "R",
    block: float = 7,
    reps: int = 1000,
    seed: SeedLike = 0,
    *,
    indices=None,
) -> MCSResult:
    """Model confidence set (Hansen, Lunde & Nason 2011) with the stationary bootstrap.

    The set of models whose mean loss is not significantly worse than the best one's: starting from all ``K`` models,
    test equal predictive ability, eliminate the worst model, and repeat until one model is left. A model's MCS p-value
    is the largest elimination p-value up to its own elimination (1 for the last model); the ``1 - alpha`` set is the
    models with a p-value above ``alpha``.

    Statistics (same definitions as ``arch.bootstrap.MCS(losses, size=alpha, reps=reps, block_size=block,
    method=statistic, bootstrap="stationary")``):

    - ``"R"`` (T_R, default): ``t_ij = (L_i - L_j) / sqrt(var*(L_i - L_j))`` for every pair of included models, with
      ``L`` the mean loss and ``var*`` the bootstrap variance of the mean difference; ``T_R = max_ij t_ij`` and the
      model ``i`` of the maximising pair is eliminated.
    - ``"max"`` (T_max): ``t_i = (L_i - mean_j L_j) / sd*`` for every included model; ``T_max = max_i t_i`` and the
      maximising model is eliminated (all of them on an exact tie).

    The p-value of a step is the share of bootstrap samples whose recentred statistic is larger than the observed one.
    The evaluation-protocol study recommends T_R: T_max changed with the bootstrap seed and kept models up to 17 %
    worse than the best.

    The bootstrap indices come from :func:`block_bootstrap_indices` (``kind="stationary"``) with
    ``numpy.random.default_rng(seed)``, so the numbers differ from arch's for the same seed (arch draws its own
    indices); fed the same indices, the p-values, the set and the elimination order are the same (checked against
    arch 8.0). On an exact tie of T_R between two pairs the first pair in row-major order is eliminated.

    Args:
        losses: ``n x K`` daily losses, one column per model, on common days.
        alpha: Size of the test; the result's :meth:`MCSResult.included` uses it.
        statistic: ``"R"`` or ``"max"``.
        block: Mean block length of the stationary bootstrap (days).
        reps: Number of bootstrap samples (at least 1,000 recommended).
        seed: Seed or ``numpy.random.Generator`` of the bootstrap.
        indices: Precomputed bootstrap indices of shape ``(reps, n)``; when given, ``block``, ``reps`` and ``seed``
            are ignored.

    Raises:
        ValueError: on fewer than 2 models, non-finite losses, or two models with identical losses on every day
            (their difference has no variance; drop one).
    """
    L = _as_losses(losses, min_models=2)
    alpha = _check_level(alpha, "alpha")
    if statistic not in ("R", "max"):
        raise ValueError(f"statistic must be 'R' or 'max', got {statistic!r}")
    idx = _resample_indices(L.shape[0], block, reps, seed, "stationary", indices)
    order, pvals, included = (_mcs_r if statistic == "R" else _mcs_max)(L, idx)
    last = np.flatnonzero(included)
    order.extend(int(i) for i in last)
    pvals.extend([1.0] * last.size)
    p = np.empty(L.shape[1])
    p[order] = np.maximum.accumulate(pvals)
    return MCSResult(
        p_values=_frozen(p),
        elimination_order=tuple(order),
        alpha=alpha,
        statistic=statistic,
        n=L.shape[0],
        reps=idx.shape[0],
    )


# --------------------------------------------------------------------------------------------------------------------
# Ranking uncertainty and test power
# --------------------------------------------------------------------------------------------------------------------


def rank_bootstrap(
    losses,
    block: float = 7,
    reps: int = 4000,
    seed: SeedLike = 0,
    level: float = 0.90,
    kind: str = "circular",
    *,
    indices=None,
) -> RankResult:
    """How decisive is the ranking? Block-bootstrap distribution of every model's rank by mean loss.

    In every bootstrap sample of days (weekly circular blocks by default) the models are ranked by mean loss
    (1 = lowest; exact ties go to the earlier column). ``p_best`` is the share of samples in which a model ranks
    first, and the rank interval is the ``(1 - level) / 2`` and ``(1 + level) / 2`` quantiles of its rank, rounded
    outwards to whole ranks (the lower end takes the lower order statistic, the upper end the higher one).

    Args:
        losses: ``n x K`` daily losses, one column per model, on common days.
        block: Block length (mean block length for the stationary bootstrap).
        reps: Number of bootstrap samples.
        seed: Seed or ``numpy.random.Generator`` of the bootstrap.
        level: Coverage of the rank interval.
        kind: ``"circular"`` or ``"stationary"``, see :func:`block_bootstrap_indices`.
        indices: Precomputed bootstrap indices of shape ``(reps, n)``; when given, ``block``, ``reps``, ``seed``
            and ``kind`` are ignored.
    """
    L = _as_losses(losses, min_models=1)
    level = _check_level(level)
    idx = _resample_indices(L.shape[0], block, reps, seed, kind, indices)
    boot = _bootstrap_means(L, idx)
    ranks = boot.argsort(axis=1, kind="stable").argsort(axis=1, kind="stable") + 1
    mean = L.mean(0)
    tail = (1 - level) / 2
    return RankResult(
        loss=_frozen(mean),
        rank=_frozen(mean.argsort(kind="stable").argsort(kind="stable") + 1),
        p_best=_frozen((ranks == 1).mean(axis=0)),
        rank_low=_frozen(np.quantile(ranks, tail, axis=0, method="lower").astype(int)),
        rank_high=_frozen(np.quantile(ranks, 1 - tail, axis=0, method="higher").astype(int)),
        level=level,
        n=L.shape[0],
        reps=idx.shape[0],
    )


def minimum_detectable_effect(
    loss_model,
    loss_benchmark,
    lag: Optional[int] = None,
    alpha: float = 0.05,
    power: float = 0.80,
    days: Optional[int] = None,
) -> float:
    """Smallest difference in mean loss, as a fraction of the benchmark's mean loss, that the HAC DM test detects.

    ``MDE = (z_{1 - alpha/2} + z_{power}) * sigma_LR / (sqrt(days) * mean(loss_benchmark))`` where ``sigma_LR^2`` is
    the Newey–West long-run variance of the daily loss differential (``n * newey_west_variance(d, lag)``) measured on
    the given days. At the defaults the factor is 1.9600 + 0.8416 = 2.8016 (the evaluation-protocol study used the
    rounded 2.80). A smaller gap is likely to go undetected; on the model zoo the MDE was about 5 % of the MAE at 365
    days.

    Args:
        loss_model: Daily losses of the model, 1-D.
        loss_benchmark: Daily losses of the benchmark on the same days, 1-D.
        lag: Newey–West lag (default :func:`default_lag` of the number of days).
        alpha: Two-sided size of the test.
        power: Probability of detecting a difference of the MDE.
        days: Test length to evaluate the MDE for (default: the number of days given); the long-run variance is
            still measured on the given days.
    """
    lm, lb = _as_pair(loss_model, loss_benchmark)
    alpha = _check_level(alpha, "alpha")
    power = _check_level(power, "power")
    n = lm.size
    lag = default_lag(n) if lag is None else int(lag)
    m = n if days is None else int(days)
    if m < 1:
        raise ValueError(f"days must be positive, got {m}")
    scale = float(lb.mean())
    if scale <= 0:
        raise ValueError("the benchmark's mean loss must be positive")
    sigma_lr = math.sqrt(n * newey_west_variance(lb - lm, lag))
    z = float(_sstats.norm.ppf(1 - alpha / 2) + _sstats.norm.ppf(power))
    return z * sigma_lr / math.sqrt(m) / scale
