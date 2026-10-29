"""Known-answer tests for epftoolbox2.evaluators.stats on synthetic losses.

The reproduction of the evaluation-protocol study on real forecasts is in tests/test_stats_evalproto.py. The
cross-checks against ``arch.bootstrap.MCS`` run only where arch is installed (it is not a dependency):
``uv run --isolated --python 3.12 --extra dev --with arch pytest tests/test_stats.py``.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sstats

from epftoolbox2.evaluators.stats import (
    block_bootstrap_indices,
    default_lag,
    dm_test,
    holm,
    minimum_detectable_effect,
    model_confidence_set,
    newey_west_variance,
    rank_bootstrap,
    ratio_interval,
)

FIXTURE = Path(__file__).parent / "fixtures" / "evalproto" / "de_honest_placebo.npz"


def ar1(n, rho, rng):
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0] / math.sqrt(1 - rho**2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x


def daily_losses(n, k, rng, shift=None):
    """Positive, cross-correlated daily losses of k models; column j has mean about 10 * (1 + shift[j])."""
    shift = np.zeros(k) if shift is None else np.asarray(shift, float)
    common = 3 * rng.standard_normal((n, 1))
    return np.abs(10 + common + 2 * rng.standard_normal((n, k))) * (1 + shift)


# --------------------------------------------------------------------------------------------------------------------
# Newey–West variance and the lag
# --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n, lag", [(1, 1), (100, 4), (182, 4), (365, 5), (730, 6), (2700, 8)])
def test_default_lag(n, lag):
    assert default_lag(n) == lag == math.floor(4 * (n / 100) ** (2 / 9))


def test_newey_west_variance_by_hand():
    # x = d - mean = (-1.5, -0.5, 0.5, 1.5); g0 = 5 / 4; g1 = (0.75 - 0.25 + 0.75) / 4 = 0.3125
    # lag 1: (g0 + 2 * (1 - 1/2) * g1) / n = (1.25 + 0.3125) / 4
    d = np.array([1.0, 2.0, 3.0, 4.0])
    assert newey_west_variance(d, 1) == pytest.approx(1.5625 / 4, abs=1e-15)
    assert newey_west_variance(d, 0) == pytest.approx(d.var() / d.size, abs=1e-15)


def test_newey_west_variance_is_never_negative():
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = rng.standard_normal(int(rng.integers(5, 60)))
        d += 3 * (-1) ** np.arange(d.size)  # strongly negatively autocorrelated
        for lag in range(d.size):
            assert newey_west_variance(d, lag) >= 0


def test_newey_west_variance_rejects_bad_lags_and_values():
    with pytest.raises(ValueError, match="lag"):
        newey_west_variance(np.ones(5), 5)
    with pytest.raises(ValueError, match="lag"):
        newey_west_variance(np.ones(5), -1)
    with pytest.raises(ValueError, match="non-finite"):
        newey_west_variance(np.array([1.0, np.nan, 2.0]), 1)


# --------------------------------------------------------------------------------------------------------------------
# Diebold–Mariano test
# --------------------------------------------------------------------------------------------------------------------


def test_dm_statistic_is_positive_when_the_model_is_better():
    rng = np.random.default_rng(1)
    benchmark = 10 + rng.standard_normal(365)
    model = benchmark - 0.5 + 0.3 * rng.standard_normal(365)

    better = dm_test(model, benchmark)
    worse = dm_test(benchmark, model)

    assert better.statistic > 0 and better.p_value < 0.05
    assert worse.statistic == pytest.approx(-better.statistic)
    assert worse.p_value == pytest.approx(better.p_value)


def test_plain_dm_is_epftoolboxs_test():
    """epftoolbox: d = |e_1| - |e_2| averaged per day, DM = mean(d) / sqrt(var(d, ddof=0) / n), one-sided p."""
    rng = np.random.default_rng(2)
    benchmark, model = np.abs(rng.standard_normal((2, 365)))
    d = benchmark - model

    res = dm_test(model, benchmark, variance="plain")

    assert res.statistic == pytest.approx(d.mean() / np.sqrt(d.var() / d.size), rel=1e-12)
    assert res.p_value == pytest.approx(2 * sstats.norm.sf(abs(res.statistic)), rel=1e-12)
    assert (res.lag, res.n, res.variance) == (0, 365, "plain")


def test_hac_variance_matches_plain_for_an_iid_differential():
    rng = np.random.default_rng(3)
    benchmark = 10 + rng.standard_normal(20_000)
    model = benchmark + rng.standard_normal(20_000)

    plain = dm_test(model, benchmark, variance="plain")
    hac = dm_test(model, benchmark)

    assert hac.lag == default_lag(20_000)
    assert (plain.statistic / hac.statistic) ** 2 == pytest.approx(1.0, abs=0.05)


def test_hac_variance_is_larger_for_an_ar1_differential():
    """AR(1) with rho = 0.5: the Bartlett-weighted long-run variance is 1 + 2 sum_k (1 - k/(L+1)) rho^k times the
    marginal variance (2.60 at L = 9), while the plain variance ignores the autocorrelation."""
    rng = np.random.default_rng(4)
    n, rho = 20_000, 0.5
    d = ar1(n, rho, rng)
    benchmark = 10 + rng.standard_normal(n)
    model = benchmark - d

    plain = dm_test(model, benchmark, variance="plain")
    hac = dm_test(model, benchmark)
    lag = hac.lag
    expected = 1 + 2 * sum((1 - k / (lag + 1)) * rho**k for k in range(1, lag + 1))

    assert (plain.statistic / hac.statistic) ** 2 == pytest.approx(expected, rel=0.08)


def test_hac_keeps_the_size_closer_to_nominal_than_plain_under_autocorrelation():
    """Equal accuracy, AR(1) differential (rho = 0.5), 365 days: the plain DM rejects far more often than 5 %."""
    rng = np.random.default_rng(5)
    reject_plain = reject_hac = 0
    sims = 600
    for _ in range(sims):
        d = ar1(365, 0.5, rng)
        model, benchmark = np.zeros(365), d
        reject_plain += dm_test(model, benchmark, variance="plain").p_value < 0.05
        reject_hac += dm_test(model, benchmark).p_value < 0.05
    assert reject_plain / sims > 0.18
    assert reject_hac / sims < 0.12


def test_dm_constant_differentials():
    x = np.linspace(1, 2, 30)
    same = dm_test(x, x)
    assert (same.statistic, same.p_value) == (0.0, 1.0)
    better = dm_test(x - 0.1, x, variance="plain")
    assert better.p_value < 1e-12 and better.statistic > 0


def test_dm_rejects_bad_input():
    x = np.ones(10)
    with pytest.raises(ValueError, match="non-finite"):
        dm_test(np.r_[x[:-1], np.nan], x)
    with pytest.raises(ValueError, match="length"):
        dm_test(x, x[:-1])
    with pytest.raises(ValueError, match="1-D"):
        dm_test(np.ones((10, 24)), np.ones((10, 24)))
    with pytest.raises(ValueError, match="lag"):
        dm_test(x, x, variance="plain", lag=5)
    with pytest.raises(ValueError, match="variance"):
        dm_test(x, x, variance="hln")


# --------------------------------------------------------------------------------------------------------------------
# Holm
# --------------------------------------------------------------------------------------------------------------------


def test_holm_by_hand():
    # sorted: 0.005 (x4), 0.01 (x3), 0.03 (x2), 0.04 (x1) -> 0.02, 0.03, 0.06, max(0.06, 0.04)
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02], abs=1e-15)
    assert holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])
    assert holm([0.02]) == pytest.approx([0.02])
    assert holm([]).size == 0


def test_holm_matches_the_definition_on_random_p_values():
    rng = np.random.default_rng(6)
    for _ in range(20):
        p = rng.random(int(rng.integers(1, 40))) ** 3
        m = p.size
        order = np.argsort(p)
        expected = np.empty(m)
        for i in range(m):
            rank = int(np.flatnonzero(order == i)[0])
            expected[i] = min(1.0, max((m - j) * p[order[j]] for j in range(rank + 1)))
        assert holm(p) == pytest.approx(expected, abs=1e-15)


def test_holm_leaves_untested_hypotheses_out():
    out = holm([0.01, np.nan, 0.04])
    assert out[0] == pytest.approx(0.02) and np.isnan(out[1]) and out[2] == pytest.approx(0.04)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm([0.5, 1.5])


# --------------------------------------------------------------------------------------------------------------------
# Block bootstrap indices
# --------------------------------------------------------------------------------------------------------------------


def reference_stationary(n, mean_block, reps, rng):
    """The evaluation-protocol study's loop (evalproto/ev.py stationary_index)."""
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    new = rng.random((reps, n)) < 1 / mean_block
    rnd = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(new[:, t], rnd[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def reference_circular(n, block, reps, rng):
    """The evaluation-protocol study's circular block bootstrap (evalproto/ev.py mbb_index)."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(reps, nb))
    return ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(reps, -1)[:, :n]


@pytest.mark.parametrize("n, block", [(365, 7), (50, 3.5), (10, 1)])
def test_stationary_indices_reproduce_the_reference_loop(n, block):
    ours = block_bootstrap_indices(n, block, 300, np.random.default_rng(11), kind="stationary")
    assert np.array_equal(ours, reference_stationary(n, block, 300, np.random.default_rng(11)))


@pytest.mark.parametrize("n, block", [(365, 7), (50, 3), (10, 20)])
def test_circular_indices_reproduce_the_reference(n, block):
    ours = block_bootstrap_indices(n, block, 300, np.random.default_rng(12))
    assert np.array_equal(ours, reference_circular(n, block, 300, np.random.default_rng(12)))


def test_circular_blocks_are_consecutive_and_wrap():
    idx = block_bootstrap_indices(30, 7, 200, 0)
    assert idx.shape == (200, 30) and idx.min() >= 0 and idx.max() < 30
    step = (np.diff(idx, axis=1) % 30)[:, [t for t in range(29) if (t + 1) % 7]]
    assert (step == 1).all()


def test_stationary_blocks_have_the_requested_mean_length():
    n, block = 365, 7
    idx = block_bootstrap_indices(n, block, 4000, 1, kind="stationary")
    breaks = (idx[:, 1:] != (idx[:, :-1] + 1) % n).sum()
    # a new block lands on the continuation with probability 1/n and is invisible
    p_visible = (1 / block) * (1 - 1 / n)
    assert breaks / (idx.shape[0] * (n - 1)) == pytest.approx(p_visible, rel=0.02)
    counts = np.bincount(idx.ravel(), minlength=n) / idx.size
    assert counts == pytest.approx(np.full(n, 1 / n), rel=0.06)


def test_indices_are_reproducible_and_advance_a_generator():
    a = block_bootstrap_indices(100, 7, 50, 3, kind="stationary")
    b = block_bootstrap_indices(100, 7, 50, 3, kind="stationary")
    assert np.array_equal(a, b)
    rng = np.random.default_rng(3)
    first = block_bootstrap_indices(100, 7, 50, rng, kind="stationary")
    second = block_bootstrap_indices(100, 7, 50, rng, kind="stationary")
    assert np.array_equal(first, a) and not np.array_equal(first, second)


def test_block_bootstrap_rejects_bad_arguments():
    with pytest.raises(ValueError, match="integer"):
        block_bootstrap_indices(100, 3.5, 10, 0, kind="circular")
    with pytest.raises(ValueError, match=">= 1"):
        block_bootstrap_indices(100, 0.5, 10, 0, kind="stationary")
    with pytest.raises(ValueError, match="kind"):
        block_bootstrap_indices(100, 7, 10, 0, kind="moving")


# --------------------------------------------------------------------------------------------------------------------
# Ratio interval
# --------------------------------------------------------------------------------------------------------------------


def test_ratio_interval_of_a_proportional_loss_is_exact():
    benchmark = daily_losses(365, 1, np.random.default_rng(7))[:, 0]
    res = ratio_interval(0.9 * benchmark, benchmark)
    assert (res.ratio, res.low, res.high) == pytest.approx((0.9, 0.9, 0.9), abs=1e-12)
    assert (res.level, res.n, res.reps) == (0.95, 365, 4000)


def test_ratio_interval_brackets_the_ratio_and_widens_with_the_level():
    losses = daily_losses(365, 2, np.random.default_rng(8), shift=[0.0, 0.03])
    narrow = ratio_interval(losses[:, 1], losses[:, 0], level=0.80)
    wide = ratio_interval(losses[:, 1], losses[:, 0], level=0.99)
    assert wide.low < narrow.low < narrow.ratio < narrow.high < wide.high
    assert narrow.ratio == pytest.approx(losses[:, 1].mean() / losses[:, 0].mean())


def test_ratio_interval_uses_given_indices():
    losses = daily_losses(200, 2, np.random.default_rng(9))
    idx = block_bootstrap_indices(200, 7, 500, np.random.default_rng(1), kind="stationary")
    given = ratio_interval(losses[:, 1], losses[:, 0], indices=idx)
    seeded = ratio_interval(losses[:, 1], losses[:, 0], reps=500, seed=1, kind="stationary")
    assert (given.low, given.high) == (seeded.low, seeded.high)
    with pytest.raises(ValueError, match="shape"):
        ratio_interval(losses[:, 1], losses[:, 0], indices=idx[:, :-1])


# --------------------------------------------------------------------------------------------------------------------
# Model confidence set
# --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("statistic", ["R", "max"])
def test_mcs_keeps_only_the_true_best_on_a_clear_cut_case(statistic):
    rng = np.random.default_rng(10)
    losses = daily_losses(365, 6, rng, shift=[0.4, 0.4, 0.0, 0.4, 0.45, 0.5])

    res = model_confidence_set(losses, statistic=statistic)

    assert np.flatnonzero(res.included()).tolist() == [2]
    assert res.p_values[2] == 1.0 and res.elimination_order[-1] == 2
    assert (np.delete(res.p_values, 2) < 0.01).all()


@pytest.mark.parametrize("statistic", ["R", "max"])
def test_mcs_keeps_the_true_best_among_close_models(statistic):
    rng = np.random.default_rng(13)
    losses = daily_losses(365, 8, rng, shift=[0.02, 0.0, 0.01, 0.3, 0.3, 0.35, 0.4, 0.03])

    res = model_confidence_set(losses, statistic=statistic, reps=2000)

    assert res.included()[1]
    assert not res.included()[3:7].any()
    assert res.included(alpha=0.5).sum() <= res.included().sum()


def test_mcs_p_values_are_monotone_in_elimination_order_and_follow_columns():
    rng = np.random.default_rng(14)
    losses = daily_losses(300, 7, rng, shift=np.linspace(0, 0.08, 7))
    idx = block_bootstrap_indices(300, 7, 1000, 0, kind="stationary")

    res = model_confidence_set(losses, indices=idx)
    perm = rng.permutation(7)
    permuted = model_confidence_set(losses[:, perm], indices=idx)

    assert sorted(res.elimination_order) == list(range(7))
    assert np.all(np.diff(res.p_values[list(res.elimination_order)]) >= 0)
    assert permuted.p_values == pytest.approx(res.p_values[perm])
    assert res.p_values.flags.writeable is False


def test_mcs_is_reproducible_with_a_seed():
    losses = daily_losses(200, 5, np.random.default_rng(15), shift=np.linspace(0, 0.05, 5))
    a, b = model_confidence_set(losses, seed=4), model_confidence_set(losses, seed=4)
    assert np.array_equal(a.p_values, b.p_values) and a.elimination_order == b.elimination_order


def test_mcs_rejects_duplicated_models_and_bad_input():
    losses = daily_losses(100, 3, np.random.default_rng(16))
    with pytest.raises(ValueError, match="identical losses"):
        model_confidence_set(np.column_stack([losses, losses[:, 0]]))
    with pytest.raises(ValueError, match="2 models"):
        model_confidence_set(losses[:, :1])
    with pytest.raises(ValueError, match="non-finite"):
        model_confidence_set(np.where(losses > 15, np.nan, losses))
    with pytest.raises(ValueError, match="statistic"):
        model_confidence_set(losses, statistic="SQ")


def arch_mcs(losses, method, reps, seed):
    arch = pytest.importorskip("arch.bootstrap")
    mcs = arch.MCS(losses, size=0.10, reps=reps, block_size=7, method=method, bootstrap="stationary", seed=seed)
    mcs.compute()
    p = mcs.pvalues["Pvalue"].reindex(range(losses.shape[1])).to_numpy()
    return mcs, p


def arch_designs():
    rng = np.random.default_rng(17)
    for _ in range(12):
        k, n = int(rng.integers(3, 25)), int(rng.integers(60, 400))
        yield daily_losses(n, k, rng, shift=rng.uniform(0, 0.12, k)), int(rng.integers(1_000_000))
    with np.load(FIXTURE) as z:
        y = z["Y"]
        yield np.column_stack([np.abs(y - z[k]).mean(axis=1) for k in z.files if "__" in k]), 1


@pytest.mark.parametrize("method", ["R", "max"])
def test_mcs_equals_arch_given_the_same_bootstrap_samples(method):
    """Fed arch's own bootstrap indices, the p-values, the set and the elimination order are identical."""
    for losses, seed in arch_designs():
        mcs, p_arch = arch_mcs(losses, method, reps=1000, seed=seed)
        if not hasattr(mcs, "_bootstrap_indices"):
            pytest.skip("this arch version does not expose its bootstrap indices")

        ours = model_confidence_set(losses, statistic=method, indices=np.vstack(mcs._bootstrap_indices))

        assert np.array_equal(ours.p_values, p_arch)
        assert np.flatnonzero(ours.included()).tolist() == sorted(mcs.included)
        assert list(ours.elimination_order) == list(mcs.pvalues.index)


@pytest.mark.parametrize("method", ["R", "max"])
def test_mcs_agrees_with_arch_under_independent_bootstraps(method):
    """With independent random numbers the p-values differ by Monte Carlo error only.

    At 2000 samples the standard error of one bootstrap p-value is at most sqrt(0.25 / 2000) = 0.011, so two
    independent estimates differ by at most about 4 * sqrt(2) * 0.011 = 0.063; arch against itself with two seeds
    differs by as much (up to 0.047 over 30 such sets).
    """
    rng = np.random.default_rng(18)
    for _ in range(4):
        losses = daily_losses(365, 8, rng, shift=np.linspace(0, 0.06, 8))
        _, p_arch = arch_mcs(losses, method, reps=2000, seed=1)

        ours = model_confidence_set(losses, statistic=method, reps=2000, seed=0)

        assert np.abs(ours.p_values - p_arch).max() < 0.063


def test_mcs_with_64_models_takes_seconds():
    import time

    losses = daily_losses(365, 64, np.random.default_rng(19), shift=np.linspace(0, 0.2, 64))
    start = time.perf_counter()
    model_confidence_set(losses, reps=1000)
    assert time.perf_counter() - start < 10


# --------------------------------------------------------------------------------------------------------------------
# Rank bootstrap
# --------------------------------------------------------------------------------------------------------------------


def test_rank_bootstrap_on_a_clear_ordering():
    losses = daily_losses(365, 3, np.random.default_rng(20), shift=[0.5, 0.0, 1.0])
    res = rank_bootstrap(losses)
    assert res.rank.tolist() == [2, 1, 3]
    assert res.p_best.tolist() == [0.0, 1.0, 0.0]
    assert res.rank_low.tolist() == res.rank_high.tolist() == [2, 1, 3]
    assert res.loss == pytest.approx(losses.mean(axis=0))


def test_rank_bootstrap_splits_p_best_between_equal_models():
    """Two models with the same mean loss whose daily differences are symmetric about zero: P(best) is about 1/2
    each, and their rank interval is 1-2."""
    rng = np.random.default_rng(21)
    base = daily_losses(365, 1, rng)[:, 0]
    e = rng.standard_normal(365)
    losses = np.column_stack([base + e - e.mean(), base - e + e.mean(), 1.5 * base])

    res = rank_bootstrap(losses)

    assert res.loss[0] == pytest.approx(res.loss[1])
    assert res.p_best.sum() == pytest.approx(1.0)
    assert res.p_best[:2] == pytest.approx([0.5, 0.5], abs=0.05)
    assert res.rank_low.tolist() == [1, 1, 3] and res.rank_high.tolist() == [2, 2, 3]


def test_rank_bootstrap_follows_the_studys_definition():
    """evalproto/q3_rank.py: ranks of the bootstrap mean losses; p_best = share of rank 1; the interval rounds the
    linear-interpolation quantiles outwards to whole ranks."""
    rng = np.random.default_rng(22)
    losses = daily_losses(365, 10, rng, shift=np.linspace(0, 0.05, 10))
    idx = block_bootstrap_indices(365, 7, 4000, np.random.default_rng(7))

    res = rank_bootstrap(losses, indices=idx)
    ranks = losses[idx].mean(axis=1).argsort(axis=1).argsort(axis=1) + 1
    low, high = np.quantile(ranks, [0.05, 0.95], axis=0)

    assert res.p_best == pytest.approx((ranks == 1).mean(axis=0), abs=0)
    assert (res.rank_low <= low).all() and (low - res.rank_low < 1).all()
    assert (res.rank_high >= high).all() and (res.rank_high - high < 1).all()


# --------------------------------------------------------------------------------------------------------------------
# Minimum detectable effect
# --------------------------------------------------------------------------------------------------------------------


def test_a_gap_of_one_mde_gives_the_hac_statistic_its_power_quantile():
    """If the mean differential equals MDE * MAE_benchmark, the HAC DM statistic is z_{0.975} + z_{0.80}."""
    rng = np.random.default_rng(23)
    benchmark = daily_losses(365, 1, rng)[:, 0]
    noise = ar1(365, 0.2, rng)
    model = benchmark - (noise - noise.mean())  # mean differential exactly 0, same long-run variance

    mde = minimum_detectable_effect(model, benchmark)
    shifted = model - mde * benchmark.mean()

    z = sstats.norm.ppf(0.975) + sstats.norm.ppf(0.80)
    assert z == pytest.approx(2.8016, abs=1e-4)
    assert dm_test(shifted, benchmark).statistic == pytest.approx(z, rel=1e-9)
    assert minimum_detectable_effect(shifted, benchmark) == pytest.approx(mde, rel=1e-9)


def test_mde_scales_with_the_square_root_of_the_test_length():
    rng = np.random.default_rng(24)
    losses = daily_losses(365, 2, rng)
    mde = minimum_detectable_effect(losses[:, 1], losses[:, 0])
    assert minimum_detectable_effect(losses[:, 1], losses[:, 0], days=4 * 365) == pytest.approx(mde / 2)
    # the study's formula (evalproto/q7_mde.py), which rounds the factor to 2.80
    lrv = 365 * newey_west_variance(losses[:, 0] - losses[:, 1], 5)
    study = 2.80 * math.sqrt(lrv) / math.sqrt(365) / losses[:, 0].mean()
    z = sstats.norm.ppf(0.975) + sstats.norm.ppf(0.80)
    assert mde == pytest.approx(study * z / 2.80, rel=1e-12)
