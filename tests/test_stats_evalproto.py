"""The statistics module reproduces the evaluation-protocol study (evalproto) on stored model-zoo forecasts.

Fixture ``tests/fixtures/evalproto/``:

- ``de_honest_placebo.npz``: the DE_LU HONEST test year 2025-10-01..2026-09-30 (365 days x 24 hours, float64) as
  ``evalproto/build.py`` rebuilds it. ``Y`` is ``P[i0:i1]`` of ``recent/data/DE_LU.npz`` (``test = (i0, i1)``) and
  ``<family>__<variant>__<seed>`` are the test forecasts of five placebo groups (one model trained with different
  seeds) from ``pred_<family>_DE_LU_honest_<variant>[_s<k>]_test.csv`` under ``zoo_dl/runs`` and ``zoo_tsfm/runs``,
  parsed with build.py's ``RX`` and ``SEED_ALIAS`` (t0afts1/t0afts2 are seeds s1/s2 of t0aft). Source: WCSS,
  ``/lustre/pd03/hpc-dawlin1140-1769424956/epf2_modeling_research/``. Every day of the test year is complete, so
  build.py's ``ok`` mask keeps all 365 days.
- ``q1_placebo_de_honest.csv``: the 15 matching rows of ``evalproto/out/q1_placebo.csv``, copied verbatim.

``evalproto/q1_tests.py`` compares seed ``a`` (the benchmark) with seed ``b`` (the model) on the daily mean absolute
error: positive statistics mean ``b`` is better and ``ratio = MAE_b / MAE_a``. Its bootstrap indices come from one
generator, ``default_rng(20261018)``, drawn per setting as circular (7-day blocks, 4000 samples) and then stationary
(mean block 7, 4000 samples); DE HONEST is the first setting, so those two draws reproduce its intervals exactly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sstats

from epftoolbox2.evaluators.stats import (
    block_bootstrap_indices,
    dm_test,
    holm,
    ratio_interval,
)

FIXTURE = Path(__file__).parent / "fixtures" / "evalproto"
ROWS = pd.read_csv(FIXTURE / "q1_placebo_de_honest.csv")
ROW_IDS = [f"{r.family}-{r.a}-{r.b}" for r in ROWS.itertuples()]
TOL = 1e-10


@pytest.fixture(scope="module")
def data():
    with np.load(FIXTURE / "de_honest_placebo.npz") as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def evalproto_indices():
    rng = np.random.default_rng(20261018)
    circular = block_bootstrap_indices(365, 7, 4000, rng, kind="circular")
    stationary = block_bootstrap_indices(365, 7, 4000, rng, kind="stationary")
    return circular, stationary


def errors(data, row):
    """Absolute errors (days x hours) of the benchmark seed a and the model seed b."""
    y = data["Y"]
    e_a = np.abs(y - data[f"{row.family}__{row.variant}__{row.a}"])
    e_b = np.abs(y - data[f"{row.family}__{row.variant}__{row.b}"])
    return e_a, e_b


def test_fixture_is_the_full_de_honest_test_year(data):
    assert data["Y"].shape == (365, 24)
    assert data["Y"].dtype == np.float64
    assert np.isfinite(data["Y"]).all()
    assert (data["dates"][0], data["dates"][-1]) == ("2025-10-01", "2026-09-30")
    assert len(ROWS) == 15 and (ROWS.n_days == 365).all()


@pytest.mark.parametrize("row", list(ROWS.itertuples()), ids=ROW_IDS)
def test_dm_statistics_p_values_and_ratio(data, row):
    e_a, e_b = errors(data, row)
    loss_a, loss_b = e_a.mean(axis=1), e_b.mean(axis=1)

    plain = dm_test(loss_b, loss_a, variance="plain")
    hac5 = dm_test(loss_b, loss_a)
    hac7 = dm_test(loss_b, loss_a, lag=7)
    hac14 = dm_test(loss_b, loss_a, lag=14)

    assert (plain.n, plain.lag, hac5.lag) == (365, 0, 5)
    assert plain.statistic == pytest.approx(row.z_plain, abs=TOL)
    assert plain.p_value == pytest.approx(row.p_plain, abs=TOL)
    assert sstats.norm.sf(plain.statistic) == pytest.approx(row.p_one, abs=TOL)  # epftoolbox's one-sided p-value
    assert hac5.statistic == pytest.approx(row.z_hac5, abs=TOL)
    assert hac5.p_value == pytest.approx(row.p_hac5, abs=TOL)
    assert hac7.p_value == pytest.approx(row.p_hac7, abs=TOL)
    assert hac14.p_value == pytest.approx(row.p_hac14, abs=TOL)
    assert ratio_interval(loss_b, loss_a, reps=10).ratio == pytest.approx(row.ratio, abs=TOL)


@pytest.mark.parametrize("row", list(ROWS.itertuples()), ids=ROW_IDS)
def test_per_hour_dm_with_holm_over_hours(data, row):
    e_a, e_b = errors(data, row)
    results = [dm_test(e_b[:, h], e_a[:, h], variance="plain") for h in range(24)]
    z = np.array([r.statistic for r in results])
    p = np.array([r.p_value for r in results])
    adjusted = holm(p)

    assert int(((adjusted < 0.05) & (z > 0)).sum()) == row.hours_better
    assert int(((adjusted < 0.05) & (z < 0)).sum()) == row.hours_worse
    assert int((p < 0.05).sum()) == row.hours_raw
    assert adjusted.min() == pytest.approx(row.min_padj, abs=TOL)


@pytest.mark.parametrize("row", list(ROWS.itertuples()), ids=ROW_IDS)
def test_bootstrap_ratio_intervals_match_exactly_with_evalprotos_indices(data, evalproto_indices, row):
    e_a, e_b = errors(data, row)
    loss_a, loss_b = e_a.mean(axis=1), e_b.mean(axis=1)
    circular, stationary = evalproto_indices

    mbb = ratio_interval(loss_b, loss_a, indices=circular)
    sb = ratio_interval(loss_b, loss_a, indices=stationary)

    assert (mbb.low, mbb.high) == pytest.approx((row.mbb_lo, row.mbb_hi), abs=1e-12)
    assert (sb.low, sb.high) == pytest.approx((row.sb_lo, row.sb_hi), abs=1e-12)


@pytest.mark.parametrize("row", list(ROWS.itertuples()), ids=ROW_IDS)
def test_bootstrap_ratio_intervals_with_the_default_seed_are_close(data, row):
    """Another seed moves the 4000-sample percentile bounds by Monte Carlo noise only (here at most about 0.002)."""
    e_a, e_b = errors(data, row)
    loss_a, loss_b = e_a.mean(axis=1), e_b.mean(axis=1)

    mbb = ratio_interval(loss_b, loss_a)
    sb = ratio_interval(loss_b, loss_a, kind="stationary")

    assert (mbb.low, mbb.high) == pytest.approx((row.mbb_lo, row.mbb_hi), abs=0.005)
    assert (sb.low, sb.high) == pytest.approx((row.sb_lo, row.sb_hi), abs=0.005)
