"""OLS's refinement step and its issue-row guard (`_stats.refined`, `_stats.solve_error`, `solve_error_ok`,
`OLS_SOLVE_ERROR_MAX`; `fit_path` "engine-2").

The engine fits OLS by the normal equations, which err as eps·cond(Z)² where the model path's SVD errs as eps·cond(Z), and an
issue row far out reads that error times its distance: on Lago's DE inputs (n 364, p 247, every issue row 5 to 30 training
standard deviations out) engine-1 kept forecasts 1.4e-8 EUR/MWh from engine off, the model path itself within 1e-11 of a
long-double solution. One step of iterative refinement (the residual well below float64's rounding, by error-free
transformations) removes the factorisation's error; what it can't remove, the rounding of the equations' entries (larger where
the centring of the sums cancelled, the fit's rows away from its span's mean), the guard estimates per row and sends a row whose
estimate is above `OLS_SOLVE_ERROR_MAX` back to the model path (`far_issue`). The calibration is in
`epftoolbox2-research/library_check_step5/ols_issue_guard/`.
"""

import json
from collections import Counter
from fractions import Fraction

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from epftoolbox2.models import OLSModel, RidgeModel, _engine_worker, _stats
from epftoolbox2.models._sums import SpanSums
from epftoolbox2.models._worker import StatsFit, TaskInputs
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import ManifestMismatchError
from epftoolbox2.scalers.standard import StandardScaler
from tests.models.test_engine_vst_cv import engine_on_arrays
from tests.models.test_stats_fits import lear_design
from tests.synthetic import make_market_data


@pytest.fixture(autouse=True)
def one_blas_thread():
    with threadpool_limits(limits=1):  # what the engine's jobs with fits from statistics run on
        yield


def ill_conditioned(seed=36, n=100, p=60, spread=1e-2):
    """Columns that share three factors and differ by noise of `spread` (cond(Z) about 3.6e3), scales 1 to 100, means off 0,
    a target of five of them plus noise."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n + 1, 3))
    x = f @ rng.normal(size=(3, p)) + spread * rng.normal(size=(n + 1, p))
    x = x * 10.0 ** rng.uniform(0, 2, p) + rng.normal(0, 50, p)
    y = 50 + x[:, :5] @ rng.normal(size=5) + rng.normal(0, 20, n + 1)
    return x[:n], y[:n]


def market_pair(seed=151, n=28, ratio=3e-3):
    """A small market-like design (load, wind, a weekend one-hot pair, a lag, noise; spiky prices) with a near-copy of the load
    `ratio` of its sd apart, whose difference the target reads: the entries' own rounding then reaches the forecast of a row
    out along that difference."""
    rng = np.random.default_rng(seed)
    load, wind = rng.normal(1000, 200, n), rng.gamma(2.0, 50.0, n)
    weekend = (rng.random(n) < 2 / 7).astype(float)
    y = 20 + 0.08 * load - 0.15 * wind + 10 * weekend + rng.normal(0, 5, n)
    spikes = rng.random(n) < 0.08
    y[spikes] *= rng.uniform(3, 10, spikes.sum())
    lag = np.roll(y, 1) + rng.normal(0, 3, n)
    x = np.column_stack([load, wind, weekend, 1 - weekend, lag, rng.normal(0, 1, n)])
    sd = ratio * load.std(ddof=1)
    e = rng.normal(0, sd, n)
    y = y + 3.0 * e / sd
    mask = np.ones(x.shape[1] + 1, dtype=bool)
    mask[2:4] = False
    return np.column_stack([x, load + e]), y, mask, sd


def engine_fit(x, y, mask):
    raw = SpanSums(x, y).over(np.ones(len(y), dtype=bool), np.arange(x.shape[1]))
    stats, params = _stats.standard_stats(raw, (x, y), mask, False)
    return stats, params, _stats.ols_fit(stats, False)


def unrefined(stats, params, row):
    """The forecast of the normal equations' first solution, before the step (engine-1's)."""
    M, b, _ = stats.uncentred()
    w = _stats.cholesky_solve(M, b)
    return params.inverse(float(params.transform(row)[0] @ w))


def model_path(x, y, mask, rows):
    scaler = StandardScaler()
    tx, ty, tt = scaler.fit_transform(x.copy(), y.copy(), rows.copy(), scalable_mask=mask)
    fitted = LinearRegression(fit_intercept=False).fit(tx, ty)
    return [scaler.inverse(float(fitted.predict(tt[i : i + 1])[0])) for i in range(len(tt))]


def far_along_weakest(x, metric, k):
    return (x.mean(axis=0) + x.std(axis=0, ddof=1) * (k * metric.vectors[:, 0] / metric.scale[0]))[None, :]


def test_the_refinement_removes_the_factorisations_error_at_a_row_20_sds_out():
    """Where the first solution's error is the factorisation's (columns of three factors and 3 % noise each, a row 20 training
    sds out along the fit's weakest direction): 3.7e-9 from the model path before the step, 6e-12 after, kept by the guard."""
    x, y = ill_conditioned(seed=31, n=120, p=80, spread=3e-2)
    mask = np.ones(x.shape[1], dtype=bool)
    stats, params, fitted = engine_fit(x, y, mask)
    far = far_along_weakest(x, _stats.issue_metric(stats, fitted), 20.0)
    z = params.transform(far)
    (off,) = model_path(x, y, mask, far)
    assert abs(unrefined(stats, params, far) - off) > 3e-9
    assert abs(params.inverse(float(fitted.predict(z)[0])) - off) <= 2e-11
    assert _stats.solve_error_ok(fitted, stats, z)
    out, counts, _ = engine_on_arrays(OLSModel([], 120), x, y, mask, far)
    assert not counts.fallbacks and abs(out[0][0] - off) <= 2e-11


def test_an_ill_conditioned_fit_keeps_a_row_10_sds_out_and_hands_back_one_20_out():
    """Columns of three factors and 1 % noise (cond(Z) about 3.6e3): at 20 training sds out along the weakest direction (inside
    OLS's bound of 30) the first solution was 1.4e-8 from the model path (engine-1's kind of miss) and the refined one is
    still 5.8e-9, the rest the rounding of the equations' entries; the guard's estimate (9e-9) sends the row back. At 10 sds
    the refined forecast is within 3e-9 and kept."""
    x, y = ill_conditioned()
    mask = np.ones(x.shape[1], dtype=bool)
    stats, params, fitted = engine_fit(x, y, mask)
    metric = _stats.issue_metric(stats, fitted)
    far, mid = far_along_weakest(x, metric, 20.0), far_along_weakest(x, metric, 10.0)
    z_far, z_mid = params.transform(far), params.transform(mid)
    assert 19.9 < _stats.issue_distance(metric, z_far) < 30
    off_far, off_mid = model_path(x, y, mask, np.vstack([far, mid]))
    first = abs(unrefined(stats, params, far) - off_far)
    gap = abs(params.inverse(float(fitted.predict(z_far)[0])) - off_far)
    assert first > 1e-8 and _stats.OLS_SOLVE_ERROR_MAX < gap < first / 2
    assert _stats.solve_error(fitted, stats, z_far) * stats.yscale > gap
    assert not _stats.solve_error_ok(fitted, stats, z_far) and _stats.solve_error_ok(fitted, stats, z_mid)
    assert abs(params.inverse(float(fitted.predict(z_mid)[0])) - off_mid) <= 3e-9
    out, counts, _ = engine_on_arrays(OLSModel([], 100), x, y, mask, np.vstack([mid, far]))
    assert out[1] is None and counts.fallbacks["far_issue"] == 1, counts.fallbacks
    assert abs(out[0][0] - off_mid) <= 3e-9


def test_the_guard_sends_back_a_row_whose_entries_rounding_the_refinement_cant_remove():
    """A near-copy pair 3e-3 of the load's sd apart and a row 20 of the difference's sds out along it (18 in the fit's metric,
    inside the bound): the entries' own rounding puts even the refined forecast 1.8e-8 from the model path's, the guard's
    estimate sees it, and the row goes back; a training row stays, within 1e-9."""
    x, y, mask, sd = market_pair()
    stats, params, fitted = engine_fit(x, y, mask)
    far = x[len(y) // 2].copy()
    far[-1] = far[0] + (x[:, -1] - x[:, 0]).mean() + 20 * sd
    far = far[None, :]
    z = params.transform(far)
    assert _stats.issue_distance(_stats.issue_metric(stats, fitted), z) < 30
    (off,) = model_path(x, y, mask, far)
    gap = abs(params.inverse(float(fitted.predict(z)[0])) - off)
    estimate = _stats.solve_error(fitted, stats, z) * stats.yscale
    assert gap > 1e-8 and estimate > max(gap, _stats.OLS_SOLVE_ERROR_MAX)
    assert not _stats.solve_error_ok(fitted, stats, z)
    near = x[1:2]
    out, counts, _ = engine_on_arrays(OLSModel([], 28), x, y, mask, np.vstack([near, far]))
    assert out[1] is None and counts.fallbacks["far_issue"] == 1, counts.fallbacks
    assert abs(out[0][0] - model_path(x, y, mask, near)[0]) <= 1e-9


def gappy_span(seed, level, n=120, lead=50, ratio=3e-3):
    """The review's design: `market_pair`'s columns over a span of n rows whose first `lead` rows lack one column (a feed that
    starts inside the window: NaN there, so the fit's rows ρ are the rest) and sit `level` higher in load. The span's shifts
    come from all its rows, so the centring of the fit's sums cancels: κ = S/(S - u²/r) about 20 at a level of 1,600."""
    rng = np.random.default_rng(seed)
    load = rng.normal(1000, 200, n)
    load[:lead] += level
    wind = rng.gamma(2.0, 50.0, n)
    weekend = (rng.random(n) < 2 / 7).astype(float)
    y = 20 + 0.08 * load - 0.15 * wind + 10 * weekend + rng.normal(0, 5, n)
    lag = np.roll(y, 1) + rng.normal(0, 3, n)
    sd = ratio * 200
    e = rng.normal(0, sd, n)
    y = y + 3.0 * e / sd
    extra = rng.normal(0, 1, n)
    extra[:lead] = np.nan
    x = np.column_stack([load, wind, weekend, 1 - weekend, lag, extra, load + e])
    mask = np.ones(x.shape[1], dtype=bool)
    mask[2:4] = False
    return x, y, mask, sd


def engine_on_span(model, x, y, mask, tests):
    """`EngineFits` on one span of rows `x`, `y` with gaps: ρ the rows whose every column is finite, the span's sums over all
    rows (so their shifts come from rows the fit doesn't read), as an engine group builds them."""
    n, p = x.shape
    arrays = {
        0: {
            "x": np.vstack([x, tests]),
            "y": np.r_[y, np.full(len(tests), np.nan)][:, None],
            "days": np.arange(n + len(tests)),
            "col_index": {f"c{j}": j for j in range(p)},
        }
    }
    inputs = _engine_worker.ModelInputs("m", TaskInputs(arrays, {}, {"scaler": model.scaler}, model), False)
    counts = _engine_worker.EngineCounts()
    fits = _engine_worker.EngineFits(
        _engine_worker._Spans(arrays, {(0, 1, 0, n): np.arange(p)}, Counter()), inputs, counts
    )
    finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
    out = fits(StatsFit(0, 1, slice(0, n), list(range(p)), finite, x[finite], y[finite], mask, tests))
    return out, counts


@pytest.mark.parametrize("level, seed", [(800.0, 39), (1600.0, 42)])
def test_a_fit_whose_rows_sit_away_from_its_spans_mean_counts_the_centrings_rounding(level, seed):
    """The review's case: a span whose first 50 of 120 rows lack a column and sit higher in load, a near-copy pair 3e-3 sd apart
    and an issue row 20 of the difference's sds out along it. The sums over ρ are about the span's shifts, so G = (S - uu'/r)
    / (σσ') cancels (κ 4.8 and 21), and its entries' rounding is eps (|S| + |uu'|/r) / (σσ'), not eps |G|: an estimate from |M|
    kept the row although even the refined forecast is 7.6e-9 and 4.7e-8 off. With the entries' scale from the sums
    (`ScaledStats.entries`) the row goes back."""
    x, y, mask, sd = gappy_span(seed, level)
    finite = np.isfinite(x).all(axis=1)
    xr, yr = x[finite], y[finite]
    far = xr[len(xr) // 2].copy()
    far[-1] = far[0] + (xr[:, -1] - xr[:, 0]).mean() + 20 * sd
    far = far[None, :]
    raw = SpanSums(x, y).over(finite, np.arange(x.shape[1]))
    kappa = np.diag(raw.S) / (np.diag(raw.S) - raw.u**2 / raw.n)
    assert kappa.max() > 4  # 4.8 at a level of 800, 21 at 1,600
    stats, params = _stats.standard_stats(raw, (xr, yr), mask, False)
    fitted = _stats.ols_fit(stats, False)
    z = params.transform(far)
    (off,) = model_path(xr, yr, mask, far)
    assert abs(params.inverse(float(fitted.predict(z)[0])) - off) > _stats.OLS_SOLVE_ERROR_MAX  # 7.6e-9 and 4.7e-8
    plain = stats._replace(entries=None)  # the scale of |M| and |b|, as before
    assert _stats.solve_error_ok(_stats.ols_fit(plain, False), plain, z)
    assert not _stats.solve_error_ok(fitted, stats, z)
    out, counts = engine_on_span(OLSModel([], 120), x, y, mask, far)
    assert out is None and counts.fallbacks["far_issue"] == 1, counts.fallbacks


def test_the_entries_scale_is_the_gram_itself_where_nothing_cancels():
    """Where the fit's rows are the whole span (κ = 1) the entries' scale is |G| and |q| up to rounding, so the guard reads what
    it read before the scale was carried."""
    x, y = ill_conditioned()
    stats, _, _ = engine_fit(x, y, np.ones(x.shape[1], dtype=bool))
    E, Eq = stats.entries
    assert np.allclose(E, np.abs(stats.G), rtol=1e-9, atol=1e-9 * np.abs(stats.G).max())
    assert np.allclose(Eq, np.abs(stats.q), rtol=1e-9, atol=1e-9 * np.abs(stats.q).max())


def lago_rows(hour, end, n=364):
    """The DE fixture's Lago-shaped rows (`lear_design`: 247 columns, prices of t-1, t-2, t-3, t-7, both exogenous series on t,
    t-1, t-7, the weekday dummies of t, left unscaled), the n days before `end` and day `end` as the issue row."""
    X, Y = lear_design(hour)
    mask = np.ones(X.shape[1], dtype=bool)
    mask[-7:] = False
    return X[end - n : end], Y[end - n : end], X[end : end + 1], mask


# Of the DE fixture's 25,872 (hour, day) fits at n 364 whose issue row the 30-sd bound keeps, the two that engine-1 forecast
# over 5e-9 from the model path (half OLS's tolerance) and one ordinary row (`ols_issue_guard/fixture_scan.py`).
@pytest.mark.parametrize("hour, end", [(14, 1245), (15, 1447), (0, 1245)])
def test_lago_shaped_de_rows_like_the_experiment_checks_are_within_the_budget(hour, end):
    """As the experiment-check's miss (Lago's DE inputs, W 364, no intercept, the weekday dummies raw, an issue row 26 sds
    out): the refined forecast is within 2e-9 of the model path's (engine-1's were up to 5.4e-9) and the guard keeps it."""
    x, y, row, mask = lago_rows(hour, end)
    stats, params, fitted = engine_fit(x, y, mask)
    z = params.transform(row)
    assert _stats.issue_distance(_stats.issue_metric(stats, fitted), z) <= 30
    (off,) = model_path(x, y, mask, row)
    assert abs(params.inverse(float(fitted.predict(z)[0])) - off) <= 2e-9
    assert _stats.solve_error_ok(fitted, stats, z)
    out, counts, _ = engine_on_arrays(OLSModel([], 364), x, y, mask, row)
    assert not counts.fallbacks and abs(out[0][0] - off) <= 2e-9


def test_the_error_free_residual_is_exact_to_the_residuals_last_bits():
    """`_residual_exact` against exact rational arithmetic, on normal equations of a near-collinear pair, where b - M w cancels
    to about 1e-13 of its terms: within 4 eps of the residual itself (a float64 residual would lose it all)."""
    rng = np.random.default_rng(3)
    A = rng.normal(size=(60, 8))
    A[:, 1] = A[:, 0] + 1e-4 * A[:, 1]
    M, b = A.T @ A, A.T @ rng.normal(size=60)
    w = np.linalg.solve(M, b)
    exact = np.array(
        [float(Fraction(b[i]) - sum(Fraction(M[i, j]) * Fraction(w[j]) for j in range(8))) for i in range(8)]
    )
    got = _stats._residual_exact(M, b, w)
    assert np.all(np.abs(got - exact) <= 4 * _stats.EPS * np.abs(exact))
    assert np.abs(exact).max() < 1e-10 * np.abs(M).max() * np.abs(w).max()  # what a float64 residual would lose


def test_the_guard_reads_a_vst_forecast_in_target_units_where_the_inverse_is_steeper():
    """With a VST the estimate is in target units through the inverse's slope at the forecast where that is steeper than at
    the training targets (`_engine_worker._steep_ok`), as Ridge's budget is: a budget the training-target slope meets, a
    steeper one doesn't."""
    x, y = ill_conditioned()
    mask = np.ones(x.shape[1], dtype=bool)
    stats, params, fitted = engine_fit(x, y, mask)
    z = params.transform(far_along_weakest(x, _stats.issue_metric(stats, fitted), 5.0))
    budget = 2.0 * _stats.solve_error(fitted, stats, z) * stats.yscale
    assert _stats.solve_error_ok(fitted, stats, z, error_max=budget)
    assert not _stats.solve_error_ok(fitted, stats, z, scale=3.0 * stats.yscale, error_max=budget)


def test_only_ols_fits_carry_the_solve_error():
    """Ridge's fits carry its own rounding measure, not the normal equations' error: the guard leaves them alone."""
    x, y = ill_conditioned(n=150, spread=1.0)  # well conditioned, so Ridge fits it from statistics
    mask = np.ones(x.shape[1], dtype=bool)
    _, _, ridge = engine_on_arrays(RidgeModel([], 150), x, y, mask, x[:1])
    assert ridge is not None and getattr(ridge, "solve", None) is None
    _, _, fitted = engine_fit(x, y, mask)
    assert isinstance(fitted.solve, _stats.SolveError) and fitted.solve.correction.shape == (x.shape[1],)


def test_results_of_engine_1_refuse_a_resume(tmp_path, monkeypatch):
    """The refinement moves OLS's numbers within the tolerance, so its fit path is "engine-2": results written by engine-1
    refuse a resume (`fit_path` compared), and `allow_manifest_mismatch=True` resumes them anyway."""
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    assert _engine_worker.ENGINE_FIT_PATH == "engine-2"
    data = make_market_data(start="2023-01-01", days=50, noise=5.0, seed=3, horizon=1)

    def run(**kwargs):
        pipeline = ModelPipeline(engine="on").add_model(OLSModel(["load_d+1", "wind_d+1"], 28, name="ols"))
        return pipeline.run(data, "2023-02-14", "2023-02-15", horizon=1, save_dir=str(tmp_path), **kwargs)

    run()
    path = tmp_path / "ols.manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["fit_path"] == "engine-2"
    path.write_text(json.dumps({**manifest, "fit_path": "engine-1"}))
    with pytest.raises(ManifestMismatchError, match="fit_path"):
        run()
    run(allow_manifest_mismatch=True)
    assert json.loads(path.read_text())["fit_path"] == "engine-2"
