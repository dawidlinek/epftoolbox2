"""`LEARModel.lago()` against the original epftoolbox of Lago, Marcjasz, De Schutter & Weron (2021), on their data.

`tests/fixtures/lago/` holds what the original `LEAR.recalibrate_and_forecast_next_day` forecast under scikit-learn 0.24.2 for the
first target day of the DE and BE test periods (issue days 2016-01-03 and 2015-01-03), each of its four calibration windows and
24 hours, its unscaled test row, and the data it read (`build.py` made them; it isn't run in CI, see its docstring). Here the
library builds the frame from that data by the documented chain (`fixtures/lago/frames.py`), and:

- its test row is the original's to the bit, column for column, which holds the order of the 247 inputs too;
- `lago()` forecasts every hour of each window within 1e-6 EUR/MWh of the original (measured: within 1e-9), one test per market and
  window, so that xdist spreads the numpy kernels' coordinate descent;
- `solver="lars"` on the same rows is scikit-learn's `Lasso(α*, tol=1e-12)` at old `LassoLarsIC`'s α* within 1e-6;
- the four windows' average is `fsum` of the single-window models' forecasts over 4, to the bit, and the numba kernels agree with
  the numpy ones with the same kink;
- `criterion="cv"` is old `LassoLarsCV(cv=KFold(5), normalize=True, max_iter=2500)` under 0.24.2, which `build.py --cv` ran on
  the original's scaled rows (`cv_{M}.csv`, `cv_coefs_{M}.parquet`): α* within 1e-9 (relative), every coefficient within 1e-9,
  the forecast within 1e-6 EUR/MWh, and α* with `max_n_alphas=50`, whose grid is strided, within 1e-9.
"""

import functools
import math
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from epftoolbox2._extras import has_extra
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import LEARModel
from epftoolbox2.models import _lars
from epftoolbox2.models import _worker as worker
from epftoolbox2.models._lars import LearFit
from epftoolbox2.models.lear import LAGO_WINDOWS
from tests.fixtures.lago.frames import EXOGENOUS, FIXTURE_ISSUE_DAYS, HERE, FixtureSource, lear_frame
from tests.models.test_lars_kernels import _lars_old_ic
from tests.models.test_lear import InProcessPool

MARKETS = ("DE", "BE")
CALIBRATION = dict(zip(LAGO_WINDOWS, (56, 84, 1092, 1456)))  # training rows -> Lago's calibration window in days
CASES = [(market, window) for market in MARKETS for window in LAGO_WINDOWS]
INFO = InformationSet(
    "11:45", "Europe/Brussels", rules={"price": Availability.published("13:00", 1, timezone="Etc/GMT-1")}
)
TOLERANCE = 1e-6  # EUR/MWh


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)


@functools.lru_cache(maxsize=None)
def frame(market):
    raw = pd.read_parquet(HERE / f"data_{market}.parquet")
    return lear_frame(FixtureSource(market), str(raw.index[0].date()), str(raw.index[-1].date()))


@functools.lru_cache(maxsize=None)
def original(market):
    """(window, hour) -> the original's forecast, and the original's test row."""
    forecasts = pd.read_csv(HERE / f"original_{market}.csv")
    assert set(forecasts["issue_day"]) == {FIXTURE_ISSUE_DAYS[market]}
    by_key = {(int(r.window), int(r.hour)): float(r.forecast) for r in forecasts.itertuples()}
    xtest = pd.read_csv(HERE / f"xtest_{market}.csv", index_col="issue_day")
    return by_key, xtest.loc[FIXTURE_ISSUE_DAYS[market]].to_numpy(dtype=float)


def forecasts(model, market):
    """hour -> the record of `model`'s forecast issued on the fixture's day, h = 1."""
    day = FIXTURE_ISSUE_DAYS[market]
    records = model.run(frame(market), day, day, horizon=1, cut_rules=INFO)._results
    assert len(records) == 24
    return {r["hour"]: r for r in records}


def lago(market, window=None, **settings):
    model = LEARModel.lago(EXOGENOUS[market])
    if window is not None:
        model.training_window = window
    for key, value in settings.items():
        setattr(model, key, value)
    return model


@functools.lru_cache(maxsize=None)
def single(market, window):
    return forecasts(lago(market, window), market)


@pytest.mark.parametrize("market", MARKETS)
def test_the_design_is_the_original_s_test_row_column_for_column(market):
    names = [name.replace("{horizon}", "1") for name in lago(market).predictors]
    _, xtest = original(market)
    rows = frame(market).loc[FIXTURE_ISSUE_DAYS[market], names].to_numpy(dtype=float)
    assert rows.shape == (24, 247)
    assert (rows == xtest).all()  # every period's row of D: the same 247 values, in the original's order


@pytest.mark.parametrize("market, window", CASES, ids=[f"{m}-{w}" for m, w in CASES])
def test_lago_forecasts_are_the_original_s(market, window):
    expected, _ = original(market)
    found = single(market, window)
    differences = [abs(found[hour]["prediction"] - expected[(CALIBRATION[window], hour)]) for hour in range(24)]
    assert max(differences) <= TOLERANCE, max(differences)


class Reference(LEARModel):
    """The fit of hours 0, 8 and 16 by scikit-learn alone: old LassoLarsIC's α* on the normalised path (the in-test copy, X form
    where n <= p as old LassoLarsIC chose), then `Lasso(α*, tol=1e-12)` on the rows as they are. Every other hour, and the kernels'
    own α* for the comparison, by the kernels."""

    hour = None
    alphas = []

    def _fit_model(self, train_x, train_y):
        fit = super()._fit_model(train_x, train_y)
        if Reference.hour % 8:
            return fit
        alpha, _, _, _ = _lars_old_ic(train_x, train_y, self.criterion)
        Reference.alphas.append((fit.alpha, alpha))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            lasso = Lasso(alpha=alpha, tol=1e-12, max_iter=10**6, precompute=True).fit(train_x, train_y)
        xbar = train_x.mean(axis=0)
        return LearFit(lasso.coef_, xbar, float(lasso.intercept_ + xbar @ lasso.coef_), alpha, fit.kink, True)


@pytest.mark.parametrize("market, window", CASES, ids=[f"{m}-{w}" for m, w in CASES])
def test_the_exact_refit_is_the_lasso_at_old_lasso_lars_ic_s_alpha(market, window, monkeypatch):
    original_fit = worker._fit_member

    def noting_the_hour(arrays, cfg, hour, *args):
        Reference.hour = hour
        return original_fit(arrays, cfg, hour, *args)

    monkeypatch.setattr(worker, "_fit_member", noting_the_hour)
    monkeypatch.setattr(Reference, "alphas", [])
    exact = forecasts(lago(market, window, solver="lars"), market)
    reference_model = Reference(lago(market).predictors, window, "aic", "lars", scaler=lago(market).scaler)
    reference_model.newest_target = "published"
    reference = forecasts(reference_model, market)
    assert len(Reference.alphas) == 3
    for kernel_alpha, sklearn_alpha in Reference.alphas:
        assert kernel_alpha == pytest.approx(sklearn_alpha, rel=1e-9)
    for hour in range(0, 24, 8):
        assert exact[hour]["prediction"] == pytest.approx(reference[hour]["prediction"], abs=TOLERANCE), hour


@pytest.mark.skipif(
    _lars._kernels().BACKEND == "numpy", reason="96 fits by the numpy kernels' coordinate descent take minutes"
)
@pytest.mark.parametrize("market", MARKETS)
def test_the_window_average_is_fsum_of_the_four_single_window_models_over_four(market):
    listed = forecasts(lago(market), market)
    expected, _ = original(market)
    for hour, record in listed.items():
        members = [single(market, window)[hour]["prediction"] for window in LAGO_WINDOWS]
        assert [m.hex() for m in record["member_predictions"]] == [m.hex() for m in members]
        assert record["prediction"].hex() == (math.fsum(members) / 4).hex()
        for window, member in zip(LAGO_WINDOWS, members):
            assert member == pytest.approx(expected[(CALIBRATION[window], hour)], abs=TOLERANCE)


@pytest.mark.skipif(not has_extra("fast"), reason="needs the [fast] extra (numba)")
@pytest.mark.parametrize("market", MARKETS)
def test_the_backends_agree_on_the_fixture_with_the_same_kink(market, monkeypatch):
    """The numba kernels' fits of every window at hours 0, 8 and 16 against the numpy kernels': the same kink, and forecasts in
    the scaled space within 1e-9 (in EUR/MWh the asinh's inverse scales that by at most about the price's MAD)."""
    monkeypatch.setenv("EPFTOOLBOX2_KERNELS", "numba")
    seen = []
    original_fit = worker._fit_member

    def keeping(arrays, cfg, hour, *args):
        if hour % 8 == 0:
            seen.append(hour)
        return original_fit(arrays, cfg, hour, *args)

    stats = []
    kept = LEARModel._fit_predict

    def keep_stats(self, train_x, train_y, test_x):
        if seen and len(stats) < len(seen):
            stats.append((_lars.gram_stats(train_x, train_y), test_x.copy(), self.criterion, self.solver))
        return kept(self, train_x, train_y, test_x)

    monkeypatch.setattr(worker, "_fit_member", keeping)
    monkeypatch.setattr(LEARModel, "_fit_predict", keep_stats)
    forecasts(lago(market), market)
    assert len(stats) == 3 * 4
    numba = _lars._numba_kernels()
    for gram, row, criterion, solver in stats:
        a, b = _lars.lear_fit(gram, criterion, solver), numba.lear_fit(gram, criterion, solver)
        assert (a.kink, a.converged) == (b.kink, b.converged)
        assert a.alpha == pytest.approx(b.alpha, rel=1e-12)
        np.testing.assert_allclose(a.w, b.w, rtol=0, atol=1e-9)
        pa = LEARModel([])._predict_fitted(a, row)[0]
        pb = LEARModel([])._predict_fitted(b, row)[0]
        assert pa == pytest.approx(pb, abs=1e-9)


def test_the_fixture_is_what_its_sources_file_says(tmp_path):
    """The files the tests read are the ones `build.py` wrote (their sha256 in SOURCES.json), from the original at a commit on
    or after f234b08, under scikit-learn 0.24.2."""
    import hashlib
    import json

    sources = json.loads((HERE / "SOURCES.json").read_text())
    assert sources["versions"]["scikit-learn"] == "0.24.2" and sources["versions"]["numpy"].startswith("1.")
    assert len(sources["epftoolbox_commit"]) == 40
    for name, digest in sources["outputs"].items():
        assert hashlib.sha256(Path(HERE / name).read_bytes()).hexdigest() == digest, name


# ---------------------------------------------------------------------------
# criterion="cv": old LassoLarsCV under 0.24.2
# ---------------------------------------------------------------------------

CV_CAP = 50  # build.py's: `alpha_cap50` is LassoLarsCV's alpha_ with max_n_alphas=50
CV_TOLERANCE = 1e-9  # α* (relative) and the coefficients (absolute, on the scaled inputs)


@functools.lru_cache(maxsize=None)
def cv_reference(market):
    """(window, hour) -> (forecast, alpha_, alpha_ with max_n_alphas=50, len(cv_alphas_)), and (window, hour) -> coef_."""
    rows = pd.read_csv(HERE / f"cv_{market}.csv")
    assert set(rows["issue_day"]) == {FIXTURE_ISSUE_DAYS[market]}
    coefs = pd.read_parquet(HERE / f"cv_coefs_{market}.parquet")
    found = {
        (int(r.window), int(r.hour)): (
            float(r.forecast),
            float(r.alpha),
            float(getattr(r, f"alpha_cap{CV_CAP}")),
            int(r.grid),
        )
        for r in rows.itertuples()
    }
    dense = {key: np.zeros(247) for key in found}
    for c in coefs.itertuples():
        dense[(int(c.window), int(c.hour))][int(c.column)] = float(c.coef)
    return found, dense


class CvCapture(LEARModel):
    """`criterion="cv"` keeping each hour's blocks' sums and fit."""

    hour = None
    fits = {}

    def _fit_stats(self, stats):
        fit = super()._fit_stats(stats)
        CvCapture.fits[CvCapture.hour] = (stats, fit)
        return fit


@functools.lru_cache(maxsize=None)
def cv_single(market, window):
    """hour -> the record, and hour -> (CvStats, LearFit), of `lago()` with `criterion="cv"` (and `solver="lars"`, which it
    takes) at `window`."""
    original_fit = worker._fit_member

    def noting_the_hour(arrays, cfg, hour, *args):
        CvCapture.hour = hour
        return original_fit(arrays, cfg, hour, *args)

    base = lago(market)
    model = CvCapture(base.predictors, window, "cv", "lars", scaler=base.scaler, newest_target="published")
    CvCapture.fits = {}
    with mock.patch.object(worker, "_fit_member", noting_the_hour):
        records = forecasts(model, market)
    fits = dict(CvCapture.fits)
    assert sorted(fits) == list(range(24))
    return records, fits


@pytest.mark.parametrize("market, window", CASES, ids=[f"{m}-{w}" for m, w in CASES])
def test_cv_is_old_lasso_lars_cv(market, window):
    """Every hour: α*, the coefficients and the forecast. At hours 0, 8 and 16 also α* with `max_n_alphas=50`, from the run's
    blocks' sums: every fixture fit's grid is strided then (0.24.2 counted 394 to 1998 penalties). At the default 1,000, 34 of
    the 48 DE fits of W 1085 and 1449 are strided too (more than 2000 kinks), no BE fit."""
    reference, coefs = cv_reference(market)
    records, fits = cv_single(market, window)
    for hour in range(24):
        forecast, alpha, capped, grid = reference[(CALIBRATION[window], hour)]
        stats, fit = fits[hour]
        assert fit.alpha == pytest.approx(alpha, rel=CV_TOLERANCE), hour
        np.testing.assert_allclose(
            records[hour]["coefficients"], coefs[(CALIBRATION[window], hour)], rtol=0, atol=CV_TOLERANCE
        )
        assert records[hour]["prediction"] == pytest.approx(forecast, abs=TOLERANCE), hour
        if hour % 8 == 0:
            assert grid >= 2 * CV_CAP
            assert _lars._kernels().lear_fit_cv(stats, CV_CAP).alpha == pytest.approx(capped, rel=CV_TOLERANCE), hour


@pytest.mark.skipif(not has_extra("fast"), reason="needs the [fast] extra (numba)")
@pytest.mark.parametrize("market, window", CASES, ids=[f"{m}-{w}" for m, w in CASES])
def test_the_backends_agree_on_cv_on_the_fixture(market, window):
    numba = _lars._numba_kernels()
    _, fits = cv_single(market, window)
    for hour in (0, 8, 16):
        stats, _ = fits[hour]
        a, b = _lars.lear_fit_cv(stats), numba.lear_fit_cv(stats)
        assert (a.kink, a.converged) == (b.kink, b.converged), hour
        assert a.alpha == pytest.approx(b.alpha, rel=1e-12)
        np.testing.assert_allclose(a.w, b.w, rtol=0, atol=1e-9)
