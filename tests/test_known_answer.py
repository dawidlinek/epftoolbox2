"""Known-answer and information-set tests on synthetic data (see tests/synthetic.py).

These check numbers, not shapes: an exactly linear price must be reproduced, and a forecast issued on
day D must not depend on prices dated D or later. Run days avoid DST switches; DST behaviour is
pinned separately in tests/test_known_issues.py.
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.models import LassoCVModel, OLSModel
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


def forecasts(model, df, test_start, test_end, horizon=2):
    ref = model.run(df, test_start=test_start, test_end=test_end, target="price", horizon=horizon)
    return ref, {(r["run_date"], r["hour"], r["horizon"]): r for r in ref._results}


def test_ols_reproduces_an_exactly_linear_price():
    df = make_market_data(noise=0.0)
    ref, results = forecasts(OLSModel(PREDICTORS, training_window=60), df, "2023-04-10", "2023-04-16")

    assert len(results) == ref.count == 7 * 24 * 2
    errors = np.array([r["prediction"] - r["actual"] for r in results.values()])
    assert np.abs(errors).max() < 1e-6


def test_actual_is_the_price_at_the_labelled_target():
    df = make_market_data(noise=5.0, seed=3)
    _, results = forecasts(OLSModel(PREDICTORS, training_window=60), df, "2023-04-10", "2023-04-12")

    for r in results.values():
        expected = df.loc[pd.Timestamp(f"{r['target_date']} {r['hour']:02d}:00", tz=df.index.tz), "price"]
        assert r["actual"] == pytest.approx(expected), r


@pytest.mark.parametrize(
    "make_model, run_day",
    [
        (lambda p: OLSModel(p, training_window=45), "2023-04-12"),
        (lambda p: OLSModel(p, training_window=45), "2023-05-02"),
        (lambda p: LassoCVModel(p, training_window=45, cv=3), "2023-04-12"),
    ],
    ids=["ols-apr", "ols-may", "lasso-apr"],
)
def test_forecast_ignores_prices_dated_on_or_after_the_issue_day(make_model, run_day):
    """Current information set: training targets end at D-1, so prices from D onward must not matter.

    If this fails after an intentional change of the boundary (e.g. using day D's auction result),
    update the test together with the documentation of the boundary in .claude/rules/models.md.
    """
    df = make_market_data(noise=5.0, seed=7)
    predictors = PREDICTORS + ["price_d-1"]

    scrambled = df.copy()
    future = pd.Series(df.index.date >= pd.Timestamp(run_day).date(), index=df.index)
    scrambled.loc[future, "price"] = 1e4 + np.arange(future.sum())
    scrambled["price_d-1"] = calendar_shift(scrambled["price"], -1)  # as a real pipeline would recompute it

    _, original = forecasts(make_model(predictors), df, run_day, run_day)
    _, perturbed = forecasts(make_model(predictors), scrambled, run_day, run_day)

    assert original.keys() == perturbed.keys()
    assert len(original) == 24 * 2
    for key, r in original.items():
        assert perturbed[key]["prediction"] == pytest.approx(r["prediction"], abs=1e-9), key
