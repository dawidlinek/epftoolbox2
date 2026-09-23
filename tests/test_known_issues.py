"""Known methodology issues, pinned with xfail(strict=True).

Each test asserts the *correct* behaviour and currently fails. When an issue is fixed its test starts
passing, strict mode turns that into a failure, and the marker must be removed (move the test next to
the code it covers). Details: .claude/rules/{models,data,evaluation}.md.
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import LagTransformer
from epftoolbox2.evaluators import rMAEEvaluator
from epftoolbox2.models import OLSModel
from epftoolbox2.results.report import EvaluationReport
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="LagTransformer shifts by elapsed time, not local calendar days"
)
def test_lag_transformer_day_lag_keeps_wall_clock_hour_across_dst():
    index = pd.date_range("2024-03-29 22:00", "2024-04-01 00:00", freq="1h", tz="UTC").tz_convert("Europe/Warsaw")
    df = pd.DataFrame({"price": np.arange(len(index), dtype=float)}, index=index)

    out = LagTransformer(columns=["price"], lags=[1], freq="day").transform(df)

    on_switch_day = pd.Timestamp("2024-03-31 10:00", tz="Europe/Warsaw")
    day_before = pd.Timestamp("2024-03-30 10:00", tz="Europe/Warsaw")
    assert out.loc[on_switch_day, "price_d-1"] == df.loc[day_before, "price"]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="rMAE divides MAEs computed on different forecast sets")
def test_rmae_compares_models_on_shared_forecast_keys():
    def row(run_date, hour, prediction, actual):
        target = (pd.Timestamp(run_date) + pd.Timedelta(days=1)).date().isoformat()
        return {"run_date": run_date, "target_date": target, "hour": hour, "horizon": 1, "day_in_test": 0,
                "prediction": prediction, "actual": actual}  # fmt: skip

    benchmark = [row("2024-01-01", 0, 11.0, 10.0), row("2024-01-01", 1, 19.0, 10.0)]  # errors 1 and 9
    candidate = [row("2024-01-01", 0, 12.0, 10.0)]  # error 2; the hour-1 forecast is missing

    report = EvaluationReport({"base": benchmark, "model": candidate}, [rMAEEvaluator(base_model="base")])
    summary = report.summary().set_index("model")

    # On the shared key the candidate's error is 2 vs the benchmark's 1.
    assert summary.loc["model", "rMAE"] == pytest.approx(2.0)


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="duplicated hour after fall-back DST shifts the training cut by one row"
)
@pytest.mark.parametrize("run_day", ["2023-10-30", "2023-10-31"])
def test_no_lookahead_in_days_after_fall_back_dst(run_day):
    df = make_market_data(start="2023-08-01", days=120, noise=5.0, seed=1)
    predictors = PREDICTORS + ["price_d-1"]
    scrambled = df.copy()
    future = pd.Series(df.index.date >= pd.Timestamp(run_day).date(), index=df.index)
    scrambled.loc[future, "price"] = 1e4 + np.arange(future.sum())
    scrambled["price_d-1"] = calendar_shift(scrambled["price"], -1)

    def predictions(data):
        ref = OLSModel(predictors, training_window=40).run(data, test_start=run_day, test_end=run_day, horizon=2)
        return {(r["hour"], r["horizon"]): r["prediction"] for r in ref._results}

    original, perturbed = predictions(df), predictions(scrambled)
    changed = [key for key in original if abs(original[key] - perturbed[key]) > 1e-9]
    assert changed == []


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="on the spring DST day the missing hour uses the previous day's row"
)
def test_spring_dst_day_forecasts_carry_their_own_target():
    df = make_market_data(noise=5.0, seed=1)
    ref = OLSModel(PREDICTORS, training_window=40).run(df, test_start="2023-03-26", test_end="2023-03-26", horizon=2)

    produced = {(r["hour"], r["horizon"]) for r in ref._results}
    mislabelled = [
        (r["hour"], r["horizon"])
        for r in ref._results
        if r["actual"]
        != pytest.approx(df.loc[pd.Timestamp(f"{r['target_date']} {r['hour']:02d}:00", tz=df.index.tz), "price"])
    ]
    assert mislabelled == []
    assert len(produced) == ref.count
