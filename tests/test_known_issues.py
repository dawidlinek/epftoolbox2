"""Known methodology issues, pinned with xfail(strict=True).

Each test asserts the *correct* behaviour and currently fails. When an issue is fixed its test starts
passing, strict mode turns that into a failure, and the marker must be removed (move the test next to
the code it covers). Details: .claude/rules/{models,data,evaluation}.md.
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import rMAEEvaluator
from epftoolbox2.information_set import Availability, InformationSet, perturb_unknown
from epftoolbox2.models import OLSModel
from epftoolbox2.results.report import EvaluationReport
from tests.synthetic import PREDICTORS, calendar_shift, make_market_data


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


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


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="issue time inside the data zone's repeated autumn hour is compared in wall-clock minutes, not UTC instants",
)
def test_weather_cutoff_known_after_issue_across_the_repeated_autumn_hour():
    """A New York index, issue time 01:30 on 2024-11-03 (ambiguous: that day repeats 01:00-02:00), resolves to
    its earliest reading, 05:30Z. A weather_cutoff value known at 06:00Z (run="00", delay="6h") is therefore
    not yet known -- but the comparison is done in New York wall-clock minutes (01:00 < 01:30), not the UTC
    instants (06:00Z > 05:30Z), so it is let through unperturbed."""
    idx = pd.date_range("2024-11-01", "2024-11-05", freq="1h", tz="America/New_York")
    data = pd.DataFrame({"temp": np.arange(len(idx), dtype=float)}, index=idx)
    info = InformationSet("01:30", "America/New_York", rules={"temp": Availability.weather_cutoff("00", "6h")})

    perturbed, _counts, unchecked = perturb_unknown(data, "2024-11-03", info, freq="1h")

    assert unchecked == []
    nov_3 = np.asarray(data.index.tz_localize(None).normalize() == pd.Timestamp("2024-11-03"))
    assert (perturbed.loc[nov_3, "temp"].to_numpy() != data.loc[nov_3, "temp"].to_numpy()).all()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="realised/weather_run availability adds elapsed time to a wall-clock number instead of a UTC instant",
)
def test_realised_load_actual_known_after_issue_across_a_spring_forward():
    """A Warsaw index, issue time 03:30 on the 2024-03-31 spring-forward day (01:30Z). load_actual at 01:00 CET
    (00:00Z) is realised 1h after its period plus a 1h delay -- known at 02:00Z, after the 01:30Z issue -- but
    `point + step + delay` adds those as wall-clock minutes (01:00 + 1h + 1h = 03:00), which reads as before the
    issue time's own wall-clock reading (03:30), so it is let through unperturbed."""
    idx = pd.date_range("2024-03-29", "2024-04-02", freq="1h", tz="Europe/Warsaw")
    data = pd.DataFrame({"load_actual": np.arange(len(idx), dtype=float)}, index=idx)
    info = InformationSet("03:30", "Europe/Warsaw")  # default rules: load_actual = Availability.realised("1h")

    perturbed, _counts, unchecked = perturb_unknown(data, "2024-03-31", info, freq="1h")

    assert unchecked == []
    row = pd.Timestamp("2024-03-31 01:00", tz="Europe/Warsaw")
    assert perturbed.loc[row, "load_actual"] != data.loc[row, "load_actual"]
