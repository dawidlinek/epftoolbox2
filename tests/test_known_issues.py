"""Known methodology issues, pinned with xfail(strict=True).

Each test asserts the *correct* behaviour and currently fails. When an issue is fixed its test starts
passing, strict mode turns that into a failure, and the marker must be removed (move the test next to
the code it covers). Details: .claude/rules/{models,data,evaluation}.md.
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import DailyProfileTransformer, DailyStatsTransformer
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


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="`_sNN` profile names don't say their slot length; the information set reads each as a quarter-hour",
)
def test_a_half_hour_profile_slot_is_judged_at_its_own_time():
    """DailyProfileTransformer names the slots of every length but one hour `_sNN`. With freq="30min" on hourly data,
    `load_actual_s30` is the load at 15:00 on D, realised at 16:30 with a 1 h rule, but the information set reads it
    as quarter-hour 30 (07:30-07:45, known 08:45) and accepts it at every period of an 11:45 issue: a look-ahead."""
    index = pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="Europe/Warsaw")
    raw = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    data = DailyProfileTransformer(columns="load_actual", freq="30min").transform(raw)
    day = data.loc["2024-01-10"]
    assert (day["load_actual_s30"] == raw.loc["2024-01-10 15:00", "load_actual"]).all()  # 15:00 on every row
    info = InformationSet("11:45", "Europe/Brussels", rules={"load_actual_s*": Availability.realised("1h")})
    expanded = {(period, 1): ["load_actual_s30"] for period in range(24)}

    found, _ = info.violations(expanded, "price", "1h", index, "2024-01-10", "2024-01-16")

    assert {v.period for v in found} == set(range(24))


_DAY_STATS_REASON = "a DailyStatsTransformer column is judged at the row's own period, not at the last slot of its day"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_DAY_STATS_REASON)
@pytest.mark.parametrize("stat", ["min", "max", "mean", "last"])
def test_day_aggregate_realised_after_the_issue_is_refused_at_every_period(stat):
    """Look-ahead. With a user rule `load_actual*` realised 1 h after its period, `load_actual_day_max` on the rows of D
    is the maximum over all 24 hours of D, known only when the last one is (01:00 on D+1): not at the 11:45 issue on D,
    whatever the row's period. Judged at the row's own period, it is accepted for periods 0-9 (each realised by 11:00)."""
    index = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="Europe/Warsaw")
    info = InformationSet("11:45", "Europe/Brussels", rules={"load_actual*": Availability.realised("1h")})
    expanded = {(period, 1): [f"load_actual_day_{stat}"] for period in range(24)}

    found, unknown = info.violations(expanded, "price", "1h", index, "2024-01-10", "2024-01-23")

    assert unknown == set()
    assert {v.period for v in found} == set(range(24))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_DAY_STATS_REASON)
def test_day_aggregate_of_a_lagged_column_is_accepted_at_every_period():
    """False refusal. LagTransformer before DailyStatsTransformer: `load_actual_d-1_day_max` on the rows of D is the
    maximum of `load_actual_d-1` over D, i.e. of the load on D-1, known at 01:00 on D. Judged at the row's own period
    as unshifted `load_actual`, it is refused for periods 10-23 ("known D+1 01:00")."""
    index = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="Europe/Warsaw")
    info = InformationSet("11:45", "Europe/Brussels", rules={"load_actual*": Availability.realised("1h")})
    expanded = {(period, 1): ["load_actual_d-1_day_max"] for period in range(24)}

    found, unknown = info.violations(expanded, "price", "1h", index, "2024-01-10", "2024-01-23")

    assert unknown == set()
    assert [(v.predictor, v.period) for v in found] == []


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_DAY_STATS_REASON)
def test_probe_perturbs_a_day_aggregate_realised_after_the_issue_on_every_row():
    """The leakage probe's side of the look-ahead above: every row of the run day carries the same
    `load_actual_day_max` (known 01:00 the next day), so every row must be perturbed, not only the rows from 10:00 on."""
    index = pd.date_range("2024-01-08", "2024-01-13", freq="h", tz="Europe/Warsaw", inclusive="left")
    raw = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)
    data = DailyStatsTransformer(columns="load_actual", stats=["max"]).transform(raw)[["load_actual_day_max"]]
    info = InformationSet("11:45", "Europe/Brussels", rules={"load_actual*": Availability.realised("1h")})

    perturbed, _counts, unchecked = perturb_unknown(data, "2024-01-10", info)

    assert unchecked == []
    run_day = np.asarray(index.tz_localize(None).normalize() == pd.Timestamp("2024-01-10"))
    assert (perturbed.loc[run_day, "load_actual_day_max"] != data.loc[run_day, "load_actual_day_max"]).all()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_DAY_STATS_REASON)
@pytest.mark.parametrize(
    "rule",
    [Availability.as_of("09:00"), Availability.weather_cutoff("00", "6h"), Availability.weather_run("6h")],
    ids=["as_of", "weather_cutoff", "weather_run"],
)
def test_a_day_lead_of_a_day_aggregate_is_a_shift_of_the_row(rule):
    """Look-ahead. `x_day_max_d+1` (LagTransformer(lags=[-1]) on the day aggregate `x_day_max`) is the maximum of x over
    D+1, known only on D+1 under these rules. They drop a leading `_d+k` as the source's own lead, but a `_d+k` after
    an aggregate always shifts the row, so dropping it judges the value of D instead: accepted at the 11:45 issue on D
    (all periods for as_of and weather_cutoff, periods 0-5 for weather_run)."""
    index = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="Europe/Warsaw")
    info = InformationSet("11:45", "Europe/Brussels", rules={"x*": rule})
    expanded = {(period, 1): ["x_day_max_d+1"] for period in range(24)}

    found, unknown = info.violations(expanded, "price", "1h", index, "2024-01-10", "2024-01-23")

    assert unknown == set()
    assert {v.period for v in found} == set(range(24))
