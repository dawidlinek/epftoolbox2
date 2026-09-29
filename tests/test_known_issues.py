"""Known methodology issues, pinned with xfail(strict=True).

Each test asserts the *correct* behaviour and currently fails. When an issue is fixed its test starts
passing, strict mode turns that into a failure, and the marker must be removed (move the test next to
the code it covers). Details: .claude/rules/{models,data,evaluation}.md.
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import rMAEEvaluator
from epftoolbox2.information_set import InformationSet
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
    strict=True, raises=AssertionError, reason="InformationSet.violations converts zone times with D's noon offsets"
)
@pytest.mark.parametrize("data_zone", ["UTC", "Etc/GMT-1", "Asia/Tokyo"])
@pytest.mark.parametrize("issue_time", ["11:30", "12:30", "13:30"])
def test_violations_use_the_delivery_and_publication_instants_own_offsets(data_zone, issue_time):
    """The pre-fit check's counterpart of test_perturbation_uses_the_delivery_and_publication_instants_own_offsets.
    On a frame whose zone has no DST (UTC, a benchmark source's fixed offset) checked against Brussels-anchored rules,
    `violations()` converts every delivery and publication time with the zone offsets at noon of the issue day D, and
    lets an earlier day with the same weekday and noon offsets stand for a day just before a switch. Issued on
    2024-03-30, `load_forecast_d+1` at 22:00Z (delivered 2024-04-01 00:00 CEST, published 10:00 on 2024-03-31)
    passes: a leak. Issued on Thursday 2024-10-24, `load_forecast_daily_max_d+3` at 22:00Z (delivered Sunday
    2024-10-27 23:00 CET, published the Friday before that week) is rejected. The expected verdict is computed
    independently, from real instants, for test ranges ending two days before to one day after each switch."""
    columns = {  # name: (published time Brussels, days_before or "weekly", day shift)
        "price": ("13:00", 1, 0),
        "price_d-1": ("13:00", 1, -1),
        "price_d+1": ("13:00", 1, 1),
        "load_forecast_d+1": ("10:00", 1, 1),
        "load_forecast_d+2": ("10:00", 1, 2),
        "load_forecast_daily_max_d+2": ("10:00", "weekly", 2),
        "load_forecast_daily_max_d+3": ("10:00", "weekly", 3),
    }
    info = InformationSet(issue_time, "Europe/Brussels")
    expanded = {(period, 1): list(columns) for period in range(24)}

    wrong = {}
    for switch in ["2024-03-31", "2024-10-27", "2025-10-26"]:
        for end in pd.date_range(pd.Timestamp(switch) - pd.Timedelta(days=2), periods=4, freq="D"):
            days = pd.date_range(end - pd.Timedelta(days=9), end, freq="D")
            index = pd.date_range(days[0] - pd.Timedelta(days=7), end + pd.Timedelta(days=7), freq="h", tz=data_zone)
            found, _ = info.violations(expanded, "price", "1h", index, str(days[0].date()), str(end.date()))
            got = {(v.predictor, v.period) for v in found if v.role == "predictor"}

            expected = set()
            for day in days:
                rows = pd.date_range(day, periods=24, freq="h").tz_localize(data_zone)
                issue = pd.Timestamp(f"{day.date()} {issue_time}", tz="Europe/Brussels")
                for name, (time, days_before, shift) in columns.items():
                    delivery = rows + pd.Timedelta(days=shift)  # no DST in the data zone: a calendar day is 24 h
                    delivery_day = delivery.tz_convert("Europe/Brussels").normalize().tz_localize(None)
                    if days_before == "weekly":  # Friday of the week before the delivery's Monday-Sunday week
                        published_day = delivery_day - pd.to_timedelta(delivery_day.weekday + 3, unit="D")
                    else:
                        published_day = delivery_day - pd.Timedelta(days=days_before)
                    published = (published_day + pd.Timedelta(f"{time}:00")).tz_localize("Europe/Brussels")
                    expected |= {(name, int(period)) for period in np.flatnonzero(published > issue)}
            if got != expected:
                wrong[str(end.date())] = {"missed": sorted(expected - got), "extra": sorted(got - expected)}
    assert wrong == {}
