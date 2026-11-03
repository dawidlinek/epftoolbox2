"""The published week-ahead study for Poland, reproduced through the library (step 3i).

`tests/fixtures/week_ahead/` holds the study's daily frame stamped at noon (`PL_noon.parquet`, the rows the issue of Friday
2024-10-25 reads) and its stored forecasts of the 104 Fridays from 2024-01-05 (`PL_direct.csv`); `build.py` there writes both from
the local copy of the study, outside CI. The Friday form of the docs page Week-ahead Forecasting refits the study's models: the
regional NWP Lasso and the expert OLS within 0.01 MW of the stored forecasts (room for another scikit-learn, the CSV keeps three
decimals), the TSO's forecast and the actual to the CSV's rounding. That needs the study's settings, which aren't today's
defaults: with them the forecasts land tens of MW away. The stored forecasts of all 104 Fridays then go through a report: MAE per
statistic and horizon, the study's log-gain over the TSO, and the DM statistic per horizon by the study's own formula with 4 lags,
on 104 days.

All 1,456 fits of Poland and the 20 markets are refitted outside CI (`epftoolbox2-research/library_check_step3/week_ahead/`).
"""

import datetime
import math
import warnings
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.evaluators.stats import default_lag
from epftoolbox2.information_set import Availability, InformationSet, InformationSetError
from epftoolbox2.models import LassoCVModel, NaiveModel, OLSModel
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.ref import ModelResultRef
from epftoolbox2.results.report import EvaluationReport
from tests.fixtures.week_ahead import study

HERE = Path(__file__).parent / "fixtures" / "week_ahead"
ZONE = "Europe/Warsaw"
ISSUE = "2024-10-25"  # a Friday whose week holds the autumn switch of 27 October
FRIDAYS = ("2024-01-05", "2025-12-26")
HORIZONS = tuple(study.HORIZONS)
RULES = {
    "actual_daily_*": Availability.realised("1h"),
    "all_avg_*": Availability.weather_cutoff("00", "8h"),
    "capital_*": Availability.weather_cutoff("00", "8h"),
    "is_sunday_or_holiday": Availability.static(),
}
S02 = dict(cv=7, max_iter=100_000, fit_intercept=False, scaler={"name": "standard", "params": {"scale_binary": True}})
STORED = {"TSO": "tso_prediction", "lasso": "regional_nwp_prediction", "expert": "expert_prediction"}
# 100 ln(MAE of the TSO / MAE of the Lasso) over the 728 forecasts of a statistic: the study's log-gain for Poland.
LOG_GAIN = {"min": 23.573, "max": 20.317}


class InProcessPool:
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        _worker_init(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, task, *args):
        future = Future()
        try:
            future.set_result(task(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    """One process and one BLAS thread: the fits run here, where a pool would limit its threads (`_pool_init`)."""
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def df_daily():
    return pd.read_parquet(HERE / "PL_noon.parquet")


@pytest.fixture(scope="module")
def stored():
    return pd.read_csv(HERE / "PL_direct.csv", dtype={"run_date": str})


def information_set():
    return InformationSet("12:00", ZONE, rules=RULES)


def friday_form(stat, availability="warn"):
    """The docs page's Friday form, without the validation window."""
    return (
        ModelPipeline(information_set=information_set(), availability=availability)
        .add_model(NaiveModel(predictors=[f"load_forecast_daily_{stat}_d+{{horizon}}"], name="TSO"))
        .add_model(OLSModel(predictors=study.expert(stat), training_window=1092, fit_intercept=True, name="expert"))
        .add_model(LassoCVModel(predictors=study.REGIONAL_NWP, training_window=1092, name="lasso", **S02))
        .add_ensemble(AverageEnsemble(members=["lasso", "expert"], name="avg"))
        .add_evaluator(MAEEvaluator())
    )


def run_issue(pipeline, frame, stat):
    return pipeline.run(
        frame, target=f"actual_daily_{stat}", freq="1D", horizons=HORIZONS, issue_days=["friday"], test=(ISSUE, ISSUE)
    )


def by_horizon(records):
    return {r["horizon"]: r for r in records}


# --- the fixture ------------------------------------------------------------------------------------------------------------


def test_the_frame_is_the_study_s_stamped_at_noon(df_daily, stored):
    index = df_daily.index
    assert str(index.tz) == ZONE and len(df_daily) == 1113
    local = index.tz_localize(None)
    assert (local.hour == 12).all() and (local.minute == 0).all() and not local.normalize().has_duplicates
    assert (local[0].date().isoformat(), local[-1].date().isoformat()) == ("2021-10-20", "2024-11-05")
    assert list(df_daily.columns) == study.columns() + ["actual_daily_min", "actual_daily_max"]
    assert not any(name.startswith("actual_daily") and "_d+" in name for name in df_daily.columns)
    # The targets, built from the study's `_d+k` columns, are the stored actuals wherever the frame reaches.
    dates = pd.Series(df_daily.index.date, index=df_daily.index)
    for row in stored[stored["run_date"] <= ISSUE].itertuples():
        target_day = datetime.date.fromisoformat(row.run_date) + datetime.timedelta(days=int(row.horizon))
        value = df_daily.loc[(dates == target_day).to_numpy(), f"actual_daily_{row.target}"].iloc[0]
        assert abs(value - row.actual) <= 0.0005, (row.run_date, row.horizon, row.target)


# --- refitting one Friday ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("stat", study.STATS)
def test_the_friday_form_refits_the_study_on_25_october_2024(df_daily, stored, stat):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = run_issue(friday_form(stat), df_daily, stat)
    expected = stored[(stored["run_date"] == ISSUE) & (stored["target"] == stat)].set_index("horizon")
    for name, column in STORED.items():
        found = by_horizon(report.refs[name]._results)
        assert sorted(found) == list(HORIZONS), name
        tolerance = 0.0005 if name == "TSO" else 0.01
        for horizon, record in found.items():
            assert abs(record["prediction"] - expected.loc[horizon, column]) <= tolerance, (name, horizon)
            assert abs(record["actual"] - expected.loc[horizon, "actual"]) <= 0.0005, (name, horizon)
    average = by_horizon(report.refs["avg"]._results)
    for horizon in HORIZONS:
        mean = (expected.loc[horizon, "regional_nwp_prediction"] + expected.loc[horizon, "expert_prediction"]) / 2
        assert abs(average[horizon]["prediction"] - mean) <= 0.01
    # The one late input, by the rules: the hour before noon, known 13:00, which the study's list holds.
    listing = report.availability()
    assert listing[["model", "predictor", "status"]].values.tolist() == [["lasso", "load_actual_h-1", "late"]]
    assert any("load_actual_h-1 is late" in str(w.message) for w in caught)


@pytest.mark.parametrize("stat", study.STATS)
def test_today_s_defaults_don_t_reproduce_the_study(df_daily, stored, stat):
    """Without an intercept-free Lasso on every column scaled, 7 folds, and an OLS with an intercept, the forecasts land tens of
    MW away: what a reproduction of the plan's snippet would get."""
    expected = stored[(stored["run_date"] == ISSUE) & (stored["target"] == stat)].set_index("horizon")
    pipeline = (
        ModelPipeline()
        .add_model(LassoCVModel(predictors=study.REGIONAL_NWP, training_window=1092, name="lasso"))
        .add_model(OLSModel(predictors=study.expert(stat), training_window=1092, name="expert"))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = run_issue(pipeline, df_daily, stat)
    for name in ("lasso", "expert"):
        gaps = [abs(r["prediction"] - expected.loc[r["horizon"], STORED[name]]) for r in report.refs[name]._results]
        assert min(gaps) > 1.0 and np.median(gaps) > 10.0, (name, gaps)


def test_a_midnight_stamp_hides_the_late_hour(df_daily):
    """Stamped at midnight, as the study's file is, `load_actual_h-1` reads as 23:00 on D-1 and passes as known (decision 11):
    nothing is listed and only a warning about the stamp is left. At noon it is listed at "warn", refused at "raise", and the
    honest list, without it, passes "raise". An OLS on the Lasso's list stands in for it: the check reads names, not fits."""
    frame = df_daily.loc[:"2024-10-26"]
    midnight = frame.copy()
    midnight.index = (frame.index.tz_localize(None) - pd.Timedelta(hours=12)).tz_localize(ZONE)

    def listing(data, names, level="warn"):
        pipeline = ModelPipeline(information_set=information_set(), availability=level)
        pipeline.add_model(OLSModel(predictors=names, training_window=1092, name="nwp"))
        return pipeline.run(
            data, target="actual_daily_max", freq="1D", horizons=[3], issue_days=["friday"], test=(ISSUE, ISSUE)
        ).availability()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert listing(midnight, study.REGIONAL_NWP).empty
    assert any("00:00" in str(w.message) and "load_actual_h-1" in str(w.message) for w in caught)

    assert listing(frame, study.REGIONAL_NWP)["predictor"].tolist() == ["load_actual_h-1"]
    with pytest.raises(InformationSetError, match="load_actual_h-1"):
        listing(frame, study.REGIONAL_NWP, "raise")
    honest = [name for name in study.REGIONAL_NWP if name != study.LATE_AT_NOON]
    assert listing(frame, honest, "raise").empty


# --- verify -----------------------------------------------------------------------------------------------------------------


def test_verify_of_the_friday_form_finds_the_late_hour(df_daily):
    """The probe and its control on the Friday of the CI frame: the TSO and the expert are `ok`, and the expert's list with the
    late hour added is a `leak` at every horizon, as the study's Lasso list is (the research check verifies that list itself,
    which costs a few hundred fits)."""
    late = [*study.expert("max"), study.LATE_AT_NOON]
    pipeline = (
        ModelPipeline(information_set=information_set())
        .add_model(NaiveModel(predictors=["load_forecast_daily_max_d+{horizon}"], name="TSO"))
        .add_model(OLSModel(predictors=study.expert("max"), training_window=1092, fit_intercept=True, name="expert"))
        .add_model(OLSModel(predictors=late, training_window=1092, fit_intercept=True, name="with the late hour"))
    )
    report = pipeline.verify(
        df_daily, ISSUE, ISSUE, target="actual_daily_max", freq="1D", horizons=HORIZONS, issue_days=["friday"], days=1
    )
    statuses = {model: report.status(model) for model in report.models}
    assert statuses == {"TSO": "ok", "expert": "ok", "with the late hour": "leak"}, report.problems()
    leak = next(check for check in report.checks if check.model == "with the late hour")
    assert leak.probe.perturbed["load_actual_h-1"] > 0 and len(leak.probe.changed) == len(HORIZONS)


def test_verify_of_the_daily_form_is_ok_on_every_weekday(df_daily):
    """The daily form: issued every day, a list drops what `ctx.published` says isn't out, and the TSO's forecast that isn't out is
    no forecast. A week from Monday 21 October at horizons 3 to 7 (the frame's weather starts at `_d+3`), every weekday once. The
    expert's function stands in for the Lasso's, which drops the same names (the research check verifies it with the Lasso)."""
    pipeline = (
        ModelPipeline(information_set=information_set())
        .add_model(NaiveModel(predictors=["load_forecast_daily_max_d+{horizon}"], name="TSO"))
        .add_model(OLSModel(predictors=study.expert_known_max, training_window=1092, fit_intercept=True, name="expert"))
    )
    report = pipeline.verify(
        df_daily, "2024-10-21", "2024-10-27", target="actual_daily_max", freq="1D", horizons=range(3, 8), days=7
    )
    weekdays = sorted(datetime.date.fromisoformat(day).weekday() for day in report.issue_days)
    assert weekdays == list(range(7))
    assert {model: report.status(model) for model in report.models} == {"TSO": "ok", "expert": "ok"}, report.problems()
    tso = {check.run_day: check.probe.forecasts for check in report.checks if check.model == "TSO"}
    assert [tso[day] for day in sorted(tso)] == [4, 3, 2, 1, 5, 5, 5]  # h=7 late on Monday ... h=4-7 on Thursday
    # The expert's list holds the TSO's forecast of the target day where it is out, and not where it isn't.
    plan = OLSModel(predictors=study.expert_known_max, training_window=1092)._plan_predictors(
        df_daily, "actual_daily_max", 7, "1D", information_set(), "2024-10-21", "2024-10-27", horizons=range(3, 8)
    )
    for position, day in enumerate(plan.issue_days):
        for horizon in range(3, 8):
            names = plan.lists[plan.assignment[(0, horizon)][position]]
            out = horizon not in {0: {7}, 1: {6, 7}, 2: {5, 6, 7}, 3: {4, 5, 6, 7}}.get(day.weekday(), set())
            assert (f"load_forecast_daily_max_d+{horizon}" in names) == out, (day, horizon)


# --- the 104 Fridays through a report ---------------------------------------------------------------------------------------


def report_of(stored, stat, evaluators=()):
    """A report of the stored forecasts of one statistic, as plain records of a daily run on Fridays at horizons 3 to 9."""
    frame = stored[stored["target"] == stat]
    first = datetime.date.fromisoformat(FRIDAYS[0])
    refs = {}
    for name, column in STORED.items():
        records = []
        for row in frame.itertuples(index=False):
            run = datetime.date.fromisoformat(row.run_date)
            records.append(
                {
                    "run_date": row.run_date,
                    "target_date": (run + datetime.timedelta(days=int(row.horizon))).isoformat(),
                    "run_weekday": run.weekday(),
                    "hour": 0,
                    "horizon": int(row.horizon),
                    "day_in_test": (run - first).days,
                    "prediction": float(getattr(row, column)),
                    "actual": float(row.actual),
                    "freq": "1D",
                }
            )
        refs[name] = ModelResultRef(
            name, None, len(records), *FRIDAYS, horizon=9, freq="1D", _results=records, produced=len(records),
            horizons=HORIZONS, issue_days=(4,),
        )  # fmt: skip
    return EvaluationReport(refs, list(evaluators))


def study_dm(d, lags=4):
    """`4_dm_test.dm_pvalue`'s statistic: the mean of d over the square root of its Newey-West (Bartlett) variance over n."""
    n = len(d)
    dev = d - d.mean()
    omega = (dev**2).mean()
    for j in range(1, lags + 1):
        omega += 2 * (1 - j / (lags + 1)) * (dev[j:] * dev[:-j]).sum() / n
    return d.mean() / math.sqrt(omega / n)


@pytest.mark.parametrize("stat", study.STATS)
def test_the_104_fridays_through_a_report(stored, stat):
    report = report_of(stored, stat, [MAEEvaluator()])
    frame = stored[stored["target"] == stat]
    assert len(frame) == 104 * 7

    errors = {name: (frame[column] - frame["actual"]).abs() for name, column in STORED.items()}
    by = report.by_horizon().set_index(["model", "horizon"])
    for name in STORED:
        for horizon in HORIZONS:
            expected = errors[name][frame["horizon"] == horizon].mean()
            assert by.loc[(name, horizon), "MAE"] == pytest.approx(expected, abs=1e-9), (name, horizon)
    summary = report.summary().set_index("model")
    gain = 100 * math.log(summary.loc["TSO", "MAE"] / summary.loc["lasso", "MAE"])
    assert gain == pytest.approx(100 * math.log(errors["TSO"].mean() / errors["lasso"].mean()), abs=1e-9)
    assert round(gain, 3) == LOG_GAIN[stat]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tests = report.tests(benchmark="TSO", reps=200).set_index(["model", "horizon"])
    assert not [w for w in caught if "rests on" in str(w.message)]
    for name in ("lasso", "expert"):
        for horizon in HORIZONS:
            row = tests.loc[(name, horizon)]
            on = frame[frame["horizon"] == horizon].sort_values("run_date")
            d = ((on["tso_prediction"] - on["actual"]).abs() - (on[STORED[name]] - on["actual"]).abs()).to_numpy()
            assert row["statistic"] == pytest.approx(study_dm(d, lags=4), abs=1e-10), (name, horizon)
            assert (row["n_days"], row["n_forecasts"]) == (104, 104)
    assert default_lag(104) == 4  # the study's HAC_LAGS, the library's default at 104 days
