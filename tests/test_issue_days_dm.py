"""The Diebold-Mariano test on weekly issues: the 104 Fridays of the published week-ahead study for Poland.

`tests/fixtures/week_ahead/PL_direct.csv` holds the study's stored forecasts (`build.py` there copies them from the local study).
The reference statistics are what the study's own `4_dm_test.py` gives on that file with `dm_pvalue(d, lags=HAC_LAGS)`, HAC_LAGS
being 4: `epftoolbox2-research/library_check_step3/issue_days/dm_reference.py` computed them outside CI. The library's
Newey-West lag for 104 observations is that 4, so nothing is told to it.

`n_days` counts observations, so it is 104, and the test-length messages read the calendar span, which is two years.
"""

import datetime
import warnings
from pathlib import Path

import pandas as pd
import pytest

from epftoolbox2.evaluators import DMEvaluator, MAEEvaluator
from epftoolbox2.evaluators._compare import short_test_message
from epftoolbox2.evaluators.stats import default_lag
from epftoolbox2.results.ref import ModelResultRef
from epftoolbox2.results.report import EvaluationReport

FIXTURE = Path(__file__).parent / "fixtures" / "week_ahead" / "PL_direct.csv"
HORIZONS = tuple(range(3, 10))
FRIDAYS = ("2024-01-05", "2025-12-26")

# (target, horizon) -> (statistic, one-sided p-value) of 4_dm_test.dm_pvalue(d, lags=4), d = |TSO error| - |Lasso error|,
# positive where the Lasso is better; "all" is that script's joint test, d averaged over the seven horizons of one Friday.
REFERENCE = {
    ("min", 3): (2.5055139742543413, 0.006113678323511396),
    ("min", 4): (2.5714133912040884, 0.005064217281353756),
    ("min", 5): (3.4451336385710465, 0.00028538856138793367),
    ("min", 6): (4.998923056474265, 2.8825700304491164e-07),
    ("min", 7): (2.489432856886471, 0.00639735416872822),
    ("min", 8): (3.826841430834041, 6.489903444506506e-05),
    ("min", 9): (3.574070289571363, 0.00017573725531960793),
    ("min", "all"): (4.849757313211039, 6.180631005880954e-07),
    ("max", 3): (0.7611191404632462, 0.22329295450992936),
    ("max", 4): (3.4236681078183224, 0.00030891012811251617),
    ("max", 5): (2.5040624417738355, 0.006138818195386753),
    ("max", 6): (2.2719749764196093, 0.01154401088598267),
    ("max", 7): (2.3574724143951844, 0.009199911162068641),
    ("max", 8): (3.9956819710995277, 3.225414228302803e-05),
    ("max", 9): (3.2083267191813767, 0.0006675486746888781),
    ("max", "all"): (3.4900683983999743, 0.00024144846779172724),
}
MODELS = {"TSO": "tso_prediction", "lasso": "regional_nwp_prediction"}


def records(frame, column):
    """One record per (Friday, horizon), as a daily model writes them: one period, `freq` "1D"."""
    out = []
    first = datetime.date.fromisoformat(FRIDAYS[0])
    for row in frame.itertuples(index=False):
        run = datetime.date.fromisoformat(row.run_date)
        out.append(
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
    return out


def report_of(target, evaluators=()):
    frame = pd.read_csv(FIXTURE, dtype={"run_date": str})
    frame = frame[frame["target"] == target]
    refs = {}
    for name, column in MODELS.items():
        found = records(frame, column)
        refs[name] = ModelResultRef(
            name, None, len(found), *FRIDAYS, horizon=9, freq="1D", _results=found, produced=len(found),
            horizons=HORIZONS, issue_days=(4,),
        )  # fmt: skip
    return EvaluationReport(refs, list(evaluators))


def test_the_fixture_is_104_fridays_at_seven_horizons():
    frame = pd.read_csv(FIXTURE)
    assert len(frame) == 104 * 7 * 2
    assert set(pd.to_datetime(frame["run_date"]).dt.weekday) == {4}
    assert (frame["run_date"].min(), frame["run_date"].max()) == FRIDAYS
    assert sorted(frame["horizon"].unique()) == list(HORIZONS)


@pytest.mark.parametrize("target", ["min", "max"])
def test_the_dm_statistic_on_104_fridays_is_the_study_s(target):
    report = report_of(target)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        per_horizon = report.tests(benchmark="TSO", reps=200).set_index("horizon")
        joint = report.tests(benchmark="TSO", horizon="all", reps=200).set_index("horizon")
    # Two years of Fridays: no length warning, no note, and no day counts as partial (a daily series has one period a day).
    assert not [w for w in caught if "rests on" in str(w.message) or "part of their periods" in str(w.message)]
    assert "note" not in per_horizon.attrs and "note" not in joint.attrs
    assert default_lag(104) == 4

    table = pd.concat([per_horizon, joint])
    assert table["n_days"].tolist() == [104] * 8
    assert table.loc[list(HORIZONS), "n_forecasts"].tolist() == [104] * 7
    for horizon, row in table.iterrows():
        statistic, p_one_sided = REFERENCE[(target, horizon)]
        assert row["statistic"] == pytest.approx(statistic, abs=1e-10), horizon
        # The library's p-value is two-sided, the study's one-sided (H1: the Lasso beats the TSO).
        assert row["p_value"] == pytest.approx(2 * min(p_one_sided, 1 - p_one_sided), abs=1e-10), horizon

    # Told the study's lag, the library gives the same numbers.
    told = report.tests(benchmark="TSO", lag=4, reps=200).set_index("horizon")
    pd.testing.assert_series_equal(told["statistic"], per_horizon["statistic"])


def test_a_summary_on_104_fridays_has_104_days_and_no_length_warning():
    report = report_of("max", [MAEEvaluator(), DMEvaluator(benchmark="TSO", reps=200)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = report.summary().set_index("model")
    assert not [w for w in caught if "rests on" in str(w.message)]
    assert summary.loc["lasso", "DM n_days"] == 104
    assert summary.loc["lasso", "DM verdict"] == "better"


def test_a_comparison_shorter_than_a_year_says_so_by_its_span():
    """30 Fridays span 204 days: short, and said with both counts. A run of every day keeps its message word for word."""
    assert short_test_message(104, 722) is None
    assert short_test_message(30, 204).startswith(
        "The comparison rests on 30 days of forecasts over 204 calendar days, less than a year"
    )
    assert short_test_message(300) == short_test_message(300, 300)
    assert short_test_message(300).startswith("The comparison rests on 300 days, less than a year")

    def first_fridays(last):
        report = report_of("max")
        for ref in report.refs.values():
            ref.test_end = last
        return report

    # 30 Fridays: a note, and no warning above 182 days of span, though there are 30 observations.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = first_fridays("2024-07-26").tests(benchmark="TSO", horizon=3, reps=200)
    assert table["n_days"].tolist() == [30]
    assert table.attrs["note"].startswith("The comparison rests on 30 days of forecasts over 204 calendar days")
    # 20 Fridays span 134 days: indicative only.
    with pytest.warns(UserWarning, match="rests on 20 days of forecasts over 134 calendar days"):
        first_fridays("2024-05-17").tests(benchmark="TSO", horizon=3, reps=200)
