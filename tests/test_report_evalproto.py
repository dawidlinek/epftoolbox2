"""The public report reproduces the evaluation-protocol study (evalproto), not only the kernels behind it.

`tests/test_stats_evalproto.py` feeds the kernels by hand. Here a report is built from the same stored forecasts as plain
result records, and `tests()`, `compare()` and the evaluator columns must give the study's numbers: z, p and ratio within
1e-10 (`q1_placebo`), and the intervals within 0.005 (another seed moves them by Monte Carlo noise only), or exactly with
the study's seed. The profit, MAE and bias of `q4_metrics` come through `summary()`. In this test year every day has 24
periods, so the ratio of daily sums and the DM on daily means are what the study computed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import BiasEvaluator, DMEvaluator, EconomicEvaluator, MAEEvaluator
from epftoolbox2.results.report import EvaluationReport

FIXTURE = Path(__file__).parent / "fixtures" / "evalproto"
PLACEBO = pd.read_csv(FIXTURE / "q1_placebo_de_honest.csv")
METRICS = pd.read_csv(FIXTURE / "q4_metrics_de_honest.csv")
ROW_IDS = [f"{r.family}-{r.a}-{r.b}" for r in PLACEBO.itertuples()]
TOL = 1e-10
# `evalproto/q1_tests.py` draws its bootstrap indices from one generator with this seed; DE HONEST comes first, and the
# circular draw first within it, so `seed=STUDY_SEED` is the study's own draw.
STUDY_SEED = 20261018
VARIANTS = dict(zip(METRICS["model"], ("in168", "in168", "in168", "in168", "ctx364_exog")))


@pytest.fixture(scope="module")
def data():
    with np.load(FIXTURE / "de_honest_placebo.npz") as z:
        return {k: z[k] for k in z.files}


def records(forecast: np.ndarray, actual: np.ndarray, dates: np.ndarray) -> list[dict]:
    """Day-ahead result records of one model: hour h of `dates[i]`, issued the day before."""
    rows = []
    for i, day in enumerate(pd.to_datetime(dates)):
        run_date = (day - pd.Timedelta(days=1)).date().isoformat()
        target_date = day.date().isoformat()
        rows.extend(
            {
                "run_date": run_date,
                "target_date": target_date,
                "hour": hour,
                "horizon": 1,
                "day_in_test": i,
                "prediction": float(forecast[i, hour]),
                "actual": float(actual[i, hour]),
            }
            for hour in range(24)
        )
    return rows


def family_report(data, family: str, variant: str, evaluators=()) -> EvaluationReport:
    prefix = f"{family}__{variant}__"
    seeds = [k.removeprefix(prefix) for k in data if k.startswith(prefix)]
    return EvaluationReport({s: records(data[prefix + s], data["Y"], data["dates"]) for s in seeds}, list(evaluators))


@pytest.fixture(scope="module")
def reports(data):
    return {(r.family, r.variant): family_report(data, r.family, r.variant) for r in PLACEBO.itertuples()}


def row_of_tests(report: EvaluationReport, row, **settings) -> pd.Series:
    table = report.tests(benchmark=row.a, **settings).set_index("model")
    return table.loc[row.b]


@pytest.mark.parametrize("row", list(PLACEBO.itertuples()), ids=ROW_IDS)
def test_tests_gives_the_study_numbers(reports, row):
    report = reports[(row.family, row.variant)]

    plain = row_of_tests(report, row, variance="plain", interval=None)
    hac5 = row_of_tests(report, row, reps=4000)
    hac7 = row_of_tests(report, row, lag=7, interval=None)
    hac14 = row_of_tests(report, row, lag=14, interval=None)

    assert (plain["n_days"], plain["n_forecasts"]) == (365, 8760)
    assert plain["statistic"] == pytest.approx(row.z_plain, abs=TOL)
    assert plain["p_value"] == pytest.approx(row.p_plain, abs=TOL)
    assert hac5["statistic"] == pytest.approx(row.z_hac5, abs=TOL)
    assert hac5["p_value"] == pytest.approx(row.p_hac5, abs=TOL)
    assert hac7["p_value"] == pytest.approx(row.p_hac7, abs=TOL)
    assert hac14["p_value"] == pytest.approx(row.p_hac14, abs=TOL)
    assert hac5["ratio"] == pytest.approx(row.ratio, abs=TOL)
    assert (hac5["ci_low"], hac5["ci_high"]) == pytest.approx((row.mbb_lo, row.mbb_hi), abs=0.005)


@pytest.mark.parametrize("row", list(PLACEBO.itertuples()), ids=ROW_IDS)
def test_the_study_seed_gives_the_study_intervals(reports, row):
    found = row_of_tests(reports[(row.family, row.variant)], row, seed=STUDY_SEED)
    assert (found["ci_low"], found["ci_high"]) == pytest.approx((row.mbb_lo, row.mbb_hi), abs=1e-12)


@pytest.mark.parametrize("row", list(PLACEBO.itertuples()), ids=ROW_IDS)
def test_compare_gives_the_study_numbers(reports, row):
    found = reports[(row.family, row.variant)].compare(row.b, row.a).iloc[0]

    assert (found["n_days"], found["n_forecasts"]) == (365, 8760)
    assert found["ratio"] == pytest.approx(row.ratio, abs=TOL)
    assert found["p_plain"] == pytest.approx(row.p_plain, abs=TOL)
    assert found["p_hac"] == pytest.approx(row.p_hac5, abs=TOL)
    assert (found["ci_low"], found["ci_high"]) == pytest.approx((row.mbb_lo, row.mbb_hi), abs=0.005)
    assert found["hours_a_better"] == row.hours_better
    assert found["hours_b_better"] == row.hours_worse
    assert found["hours_tested"] == 24


def test_the_dm_columns_of_the_summary_are_the_tests_table(data):
    row = PLACEBO.iloc[0]
    evaluators = [DMEvaluator(benchmark=row.a)]
    report = family_report(data, row.family, row.variant, evaluators)

    summary = report.summary().set_index("model")
    table = report.tests().set_index("model")

    assert summary.loc[row.b, "DM ratio"] == table.loc[row.b, "ratio"]
    assert summary.loc[row.b, "DM ci_low"] == table.loc[row.b, "ci_low"]
    assert summary.loc[row.b, "DM ci_high"] == table.loc[row.b, "ci_high"]
    assert summary.loc[row.b, "DM p"] == pytest.approx(row.p_hac5, abs=TOL)
    assert summary.loc[row.b, "DM p_holm"] == table.loc[row.b, "p_holm"]
    assert summary.loc[row.b, "DM verdict"] == table.loc[row.b, "verdict"]
    assert summary.loc[row.b, "DM n_days"] == 365
    assert summary.loc[row.b, "DM n_forecasts"] == 8760


def test_summary_reproduces_profit_mae_and_bias_of_the_study(data):
    """The families of `q4_metrics` are the mean of their seeds' forecasts (evalproto/build.py), all in the fixture."""
    study = {}
    for family, variant in VARIANTS.items():
        prefix = f"{family}__{variant}__"
        forecast = np.mean([data[k] for k in data if k.startswith(prefix)], axis=0)
        study[family] = records(forecast, data["Y"], data["dates"])

    summary = (
        EvaluationReport(study, [MAEEvaluator(), BiasEvaluator(), EconomicEvaluator()]).summary().set_index("model")
    )

    for row in METRICS.itertuples():
        assert summary.loc[row.model, "profit"] == pytest.approx(row.profit, abs=1e-6)
        assert summary.loc[row.model, "profit % of perfect"] == pytest.approx(row.pct_pf, abs=1e-9)
        assert summary.loc[row.model, "profit days"] == 365
        assert summary.loc[row.model, "MAE"] == pytest.approx(row.MAE, abs=TOL)
        assert summary.loc[row.model, "Bias"] == pytest.approx(row.bias, abs=TOL)
