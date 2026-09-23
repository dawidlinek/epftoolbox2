"""EvaluationReport computes grouped metrics in two places; exporters use compute_all(), users call by_*().

This pins that both paths give the same numbers for every grouping, including a relative metric and
models with different forecast sets.
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator, rMAEEvaluator
from epftoolbox2.results.report import EvaluationReport

GROUPINGS = {
    "hour": "by_hour",
    "horizon": "by_horizon",
    "hour_horizon": "by_hour_horizon",
    "year": "by_year",
    "year_horizon": "by_year_horizon",
    "run_weekday_horizon": "by_run_weekday_horizon",
    "target_weekday_horizon": "by_target_weekday_horizon",
}


def fake_results(seed: int, skip_every: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for day_in_test, run_date in enumerate(pd.date_range("2023-12-28", "2024-01-05", freq="D")):
        for horizon in (1, 2):
            for hour in range(24):
                if skip_every and (day_in_test * 48 + horizon * 24 + hour) % skip_every == 0:
                    continue
                actual = 50 + 10 * np.sin(hour / 24 * 2 * np.pi)
                rows.append(
                    {
                        "run_date": run_date.date().isoformat(),
                        "target_date": (run_date + pd.Timedelta(days=horizon)).date().isoformat(),
                        "run_weekday": run_date.dayofweek,
                        "hour": hour,
                        "horizon": horizon,
                        "day_in_test": day_in_test,
                        "prediction": float(actual + rng.normal(0, 3)),
                        "actual": float(actual),
                    }
                )
    return rows


@pytest.fixture(scope="module")
def report():
    evaluators = [MAEEvaluator(), RMSEEvaluator(), rMAEEvaluator(base_model="base")]
    return EvaluationReport({"base": fake_results(1), "model": fake_results(2, skip_every=7)}, evaluators)


def normalized(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in df.columns if c not in ("MAE", "RMSE", "rMAE")]
    return df.sort_values(keys).reset_index(drop=True)[sorted(df.columns)]


def test_summary_matches(report):
    pd.testing.assert_frame_equal(normalized(report.compute_all()["summary"]), normalized(report.summary()))


@pytest.mark.parametrize("grouping, method", GROUPINGS.items())
def test_grouped_metrics_match(report, grouping, method):
    from_compute_all = report.compute_all()[grouping]
    from_method = getattr(report, method)()
    assert not from_method.empty
    pd.testing.assert_frame_equal(normalized(from_compute_all), normalized(from_method), check_dtype=False)
