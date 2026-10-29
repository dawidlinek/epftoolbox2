"""EvaluationReport computes grouped metrics in two places; exporters use compute_all(), users call by_*().

This pins that both paths give the same numbers for every grouping, including a relative metric and
models with different forecast sets.
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import Evaluator, MAEEvaluator, RMSEEvaluator, rMAEEvaluator
from epftoolbox2.evaluators.base import FORECAST_KEYS
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


class Recorder(Evaluator):
    """Keeps every frame the report hands to an evaluator."""

    name = "seen"

    def __init__(self):
        self.calls = []

    def compute(self, df, **kwargs):
        self.calls.append((df, kwargs["model_dfs"]))
        return float(len(df))


def keyed_report():
    recorder = Recorder()
    results = {"base": fake_results(1), "model": fake_results(2, skip_every=7)}
    return EvaluationReport(results, [recorder]), recorder


@pytest.mark.parametrize("method", ["summary", "compute_all", *GROUPINGS.values()])
def test_evaluators_receive_keyed_frames(method):
    """Every path (summary, compute_all, by_*) hands evaluators frames with the forecast keys, for the model and for
    every model in `model_dfs`."""
    report, recorder = keyed_report()

    getattr(report, method)()

    assert recorder.calls
    for df, model_dfs in recorder.calls:
        for frame in (df, *model_dfs.values()):
            assert list(frame.columns) == [*FORECAST_KEYS, "prediction", "actual"]
            assert pd.api.types.is_datetime64_any_dtype(frame["run_date"])
            assert pd.api.types.is_datetime64_any_dtype(frame["target_date"])
            assert not frame[list(FORECAST_KEYS)].isna().any().any()
            assert not frame.duplicated(list(FORECAST_KEYS)).any()


def test_grouped_frames_hold_only_their_group():
    report, recorder = keyed_report()

    report.by_hour_horizon()

    for df, model_dfs in recorder.calls:
        hour, horizon = df["hour"].iloc[0], df["horizon"].iloc[0]
        for frame in model_dfs.values():
            assert (frame["hour"] == hour).all() and (frame["horizon"] == horizon).all()
        assert ((df["target_date"] - df["run_date"]).dt.days == horizon).all()


def test_summary_frames_hold_every_forecast_of_the_model():
    report, recorder = keyed_report()

    summary = report.summary()

    (df, model_dfs) = recorder.calls[0]
    assert {name: len(frame) for name, frame in model_dfs.items()} == dict(zip(summary["model"], summary["forecasts"]))
    assert len(model_dfs["base"]) == len(fake_results(1))
    assert len(model_dfs["model"]) == len(fake_results(2, skip_every=7))
