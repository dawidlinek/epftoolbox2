"""EvaluationReport computes grouped metrics in two places; exporters use compute_all(), users call by_*().

This pins that both paths give the same numbers for every grouping, including a relative metric and
models with different forecast sets.
"""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import (
    BiasEvaluator,
    DMEvaluator,
    EconomicEvaluator,
    Evaluator,
    MAEEvaluator,
    MCSEvaluator,
    RMSEEvaluator,
    rMAEEvaluator,
)
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


SET_LEVEL = pytest.mark.filterwarnings("ignore:The model confidence set needs", "ignore:The comparison rests on")


@pytest.fixture(scope="module")
def set_level_report():
    """Evaluators that see the whole set of models, several columns each, and a model with different forecasts."""
    evaluators = [
        MAEEvaluator(),
        BiasEvaluator(),
        rMAEEvaluator(base_model="base"),
        DMEvaluator(benchmark="base", reps=100),
        MCSEvaluator(reps=100),
        EconomicEvaluator(),
    ]
    results = {"base": fake_results(1), "model": fake_results(2, skip_every=7), "third": fake_results(3)}
    return EvaluationReport(results, evaluators)


@pytest.fixture(scope="module")
def set_level_tables(set_level_report):
    return set_level_report.compute_all()


@SET_LEVEL
def test_summary_matches_with_set_level_evaluators(set_level_report, set_level_tables):
    pd.testing.assert_frame_equal(set_level_tables["summary"], set_level_report.summary(), check_exact=True)


@SET_LEVEL
@pytest.mark.parametrize("grouping, method", GROUPINGS.items())
def test_grouped_tables_match_with_set_level_evaluators(set_level_report, set_level_tables, grouping, method):
    from_method = getattr(set_level_report, method)()
    assert {"DM ratio", "DM p_holm", "MCS p", "in MCS", "profit", "profit % of perfect"} <= set(from_method.columns)
    pd.testing.assert_frame_equal(set_level_tables[grouping], from_method, check_exact=True)


@SET_LEVEL
def test_every_path_runs_on_the_same_group(set_level_report):
    """Each path gets a group's frames by the same cut: one row of the table is the evaluator on that group alone."""
    table = set_level_report.by_horizon()
    frames = {name: df[df["horizon"] == 1] for name, df in set_level_report._model_frames().items()}
    alone = DMEvaluator(benchmark="base", reps=100).compute_group(frames)
    row = table[(table["horizon"] == 1) & (table["model"] == "model")].iloc[0]
    assert row["DM ratio"] == alone["model"]["ratio"]
    assert row["DM p"] == alone["model"]["p"]


@SET_LEVEL
def test_set_level_tables_do_not_depend_on_the_order_of_the_records(set_level_report):
    results = {name: ref._results for name, ref in set_level_report.refs.items()}
    shuffled = {
        name: [records[i] for i in np.random.default_rng(9).permutation(len(records))]
        for name, records in results.items()
    }
    other = EvaluationReport(shuffled, set_level_report.evaluators)

    pd.testing.assert_frame_equal(other.summary(), set_level_report.summary(), check_exact=True)
    pd.testing.assert_frame_equal(other.by_horizon(), set_level_report.by_horizon(), check_exact=True)


def quarter_results(seed: int, skip_every: int = 0) -> list[dict]:
    """15-minute records over a few winter days (no clock change), two horizons."""
    rng = np.random.default_rng(seed)
    rows = []
    for day_in_test, run_date in enumerate(pd.date_range("2024-01-08", "2024-01-12", freq="D")):
        for horizon in (1, 2):
            for period in range(96):
                if skip_every and (day_in_test * 192 + horizon * 96 + period) % skip_every == 0:
                    continue
                actual = 50 + 10 * np.sin(period / 96 * 2 * np.pi) + (period % 4)
                rows.append(
                    {
                        "run_date": run_date.date().isoformat(),
                        "target_date": (run_date + pd.Timedelta(days=horizon)).date().isoformat(),
                        "run_weekday": run_date.dayofweek,
                        "hour": period,
                        "horizon": horizon,
                        "day_in_test": day_in_test,
                        "prediction": float(actual + rng.normal(0, 3)),
                        "actual": float(actual),
                        "freq": "15min",
                    }
                )
    return rows


@pytest.fixture(scope="module")
def period_report():
    from epftoolbox2.evaluators import NMAEEvaluator, ScaledMAEEvaluator, ShapeMAEEvaluator

    index = pd.date_range("2024-01-08", "2024-01-16", freq="15min", inclusive="left", tz="Europe/Warsaw")
    source = pd.DataFrame({"cap": 100.0 + np.arange(len(index)) % 7}, index=index)
    evaluators = [
        MAEEvaluator(),
        ShapeMAEEvaluator(),
        NMAEEvaluator(),
        ScaledMAEEvaluator("cap"),
        rMAEEvaluator("base"),
    ]
    results = {"base": quarter_results(1), "model": quarter_results(2, skip_every=11)}
    return EvaluationReport(results, evaluators, source_data=source)


def test_the_period_evaluators_agree_on_every_path(period_report):
    tables = period_report.compute_all()
    pd.testing.assert_frame_equal(tables["summary"], period_report.summary(), check_exact=True)
    assert {"ShapeMAE", "nMAE", "MAE/cap", "MAE/cap left out"} <= set(tables["summary"].columns)
    for grouping, method in GROUPINGS.items():
        pd.testing.assert_frame_equal(tables[grouping], getattr(period_report, method)(), check_exact=True)
    assert tables["year"]["ShapeMAE"].notna().all() and tables["hour"]["ShapeMAE"].isna().all()


def test_the_period_evaluators_do_not_depend_on_the_order_of_the_records(period_report):
    results = {name: ref._results for name, ref in period_report.refs.items()}
    shuffled = {
        name: [records[i] for i in np.random.default_rng(4).permutation(len(records))]
        for name, records in results.items()
    }
    other = EvaluationReport(shuffled, period_report.evaluators, source_data=period_report.source_data)
    pd.testing.assert_frame_equal(other.summary(), period_report.summary(), check_exact=True)
    pd.testing.assert_frame_equal(other.by_year(), period_report.by_year(), check_exact=True)
    pd.testing.assert_frame_equal(other.error_split(), period_report.error_split(), check_exact=True)
    pd.testing.assert_frame_equal(other.by_column("cap"), period_report.by_column("cap"), check_exact=True)
    for agg in ("mean", "max"):
        pd.testing.assert_frame_equal(
            other.aggregate("1h", agg=agg).summary(), period_report.aggregate("1h", agg=agg).summary(), check_exact=True
        )
    for agg in ("peak", "offpeak"):
        pd.testing.assert_frame_equal(
            other.aggregate("1D", agg=agg).summary(), period_report.aggregate("1D", agg=agg).summary(), check_exact=True
        )


def test_a_source_reading_evaluator_gets_its_column_next_to_the_keys(period_report):
    class Reads(Recorder):
        source_columns = ("cap",)

    recorder = Reads()
    report = EvaluationReport(
        {name: ref._results for name, ref in period_report.refs.items()},
        [recorder],
        source_data=period_report.source_data,
    )
    report.by_horizon()
    for df, model_dfs in recorder.calls:
        for frame in (df, *model_dfs.values()):
            assert list(frame.columns) == [*FORECAST_KEYS, "prediction", "actual", "cap"]
            assert frame["cap"].notna().all()
