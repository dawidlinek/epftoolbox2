"""BiasEvaluator, DMEvaluator and MCSEvaluator in the report's tables."""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import BiasEvaluator, DMEvaluator, MCSEvaluator, stats
from epftoolbox2.results.report import EvaluationReport
from tests.forecast_records import forecast_records, shared_errors

pytestmark = pytest.mark.filterwarnings("ignore:The comparison rests on")

RUN_DATES = pd.date_range("2023-12-01", periods=70, freq="D")


def study(horizons=(1,), run_dates=RUN_DATES):
    return {
        "base": forecast_records(1, run_dates, noise=6.0, horizons=horizons),
        "better": forecast_records(2, run_dates, noise=4.0, horizons=horizons),
        "same": forecast_records(3, run_dates, noise=6.0, horizons=horizons),
    }


def test_bias_is_the_mean_error():
    frame = pd.DataFrame({"prediction": [12.0, 8.0, 13.0], "actual": [10.0, 10.0, 10.0]})
    assert BiasEvaluator().compute(frame) == pytest.approx(1.0)
    assert BiasEvaluator.name == "Bias"


# ---------------------------------------------------------------------------------------------------------------------
# DMEvaluator
# ---------------------------------------------------------------------------------------------------------------------


def test_dm_columns_in_the_summary():
    table = EvaluationReport(study(), [DMEvaluator("base", reps=100)]).summary().set_index("model")

    assert list(table.columns) == [
        "forecasts",
        "DM ratio",
        "DM ci_low",
        "DM ci_high",
        "DM p",
        "DM p_holm",
        "DM verdict",
        "DM n_days",
        "DM n_forecasts",
    ]
    assert table.loc["base", "DM ratio"] == 1.0 and np.isnan(table.loc["base", "DM p"])
    assert table.loc["base", "DM verdict"] is None
    assert table.loc["better", "DM verdict"] == "better"
    assert table.loc["better", "DM n_days"] == 70 and table.loc["better", "DM n_forecasts"] == 70 * 24
    np.testing.assert_allclose(
        table.loc[["better", "same"], "DM p_holm"], stats.holm(table.loc[["better", "same"], "DM p"].to_numpy())
    )


def test_dm_uses_each_pairs_keys():
    results = study()
    results["better"] = forecast_records(2, RUN_DATES, noise=4.0, drop=lambda r: r["day_in_test"] < 10)
    errors = shared_errors(results["better"], results["base"])

    table = EvaluationReport(results, [DMEvaluator("base", reps=50)]).summary().set_index("model")

    assert table.loc["better", "DM n_days"] == 60
    assert table.loc["same", "DM n_days"] == 70
    assert table.loc["better", "DM ratio"] == pytest.approx(errors["a"].mean() / errors["b"].mean(), rel=1e-12)
    assert table.loc["base", "DM n_days"] == 70


def test_dm_by_horizon_is_tests_row_by_row():
    report = EvaluationReport(study(horizons=(1, 2)), [DMEvaluator("base", reps=100)])

    by_horizon = report.by_horizon().set_index(["horizon", "model"])
    table = report.tests(reps=100).set_index(["horizon", "model"])

    for key, row in table.iterrows():
        assert by_horizon.loc[key, "DM ratio"] == row["ratio"]
        assert by_horizon.loc[key, "DM p"] == row["p_value"]
        assert by_horizon.loc[key, "DM p_holm"] == row["p_holm"]
        assert by_horizon.loc[key, "DM ci_low"] == row["ci_low"]
        assert by_horizon.loc[key, "DM verdict"] == row["verdict"]


def test_dm_of_several_horizons_in_the_summary_is_horizon_all():
    report = EvaluationReport(study(horizons=(1, 2)), [DMEvaluator("base", reps=100)])

    summary = report.summary().set_index("model")
    table = report.tests(horizon="all", reps=100).set_index("model")

    assert summary.loc["better", "DM p"] == table.loc["better", "p_value"]
    assert summary.loc["better", "DM ratio"] == table.loc["better", "ratio"]


@pytest.mark.parametrize("method", ["by_hour", "by_hour_horizon"])
def test_per_hour_tables_give_no_verdict(method):
    table = getattr(EvaluationReport(study(), [DMEvaluator("base", reps=20)]), method)()
    assert table["DM p_holm"].isna().all()
    assert table["DM verdict"].isna().all()
    assert table.query("model == 'better'")["DM p"].notna().all()
    assert (table.query("model == 'better'")["DM n_days"] == 70).all()


def test_dm_without_the_benchmark_warns_and_gives_nan():
    with pytest.warns(UserWarning, match="'missing' has no forecasts"):
        table = EvaluationReport(study(), [DMEvaluator("missing", reps=20)]).summary()
    assert table["DM ratio"].isna().all()


def test_dm_with_a_model_without_forecasts():
    results = study()
    results["empty"] = []
    table = EvaluationReport(results, [DMEvaluator("base", reps=20)]).summary().set_index("model")
    assert np.isnan(table.loc["empty", "DM ratio"]) and table.loc["empty", "DM n_days"] == 0
    assert table.loc["better", "DM n_days"] == 70


def test_a_perfect_benchmark_does_not_crash_the_table():
    results = study()
    for record in results["base"]:
        record["prediction"] = record["actual"]
    table = EvaluationReport(results, [DMEvaluator("base", reps=20)]).summary().set_index("model")
    assert table.loc["better", "DM ratio"] == np.inf
    assert table.loc["better", "DM verdict"] == "worse"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variance": "x"},
        {"correction": "x"},
        {"lag": -1},
        {"lag": 1.5},
        {"variance": "plain", "lag": 2},
        {"seed": None},
    ],
)
def test_dm_refuses_bad_settings(kwargs):
    with pytest.raises((ValueError, TypeError)):
        DMEvaluator("base", **kwargs)


# ---------------------------------------------------------------------------------------------------------------------
# MCSEvaluator
# ---------------------------------------------------------------------------------------------------------------------


def test_mcs_columns_match_the_report_method():
    report = EvaluationReport(
        study(run_dates=pd.date_range("2023-01-01", periods=200)), [MCSEvaluator(reps=200, seed=3)]
    )

    summary = report.summary().set_index("model")
    table = report.mcs(reps=200, seed=3).set_index("model")

    assert list(summary.columns) == ["forecasts", "MCS p", "in MCS", "MCS n_days", "MCS n_forecasts"]
    for model, row in table.iterrows():
        assert summary.loc[model, "MCS p"] == row["p_value"]
        assert summary.loc[model, "in MCS"] == row["in_90"]
        assert summary.loc[model, "MCS n_days"] == 200


def test_mcs_with_fewer_than_three_models_is_nan_with_a_warning():
    results = study()
    del results["same"]
    with pytest.warns(UserWarning, match="at least 3 models"):
        table = EvaluationReport(results, [MCSEvaluator(reps=20)]).summary()
    assert table["MCS p"].isna().all() and table["in MCS"].isna().all()


def test_mcs_of_identical_models_is_nan_with_a_warning_in_a_table():
    results = study()
    results["copy"] = results["base"]
    with pytest.warns(UserWarning, match="No model confidence set.*base, better, same, copy"):
        table = EvaluationReport(results, [MCSEvaluator(reps=20)]).summary()
    assert table["MCS p"].isna().all() and (table["MCS n_days"] == 70).all()


def test_mcs_on_the_keys_of_the_group():
    results = study()
    results["better"] = forecast_records(2, RUN_DATES, noise=4.0, drop=lambda r: r["day_in_test"] < 10)
    table = EvaluationReport(results, [MCSEvaluator(reps=20)]).summary()
    assert (table["MCS n_days"] == 60).all() and (table["MCS n_forecasts"] == 60 * 24).all()


@pytest.mark.parametrize("kwargs", [{"alpha": 0}, {"alpha": 1.5}, {"statistic": "x"}, {"seed": None}])
def test_mcs_refuses_bad_settings(kwargs):
    with pytest.raises((ValueError, TypeError)):
        MCSEvaluator(**kwargs)
