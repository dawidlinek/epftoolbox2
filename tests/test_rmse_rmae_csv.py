import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator, rMAEEvaluator
from epftoolbox2.exporters import ExcelExporter
from epftoolbox2.exporters.csv import CsvExporter
from epftoolbox2.models import OLSModel
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.report import EvaluationReport
from tests.synthetic import PREDICTORS, make_market_data


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_results(model_name, predictions, actuals, hours, horizons, target_dates, run_dates=None, day_in_test=None):
    """Build a list of result dicts for a single model."""
    if run_dates is None:
        run_dates = ["2023-12-31"] * len(predictions)
    if day_in_test is None:
        day_in_test = [0] * len(predictions)
    return [
        {
            "prediction": p,
            "actual": a,
            "hour": h,
            "horizon": hz,
            "target_date": td,
            "run_date": rd,
            "day_in_test": dit,
        }
        for p, a, h, hz, td, rd, dit in zip(
            predictions,
            actuals,
            hours,
            horizons,
            target_dates,
            run_dates,
            day_in_test,
        )
    ]


@pytest.fixture
def two_model_results():
    """Two models, 4 rows each, same forecast points."""
    return {
        "OLS": _make_results(
            "OLS",
            predictions=[10, 20, 30, 40],
            actuals=[12, 18, 33, 38],
            hours=[0, 1, 0, 1],
            horizons=[1, 1, 2, 2],
            target_dates=["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        ),
        "Lasso": _make_results(
            "Lasso",
            predictions=[11, 19, 32, 39],
            actuals=[12, 18, 33, 38],
            hours=[0, 1, 0, 1],
            horizons=[1, 1, 2, 2],
            target_dates=["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        ),
    }


def _record(run_date, hour, prediction, actual, horizon=1):
    """One result record; target_date is run_date + horizon days."""
    target = (pd.Timestamp(run_date) + pd.Timedelta(days=horizon)).date().isoformat()
    return {
        "run_date": run_date,
        "target_date": target,
        "hour": hour,
        "horizon": horizon,
        "day_in_test": 0,
        "prediction": prediction,
        "actual": actual,
    }


def _keyed(rows):
    """A keyed per-model frame from (run_date, hour, horizon, prediction, actual) rows."""
    return pd.DataFrame(
        [_record(run_date, hour, prediction, actual, horizon) for run_date, hour, horizon, prediction, actual in rows]
    ).assign(run_date=lambda d: pd.to_datetime(d["run_date"]), target_date=lambda d: pd.to_datetime(d["target_date"]))


# ===========================================================================
# RMSEEvaluator
# ===========================================================================


class TestRMSEEvaluator:
    def test_compute(self):
        df = pd.DataFrame({"prediction": [10, 20, 30], "actual": [12, 18, 33]})
        result = RMSEEvaluator().compute(df)
        expected = np.sqrt(((np.array([10, 20, 30]) - np.array([12, 18, 33])) ** 2).mean())
        assert abs(result - expected) < 1e-9

    def test_compute_zero_error(self):
        df = pd.DataFrame({"prediction": [5, 5], "actual": [5, 5]})
        assert RMSEEvaluator().compute(df) == 0.0

    def test_name(self):
        assert RMSEEvaluator().name == "RMSE"

    def test_in_report_summary(self, two_model_results):
        report = EvaluationReport(two_model_results, [RMSEEvaluator()])
        summary = report.summary()
        assert "RMSE" in summary.columns
        assert len(summary) == 2


# ===========================================================================
# rMAEEvaluator
# ===========================================================================


class TestRMAEEvaluator:
    def test_name(self):
        assert rMAEEvaluator(base_model="X").name == "rMAE"

    def test_base_model_equals_one(self, two_model_results):
        """rMAE of the base model against itself should be 1.0."""
        report = EvaluationReport(two_model_results, [rMAEEvaluator(base_model="OLS")])
        summary = report.summary()
        ols_rmae = summary[summary["model"] == "OLS"]["rMAE"].iloc[0]
        assert abs(ols_rmae - 1.0) < 1e-9

    def test_better_model_below_one(self, two_model_results):
        """Lasso has smaller errors than OLS in the fixture → rMAE < 1."""
        report = EvaluationReport(two_model_results, [rMAEEvaluator(base_model="OLS")])
        summary = report.summary()
        lasso_rmae = summary[summary["model"] == "Lasso"]["rMAE"].iloc[0]
        assert lasso_rmae < 1.0

    def test_raises_when_no_model_dfs(self):
        """Should raise when model_dfs is completely empty (no pipeline context)."""
        df = pd.DataFrame({"prediction": [1], "actual": [2]})
        ev = rMAEEvaluator(base_model="Missing")
        with pytest.raises(ValueError, match="requires 'model_dfs'"):
            ev.compute(df, model_dfs={})

    def test_nan_when_base_missing_from_slice(self):
        """Should return NaN when base model has no data for a given group."""
        df = pd.DataFrame({"prediction": [1, 2], "actual": [3, 4]})
        ev = rMAEEvaluator(base_model="Missing")
        result = ev.compute(df, model_dfs={"Other": df})
        assert math.isnan(result)

    def test_nan_when_base_empty(self):
        """Should return NaN when base model df is empty."""
        df = pd.DataFrame({"prediction": [1], "actual": [2]})
        empty = pd.DataFrame({"prediction": [], "actual": []})
        ev = rMAEEvaluator(base_model="Base")
        result = ev.compute(df, model_dfs={"Base": empty, "Other": df})
        assert math.isnan(result)

    def test_inf_when_base_mae_zero(self):
        """Should return inf when base model has zero MAE."""
        df = _keyed([("2024-01-01", 0, 1, 10, 20)])
        perfect = _keyed([("2024-01-01", 0, 1, 5, 5)])
        ev = rMAEEvaluator(base_model="Perfect")
        result = ev.compute(df, model_dfs={"Perfect": perfect})
        assert result == float("inf")

    def test_compares_models_on_shared_forecast_keys(self):
        """The candidate lacks the hour-1 forecast, so the benchmark's MAE is taken on hour 0 only."""
        benchmark = [_record("2024-01-01", 0, 11.0, 10.0), _record("2024-01-01", 1, 19.0, 10.0)]  # errors 1 and 9
        candidate = [_record("2024-01-01", 0, 12.0, 10.0)]  # error 2; the hour-1 forecast is missing

        report = EvaluationReport({"base": benchmark, "model": candidate}, [rMAEEvaluator(base_model="base")])
        summary = report.summary().set_index("model")

        # On the shared key the candidate's error is 2 vs the benchmark's 1 (not 2 vs the benchmark's mean of 5).
        assert summary.loc["model", "rMAE"] == pytest.approx(2.0)
        # The benchmark is scored on the same single key; its own count is still its full count.
        assert summary.loc["base", "rMAE"] == pytest.approx(1.0)
        assert summary.loc["base", "forecasts"] == 2

    def test_model_only_forecasts_do_not_enter_the_model_mae(self):
        """A forecast only the candidate has (here with a huge error) is left out of its MAE."""
        benchmark = [_record("2024-01-01", 0, 11.0, 10.0)]  # error 1
        candidate = [_record("2024-01-01", 0, 12.0, 10.0), _record("2024-01-01", 1, 500.0, 10.0)]  # errors 2, 490

        report = EvaluationReport({"base": benchmark, "model": candidate}, [rMAEEvaluator(base_model="base")])

        assert report.summary().set_index("model").loc["model", "rMAE"] == pytest.approx(2.0)

    def test_keys_include_run_date_and_horizon(self):
        """The same hour on another run date, or at another horizon, is another forecast."""
        benchmark = [_record("2024-01-01", 0, 11.0, 10.0), _record("2024-01-02", 0, 15.0, 10.0)]  # errors 1, 5
        candidate = [_record("2024-01-02", 0, 12.0, 10.0), _record("2024-01-02", 0, 30.0, 10.0, horizon=2)]  # 2, 20

        report = EvaluationReport({"base": benchmark, "model": candidate}, [rMAEEvaluator(base_model="base")])

        # Only (2024-01-02, hour 0, horizon 1) is shared: candidate error 2 against benchmark error 5.
        assert report.summary().set_index("model").loc["model", "rMAE"] == pytest.approx(2 / 5)

    def test_same_forecast_keys_give_the_plain_mae_ratio(self, two_model_results):
        """With identical keys the result equals the ratio of each model's own MAEs (the old behaviour)."""
        report = EvaluationReport(two_model_results, [MAEEvaluator(), rMAEEvaluator(base_model="OLS")])
        summary = report.summary().set_index("model")

        assert summary.loc["Lasso", "rMAE"] == pytest.approx(summary.loc["Lasso", "MAE"] / summary.loc["OLS", "MAE"])
        assert summary.loc["Lasso", "rMAE"] == pytest.approx(1.0 / 2.25)

    def test_nan_when_no_shared_keys(self):
        benchmark = [_record("2024-01-01", 0, 11.0, 10.0)]
        candidate = [_record("2024-01-01", 1, 12.0, 10.0)]

        report = EvaluationReport({"base": benchmark, "model": candidate}, [rMAEEvaluator(base_model="base")])
        summary = report.summary().set_index("model")

        assert math.isnan(summary.loc["model", "rMAE"])
        assert summary.loc["base", "rMAE"] == pytest.approx(1.0)

    def test_non_finite_values_drop_the_key_for_both_models(self):
        """A key whose prediction is NaN on either side is out of both MAEs."""
        benchmark = [_record("2024-01-01", 0, 11.0, 10.0), _record("2024-01-01", 1, 60.0, 10.0)]  # errors 1, 50
        candidate = [_record("2024-01-01", 0, 14.0, 10.0), _record("2024-01-01", 1, math.nan, 10.0)]  # errors 4, NaN

        report = EvaluationReport({"base": benchmark, "model": candidate}, [rMAEEvaluator(base_model="base")])

        assert report.summary().set_index("model").loc["model", "rMAE"] == pytest.approx(4.0)

    def test_grouped_tables_use_shared_keys_and_agree_with_compute_all(self):
        """By hour, the candidate (missing every other day of hour 1) is scored against the benchmark on its keys."""
        days = [d.date().isoformat() for d in pd.date_range("2024-01-01", periods=4, freq="D")]
        base_errors = {0: [1.0, 1.0, 1.0, 1.0], 1: [2.0, 4.0, 6.0, 8.0]}
        model_errors = {0: [2.0, 2.0, 2.0, 2.0], 1: [3.0, None, 3.0, None]}  # hour 1 only on days 0 and 2
        base, model = [], []
        for hour in (0, 1):
            for day, b_err, m_err in zip(days, base_errors[hour], model_errors[hour]):
                base.append(_record(day, hour, 100.0 + b_err, 100.0))
                if m_err is not None:
                    model.append(_record(day, hour, 100.0 + m_err, 100.0))
        report = EvaluationReport({"base": base, "model": model}, [rMAEEvaluator(base_model="base")])

        by_hour = report.by_hour().set_index(["hour", "model"])["rMAE"]
        assert by_hour[(0, "model")] == pytest.approx(2.0)
        # Hour 1 on days 0 and 2: candidate errors 3, 3; benchmark errors 2, 6 (mean 4), not its mean of 5 over all days.
        assert by_hour[(1, "model")] == pytest.approx(3.0 / 4.0)
        assert by_hour[(1, "base")] == pytest.approx(1.0)

        from_compute_all = report.compute_all()["hour"].set_index(["hour", "model"])["rMAE"]
        pd.testing.assert_series_equal(by_hour.sort_index(), from_compute_all.sort_index())

        # The overall number is on the 6 shared keys: hour 0 on four days and hour 1 on two.
        assert report.summary().set_index("model").loc["model", "rMAE"] == pytest.approx((8 + 6) / (4 + 8))

    def test_raises_when_key_columns_are_missing(self):
        """Result records without run_date cannot be aligned; say so instead of comparing unaligned rows."""
        rows = [{k: v for k, v in _record("2024-01-01", 0, 11.0, 10.0).items() if k != "run_date"}]
        report = EvaluationReport({"base": rows, "model": rows}, [rMAEEvaluator(base_model="base")])

        with pytest.raises(ValueError, match="run_date"):
            report.summary()

    def test_raises_when_some_keys_are_missing(self):
        rows = [_record("2024-01-01", 0, 11.0, 10.0), {"prediction": 1.0, "actual": 2.0, "hour": 1, "horizon": 1}]
        report = EvaluationReport({"base": rows, "model": rows}, [rMAEEvaluator(base_model="base")])

        with pytest.raises(ValueError, match="missing forecast key"):
            report.summary()

    def test_raises_on_duplicated_keys(self):
        """The same forecast twice in one model would count twice in the join."""
        base = [_record("2024-01-01", 0, 11.0, 10.0)]
        duplicated = [_record("2024-01-01", 0, 12.0, 10.0), _record("2024-01-01", 0, 13.0, 10.0)]
        report = EvaluationReport({"base": base, "model": duplicated}, [rMAEEvaluator(base_model="base")])

        with pytest.raises(ValueError, match="duplicated"):
            report.summary()

    def test_other_evaluators_do_not_need_the_keys(self):
        """A report with only MAE still works on records that carry just prediction and actual."""
        rows = {"M": [{"prediction": 10.0, "actual": 12.0}, {"prediction": 20.0, "actual": 18.0}]}
        summary = EvaluationReport(rows, [MAEEvaluator()]).summary()
        assert summary.loc[0, "MAE"] == pytest.approx(2.0)

    def test_grouped_evaluation(self, two_model_results):
        """rMAE should be computed per group in by_horizon."""
        report = EvaluationReport(two_model_results, [rMAEEvaluator(base_model="OLS")])
        by_hz = report.by_horizon()
        assert "rMAE" in by_hz.columns
        ols_h1 = by_hz[(by_hz["model"] == "OLS") & (by_hz["horizon"] == 1)]["rMAE"].iloc[0]
        assert abs(ols_h1 - 1.0) < 1e-9

    def test_serialization_roundtrip(self):
        """rMAEEvaluator should serialize/deserialize via COMPONENT_REGISTRY."""
        from epftoolbox2.pipelines.model_pipeline import ModelPipeline
        import tempfile
        import yaml

        pipeline = ModelPipeline().add_evaluator(rMAEEvaluator(base_model="OLS"))
        d = pipeline.to_dict()
        assert d["evaluators"][0]["class"] == "rMAEEvaluator"
        assert d["evaluators"][0]["params"]["base_model"] == "OLS"

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.dump(d, f)
            f_path = f.name
        try:
            loaded = ModelPipeline.load(f_path)
            assert len(loaded.evaluators) == 1
            assert isinstance(loaded.evaluators[0], rMAEEvaluator)
            assert loaded.evaluators[0].base_model == "OLS"
        finally:
            os.unlink(f_path)


# ===========================================================================
# CsvExporter
# ===========================================================================


class TestCsvExporter:
    def test_basic_export(self, two_model_results):
        report = EvaluationReport(two_model_results, [MAEEvaluator()])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.csv")
            CsvExporter(path).export(report)
            df = pd.read_csv(path)

            assert "OLS_prediction" in df.columns
            assert "Lasso_prediction" in df.columns
            assert "OLS_error" in df.columns
            assert "Lasso_error" in df.columns
            assert "actual" in df.columns

    def test_error_is_residual(self, two_model_results):
        report = EvaluationReport(two_model_results, [MAEEvaluator()])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.csv")
            CsvExporter(path).export(report)
            df = pd.read_csv(path)

            np.testing.assert_array_almost_equal(
                df["OLS_error"].values,
                df["OLS_prediction"].values - df["actual"].values,
            )

    def test_creates_parent_dirs(self, two_model_results):
        report = EvaluationReport(two_model_results, [MAEEvaluator()])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "out.csv")
            CsvExporter(path).export(report)
            assert os.path.exists(path)

    def test_empty_report(self):
        report = EvaluationReport({}, [MAEEvaluator()])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.csv")
            CsvExporter(path).export(report)
            assert not os.path.exists(path)

    def test_mismatched_rows_across_models(self):
        """Models with different forecast points should merge correctly."""
        results = {
            "Full": _make_results(
                "Full",
                predictions=[10, 20, 30],
                actuals=[12, 18, 33],
                hours=[0, 1, 0],
                horizons=[1, 1, 2],
                target_dates=["2024-01-01", "2024-01-01", "2024-01-02"],
            ),
            "Partial": _make_results(
                "Partial",
                predictions=[11, 19],
                actuals=[12, 18],
                hours=[0, 1],
                horizons=[1, 1],
                target_dates=["2024-01-01", "2024-01-01"],
            ),
        }
        report = EvaluationReport(results, [MAEEvaluator()])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.csv")
            CsvExporter(path).export(report)
            df = pd.read_csv(path)

            assert len(df) == 3
            assert "Full_prediction" in df.columns
            assert "Partial_prediction" in df.columns
            # Partial model should have NaN for the missing row
            partial_h2 = df[df["horizon"] == 2]["Partial_prediction"]
            assert partial_h2.isna().all()

    def test_extra_columns(self):
        results = {
            "M": _make_results(
                "M",
                predictions=[10],
                actuals=[12],
                hours=[14],
                horizons=[1],
                target_dates=["2024-01-02"],
                run_dates=["2024-01-01"],
            ),
        }
        source_idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        source_data = pd.DataFrame(
            {"is_holiday": [0] * 48, "temperature": np.random.randn(48)},
            index=source_idx,
        )
        report = EvaluationReport(results, [MAEEvaluator()], source_data=source_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.csv")
            CsvExporter(path, extra_columns=["is_holiday", "temperature"]).export(report)
            df = pd.read_csv(path)

            assert "is_holiday" in df.columns
            assert "temperature" in df.columns

    def test_extra_columns_missing_gracefully(self):
        """Requesting a non-existent column should not crash."""
        results = {
            "M": _make_results(
                "M",
                predictions=[10],
                actuals=[12],
                hours=[0],
                horizons=[1],
                target_dates=["2024-01-01"],
            ),
        }
        source_idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        source_data = pd.DataFrame({"price": [50] * 24}, index=source_idx)
        report = EvaluationReport(results, [], source_data=source_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.csv")
            CsvExporter(path, extra_columns=["nonexistent"]).export(report)
            df = pd.read_csv(path)
            assert "nonexistent" not in df.columns

    def test_no_source_data(self):
        """extra_columns requested but no source_data — should not crash."""
        results = {
            "M": _make_results(
                "M",
                predictions=[10],
                actuals=[12],
                hours=[0],
                horizons=[1],
                target_dates=["2024-01-01"],
            ),
        }
        report = EvaluationReport(results, [])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.csv")
            CsvExporter(path, extra_columns=["is_holiday"]).export(report)
            df = pd.read_csv(path)
            assert "is_holiday" not in df.columns

    def test_sorted_output(self, two_model_results):
        report = EvaluationReport(two_model_results, [])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.csv")
            CsvExporter(path).export(report)
            df = pd.read_csv(path)
            assert df["target_date"].is_monotonic_increasing or (
                df.sort_values(["target_date", "hour", "horizon"]).reset_index(drop=True).equals(df)
            )


# ===========================================================================
# CsvExporter: pinned hourly output and 15-min results
# ===========================================================================


def _hourly_records(offset):
    """Two run days x horizons 1-2 x hours 0, 1, 23, with values that are exact in binary floating point."""
    rows = []
    for run_date, day_in_test in (("2024-01-01", 0), ("2024-01-02", 1)):
        for horizon in (1, 2):
            target_date = (pd.Timestamp(run_date) + pd.Timedelta(days=horizon)).date().isoformat()
            for hour in (0, 1, 23):
                actual = 40.0 + hour + 10 * horizon + 5 * day_in_test
                rows.append(
                    {
                        "run_date": run_date,
                        "target_date": target_date,
                        "hour": hour,
                        "horizon": horizon,
                        "day_in_test": day_in_test,
                        "actual": actual,
                        "prediction": actual + offset + hour / 4,
                    }
                )
    return rows


# What CsvExporter wrote for these records before 15-min results were handled; hourly files must not change.
_HOURLY_CSV = """\
run_date,target_date,hour,horizon,day_in_test,actual,OLS_prediction,OLS_error,Lasso_prediction,Lasso_error,is_holiday,temperature
2024-01-01,2024-01-02,0,1,0,50.0,51.5,1.5,49.5,-0.5,0,6.0
2024-01-01,2024-01-02,1,1,0,51.0,52.75,1.75,50.75,-0.25,0,6.25
2024-01-01,2024-01-02,23,1,0,73.0,80.25,7.25,78.25,5.25,0,11.75
2024-01-02,2024-01-03,0,1,1,55.0,56.5,1.5,54.5,-0.5,1,12.0
2024-01-01,2024-01-03,0,2,0,60.0,61.5,1.5,59.5,-0.5,1,12.0
2024-01-02,2024-01-03,1,1,1,56.0,57.75,1.75,55.75,-0.25,1,12.25
2024-01-01,2024-01-03,1,2,0,61.0,62.75,1.75,60.75,-0.25,1,12.25
2024-01-02,2024-01-03,23,1,1,78.0,85.25,7.25,83.25,5.25,1,17.75
2024-01-01,2024-01-03,23,2,0,83.0,90.25,7.25,88.25,5.25,1,17.75
2024-01-02,2024-01-04,0,2,1,65.0,66.5,1.5,,,0,18.0
2024-01-02,2024-01-04,1,2,1,66.0,67.75,1.75,,,0,18.25
2024-01-02,2024-01-04,23,2,1,88.0,95.25,7.25,,,0,23.75
"""


def _quarter_hour_records(run_date, horizon, offset, periods=range(96)):
    target_date = (pd.Timestamp(run_date) + pd.Timedelta(days=horizon)).date().isoformat()
    return [
        {
            "run_date": run_date,
            "target_date": target_date,
            "hour": p,  # the period of the day: 0-95 at 15 min
            "horizon": horizon,
            "day_in_test": 0,
            "actual": 100.0 + p,
            "prediction": 100.0 + p + offset,
            "freq": "15min",
        }
        for p in periods
    ]


def _clock_time(period):
    return f"{period // 4:02d}:{period % 4 * 15:02d}"


class TestCsvExporterFreq:
    def test_hourly_output_is_pinned(self, tmp_path):
        results = {"OLS": _hourly_records(1.5), "Lasso": _hourly_records(-0.5)[:-3]}
        index = pd.date_range("2024-01-01", periods=96, freq="h", tz="Europe/Warsaw")
        source_data = pd.DataFrame(
            {"is_holiday": (index.day == 3).astype(int), "temperature": np.arange(96) / 4}, index=index
        )
        report = EvaluationReport(results, [MAEEvaluator()], source_data=source_data)
        path = tmp_path / "out.csv"
        CsvExporter(path, extra_columns=["is_holiday", "temperature"]).export(report)
        assert path.read_text() == _HOURLY_CSV

    def test_each_quarter_hour_is_one_row_with_its_clock_time(self, tmp_path):
        results = {
            "OLS": _quarter_hour_records("2024-01-01", 1, 1.0) + _quarter_hour_records("2024-01-01", 2, 2.0),
            "Lasso": _quarter_hour_records("2024-01-01", 1, -1.0),
        }
        path = tmp_path / "out.csv"
        CsvExporter(path).export(EvaluationReport(results, [MAEEvaluator()]))
        df = pd.read_csv(path)

        assert len(df) == 2 * 96
        assert not df.duplicated(["run_date", "hour", "horizon"]).any()
        assert sorted(df.loc[df["horizon"] == 1, "hour"]) == list(range(96))
        assert df["target_time"].tolist() == [_clock_time(p) for p in df["hour"]]
        h1 = df[df["horizon"] == 1].set_index("hour")
        assert (h1["OLS_error"] == 1.0).all() and (h1["Lasso_error"] == -1.0).all()
        assert df.loc[df["horizon"] == 2, "Lasso_prediction"].isna().all()
        # Rows come out in time order within each target day.
        assert df.sort_values(["target_date", "hour", "horizon"]).index.is_monotonic_increasing

    def test_extra_columns_follow_the_quarter_hour(self, tmp_path):
        results = {"M": _quarter_hour_records("2024-01-01", 1, 0.5)}
        index = pd.date_range("2024-01-02", periods=2 * 96, freq="15min", tz="Europe/Warsaw")
        source_data = pd.DataFrame({"slot": np.arange(len(index), dtype=float)}, index=index)
        path = tmp_path / "out.csv"
        CsvExporter(path, extra_columns=["slot"]).export(EvaluationReport(results, [], source_data=source_data))
        df = pd.read_csv(path)

        assert len(df) == 96
        assert df["slot"].tolist() == list(range(96))  # period p of 2024-01-02 is source row p, not a clock hour

    def test_report_from_a_15min_model_run(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MAX_PROCESSES", "1")
        monkeypatch.setenv("THREADS_PER_PROCESS", "1")
        data = make_market_data(start="2023-01-01", days=30, freq="15min", horizon=2, noise=1.0, seed=3)
        path = tmp_path / "out.csv"
        (
            ModelPipeline()
            .add_model(OLSModel(PREDICTORS, training_window=7, name="OLS"))
            .add_evaluator(MAEEvaluator())
            .add_exporter(CsvExporter(path, extra_columns=["load"]))
            .run(data, test_start="2023-01-20", test_end="2023-01-21", target="price", horizon=2, freq="15min")
        )
        df = pd.read_csv(path)

        assert len(df) == 2 * 2 * 96  # run days x horizons x periods
        assert not df.duplicated(["run_date", "hour", "horizon"]).any()
        assert df["load"].notna().all()
        assert df["target_time"].tolist() == [_clock_time(p) for p in df["hour"]]
        wall = data.set_index(data.index.tz_localize(None))
        stamps = pd.to_datetime(df["target_date"] + " " + df["target_time"])
        np.testing.assert_allclose(df["actual"], wall.loc[stamps, "price"].to_numpy())
        np.testing.assert_allclose(df["load"], wall.loc[stamps, "load"].to_numpy())

    @pytest.mark.parametrize(
        "start, run_day, rows",
        [
            ("2023-03-01", "2023-03-25", 92),  # the target day is the spring-forward day: no 02:00-02:45
            ("2023-10-01", "2023-10-29", 96),  # the run day is the fall-back day
        ],
    )
    def test_15min_run_around_a_dst_switch(self, tmp_path, monkeypatch, start, run_day, rows):
        monkeypatch.setenv("MAX_PROCESSES", "1")
        monkeypatch.setenv("THREADS_PER_PROCESS", "1")
        data = make_market_data(start=start, days=40, freq="15min", horizon=1, noise=1.0, seed=5)
        path = tmp_path / "out.csv"
        (
            ModelPipeline()
            .add_model(OLSModel(PREDICTORS, training_window=7, name="OLS"))
            .add_exporter(CsvExporter(path, extra_columns=["load"]))
            .run(data, test_start=run_day, test_end=run_day, target="price", horizon=1, freq="15min")
        )
        df = pd.read_csv(path)

        assert len(df) == rows
        assert not df.duplicated(["run_date", "hour", "horizon"]).any()
        assert df["load"].notna().all()
        wall = data.set_index(data.index.tz_localize(None))
        stamps = pd.to_datetime(df["target_date"] + " " + df["target_time"])
        np.testing.assert_allclose(df["load"], wall.loc[stamps, "load"].to_numpy())

    def test_report_over_saved_15min_files_ignores_the_default_freq(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MAX_PROCESSES", "1")
        monkeypatch.setenv("THREADS_PER_PROCESS", "1")
        data = make_market_data(start="2023-01-01", days=30, freq="15min", horizon=1, noise=1.0, seed=3)
        pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=7, name="OLS"))
        window = {"test_start": "2023-01-20", "test_end": "2023-01-20"}
        pipeline.run(data, target="price", horizon=1, freq="15min", save_dir=tmp_path / "runs", **window)

        path = tmp_path / "out.csv"
        merge = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=7, name="OLS"))
        # freq is left at its "1h" default: the records say 15min and win.
        merge.add_exporter(CsvExporter(path, extra_columns=["load"])).report(
            tmp_path / "runs", horizon=1, data=data, **window
        )
        df = pd.read_csv(path)

        assert len(df) == 96
        assert df["load"].notna().all()
        assert df["target_time"].iloc[5] == "01:15"

    def test_extra_columns_need_15min_source_rows_for_15min_results(self, tmp_path):
        results = {"M": _quarter_hour_records("2024-01-01", 1, 0.5)}
        hourly = pd.DataFrame({"slot": np.arange(48.0)}, index=pd.date_range("2024-01-02", periods=48, freq="h"))
        report = EvaluationReport(results, [], source_data=hourly)
        with pytest.raises(ValueError, match="15min results"):
            CsvExporter(tmp_path / "out.csv", extra_columns=["slot"]).export(report)

    def test_result_freqs_are_read_from_the_records(self):
        results = {
            "quarter": _quarter_hour_records("2024-01-01", 1, 0.0),
            "hourly": _hourly_records(0.0),  # records without a freq key are hourly
            "empty": [],  # no results, no frequency
        }
        assert EvaluationReport(results, [])._result_freqs() == {"quarter": "15min", "hourly": "1h"}

    def test_models_with_different_frequencies_do_not_share_a_file(self, tmp_path):
        results = {"hourly": _hourly_records(0.0), "quarter": _quarter_hour_records("2024-01-01", 1, 0.0)}
        report = EvaluationReport(results, [])
        with pytest.raises(ValueError, match="frequenc"):
            CsvExporter(tmp_path / "out.csv").export(report)


class TestExcelExporterFreq:
    def test_15min_results_keep_every_quarter_hour(self, tmp_path):
        results = {
            "A": _quarter_hour_records("2024-01-01", 1, 1.0),
            "B": _quarter_hour_records("2024-01-01", 1, -1.0),
        }
        path = tmp_path / "out.xlsx"
        ExcelExporter(path).export(EvaluationReport(results, [MAEEvaluator()]))
        sheets = pd.read_excel(path, sheet_name=None)

        details = sheets["Details"]
        assert len(details) == 96
        assert not details.duplicated(["run_date", "hour", "horizon"]).any()
        assert sorted(details["hour"]) == list(range(96))
        assert (details["prediction_A"] - details["actual"] == 1.0).all()
        assert (details["prediction_B"] - details["actual"] == -1.0).all()
        # One column per quarter-hour (the sheet keeps the library's name for the period of the day, "hour").
        assert sheets["hour_MAE"].shape == (2, 1 + 96)
