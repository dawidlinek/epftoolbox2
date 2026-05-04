import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator, rMAEEvaluator
from epftoolbox2.exporters.csv import CsvExporter
from epftoolbox2.results.report import EvaluationReport


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
            "prediction": p, "actual": a, "hour": h, "horizon": hz,
            "target_date": td, "run_date": rd, "day_in_test": dit,
        }
        for p, a, h, hz, td, rd, dit in zip(
            predictions, actuals, hours, horizons, target_dates, run_dates, day_in_test,
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
        df = pd.DataFrame({"prediction": [10], "actual": [20]})
        perfect = pd.DataFrame({"prediction": [5], "actual": [5]})
        ev = rMAEEvaluator(base_model="Perfect")
        result = ev.compute(df, model_dfs={"Perfect": perfect})
        assert result == float("inf")

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
        import tempfile, yaml
        from pathlib import Path

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
                df.sort_values(["target_date", "hour", "horizon"])
                .reset_index(drop=True)
                .equals(df)
            )
