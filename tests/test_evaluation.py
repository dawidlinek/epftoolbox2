import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import tempfile
import os

from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.results.report import EvaluationReport
from epftoolbox2.pipelines.model_pipeline import ModelPipeline
from epftoolbox2.exporters.terminal import TerminalExporter
from epftoolbox2.exporters.excel import ExcelExporter


class TestMAEEvaluator:
    def test_compute_simple(self):
        df = pd.DataFrame({"prediction": [10, 20, 30], "actual": [12, 18, 33]})
        evaluator = MAEEvaluator()
        result = evaluator.compute(df)
        expected = (2 + 2 + 3) / 3
        assert abs(result - expected) < 1e-9

    def test_compute_zero_error(self):
        df = pd.DataFrame({"prediction": [10, 20, 30], "actual": [10, 20, 30]})
        evaluator = MAEEvaluator()
        assert evaluator.compute(df) == 0.0

    def test_name(self):
        evaluator = MAEEvaluator()
        assert evaluator.name == "MAE"


class TestEvaluationReport:
    @pytest.fixture
    def sample_results(self):
        return {
            "model_a": [
                {"prediction": 10, "actual": 12, "hour": 0, "horizon": 1, "target_date": "2024-01-01"},
                {"prediction": 20, "actual": 18, "hour": 1, "horizon": 1, "target_date": "2024-01-01"},
                {"prediction": 30, "actual": 33, "hour": 0, "horizon": 2, "target_date": "2024-01-02"},
                {"prediction": 40, "actual": 38, "hour": 1, "horizon": 2, "target_date": "2024-01-02"},
            ],
            "model_b": [
                {"prediction": 11, "actual": 12, "hour": 0, "horizon": 1, "target_date": "2024-01-01"},
                {"prediction": 19, "actual": 18, "hour": 1, "horizon": 1, "target_date": "2024-01-01"},
                {"prediction": 32, "actual": 33, "hour": 0, "horizon": 2, "target_date": "2024-01-02"},
                {"prediction": 39, "actual": 38, "hour": 1, "horizon": 2, "target_date": "2024-01-02"},
            ],
        }

    def test_summary(self, sample_results):
        report = EvaluationReport(sample_results, [MAEEvaluator()])
        summary = report.summary()

        assert len(summary) == 2
        assert "model" in summary.columns
        assert "MAE" in summary.columns

        model_a_mae = summary[summary["model"] == "model_a"]["MAE"].iloc[0]
        expected_a = (2 + 2 + 3 + 2) / 4
        assert abs(model_a_mae - expected_a) < 1e-9

    def test_by_hour(self, sample_results):
        report = EvaluationReport(sample_results, [MAEEvaluator()])
        by_hour = report.by_hour()

        assert "hour" in by_hour.columns
        assert "model" in by_hour.columns
        assert len(by_hour) == 4  # 2 models × 2 hours

    def test_by_horizon(self, sample_results):
        report = EvaluationReport(sample_results, [MAEEvaluator()])
        by_horizon = report.by_horizon()

        assert "horizon" in by_horizon.columns
        assert len(by_horizon) == 4  # 2 models × 2 horizons

    def test_by_hour_horizon(self, sample_results):
        report = EvaluationReport(sample_results, [MAEEvaluator()])
        by_hh = report.by_hour_horizon()

        assert "hour" in by_hh.columns
        assert "horizon" in by_hh.columns
        assert len(by_hh) == 8  # 2 models × 2 hours × 2 horizons

    def test_by_year(self, sample_results):
        report = EvaluationReport(sample_results, [MAEEvaluator()])
        by_year = report.by_year()

        assert "year" in by_year.columns
        assert len(by_year) == 2  # 2 models × 1 year


class TestModelPipeline:
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range("2024-01-01", periods=48 * 10, freq="h")
        return pd.DataFrame({"price": np.random.randn(len(dates))}, index=dates)

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.name = "MockModel"
        model.run.return_value = [
            {"prediction": 10, "actual": 12, "hour": 0, "horizon": 1, "target_date": "2024-01-01"},
            {"prediction": 20, "actual": 18, "hour": 1, "horizon": 1, "target_date": "2024-01-01"},
        ]
        return model

    def test_init(self):
        pipeline = ModelPipeline()
        assert pipeline.models == []
        assert pipeline.evaluators == []
        assert pipeline.exporters == []

    def test_add_model_returns_self(self, mock_model):
        pipeline = ModelPipeline()
        result = pipeline.add_model(mock_model)
        assert result is pipeline
        assert len(pipeline.models) == 1

    def test_add_evaluator_returns_self(self):
        pipeline = ModelPipeline()
        evaluator = MAEEvaluator()
        result = pipeline.add_evaluator(evaluator)
        assert result is pipeline
        assert len(pipeline.evaluators) == 1

    def test_add_exporter_returns_self(self):
        pipeline = ModelPipeline()
        exporter = MagicMock()
        result = pipeline.add_exporter(exporter)
        assert result is pipeline
        assert len(pipeline.exporters) == 1

    def test_builder_pattern_chaining(self, mock_model):
        pipeline = ModelPipeline().add_model(mock_model).add_evaluator(MAEEvaluator()).add_exporter(MagicMock())
        assert len(pipeline.models) == 1
        assert len(pipeline.evaluators) == 1
        assert len(pipeline.exporters) == 1

    def test_run_without_models_raises_error(self, sample_data):
        pipeline = ModelPipeline()
        with pytest.raises(ValueError, match="At least one model is required"):
            pipeline.run(sample_data, "2024-01-05", "2024-01-10")

    def test_run_calls_model_run(self, sample_data, mock_model):
        pipeline = ModelPipeline().add_model(mock_model).add_evaluator(MAEEvaluator())
        pipeline.run(sample_data, "2024-01-05", "2024-01-10")
        mock_model.run.assert_called_once()

    def test_run_returns_evaluation_report(self, sample_data, mock_model):
        pipeline = ModelPipeline().add_model(mock_model).add_evaluator(MAEEvaluator())
        report = pipeline.run(sample_data, "2024-01-05", "2024-01-10")
        assert isinstance(report, EvaluationReport)

    def test_run_calls_exporters(self, sample_data, mock_model):
        exporter = MagicMock()
        pipeline = ModelPipeline().add_model(mock_model).add_evaluator(MAEEvaluator()).add_exporter(exporter)
        report = pipeline.run(sample_data, "2024-01-05", "2024-01-10")
        exporter.export.assert_called_once_with(report)

    def test_run_multiple_models(self, sample_data, mock_model):
        model2 = MagicMock()
        model2.name = "MockModel2"
        model2.run.return_value = [
            {"prediction": 11, "actual": 12, "hour": 0, "horizon": 1, "target_date": "2024-01-01"},
        ]
        pipeline = ModelPipeline().add_model(mock_model).add_model(model2).add_evaluator(MAEEvaluator())
        report = pipeline.run(sample_data, "2024-01-05", "2024-01-10")
        assert "MockModel" in report.results
        assert "MockModel2" in report.results


class TestTerminalExporter:
    @pytest.fixture
    def sample_report(self):
        results = {
            "model_a": [
                {"prediction": 10, "actual": 12, "hour": 0, "horizon": 1, "target_date": "2024-01-01"},
                {"prediction": 20, "actual": 18, "hour": 1, "horizon": 1, "target_date": "2024-01-01"},
            ],
        }
        return EvaluationReport(results, [MAEEvaluator()])

    def test_init_default_show(self):
        exporter = TerminalExporter()
        assert exporter.show == ["summary", "horizon"]

    def test_init_custom_show(self):
        exporter = TerminalExporter(show=["summary", "hour", "year"])
        assert exporter.show == ["summary", "hour", "year"]

    @patch("epftoolbox2.exporters.terminal.Console")
    def test_export_summary(self, mock_console_class, sample_report):
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        exporter = TerminalExporter(show=["summary"])
        exporter.export(sample_report)
        assert mock_console.print.called

    @patch("epftoolbox2.exporters.terminal.Console")
    def test_export_all_sections(self, mock_console_class, sample_report):
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        exporter = TerminalExporter(show=["summary", "hour", "horizon", "hour_horizon", "year", "year_horizon"])
        exporter.export(sample_report)
        # Should have multiple print calls for headers and tables
        assert mock_console.print.call_count >= 6

    def test_df_to_table(self, sample_report):
        exporter = TerminalExporter()
        df = sample_report.summary()
        table = exporter._df_to_table(df)
        assert table is not None

    def test_format_float(self):
        exporter = TerminalExporter()
        assert exporter._format(1.23456789) == "1.2346"

    def test_format_string(self):
        exporter = TerminalExporter()
        assert exporter._format("test") == "test"

    def test_format_int(self):
        exporter = TerminalExporter()
        assert exporter._format(42) == "42"


class TestExcelExporter:
    @pytest.fixture
    def sample_report(self):
        results = {
            "model_a": [
                {"prediction": 10, "actual": 12, "hour": 0, "horizon": 1, "target_date": "2024-01-01", "run_date": "2023-12-31", "day_in_test": 0},
                {"prediction": 20, "actual": 18, "hour": 1, "horizon": 1, "target_date": "2024-01-01", "run_date": "2023-12-31", "day_in_test": 0},
                {"prediction": 30, "actual": 33, "hour": 0, "horizon": 2, "target_date": "2024-01-02", "run_date": "2024-01-01", "day_in_test": 1},
                {"prediction": 40, "actual": 38, "hour": 1, "horizon": 2, "target_date": "2024-01-02", "run_date": "2024-01-01", "day_in_test": 1},
            ],
            "model_b": [
                {"prediction": 11, "actual": 12, "hour": 0, "horizon": 1, "target_date": "2024-01-01", "run_date": "2023-12-31", "day_in_test": 0},
                {"prediction": 19, "actual": 18, "hour": 1, "horizon": 1, "target_date": "2024-01-01", "run_date": "2023-12-31", "day_in_test": 0},
                {"prediction": 32, "actual": 33, "hour": 0, "horizon": 2, "target_date": "2024-01-02", "run_date": "2024-01-01", "day_in_test": 1},
                {"prediction": 39, "actual": 38, "hour": 1, "horizon": 2, "target_date": "2024-01-02", "run_date": "2024-01-01", "day_in_test": 1},
            ],
        }
        return EvaluationReport(results, [MAEEvaluator()])

    def test_init_default_sheets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.xlsx")
            exporter = ExcelExporter(path)
            assert exporter.sheets == ["summary", "hour", "horizon", "hour_horizon", "year", "year_horizon", "details"]

    def test_init_custom_sheets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.xlsx")
            exporter = ExcelExporter(path, sheets=["summary", "horizon"])
            assert exporter.sheets == ["summary", "horizon"]

    def test_export_creates_file(self, sample_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.xlsx")
            exporter = ExcelExporter(path)
            exporter.export(sample_report)
            assert os.path.exists(path)

    def test_export_creates_parent_directory(self, sample_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "test.xlsx")
            exporter = ExcelExporter(path)
            exporter.export(sample_report)
            assert os.path.exists(path)

    def test_export_summary_sheet(self, sample_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.xlsx")
            exporter = ExcelExporter(path, sheets=["summary"])
            exporter.export(sample_report)
            df = pd.read_excel(path, sheet_name="Summary")
            assert "model" in df.columns
            assert len(df) == 2

    def test_export_horizon_sheet(self, sample_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.xlsx")
            exporter = ExcelExporter(path, sheets=["horizon"])
            exporter.export(sample_report)
            df = pd.read_excel(path, sheet_name="horizon_MAE")
            assert "horizon_1" in df.columns or "horizon_2" in df.columns

    def test_export_all_sheets(self, sample_report):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.xlsx")
            exporter = ExcelExporter(path)
            exporter.export(sample_report)
            with pd.ExcelFile(path) as xl:
                expected_sheets = ["Summary", "hour_MAE", "horizon_MAE", "HourHorizon_MAE", "year_MAE", "YearHorizon_MAE"]
                for sheet in expected_sheets:
                    assert sheet in xl.sheet_names
