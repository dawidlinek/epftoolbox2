import pytest
import pandas as pd
import numpy as np

from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.results.report import EvaluationReport


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
