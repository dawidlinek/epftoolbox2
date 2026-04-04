"""Tests for forecast-only mode, date keywords, and related report.predictions()."""

import datetime
import pytest
import numpy as np
import pandas as pd

from epftoolbox2._date_utils import resolve_date
from epftoolbox2.models import OLSModel
from epftoolbox2.pipelines.model_pipeline import ModelPipeline
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.results.report import EvaluationReport


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """400 days of hourly data with price + one predictor, no NaNs."""
    np.random.seed(0)
    n_days = 400
    dates = pd.date_range("2023-01-01", periods=n_days * 24, freq="h")
    df = pd.DataFrame(
        {
            "price": np.random.randn(n_days * 24) * 10 + 50,
            "load": np.random.randn(n_days * 24) * 500 + 5000,
        },
        index=dates,
    )
    df["price_d-1"] = df["price"].shift(24)
    df["price_d-2"] = df["price"].shift(48)
    return df.dropna()


# ---------------------------------------------------------------------------
# resolve_date
# ---------------------------------------------------------------------------

class TestResolveDate:
    def test_today(self):
        assert resolve_date("today") == datetime.date.today().isoformat()

    def test_now(self):
        assert resolve_date("now") == datetime.date.today().isoformat()

    def test_case_insensitive(self):
        assert resolve_date("TODAY") == datetime.date.today().isoformat()
        assert resolve_date("Now") == datetime.date.today().isoformat()

    def test_positive_offset(self):
        expected = (datetime.date.today() + datetime.timedelta(days=7)).isoformat()
        assert resolve_date("today_d+7") == expected

    def test_negative_offset(self):
        expected = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
        assert resolve_date("now_d-365") == expected

    def test_large_offset(self):
        expected = (datetime.date.today() - datetime.timedelta(days=800)).isoformat()
        assert resolve_date("today_d-800") == expected

    def test_iso_string_passthrough(self):
        assert resolve_date("2024-01-01") == "2024-01-01"

    def test_non_string_passthrough(self):
        assert resolve_date(None) is None
        assert resolve_date(42) == 42

    def test_plain_word_passthrough(self):
        assert resolve_date("price") == "price"


# ---------------------------------------------------------------------------
# forecast_only=True — test_start/test_end default to today
# ---------------------------------------------------------------------------

class TestForecastOnlyDefaults:
    def test_model_runs_without_test_dates(self, sample_data):
        """forecast_only=True should run using today as test_start/test_end."""
        today = datetime.date.today().isoformat()
        # Build data that includes today so the model can find test rows
        extra_dates = pd.date_range(today, periods=24, freq="h")
        extra = pd.DataFrame(
            {
                "price": np.random.randn(24) * 10 + 50,
                "load": np.random.randn(24) * 500 + 5000,
                "price_d-1": np.random.randn(24) * 10 + 50,
                "price_d-2": np.random.randn(24) * 10 + 50,
            },
            index=extra_dates,
        )
        data = pd.concat([sample_data, extra])

        model = OLSModel(predictors=["price_d-1", "price_d-2"], training_window=30)
        ref = model.run(data=data, target="price", horizon=1, forecast_only=True)
        assert ref.count > 0

    def test_pipeline_runs_without_test_dates(self, sample_data):
        """ModelPipeline.run(forecast_only=True) should work without explicit dates."""
        today = datetime.date.today().isoformat()
        extra_dates = pd.date_range(today, periods=24, freq="h")
        extra = pd.DataFrame(
            {
                "price": np.random.randn(24) * 10 + 50,
                "load": np.random.randn(24) * 500 + 5000,
                "price_d-1": np.random.randn(24) * 10 + 50,
                "price_d-2": np.random.randn(24) * 10 + 50,
            },
            index=extra_dates,
        )
        data = pd.concat([sample_data, extra])

        report = (
            ModelPipeline()
            .add_model(OLSModel(predictors=["price_d-1"], training_window=30, name="OLS"))
            .run(data=data, target="price", horizon=1, forecast_only=True)
        )
        assert report is not None

    def test_requires_test_dates_without_forecast_only(self, sample_data):
        """Without forecast_only, omitting test_start/test_end must raise ValueError."""
        model = OLSModel(predictors=["price_d-1"], training_window=30)
        with pytest.raises(ValueError, match="test_start and test_end are required"):
            model.run(data=sample_data, target="price", horizon=1)


# ---------------------------------------------------------------------------
# forecast_only skips evaluators and exporters
# ---------------------------------------------------------------------------

class TestForecastOnlySkipsEvaluatorsExporters:
    def test_evaluators_not_run(self, sample_data):
        """With forecast_only=True evaluators should be skipped — report.summary() is empty."""
        today = datetime.date.today().isoformat()
        extra_dates = pd.date_range(today, periods=24, freq="h")
        extra = pd.DataFrame(
            {
                "price": np.random.randn(24) * 10 + 50,
                "load": np.random.randn(24) * 500 + 5000,
                "price_d-1": np.random.randn(24) * 10 + 50,
            },
            index=extra_dates,
        )
        data = pd.concat([sample_data, extra])

        report = (
            ModelPipeline()
            .add_model(OLSModel(predictors=["price_d-1"], training_window=30, name="OLS"))
            .add_evaluator(MAEEvaluator())
            .run(data=data, target="price", horizon=1, forecast_only=True)
        )
        # No evaluators ran → summary should have no metric columns
        summary = report.summary()
        assert "MAE" not in summary.columns


# ---------------------------------------------------------------------------
# _resolve_day raises ValueError for missing dates
# ---------------------------------------------------------------------------

class TestResolveDayError:
    def test_missing_test_start_raises(self, sample_data):
        """Requesting a test_start beyond data end must raise a clear ValueError."""
        model = OLSModel(predictors=["price_d-1"], training_window=30)
        with pytest.raises(ValueError, match="not in the data"):
            model.run(
                data=sample_data,
                test_start="2099-01-01",
                test_end="2099-01-01",
                target="price",
                horizon=1,
            )


# ---------------------------------------------------------------------------
# EvaluationReport.predictions()
# ---------------------------------------------------------------------------

class TestPredictions:
    @pytest.fixture
    def simple_report(self):
        results = {
            "ModelA": [
                {"run_date": "2024-01-01", "target_date": "2024-01-02", "hour": 3, "horizon": 1,
                 "day_in_test": 0, "prediction": 55.0, "actual": 54.0},
                {"run_date": "2024-01-01", "target_date": "2024-01-02", "hour": 0, "horizon": 1,
                 "day_in_test": 0, "prediction": 50.0, "actual": 51.0},
                {"run_date": "2024-01-01", "target_date": "2024-01-03", "hour": 0, "horizon": 2,
                 "day_in_test": 0, "prediction": 60.0, "actual": 61.0},
            ],
            "ModelB": [
                {"run_date": "2024-01-01", "target_date": "2024-01-02", "hour": 0, "horizon": 1,
                 "day_in_test": 0, "prediction": 48.0, "actual": 51.0},
            ],
        }
        return EvaluationReport(results, [])

    def test_returns_dataframe(self, simple_report):
        df = simple_report.predictions()
        assert isinstance(df, pd.DataFrame)

    def test_columns(self, simple_report):
        df = simple_report.predictions()
        assert set(df.columns) == {"run_date", "target_date", "hour", "horizon", "prediction", "model"}

    def test_no_day_in_test(self, simple_report):
        df = simple_report.predictions()
        assert "day_in_test" not in df.columns

    def test_no_actual(self, simple_report):
        df = simple_report.predictions()
        assert "actual" not in df.columns

    def test_sorted_by_model_horizon_hour(self, simple_report):
        df = simple_report.predictions()
        assert list(df["model"]) == ["ModelA", "ModelA", "ModelA", "ModelB"]
        assert list(df["horizon"]) == [1, 1, 2, 1]
        assert list(df["hour"].iloc[:2]) == [0, 3]  # hour sorted within horizon

    def test_all_models_present(self, simple_report):
        df = simple_report.predictions()
        assert set(df["model"].unique()) == {"ModelA", "ModelB"}

    def test_empty_report(self):
        report = EvaluationReport({}, [])
        df = report.predictions()
        assert df.empty
