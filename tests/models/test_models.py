import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from epftoolbox2.models import OLSModel, LassoCVModel


class TestModels:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n_days = 400
        n_hours = n_days * 24

        dates = pd.date_range("2023-01-01", periods=n_hours, freq="h")

        data = pd.DataFrame(
            {
                "price": np.random.randn(n_hours) * 10 + 50,
                "load": np.random.randn(n_hours) * 1000 + 5000,
                "temperature": np.random.randn(n_hours) * 5 + 15,
            },
            index=dates,
        )

        data["price_lag_24"] = data["price"].shift(24)
        data["price_lag_48"] = data["price"].shift(48)
        data = data.dropna()

        return data

    def test_ols_model_runs(self, sample_data):
        model = OLSModel(
            predictors=["load", "temperature", "price_lag_24"],
            training_window=30,
            name="TestOLS",
        )

        results = model.run(
            data=sample_data,
            test_start="2023-12-01",
            test_end="2023-12-03",
            target="price",
            horizon=1,
        )

        assert len(results) > 0
        assert "prediction" in results[0]
        assert "actual" in results[0]
        assert "coefficients" in results[0]

    def test_lasso_model_runs(self, sample_data):
        model = LassoCVModel(
            predictors=["load", "temperature", "price_lag_24"],
            training_window=30,
            cv=3,
            name="TestLasso",
        )

        results = model.run(
            data=sample_data,
            test_start="2023-12-01",
            test_end="2023-12-02",
            target="price",
            horizon=1,
        )

        assert len(results) > 0

    def test_lambda_predictors(self, sample_data):
        model = OLSModel(
            predictors=[
                "load",
                lambda h: f"price_lag_{h * 24}",
            ],
            training_window=30,
        )

        results = model.run(
            data=sample_data,
            test_start="2023-12-01",
            test_end="2023-12-02",
            target="price",
            horizon=2,
        )

        assert len(results) > 0

    def test_template_predictors(self, sample_data):
        sample_data["price_lag_1"] = sample_data["price"].shift(1)
        sample_data = sample_data.dropna()

        model = OLSModel(
            predictors=["load", "price_lag_{horizon}"],
            training_window=30,
        )

        results = model.run(
            data=sample_data,
            test_start="2023-12-01",
            test_end="2023-12-02",
            target="price",
            horizon=1,
        )

        assert len(results) > 0

    def test_result_fields(self, sample_data):
        model = OLSModel(
            predictors=["load"],
            training_window=30,
        )

        results = model.run(
            data=sample_data,
            test_start="2023-12-01",
            test_end="2023-12-01",
            target="price",
            horizon=1,
        )

        result = results[0]
        assert "run_date" in result
        assert "target_date" in result
        assert "hour" in result
        assert "horizon" in result
        assert "day_in_test" in result
        assert "prediction" in result
        assert "actual" in result
        assert "coefficients" in result

    def test_save_and_resume(self, sample_data):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name

        try:
            model = OLSModel(
                predictors=["load"],
                training_window=30,
            )

            results1 = model.run(
                data=sample_data,
                test_start="2023-12-01",
                test_end="2023-12-01",
                target="price",
                horizon=1,
                save_to=temp_path,
            )

            results2 = model.run(
                data=sample_data,
                test_start="2023-12-01",
                test_end="2023-12-01",
                target="price",
                horizon=1,
                save_to=temp_path,
            )

            assert len(results1) == len(results2)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_multi_horizon(self, sample_data):
        model = OLSModel(
            predictors=["load"],
            training_window=30,
        )

        results = model.run(
            data=sample_data,
            test_start="2023-12-01",
            test_end="2023-12-01",
            target="price",
            horizon=3,
        )

        horizons = set(r["horizon"] for r in results)
        assert horizons == {1, 2, 3}
