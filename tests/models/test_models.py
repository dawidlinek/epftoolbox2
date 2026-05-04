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

        assert results.count > 0
        assert "prediction" in results._results[0]
        assert "actual" in results._results[0]
        assert "coefficients" in results._results[0]

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

        assert results.count > 0

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

        assert results.count > 0

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

        assert results.count > 0

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

        result = results._results[0]
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

            assert results1.count == results2.count
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

        horizons = set(r["horizon"] for r in results._results)
        assert horizons == {1, 2, 3}

    def test_day_ids_follow_calendar_days_across_dst(self):
        index = pd.date_range(
            "2024-03-28 00:00",
            "2024-04-03 23:00",
            freq="h",
            tz="Europe/Warsaw",
        )
        data = pd.DataFrame({"load_actual": np.arange(len(index), dtype=float)}, index=index)

        model = OLSModel(predictors=["load_actual"], training_window=5)
        model._freq = "1h"
        processed = model._preprocess(data, horizon=1, target="load_actual")

        day_ids_by_date = processed.groupby(processed.index.date)["day"].first().to_numpy()
        expected = np.arange(len(day_ids_by_date))
        assert np.array_equal(day_ids_by_date, expected)

    def test_results_align_actual_with_target_date_after_dst(self, monkeypatch):
        monkeypatch.setenv("MAX_PROCESSES", "1")
        monkeypatch.setenv("THREADS_PER_PROCESS", "1")

        index = pd.date_range(
            "2024-01-01 00:00",
            "2024-04-05 23:00",
            freq="h",
            tz="Europe/Warsaw",
        )
        ordinals = np.array([ts.date().toordinal() for ts in index], dtype=float)
        load_actual = ordinals * 100.0 + index.hour.to_numpy(dtype=float)

        data = pd.DataFrame(
            {
                "load_actual": load_actual,
                "feature": np.sin(np.arange(len(index)) / 24.0),
            },
            index=index,
        )

        model = OLSModel(predictors=["feature"], training_window=30)
        results = model.run(
            data=data,
            test_start="2024-03-01",
            test_end="2024-04-01",
            target="load_actual",
            horizon=1,
        )

        rows = [
            r
            for r in (results._results or [])
            if r["run_date"] == "2024-03-31" and r["hour"] == 18 and r["horizon"] == 1
        ]
        assert rows, "Expected a row for run_date=2024-03-31, hour=18, horizon=1"

        row = rows[0]
        expected_ts = pd.Timestamp("2024-04-01 18:00:00", tz="Europe/Warsaw")
        expected_actual = float(data.loc[expected_ts, "load_actual"])

        assert row["target_date"] == "2024-04-01"
        assert row["actual"] == pytest.approx(expected_actual)

        run_dates = {r["run_date"] for r in (results._results or [])}
        assert "2024-04-01" in run_dates

    def test_multi_horizon_run_across_dst_completes(self, monkeypatch):
        monkeypatch.setenv("MAX_PROCESSES", "1")
        monkeypatch.setenv("THREADS_PER_PROCESS", "1")

        index = pd.date_range(
            "2023-12-01 00:00",
            "2024-04-10 23:00",
            freq="h",
            tz="Europe/Warsaw",
        )
        ordinals = np.array([ts.date().toordinal() for ts in index], dtype=float)
        data = pd.DataFrame(
            {
                "load_actual": ordinals * 100.0 + index.hour.to_numpy(dtype=float),
                "feature": np.cos(np.arange(len(index)) / 12.0),
            },
            index=index,
        )

        model = OLSModel(predictors=["feature"], training_window=30)
        results = model.run(
            data=data,
            test_start="2024-03-28",
            test_end="2024-04-01",
            target="load_actual",
            horizon=7,
        )

        rows = results._results or []
        assert rows
        assert all(np.isfinite(r["prediction"]) for r in rows)

        horizons = {r["horizon"] for r in rows}
        assert 1 in horizons
        assert 7 in horizons
