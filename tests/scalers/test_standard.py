import pytest
import pandas as pd
import numpy as np
from epftoolbox2.scalers import StandardScaler


class TestStandardScaler:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        train = pd.DataFrame(
            {
                "price": np.random.randn(100) * 10 + 50,
                "load": np.random.randn(100) * 1000 + 5000,
                "temperature": np.random.randn(100) * 5 + 15,
                "is_holiday": np.random.choice([0, 1], 100),
                "hour": np.tile(np.arange(24), 5)[:100],
            }
        )
        test = pd.DataFrame(
            {
                "price": [55.0],
                "load": [5500.0],
                "temperature": [18.0],
                "is_holiday": [1],
                "hour": [12],
            }
        )
        return train, test

    def test_fit_transform_scales_continuous(self, sample_data):
        train, test = sample_data
        scaler = StandardScaler()

        train_scaled, test_scaled = scaler.fit_transform(train, test, ["load", "temperature"], "price")

        assert abs(train_scaled["load"].mean()) < 0.1
        assert abs(train_scaled["load"].std() - 1.0) < 0.1
        assert abs(train_scaled["temperature"].mean()) < 0.1

    def test_fit_transform_skips_binary(self, sample_data):
        train, test = sample_data
        scaler = StandardScaler()

        train_scaled, test_scaled = scaler.fit_transform(train, test, ["load", "is_holiday"], "price")

        assert set(train_scaled["is_holiday"].unique()).issubset({0, 1})
        assert train_scaled["is_holiday"].mean() == train["is_holiday"].mean()

    def test_inverse_restores_target(self, sample_data):
        train, test = sample_data
        scaler = StandardScaler()

        train_scaled, _ = scaler.fit_transform(train, test, ["load"], "price")

        original_mean = train["price"].mean()
        restored = scaler.inverse(0.0)

        assert abs(restored - original_mean) < 0.01

    def test_handles_zero_std(self):
        train = pd.DataFrame({"x": [5.0] * 10, "y": [1.0] * 10})
        test = pd.DataFrame({"x": [5.0], "y": [1.0]})

        scaler = StandardScaler()
        train_scaled, test_scaled = scaler.fit_transform(train, test, ["x"], "y")

        assert not train_scaled["x"].isna().any()

    def test_categorical_dtype_not_scaled(self):
        train = pd.DataFrame(
            {
                "category": pd.Categorical(["A", "B", "A", "C"]),
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        test = pd.DataFrame(
            {
                "category": pd.Categorical(["B"]),
                "value": [2.5],
            }
        )

        scaler = StandardScaler()
        train_scaled, test_scaled = scaler.fit_transform(train, test, ["category"], "value")

        assert train_scaled["category"].equals(train["category"])

    def test_does_not_modify_original(self, sample_data):
        train, test = sample_data
        train_orig = train.copy()
        test_orig = test.copy()

        scaler = StandardScaler()
        scaler.fit_transform(train, test, ["load"], "price")

        pd.testing.assert_frame_equal(train, train_orig)
        pd.testing.assert_frame_equal(test, test_orig)
