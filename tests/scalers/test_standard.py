import pytest
import numpy as np
from epftoolbox2.scalers import StandardScaler


class TestStandardScaler:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        train_x = np.column_stack(
            [
                np.random.randn(100) * 1000 + 5000,  # load
                np.random.randn(100) * 5 + 15,  # temperature
                np.random.choice([0, 1], 100),  # is_holiday (binary)
            ]
        )
        train_y = np.random.randn(100) * 10 + 50  # price
        test_x = np.array([[5500.0, 18.0, 1]])
        return train_x, train_y, test_x

    def test_fit_transform_scales_continuous(self, sample_data):
        train_x, train_y, test_x = sample_data
        scaler = StandardScaler()

        train_x_scaled, train_y_scaled, test_x_scaled = scaler.fit_transform(train_x, train_y, test_x)

        # load column (index 0) should be scaled
        assert abs(train_x_scaled[:, 0].mean()) < 0.1
        assert abs(train_x_scaled[:, 0].std() - 1.0) < 0.1
        # temperature column (index 1) should be scaled
        assert abs(train_x_scaled[:, 1].mean()) < 0.1

    def test_fit_transform_skips_binary(self, sample_data):
        train_x, train_y, test_x = sample_data
        scaler = StandardScaler()

        train_x_scaled, _, _ = scaler.fit_transform(train_x, train_y, test_x)

        # is_holiday column (index 2) should NOT be scaled - values should remain 0/1
        unique = np.unique(train_x_scaled[:, 2])
        assert set(unique).issubset({0, 1})
        assert train_x_scaled[:, 2].mean() == train_x[:, 2].mean()

    def test_inverse_restores_target(self, sample_data):
        train_x, train_y, test_x = sample_data
        scaler = StandardScaler()

        scaler.fit_transform(train_x, train_y, test_x)

        original_mean = train_y.mean()
        restored = scaler.inverse(0.0)

        assert abs(restored - original_mean) < 0.01

    def test_handles_zero_std(self):
        train_x = np.array([[5.0]] * 10)
        train_y = np.array([1.0] * 10)
        test_x = np.array([[5.0]])

        scaler = StandardScaler()
        train_x_scaled, _, _ = scaler.fit_transform(train_x, train_y, test_x)

        assert not np.isnan(train_x_scaled).any()

    def test_does_not_modify_original(self, sample_data):
        train_x, train_y, test_x = sample_data
        train_x_orig = train_x.copy()
        train_y_orig = train_y.copy()
        test_x_orig = test_x.copy()

        scaler = StandardScaler()
        scaler.fit_transform(train_x, train_y, test_x)

        np.testing.assert_array_equal(train_x, train_x_orig)
        np.testing.assert_array_equal(train_y, train_y_orig)
        np.testing.assert_array_equal(test_x, test_x_orig)

    def test_get_scalable_mask(self):
        # Mix of continuous and binary columns
        arr = np.column_stack(
            [
                np.random.randn(100),  # continuous - should be True
                np.random.choice([0, 1], 100),  # binary - should be False
                np.array([0.0] * 100),  # constant 0 - should be False
                np.array([1.0] * 100),  # constant 1 - should be False
                np.arange(100, dtype=float),  # continuous - should be True
            ]
        )

        mask = StandardScaler.get_scalable_mask(arr)

        assert mask[0] == True  # continuous
        assert mask[1] == False  # binary
        assert mask[2] == False  # constant 0
        assert mask[3] == False  # constant 1
        assert mask[4] == True  # continuous
