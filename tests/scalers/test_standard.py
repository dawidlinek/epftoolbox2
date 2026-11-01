import warnings

import pytest
import numpy as np
from epftoolbox2.models import LassoCVModel, OLSModel
from epftoolbox2.scalers import StandardScaler
from epftoolbox2.scalers.base import CONSTANT_STD_RTOL, is_constant, mean_std


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

        assert mask[0]  # continuous
        assert not mask[1]  # binary
        assert not mask[2]  # constant 0
        assert not mask[3]  # constant 1
        assert mask[4]  # continuous


def near_constant_case(constant=0.3, noise=1e-15, n=30, seed=0):
    """A load-like column, a column that is `constant` up to rounding noise, a price that depends on the first."""
    rng = np.random.default_rng(seed)
    load = rng.normal(50, 10, n)
    flat = constant + rng.normal(0, noise, n)
    price = 2 * load + 5 + rng.normal(0, 3, n)
    return np.column_stack([load, flat]), price, np.array([[52.0, 0.31]])


def spread_column(ratio, centre=1000.0, n=40, seed=3):
    """Values around `centre` whose sample standard deviation is about `ratio` times their largest absolute value."""
    z = np.random.default_rng(seed).normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    return centre * (1 + ratio * z)


def ols_forecast(train_x, train_y, test_x):
    """What OLSModel does with the scaled data."""
    return OLSModel(predictors=[])._fit_predict(train_x, train_y, test_x)[0]


class TestNearConstantColumns:
    """A column constant up to rounding noise has a standard deviation of about 1e-16, not 0, and is not divided by it."""

    def test_scale_is_one(self):
        train_x, train_y, test_x = near_constant_case()
        flat = train_x[:, 1]
        assert 0 < np.std(flat, ddof=1) <= CONSTANT_STD_RTOL * np.abs(flat).max()

        scaled_train, _, scaled_test = StandardScaler().fit_transform(train_x, train_y, test_x)

        np.testing.assert_array_equal(scaled_train[:, 1], 0.0)  # not its centred noise
        assert scaled_test[0, 1] == (0.31 - np.mean(flat)) / 1.0  # the test row is centred and divided by 1.0
        assert scaled_train[:, 0].std(ddof=1) == pytest.approx(1.0)  # the ordinary column is standardised as before

    @pytest.mark.parametrize("seed", range(5))
    def test_forecast_is_finite_and_ordinary(self, seed):
        train_x, train_y, test_x = near_constant_case(seed=seed)

        scaler = StandardScaler()
        forecast = scaler.inverse(ols_forecast(*scaler.fit_transform(train_x, train_y, test_x)))
        alone = StandardScaler()
        reference = alone.inverse(ols_forecast(*alone.fit_transform(train_x[:, :1], train_y, test_x[:, :1])))

        assert np.isfinite(forecast)
        assert forecast == pytest.approx(reference, rel=1e-9)  # the column neither helps nor hurts
        assert abs(forecast - (2 * 52 + 5)) < 10  # 109 plus the price's own noise (sd 3) in the fit

    def test_exactly_constant_column_with_an_inexact_mean(self):
        # 0.3 thirty times: the mean is rounded, so numpy's standard deviation is about 5e-17, not 0.
        train_x, train_y, test_x = near_constant_case(noise=0.0)

        scaled_train, _, scaled_test = StandardScaler().fit_transform(train_x, train_y, test_x)

        np.testing.assert_array_equal(scaled_train[:, 1], 0.0)
        assert scaled_test[0, 1] == pytest.approx(0.01, abs=1e-15)

    @pytest.mark.parametrize("centre", [1e-3, 0.3, 1000.0, 1e9])
    def test_threshold_is_relative_to_the_largest_value(self, centre):
        # 5e-13 of the largest value is constant, 2e-12 is a real spread, at every magnitude.
        for ratio, scaled in ((0.5e-12, False), (2e-12, True)):
            train_x = np.column_stack([np.arange(40.0), spread_column(ratio, centre)])
            test_x = np.array([[3.0, train_x[:, 1].mean() + np.std(train_x[:, 1], ddof=1)]])
            scaled_train, _, scaled_test = StandardScaler().fit_transform(train_x, np.arange(40.0), test_x)
            assert bool(scaled_train[:, 1].std(ddof=1) == pytest.approx(1.0, rel=1e-2)) is scaled
            assert bool(scaled_test[0, 1] == pytest.approx(1.0, rel=1e-2)) is scaled

    def test_ratio_of_1e9_still_scales_normally(self):
        col = spread_column(1e-9)
        assert 5e-10 < np.std(col, ddof=1) / np.abs(col).max() < 2e-9
        test_x = np.array([[col.mean() + 2 * np.std(col, ddof=1)]])

        scaled_train, _, scaled_test = StandardScaler().fit_transform(col.reshape(-1, 1), np.arange(40.0), test_x)

        assert scaled_train[:, 0].mean() == pytest.approx(0.0, abs=1e-9)
        assert scaled_train[:, 0].std(ddof=1) == pytest.approx(1.0, rel=1e-5)
        assert scaled_test[0, 0] == pytest.approx(2.0, rel=1e-5)

    def test_small_values_with_a_real_spread_are_scaled(self):
        col = 1e-20 * np.random.default_rng(0).normal(size=50)  # tiny in absolute terms, not relative to itself
        scaled_train, _, _ = StandardScaler().fit_transform(col.reshape(-1, 1), np.arange(50.0), col[:1].reshape(1, 1))
        assert scaled_train[:, 0].std(ddof=1) == pytest.approx(1.0)

    def test_all_zero_column_stays_at_scale_one(self):
        train_x = np.column_stack([np.arange(20.0), np.zeros(20)])
        mask = np.array([True, True])  # an all-zero column is "binary" and unscaled unless a mask says otherwise

        scaled_train, _, scaled_test = StandardScaler().fit_transform(
            train_x, np.arange(20.0), np.array([[3.0, 0.5]]), scalable_mask=mask
        )

        assert not scaled_train[:, 1].any()
        assert scaled_test[0, 1] == 0.5

    def test_one_finite_value_has_no_spread_to_divide_by(self):
        col = np.full(10, np.nan)
        col[4] = 0.3
        train_x = np.column_stack([np.arange(10.0), col])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # numpy: degrees of freedom <= 0
            scaled_train, _, scaled_test = StandardScaler().fit_transform(
                train_x, np.arange(10.0), np.array([[3.0, 0.31]]), scalable_mask=np.array([True, True])
            )

        assert scaled_train[4, 1] == 0.0
        assert np.isnan(np.delete(scaled_train[:, 1], 4)).all()
        assert scaled_test[0, 1] == pytest.approx(0.01)

    def test_nans_are_ignored_by_the_rule(self):
        train_x, train_y, test_x = near_constant_case()
        train_x[[2, 11], 1] = np.nan
        train_x[[5, 7], 0] = np.nan
        scaled_train, _, scaled_test = StandardScaler().fit_transform(train_x, train_y, test_x)

        flat = train_x[:, 1]
        np.testing.assert_array_equal(scaled_train[:, 1], np.where(np.isnan(flat), np.nan, 0.0))  # NaN stays NaN
        assert scaled_test[0, 1] == (0.31 - np.nanmean(flat)) / 1.0
        assert np.nanstd(scaled_train[:, 0], ddof=1) == pytest.approx(1.0)  # an ordinary column with NaNs

    def test_target_constant_up_to_rounding_noise_has_scale_one(self):
        rng = np.random.default_rng(0)
        train_y = 0.3 + rng.normal(0, 1e-15, 30)
        scaler = StandardScaler()

        _, scaled_y, _ = scaler.fit_transform(rng.normal(size=(30, 1)), train_y, np.zeros((1, 1)))

        assert scaler._target_std == 1.0
        np.testing.assert_array_equal(scaled_y, train_y - np.mean(train_y))
        assert scaler.inverse(2.0) == pytest.approx(np.mean(train_y) + 2.0)

    def test_target_with_ratio_of_1e9_is_scaled(self):
        train_y = spread_column(1e-9)
        scaler = StandardScaler()

        _, scaled_y, _ = scaler.fit_transform(np.arange(40.0).reshape(-1, 1), train_y, np.zeros((1, 1)))

        assert scaler._target_std == pytest.approx(np.std(train_y, ddof=1))
        assert scaled_y.std(ddof=1) == pytest.approx(1.0, rel=1e-5)

    def test_target_all_zero_one_value_and_nans(self):
        x = np.arange(10.0).reshape(-1, 1)
        zeros = StandardScaler()
        zeros.fit_transform(x, np.zeros(10), np.zeros((1, 1)))
        assert zeros._target_std == 1.0 and zeros.inverse(0.5) == 0.5

        one = np.full(10, np.nan)
        one[3] = 7.0
        single = StandardScaler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            single.fit_transform(x, one, np.zeros((1, 1)))
        assert single._target_std == 1.0 and single._target_mean == 7.0

        noisy = 0.3 + np.random.default_rng(1).normal(0, 1e-15, 10)
        noisy[[1, 6]] = np.nan
        with_nans = StandardScaler()
        with_nans.fit_transform(x, noisy, np.zeros((1, 1)))
        assert with_nans._target_std == 1.0 and with_nans._target_mean == np.nanmean(noisy)


class TestConstantPredictorTrainsAsZeros:
    """The centred noise of a constant predictor isn't left to the solver to ignore: it is 0.0, so any solver gives
    the column the coefficient 0."""

    def test_training_values_are_exactly_zero_and_nans_stay_nan(self):
        train_x, train_y, test_x = near_constant_case()
        train_x[[2, 11], 1] = np.nan

        scaled_train, _, scaled_test = StandardScaler().fit_transform(train_x, train_y, test_x)

        assert np.isnan(scaled_train[[2, 11], 1]).all()
        present = np.delete(scaled_train[:, 1], [2, 11])
        assert present.shape == (28,) and (present == 0.0).all() and not np.signbit(present).any()
        assert scaled_test[0, 1] == (0.31 - np.nanmean(train_x[:, 1])) / 1.0  # the test row keeps its deviation

    @pytest.mark.parametrize("seed", range(5))
    def test_ols_coefficient_is_exactly_zero(self, seed):
        train_x, train_y, test_x = near_constant_case(seed=seed)
        _, coefficients = OLSModel(predictors=[])._fit_predict(
            *StandardScaler().fit_transform(train_x, train_y, test_x)
        )
        assert coefficients[1] == 0.0

    @pytest.mark.parametrize("seed", range(3))
    def test_lasso_forecast_equals_the_frame_without_the_column(self, seed):
        train_x, train_y, test_x = near_constant_case(seed=seed)
        model = LassoCVModel(predictors=[], cv=3)

        scaler = StandardScaler()
        forecast = scaler.inverse(model._fit_predict(*scaler.fit_transform(train_x, train_y, test_x))[0])
        alone = StandardScaler()
        reference = alone.inverse(model._fit_predict(*alone.fit_transform(train_x[:, :1], train_y, test_x[:, :1]))[0])

        assert abs(forecast - reference) < 1e-9

    def test_a_column_with_ratio_1e9_is_untouched(self):
        col = spread_column(1e-9)
        assert 5e-10 < np.std(col, ddof=1) / np.abs(col).max() < 2e-9

        scaled_train, _, _ = StandardScaler().fit_transform(col.reshape(-1, 1), np.arange(40.0), col[:1].reshape(1, 1))

        np.testing.assert_allclose(scaled_train[:, 0], (col - np.mean(col)) / np.std(col, ddof=1), rtol=1e-12)
        assert scaled_train.any()


class TestSharedRule:
    def test_mean_std_is_the_sample_statistics_unless_constant_or_undefined(self):
        col = np.array([1.0, 2.0, 4.0, np.nan])
        assert mean_std(col) == (np.nanmean(col), np.nanstd(col, ddof=1), False)
        assert mean_std(np.full(5, 0.3))[1:] == (1.0, True)
        assert mean_std(np.zeros(5)) == (0.0, 1.0, True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert mean_std(np.array([0.3, np.nan])) == (0.3, 1.0, False)  # undefined, not constant
            assert mean_std(np.array([0.3, np.inf]))[1:] == (1.0, False)  # undefined: infinity minus infinity

    def test_is_constant_needs_two_values_and_a_defined_spread(self):
        assert is_constant(np.full(30, 0.3))
        assert is_constant(0.3 + np.random.default_rng(0).normal(0, 1e-15, 30))
        assert is_constant(np.array([0.3, 0.3, np.nan]))
        assert not is_constant(np.array([0.3]))
        assert not is_constant(np.array([np.nan, np.nan]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert not is_constant(np.array([0.3, np.inf]))
        assert not is_constant(spread_column(1e-9))

    def test_is_constant_follows_the_definition_around_the_threshold(self):
        # The early exit on the range must never turn a constant column into a non-constant one.
        rng = np.random.default_rng(5)
        for _ in range(300):
            n = int(rng.integers(2, 60))
            centre = float(rng.choice([-1e6, -0.3, 1e-3, 1.0, 0.3, 42.0, 1e8]))
            ratio = float(10 ** rng.uniform(-15, -9))
            col = centre * (1 + ratio * rng.normal(size=n))
            if rng.random() < 0.3:
                col[rng.integers(0, n)] = np.nan
            if np.count_nonzero(~np.isnan(col)) < 2:
                continue
            finite = col[np.isfinite(col)]
            expected = bool(np.nanstd(col, ddof=1) <= CONSTANT_STD_RTOL * np.abs(finite).max())
            assert is_constant(col) is expected
