"""AsinhScaler and NPITScaler: fitted on training rows only, invertible, defined for negative prices."""

import warnings

import numpy as np
import pytest

from epftoolbox2.models import LassoCVModel, OLSModel
from epftoolbox2.scalers import AsinhScaler, NPITScaler, StandardScaler
from epftoolbox2.scalers.base import CONSTANT_STD_RTOL

VST_SCALERS = [AsinhScaler, NPITScaler]
MAD_TO_STD = 1.482602218505602  # 1 / Phi^-1(0.75)


@pytest.fixture
def spiky_prices():
    """Prices with negative hours and spikes, plus a load-like predictor and a binary dummy."""
    rng = np.random.default_rng(7)
    n = 200
    price = rng.normal(60, 15, n)
    price[rng.choice(n, 10, replace=False)] = rng.uniform(300, 900, 10)
    price[rng.choice(n, 10, replace=False)] = rng.uniform(-80, -5, 10)
    load = rng.normal(15000, 2000, n)
    dummy = rng.integers(0, 2, n).astype(float)
    train_x = np.column_stack([price, load, dummy])
    train_y = np.roll(price, -1)
    test_x = np.array([[-40.0, 16000.0, 1.0]])
    return train_x, train_y, test_x


def fitted_state(scaler):
    return {
        "columns": [None if p is None else (p[0], p[1], p[2]) for p in scaler._column_params],
        "target": scaler._target_params,
        "target_mean": scaler._target_mean,
        "target_std": scaler._target_std,
    }


def assert_same_state(a, b):
    assert len(a["columns"]) == len(b["columns"])
    for col_a, col_b in zip(a["columns"], b["columns"]):
        if col_a is None:
            assert col_b is None
            continue
        np.testing.assert_array_equal(np.asarray(col_a[0], dtype=float), np.asarray(col_b[0], dtype=float))
        assert col_a[1:] == col_b[1:]
    np.testing.assert_array_equal(np.asarray(a["target"], dtype=float), np.asarray(b["target"], dtype=float))
    assert (a["target_mean"], a["target_std"]) == (b["target_mean"], b["target_std"])


@pytest.mark.parametrize("scaler_cls", VST_SCALERS)
def test_target_round_trips_through_inverse(scaler_cls, spiky_prices):
    train_x, train_y, test_x = spiky_prices
    scaler = scaler_cls()
    _, y_scaled, _ = scaler.fit_transform(train_x, train_y, test_x)

    np.testing.assert_allclose(scaler.inverse(y_scaled), train_y, rtol=1e-12, atol=1e-9)
    assert isinstance(scaler.inverse(float(y_scaled[3])), float)


@pytest.mark.parametrize("scaler_cls", VST_SCALERS)
def test_transformed_columns_are_standardised_and_dummies_untouched(scaler_cls, spiky_prices):
    train_x, train_y, test_x = spiky_prices
    x_scaled, y_scaled, test_scaled = scaler_cls().fit_transform(train_x, train_y, test_x)

    for values in (x_scaled[:, 0], x_scaled[:, 1], y_scaled):
        assert values.mean() == pytest.approx(0.0, abs=1e-12)
        assert values.std(ddof=1) == pytest.approx(1.0)
    np.testing.assert_array_equal(x_scaled[:, 2], train_x[:, 2])
    assert test_scaled[0, 2] == 1.0
    assert np.isfinite(x_scaled).all() and np.isfinite(test_scaled).all()


@pytest.mark.parametrize("scaler_cls", VST_SCALERS)
def test_fitted_parameters_ignore_the_test_row(scaler_cls, spiky_prices):
    train_x, train_y, test_x = spiky_prices
    base = scaler_cls()
    x_a, y_a, test_a = base.fit_transform(train_x, train_y, test_x)

    perturbed = scaler_cls()
    x_b, y_b, test_b = perturbed.fit_transform(train_x, train_y, np.array([[5000.0, -1e6, 0.0]]))

    assert_same_state(fitted_state(base), fitted_state(perturbed))
    np.testing.assert_array_equal(x_a, x_b)
    np.testing.assert_array_equal(y_a, y_b)
    assert not np.array_equal(test_a, test_b)


@pytest.mark.parametrize("scaler_cls", VST_SCALERS)
def test_inputs_are_not_modified(scaler_cls, spiky_prices):
    train_x, train_y, test_x = spiky_prices
    copies = [a.copy() for a in spiky_prices]
    scaler_cls().fit_transform(train_x, train_y, test_x)
    for original, copy in zip(spiky_prices, copies):
        np.testing.assert_array_equal(original, copy)


@pytest.mark.parametrize("scaler_cls", VST_SCALERS)
def test_negative_prices_are_finite_and_invertible(scaler_cls):
    train_y = np.array([-500.0, -120.0, -3.0, 0.0, 0.0, 4.0, 35.0, 80.0, 120.0, 2500.0])
    train_x = train_y.reshape(-1, 1).copy()
    scaler = scaler_cls()
    x_scaled, y_scaled, test_scaled = scaler.fit_transform(train_x, train_y, np.array([[-250.0]]))

    assert np.isfinite(x_scaled).all() and np.isfinite(y_scaled).all() and np.isfinite(test_scaled).all()
    assert np.all(np.diff(y_scaled) >= 0)  # monotone: the order of prices is kept
    np.testing.assert_allclose(scaler.inverse(y_scaled), train_y, atol=1e-9)
    assert scaler.inverse(float(test_scaled[0, 0])) == pytest.approx(-250.0)


def test_asinh_known_answer():
    train_y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    scaler = AsinhScaler()
    _, y_scaled, _ = scaler.fit_transform(np.empty((5, 0)), train_y, np.empty((1, 0)))

    # median 0, MAD 1 -> scale 1 / 0.6745; asinh(0.6745) = 0.63164..., asinh(1.3490) = 1.10796...
    assert scaler._target_params == pytest.approx((0.0, MAD_TO_STD), rel=1e-15)
    z = np.array([-1.1079648640353739, -0.6316425616165524, 0.0, 0.6316425616165524, 1.1079648640353739])
    np.testing.assert_allclose(y_scaled, z / 0.9018199558621568, rtol=1e-12)
    assert scaler.inverse(0.0) == pytest.approx(0.0, abs=1e-12)  # the transformed mean maps to the median


def test_npit_known_answer():
    scaler = NPITScaler()
    train_y = np.array([30.0, 10.0, 20.0])
    x_scaled, y_scaled, test_scaled = scaler.fit_transform(train_y.reshape(-1, 1), train_y, np.array([[25.0]]))

    # F = 1/4, 2/4, 3/4 -> z = -0.6745, 0, 0.6745 -> standardised -1, 0, 1 (in input order 30, 10, 20)
    np.testing.assert_allclose(y_scaled, [1.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(x_scaled[:, 0], [1.0, -1.0, 0.0], atol=1e-12)
    assert test_scaled[0, 0] == pytest.approx(0.5)  # halfway between the 20 and 30 knots
    assert scaler.inverse(0.5) == pytest.approx(25.0)
    # Beyond the extreme knots the forecast is clamped to the training range.
    assert scaler.inverse(5.0) == 30.0
    assert scaler.inverse(-5.0) == 10.0


def test_npit_ties_share_their_mid_rank():
    scaler = NPITScaler()
    scaler.fit_transform(np.empty((4, 0)), np.array([10.0, 20.0, 10.0, 30.0]), np.empty((1, 0)))

    knots_x, knots_z = scaler._target_params
    np.testing.assert_array_equal(knots_x, [10.0, 20.0, 30.0])
    # positions 1,2 -> mid-rank 1.5 -> F = 0.3; 20 -> 3/5; 30 -> 4/5
    np.testing.assert_allclose(knots_z, [-0.5244005127080407, 0.2533471031357998, 0.8416212335729144], rtol=1e-12)


def test_asinh_zero_mad_falls_back_to_standard_deviation():
    train_y = np.array([5.0, 5.0, 5.0, 5.0, 6.0, 9.0])
    scaler = AsinhScaler()
    _, y_scaled, _ = scaler.fit_transform(np.empty((6, 0)), train_y, np.empty((1, 0)))

    assert scaler._target_params == pytest.approx((5.0, np.std(train_y, ddof=1)))
    assert np.isfinite(y_scaled).all()
    np.testing.assert_allclose(scaler.inverse(y_scaled), train_y, atol=1e-12)


@pytest.mark.parametrize("scaler_cls", VST_SCALERS)
def test_constant_columns_stay_finite(scaler_cls):
    train_x = np.full((10, 1), 42.0)
    train_y = np.full(10, 7.0)
    scaler = scaler_cls()
    x_scaled, y_scaled, test_scaled = scaler.fit_transform(train_x, train_y, np.array([[43.0]]))

    assert np.isfinite(x_scaled).all() and np.isfinite(y_scaled).all() and np.isfinite(test_scaled).all()
    assert scaler.inverse(0.0) == pytest.approx(7.0)


@pytest.mark.parametrize("scaler_cls", VST_SCALERS)
def test_apply_to_target_standardises_predictors_like_standard_scaler(scaler_cls, spiky_prices):
    train_x, train_y, test_x = spiky_prices
    x_vst, _, test_vst = scaler_cls(apply_to="target").fit_transform(train_x, train_y, test_x)
    x_std, _, test_std = StandardScaler().fit_transform(train_x, train_y, test_x)

    np.testing.assert_allclose(x_vst, x_std, rtol=1e-12)
    np.testing.assert_allclose(test_vst, test_std, rtol=1e-12)


@pytest.mark.parametrize("scaler_cls", VST_SCALERS)
def test_apply_to_patterns_select_columns_by_name(scaler_cls, spiky_prices):
    train_x, train_y, test_x = spiky_prices
    columns = ["price_d-1", "load_d+1", "is_holiday_d+1"]
    scaler = scaler_cls(apply_to=["price", "price_d-*", "is_holiday*"])
    x_scaled, _, _ = scaler.fit_transform(train_x, train_y, test_x, columns=columns)
    x_all, _, _ = scaler_cls().fit_transform(train_x, train_y, test_x)
    x_std, _, _ = StandardScaler().fit_transform(train_x, train_y, test_x)

    np.testing.assert_allclose(x_scaled[:, 0], x_all[:, 0], rtol=1e-12)  # price lag: transformed
    np.testing.assert_allclose(x_scaled[:, 1], x_std[:, 1], rtol=1e-12)  # load: standardised only
    np.testing.assert_array_equal(x_scaled[:, 2], train_x[:, 2])  # binary: never transformed


def test_apply_to_patterns_that_match_nothing_warn():
    x = np.arange(20.0).reshape(10, 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AsinhScaler(apply_to=["pirce*"]).fit_transform(x, np.arange(10.0), x[:1], columns=["price_x1", "load_x1"])
        AsinhScaler(apply_to=["price", "price_*"]).fit_transform(x, np.arange(10.0), x[:1], columns=["price_x2", "l"])
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1 and "pirce*" in messages[0]


def test_apply_to_patterns_need_column_names():
    x = np.arange(20.0).reshape(10, 2)
    with pytest.raises(ValueError, match="predictor name"):
        AsinhScaler(apply_to=["price*"]).fit_transform(x, np.arange(10.0), x[:1])


@pytest.mark.parametrize("bad", ["price*", "none", [], [1, 2], None])
def test_invalid_apply_to_is_rejected(bad):
    with pytest.raises(ValueError, match="apply_to"):
        AsinhScaler(apply_to=bad)


def test_apply_to_tuple_is_stored_as_list():
    assert NPITScaler(apply_to=("price", "price_d-*")).apply_to == ["price", "price_d-*"]


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


APPLY_TO = ["all", "target", ["load*"]]  # the near-constant column ("flat") is transformed, or only standardised


def scaler_for(scaler_cls, apply_to):
    return scaler_cls(apply_to=apply_to if apply_to != ["load*"] else ["flat*"])


@pytest.mark.parametrize("scaler_cls", VST_SCALERS)
@pytest.mark.parametrize("apply_to", APPLY_TO)
class TestNearConstantColumns:
    """A column constant up to rounding noise isn't divided by its standard deviation of about 1e-16."""

    columns = ["load", "flat"]

    def test_scale_is_one_and_the_test_row_is_ordinary(self, scaler_cls, apply_to):
        train_x, train_y, test_x = near_constant_case()
        flat = train_x[:, 1]
        assert 0 < np.std(flat, ddof=1) <= CONSTANT_STD_RTOL * np.abs(flat).max()

        scaler = scaler_for(scaler_cls, apply_to)
        scaled_train, _, scaled_test = scaler.fit_transform(train_x, train_y, test_x, columns=self.columns)

        assert scaler._column_params[1][2] == 1.0  # the standardising scale
        np.testing.assert_array_equal(scaled_train[:, 1], 0.0)  # not its centred noise
        assert abs(scaled_test[0, 1]) < 0.02  # 0.31 is 0.01 above the constant
        assert scaled_train[:, 0].std(ddof=1) == pytest.approx(1.0)  # the ordinary column is standardised

    @pytest.mark.parametrize("seed", range(3))
    def test_forecast_is_finite_and_ordinary(self, scaler_cls, apply_to, seed):
        train_x, train_y, test_x = near_constant_case(seed=seed)

        scaler = scaler_for(scaler_cls, apply_to)
        forecast = scaler.inverse(ols_forecast(*scaler.fit_transform(train_x, train_y, test_x, columns=self.columns)))
        alone = scaler_for(scaler_cls, apply_to if apply_to != ["load*"] else "target")
        reference = alone.inverse(
            ols_forecast(*alone.fit_transform(train_x[:, :1], train_y, test_x[:, :1], columns=["load"]))
        )

        assert np.isfinite(forecast)
        assert abs(forecast - reference) < 1e-9 * abs(reference)  # the column neither helps nor hurts
        assert abs(forecast - (2 * 52 + 5)) < 15

    def test_training_values_are_exactly_zero_and_nans_stay_nan(self, scaler_cls, apply_to):
        train_x, train_y, test_x = near_constant_case()
        train_x[[2, 11], 1] = np.nan

        scaler = scaler_for(scaler_cls, apply_to)
        scaled_train, _, scaled_test = scaler.fit_transform(train_x, train_y, test_x, columns=self.columns)

        assert np.isnan(scaled_train[[2, 11], 1]).all()
        present = np.delete(scaled_train[:, 1], [2, 11])
        assert present.shape == (28,) and (present == 0.0).all() and not np.signbit(present).any()
        assert np.isfinite(scaled_test).all() and abs(scaled_test[0, 1]) < 0.02  # the test row isn't zeroed

    @pytest.mark.parametrize("seed", range(3))
    def test_ols_coefficient_is_exactly_zero(self, scaler_cls, apply_to, seed):
        train_x, train_y, test_x = near_constant_case(seed=seed)
        scaler = scaler_for(scaler_cls, apply_to)

        _, coefficients = OLSModel(predictors=[])._fit_predict(
            *scaler.fit_transform(train_x, train_y, test_x, columns=self.columns)
        )

        assert coefficients[1] == 0.0

    @pytest.mark.parametrize("seed", range(3))
    def test_lasso_forecast_equals_the_frame_without_the_column(self, scaler_cls, apply_to, seed):
        train_x, train_y, test_x = near_constant_case(seed=seed)
        model = LassoCVModel(predictors=[], cv=3)

        scaler = scaler_for(scaler_cls, apply_to)
        forecast = scaler.inverse(
            model._fit_predict(*scaler.fit_transform(train_x, train_y, test_x, columns=self.columns))[0]
        )
        alone = scaler_for(scaler_cls, apply_to if apply_to != ["load*"] else "target")
        reference = alone.inverse(
            model._fit_predict(*alone.fit_transform(train_x[:, :1], train_y, test_x[:, :1], columns=["load"]))[0]
        )

        assert abs(forecast - reference) < 1e-9

    def test_exactly_constant_column_with_an_inexact_mean(self, scaler_cls, apply_to):
        train_x, train_y, test_x = near_constant_case(noise=0.0)

        scaler = scaler_for(scaler_cls, apply_to)
        scaled_train, _, scaled_test = scaler.fit_transform(train_x, train_y, test_x, columns=self.columns)

        assert scaler._column_params[1][2] == 1.0
        np.testing.assert_array_equal(scaled_train[:, 1], 0.0)
        assert abs(scaled_test[0, 1]) < 0.02

    def test_ratio_of_1e9_still_scales_normally(self, scaler_cls, apply_to):
        col = spread_column(1e-9)
        assert 5e-10 < np.std(col, ddof=1) / np.abs(col).max() < 2e-9
        train_x = np.column_stack([np.arange(40.0), col])
        test_x = np.array([[3.0, col.mean() + 2 * np.std(col, ddof=1)]])

        scaler = scaler_for(scaler_cls, apply_to)
        scaled_train, _, scaled_test = scaler.fit_transform(train_x, np.arange(40.0), test_x, columns=self.columns)

        assert scaler._column_params[1][2] != 1.0
        assert np.abs(scaled_train[:, 1]).max() > 1.0  # untouched by the rule: a real standardised column
        assert scaled_train[:, 1].mean() == pytest.approx(0.0, abs=1e-9)
        assert scaled_train[:, 1].std(ddof=1) == pytest.approx(1.0, rel=1e-5)
        assert np.isfinite(scaled_test).all()

    def test_all_zero_column(self, scaler_cls, apply_to):
        train_x = np.column_stack([np.arange(20.0), np.zeros(20)])
        scaler = scaler_for(scaler_cls, apply_to)

        scaled_train, _, scaled_test = scaler.fit_transform(
            train_x, np.arange(20.0), np.array([[3.0, 0.5]]), scalable_mask=np.array([True, True]), columns=self.columns
        )

        assert not scaled_train[:, 1].any()
        assert scaler._column_params[1][2] == 1.0
        assert np.isfinite(scaled_test).all()

    def test_one_finite_value(self, scaler_cls, apply_to):
        col = np.full(10, np.nan)
        col[4] = 0.3
        train_x = np.column_stack([np.arange(10.0), col])
        scaler = scaler_for(scaler_cls, apply_to)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scaled_train, _, scaled_test = scaler.fit_transform(
                train_x,
                np.arange(10.0),
                np.array([[3.0, 0.31]]),
                scalable_mask=np.array([True, True]),
                columns=self.columns,
            )

        assert scaler._column_params[1][2] == 1.0
        assert scaled_train[4, 1] == 0.0
        assert np.isfinite(scaled_test).all() and abs(scaled_test[0, 1]) < 0.02

    def test_nans_in_the_column(self, scaler_cls, apply_to):
        train_x, train_y, test_x = near_constant_case()
        train_x[[2, 11], 1] = np.nan
        train_x[[5, 7], 0] = np.nan

        scaler = scaler_for(scaler_cls, apply_to)
        scaled_train, _, scaled_test = scaler.fit_transform(train_x, train_y, test_x, columns=self.columns)

        assert scaler._column_params[1][2] == 1.0
        assert np.isfinite(np.delete(scaled_train[:, 1], [2, 11])).all()
        assert abs(scaled_test[0, 1]) < 0.02
        assert np.nanstd(scaled_train[:, 0], ddof=1) == pytest.approx(1.0)


@pytest.mark.parametrize("scaler_cls", VST_SCALERS)
class TestNearConstantTarget:
    def test_scale_is_one_and_the_inverse_stays_at_the_constant(self, scaler_cls):
        rng = np.random.default_rng(0)
        train_y = 0.3 + rng.normal(0, 1e-15, 30)
        scaler = scaler_cls()

        _, scaled_y, _ = scaler.fit_transform(rng.normal(size=(30, 1)), train_y, np.zeros((1, 1)))

        assert scaler._target_std == 1.0
        assert np.abs(scaled_y).max() < 1e-14  # the target keeps its centred noise: only predictors are zeroed
        for z in (-3.0, 0.0, 3.0):  # a model's prediction in the scaled space: an ordinary number maps back
            assert np.isfinite(scaler.inverse(z))
        assert scaler.inverse(0.0) == pytest.approx(0.3, abs=1e-12)

    def test_exactly_constant_target_with_an_inexact_mean(self, scaler_cls):
        scaler = scaler_cls()
        scaler.fit_transform(np.arange(30.0).reshape(-1, 1), np.full(30, 0.3), np.zeros((1, 1)))
        assert scaler._target_std == 1.0 and scaler.inverse(0.0) == pytest.approx(0.3, abs=1e-12)

    def test_target_with_ratio_of_1e9_is_scaled(self, scaler_cls):
        scaler = scaler_cls()
        _, scaled_y, _ = scaler.fit_transform(np.arange(40.0).reshape(-1, 1), spread_column(1e-9), np.zeros((1, 1)))

        assert scaler._target_std != 1.0
        assert scaled_y.std(ddof=1) == pytest.approx(1.0, rel=1e-5)

    def test_all_zero_one_value_and_nans(self, scaler_cls):
        x = np.arange(10.0).reshape(-1, 1)
        zeros = scaler_cls()
        zeros.fit_transform(x, np.zeros(10), np.zeros((1, 1)))
        assert zeros._target_std == 1.0 and zeros.inverse(0.0) == 0.0

        one = np.full(10, np.nan)
        one[3] = 7.0
        single = scaler_cls()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            single.fit_transform(x, one, np.zeros((1, 1)))
        assert single._target_std == 1.0

        noisy = 0.3 + np.random.default_rng(1).normal(0, 1e-15, 10)
        noisy[[1, 6]] = np.nan
        with_nans = scaler_cls()
        with_nans.fit_transform(x, noisy, np.zeros((1, 1)))
        assert with_nans._target_std == 1.0 and with_nans.inverse(0.0) == pytest.approx(0.3, abs=1e-12)
