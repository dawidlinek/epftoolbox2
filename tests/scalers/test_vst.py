"""AsinhScaler and NPITScaler: fitted on training rows only, invertible, defined for negative prices."""

import warnings

import numpy as np
import pytest

from epftoolbox2.scalers import AsinhScaler, NPITScaler, StandardScaler

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
