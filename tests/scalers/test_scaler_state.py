"""`state()` / `from_state()` / `transform()` of every scaler (step 8d, stored fits): a fitted scaler as arrays, written by
`numpy.savez` and read back without pickle, transforms test rows and maps forecasts back to the bit of the scaler that was fitted,
and `transform` is what `fit_transform` did to its test rows, row by row too.
"""

import io
import warnings

import numpy as np
import pytest

from epftoolbox2.scalers import SCALERS, BaseScaler, StandardScaler, make_scaler

pytestmark = pytest.mark.usefixtures("no_network")

SPECS = [
    "standard",
    {"name": "standard", "params": {"scale_binary": True}},
    "asinh",
    {"name": "asinh", "params": {"standardise": False}},
    {"name": "asinh", "params": {"apply_to": "target"}},
    {"name": "asinh", "params": {"apply_to": ["a*"]}},
    "npit",
    {"name": "npit", "params": {"standardise": False}},
    "mlog",
    {"name": "mlog", "params": {"c": 0.2, "standardise": False}},
    "poly",
    {"name": "poly", "params": {"lam": 0.3, "c": 0.1, "apply_to": ["b*"]}},
]


def bits(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.int64)


def round_trip(state):
    """The state through `numpy.savez` and `numpy.load(allow_pickle=False)`, as a stored fit keeps it."""
    buffer = io.BytesIO()
    np.savez(buffer, **state)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def design(rng, n, p):
    """Spiky columns, a 0/1 dummy, a column constant up to rounding noise and one with ties, and a spiky target."""
    x = rng.standard_t(2, (n, p)) * 30
    if p > 1:
        x[:, 1] = rng.random(n) > 0.5
    if p > 2:
        x[:, 2] = 0.3 + np.arange(n) * 1e-17
    if p > 3:
        x[:, 3] = np.round(x[:, 3] / 40)
    y = rng.standard_t(2, n) * 50 + 40
    return x, y


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s if isinstance(s, str) else repr(s))
def test_a_stored_scaler_transforms_and_inverts_to_the_bit(spec):
    rng = np.random.default_rng(3)
    for _ in range(25):
        n, p = int(rng.integers(3, 80)), int(rng.integers(0, 7))
        x, y = design(rng, n, p)
        test = rng.standard_t(2, (5, p)) * 30
        columns = [f"a{i}" if i % 2 else f"b{i}" for i in range(p)]
        scaler = make_scaler(spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # apply_to patterns that match no column of a narrow design
            _, _, transformed = scaler.fit_transform(
                x, y, test, scalable_mask=BaseScaler.get_scalable_mask(x), columns=columns
            )
        restored = BaseScaler.from_state(round_trip(scaler.state()))
        assert type(restored) is type(scaler) and restored.get_params() == scaler.get_params()
        assert np.array_equal(bits(restored.transform(test)), bits(transformed))
        assert np.array_equal(bits(scaler.transform(test)), bits(transformed))
        # Row by row, as a forecast of one issue day transforms its row: the same bits as the block of rows.
        if p:
            rows = np.vstack([restored.transform(test[i : i + 1]) for i in range(len(test))])
            assert np.array_equal(bits(rows), bits(transformed))
        z = np.concatenate([rng.normal(0, 2, 20), [0.0, -0.0, 8.0, -8.0]])
        assert np.array_equal(
            bits([restored.inverse(float(v)) for v in z]), bits([scaler.inverse(float(v)) for v in z])
        )


def test_every_registered_scaler_has_a_state():
    """A scaler added to the registry without `state()` couldn't be stored: every one of them round trips."""
    x, y = design(np.random.default_rng(0), 40, 4)
    for name in SCALERS:
        scaler = make_scaler(name)
        scaler.fit_transform(x, y, x[:1], columns=["a", "b", "c", "d"])
        restored = BaseScaler.from_state(round_trip(scaler.state()))
        assert np.array_equal(bits(restored.transform(x[:3])), bits(scaler.transform(x[:3]))), name


def test_a_scaler_that_wasnt_fitted_has_no_state_and_no_transform():
    for name in SCALERS:
        with pytest.raises(ValueError, match="wasn't fitted"):
            make_scaler(name).state()
        with pytest.raises(ValueError, match="wasn't fitted"):
            make_scaler(name).transform(np.zeros((1, 2)))


def test_rows_of_another_width_and_another_class_are_refused():
    x, y = design(np.random.default_rng(1), 30, 3)
    scaler = make_scaler("asinh")
    scaler.fit_transform(x, y, x[:1])
    with pytest.raises(ValueError, match="fitted on 3 columns"):
        scaler.transform(np.zeros((1, 4)))
    with pytest.raises(ValueError, match="not of a StandardScaler"):
        StandardScaler.from_state(round_trip(scaler.state()))
    with pytest.raises(ValueError, match="no spec"):
        BaseScaler.from_state({})


def test_state_holds_arrays_only():
    """No object array, so `numpy.load(allow_pickle=False)` reads every state."""
    x, y = design(np.random.default_rng(2), 30, 4)
    for name in SCALERS:
        scaler = make_scaler(name)
        scaler.fit_transform(x, y, x[:1])
        for key, value in scaler.state().items():
            assert isinstance(value, np.ndarray) and value.dtype != object, (name, key)
