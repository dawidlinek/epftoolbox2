"""MirrorLogScaler and PolyScaler: the transformations of Uniejewski, Weron & Ziel (2018), after AsinhScaler's normalisation.

The shared VST behaviour (training rows only, `apply_to`, binary columns, the near-constant rule, `standardise=False`) is in
test_vst.py, which runs every test there on these two as well. Here: the formulas against the paper's (as the regimes research's
`scripts/rg.py` wrote them, transcribed below), round trips within 1e-14, monotone inverses defined everywhere, the parameters,
and YAML.
"""

import numpy as np
import pytest
import yaml

from epftoolbox2.models import LEARModel, OLSModel
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.scalers import SCALERS, AsinhScaler, MirrorLogScaler, PolyScaler, make_scaler, scaler_spec

# The paper's equations, as `epf2_modeling_research/regimes/scripts/rg.py` (`_fwd`, `_bwd`) has them, on normalised x.
MC = 1.0 / 3.0
PL, PC = 0.125, 0.05
PA = (PC / PL) ** (1.0 / (PL - 1.0))
PB = (PC / PL) ** (PL / (PL - 1.0))


def paper_forward(kind, x):
    if kind == "mlog":
        return np.sign(x) * (np.log(np.abs(x) + 1.0 / MC) + np.log(MC))
    return np.sign(x) * ((np.abs(x) + PA) ** PL - PB)


def paper_inverse(kind, z):
    if kind == "mlog":
        return np.sign(z) * (np.exp(np.abs(z) - np.log(MC)) - 1.0 / MC)
    return np.sign(z) * (np.maximum(np.abs(z) + PB, 0.0) ** (1.0 / PL) - PA)


SCALERS_BY_KIND = {"mlog": MirrorLogScaler, "poly": PolyScaler}
NORMALISED = (0.0, 1.0)  # (median, scale) that leaves x as it is: the transformation alone
GRID = np.concatenate(
    [np.linspace(-1000.0, 1000.0, 20001), np.geomspace(1e-12, 1e4, 400), -np.geomspace(1e-12, 1e4, 400), [0.0]]
)


def relative_to_one(a, b):
    """|a - b| over max(1, |b|): absolute near the median, relative in the tails."""
    return np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)))


@pytest.fixture
def prices():
    """Spiky prices with negative hours, a load-like predictor and a dummy."""
    rng = np.random.default_rng(11)
    n = 300
    price = rng.normal(55, 18, n)
    price[rng.choice(n, 12, replace=False)] = rng.uniform(250, 3000, 12)
    price[rng.choice(n, 12, replace=False)] = rng.uniform(-500, -1, 12)
    load = rng.normal(15000, 2500, n)
    dummy = rng.integers(0, 2, n).astype(float)
    return np.column_stack([price, load, dummy]), np.roll(price, -1), np.array([[-120.0, 17000.0, 0.0]])


@pytest.mark.parametrize("kind", ["mlog", "poly"])
def test_the_transformation_is_the_paper_s(kind):
    scaler = SCALERS_BY_KIND[kind]()
    assert relative_to_one(scaler._forward(NORMALISED, GRID), paper_forward(kind, GRID)) < 2e-15
    z = np.linspace(-12.0, 12.0, 2401)
    assert relative_to_one(scaler._backward(NORMALISED, z), paper_inverse(kind, z)) < 4e-15


def test_the_parameters_are_the_paper_s_and_enter_the_formulas():
    assert MirrorLogScaler().c == 1.0 / 3.0
    assert (PolyScaler().lam, PolyScaler().c) == (0.125, 0.05)
    x = np.array([-7.0, -0.5, 0.5, 7.0])
    # mirror-log with c: sgn(x) (log(|x| + 1/c) + log c)
    c = 0.8
    expected = np.sign(x) * (np.log(np.abs(x) + 1 / c) + np.log(c))
    np.testing.assert_allclose(MirrorLogScaler(c=c)._forward(NORMALISED, x), expected, rtol=2e-15)
    # poly with (lam, c): sgn(x) ((|x| + (c/lam)^(1/(lam-1)))^lam - (c/lam)^(lam/(lam-1)))
    lam, c = 0.4, 0.3
    a, b = (c / lam) ** (1 / (lam - 1)), (c / lam) ** (lam / (lam - 1))
    expected = np.sign(x) * ((np.abs(x) + a) ** lam - b)
    np.testing.assert_allclose(PolyScaler(lam=lam, c=c)._forward(NORMALISED, x), expected, rtol=4e-15)
    assert not np.allclose(PolyScaler(lam=lam, c=c)._forward(NORMALISED, x), PolyScaler()._forward(NORMALISED, x))


@pytest.mark.parametrize("kind", ["mlog", "poly"])
def test_the_transformation_is_odd_zero_at_the_median_with_slope_c(kind):
    scaler = SCALERS_BY_KIND[kind]()
    positive = GRID[GRID > 0]
    np.testing.assert_array_equal(scaler._forward(NORMALISED, -positive), -scaler._forward(NORMALISED, positive))
    assert scaler._forward(NORMALISED, np.array([0.0]))[0] == 0.0
    assert scaler._backward(NORMALISED, np.array([0.0]))[0] == 0.0
    slope = scaler._forward(NORMALISED, np.array([1e-9]))[0] / 1e-9
    assert slope == pytest.approx(scaler.c, rel=1e-8)


@pytest.mark.parametrize("kind", ["mlog", "poly"])
def test_the_transformation_round_trips_within_1e_14(kind):
    scaler = SCALERS_BY_KIND[kind]()
    back = scaler._backward(NORMALISED, scaler._forward(NORMALISED, GRID))
    assert relative_to_one(back, GRID) < 1e-14
    tiny = np.geomspace(1e-300, 1e-6, 50)
    np.testing.assert_allclose(scaler._backward(NORMALISED, scaler._forward(NORMALISED, tiny)), tiny, rtol=1e-14)


@pytest.mark.parametrize("cls", [MirrorLogScaler, PolyScaler])
@pytest.mark.parametrize("standardise", [True, False])
def test_the_scaler_round_trips_the_target_within_1e_14(cls, standardise, prices):
    train_x, train_y, test_x = prices
    scaler = cls(standardise=standardise)
    _, y_scaled, test_scaled = scaler.fit_transform(train_x, train_y, test_x)

    back = scaler.inverse(y_scaled)
    np.testing.assert_allclose(back, train_y, rtol=1e-14, atol=1e-14 * np.abs(train_y).max())
    # a transformed predictor maps back through its own column's parameters
    params, mean, std = scaler._column_params[0]
    price_back = scaler._backward(params, test_scaled[0, 0] * std + mean)
    assert price_back == pytest.approx(-120.0, rel=1e-14)


@pytest.mark.parametrize("cls", [MirrorLogScaler, PolyScaler])
def test_the_forward_map_and_the_inverse_are_strictly_increasing_everywhere(cls, prices):
    train_x, train_y, test_x = prices
    scaler = cls()
    scaler.fit_transform(train_x, train_y, test_x)
    params = scaler._target_params

    values = np.linspace(-5000.0, 20000.0, 250001)
    assert np.all(np.diff(scaler._forward(params, values)) > 0)
    z = np.linspace(-25.0, 25.0, 500001)  # far beyond any standardised training value
    restored = scaler.inverse(z)
    assert np.isfinite(restored).all() and np.all(np.diff(restored) > 0)
    assert scaler.inverse(-50.0) < train_y.min() and scaler.inverse(50.0) > train_y.max()  # not clamped, unlike N-PIT


@pytest.mark.parametrize("cls", [MirrorLogScaler, PolyScaler])
def test_the_normalisation_is_asinh_s_median_and_mad(cls, prices):
    train_x, train_y, test_x = prices
    asinh, other = AsinhScaler(), cls()
    asinh.fit_transform(train_x, train_y, test_x)
    other.fit_transform(train_x, train_y, test_x)
    assert other._target_params == asinh._target_params
    assert [p[0] for p in other._column_params[:2]] == [p[0] for p in asinh._column_params[:2]]
    assert other._column_params[2] is None  # the dummy

    median, mad = np.median(train_y), np.median(np.abs(train_y - np.median(train_y)))
    assert other._target_params == pytest.approx((median, mad / 0.6744897501960817), rel=1e-15)


@pytest.mark.parametrize("cls", [MirrorLogScaler, PolyScaler])
def test_the_mad_fallback_is_asinh_s(cls):
    train_y = np.array([5.0, 5.0, 5.0, 5.0, 6.0, 9.0])
    scaler = cls()
    _, y_scaled, _ = scaler.fit_transform(np.empty((6, 0)), train_y, np.empty((1, 0)))
    assert scaler._target_params == pytest.approx((5.0, np.std(train_y, ddof=1)))
    np.testing.assert_allclose(scaler.inverse(y_scaled), train_y, rtol=1e-14)


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: MirrorLogScaler(c=0), "c"),
        (lambda: MirrorLogScaler(c=-1.0), "c"),
        (lambda: MirrorLogScaler(c=float("nan")), "c"),
        (lambda: MirrorLogScaler(c=True), "c"),
        (lambda: MirrorLogScaler(c="1/3"), "c"),
        (lambda: PolyScaler(lam=0), "lam"),
        (lambda: PolyScaler(lam=1), "lam"),
        (lambda: PolyScaler(lam=float("inf")), "lam"),
        (lambda: PolyScaler(c=0.0), "c"),
        (lambda: PolyScaler(c=None), "c"),
        (lambda: PolyScaler(apply_to="prices"), "apply_to"),
    ],
)
def test_invalid_parameters_are_refused(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_the_registry_offers_both():
    assert SCALERS["mlog"] is MirrorLogScaler and SCALERS["poly"] is PolyScaler
    assert type(make_scaler("mlog")) is MirrorLogScaler and type(make_scaler("poly")) is PolyScaler
    assert scaler_spec("mlog") == "mlog" and scaler_spec("poly") == "poly"


def test_defaults_are_left_out_of_the_spec_and_other_values_kept():
    assert MirrorLogScaler().get_params() == {"apply_to": "all"} == MirrorLogScaler(c=1 / 3).get_params()
    assert PolyScaler().get_params() == {"apply_to": "all"} == PolyScaler(lam=0.125, c=0.05).get_params()
    assert MirrorLogScaler(c=0.5).get_params() == {"apply_to": "all", "c": 0.5}
    assert PolyScaler(lam=0.25, standardise=False).get_params() == {
        "apply_to": "all",
        "standardise": False,
        "lam": 0.25,
    }
    assert scaler_spec({"name": "poly", "params": {"c": 0.1, "apply_to": ["price*"]}}) == {
        "name": "poly",
        "params": {"apply_to": ["price*"], "c": 0.1},
    }
    rebuilt = make_scaler(PolyScaler(lam=0.5, c=0.2))
    assert (rebuilt.lam, rebuilt.c) == (0.5, 0.2)
    assert OLSModel(["x"], scaler=MirrorLogScaler(c=2)).scaler == {
        "name": "mlog",
        "params": {"apply_to": "all", "c": 2.0},
    }


def test_both_survive_a_yaml_round_trip(tmp_path):
    specs = [
        "mlog",
        "poly",
        {"name": "mlog", "params": {"apply_to": ["price", "price_d-*"], "c": 0.5}},
        {"name": "poly", "params": {"apply_to": "target", "lam": 0.25, "c": 0.1, "standardise": False}},
    ]
    pipeline = ModelPipeline()
    for i, spec in enumerate(specs):
        pipeline.add_model(OLSModel(["price_d-1", "load_d+{horizon}"], name=f"OLS {i}", scaler=spec))
    pipeline.add_model(LEARModel(["price_d-1"], scaler="poly", name="LEAR_poly"))
    path = tmp_path / "mp.yaml"
    pipeline.save(path)
    saved = yaml.safe_load(path.read_text())
    loaded = ModelPipeline.load(path)

    expected = [
        "mlog",
        "poly",
        {"name": "mlog", "params": {"apply_to": ["price", "price_d-*"], "c": 0.5}},
        {"name": "poly", "params": {"apply_to": "target", "standardise": False, "lam": 0.25, "c": 0.1}},
        "poly",
    ]
    assert [m["params"]["scaler"] for m in saved["models"]] == expected
    assert [m.scaler for m in loaded.models] == expected
    rebuilt = make_scaler(loaded.models[3].scaler)
    assert (rebuilt.lam, rebuilt.c, rebuilt.standardise, rebuilt.apply_to) == (0.25, 0.1, False, "target")
