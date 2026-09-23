"""Scaler specs: what a model stores in `scaler` and what the worker builds from it."""

import pytest
import yaml

from epftoolbox2.models import LassoCVModel, OLSModel
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.scalers import AsinhScaler, NPITScaler, StandardScaler, make_scaler, scaler_spec


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("standard", "standard"),
        ("asinh", "asinh"),
        ("npit", "npit"),
        (StandardScaler(), "standard"),
        (AsinhScaler(), {"name": "asinh", "params": {"apply_to": "all"}}),
        (NPITScaler(apply_to=("price*",)), {"name": "npit", "params": {"apply_to": ["price*"]}}),
        ({"name": "asinh"}, "asinh"),
        ({"name": "asinh", "params": {"apply_to": "target"}}, {"name": "asinh", "params": {"apply_to": "target"}}),
    ],
)
def test_scaler_spec_is_yaml_safe(spec, expected):
    result = scaler_spec(spec)
    assert result == expected
    assert yaml.safe_load(yaml.dump(result)) == expected


@pytest.mark.parametrize(
    "spec, cls",
    [("standard", StandardScaler), ("asinh", AsinhScaler), ({"name": "npit", "params": {}}, NPITScaler)],
)
def test_make_scaler_builds_a_fresh_instance(spec, cls):
    a, b = make_scaler(spec), make_scaler(spec)
    assert type(a) is cls and a is not b


def test_make_scaler_from_instance_copies_its_parameters():
    original = NPITScaler(apply_to=["price*"])
    rebuilt = make_scaler(original)
    assert rebuilt is not original and rebuilt.apply_to == ["price*"]
    rebuilt.apply_to.append("load*")
    assert original.apply_to == ["price*"]


@pytest.mark.parametrize(
    "spec, error",
    [
        ("boxcox", ValueError),
        ({"name": "asinh", "params": {"shift": 1}}, ValueError),
        ({"name": "asinh", "params": {"apply_to": "prices"}}, ValueError),
        ({"class": "AsinhScaler"}, ValueError),
        ({"name": "asinh", "extra": 1}, ValueError),
        (None, TypeError),
        (42, TypeError),
    ],
)
def test_invalid_specs_fail_when_the_model_is_built(spec, error):
    with pytest.raises(error):
        OLSModel(["load"], scaler=spec)


def test_unregistered_scaler_subclass_is_rejected():
    class MyScaler(StandardScaler):
        pass

    with pytest.raises(ValueError, match="not a registered scaler"):
        scaler_spec(MyScaler())


def test_models_default_to_standard_scaler():
    assert OLSModel(["load"]).scaler == "standard"
    assert LassoCVModel(["load"], cv=3).scaler == "standard"


def test_scaler_survives_yaml_round_trip(tmp_path):
    spec = {"name": "npit", "params": {"apply_to": ["price", "price_d-*"]}}
    pipeline = (
        ModelPipeline()
        .add_model(OLSModel(["price_d-1", "load_d+{horizon}"], name="OLS npit", scaler=spec))
        .add_model(LassoCVModel(["price_d-1"], cv=3, name="Lasso asinh", scaler=AsinhScaler()))
        .add_model(OLSModel(["price_d-1"], name="OLS"))
    )
    path = tmp_path / "mp.yaml"
    pipeline.save(path)
    saved = yaml.safe_load(path.read_text())
    loaded = ModelPipeline.load(path)

    assert saved["models"][0]["params"]["scaler"] == spec
    assert [m.scaler for m in loaded.models] == [
        spec,
        {"name": "asinh", "params": {"apply_to": "all"}},
        "standard",
    ]


def test_yaml_without_scaler_loads_the_standard_scaler(tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text(
        yaml.dump({"models": [{"class": "OLSModel", "params": {"predictors": ["load"], "training_window": 30}}]})
    )
    assert ModelPipeline.load(path).models[0].scaler == "standard"
