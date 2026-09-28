import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.validators import PlausibilityValidator
from epftoolbox2.data.validators.plausibility import DEFAULT_RANGES

TZ = "Europe/Warsaw"


@pytest.fixture
def frame():
    index = pd.date_range("2024-01-01", periods=24 * 7, freq="1h", tz="UTC").tz_convert(TZ)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "price": rng.normal(80, 30, len(index)),
            "price_d-1": rng.normal(80, 30, len(index)),
            "load_actual": rng.normal(15000, 2000, len(index)),
            "is_weekend": (index.dayofweek >= 5).astype(float),
            "country": "PL",
        },
        index=index,
    )


def test_realistic_frame_is_valid(frame):
    result = PlausibilityValidator().validate(frame)

    assert result.is_valid, result.errors
    assert result.warnings == []
    assert set(result.info["checked_columns"]) == {"price", "price_d-1", "load_actual"}


def test_default_price_limits_are_sdac(frame):
    assert DEFAULT_RANGES["price*"] == [-600.0, 4000.0]
    frame.iloc[5, frame.columns.get_loc("price")] = 4000.5
    frame.iloc[9, frame.columns.get_loc("price_d-1")] = -650.0
    frame.iloc[11, frame.columns.get_loc("price")] = -550.0  # below the pre-2026 floor of -500, still allowed

    result = PlausibilityValidator().validate(frame)

    assert not result.is_valid
    assert result.info["out_of_range"] == {"price": 1, "price_d-1": 1}
    assert "'price' has 1 values outside [-600, 4000]" in result.errors[0]
    assert str(frame.index[5]) in result.errors[0]


def test_load_must_be_positive(frame):
    frame.iloc[3, frame.columns.get_loc("load_actual")] = 0.0

    result = PlausibilityValidator().validate(frame)

    assert not result.is_valid
    assert result.info["out_of_range"] == {"load_actual": 1}


def test_custom_ranges_replace_the_defaults(frame):
    frame["temperature"] = 20.0
    frame.iloc[0, frame.columns.get_loc("temperature")] = 80.0
    frame.iloc[0, frame.columns.get_loc("price")] = 9999.0

    result = PlausibilityValidator(ranges={"temp*": [-50, 60]}, max_constant_run=None).validate(frame)

    assert result.info["out_of_range"] == {"temperature": 1}
    assert PlausibilityValidator(ranges={}, max_constant_run=None).validate(frame).is_valid


def test_first_matching_pattern_wins(frame):
    frame.iloc[0, frame.columns.get_loc("price_d-1")] = 9999.0

    result = PlausibilityValidator(ranges={"price_d*": [None, None], "price*": [-600, 4000]}).validate(frame)

    assert result.is_valid
    rules = result.stats.set_index("column")["rule"]
    assert rules["price_d-1"] == "price_d*" and rules["price"] == "price*"


def test_unmatched_user_pattern_warns(frame):
    result = PlausibilityValidator(ranges={"wind*": [0, None]}).validate(frame)

    assert result.is_valid
    assert any("'wind*'" in w for w in result.warnings)


def test_stuck_price_run_is_a_warning(frame):
    frame.iloc[48:78, frame.columns.get_loc("price")] = 92.54  # 30 identical hours, like an interpolated outage

    result = PlausibilityValidator().validate(frame)

    assert result.is_valid
    assert result.info["longest_constant_run"]["price"] == 30
    assert len(result.warnings) == 1
    assert "'price' repeats 92.54 for 30 consecutive rows" in result.warnings[0]
    assert str(frame.index[48]) in result.warnings[0]


def test_constant_run_threshold_and_columns(frame):
    frame.iloc[48:72, frame.columns.get_loc("price")] = 50.0  # exactly 24

    assert PlausibilityValidator().validate(frame).warnings == []
    assert len(PlausibilityValidator(max_constant_run=23).validate(frame).warnings) == 1
    assert PlausibilityValidator(max_constant_run=None).validate(frame).warnings == []
    # is_weekend is constant for 48 hours but isn't checked unless asked for
    assert len(PlausibilityValidator(constant_columns=["is_*"]).validate(frame).warnings) == 1


def test_missing_values_break_constant_runs(frame):
    frame.iloc[48:78, frame.columns.get_loc("price")] = 50.0
    frame.iloc[60, frame.columns.get_loc("price")] = np.nan

    assert PlausibilityValidator().validate(frame).warnings == []


@pytest.mark.parametrize(
    "ranges",
    [{"price": [10, 0]}, {"price": [1, 2, 3]}, {"price": 5}, ["price"]],
)
def test_invalid_ranges_raise(ranges):
    with pytest.raises(ValueError, match="ranges"):
        PlausibilityValidator(ranges=ranges)


def test_invalid_max_constant_run_raises():
    with pytest.raises(ValueError, match="max_constant_run"):
        PlausibilityValidator(max_constant_run=0)


def test_tuples_are_stored_as_lists():
    assert PlausibilityValidator(ranges={"price": (None, 100)}).ranges == {"price": [None, 100.0]}


def test_nullable_integer_columns(frame):
    frame["load_actual"] = pd.array([15000] * (len(frame) - 2) + [None, 0], dtype="Int64")

    result = PlausibilityValidator().validate(frame)

    assert result.info["out_of_range"] == {"load_actual": 1}


def test_imputed_flag_columns_are_not_checked_for_constant_runs(frame):
    frame = frame.copy()
    frame["price_imputed"] = 0  # ImputeTransformer's flag: all zeros when nothing needed filling
    result = PlausibilityValidator().validate(frame)
    assert not any("price_imputed" in w for w in result.warnings + result.errors)
