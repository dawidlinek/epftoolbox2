"""Every validator survives DataPipeline.save → load with its parameters."""

import pytest

from epftoolbox2.data.validators import (
    ContinuityValidator,
    CoverageValidator,
    DSTStructureValidator,
    EdaValidator,
    PlausibilityValidator,
)
from epftoolbox2.pipelines import DataPipeline

VALIDATORS = [
    ContinuityValidator(freq="15min"),
    EdaValidator(columns=["price"], verbose=False),
    DSTStructureValidator(tz="Europe/Warsaw", freq="1h", allow_partial_edges=False),
    CoverageValidator(
        columns=["price"], start="2024-01-01", end="today_d-1", freq="1h", min_coverage=0.99, max_gap="6h"
    ),
    PlausibilityValidator(
        ranges={"price": (-500, 3000), "load*": [0, None]}, max_constant_run=12, constant_columns=["price"]
    ),
    PlausibilityValidator(ranges={}, max_constant_run=None),
]


def public_params(component):
    return {k: v for k, v in vars(component).items() if not k.startswith("_") and k != "console"}


@pytest.mark.parametrize("validator", VALIDATORS, ids=lambda v: type(v).__name__)
def test_round_trip(tmp_path, validator):
    path = tmp_path / "dp.yaml"
    DataPipeline().add_validator(validator).save(path)

    loaded = DataPipeline.load(path).validators[0]

    assert type(loaded) is type(validator)
    assert public_params(loaded) == public_params(validator)
