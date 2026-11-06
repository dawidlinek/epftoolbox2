"""Optional dependency groups: `has_extra` and the error that names the extra to install."""

import importlib.util
import pickle
import tomllib
from pathlib import Path

import pytest

from epftoolbox2._extras import EXTRAS, MissingExtraError, has_extra


def test_missing_extra_error_names_the_extra_and_pickles():
    error = MissingExtraError("TabPFNModel", "tabpfn")
    assert isinstance(error, ImportError)
    assert str(error) == "TabPFNModel needs: pip install 'epftoolbox2[tabpfn]'"
    copy = pickle.loads(pickle.dumps(error))
    assert (str(copy), copy.what, copy.extra) == (str(error), "TabPFNModel", "tabpfn")
    assert has_extra("fast") == (importlib.util.find_spec("numba") is not None)
    with pytest.raises(ValueError, match="Unknown extra"):
        has_extra("gpu")


def test_every_extra_asked_for_is_declared_in_pyproject():
    declared = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    assert set(EXTRAS) <= set(declared)
