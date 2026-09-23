"""The TUI's model-side component lists and parameter schemas match the classes they build."""

import inspect

import pytest

from epftoolbox2 import tui
from epftoolbox2.pipelines.model_pipeline import COMPONENT_REGISTRY


@pytest.mark.parametrize("name", tui._MODELS + tui._EVALUATORS + tui._EXPORTERS)
def test_schema_fields_are_constructor_parameters(name):
    registry = {**COMPONENT_REGISTRY["models"], **COMPONENT_REGISTRY["evaluators"], **COMPONENT_REGISTRY["exporters"]}
    assert name in registry
    cls = getattr(__import__(registry[name], fromlist=[name]), name)
    accepted = set(inspect.signature(cls.__init__).parameters) | set(
        inspect.signature(cls.__mro__[1].__init__).parameters
    )
    fields = {f["name"] for f in tui.PARAM_SCHEMAS[name]}
    assert fields <= accepted, fields - accepted


def test_every_registered_evaluator_and_exporter_is_offered():
    assert set(tui._EVALUATORS) == set(COMPONENT_REGISTRY["evaluators"])
    assert set(tui._EXPORTERS) == set(COMPONENT_REGISTRY["exporters"])


def test_rmae_and_csv_exporter_can_be_built(tmp_path):
    rmae = tui._instantiate("rMAEEvaluator", {"base_model": "OLS"})
    assert rmae.base_model == "OLS"
    exporter = tui._instantiate("CsvExporter", {"path": str(tmp_path / "out.csv"), "extra_columns": ["is_holiday"]})
    assert exporter.extra_columns == ["is_holiday"]
