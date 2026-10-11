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


def test_issue_time_prompt_defaults_to_the_information_set_default(monkeypatch, tmp_path):
    """The wizard offers the same issue time as `InformationSet()`."""
    import pandas as pd

    from epftoolbox2.information_set import InformationSet

    class Stop(Exception):
        pass

    class Answer:
        def __init__(self, value):
            self.value = value

        def ask(self):
            return self.value

    offered = {}

    def text(message, default=""):
        if "Issue time" in message:
            offered["issue_time"] = default
            raise Stop
        return Answer(str(tmp_path) if "CSV" in message else default or "7")

    frame = pd.DataFrame({"price": [1.0]}, index=pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC"))
    monkeypatch.setattr(tui, "_load_data_csv", lambda path: frame)
    monkeypatch.setattr(tui, "_build_component_list", lambda *args, **kwargs: [])
    monkeypatch.setattr(tui, "_ask_date", lambda label, default="": "2024-01-01")
    monkeypatch.setattr(tui, "_ask_freq", lambda: "1h")
    monkeypatch.setattr(tui.questionary, "text", text)
    monkeypatch.setattr(tui.questionary, "confirm", lambda *args, **kwargs: Answer(False))
    monkeypatch.setattr(tui.questionary, "select", lambda *args, **kwargs: Answer("price"))

    with pytest.raises(Stop):
        tui.build_model_pipeline_interactive()

    assert offered["issue_time"] == InformationSet().issue_time == "11:45"
