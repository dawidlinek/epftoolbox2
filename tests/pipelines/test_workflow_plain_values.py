"""`Workflow.to_dict()` and `save()` hand over plain values, whatever numpy and pandas types the Workflow was built from."""

import numpy as np
import pandas as pd
import pytest
import yaml

from epftoolbox2.pipelines import DataPipeline, Workflow
from epftoolbox2.pipelines._yaml import dump_yaml

PLAIN = (str, int, float, bool, type(None))


def assert_plain(value, where="config"):
    """Every value is of exactly the type `yaml.safe_load` gives back (a numpy scalar or a Timestamp is a subclass)."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert type(key) in (str, int), f"{where}: key {key!r} is a {type(key).__name__}"
            assert_plain(item, f"{where}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            assert_plain(item, f"{where}[{i}]")
    else:
        assert type(value) in PLAIN, f"{where} is a {type(value).__name__}: {value!r}"


def numpy_workflow(**horizons):
    workflow = Workflow(
        data_start=pd.Timestamp("2023-01-01"),
        data_end=pd.Timestamp("2024-01-01 06:00", tz="Europe/Warsaw"),
        data_cache=np.bool_(True),
        model_test_start=pd.Timestamp("2023-10-01 12:00"),
        model_test_end=np.datetime64("2023-12-31"),
        model_target=np.str_("price"),
        model_freq=np.str_("1h"),
        model_forecast_only=np.bool_(False),
        model_issue_days=np.array(["friday", "monday"]),
        max_processes=np.int64(4),
        threads_per_process=np.int32(2),
        cache_path=np.str_("/tmp/cache"),
        model_index=np.int64(2),
        models_per_task=3,
        model_save_dir="results",
        model_allow_manifest_mismatch=np.bool_(True),
        model_require_manifest=np.bool_(True),
        model_validation=[pd.Timestamp("2023-06-01"), np.datetime64("2023-09-30")],
        model_test=(pd.Timestamp("2023-10-01"), pd.Timestamp("2023-12-31", tz="Europe/Warsaw")),
        **horizons,
    )
    # Inline sections come from a YAML file, but a program can fill them (the experiment-check snapshot does).
    workflow._mp_inline = {
        "models": [
            {
                "class": "OLSModel",
                "params": {"predictors": ["price_d-1"], "ridge": np.float64(0.1), "weights": np.array([0.5, 1.5])},
            }
        ],
        "availability": np.str_("warn"),
    }
    workflow._dp_inline = {"validators": [{"class": "CoverageValidator", "params": {"min_coverage": np.float64(0.95)}}]}
    return workflow


@pytest.mark.parametrize("horizons", [{"model_horizon": np.int64(5)}, {"model_horizons": np.array([1, 3, 7])}])
def test_numpy_and_timestamp_values_save_and_load_as_the_plain_values_they_stand_for(tmp_path, horizons):
    original = numpy_workflow(**horizons)
    config = original.to_dict()
    assert_plain(config)

    text = dump_yaml(config)
    assert yaml.safe_load(text) == config

    path = tmp_path / "experiment.yaml"
    original.save(path)
    assert path.read_text() == text
    loaded = Workflow.load(path)
    assert loaded.to_dict() == config

    for name in (
        "data_cache model_target model_freq model_forecast_only max_processes threads_per_process cache_path "
        "model_index models_per_task model_save_dir model_allow_manifest_mismatch model_require_manifest model_horizon "
        "model_horizons"
    ).split():
        before, after = getattr(original, name), getattr(loaded, name)
        assert np.asarray(before).tolist() == np.asarray(after).tolist(), name
        assert type(after) in (*PLAIN, list), f"{name} is a {type(after).__name__} after the round trip"
    assert loaded.model_issue_days == ["monday", "friday"]  # saved as weekday names, in the week's order
    assert loaded._mp_inline["models"][0]["params"]["ridge"] == 0.1
    assert loaded._mp_inline["models"][0]["params"]["weights"] == [0.5, 1.5]
    assert loaded._dp_inline["validators"][0]["params"]["min_coverage"] == 0.95


def test_a_timestamp_keeps_the_instant_it_names_when_the_workflow_is_saved_and_loaded(tmp_path):
    """`DataPipeline.run` reads the text "2024-01-01" as the whole day and a Timestamp as the instant, so the Timestamp
    is saved with its time of day: the loaded range must be the range the original runs."""
    original = numpy_workflow(model_horizon=1)
    original.save(tmp_path / "experiment.yaml")
    loaded = Workflow.load(tmp_path / "experiment.yaml")

    assert loaded.data_start == "2023-01-01T00:00:00"
    assert loaded.data_end == "2024-01-01T06:00:00+01:00"
    pipeline = DataPipeline(timezone="Europe/Warsaw")
    assert pipeline.resolve_range(loaded.data_start, loaded.data_end) == pipeline.resolve_range(
        original.data_start, original.data_end
    )
    # The models read days, so a test bound is saved as its local day, like the days of a validation or test window.
    assert (loaded.model_test_start, loaded.model_test_end) == ("2023-10-01", "2023-12-31")
    assert loaded.model_validation == ["2023-06-01", "2023-09-30"]
    assert loaded.model_test == ["2023-10-01", "2023-12-31"]
