"""The tracked root configs (data_pipeline.yaml, model_pipeline.yaml, experiment.yaml) double as a worked,
leak-free example: they must load without touching the network, and every model's predictors must pass the
information-set check against the declared issue_time. This mirrors what `Workflow.run()` does before fitting
(attach each source's rules, then check every expanded predictor) without actually fetching data.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from epftoolbox2.pipelines import DataPipeline, ModelPipeline
from epftoolbox2.data.transformers import TimezoneTransformer

ROOT = Path(__file__).resolve().parent.parent
SHIPPED_YAMLS = ["data_pipeline.yaml", "model_pipeline.yaml", "experiment.yaml"]


def _expand(predictors, horizon, periods=24):
    """{(period, horizon): [predictor names with {horizon} filled in]}, the shape InformationSet.check() wants."""
    return {
        (period, hz): [str(p).replace("{horizon}", str(hz)) for p in predictors]
        for period in range(periods)
        for hz in range(1, horizon + 1)
    }


def _local_timezone(data_pipeline: DataPipeline) -> str:
    """The zone the data actually ends up in: TimezoneTransformer.target_tz, falling back to DataPipeline.timezone."""
    for transformer in data_pipeline.transformers:
        if isinstance(transformer, TimezoneTransformer):
            return transformer.target_tz
    return data_pipeline.timezone or "UTC"


@pytest.mark.parametrize("name", SHIPPED_YAMLS)
def test_shipped_yaml_exists(name):
    assert (ROOT / name).is_file(), f"{name} is a shipped config listed in CLAUDE.md but is missing from the repo"


def test_shipped_pipelines_load_without_network():
    """Loading only constructs the declared components; nothing here should call out to ENTSO-E or Open-Meteo."""
    data_pipeline = DataPipeline.load(ROOT / "data_pipeline.yaml")
    assert data_pipeline.sources, "data_pipeline.yaml should declare at least one source"

    model_pipeline = ModelPipeline.load(ROOT / "model_pipeline.yaml")
    assert model_pipeline.models, "model_pipeline.yaml should declare at least one model"


def test_shipped_model_pipeline_declares_an_issue_time():
    """The shipped config should actually be checked, not merely leak-free by accident."""
    model_pipeline = ModelPipeline.load(ROOT / "model_pipeline.yaml")
    assert model_pipeline.information_set is not None, (
        "model_pipeline.yaml has no information_set: add issue_time/issue_timezone so this config is checked "
        "(see docs/model-pipeline/information-set)"
    )


def test_shipped_configs_predictors_pass_the_information_set_check():
    data_pipeline = DataPipeline.load(ROOT / "data_pipeline.yaml")
    model_pipeline = ModelPipeline.load(ROOT / "model_pipeline.yaml")
    with open(ROOT / "experiment.yaml") as f:
        experiment = yaml.safe_load(f)

    # Workflow.run() attaches source rules the same way before checking; do the same here without fetching data.
    information_set = model_pipeline.information_set.with_source_rules(data_pipeline.sources)

    target = experiment["model_pipeline"]["target"]
    horizon = experiment["model_pipeline"]["horizon"]
    freq = experiment["model_pipeline"].get("freq", "1h")

    tz = _local_timezone(data_pipeline)
    index = pd.date_range("2024-01-01", periods=24 * 60, freq="1h", tz=tz)

    for model in model_pipeline.models:
        expanded = _expand(model.predictors, horizon)
        # Raises InformationSetError if a predictor (or the newest training target) isn't known at issue time.
        information_set.check(expanded, target, freq, index, "2024-02-01", "2024-02-07", model_name=model.name)
