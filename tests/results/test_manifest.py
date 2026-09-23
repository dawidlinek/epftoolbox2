"""The results manifest: what it records, what it compares, and the resume-compatibility policy
(roadmap item 7)."""

import json

import pandas as pd
import pytest

from epftoolbox2.results.manifest import (
    Manifest,
    ManifestMismatchError,
    build_manifest,
    check_and_write,
    hash_columns,
    manifest_path,
)


class _DummyModel:
    """Enough of BaseModel's shape for build_manifest: training_window and _model_kwargs."""

    def __init__(self, training_window=30, model_kwargs=None):
        self.training_window = training_window
        self._model_kwargs = model_kwargs or {}


def make_data(n=48, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="Europe/Warsaw")
    return pd.DataFrame({"price": rng.normal(50, 5, n), "load": rng.normal(1000, 50, n)}, index=index)


def make_manifest(data, **overrides):
    kwargs = dict(
        model=_DummyModel(),
        data=data,
        used_columns=["price", "load"],
        target="price",
        horizon=1,
        freq="1h",
        test_start="2024-01-02",
        test_end="2024-01-02",
        information_set=None,
    )
    kwargs.update(overrides)
    return build_manifest(**kwargs)


def test_manifest_path_is_named_after_the_results_file(tmp_path):
    assert manifest_path(tmp_path / "ols.jsonl") == tmp_path / "ols.manifest.json"


def test_build_manifest_records_everything_that_defines_a_forecast():
    data = make_data()
    manifest = make_manifest(data, model=_DummyModel(training_window=45, model_kwargs={"cv": 5}))

    assert manifest.predictors == ["load"]  # used_columns minus target
    assert manifest.target == "price"
    assert manifest.horizon == 1
    assert manifest.freq == "1h"
    assert manifest.training_window == 45
    assert manifest.model_params == {"cv": 5}
    assert manifest.data_columns_hash == hash_columns(data, ["price", "load"])
    assert manifest.package_version  # recorded
    assert manifest.schema_version == 1


def test_data_hash_changes_with_the_data_and_not_with_column_order():
    data = make_data()
    same_columns_reordered = data[["load", "price"]]
    changed = data.copy()
    changed.iloc[0, changed.columns.get_loc("price")] += 1.0

    assert hash_columns(data, ["price", "load"]) == hash_columns(same_columns_reordered, ["price", "load"])
    assert hash_columns(data, ["price", "load"]) != hash_columns(changed, ["price", "load"])


def test_differences_only_reports_compared_fields():
    data = make_data()
    a = make_manifest(data)
    b = make_manifest(data, horizon=2, test_start="2024-01-03")  # test_start is provenance, not compared

    diffs = a.differences(b)
    assert len(diffs) == 1 and "horizon" in diffs[0]


def test_to_dict_from_dict_round_trip():
    manifest = make_manifest(make_data())
    restored = Manifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    assert restored == manifest


def test_from_dict_ignores_unknown_future_fields():
    manifest = make_manifest(make_data())
    data = manifest.to_dict()
    data["some_field_from_a_newer_schema"] = "value"
    restored = Manifest.from_dict(data)
    assert restored.extra == {"some_field_from_a_newer_schema": "value"}


def test_check_and_write_creates_a_fresh_manifest(tmp_path):
    path = tmp_path / "ols.manifest.json"
    manifest = make_manifest(make_data())

    check_and_write(path, manifest, has_existing_results=False)

    assert path.exists()
    assert Manifest.from_dict(json.loads(path.read_text())) == manifest


def test_check_and_write_passes_silently_when_compatible(tmp_path):
    path = tmp_path / "ols.manifest.json"
    data = make_data()
    first = make_manifest(data)
    check_and_write(path, first, has_existing_results=False)

    second = make_manifest(data, test_start="2024-01-03")  # same config, later resume: not a mismatch
    check_and_write(path, second, has_existing_results=True)

    stored = Manifest.from_dict(json.loads(path.read_text()))
    assert stored.created_at == first.created_at  # provenance of the first run is preserved
    assert stored.test_start == "2024-01-03"  # but the manifest reflects the latest run


def test_check_and_write_raises_on_mismatch_by_default(tmp_path):
    path = tmp_path / "ols.manifest.json"
    data = make_data()
    check_and_write(path, make_manifest(data), has_existing_results=False)

    changed = make_manifest(data, horizon=7)
    with pytest.raises(ManifestMismatchError, match="horizon"):
        check_and_write(path, changed, has_existing_results=True, model_name="OLS")

    # Refused: the file on disk must be untouched.
    assert Manifest.from_dict(json.loads(path.read_text())).horizon == 1


def test_check_and_write_allow_mismatch_overwrites(tmp_path):
    path = tmp_path / "ols.manifest.json"
    data = make_data()
    check_and_write(path, make_manifest(data), has_existing_results=False)

    changed = make_manifest(data, horizon=7)
    check_and_write(path, changed, has_existing_results=True, allow_mismatch=True)

    assert Manifest.from_dict(json.loads(path.read_text())).horizon == 7


def test_missing_manifest_is_accepted_by_default_and_can_be_required(tmp_path):
    path = tmp_path / "ols.manifest.json"  # no file written yet, simulating results from before this feature
    manifest = make_manifest(make_data())

    check_and_write(path, manifest, has_existing_results=True)  # default: warn and write, not raise
    assert path.exists()

    path.unlink()
    with pytest.raises(ManifestMismatchError, match="no manifest"):
        check_and_write(path, manifest, has_existing_results=True, require_manifest=True)


def test_information_set_settings_are_part_of_the_comparison():
    from epftoolbox2.information_set import InformationSet

    data = make_data()
    a = make_manifest(data, information_set=InformationSet("11:30", "Europe/Brussels"))
    b = make_manifest(data, information_set=InformationSet("13:00", "Europe/Brussels"))

    diffs = a.differences(b)
    assert len(diffs) == 1 and "information_set" in diffs[0]
