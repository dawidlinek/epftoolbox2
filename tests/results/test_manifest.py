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
    """Enough of BaseModel's shape for build_manifest: training_window, scaler and _model_kwargs.

    `scaler` is a separate constructor parameter, not part of `model_kwargs`, matching BaseModel: it is
    kept out of `_model_kwargs` (passed to workers separately) so a subclass overriding `_model_kwargs`
    can't accidentally drop it.
    """

    def __init__(self, training_window=30, scaler="standard", model_kwargs=None):
        self.training_window = training_window
        self.scaler = scaler
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
        expanded={(0, 1): ["load"]},
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

    assert manifest.predictors == ["load"]  # used_columns minus target (informational; order isn't compared)
    assert manifest.target == "price"
    assert manifest.horizon == 1
    assert manifest.freq == "1h"
    assert manifest.training_window == 45
    assert manifest.model_params == {"scaler": "standard", "cv": 5}
    assert manifest.data_timezone == "Europe/Warsaw"
    assert manifest.package_version  # recorded
    assert manifest.schema_version == 1


def test_data_timezone_is_part_of_the_comparison():
    """Converting Warsaw-local data to UTC keeps the same instants, and so the same data_columns_hash (the
    hash is over epoch values, not the tz label) - but resuming would silently mix rows recorded under two
    different wall-clock conventions unless the timezone itself is compared."""
    data = make_data()
    utc = data.tz_convert("UTC")
    a = make_manifest(data)
    b = make_manifest(utc)

    assert a.data_columns_hash == b.data_columns_hash  # same instants: the hash alone can't tell them apart
    diffs = a.differences(b)
    assert len(diffs) == 1 and "data_timezone" in diffs[0]


def test_predictor_layout_hash_is_sensitive_to_order_and_per_task_layout():
    """A sorted union of names can't tell `["price_d-1","price_d-2"]` apart from `["price_d-{horizon}"]`
    expanded per horizon, and doesn't notice a reordering that changes what stored coefficients mean."""
    data = make_data()
    fixed = make_manifest(data, expanded={(0, 1): ["price_d-1", "price_d-2"]})
    per_horizon = make_manifest(data, expanded={(0, 1): ["price_d-1"], (0, 2): ["price_d-2"]})
    reordered = make_manifest(data, expanded={(0, 1): ["price_d-2", "price_d-1"]})
    same_again = make_manifest(data, expanded={(0, 1): ["price_d-1", "price_d-2"]})

    assert fixed.predictor_layout_hash == same_again.predictor_layout_hash
    assert fixed.predictor_layout_hash != per_horizon.predictor_layout_hash
    assert fixed.predictor_layout_hash != reordered.predictor_layout_hash

    diffs = fixed.differences(reordered)
    assert len(diffs) == 1 and "predictor_layout_hash" in diffs[0]


def test_data_hash_changes_with_the_data_and_not_with_column_order():
    data = make_data()
    same_columns_reordered = data[["load", "price"]]
    changed = data.copy()
    changed.iloc[0, changed.columns.get_loc("price")] += 1.0

    assert hash_columns(data, ["price", "load"]) == hash_columns(same_columns_reordered, ["price", "load"])
    assert hash_columns(data, ["price", "load"]) != hash_columns(changed, ["price", "load"])


def test_data_hash_excludes_rows_after_test_end():
    """A source's own fetch window can catch up a little between two runs on the same day, appending rows
    after test_end; that must not raise a manifest mismatch. Rows at or before test_end still matter."""
    data = make_data(n=48)  # 2024-01-01 00:00 .. 2024-01-02 23:00
    base = dict(model=_DummyModel(training_window=1), test_start="2024-01-02", test_end="2024-01-02")
    a = make_manifest(data, **base)

    extra_index = pd.date_range("2024-01-03", periods=24, freq="h", tz="Europe/Warsaw")
    extra = pd.DataFrame({"price": 0.0, "load": 0.0}, index=extra_index)
    extended = pd.concat([data, extra])  # the first 48 rows are byte-identical to `data`
    b = make_manifest(extended, **base)
    assert a.data_columns_hash == b.data_columns_hash

    changed_within_range = data.copy()
    changed_within_range.iloc[0, changed_within_range.columns.get_loc("price")] += 1.0
    c = make_manifest(changed_within_range, **base)
    assert a.data_columns_hash != c.data_columns_hash


def test_data_hash_is_unaffected_by_a_different_test_start_over_the_same_data():
    """Resuming into the same save_dir with a different (e.g. narrower) test_start over otherwise
    unchanged data - as when a wide run is followed by a narrower one - must not look like a configuration
    change: only the data (up to test_end) matters, not where test_start happens to fall within it."""
    data = make_data(n=96)  # 2024-01-01 00:00 .. 2024-01-04 23:00
    a = make_manifest(data, test_start="2024-01-01", test_end="2024-01-04")
    b = make_manifest(data, test_start="2024-01-03", test_end="2024-01-04")

    assert a.data_columns_hash == b.data_columns_hash
    assert a.differences(b) == []


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
    path = tmp_path / "ols.manifest.json"  # results exist but no manifest file was written
    manifest = make_manifest(make_data())

    check_and_write(path, manifest, has_existing_results=True)  # default: warn and write, not raise
    assert path.exists()

    path.unlink()
    with pytest.raises(ManifestMismatchError, match="no manifest"):
        check_and_write(path, manifest, has_existing_results=True, require_manifest=True)


def test_information_set_is_recorded_but_not_compared():
    """The information set gates which predictors are allowed before fitting; it doesn't change what a
    fitted model predicts, so a change to it alone must not force a resume mismatch. It is still recorded
    for provenance."""
    from epftoolbox2.information_set import InformationSet

    data = make_data()
    a = make_manifest(data, information_set=InformationSet("11:45", "Europe/Brussels"))
    b = make_manifest(data, information_set=InformationSet("13:00", "Europe/Brussels"))

    assert a.information_set != b.information_set  # still recorded
    assert a.differences(b) == []  # but not compared


def test_model_params_generically_covers_any_constructor_parameter():
    """model_params comes from model._model_kwargs generically, so a brand-new constructor parameter is
    covered by the comparison without any manifest change, as long as the model returns it from
    _model_kwargs."""
    data = make_data()
    a = make_manifest(data, model=_DummyModel(model_kwargs={"cv": 5}))
    b = make_manifest(data, model=_DummyModel(model_kwargs={"cv": 10}))

    diffs = a.differences(b)
    assert len(diffs) == 1 and "model_params" in diffs[0]


def test_model_params_includes_the_scaler_so_resume_refuses_on_a_change(tmp_path):
    """integration check e: BaseModel keeps `scaler` out of `_model_kwargs` (passed to workers separately,
    see base.py) so a subclass overriding `_model_kwargs` can't drop it; build_manifest must add it back
    itself, or resuming into an existing directory after changing the scaler would silently mix forecasts
    fit in two different scales."""
    data = make_data()
    a = make_manifest(data, model=_DummyModel(scaler="standard"))
    b = make_manifest(data, model=_DummyModel(scaler="asinh"))

    assert a.model_params == {"scaler": "standard"}
    assert b.model_params == {"scaler": "asinh"}
    assert len(a.differences(b)) == 1 and "model_params" in a.differences(b)[0]

    path = tmp_path / "ols.manifest.json"
    check_and_write(path, a, has_existing_results=False)
    with pytest.raises(ManifestMismatchError, match="model_params"):
        check_and_write(path, b, has_existing_results=True)
