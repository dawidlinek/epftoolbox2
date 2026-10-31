"""The results manifest: what it records, what it compares, and the resume-compatibility policy
(roadmap item 7)."""

import json

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.results import manifest
from epftoolbox2.results.manifest import (
    FrameHashes,
    Manifest,
    ManifestMismatchError,
    build_manifest,
    check_and_write,
    hash_column,
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


def test_every_column_read_and_the_whole_frame_are_hashed_through_the_test_end():
    data = make_data()
    data["unread"] = 1.0
    manifest = make_manifest(data, used_columns=["price", "load"], test_end="2024-01-01")
    rows = data.loc[:"2024-01-01"]
    # On a float64 column `hash_column` is `hash_columns`, bit for bit.
    assert manifest.column_hashes == {"price": hash_columns(rows, ["price"]), "load": hash_columns(rows, ["load"])}
    assert manifest.frame_hash == hash_columns(rows, list(rows.columns))
    written = manifest.to_dict()
    assert written["column_hashes"] == manifest.column_hashes and written["frame_hash"] == manifest.frame_hash


def test_the_column_hashes_and_the_frame_hash_never_refuse_a_resume(tmp_path):
    data = make_data()
    data["unread"] = 1.0
    path = tmp_path / "m.manifest.json"
    check_and_write(path, make_manifest(data), has_existing_results=True)
    moved = data.copy()
    moved["unread"] = 2.0
    later = make_manifest(moved)
    assert later.frame_hash != make_manifest(data).frame_hash
    check_and_write(path, later, has_existing_results=True)
    assert json.loads(path.read_text())["frame_hash"] == later.frame_hash


def test_a_manifest_older_than_the_column_hashes_loads_resumes_and_gains_them(tmp_path):
    data = make_data()
    written = make_manifest(data).to_dict()
    del written["column_hashes"], written["frame_hash"]
    loaded = Manifest.from_dict(written)
    assert loaded.column_hashes is None and loaded.frame_hash is None
    assert "column_hashes" not in loaded.to_dict() and "frame_hash" not in loaded.to_dict()
    path = tmp_path / "m.manifest.json"
    path.write_text(json.dumps(written))
    check_and_write(path, make_manifest(data), has_existing_results=True)
    assert set(json.loads(path.read_text())["column_hashes"]) == {"price", "load"}


def test_a_frame_that_cant_be_hashed_doesnt_stop_a_run():
    data = make_data()
    data["notes"] = [[1, 2]] * len(data)
    manifest = make_manifest(data)
    assert manifest.frame_hash is None and "frame_hash" not in manifest.to_dict()
    assert set(manifest.column_hashes) == {"price", "load"}


def test_a_columns_hash_doesnt_depend_on_the_dtype_of_equal_values():
    index = pd.date_range("2024-01-01", periods=6, freq="h", tz="Europe/Warsaw")
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    frame = pd.DataFrame({"x": values}, index=index)
    same = {
        "float32": pd.Series(values, index=index, dtype="float32"),
        "int": pd.Series(values, index=index).astype(int),
        "bool": pd.Series(values, index=index).astype(bool),
        "Int64": pd.Series(values, index=index).astype("Int64"),
        "negative zero": pd.Series(values, index=index) * -1.0 * -1.0 + pd.Series([-0.0] * 6, index=index),
    }
    expected = hash_column(frame, "x")
    assert expected == hash_columns(frame, ["x"])
    for name, series in same.items():
        assert hash_column(series.to_frame("x"), "x") == expected, name
    assert hash_column(pd.DataFrame({"x": [v + 1e-9 for v in values]}, index=index), "x") != expected
    assert hash_columns(same["float32"].to_frame("x"), ["x"]) != expected, "data_columns_hash stays as it is"


def test_a_columns_hash_takes_one_nan_and_text():
    index = pd.date_range("2024-01-01", periods=3, freq="h", tz="Europe/Warsaw")
    quiet = np.array([1.0, np.nan, 3.0])
    computed = np.array([1.0, 0.0, 3.0])
    with np.errstate(invalid="ignore"):
        computed[1] = np.divide(0.0, 0.0)  # a NaN with its own sign bit on most machines
    assert hash_column(pd.DataFrame({"x": quiet}, index=index), "x") == hash_column(
        pd.DataFrame({"x": computed}, index=index), "x"
    )
    with_na = pd.Series([1, None, 3], index=index, dtype="Int64").to_frame("x")
    assert hash_column(with_na, "x") == hash_column(pd.DataFrame({"x": quiet}, index=index), "x")
    words = pd.DataFrame({"x": ["a", "b", "c"]}, index=index)
    assert hash_column(words, "x") == hash_columns(words, ["x"])


def test_frame_hashes_compute_each_thing_once_and_answer_what_the_functions_do(monkeypatch):
    data = make_data()
    data["other"] = 1.0
    calls = {"column": 0, "frame": 0, "columns": 0}
    real = {"column": manifest.hash_column, "frame": manifest.hash_frame, "columns": manifest.hash_columns}

    def counted(kind):
        def call(*args):
            calls[kind] += 1
            return real[kind](*args)

        return call

    hashes = FrameHashes(data)
    assert hashes.of_column("2024-01-01", "load") == hash_column(data.loc[:"2024-01-01"], "load")
    assert hashes.of_columns("2024-01-01", ["load", "price"]) == hash_columns(
        data.loc[:"2024-01-01"], ["load", "price"]
    )
    assert hashes.of_frame("2024-01-01") == hash_columns(data.loc[:"2024-01-01"], list(data.columns))
    monkeypatch.setattr(manifest, "hash_column", counted("column"))
    monkeypatch.setattr(manifest, "hash_frame", counted("frame"))
    monkeypatch.setattr(manifest, "hash_columns", counted("columns"))
    fresh = FrameHashes(data)
    for _ in range(3):
        fresh.of_column("2024-01-01", "load")
        fresh.of_frame("2024-01-01")
        fresh.of_columns("2024-01-01", ["load", "price"])
        fresh.of_columns("2024-01-01", ["price", "load"])
    assert calls["column"] == 1 and calls["frame"] == 1
    fresh.of_column("2024-01-02", "load")
    assert calls["column"] == 2, "another day is another hash"


def _written(path):
    return json.loads(path.read_text())


def test_a_mismatch_accepted_over_stored_forecasts_is_recorded_and_stays(tmp_path):
    data = make_data()
    path = tmp_path / "m.manifest.json"
    check_and_write(path, make_manifest(data), has_existing_results=False)
    assert "data_unverified" not in _written(path)
    moved = data.copy()
    moved["load"] = moved["load"] + 1.0
    with pytest.raises(ManifestMismatchError):
        check_and_write(path, make_manifest(moved), has_existing_results=True, frame_hashes=FrameHashes(moved))
    check_and_write(
        path, make_manifest(moved), has_existing_results=True, allow_mismatch=True, frame_hashes=FrameHashes(moved)
    )
    assert _written(path)["data_unverified"] is True
    check_and_write(path, make_manifest(moved), has_existing_results=True, frame_hashes=FrameHashes(moved))
    assert _written(path)["data_unverified"] is True, "it stays across resumes"


def test_a_mismatch_accepted_with_nothing_stored_is_no_matter(tmp_path):
    data = make_data()
    path = tmp_path / "m.manifest.json"
    check_and_write(path, make_manifest(data), has_existing_results=False)
    moved = data.copy()
    moved["load"] = moved["load"] + 1.0
    check_and_write(
        path, make_manifest(moved), has_existing_results=False, allow_mismatch=True, frame_hashes=FrameHashes(moved)
    )
    assert "data_unverified" not in _written(path)


def test_an_override_that_only_moves_the_test_end_says_nothing_of_the_data(tmp_path):
    """A plain run hashes other rows if it ends on another day. The rows on file are still what they were."""
    data = make_data()
    path = tmp_path / "m.manifest.json"
    check_and_write(path, make_manifest(data, test_end="2024-01-01"), has_existing_results=False)
    later = make_manifest(data, test_end="2024-01-02")
    check_and_write(path, later, has_existing_results=True, allow_mismatch=True, frame_hashes=FrameHashes(data))
    assert "data_unverified" not in _written(path)
    moved = data.copy()
    moved.iloc[0, 1] += 1.0
    other = make_manifest(moved, test_end="2024-01-02")
    check_and_write(path, other, has_existing_results=True, allow_mismatch=True, frame_hashes=FrameHashes(moved))
    assert _written(path)["data_unverified"] is True


def test_results_resumed_with_no_manifest_are_recorded_as_unverified(tmp_path):
    data = make_data()
    path = tmp_path / "m.manifest.json"
    check_and_write(path, make_manifest(data), has_existing_results=False)
    assert "data_unverified" not in _written(path)
    path.unlink()
    check_and_write(path, make_manifest(data), has_existing_results=True)
    assert _written(path)["data_unverified"] is True
    path.unlink()
    check_and_write(path, make_manifest(data), has_existing_results=False)
    assert "data_unverified" not in _written(path)
