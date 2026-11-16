"""What a live revision keeps of its issue (step 8f, `epftoolbox2.pipelines._issue_records`): the input snapshot and its hash,
the data-quality record, the check, refresh and wait records, and how they are written.

The hash is checked against its definition, recomputed here with hashlib from the values (no library code): the issue rows'
UTC instants, then each column in sorted name order with its float64 bytes.
"""

import dataclasses
import datetime
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.transformers import ImputeTransformer
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models.context import list_hash
from epftoolbox2.pipelines._issue_records import (
    CHECKS_FILE,
    QUALITY_FILE,
    SNAPSHOT_FILE,
    check,
    imputed_flag,
    issue_rows,
    jsonable,
    quality,
    read_snapshot,
    refresh_record,
    snapshot,
    snapshot_hash,
    waits_record,
    write_json,
    write_snapshot,
)
from epftoolbox2.results.checks import CHECK_FAILED, ForecastChecks, planned_periods

ZONE = "Europe/Berlin"
ISSUE = "2026-10-25"  # the autumn switch: 25 hours, 100 quarter-hours


@pytest.fixture(autouse=True)
def offline(no_network):
    yield


def live_frame(freq="15min", seed=0):
    """Four days around the autumn switch in Berlin, three inputs and the target, in shuffled column order."""
    index = pd.date_range(
        pd.Timestamp("2026-10-23", tz=ZONE), pd.Timestamp("2026-10-27", tz=ZONE), freq=freq, inclusive="left"
    )
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "price": rng.normal(90, 20, len(index)),
            "load_d+1": rng.normal(50_000, 3_000, len(index)),
            "exaa_d+1": rng.normal(85, 15, len(index)),
            "wind_d+1": rng.gamma(2, 3_000, len(index)),
        },
        index=index,
    )
    return frame


PLAN = {
    "lear": {(period, 1): ["load_d+1", "exaa_d+1"] for period in range(96)},
    "fallback": {(period, 1): ["load_d+1"] for period in range(96)},
}


def with_payload_nan(frame, row, column):
    """`frame` with a NaN of another payload than numpy's at (row, column), set through the array so pandas keeps its bits."""
    values = frame.to_numpy(dtype=np.float64, copy=True)
    values.view(np.uint64)[row, column] = 0x7FF8000000000001
    out = pd.DataFrame(values, index=frame.index, columns=frame.columns)
    assert out.to_numpy()[row, column].tobytes() != np.float64(np.nan).tobytes()
    return out


def oracle_hash(frame, columns, issue):
    """The definition, written out: the rows dated `issue` in the frame's zone, their UTC instants, then each column by name."""
    wall = frame.index.tz_localize(None)
    rows = frame[(wall >= pd.Timestamp(issue)) & (wall < pd.Timestamp(issue) + pd.Timedelta(days=1))]
    digest = hashlib.sha256(b"epftoolbox2 snapshot 1\n")
    instants = rows.index.tz_convert("UTC").as_unit("ns").asi8
    digest.update(len(instants).to_bytes(8, "little") + instants.astype("<i8").tobytes())
    digest.update(len(columns).to_bytes(8, "little"))
    for column in sorted(columns):
        digest.update(len(column.encode()).to_bytes(8, "little") + column.encode())
        digest.update(rows[column].to_numpy(dtype="<f8").tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------------------------------------------------


def test_a_snapshot_holds_the_issue_rows_of_what_the_models_read():
    frame = live_frame()
    snap = snapshot(frame, ["lear", "fallback"], PLAN, ISSUE)
    assert snap.issue == ISSUE
    assert list(snap.inputs.columns) == ["exaa_d+1", "load_d+1"]  # wind and the target nobody reads aren't in it
    assert len(snap.inputs) == 100  # both occurrences of the repeated hour's quarter-hours
    assert (snap.inputs.index.tz_localize(None).normalize() == pd.Timestamp(ISSUE)).all()
    assert snap.inputs.dtypes.unique().tolist() == [np.float64]
    assert snap.hash == oracle_hash(frame, ["exaa_d+1", "load_d+1"], ISSUE) == snapshot_hash(snap.inputs)
    assert snap.lists[0] == {
        "model": "lear",
        "period": 0,
        "horizon": 1,
        "list_hash": list_hash(["load_d+1", "exaa_d+1"]),
    }
    assert len(snap.lists) == 192
    one_list = snapshot(frame, ["fallback"], {"fallback": ["load_d+1"]}, datetime.date(2026, 10, 25))
    assert one_list.lists == [
        {"model": "fallback", "period": None, "horizon": None, "list_hash": list_hash(["load_d+1"])}
    ]


def test_the_hash_is_the_same_for_the_same_values_whatever_the_column_order():
    frame = live_frame()
    first = snapshot(frame, ["lear"], PLAN, ISSUE).hash
    shuffled = frame[list(reversed(frame.columns))]
    assert snapshot(shuffled, ["lear"], PLAN, ISSUE).hash == first
    # The rows in another order, other days changed, an int column of the same values, -0.0 for 0.0, one NaN for another.
    assert snapshot(frame.iloc[::-1], ["lear"], PLAN, ISSUE).hash == first
    other_days = frame.copy()
    other_days.loc[other_days.index.tz_localize(None).normalize() != pd.Timestamp(ISSUE), "load_d+1"] += 1.0
    assert snapshot(other_days, ["lear"], PLAN, ISSUE).hash == first
    zeros = frame.copy()
    zeros["exaa_d+1"] = 0.0
    negative = zeros.copy()
    negative["exaa_d+1"] = -0.0
    assert snapshot(zeros, ["lear"], PLAN, ISSUE).hash == snapshot(negative, ["lear"], PLAN, ISSUE).hash
    nans = frame.copy()
    at = int(np.flatnonzero(issue_rows(frame, ISSUE))[10])  # a cell the snapshot holds
    nans.iloc[at, 2] = np.nan
    payload = with_payload_nan(frame, at, 2)
    assert snapshot(nans, ["lear"], PLAN, ISSUE).hash == snapshot(payload, ["lear"], PLAN, ISSUE).hash
    as_ints = frame.copy()
    as_ints["exaa_d+1"] = np.round(as_ints["exaa_d+1"])
    as_ints_int = as_ints.astype({"exaa_d+1": "int64"})
    assert snapshot(as_ints, ["lear"], PLAN, ISSUE).hash == snapshot(as_ints_int, ["lear"], PLAN, ISSUE).hash


def test_the_hash_differs_on_one_changed_cell_and_on_other_instants():
    frame = live_frame()
    first = snapshot(frame, ["lear"], PLAN, ISSUE).hash
    rows = np.flatnonzero(issue_rows(frame, ISSUE))
    seen = {first}
    for row in rows[[0, 37, 99]]:
        for column in ("load_d+1", "exaa_d+1"):
            changed = frame.copy()
            changed.iloc[row, changed.columns.get_loc(column)] = np.nextafter(changed.iloc[row][column], np.inf)
            seen.add(snapshot(changed, ["lear"], PLAN, ISSUE).hash)
    missing = frame.copy()
    missing.iloc[rows[5], missing.columns.get_loc("load_d+1")] = np.nan
    seen.add(snapshot(missing, ["lear"], PLAN, ISSUE).hash)
    renamed = frame.rename(columns={"exaa_d+1": "exab_d+1"})
    seen.add(snapshot(renamed, ["lear"], {"lear": {(0, 1): ["load_d+1", "exab_d+1"]}}, ISSUE).hash)
    shifted = frame.copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=15)
    seen.add(snapshot(shifted, ["lear"], PLAN, ISSUE).hash)
    assert len(seen) == 10  # the first, six changed cells, a missing one, a renamed column, other instants


def test_a_snapshot_refuses_what_it_can_t_hold():
    frame = live_frame()
    with pytest.raises(KeyError, match="no plan"):
        snapshot(frame, ["lear", "other"], PLAN, ISSUE)
    with pytest.raises(KeyError, match="no column"):
        snapshot(frame, ["x"], {"x": ["solar_d+1"]}, ISSUE)
    text = frame.assign(**{"load_d+1": "high"})
    with pytest.raises(TypeError, match="float64"):
        snapshot(text, ["lear"], PLAN, ISSUE)
    with pytest.raises(TypeError, match="a day, not an instant"):
        snapshot(frame, ["lear"], PLAN, pd.Timestamp(ISSUE, tz=ZONE))
    with pytest.raises(ValueError, match="two models"):
        snapshot(frame, ["lear", "lear"], PLAN, ISSUE)


def test_write_and_read_a_snapshot(tmp_path):
    snap = snapshot(live_frame(), ["lear", "fallback"], PLAN, ISSUE)
    path = write_snapshot(snap, tmp_path)
    assert path == tmp_path / SNAPSHOT_FILE and not list(tmp_path.glob(".*"))
    back = read_snapshot(path)
    assert back.hash == snap.hash and back.issue == ISSUE and back.lists == snap.lists
    pd.testing.assert_frame_equal(back.inputs, snap.inputs, check_exact=True, check_freq=False)
    # A file whose values aren't those it records the hash of raises.
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    changed = snap.inputs.copy()
    changed.iloc[0, 0] += 1
    pq.write_table(
        pa.Table.from_pandas(changed, preserve_index=True).replace_schema_metadata(table.schema.metadata), path
    )
    with pytest.raises(ValueError, match="hash to"):
        read_snapshot(path)


# ---------------------------------------------------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Source:  # stands in for one source of 8b's RefreshRecord
    status: str
    last_valid: dict
    seconds: float = 1.5


@dataclasses.dataclass
class Refresh:  # stands in for 8b's RefreshRecord
    sources: dict


def test_quality_counts_missing_and_imputed_cells_on_the_issue_rows():
    frame = live_frame(freq="1h")
    raw = frame.copy()
    rows = np.flatnonzero(issue_rows(raw, ISSUE))
    raw.iloc[rows[:3], raw.columns.get_loc("load_d+1")] = np.nan  # three cells of the issue day, imputed below
    raw.iloc[rows[0] - 2, raw.columns.get_loc("load_d+1")] = np.nan  # one of the day before
    raw.iloc[rows[4:6], raw.columns.get_loc("exaa_d+1")] = np.nan  # two left missing
    raw.iloc[-5:, raw.columns.get_loc("wind_d+1")] = np.nan  # the last five hours
    imputed = ImputeTransformer(columns=["load_d+1"], method="ffill", flag=True).transform(raw)
    rules = InformationSet(
        "11:45",
        ZONE,
        rules={"exaa_d+1": Availability.published("10:15", days_before=1, timezone=ZONE, vintage="snapshot")},
    )
    refresh = Refresh(
        {
            "entsoe": Source("ok", {"load_d+1": pd.Timestamp("2026-10-26 23:00", tz=ZONE)}),
            "exaa": {"status": "timeout", "last_valid": {"exaa_d+1": None}},
        }
    )
    found = quality(imputed, refresh, rules, ISSUE)
    assert found["issue"] == ISSUE
    assert found["sources"] == {
        "entsoe": {"status": "ok", "last_valid": {"load_d+1": "2026-10-26T23:00:00+01:00"}},
        "exaa": {"status": "timeout", "last_valid": {"exaa_d+1": None}},
    }
    columns = found["columns"]
    assert set(columns) == {"price", "load_d+1", "exaa_d+1", "wind_d+1"}  # not the flag
    assert columns["load_d+1"] == {
        "issue_rows": 25,
        "missing": 0,
        "imputed": 3,
        "flag": "load_d+1_imputed",
        "last_valid": imputed.index[-1].isoformat(),
        "vintage": "unknown",
    }
    assert columns["exaa_d+1"]["missing"] == 2 and columns["exaa_d+1"]["imputed"] is None
    assert columns["exaa_d+1"]["vintage"] == "snapshot"
    assert columns["wind_d+1"]["last_valid"] == imputed.index[-6].isoformat() and columns["wind_d+1"]["missing"] == 0
    plain = quality(imputed, None, None, ISSUE, columns=["exaa_d+1"])
    assert plain["sources"] is None and plain["columns"]["exaa_d+1"]["vintage"] is None
    json.dumps(found, allow_nan=False)  # plain JSON
    with pytest.raises(KeyError):
        quality(imputed, None, None, ISSUE, columns=["solar"])
    with pytest.raises(TypeError, match="sources"):
        quality(imputed, {"entsoe": "ok"}, None, ISSUE)


def test_a_shifted_column_finds_its_shifted_flag():
    frame = pd.DataFrame(
        {"load_d+1": [1.0], "load_imputed_d+1": [1], "wind": [2.0], "wind_imputed": [0]},
        index=pd.DatetimeIndex([pd.Timestamp(ISSUE, tz=ZONE)]),
    )
    assert imputed_flag(frame, "load_d+1") == "load_imputed_d+1"
    assert imputed_flag(frame, "wind") == "wind_imputed"
    assert imputed_flag(frame, "load") is None
    assert quality(frame, None, None, ISSUE)["columns"]["load_d+1"]["imputed"] == 1


# ---------------------------------------------------------------------------------------------------------------------
# Checks, refresh and waits records, and their files
# ---------------------------------------------------------------------------------------------------------------------


def test_check_is_the_record_check_and_checks_json_is_its_outcome(tmp_path):
    records = [
        {
            "run_date": "2026-10-24",
            "target_date": ISSUE,
            "run_weekday": 5,
            "hour": p,
            "horizon": 1,
            "day_in_test": 0,
            "prediction": 50.0,
            "actual": None,
            "coefficients": [],
            "freq": "15min",
            "status": "ok",
            "list_hash": "x",
            "refit_anchor": "2026-10-24",
            "revision": 2,
        }
        for p in range(96)
    ]
    checks = ForecastChecks()
    outcome = check({"lear": records}, checks, planned_periods("15min", ZONE), freq="15min", zone=ZONE)
    assert {r["status"] for r in outcome.records["lear"]} == {CHECK_FAILED}
    assert {r["revision"] for r in outcome.records["lear"]} == {2}  # the revision's own number
    assert outcome.findings[0]["products"] == 100 and outcome.findings[0]["failed"] == {"not_constant": "all 96 at 50"}
    path = write_json(tmp_path, CHECKS_FILE, outcome.to_dict(checks))
    assert json.loads(path.read_text())["failed"][0]["member"] == "lear"


def test_refresh_and_waits_records_are_plain_json(tmp_path):
    refresh = Refresh(
        {"entsoe": Source("ok", {"price": pd.Timestamp("2026-10-24 23:00", tz=ZONE)}, seconds=np.float64(2.0))}
    )
    record = refresh_record(refresh)
    assert record == {
        "sources": {"entsoe": {"status": "ok", "last_valid": {"price": "2026-10-24T23:00:00+02:00"}, "seconds": 2.0}}
    }
    assert refresh_record(None) is None
    polls = [
        {
            "at": pd.Timestamp("2026-10-24 09:45", tz="UTC"),
            "column": "load_d+1",
            "missing": np.int64(4),
            "gave_up": False,
            "slept": pd.Timedelta(minutes=5),
        }
    ]
    assert waits_record(polls) == {
        "polls": [
            {"at": "2026-10-24T09:45:00+00:00", "column": "load_d+1", "missing": 4, "gave_up": False, "slept": 300.0}
        ]
    }
    assert jsonable({"x": float("nan"), "y": {3, 1}, "z": np.array([1.5])}) == {"x": None, "y": [1, 3], "z": [1.5]}
    path = write_json(tmp_path, QUALITY_FILE, {"issue": ISSUE, "missing": float("inf")})
    assert json.loads(path.read_text()) == {"issue": ISSUE, "missing": None}
    assert not list(tmp_path.glob(".*"))


def test_snapshot_hash_itself_sorts_columns_and_rows_and_reads_one_nan():
    """`snapshot_hash` on inputs in any column and row order, as a later check may build them, and NaNs of any payload."""
    inputs = snapshot(live_frame(), ["lear"], PLAN, ISSUE).inputs
    first = snapshot_hash(inputs)
    assert snapshot_hash(inputs[list(reversed(inputs.columns))]) == first
    assert snapshot_hash(inputs.iloc[::-1]) == first
    nan = inputs.copy()
    nan.iloc[3, 0] = np.nan
    payload = with_payload_nan(inputs, 3, 0)
    assert snapshot_hash(nan) == snapshot_hash(payload) != first
