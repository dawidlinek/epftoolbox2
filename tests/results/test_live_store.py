"""Live directories (step 8c, `epftoolbox2.results.live`): revisions visible once committed, the commit log, deliveries, the
readers' policies and cut-offs, the live lock, the refusals between live and backtest directories, and `ForecastSource` on a
live upstream.

The readers are checked on hand-made files (`write_live`): three issue days of one model, written as literal JSON lines, with
an oracle (`expected`) that knows nothing of the library's rule but the spec's: per key the highest revision among the issue
day's committed ones the policy and cut-off let through. Issue day 1 has three revisions, r2 skipping one key and r3 never
committed; day 2 delivers its first revision only; day 3 commits r2 after the deadline and delivers nothing. Every reading is
made from the JSONL file and from its Parquet twin (`convert`, exact), which must give the same records.
"""

import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.sources import ForecastSource, UpstreamError
from epftoolbox2.models import OLSModel
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.live import (
    LiveBusyError,
    LiveDirectoryError,
    LiveLateError,
    LiveStore,
    live_reading,
    with_live,
)
from epftoolbox2.results.manifest import Manifest, ManifestMismatchError, check_and_write, check_and_write_live
from epftoolbox2.results.parquet_store import convert, iter_file_records
from epftoolbox2.results.ref import ModelResultRef
from epftoolbox2.results.store import (
    ResultStore,
    iter_file_records as iter_any,
    open_store,
    read_records,
    visible_revisions,
    with_status,
)

from tests.synthetic import PREDICTORS, make_market_data

pytestmark = pytest.mark.usefixtures("no_network")

ZONE = "Europe/Berlin"  # CEST (UTC+2) on every issue day here
DAYS = ("2026-10-21", "2026-10-22", "2026-10-23")
HOURS = (0, 1, 2, 3)
HORIZONS = (1, 2)
SKIPPED = (2, 1)  # (hour, horizon) day 1's r2 skips
# (day, revision) -> commit instant (UTC), None for one never committed. Deadline 11:55 CEST = 09:55 UTC.
REVISIONS = {
    (DAYS[0], 1): "2026-10-21T08:15:00+00:00",
    (DAYS[0], 2): "2026-10-21T09:20:00+00:00",
    (DAYS[0], 3): None,
    (DAYS[1], 1): "2026-10-22T08:10:00+00:00",
    (DAYS[1], 2): "2026-10-22T09:00:00+00:00",
    (DAYS[2], 1): "2026-10-23T08:05:00+00:00",
    (DAYS[2], 2): "2026-10-23T10:30:00+00:00",
}
ISSUED = {key: at or "2026-10-21T09:40:00+00:00" for key, at in REVISIONS.items()}
# The order the revisions' records were written in.
WRITTEN = [(DAYS[0], 1), (DAYS[1], 1), (DAYS[2], 1), (DAYS[0], 2), (DAYS[1], 2), (DAYS[0], 3), (DAYS[2], 2)]
DELIVERIES = [(DAYS[0], 2, "2026-10-21T09:30:00+00:00"), (DAYS[1], 1, "2026-10-22T08:20:00+00:00")]


def value(day: str, revision: int, hour: int, horizon: int) -> float:
    """A forecast naming its key and revision."""
    return revision * 1000 + DAYS.index(day) * 100 + hour * 10 + horizon + 0.25


def snapshot_of(day: str, revision: int) -> str:
    return hashlib.sha256(f"{day} r{revision}".encode()).hexdigest()


def base_record(day: str, hour: int, horizon: int, prediction, status: str = "ok") -> dict:
    run = datetime.date.fromisoformat(day)
    target = run + datetime.timedelta(days=horizon)
    record = {
        "run_date": day,
        "target_date": target.isoformat(),
        "run_weekday": run.weekday(),
        "hour": hour,
        "horizon": horizon,
        "day_in_test": (run - datetime.date(2026, 10, 19)).days,
        "prediction": prediction,
        "actual": 50.0 + hour,
        "coefficients": [] if prediction is None else [0.5, -0.25],
        "freq": "1h",
    }
    return with_status(record, status, list_hash="abc123", refit_anchor=day)


def live_record(day: str, revision: int, hour: int, horizon: int) -> dict:
    skipped = (day, revision) == (DAYS[0], 2) and (hour, horizon) == SKIPPED
    record = base_record(
        day,
        hour,
        horizon,
        None if skipped else value(day, revision, hour, horizon),
        "missing_test_predictors" if skipped else "ok",
    )
    return with_live(record, revision, ISSUED[(day, revision)], snapshot_of(day, revision))


def commit_line(day: str, revision: int, at: str, **extra) -> dict:
    return {
        "issue": day,
        "call": f"c-{day}-{revision}",
        "started": ISSUED[(day, revision)],
        "finished": at,
        "status": "ok",
        "exit_code": 0,
        "revision": revision,
        "committed": True,
        "issued_at": ISSUED[(day, revision)],
        "snapshot": snapshot_of(day, revision),
        **extra,
    }


def lines(records) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


def write_live(directory: Path, save_format: str = "jsonl", name: str = "ols") -> Path:
    """The hand-made live directory of the module docstring; returns the model's results path."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "live.json").write_text(json.dumps({"mode": "live", "issue_timezone": ZONE, "deadline": "11:55"}))
    records = [live_record(d, r, h, k) for d, r in WRITTEN for h in HOURS for k in HORIZONS]
    (directory / f"{name}.jsonl").write_text(lines(records))
    calls = [commit_line(d, r, REVISIONS[(d, r)]) for d, r in WRITTEN if REVISIONS[(d, r)] is not None]
    # A late call commits nothing, and day 1's r3 crashed before its line.
    calls.insert(
        3,
        {
            "issue": DAYS[1],
            "call": "late",
            "finished": "2026-10-22T10:00:00+00:00",
            "status": "late",
            "exit_code": 30,
            "revision": None,
            "committed": False,
        },
    )
    (directory / "issues.jsonl").write_text(lines(calls))
    (directory / "delivered.jsonl").write_text(
        lines(
            {"issue": d, "revision": r, "delivered_at": at, "receipt_sha256": "0" * 64, "receipt_path": "/r.json"}
            for d, r, at in DELIVERIES
        )
    )
    if save_format == "parquet":
        convert(directory, "parquet")
        return directory / f"{name}.parquet"
    return directory / f"{name}.jsonl"


def expected(policy: str, cutoff, statuses=("ok",)) -> dict:
    """key -> (revision, prediction, status) by the spec: per key the highest revision among its day's committed ones that the
    policy and cut-off let through ("first": the lowest committed alone; "delivered": up to the newest delivery)."""
    deadline = {day: pd.Timestamp(f"{day}T11:55", tz=ZONE).tz_convert("UTC") for day in DAYS}
    limit = {
        None: {day: None for day in DAYS},
        "deadline": deadline,
        "10:12": {day: pd.Timestamp(f"{day}T10:12", tz=ZONE).tz_convert("UTC") for day in DAYS},
    }[cutoff]
    out = {}
    for day in DAYS:
        seen = sorted(
            r
            for (d, r), at in REVISIONS.items()
            if d == day and at is not None and (limit[day] is None or pd.Timestamp(at) <= limit[day])
        )
        if policy == "first":
            seen = seen[:1]
        elif policy == "delivered":
            heads = [
                r
                for d, r, at in DELIVERIES
                if d == day and r in seen and (limit[day] is None or pd.Timestamp(at) <= limit[day])
            ]
            seen = [r for r in seen if heads and r <= heads[-1]]
        if not seen:
            continue
        revision = max(seen)
        for hour in HOURS:
            for horizon in HORIZONS:
                record = live_record(day, revision, hour, horizon)
                if statuses is None or record["status"] in statuses:
                    out[(hour, horizon, day)] = (revision, record["prediction"], record["status"])
    return out


def read(path: Path, policy: str = "latest", cutoff="deadline", statuses=("ok",)) -> dict:
    committed = LiveStore(path.parent).visible(policy, cutoff)
    found = {}
    for r in read_records(path, statuses=statuses, committed=committed):
        key = (r["hour"], r["horizon"], r["run_date"])
        assert key not in found
        found[key] = (r["revision"], r["prediction"], r["status"])
    return found


POLICY_CELLS = [
    ("latest", "deadline"),
    ("latest", None),
    ("latest", "10:12"),
    ("first", "deadline"),
    ("first", "10:12"),
    ("delivered", "deadline"),
    ("delivered", None),
]


@pytest.mark.parametrize("save_format", ["jsonl", "parquet"])
@pytest.mark.parametrize(("policy", "cutoff"), POLICY_CELLS)
def test_hand_made_revisions_read_per_policy_and_cutoff(tmp_path, save_format, policy, cutoff):
    path = write_live(tmp_path / "live", save_format)
    assert read(path, policy, cutoff) == expected(policy, cutoff)
    assert read(path, policy, cutoff, statuses=None) == expected(policy, cutoff, statuses=None)


def test_the_policies_differ_where_the_files_say(tmp_path):
    """The oracle's cells are not all one: latest hides day 1's skipped key behind r2 and ignores r3, first reads r1, delivered
    reads day 2's r1 and nothing of day 3, the deadline leaves out day 3's r2."""
    path = write_live(tmp_path / "live")
    latest, first, delivered = (read(path, p) for p in ("latest", "first", "delivered"))
    assert (2, 1, DAYS[0]) not in latest and first[(2, 1, DAYS[0])][0] == 1
    assert {v[0] for k, v in latest.items() if k[2] == DAYS[0]} == {2}
    assert {v[0] for k, v in first.items()} == {1}
    assert {v[0] for k, v in delivered.items() if k[2] == DAYS[1]} == {1}
    assert not any(k[2] == DAYS[2] for k in delivered)
    assert {v[0] for k, v in latest.items() if k[2] == DAYS[2]} == {1}
    assert {v[0] for k, v in read(path, "latest", None).items() if k[2] == DAYS[2]} == {2}
    assert read(path, statuses=None)[(2, 1, DAYS[0])] == (2, None, "missing_test_predictors")


@pytest.mark.parametrize("save_format", ["jsonl", "parquet"])
def test_model_result_ref_reads_by_policy_frame_and_records_alike(tmp_path, save_format):
    path = write_live(tmp_path / "live", save_format)

    def ref(**kwargs):
        return ModelResultRef("ols", path, 0, DAYS[0], DAYS[-1], 2, **kwargs)

    for policy, cutoff in POLICY_CELLS:
        want = expected(policy, cutoff)
        got = {
            (r["hour"], r["horizon"], r["run_date"]): r["prediction"]
            for r in ref(policy=policy, cutoff=cutoff).iter_records()
        }
        assert got == {key: v[1] for key, v in want.items()}, (policy, cutoff)
        frame = ref(policy=policy, cutoff=cutoff).frame(["hour", "horizon", "run_date", "prediction"])
        if save_format == "parquet":
            assert len(frame) == len(want)
            assert sorted(frame["prediction"]) == sorted(v[1] for v in want.values())
        else:
            assert frame is None
    # No policy in a live directory is "latest" before the deadline.
    assert [r["prediction"] for r in ref().iter_records()] == [
        r["prediction"] for r in ref(policy="latest").iter_records()
    ]
    first = ref(policy="first").first_record()
    assert first["revision"] == 1 and first["run_date"] == DAYS[0]


def test_a_policy_on_a_backtest_s_results_raises_and_backtest_refs_read_as_before(tmp_path):
    directory = tmp_path / "bt"
    directory.mkdir()
    path = directory / "ols.jsonl"
    path.write_text(lines(base_record(DAYS[0], h, 1, float(h)) for h in HOURS))
    plain = ModelResultRef("ols", path, 0, DAYS[0], DAYS[0], 1)
    assert [r["prediction"] for r in plain.iter_records()] == [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(LiveDirectoryError, match="backtest"):
        list(ModelResultRef("ols", path, 0, DAYS[0], DAYS[0], 1, policy="latest").iter_records())
    with pytest.raises(ValueError, match="memory"):
        live_reading(None, "first", "deadline")


@pytest.mark.parametrize("save_format", ["jsonl", "parquet"])
def test_reading_a_live_directory_without_committed_raises(tmp_path, save_format):
    path = write_live(tmp_path / "live", save_format)
    with pytest.raises(LiveDirectoryError, match="committed"):
        list(read_records(path))
    with pytest.raises(LiveDirectoryError):
        list(read_records(path, statuses=None, transform=lambda r: None))
    with pytest.raises(LiveDirectoryError):
        ResultStore(str(path)) if save_format == "jsonl" else open_store(path)


def test_live_records_outside_a_live_directory_and_backtest_records_inside_one_raise(tmp_path):
    live = write_live(tmp_path / "live")
    moved = tmp_path / "moved"
    moved.mkdir()
    (moved / "ols.jsonl").write_bytes(live.read_bytes())
    with pytest.raises(LiveDirectoryError, match="outside a live directory"):
        list(read_records(moved / "ols.jsonl"))
    (live.parent / "bt.jsonl").write_text(lines(base_record(DAYS[0], h, 1, float(h)) for h in HOURS))
    committed = {DAYS[0]: {1: pd.Timestamp(REVISIONS[(DAYS[0], 1)])}}
    with pytest.raises(LiveDirectoryError, match="no live record"):
        list(read_records(live.parent / "bt.jsonl", committed=committed))


def test_read_records_issued_by_cuts_off_per_issue_day(tmp_path):
    path = write_live(tmp_path / "live")
    committed = LiveStore(path.parent).visible("latest", None)
    assert visible_revisions(committed, pd.Timestamp("2026-10-21T09:00:00+00:00")) == frozenset({(DAYS[0], 1)})
    by_day = {DAYS[0]: "2026-10-21T09:10:00+00:00", DAYS[1]: None}  # days 2 and 3: no cut-off
    found = {(r["run_date"], r["revision"]) for r in read_records(path, committed=committed, issued_by=by_day)}
    assert found == {(DAYS[0], 1), (DAYS[1], 2), (DAYS[2], 2)}
    found = {
        (r["run_date"], r["revision"])
        for r in read_records(path, committed=committed, issued_by=lambda d: f"{d}T08:12:00+00:00")
    }
    assert found == {(DAYS[1], 1), (DAYS[2], 1)}
    with pytest.raises(ValueError, match="committed="):
        list(read_records(path, issued_by="2026-10-21T09:00:00+00:00"))


def test_a_commit_after_its_own_deadline_is_out_under_the_deadline_cutoff(tmp_path):
    path = write_live(tmp_path / "live")
    (path.parent / "live.json").write_text(json.dumps({"mode": "live", "issue_timezone": ZONE}))  # no deadline on file
    log = path.parent / "issues.jsonl"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    for line in calls:
        if line.get("committed") and line["issue"] == DAYS[1] and line["revision"] == 2:
            line["deadline"] = "2026-10-22T08:30:00+00:00"  # committed 09:00 UTC, after it
    log.write_text(lines(calls))
    assert {v[0] for k, v in read(path, "latest", "deadline").items() if k[2] == DAYS[1]} == {1}
    assert {v[0] for k, v in read(path, "latest", None).items() if k[2] == DAYS[1]} == {2}
    assert {v[0] for k, v in read(path, "latest", "deadline").items() if k[2] == DAYS[2]} == {2}  # no deadline anywhere


# --- the writer -------------------------------------------------------------------------------------------------------------


def write_through_the_store(directory: Path, save_format: str = "jsonl", name: str = "ols") -> None:
    """The files of `write_live`, written through `LiveStore` (r3 abandoned, as a crash before its commit leaves it)."""
    store = LiveStore(directory)
    store.init(mode="live", issue_timezone=ZONE, deadline="11:55")
    with store.hold():
        for position, (day, revision) in enumerate(WRITTEN):
            if position == 3:
                store.record_call(DAYS[1], "late", 30, call="late", finished="2026-10-22T10:00:00+00:00")
            rev = store.begin(
                day, issued_at=ISSUED[(day, revision)], snapshot=snapshot_of(day, revision), call=f"c-{day}-{revision}"
            )
            assert rev.revision == revision and rev.directory.is_dir()
            out = rev.open(name, save_format)
            for hour in HOURS:
                for horizon in HORIZONS:
                    record = live_record(day, revision, hour, horizon)
                    plain = {k: v for k, v in record.items() if k not in ("issued_at", "mode", "snapshot")}
                    plain["revision"] = 1  # what a backtest's builder gives: the store stamps the revision
                    (out.save if plain["status"] == "ok" else out.save_skip)(plain)
            if REVISIONS[(day, revision)] is None:
                rev.abandon()
            else:
                rev.commit(finished=REVISIONS[(day, revision)])
        for day, revision, at in DELIVERIES:
            receipt = directory / f"receipt-{day}.json"
            receipt.write_text("{}")
            store.mark_delivered(day, revision, receipt, delivered_at=at)


@pytest.mark.parametrize("save_format", ["jsonl", "parquet"])
def test_the_writer_writes_the_hand_made_files(tmp_path, save_format):
    hand = write_live(tmp_path / "hand")
    write_through_the_store(tmp_path / "store", save_format)
    stored = tmp_path / "store" / f"ols.{save_format}"
    assert sorted(json.dumps(r) for r in iter_any(stored)) == sorted(hand.read_text().splitlines())
    ours = [json.loads(line) for line in (tmp_path / "store" / "issues.jsonl").read_text().splitlines()]
    theirs = [json.loads(line) for line in (tmp_path / "hand" / "issues.jsonl").read_text().splitlines()]
    assert [line for line in ours if line["committed"]] == [line for line in theirs if line["committed"]]
    assert [(line["status"], line["revision"], line["committed"]) for line in ours if not line["committed"]] == [
        ("late", None, False)
    ]
    for policy, cutoff in POLICY_CELLS:
        assert read(stored, policy, cutoff) == expected(policy, cutoff)
    record = next(iter(iter_any(stored)))
    assert list(record)[-7:] == ["status", "list_hash", "refit_anchor", "revision", "issued_at", "mode", "snapshot"]


def test_the_writer_needs_the_lock_a_live_directory_and_reads_its_own_revision_alone(tmp_path):
    store = LiveStore(tmp_path / "live")
    with pytest.raises(RuntimeError, match="held"):
        store.begin(DAYS[0])
    with store.hold():
        with pytest.raises(LiveDirectoryError, match="init"):
            store.begin(DAYS[0])
    store.init(issue_timezone=ZONE)
    with pytest.raises(LiveDirectoryError, match="another live pipeline"):
        store.init(issue_timezone="Europe/Warsaw")
    with store.hold():
        rev = store.begin(DAYS[0], issued_at="2026-10-21T08:00:00+00:00", snapshot="s")
        for name in ("issues", "delivered"):
            with pytest.raises(ValueError, match="log"):
                rev.open(name)
        rev.abandon()
        with pytest.raises(RuntimeError, match="finished"):
            rev.open("ols")
    with pytest.raises(RuntimeError, match="held"):
        rev.store.record_call(DAYS[0], "noop", 0)


def test_pending_shows_a_call_its_own_revision_and_nobody_else(tmp_path):
    directory = tmp_path / "live"
    write_through_the_store(directory)
    store = LiveStore(directory)
    with store.hold():
        rev = store.begin(DAYS[1], issued_at="2026-10-22T09:30:00+00:00", snapshot="p")
        out = rev.open("ols")
        for hour in HOURS:
            for horizon in HORIZONS:
                out.save(base_record(DAYS[1], hour, horizon, -1.0))
        out.flush()
        path = directory / "ols.jsonl"
        assert {
            r["revision"]
            for r in read_records(path, committed=store.visible("latest", None))
            if r["run_date"] == DAYS[1]
        } == {2}
        mine = store.visible("latest", None, pending=rev)
        assert {r["revision"] for r in read_records(path, committed=mine) if r["run_date"] == DAYS[1]} == {3}
        with pytest.raises(ValueError, match="latest"):
            store.visible("first", None, pending=rev)
        rev.abandon()


# --- crashes and numbers ----------------------------------------------------------------------------------------------------


def test_a_crash_before_the_commit_is_invisible_and_its_number_is_never_reused(tmp_path):
    directory = tmp_path / "live"
    write_through_the_store(directory)
    store = LiveStore(directory)
    path = directory / "ols.jsonl"
    before = read(path)
    # Day 1's r3 was abandoned: the next revision of day 1 is 4, whatever folder or line is gone.
    assert store.next_revision(DAYS[0]) == 4
    (directory / "issues" / DAYS[0] / "r3").rmdir()
    assert store.next_revision(DAYS[0]) == 4  # r3's records still say so
    # A crash mid-commit: the commit line half-written (truncated) commits nothing.
    with store.hold():
        rev = store.begin(DAYS[1], issued_at="2026-10-22T09:30:00+00:00", snapshot="x")
        out = rev.open("ols")
        for hour in HOURS:
            for horizon in HORIZONS:
                out.save(base_record(DAYS[1], hour, horizon, -3.0))
        rev.commit(finished="2026-10-22T09:31:00+00:00")
    assert rev.revision == 3
    log = directory / "issues.jsonl"
    data = log.read_bytes()
    log.write_bytes(data[: len(data) - 40])  # the last line cut in the middle
    with pytest.warns(UserWarning, match="not a whole line"):
        assert read(path) == before
    with pytest.warns(UserWarning, match="not a whole line"):
        assert store.next_revision(DAYS[1]) == 4
    # The next call cuts the torn line off and takes 4; readers see 4 and never 3.
    with store.hold():
        with pytest.warns(UserWarning):
            rev = store.begin(DAYS[1], issued_at="2026-10-22T09:35:00+00:00", snapshot="y")
        assert rev.revision == 4
        out = rev.open("ols")
        for hour in HOURS:
            for horizon in HORIZONS:
                out.save(base_record(DAYS[1], hour, horizon, -4.0))
        with pytest.warns(UserWarning, match="cut off"):
            rev.commit(finished="2026-10-22T09:36:00+00:00")
    assert {v[:2] for k, v in read(path).items() if k[2] == DAYS[1]} == {(4, -4.0)}
    assert sorted(store.committed(DAYS[1])[DAYS[1]]) == [1, 2, 4]


def test_a_crash_while_writing_the_first_record_still_takes_the_number(tmp_path):
    directory = tmp_path / "live"
    write_through_the_store(directory)
    store = LiveStore(directory)
    path = directory / "ols.jsonl"
    with store.hold():
        rev = store.begin(DAYS[2], issued_at="2026-10-23T09:00:00+00:00", snapshot="z")
    with open(path, "ab") as f:  # half of r3's first line, then the process died
        f.write(json.dumps(with_live(base_record(DAYS[2], 0, 1, 1.0), 3, rev.issued_at, "z")).encode()[:30])
    with pytest.warns(UserWarning):
        assert store.next_revision(DAYS[2]) == 4  # the folder took 3 before any record
    with pytest.warns(UserWarning):
        assert read(path, "latest", None) == expected("latest", None)


def test_a_late_commit_raises_and_commits_nothing(tmp_path):
    directory = tmp_path / "live"
    write_through_the_store(directory)
    store = LiveStore(directory)
    with store.hold():
        rev = store.begin(DAYS[2], issued_at="2026-10-23T09:50:00+00:00", snapshot="late")
        rev.open("ols").save(base_record(DAYS[2], 0, 1, 7.0))
        with pytest.raises(LiveLateError) as caught:
            rev.commit(finished="2026-10-23T09:56:00+00:00", deadline="2026-10-23T09:55:00+00:00")
        assert caught.value.exit_code == 30
        line = rev.fail("late", 30)
    assert line["committed"] is False and line["revision"] == 3
    assert 3 not in store.committed(DAYS[2])[DAYS[2]]
    assert store.next_revision(DAYS[2]) == 4


# --- the lock and deliveries ------------------------------------------------------------------------------------------------


def snapshot_files(directory: Path) -> dict:
    return {
        str(p.relative_to(directory)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def test_the_live_lock_gives_busy_and_nothing_is_written(tmp_path):
    directory = tmp_path / "live"
    write_through_the_store(directory)
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"ok": true}')
    before = snapshot_files(directory)
    fd = os.open(directory / "live.lock", os.O_RDWR)  # another open file description: another holder to flock
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(LiveBusyError) as caught:
            with LiveStore(directory).hold():
                pass
        assert (caught.value.status, caught.value.exit_code) == ("busy", 75)
        with pytest.raises(LiveBusyError):
            ModelPipeline().mark_delivered(directory, DAYS[0], 1, receipt)
        with pytest.raises(LiveBusyError):
            ModelPipeline().mark_delivered(directory, DAYS[0], 1, receipt, dry_run=True)
    finally:
        os.close(fd)
    with LiveStore(directory).hold():  # in this process, through another LiveStore
        with pytest.raises(LiveBusyError):
            with LiveStore(directory).hold():
                pass
    assert snapshot_files(directory) == before


def test_mark_delivered_refuses_unknown_and_uncommitted_revisions_and_a_dry_run_writes_nothing(tmp_path):
    directory = tmp_path / "live"
    write_live(directory)
    (directory / "live.lock").unlink(missing_ok=True)
    receipt = tmp_path / "r.json"
    receipt.write_text('{"accepted": 96}')
    pipeline = ModelPipeline()
    before = snapshot_files(directory)
    line = pipeline.mark_delivered(directory, DAYS[1], 2, receipt, dry_run=True)
    assert snapshot_files(directory) == before  # not even live.lock
    assert line["receipt_sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert line["receipt_path"] == str(receipt.resolve()) and (line["issue"], line["revision"]) == (DAYS[1], 2)
    with pytest.raises(ValueError, match="not committed"):
        pipeline.mark_delivered(directory, DAYS[0], 3, receipt)
    with pytest.raises(ValueError, match="no revision 9"):
        pipeline.mark_delivered(directory, DAYS[0], 9, receipt)
    with pytest.raises(ValueError, match="not committed"):
        pipeline.mark_delivered(directory, DAYS[0], 3, receipt, dry_run=True)
    with pytest.raises(FileNotFoundError):
        pipeline.mark_delivered(directory, DAYS[0], 1, tmp_path / "missing.json")
    assert (directory / "delivered.jsonl").read_bytes() == before["delivered.jsonl"][0]
    written = pipeline.mark_delivered(directory, DAYS[1], 2, receipt)
    assert {k: v for k, v in written.items() if k != "delivered_at"} == {
        k: v for k, v in line.items() if k != "delivered_at"
    }
    assert LiveStore(directory).deliveries(DAYS[1])[-1] == written
    # The delivered policy now reads day 2's r2.
    assert {v[0] for k, v in read(directory / "ols.jsonl", "delivered", None).items() if k[2] == DAYS[1]} == {2}
    with pytest.raises(LiveDirectoryError):
        pipeline.mark_delivered(tmp_path / "nowhere", DAYS[0], 1, receipt)


# --- refusals both ways -----------------------------------------------------------------------------------------------------


def frame() -> pd.DataFrame:
    return make_market_data(start="2026-08-20", days=60, tz=ZONE, horizon=1, noise=1.0, seed=3)


def test_a_backtest_run_into_a_live_directory_and_its_report_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    directory = tmp_path / "live"
    write_live(directory)
    before = snapshot_files(directory)
    pipeline = ModelPipeline().add_model(OLSModel(PREDICTORS, 28, name="ols"))
    with pytest.raises(LiveDirectoryError, match="live directory"):
        pipeline.run(frame(), "2026-10-01", "2026-10-02", horizon=1, save_dir=str(directory))
    with pytest.raises(LiveDirectoryError, match="live directory"):
        pipeline.run({"DE": frame()}, "2026-10-01", "2026-10-02", horizon=1, save_dir=str(directory))
    with pytest.raises(LiveDirectoryError, match="live directory"):
        pipeline.report(directory)
    with pytest.raises(LiveDirectoryError, match="backtest run"):
        open_store(directory / "other.jsonl", lock=True)
    with pytest.raises(LiveDirectoryError, match="backtest run"):
        open_store(directory / "other.parquet", lock=True)
    assert snapshot_files(directory) == before


def test_a_live_writer_into_a_backtest_directory_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    directory = tmp_path / "bt"
    ModelPipeline().add_model(OLSModel(PREDICTORS, 28, name="ols")).run(
        frame(), "2026-10-01", "2026-10-02", horizon=1, save_dir=str(directory)
    )
    before = snapshot_files(directory)
    with pytest.raises(LiveDirectoryError, match="backtest"):
        LiveStore(directory).init(issue_timezone=ZONE)
    from epftoolbox2.results.live import LiveResultStore, live_parquet_store_class

    with pytest.raises(LiveDirectoryError, match="no live directory"):
        LiveResultStore(directory / "ols.jsonl", (2, "2026-10-01T09:00:00+00:00", None))
    with pytest.raises(LiveDirectoryError, match="no live directory"):
        live_parquet_store_class()(directory / "new.parquet", (2, "2026-10-01T09:00:00+00:00", None))
    with pytest.raises(LiveDirectoryError, match="backtest"):
        with LiveStore(directory).hold():
            pass
    assert snapshot_files(directory) == before


def manifest(**changes) -> Manifest:
    fields = dict(
        model_class="OLSModel",
        predictors=["x_d+1"],
        predictor_layout_hash="h",
        target="price",
        horizon=1,
        freq="1h",
        training_window=28,
        model_params={},
        data_columns_hash="d1",
        data_timezone=ZONE,
        information_set={
            "issue_time": "11:45",
            "issue_timezone": ZONE,
            "rules": {"x": {"kind": "published", "time": "10:00", "note": "a"}},
        },
        test_start=DAYS[0],
        test_end=DAYS[0],
        package_version="0",
        git_sha=None,
        created_at="2026-10-21T08:00:00+00:00",
        updated_at="2026-10-21T08:00:00+00:00",
        training_cut={"rows": "dates"},
    )
    fields.update(changes)
    return Manifest(**fields)


def test_a_live_manifest_compares_the_configuration_and_the_information_set_but_not_the_data(tmp_path):
    directory = tmp_path / "live"
    LiveStore(directory).init(issue_timezone=ZONE)
    path = directory / "ols.manifest.json"
    check_and_write_live(path, manifest())
    stored = json.loads(path.read_text())
    assert stored["mode"] == "live"
    check_and_write_live(path, manifest(data_columns_hash="d2", test_end=DAYS[1], updated_at="later"))  # grows daily
    info = manifest().information_set
    note_only = {**info, "rules": {"x": {**info["rules"]["x"], "note": "b"}}}
    check_and_write_live(path, manifest(information_set=note_only))
    assert json.loads(path.read_text())["created_at"] == "2026-10-21T08:00:00+00:00"
    with pytest.raises(ManifestMismatchError, match="training_window"):
        check_and_write_live(path, manifest(training_window=56))
    with pytest.raises(ManifestMismatchError, match="information_set"):
        check_and_write_live(path, manifest(information_set={**info, "issue_time": "11:00"}))
    with pytest.raises(LiveDirectoryError, match="live directory's manifest"):
        check_and_write(path, manifest(), has_existing_results=True)
    backtest = tmp_path / "bt" / "ols.manifest.json"
    check_and_write(backtest, manifest(), has_existing_results=False)
    assert "mode" not in json.loads(backtest.read_text())
    with pytest.raises(LiveDirectoryError, match="no live directory"):
        check_and_write_live(backtest, manifest())
    (directory / "bt.manifest.json").write_text(backtest.read_text())
    with pytest.raises(LiveDirectoryError, match="backtest's manifest"):
        check_and_write_live(directory / "bt.manifest.json", manifest())


# --- round trips --------------------------------------------------------------------------------------------------------------


def test_jsonl_to_parquet_and_back_with_the_live_keys_is_exact(tmp_path):
    directory = tmp_path / "live"
    path = write_live(directory)
    original = path.read_bytes()
    with pytest.raises(LiveDirectoryError):
        list(read_records(path))  # (a reader that doesn't say what is committed)
    convert(directory, "parquet")
    records = list(iter_file_records(directory / "ols.parquet"))
    assert sorted(json.dumps(r) for r in records) == sorted(original.decode().splitlines())
    assert all(list(r)[-3:] == ["issued_at", "mode", "snapshot"] for r in records)
    convert(directory, "jsonl")
    assert sorted(path.read_bytes().splitlines()) == sorted(original.splitlines())
    assert not (directory / "issues.parquet").exists() and not (directory / "delivered.parquet").exists()


def test_a_model_named_live_shares_the_live_lock(tmp_path):
    directory = tmp_path / "live"
    write_through_the_store(directory, name="live")
    assert read(directory / "live.jsonl") == expected("latest", "deadline")
    convert(directory, "parquet")
    assert read(directory / "live.parquet") == expected("latest", "deadline")


def test_parts_written_before_the_live_columns_read_and_resume_with_new_ones(tmp_path):
    """A backtest's Parquet part of the schema before 8c (no issued_at, mode, snapshot) reads, and concatenates with a part of
    today's schema, record for record."""
    import pyarrow as pa

    from epftoolbox2.results import parquet_schema as schema
    from epftoolbox2.results.parquet_store import _append_entry, _write_table, read_table

    directory = tmp_path / "bt" / "ols.parquet"
    directory.mkdir(parents=True)
    old = [base_record(DAYS[0], h, 1, float(h)) for h in HOURS]
    table = schema.to_table(old)
    table = table.select([n for n in table.column_names if n not in ("issued_at", "mode", "snapshot")])
    _append_entry(directory, _write_table(directory, "2026", table, "part-old-000001.parquet"))
    store = open_store(directory, lock=True)
    assert store.has_results()
    new = base_record(DAYS[1], 0, 1, 9.0)
    store.save(dict(new))
    store.close()
    assert [r["prediction"] for r in read_records(directory)] == [0.0, 1.0, 2.0, 3.0, 9.0]
    assert sorted(json.dumps(r) for r in iter_file_records(directory)) == sorted(json.dumps(r) for r in [*old, new])
    assert read_table(directory).num_rows == 5 and isinstance(read_table(directory), pa.Table)


# --- ForecastSource on a live upstream -----------------------------------------------------------------------------------------


UP_DAYS = ("2026-10-19", "2026-10-20", "2026-10-21")
# (day, revision) -> commit instant; the upstream's issue time is 11:45 CEST = 09:45 UTC.
UP = {
    (UP_DAYS[0], 1): "2026-10-19T09:00:00+00:00",  # 11:00
    (UP_DAYS[0], 2): "2026-10-19T09:50:00+00:00",  # 11:50: after 11:45, before 12:00
    (UP_DAYS[1], 1): None,  # never committed
    (UP_DAYS[2], 1): "2026-10-21T09:30:00+00:00",
    (UP_DAYS[2], 2): "2026-10-21T09:40:00+00:00",
}


def write_upstream(directory: Path, commits: dict = UP) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "live.json").write_text(json.dumps({"mode": "live", "issue_timezone": ZONE}))
    records, calls = [], []
    for (day, revision), at in commits.items():
        issued = at or f"{day}T09:10:00+00:00"
        for hour in range(24):
            record = base_record(day, hour, 1, revision * 1000 + UP_DAYS.index(day) * 100 + hour + 0.5)
            records.append(with_live(record, revision, issued, "s"))
        if at is not None:
            calls.append(
                {
                    "issue": day,
                    "call": "c",
                    "started": issued,
                    "finished": at,
                    "status": "ok",
                    "exit_code": 0,
                    "revision": revision,
                    "committed": True,
                    "issued_at": issued,
                    "snapshot": "s",
                }
            )
    (directory / "up.jsonl").write_text(lines(records))
    (directory / "issues.jsonl").write_text(lines(calls))
    (directory / "up.manifest.json").write_text(
        json.dumps(
            {
                "model_class": "OLSModel",
                "target": "price",
                "horizon": 1,
                "freq": "1h",
                "data_timezone": ZONE,
                "mode": "live",
                "information_set": {"issue_time": "11:45", "issue_timezone": ZONE, "unknown": "raise"},
                "availability": {"level": "warn", "times": "instants", "listing": []},
            }
        )
    )


def fetched(source: ForecastSource) -> pd.Series:
    return source.fetch(pd.Timestamp("2026-10-01", tz="UTC"), pd.Timestamp("2026-11-01", tz="UTC"))["fc"]


def upstream_expected(available: str = "11:45") -> pd.Series:
    out = {}
    for day in UP_DAYS:
        limit = pd.Timestamp(f"{day}T{available}", tz=ZONE).tz_convert("UTC")
        seen = [r for (d, r), at in UP.items() if d == day and at is not None and pd.Timestamp(at) <= limit]
        if not seen:
            continue
        target = datetime.date.fromisoformat(day) + datetime.timedelta(days=1)
        for hour in range(24):
            instant = pd.Timestamp(f"{target}T{hour:02d}:00", tz=ZONE).tz_convert("UTC")
            out[instant] = max(seen) * 1000 + UP_DAYS.index(day) * 100 + hour + 0.5
    return pd.Series(out, name="fc").sort_index()


@pytest.mark.parametrize("save_format", ["jsonl", "parquet"])
def test_forecast_source_reads_each_key_s_newest_revision_committed_by_available(tmp_path, save_format):
    directory = tmp_path / "up"
    write_upstream(directory)
    if save_format == "parquet":
        convert(directory, "parquet")
    got = fetched(ForecastSource(directory, "up", "fc"))
    want = upstream_expected()
    assert got.index.equals(want.index.tz_convert("UTC"))
    assert np.array_equal(got.to_numpy(), want.to_numpy())
    # Day 1 reads r1 at 11:45 and r2 once available is 12:00; day 2 committed nothing and has no value; day 3 reads r2.
    later = fetched(ForecastSource(directory, "up", "fc", available="12:00"))
    assert np.array_equal(later.to_numpy(), upstream_expected("12:00").to_numpy())
    assert not np.array_equal(later.to_numpy(), got.to_numpy())


def test_forecast_source_on_a_live_upstream_with_nothing_committed_raises(tmp_path):
    directory = tmp_path / "up"
    write_upstream(directory, {key: None for key in UP})
    with pytest.raises(UpstreamError, match="no committed revision"):
        fetched(ForecastSource(directory, "up", "fc"))


def test_forecast_source_refuses_live_and_backtest_mixed_and_sees_a_new_commit(tmp_path):
    directory = tmp_path / "up"
    write_upstream(directory)
    source = ForecastSource(directory, "up", "fc")
    digest = source.content_digest()
    with open(
        directory / "issues.jsonl", "a"
    ) as f:  # day 2's r1 committed late: the results don't change, the reading does
        f.write(
            json.dumps(
                {
                    "issue": UP_DAYS[1],
                    "call": "c",
                    "started": "x",
                    "finished": "2026-10-20T09:15:00+00:00",
                    "status": "ok",
                    "exit_code": 0,
                    "revision": 1,
                    "committed": True,
                }
            )
            + "\n"
        )
    assert source.content_digest() != digest
    assert len(fetched(source)) == len(upstream_expected()) + 24
    manifest = json.loads((directory / "up.manifest.json").read_text())
    del manifest["mode"]
    (directory / "up.manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(UpstreamError, match="mixed"):
        fetched(source)
