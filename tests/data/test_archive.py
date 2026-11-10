"""`SnapshotArchive` (record, catalog, first_seen, rewrites) and `ArchiveSource`, its replay as an honest vintage (step 8j).

The archive is filled by the real `EntsoeSource` parsing answers of a fake ENTSO-E (`FakeEntsoe`, a mocked session) on a fake clock:
the day-ahead load forecast (A65/A01) of each delivery day d is published at 10:00 Brussels on d - 1, and some cells are revised
later. Each request takes one second of the clock, so a fetch returns three seconds after it starts (A16, A01, A31). The archive
around the autumn switch of 2026 (Sunday 25 October, 25 hours) is read back:

- three recordings with a value revised between the second and the third give `first_seen` and `rewrites` exactly;
- `ArchiveSource(known_at="11:45", days_before=1)` gives each delivery day the values of the newest fetch by 11:45 on the day
  before, and the information set says each cell is known exactly then (an oracle written from the description alone, over both
  switches, several zone pairs, `days_before` and times the clocks repeat or skip);
- no byte the archive writes holds the API key planted in the environment.

No test opens a socket (`no_network`).
"""

import json
import os
import uuid

import numpy as np
import pandas as pd
import pytest
import requests
import yaml

from epftoolbox2.clock import FakeClock, use_clock
from epftoolbox2.data.archive import (
    ArchiveError,
    SecretInArchiveError,
    SnapshotArchive,
    _refuse_secrets,
    _refuse_secrets_in_frame,
    _uncached,
    archive_key,
    record_snapshot,
    source_secrets,
)
from epftoolbox2.data.sources import ArchiveSource, CalendarSource, CsvSource, EntsoeSource
from epftoolbox2.data.sources.base import DataSource
from epftoolbox2.data.transformers import LagTransformer, TimezoneTransformer
from epftoolbox2.information_set import _STEP_MINUTES, Availability, InformationSet, InformationSetError
from epftoolbox2.models._calendar import periods_per_day
from epftoolbox2.pipelines import DataPipeline

pytestmark = pytest.mark.usefixtures("no_network")

ZONE = "Europe/Brussels"
ORIGIN = pd.Timestamp("2026-10-01", tz="UTC")
HOUR = pd.Timedelta(hours=1)
# A key shaped like an ENTSO-E one (a UUID), made at run time so that no such literal is in the repository.
SECRET = str(uuid.UUID(int=0x5F0C2E7A4B1D4C3E9A8B0D1E2F3A4B5C))


def local(text: str, zone: str = ZONE) -> pd.Timestamp:
    """A wall-clock time of `zone` as a UTC instant (a repeated one at its first occurrence)."""
    return pd.Timestamp(text).tz_localize(zone, ambiguous=True).tz_convert("UTC")


def base_value(cell: pd.Timestamp) -> float:
    return 1000.0 + (cell - ORIGIN) / HOUR


class Truth:
    """What the fake ENTSO-E publishes: the forecast of a cell of delivery day d (a Brussels date) is out from 10:00 on d - 1,
    at `base_value`, and a revision `(cell, at, value)` replaces it from `at` on."""

    def __init__(self, revisions=()):
        self.revisions = sorted(revisions, key=lambda one: one[1])

    @staticmethod
    def published_at(cell: pd.Timestamp) -> pd.Timestamp:
        day = pd.Timestamp(cell.tz_convert(ZONE).date())
        return local(f"{(day - pd.Timedelta(days=1)).date()} 10:00")

    def value(self, cell: pd.Timestamp, at: pd.Timestamp) -> float:
        if at < self.published_at(cell):
            return np.nan
        value = base_value(cell)
        for revised, when, new in self.revisions:
            if revised == cell and when <= at:
                value = new
        return value


NO_DATA = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<Acknowledgement_MarketDocument '
    'xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0"><mRID>1</mRID>'
    "<Reason><code>999</code><text>No matching data found for Data item X and interval Y.</text></Reason>"
    "</Acknowledgement_MarketDocument>"
)


def a65(cells: pd.DatetimeIndex, values: np.ndarray) -> str:
    """An A65 document of one hourly Period over `cells` (consecutive UTC hours)."""
    points = "".join(
        f"<Point><position>{i}</position><quantity>{float(v)!r}</quantity></Point>" for i, v in enumerate(values, 1)
    )
    period = (
        f"<Period><timeInterval><start>{cells[0]:%Y-%m-%dT%H:%MZ}</start><end>{cells[-1] + HOUR:%Y-%m-%dT%H:%MZ}</end>"
        f"</timeInterval><resolution>PT60M</resolution>{points}</Period>"
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n<GL_MarketDocument '
        'xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"><mRID>x</mRID>'
        f"<TimeSeries><mRID>1</mRID><curveType>A01</curveType>{period}</TimeSeries></GL_MarketDocument>"
    )


def response(text: str, status: int = 200) -> requests.Response:
    out = requests.Response()
    out.status_code = status
    out._content = text.encode()
    out.headers.update({"Content-Type": "text/xml"})
    out.url = EntsoeSource.API_URL
    out.reason = "OK" if status < 400 else "Error"
    return out


class FakeEntsoe:
    """A mocked `session.get`: every request takes one second of `clock`, and A65/A01 is answered from `truth` as published by
    then (one Period over the cells out); every other document is "No matching data"."""

    def __init__(self, clock: FakeClock, truth: Truth):
        self.clock = clock
        self.truth = truth
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.clock.advance(1)
        params = dict(params or {})
        self.calls.append({"params": params, "headers": dict(headers or {}), "at": self.clock.now()})
        if (params.get("documentType"), params.get("processType")) != ("A65", "A01"):
            return response(NO_DATA)
        start = pd.to_datetime(params["periodStart"], format="%Y%m%d%H%M", utc=True)
        end = pd.to_datetime(params["periodEnd"], format="%Y%m%d%H%M", utc=True)
        cells = pd.date_range(start, end, freq="1h", inclusive="left")
        values = np.array([self.truth.value(cell, self.clock.now()) for cell in cells])
        out = ~np.isnan(values)
        if not out.any():
            return response(NO_DATA)
        assert out[: out.sum()].all(), "the fake publishes whole days from the range's start"
        return response(a65(cells[out], values[out]))

    def close(self):
        pass


def entsoe(server: FakeEntsoe, key: str = SECRET) -> EntsoeSource:
    source = EntsoeSource("BE", key, type=["load"])
    source.session = server
    return source


def recordings(archive: SnapshotArchive, pipeline: DataPipeline, clock: FakeClock, times):
    """`record` at each of `times` (Brussels wall times, in order), the clock moved forward to each; the lines written."""
    lines = []
    for text in times:
        target = local(text)
        assert clock.now() <= target
        clock.advance((target - clock.now()).total_seconds())
        result = archive.record(pipeline)
        assert result.exit_code == 0, result.summary()
        lines.extend(result.lines)
    return lines


# --- record, first_seen, rewrites ----------------------------------------------------------------------------------------


def test_three_recordings_with_a_revision_give_first_seen_and_rewrites_exactly(tmp_path):
    # The repeated hour of the 25-hour day: its second 02:00 (01:00 UTC) is revised at 11:05 on the day before.
    revised = local("2026-10-25 02:00") + HOUR
    assert revised.tz_convert(ZONE).hour == 2
    truth = Truth([(revised, local("2026-10-24 11:05"), 7777.25)])
    clock = FakeClock(local("2026-10-24 08:00"))
    server = FakeEntsoe(clock, truth)
    pipeline = DataPipeline(sources=[entsoe(server)], timezone=ZONE)
    archive = SnapshotArchive(tmp_path / "archive")
    with use_clock(clock):
        lines = recordings(archive, pipeline, clock, ["2026-10-24 09:00", "2026-10-24 10:30", "2026-10-24 11:30"])

    started = [local("2026-10-24 09:00"), local("2026-10-24 10:30"), local("2026-10-24 11:30")]
    fetched = [at + pd.Timedelta(seconds=3) for at in started]  # three requests of one second each: the answer in hand
    catalog = archive.catalog()
    assert list(catalog["status"]) == ["ok"] * 3
    assert list(catalog["started_at"]) == started
    assert list(catalog["fetched_at"]) == fetched
    # Whole Brussels days: from three days before today to two after it.
    assert set(catalog["start"]) == {local("2026-10-21 00:00")} and set(catalog["end"]) == {local("2026-10-27 00:00")}
    for call in server.calls:
        assert call["params"]["periodStart"] == "202610202200" and call["params"]["periodEnd"] == "202610262300"
    key = archive_key(pipeline.sources[0])
    assert archive.keys() == [key] and key.startswith("EntsoeSource-")
    assert lines[0]["config"] == pipeline.sources[0].get_cache_config()
    for line in lines:
        stored = (archive.path / line["file"]).read_bytes()
        assert line["sha256"] == __import__("hashlib").sha256(stored).hexdigest()

    # The cells each recording asked for that were out by then: days up to the 24th at 09:00, the 25th from 10:30.
    cells = pd.date_range(local("2026-10-21 00:00"), local("2026-10-27 00:00"), freq="1h", inclusive="left")
    days = cells.tz_convert(ZONE).normalize().tz_localize(None)
    out = days <= pd.Timestamp("2026-10-25")
    seen_first = pd.DatetimeIndex(
        [fetched[0] if day <= pd.Timestamp("2026-10-24") else fetched[1] for day in days[out]]
    )
    expected = pd.DataFrame(
        {"fetched_at": seen_first, "value": [base_value(c) for c in cells[out]]},
        index=pd.DatetimeIndex(cells[out], name="instant"),
    )
    pd.testing.assert_frame_equal(archive.first_seen("load_forecast"), expected, check_freq=False)
    assert len(expected) == 4 * 24 + 25  # the 21st to the 24th, and the 25-hour day

    rewrites = archive.rewrites("load_forecast")
    expected = pd.DataFrame(
        {
            "before_fetched_at": [fetched[1]],
            "before": [base_value(revised)],
            "after_fetched_at": [fetched[2]],
            "after": [7777.25],
        },
        index=pd.DatetimeIndex([revised], name="instant"),
    )
    pd.testing.assert_frame_equal(rewrites, expected, check_freq=False)


def test_an_observation_is_inside_the_range_asked_for_and_a_missing_value_is_no_rewrite(tmp_path):
    class Scripted(DataSource):
        """Returns the frames of `answers` in turn, whatever is asked."""

        def __init__(self):
            self.answers = []

        def _validate_config(self):
            return True

        def get_cache_config(self):
            return {"source_type": "scripted"}

        def fetch(self, start, end):
            return self.answers.pop(0)

    hours = pd.date_range("2026-10-20 22:00", periods=6, freq="1h", tz="UTC")
    source = Scripted()
    source.answers = [
        pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]}, index=hours),
        pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0]}, index=hours[:4]),  # 2.0 missing now: no change
        pd.DataFrame({"x": [1.0, 2.0, 3.5, 4.0]}, index=hours[:4]),  # 3.0 rewritten to 3.5
    ]
    before = pd.DataFrame({"x": [99.0]}, index=[hours[0] - HOUR])  # outside the range asked for: never read
    source.answers[0] = pd.concat([before, source.answers[0]])
    archive = SnapshotArchive(tmp_path)
    clock = FakeClock("2026-10-24 07:00")
    pipeline = DataPipeline(sources=[source], timezone=ZONE)
    with use_clock(clock):
        for _ in range(3):
            archive.record(pipeline)
            clock.advance(60)
    first = archive.first_seen("x")
    assert list(first.index) == list(hours)
    assert list(first["value"]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert list(first["fetched_at"]) == [clock.start] * 2 + [clock.start + pd.Timedelta(minutes=1)] + [clock.start] * 3
    rewrites = archive.rewrites("x")
    assert list(rewrites.index) == [hours[2]]
    assert rewrites.iloc[0].to_dict() == {
        "before_fetched_at": clock.start + pd.Timedelta(minutes=1),
        "before": 3.0,
        "after_fetched_at": clock.start + pd.Timedelta(minutes=2),
        "after": 3.5,
    }
    stored = archive.read(archive.fetches(archive.keys()[0])[0])
    assert stored.loc[hours[0] - HOUR, "x"] == 99.0  # stored as returned


def test_a_recording_bypasses_the_pipeline_cache_and_stores_the_frame_as_returned(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    clock = FakeClock(local("2026-10-24 10:30"))
    server = FakeEntsoe(clock, Truth())
    source = entsoe(server)
    pipeline = DataPipeline(sources=[source], timezone=ZONE)
    returned = []
    fetch = EntsoeSource.fetch

    def spy(self, start, end):
        frame = fetch(self, start, end)
        frame.attrs["note"] = "attrs are not stored"
        returned.append(frame)
        return frame

    monkeypatch.setattr(EntsoeSource, "fetch", spy)
    with use_clock(clock):
        pipeline.run("2026-10-21", "2026-10-26", cache=True)
        cache = sorted(p.relative_to(tmp_path) for p in (tmp_path / ".cache").rglob("*") if p.is_file())
        assert cache and len(server.calls) == 3
        SnapshotArchive(tmp_path / "archive").record(pipeline)
    assert len(server.calls) == 6  # asked again: the cache holds the range, the archive doesn't read it
    assert sorted(p.relative_to(tmp_path) for p in (tmp_path / ".cache").rglob("*") if p.is_file()) == cache
    archive = SnapshotArchive(tmp_path / "archive")
    stored = archive.read(archive.fetches("EntsoeSource")[0])
    expected = returned[-1].copy()
    expected.attrs = {}
    pd.testing.assert_frame_equal(stored, expected, check_freq=False)
    assert b"attrs are not stored" not in (archive.path / archive.catalog()["file"][0]).read_bytes()


def test_a_source_s_own_response_cache_is_skipped_on_a_copy():
    class Cached(DataSource):
        def __init__(self):
            self.use_cache = True

        def _validate_config(self):
            return True

        def get_cache_config(self):
            return {"source_type": "cached"}

        def fetch(self, start, end):
            raise AssertionError

    source = Cached()
    fresh = _uncached(source)
    assert fresh is not source and fresh.use_cache is False and source.use_cache is True
    calendar = CalendarSource("BE")
    assert _uncached(calendar) is calendar


def test_failed_and_skipped_sources_and_the_pending_budget(tmp_path, monkeypatch):
    class Broken(DataSource):
        api_key = None

        def __init__(self, api_key):
            self.api_key = api_key

        def _validate_config(self):
            return True

        def get_cache_config(self):
            return {"source_type": "broken", "api_key": self.api_key, "token_hint": f"ends {self.api_key[-8:]}"}

        def fetch(self, start, end):
            raise RuntimeError(f"HTTP 401 for token {self.api_key}")

    csv = tmp_path / "x.csv"
    csv.write_text("datetime,x\n2026-10-24 00:00,1\n")
    clock = FakeClock(local("2026-10-24 09:00"))
    pipeline = DataPipeline(
        sources=[Broken(SECRET), CsvSource(str(csv)), CalendarSource("BE", prefix="cal")], timezone=ZONE
    )
    archive = SnapshotArchive(tmp_path / "archive")
    with use_clock(clock):
        result = archive.record(pipeline)
    assert [line["status"] for line in result.lines] == ["failed", "skipped", "ok"]
    assert result.exit_code == 1 and len(result.failed) == 1 and len(result.ok) == 1
    failed = result.failed[0]
    assert failed["error"] == "RuntimeError: HTTP 401 for token ***" and "file" not in failed
    assert failed["config"] == {"source_type": "broken"}  # entries named like a secret are dropped
    assert list(archive.catalog()["status"]) == ["failed", "ok"]  # a skipped source writes nothing
    assert SECRET[-8:] not in archive.catalog_path.read_text()  # a name like a secret is dropped from the config
    with pytest.raises(NotImplementedError, match="step 8b"):
        archive.record(pipeline, budget=60)
    with pytest.raises(NotImplementedError, match="timeout=, retries="):
        archive.record(pipeline, timeout=30, retries=2)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="days_back"):
            archive.record(pipeline, days_back=bad)
    with pytest.raises(TypeError, match="DataPipeline"):
        archive.record([CalendarSource("BE")])


def test_two_recordings_at_one_instant_keep_both_in_order_and_a_torn_line_or_tampered_file_is_caught(tmp_path):
    clock = FakeClock(local("2026-10-24 09:00"))
    pipeline = DataPipeline(sources=[CalendarSource("BE")], timezone=ZONE)
    archive = SnapshotArchive(tmp_path)
    with use_clock(clock):
        archive.record(pipeline)
        archive.record(pipeline)  # the clock stood still
    files = list(archive.catalog()["file"])
    assert files[0].endswith("T070000000000Z.parquet") and files[1].endswith("T070000000000Z-2.parquet")
    assert [fetch.file for fetch in archive.fetches("CalendarSource")] == files
    with open(archive.catalog_path, "a") as handle:
        handle.write('{"source": "CalendarSource-')  # a crash while appending
    assert len(archive.catalog()) == 2
    (archive.path / files[1]).write_bytes(b"not the frame")
    with pytest.raises(ArchiveError, match="sha256"):
        archive.read(archive.fetches("CalendarSource")[1])
    with pytest.raises(ArchiveError, match="nothing has been recorded"):
        SnapshotArchive(tmp_path / "empty").catalog()


# --- secrets ---------------------------------------------------------------------------------------------------------------


def every_byte(path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file())


def test_no_byte_written_holds_the_api_key_planted_in_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTSOE_API_KEY", SECRET)
    clock = FakeClock(local("2026-10-24 08:00"))
    server = FakeEntsoe(clock, Truth([(local("2026-10-25 08:00"), local("2026-10-24 11:05"), 1.5)]))
    pipeline = DataPipeline(sources=[entsoe(server, os.environ["ENTSOE_API_KEY"])], timezone=ZONE)

    class Leaky(DataSource):
        """Puts the key in a frame's attrs, then in a column: the first is dropped, the second refused."""

        def __init__(self, column: bool):
            self.column = column

        def _validate_config(self):
            return True

        def get_cache_config(self):
            return {"source_type": "leaky", "column": self.column}

        def fetch(self, start, end):
            index = pd.date_range(start, periods=3, freq="1h")
            frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=index)
            frame.attrs["key"] = os.environ["ENTSOE_API_KEY"]
            if self.column:
                frame["note"] = os.environ["ENTSOE_API_KEY"]
            return frame

    pipeline.sources += [Leaky(False), Leaky(True)]
    archive = SnapshotArchive(tmp_path / "archive")
    lines = []
    with use_clock(clock):
        for text in ["2026-10-24 09:00", "2026-10-24 10:30", "2026-10-24 11:30"]:
            clock.advance((local(text) - clock.now()).total_seconds())
            lines.extend(archive.record(pipeline).lines)
    assert all(call["headers"]["SECURITY_TOKEN"] == SECRET for call in server.calls)  # the key was in use
    assert [line["status"] for line in lines] == ["ok", "ok", "failed"] * 3
    assert all("secret value" in line["error"] for line in lines[2::3])
    assert len(list((tmp_path / "archive").rglob("*.parquet"))) == 6
    written = every_byte(tmp_path / "archive")
    assert len(written) > 10_000 and SECRET.encode() not in written
    assert SECRET[:8].encode() not in written and SECRET[-12:].encode() not in written
    assert SECRET in source_secrets(pipeline.sources[0])
    assert all("api_key" not in line["config"] and "SECURITY_TOKEN" not in json.dumps(line) for line in lines)


@pytest.mark.parametrize("where", ["many values", "categorical", "column name", "index name"])
def test_a_frame_holding_a_secret_anywhere_is_refused(tmp_path, monkeypatch, where):
    monkeypatch.setenv("SOME_SERVICE_TOKEN", SECRET)

    class Holder(DataSource):
        def _validate_config(self):
            return True

        def get_cache_config(self):
            return {"source_type": "holder"}

        def fetch(self, start, end):
            index = pd.date_range(start, end, freq="15min", inclusive="left")
            frame = pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)
            if where == "many values":  # one dictionary entry, compressed: a byte search alone needn't see it
                frame["note"] = SECRET
            elif where == "categorical":
                frame["note"] = pd.Categorical([SECRET] * len(index))
            elif where == "column name":
                frame[f"x_{SECRET}"] = 1.0
            else:
                frame.index.name = SECRET
            return frame

    with use_clock(FakeClock(local("2026-10-24 09:00"))):
        result = SnapshotArchive(tmp_path).record(DataPipeline(sources=[Holder()], timezone=ZONE))
    assert result.exit_code == 1 and "secret value" in result.failed[0]["error"]
    assert not list(tmp_path.rglob("*.parquet"))
    assert SECRET.encode() not in every_byte(tmp_path)


def test_each_guard_refuses_on_its_own():
    # The frame's text, before it is written (compression may keep a string from showing verbatim in a file) ...
    for frame in (
        pd.DataFrame({"note": [f"key {SECRET}"] * 3}),
        pd.DataFrame({SECRET: [1.0]}),
        pd.DataFrame({"x": [1.0]}, index=pd.Index([0], name=SECRET)),
    ):
        with pytest.raises(SecretInArchiveError):
            _refuse_secrets_in_frame(frame, [SECRET], "f")
    _refuse_secrets_in_frame(pd.DataFrame({"x": [1.0], "when": [pd.Timestamp("2026-10-24")]}), [SECRET], "f")
    # ... and the bytes written, a file's and a line's.
    with pytest.raises(SecretInArchiveError):
        _refuse_secrets(b"PAR1..." + SECRET.encode() + b"...", [SECRET], "f")
    _refuse_secrets(b"PAR1...", [SECRET], "f")


# --- ArchiveSource -------------------------------------------------------------------------------------------------------


# Every day from the 20th to the 27th is recorded at 09:00, 10:30, 11:30 and 12:30 Brussels, but for the 11:30 of the 22nd and the
# 10:30 and 11:30 of the 24th (the day before the 25-hour day). Each delivery day's 08:00 is revised at 11:05 the day before
# (seen at 11:30), its 09:00 at 12:00 (seen at 12:30, after 11:45), and the second 02:00 of the 25th at 11:05 on the 24th.
SKIPPED = {"2026-10-22 11:30", "2026-10-24 10:30", "2026-10-24 11:30"}
SCHEDULE = [
    f"{day.date()} {time}"
    for day in pd.date_range("2026-10-20", "2026-10-27")
    for time in ("09:00", "10:30", "11:30", "12:30")
    if f"{day.date()} {time}" not in SKIPPED
]
DELIVERY = pd.date_range("2026-10-21", "2026-10-28")


def week_revisions():
    out = []
    for day in DELIVERY:
        before = (day - pd.Timedelta(days=1)).date()
        out.append((local(f"{day.date()} 08:00"), local(f"{before} 11:05"), -1.0 * day.day))
        out.append((local(f"{day.date()} 09:00"), local(f"{before} 12:00"), -2.0 * day.day))
    out.append((local("2026-10-25 02:00") + HOUR, local("2026-10-24 11:05"), -3.0))
    return out


@pytest.fixture(scope="module")
def week(tmp_path_factory):
    """The archive of the schedule above, the truth it was fetched from, and each recording's (fetched_at, start, end)."""
    path = tmp_path_factory.mktemp("week")
    truth = Truth(week_revisions())
    clock = FakeClock(local("2026-10-20 08:00"))
    pipeline = DataPipeline(sources=[entsoe(FakeEntsoe(clock, truth))], timezone=ZONE)
    with use_clock(clock):
        lines = recordings(SnapshotArchive(path), pipeline, clock, SCHEDULE)
    made = [
        (pd.Timestamp(line["fetched_at"]), pd.Timestamp(line["start"]), pd.Timestamp(line["end"])) for line in lines
    ]
    return path, truth, made


def replayed_by_hand(truth: Truth, made, cell: pd.Timestamp, known: pd.Timestamp) -> float:
    """The truth as the newest recording that returned by `known` and asked for `cell` fetched it; NaN without one."""
    taken = [at for at, start, end in made if at <= known and start <= cell < end]
    return truth.value(cell, max(taken)) if taken else np.nan


def cut_off(day: pd.Timestamp, known_at: str, days_before: int, zone: str) -> pd.Timestamp:
    """The description: `known_at` on `day` - `days_before` in `zone`, a repeated time at its second occurrence, a skipped one
    at the switch."""
    wall = pd.Timestamp(day.date()) - pd.Timedelta(days=days_before) + pd.Timedelta(known_at + ":00")
    return wall.tz_localize(zone, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")


def cut_offs(days: pd.DatetimeIndex, known_at: str, days_before: int, zone: str) -> np.ndarray:
    """`cut_off` of many naive dates (NaT stays NaT), as naive UTC datetime64."""
    wall = days.normalize() - pd.Timedelta(days=days_before) + pd.Timedelta(known_at + ":00")
    instants = wall.tz_localize(zone, ambiguous=np.zeros(len(wall), dtype=bool), nonexistent="shift_forward")
    return instants.tz_convert("UTC").tz_localize(None).to_numpy()


def library_known(info: InformationSet, name: str, index: pd.DatetimeIndex, freq: str, days: pd.DatetimeIndex):
    """[period, issue day] -> when the information set says the value of `name` on that row is known (naive UTC), and the
    cells the clock skips."""
    clock = info._issue_clock(days, index, freq)
    step = _STEP_MINUTES[freq]
    periods = np.arange(periods_per_day(freq))
    avail = clock.available(info._resolve_for(name, step), periods, step)
    origin = clock.ctx.origin.tz_convert("UTC").tz_localize(None)
    minutes = np.where(np.isfinite(avail), avail, np.nan).ravel()
    known = (origin + pd.to_timedelta(minutes, unit="min")).to_numpy().reshape(avail.shape)
    return known, clock.skipped(periods, step)


def test_the_replay_gives_each_day_the_newest_fetch_by_known_at_the_day_before(week):
    path, truth, made = week
    source = ArchiveSource(path, "EntsoeSource", known_at="11:45", days_before=1, columns=["load_forecast"])
    start, end = local("2026-10-21 00:00"), local("2026-10-29 00:00")
    frame = source.fetch(start, end)
    assert list(frame.columns) == ["archive_load_forecast"]
    cells = pd.date_range(start, end, freq="1h", inclusive="left")
    expected = pd.Series(
        [replayed_by_hand(truth, made, cell, cut_off(cell.tz_convert(ZONE), "11:45", 1, ZONE)) for cell in cells],
        index=cells,
    )
    got = frame["archive_load_forecast"].reindex(cells)
    pd.testing.assert_series_equal(got, expected, check_names=False, check_freq=False)
    by_day = got.groupby(cells.tz_convert(ZONE).date)
    assert by_day.size().to_dict()[pd.Timestamp("2026-10-25").date()] == 25
    # The 25th: read at 11:45 on the 24th, when only the 09:00 recording had returned, before the 10:00 publication.
    assert by_day.count()[pd.Timestamp("2026-10-25").date()] == 0
    # The 23rd: from the 10:30 recording of the 22nd (11:30 skipped), before the 11:05 revision of its 08:00.
    assert got[local("2026-10-23 08:00")] == base_value(local("2026-10-23 08:00"))
    # Every other day: the 11:05 revision is in, the 12:00 one isn't.
    for day in ("2026-10-21", "2026-10-22", "2026-10-24", "2026-10-26", "2026-10-27", "2026-10-28"):
        assert got[local(f"{day} 08:00")] == -1.0 * pd.Timestamp(day).day
        assert got[local(f"{day} 09:00")] == base_value(local(f"{day} 09:00"))
    # A later known_at sees the 12:00 revisions and the 25th; the 12:30 recordings never miss.
    late = ArchiveSource(path, "EntsoeSource", known_at="13:00", columns=["load_forecast"], prefix="").fetch(start, end)
    assert late.loc[local("2026-10-26 09:00"), "load_forecast"] == -2.0 * 26
    assert late.loc[local("2026-10-25 02:00") + HOUR, "load_forecast"] == -3.0
    assert late["load_forecast"].notna().all() and len(late) == len(cells)


def test_a_fetch_that_returned_exactly_at_the_cut_off_counts_and_one_a_microsecond_later_doesn_t(tmp_path):
    class Counter(DataSource):
        """Each fetch's value is its number: 1, 2, 3, ..."""

        def __init__(self):
            self._n = 0

        def _validate_config(self):
            return True

        def get_cache_config(self):
            return {"source_type": "counter"}

        def fetch(self, start, end):
            self._n += 1
            return pd.DataFrame({"x": float(self._n)}, index=pd.date_range(start, end, freq="1h", inclusive="left"))

    clock = FakeClock(local("2026-10-23 11:44"))
    pipeline = DataPipeline(sources=[Counter()], timezone=ZONE)
    archive = SnapshotArchive(tmp_path)
    with use_clock(clock):
        archive.record(pipeline)  # 11:44: fetch 1
        clock.advance(60)
        archive.record(pipeline)  # 11:45:00 exactly: fetch 2
        clock.advance(1e-6)
        archive.record(pipeline)  # a microsecond later: fetch 3
    source = ArchiveSource(tmp_path, "Counter", known_at="11:45", days_before=1)
    frame = source.fetch(local("2026-10-24 00:00"), local("2026-10-25 00:00"))
    assert len(frame) == 24 and (frame["archive_x"] == 2.0).all()
    # Read at 11:46, the third fetch is in.
    assert (
        ArchiveSource(tmp_path, "Counter", "11:46", 1).fetch(local("2026-10-24 00:00"), local("2026-10-24 01:00"))[
            "archive_x"
        ]
        == 3.0
    ).all()


def test_the_replay_holds_for_any_known_at_days_before_and_zone(week):
    path, truth, made = week
    start, end = local("2026-10-21 00:00"), local("2026-10-29 00:00")
    cells = pd.date_range(start, end, freq="1h", inclusive="left")
    for known_at, days_before, zone, out in [
        ("10:15", 1, ZONE, False),  # only the 09:00 recordings by then, before the 10:00 publication: nothing
        ("10:45", 1, ZONE, True),  # the 10:30 recordings, before the 11:05 revisions
        ("02:30", 0, ZONE, True),  # the 25th's 02:30 repeats: its second occurrence
        ("23:00", 1, "Europe/Helsinki", True),
        ("23:00", 2, "Europe/Helsinki", False),  # two days ahead is never out
        ("12:00", -1, "Europe/Lisbon", True),
        # A Helsinki day starts at 23:00 Brussels, before the newest fetch's range (whole Brussels days): its first hour is
        # the older fetch's, which asked for it.
        ("12:00", -3, "Europe/Helsinki", True),
    ]:
        source = ArchiveSource(path, "EntsoeSource", known_at, days_before, zone, columns=["load_forecast"])
        frame = source.fetch(start, end)
        assert (frame["archive_load_forecast"].notna().sum() > 100) == out
        for cell in cells:
            day = cell.tz_convert(zone)
            assert source.known_instant(day) == cut_off(day, known_at, days_before, zone)
            want = replayed_by_hand(truth, made, cell, cut_off(day, known_at, days_before, zone))
            have = frame["archive_load_forecast"].get(cell, np.nan)
            assert (np.isnan(want) and np.isnan(have)) or want == have, (known_at, days_before, zone, cell, want, have)
        # A sub-range holds the same values (the source contract).
        part = source.fetch(local("2026-10-24 13:00"), local("2026-10-26 05:00"))
        whole = frame.loc[local("2026-10-24 13:00") : local("2026-10-26 04:00")]
        pd.testing.assert_frame_equal(part, whole, check_freq=False)


def test_the_information_set_knows_each_replayed_cell_exactly_when_the_replay_read_it(week):
    path, truth, made = week
    source = ArchiveSource(path, "EntsoeSource", known_at="11:45", days_before=1)
    assert source.column_names() == ["archive_load_forecast"]
    info = InformationSet("11:45", ZONE).with_source_rules([source])
    assert info.resolve("archive_load_forecast").rule == Availability.published(
        "11:45", 1, timezone=ZONE, vintage="snapshot", note=source.availability().note
    )
    frame = source.fetch(local("2026-10-21 00:00"), local("2026-10-29 00:00"))
    index = pd.date_range(local("2026-10-21 00:00"), local("2026-10-29 00:00"), freq="1h", inclusive="left").tz_convert(
        ZONE
    )
    days = pd.date_range("2026-10-22", "2026-10-27")
    for name, shift in (
        ("archive_load_forecast", 0),
        ("archive_load_forecast_d+1", 1),
        ("archive_load_forecast_d-1", -1),
    ):
        known, skipped = library_known(info, name, index, "1h", days)
        for column, day in enumerate(days):
            delivery = day + pd.Timedelta(days=shift)
            truth_known = cut_off(delivery, "11:45", 1, ZONE).tz_localize(None)
            assert (known[~skipped[:, column], column] == truth_known.to_datetime64()).all(), (name, day)
    # And so what a row of the issue day reads was in hand by then: a row of D reads `_d+1`, known at 11:45 on D.
    for day in days:
        delivery = day + pd.Timedelta(days=1)
        cells = index[index.normalize().tz_localize(None) == delivery].tz_convert("UTC")
        assert len(cells) in (23, 24, 25)
        for cell in cells:
            want = replayed_by_hand(truth, made, cell, local(f"{day.date()} 11:45"))
            have = frame["archive_load_forecast"].get(cell, np.nan)
            assert (np.isnan(want) and np.isnan(have)) or want == have


ZONES = ["Europe/Brussels", "Europe/Helsinki", "Europe/Lisbon", "UTC"]
WINDOWS = {"spring": ("2024-03-28", "2024-04-02"), "autumn": ("2024-10-24", "2024-10-29")}
NAMES = {"col": 0, "col_d+1": 1, "col_d+2": 2, "col_d-1": -1}
KNOWN = ["11:45", "02:30", "00:00", "23:30"]
DAYS_BEFORE = [1, 0, -1, 2]


@pytest.mark.parametrize(("archive_zone", "frame_zone"), [(a, f) for a in ZONES for f in ZONES])
def test_an_archive_column_is_known_at_known_at_on_the_day_before_its_delivery_day(archive_zone, frame_zone):
    """The oracle: a row of issue day D at period p reads `col_d+k` at the same wall time k days later in the frame's zone; that
    cell is delivered on its date in the archive's zone, and known at `known_at` on that date minus `days_before` there. Every
    days_before with every zone pair, the known_at times cycling (02:30 is skipped in spring and repeated in autumn), both
    switches, hourly and 15-minute."""
    from epftoolbox2._date_utils import shift_calendar_days

    pair = ZONES.index(archive_zone) * len(ZONES) + ZONES.index(frame_zone)
    freq = "15min" if pair % 3 == 0 else "1h"
    for i, days_before in enumerate(DAYS_BEFORE):
        known_at = KNOWN[(pair + i) % len(KNOWN)]
        source = ArchiveSource("unused", "X", known_at, days_before, archive_zone, columns=["col"], prefix="")
        info = InformationSet("11:45", "Europe/Brussels").with_source_rules([source])
        for window, (first, last) in WINDOWS.items():
            days = pd.date_range(first, last, freq="D")
            index = pd.date_range(
                (days[0] - pd.Timedelta(days=4)).tz_localize(frame_zone),
                (days[-1] + pd.Timedelta(days=4)).tz_localize(frame_zone),
                freq=freq,
                inclusive="left",
            )
            step = _STEP_MINUTES[freq]
            periods = np.arange(periods_per_day(freq))
            wall = pd.DatetimeIndex([day + pd.Timedelta(minutes=int(p * step)) for p in periods for day in days])
            rows = wall.tz_localize(frame_zone, ambiguous=np.zeros(len(wall), dtype=bool), nonexistent="NaT")
            for name, k in NAMES.items():
                known, skipped = library_known(info, name, index, freq, days)
                value_at = shift_calendar_days(rows, k) if k else rows
                delivered = value_at.tz_convert(archive_zone).tz_localize(None).normalize()
                truth = cut_offs(delivered, known_at, days_before, archive_zone).reshape(len(periods), len(days))
                judged = ~skipped & ~np.isnat(truth)
                assert judged.sum() > 0.9 * judged.size
                early, late = judged & (known < truth), judged & (known > truth)
                where = (window, name, known_at, days_before)
                assert not early.any(), (*where, "early", np.argwhere(early)[:3])
                assert not late.any(), (*where, "late", np.argwhere(late)[:3])
                for d in pd.unique(delivered[~pd.isna(delivered)]):
                    assert source.known_instant(d) == cut_off(pd.Timestamp(d), known_at, days_before, archive_zone)


def test_an_archive_column_s_rule_comes_before_every_other_and_a_rule_naming_it_raises():
    source = ArchiveSource(
        "unused", "EntsoeSource", "11:45", 1, ZONE, columns=["load_forecast", "load_actual"], prefix=""
    )
    rule = source.availability()
    assert rule.vintage == "snapshot" and (rule.time, rule.days_before, rule.timezone) == ("11:45", 1, ZONE)
    # The built-in rules of these names (A65, an actual), a pattern of yours, an assumption: none replaces the archive's.
    plain = InformationSet("11:45", ZONE)
    assert plain.resolve("load_forecast").rule != rule
    info = InformationSet("11:45", ZONE, rules={"load_*": Availability.static()}).with_source_rules([source])
    for name in (
        "load_forecast",
        "load_forecast_d+1",
        "load_actual_d-1",
        "load_forecast_day_max",
        "load_forecast_h05_d+1",
    ):
        resolved = info.resolve(name)
        assert resolved.rule == rule and resolved.layer == "sources", name
    for name in ("load_forecast", "load_forecast_d+1", "load_actual_day_max"):
        with pytest.raises(InformationSetError, match="nothing replaces it"):
            InformationSet("11:45", ZONE, rules={name: Availability.static()}).with_source_rules([source])
    # Prefixed (the default), its names are its own: no built-in rule names them, so outside with_source_rules they have none.
    prefixed = ArchiveSource("unused", "EntsoeSource", columns=["load_forecast"])
    assert prefixed.column_names() == ["archive_load_forecast"]
    assert plain.resolve("archive_load_forecast").rule is None
    assert (
        InformationSet().with_source_rules([prefixed]).resolve("archive_load_forecast_d+1").rule
        == prefixed.availability()
    )
    # Other sources' columns resolve as before.
    assert info.resolve("price").rule == plain.resolve("price").rule
    # A rule added after with_source_rules would be ignored: the run's checks refuse it.
    from epftoolbox2.models import _chained

    info.rules["load_forecast_d+1"] = Availability.static()
    frame = pd.DataFrame({"load_forecast": [1.0]}, index=pd.date_range("2026-10-24", periods=1, freq="1h", tz=ZONE))
    with pytest.raises(InformationSetError, match="nothing replaces it"):
        _chained.check("m", frame, {(0, 1): ["load_forecast_d+1"]}, info, "warn", None, None)


def test_an_archive_source_in_a_pipeline_and_its_refusals(week, tmp_path):
    path, _, _ = week
    source = ArchiveSource(path, "EntsoeSource", columns=["load_forecast"])
    pipeline = DataPipeline(
        sources=[source, CalendarSource("BE")],
        transformers=[TimezoneTransformer(), LagTransformer(columns=["archive_load_forecast"], lags=[-1], freq="day")],
        timezone=ZONE,
    )
    frame = pipeline.run("2026-10-22", "2026-10-27")
    assert str(frame.index.tz) == ZONE and "archive_load_forecast_d+1" in frame
    row = pd.Timestamp("2026-10-23 08:00", tz=ZONE)
    assert frame.loc[row, "archive_load_forecast_d+1"] == -24.0  # the 24th's 08:00 as revised at 11:05 on the 23rd
    assert source.get_cache_config() is None and source.content_digest() == SnapshotArchive(path).content_digest()
    with pytest.raises(ArchiveError, match="no fetch .* holds \\['nope'\\]"):
        ArchiveSource(path, "EntsoeSource", columns=["nope"]).fetch(
            local("2026-10-22 00:00"), local("2026-10-23 00:00")
        )
    with pytest.raises(ArchiveError, match="no source 'PseSource'"):
        ArchiveSource(path, "PseSource").fetch(local("2026-10-22 00:00"), local("2026-10-23 00:00"))
    for kwargs, message in [
        ({"known_at": "1145"}, "HH:MM"),
        ({"days_before": 1.5}, "days_before"),
        ({"days_before": True}, "days_before"),
        ({"timezone": "Mars/Olympus"}, "Unknown time zone"),
        ({"timezone": None}, "timezone"),
        ({"columns": []}, "columns"),
        ({"prefix": "a_"}, "prefix"),
    ]:
        with pytest.raises(ValueError, match=message):
            ArchiveSource(path, "EntsoeSource", **kwargs)
    # Two configurations of one class: the class name no longer names one.
    clock = FakeClock(local("2026-10-28 09:00"))
    other = EntsoeSource("BE", SECRET, type=["load"], prefix="be")
    other.session = FakeEntsoe(clock, Truth())
    copy = tmp_path / "two"
    copy.mkdir()
    for p in path.rglob("*"):
        if p.is_file():
            (copy / p.relative_to(path)).parent.mkdir(parents=True, exist_ok=True)
            (copy / p.relative_to(path)).write_bytes(p.read_bytes())
    with use_clock(clock):
        SnapshotArchive(copy).record(DataPipeline(sources=[other], timezone=ZONE))
    with pytest.raises(ArchiveError, match="2 configurations of EntsoeSource"):
        ArchiveSource(copy, "EntsoeSource").fetch(local("2026-10-22 00:00"), local("2026-10-23 00:00"))
    key = archive_key(other)
    assert ArchiveSource(copy, key, prefix="").column_names() == ["be_load_forecast"]


def test_yaml_round_trip_and_the_scheduled_job_function(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    archive = tmp_path / "archive"
    source = ArchiveSource(
        "archive", "CalendarSource", known_at="12:30", days_before=0, timezone=ZONE, columns=["is_holiday"]
    )
    DataPipeline(sources=[source]).save(tmp_path / "replay.yaml")
    written = yaml.safe_load((tmp_path / "replay.yaml").read_text())["sources"][0]
    assert written == {
        "class": "ArchiveSource",
        "params": {
            "path": "archive",
            "source": "CalendarSource",
            "known_at": "12:30",
            "days_before": 0,
            "timezone": ZONE,
            "columns": ["is_holiday"],
            "prefix": "archive",
        },
    }
    loaded = DataPipeline.load(tmp_path / "replay.yaml").sources[0]
    assert isinstance(loaded, ArchiveSource) and loaded.path == str(archive)
    assert (loaded.source, loaded.known_at, loaded.days_before, loaded.columns) == (
        "CalendarSource",
        "12:30",
        0,
        ["is_holiday"],
    )

    # The function a scheduled job (and `epf snapshot`) calls: a DataPipeline YAML, or a Workflow's data pipeline.
    DataPipeline(sources=[CalendarSource("BE")], timezone=ZONE).save(tmp_path / "data.yaml")
    workflow = {
        "data_pipeline": {"path": "data.yaml", "start": "2026-01-01", "end": "2026-02-01"},
        "model_pipeline": {"test_start": "2026-01-15", "test_end": "2026-02-01", "target": "price"},
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(workflow))
    clock = FakeClock(local("2026-10-24 09:00"))
    with use_clock(clock):
        first = record_snapshot(tmp_path / "data.yaml", archive)
        clock.advance(900)
        second = record_snapshot(tmp_path / "experiment.yaml", "archive", days_back=0, days_ahead=0)
    assert first.exit_code == 0 and second.exit_code == 0
    assert "CalendarSource-" in first.summary() and "ok (" in first.summary()
    catalog = SnapshotArchive(archive).catalog()
    assert list(catalog["start"]) == [local("2026-10-21 00:00"), local("2026-10-24 00:00")]
    assert list(catalog["end"]) == [local("2026-10-27 00:00"), local("2026-10-25 00:00")]
    frame = loaded.fetch(local("2026-10-24 00:00"), local("2026-10-26 00:00"))
    assert list(frame.columns) == ["archive_is_holiday"]
    # The 24th was fetched by 12:30 on the 24th, the 25th too (days_before 0 reads each day on itself).
    assert frame.index.min() == local("2026-10-24 00:00") and len(frame) == 49


def test_secrets_of_a_source_and_the_environment(monkeypatch):
    for name in list(os.environ):
        if name.upper().endswith(("API_KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "APIKEY")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "t0ken-value-123")
    monkeypatch.setenv("SHORT_TOKEN", "abc")  # too short to search for
    monkeypatch.setenv("TOKENIZER_PATH", "/not/a/secret")

    class WithProxy(DataSource):
        def __init__(self):
            self.proxy_url = "http://user:hunter2-pass@proxy.example:8080"
            self.password = "p4ssw0rd-x"
            self.model = "not-a-secret"

        def _validate_config(self):
            return True

        def get_cache_config(self):
            return {"source_type": "proxy"}

        def fetch(self, start, end):
            return pd.DataFrame()

    found = source_secrets(WithProxy())
    assert "t0ken-value-123" in found and "p4ssw0rd-x" in found and "hunter2-pass" in found
    assert "user:hunter2-pass" in found and "abc" not in found and "/not/a/secret" not in found
    assert "not-a-secret" not in found
    assert json.dumps(found)  # plain strings
