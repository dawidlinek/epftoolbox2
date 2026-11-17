"""The day-ahead load forecast (ENTSO-E A65/A01) and wind and solar forecast (A69) are delivered per local day of each area.

Every TSO submits these per its area's own local day: each Period's timeInterval starts at the area's local midnight (FI and GR
at 22:00Z in winter and 21:00Z in summer, PT and IE at 00:00Z and 23:00Z, for A65 and A69 alike; the live responses are in
`tests/data/fixtures/entsoe_live_checks/`). So the built-in rules read the delivery day in the data's local dates
(`Availability.published(..., day_zone="data")`), an `EntsoeSource`'s rule in its area's zone, and the publication instant
stays 10:00 (A65) or 18:00 (A69) Brussels on the date before. They used to read a Brussels day: east of Brussels the first hour
of a day (Helsinki's 00:00 is 23:00 the day before in Brussels) was taken as published a day early, a look-ahead of one hour a
day, and west of it the last hour (Lisbon's 23:00 is the next day's 00:00 in Brussels) a day late. In every zone on Brussels'
clock (DE_LU, AT, PL) nothing moves, to the bit.

The truth is written from that description alone (`published` below) and compared with the cells the check reads.
"""

import datetime
import json
import re
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.sources.entsoe import EntsoeSource
from epftoolbox2.information_set import (
    _STEP_MINUTES,
    ASSUMPTIONS,
    Availability,
    InformationSet,
    InformationSetError,
    default_rules,
)
from epftoolbox2.models import NaiveModel
from epftoolbox2.models._calendar import periods_per_day
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import DataPipeline, ModelPipeline

FIXTURES = Path(__file__).parent / "data" / "fixtures" / "entsoe_live_checks"
LOAD = "load_forecast"
WIND = "generation_wind_onshore_forecast"
# The rules as they were: a Brussels delivery day.
BEFORE = {
    LOAD: Availability.published("10:00", 1, vintage="latest_revision"),
    "generation_*_forecast": Availability.published("18:00", 1, vintage="latest_revision"),
}
TIMES = {LOAD: "10:00", WIND: "18:00"}
EAST = {"FI": "Europe/Helsinki", "RO": "Europe/Bucharest", "BG": "Europe/Sofia", "GR": "Europe/Athens"}
WEST = {"PT": "Europe/Lisbon", "IE": "Europe/Dublin"}
CET = {"DE_LU": "Europe/Berlin", "AT": "Europe/Vienna", "PL": "Europe/Warsaw"}


def published(column: str, delivery: datetime.date) -> pd.Timestamp:
    """When the value of the local delivery date `delivery` is out: 10:00 (A65) or 18:00 (A69) Brussels on the date before."""
    return pd.Timestamp(f"{delivery - datetime.timedelta(days=1)} {TIMES[column]}", tz="Europe/Brussels")


def issued(day: datetime.date, time: str, zone: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {time}", tz=zone)


def frame_index(zone: str, freq: str, first: str, last: str) -> pd.DatetimeIndex:
    """Rows from 3 days before `first` to 4 after `last`, at noon for a daily frame."""
    start = pd.Timestamp(first) - pd.Timedelta(days=3)
    end = pd.Timestamp(last) + pd.Timedelta(days=4)
    if freq == "1D":
        return (pd.date_range(start, end, freq="D") + pd.Timedelta(hours=12)).tz_localize(zone)
    return pd.date_range(start.tz_localize(zone), end.tz_localize(zone), freq=freq, inclusive="left")


def available(info: InformationSet, name: str, index: pd.DatetimeIndex, freq: str, first: str, last: str):
    """The check's availability instants of `name` as [period, issue day], the issue instants, and the skipped cells."""
    days = pd.date_range(first, last, freq="D")
    clock = info._issue_clock(days, index, freq)
    step = _STEP_MINUTES[freq]
    periods = np.arange(periods_per_day(freq))
    return clock.available(info._resolve_for(name, step), periods, step), clock.issue, clock.skipped(periods, step)


def known_cells(info, name, zone, freq, first, last, frame_zone=None):
    index = frame_index(zone, freq, first, last)
    if frame_zone is not None:
        index = index.tz_convert(frame_zone)
    found, issue, skipped = available(info, name, index, freq, first, last)
    return found <= issue[None, :], skipped


def by_hand(column, area_zone, zone, freq, first, last, horizon, hours=0, issue="12:00", issue_zone=None):
    """The truth as [period, issue day] for `column_d+{horizon}` (with `hours` an elapsed `_h+k` shift after it) on a frame of
    `zone`'s clock: the slot's instant, its local date in the area's zone, and that date's publication against the issue.
    A daily row's value is its whole date, read at the date's last slot."""
    days = pd.date_range(first, last, freq="D").date
    step = _STEP_MINUTES[freq]
    n = periods_per_day(freq)
    out = np.zeros((n, len(days)), dtype=bool)
    for d, day in enumerate(days):
        for p in range(n):
            wall = pd.Timestamp(day + datetime.timedelta(days=horizon))
            wall += pd.Timedelta(hours=23) if freq == "1D" else pd.Timedelta(minutes=p * step)
            instant = wall.tz_localize(zone, ambiguous=False, nonexistent="shift_forward") + pd.Timedelta(hours=hours)
            delivery = instant.tz_convert(area_zone).date()
            out[p, d] = published(column, delivery) <= issued(day, issue, issue_zone or zone)
    return out


# ---------------------------------------------------------------------------
# The fixtures: every Period starts at the area's own local midnight
# ---------------------------------------------------------------------------

DELIVERY = sorted(path.stem for path in FIXTURES.glob("delivery_*.json"))


def test_the_live_check_covers_a65_and_a69_of_four_zones_in_winter_and_summer():
    assert len(DELIVERY) == 16
    assert {name.split("_")[1] for name in DELIVERY} == {"FI", "GR", "PT", "IE"}


@pytest.mark.parametrize("fixture", DELIVERY)
def test_every_period_starts_at_the_area_s_own_midnight(fixture):
    """What the rules rest on: the period of local day X starts at X's midnight in the zone `EntsoeSource` gives the area."""
    summary = json.loads((FIXTURES / f"{fixture}.json").read_text())
    zone = EntsoeSource(summary["area"], api_key="test-key", type=["load"])._timezone
    assert zone == summary["zone"]
    midnight = pd.Timestamp(summary["day"]).tz_localize(zone).tz_convert("UTC")
    assert summary["periods"]
    for period in summary["periods"]:
        assert pd.Timestamp(period["time_interval_start"]) == midnight
        assert pd.Timestamp(period["time_interval_end"]) == (
            (pd.Timestamp(summary["day"]) + pd.Timedelta(days=1)).tz_localize(zone).tz_convert("UTC")
        )
    # ... and not at Brussels' midnight, which is another instant in all four zones
    assert midnight != pd.Timestamp(summary["day"]).tz_localize("Europe/Brussels").tz_convert("UTC")


def fixture_case(fixture: str):
    """The fixture's area, zone, delivery day, column and an issue the old rule got wrong: for a zone east of Brussels the first
    slot of the day (a day early before), for one west of it the last slot (a day late before)."""
    summary = json.loads((FIXTURES / f"{fixture}.json").read_text())
    column = LOAD if summary["params"]["documentType"] == "A65" else WIND
    zone, delivery = summary["zone"], datetime.date.fromisoformat(summary["day"])
    east = summary["area"] in ("FI", "GR")
    if east:
        # Its 00:00 is 23:00 of the day before in Brussels. A65: issued two days before at noon, `_d+2`; A69: the day before
        # at noon (11:00 Brussels), `_d+1`, before the 18:00 publication of the local day, after that of the Brussels day.
        horizon = 2 if column == LOAD else 1
        slot = pd.Timestamp(summary["periods"][0]["time_interval_start"])
        issue = "12:00"
    else:
        # Its 23:00 is 00:00 of the next day in Brussels: issued the day before, `_d+1`, at noon (A65) or 20:00 (A69).
        horizon = 1
        slot = pd.Timestamp(summary["periods"][0]["time_interval_end"]) - pd.Timedelta(hours=1)
        issue = "12:00" if column == LOAD else "20:00"
    freq = "15min" if east else "1h"
    period = (
        int((slot.tz_convert(zone) - pd.Timestamp(delivery).tz_localize(zone)).total_seconds() // 60)
        // (_STEP_MINUTES[freq])
    )
    return summary["area"], zone, delivery, column, horizon, slot, issue, freq, period, east


@pytest.mark.parametrize("fixture", DELIVERY)
def test_the_fixture_s_slot_is_known_exactly_when_its_local_day_is_published(fixture):
    """FI and GR: the slot at local 00:00 of day X isn't known at the issue on X-2 (A65 `_d+2`) or X-1 (A69 `_d+1`), and the
    Brussels day called it known: the look-ahead. PT and IE: the slot at local 23:00 of X is known on X-1, and the Brussels day
    called it unpublished: too strict."""
    area, zone, delivery, column, horizon, slot, issue, freq, period, east = fixture_case(fixture)
    issue_day = delivery - datetime.timedelta(days=horizon)
    index = pd.date_range(
        pd.Timestamp(issue_day - datetime.timedelta(days=5)).tz_localize(zone),
        pd.Timestamp(delivery + datetime.timedelta(days=3)).tz_localize(zone),
        freq=freq,
        inclusive="left",
    )
    # The cell (period, horizon) of the issue day's row is that very slot.
    row = pd.Timestamp(issue_day).tz_localize(zone) + pd.Timedelta(minutes=period * _STEP_MINUTES[freq])
    assert row + pd.DateOffset(days=horizon) == slot
    truth = published(column, delivery) <= issued(issue_day, issue, zone)
    assert truth is (not east)
    name = f"{column}_d+{horizon}"
    expanded = {(period, horizon): [name]}

    def check(info):
        info.check(expanded, "load_actual", freq, index, str(issue_day), str(issue_day))

    source = EntsoeSource(area, api_key="test-key", type=["load", "generation"])
    for info in (
        InformationSet(issue, zone),  # the built-in rule, issued on the market's clock
        InformationSet(issue, "Europe/Brussels").with_source_rules([source]),  # the source's, on Brussels' clock
    ):
        if truth:
            check(info)
        else:
            with pytest.raises(InformationSetError, match=re.escape(name)):
                check(info)
    before = InformationSet(issue, zone, rules=BEFORE)
    if east:
        check(before)  # the look-ahead: called known before its local day was published
    else:
        with pytest.raises(InformationSetError, match=re.escape(name)):
            check(before)


# ---------------------------------------------------------------------------
# Every cell against the truth
# ---------------------------------------------------------------------------

WINDOWS = {
    "january": ("2025-01-06", "2025-01-26"),
    "spring": ("2025-03-24", "2025-04-06"),  # the EU switch on 30 March
    "autumn": ("2025-10-20", "2025-11-02"),  # the EU switch on 26 October
}
ZONES = ["Europe/Helsinki", "Europe/Athens", "Europe/Bucharest", "Europe/Lisbon", "Europe/Dublin", "Europe/Warsaw"]
FREQS = ["1D", "1h", "15min"]
# Every pair of levels of zone, frequency and window in 18 cells: the window is (zone + frequency) mod 3.
GRID = [(zone, freq, list(WINDOWS)[(z + f) % 3]) for z, zone in enumerate(ZONES) for f, freq in enumerate(FREQS)]
NAMES = [(LOAD, 0, 0), (LOAD, 1, 0), (LOAD, 2, 0), (LOAD, 0, 35), (WIND, 1, 0), (WIND, 0, 12)]


@pytest.mark.parametrize(("zone", "freq", "window"), GRID)
def test_the_built_in_rules_publish_each_market_s_own_day(zone, freq, window):
    info = InformationSet("12:00", zone)
    first, last = WINDOWS[window]
    for column, horizon, hours in NAMES:
        if hours and freq == "1D":
            continue
        name = f"{column}_d+{horizon}" if horizon else column
        name = f"{name}_h+{hours}" if hours else name
        known, skipped = known_cells(info, name, zone, freq, first, last)
        expected = by_hand(column, zone, zone, freq, first, last, horizon, hours)
        wrong = np.argwhere((known != expected) & ~skipped)
        assert not len(wrong), (name, [(int(p), int(d), bool(known[p, d])) for p, d in wrong[:5]])


@pytest.mark.parametrize("area", [*EAST, *WEST, *CET])
@pytest.mark.parametrize("frame_zone", ["UTC", "Europe/Brussels", None])
def test_the_source_s_rule_is_exact_on_a_frame_in_any_zone(area, frame_zone):
    """An `EntsoeSource`'s rule reads the delivery day in its area's zone, so it holds on a UTC frame, or a frame on Brussels'
    clock, issued at noon Brussels: where the data's dates alone can't name the area's day."""
    zone = {**EAST, **WEST, **CET}[area]
    source = EntsoeSource(area, api_key="test-key", type=["load", "generation"])
    info = InformationSet("12:00", "Europe/Brussels").with_source_rules([source])
    first, last = WINDOWS["spring"]
    shown = frame_zone or zone
    for column, horizon in ((LOAD, 1), (LOAD, 2), (WIND, 1)):
        known, skipped = known_cells(info, f"{column}_d+{horizon}", zone, "1h", first, last, frame_zone)
        # The frame's rows are dated in its own zone: the truth reads each row's instant there.
        expected = by_hand(column, zone, shown, "1h", first, last, horizon, issue_zone="Europe/Brussels")
        wrong = np.argwhere((known != expected) & ~skipped)
        assert not len(wrong), (column, horizon, [(int(p), int(d)) for p, d in wrong[:5]])


# ---------------------------------------------------------------------------
# What moved: the look-ahead east of Brussels, the strictness west of it; nothing on Brussels' clock
# ---------------------------------------------------------------------------

YEAR = ("2025-01-01", "2025-12-31")  # both switches


def year_counts(zone: str, freq: str, name: str):
    before, skipped = known_cells(InformationSet("12:00", zone, rules=BEFORE), name, zone, freq, *YEAR)
    after, _ = known_cells(InformationSet("12:00", zone), name, zone, freq, *YEAR)
    return before & ~skipped, after & ~skipped


@pytest.mark.parametrize("zone", list(EAST.values()))
@pytest.mark.parametrize(("freq", "per_day"), [("1h", 1), ("15min", 4)])
def test_east_of_brussels_the_first_hour_of_a_day_is_no_longer_read_a_day_early(zone, freq, per_day):
    """Issued at noon on each day of 2025: A65 `_d+2` and A69 `_d+1` were called known in the first hour of the day (one hour
    a day: 365 cells at 1h, 1460 at 15min), published a day after the issue. Nothing else moves, nothing becomes known."""
    for name in (f"{LOAD}_d+2", f"{WIND}_d+1"):
        before, after = year_counts(zone, freq, name)
        removed = before & ~after
        assert int(removed.sum()) == 365 * per_day, name
        assert set(np.argwhere(removed)[:, 0].tolist()) == set(range(per_day)), name  # the local hour from 00:00
        assert not (after & ~before).any(), name
    for name in (LOAD, f"{LOAD}_d+1", f"{WIND}_d+2"):  # D's own day, D+1's whole day at 10:00, nothing of D+2 at 18:00
        before, after = year_counts(zone, freq, name)
        assert np.array_equal(before, after), name


@pytest.mark.parametrize("zone", list(WEST.values()))
@pytest.mark.parametrize(("freq", "per_day"), [("1h", 1), ("15min", 4)])
def test_west_of_brussels_the_last_hour_of_a_day_is_no_longer_read_a_day_late(zone, freq, per_day):
    """Issued at noon on the market's clock: A65 `_d+1` in the last hour of the day (Lisbon's and Dublin's 23:00 is the next
    day in Brussels) is out since 10:00 Brussels, and used to be called unpublished. Nothing becomes unknown."""
    before, after = year_counts(zone, freq, f"{LOAD}_d+1")
    added = after & ~before
    assert int(added.sum()) == 365 * per_day
    assert set(np.argwhere(added)[:, 0].tolist()) == set(range(periods_per_day(freq) - per_day, periods_per_day(freq)))
    assert not (before & ~after).any()


CET_NAMES = [
    LOAD,
    f"{LOAD}_d+1",
    f"{LOAD}_d+2",
    f"{LOAD}_d-1",
    f"{LOAD}_d-7",
    f"{LOAD}_h+35",
    f"{LOAD}_h-1",
    f"{LOAD}_d+1_h-1",
    WIND,
    f"{WIND}_d+1",
    f"{WIND}_d-1",
    "generation_solar_forecast_d+1",
]


@pytest.mark.parametrize("area", list(CET))
@pytest.mark.parametrize("freq", ["1h", "15min"])
def test_on_brussels_clock_every_instant_is_the_one_it_was(area, freq):
    """DE_LU, AT and PL over 2025: every availability instant the check reads, built-in, the source's rule and the accepted
    A69 assumption, equals the Brussels-day rule's to the bit, on a frame in the market's zone, in UTC (issued at noon UTC,
    Brussels or the market's) and at UTC+1."""
    zone = CET[area]
    source = EntsoeSource(area, api_key="test-key", type=["load", "generation"])
    old_assumption = {"generation_*_forecast": Availability.published("10:00", 1, vintage="latest_revision")}
    for frame_zone, issue_zone in [
        (None, zone),
        (None, "Europe/Brussels"),
        ("UTC", "UTC"),
        ("UTC", "Europe/Brussels"),
        ("UTC", zone),
        ("Etc/GMT-1", "Etc/GMT-1"),
    ]:
        index = frame_index(zone, freq, *YEAR)
        if frame_zone is not None:
            index = index.tz_convert(frame_zone)
        pairs = [
            (InformationSet("12:00", issue_zone, rules=BEFORE), InformationSet("12:00", issue_zone)),
            (
                InformationSet("12:00", issue_zone, rules=BEFORE),
                InformationSet("12:00", issue_zone).with_source_rules([source]),
            ),
            (
                InformationSet("12:00", issue_zone, rules={**BEFORE, **old_assumption}),
                InformationSet("12:00", issue_zone, assumptions=["entsoe_a69_by_10"]).with_source_rules([source]),
            ),
        ]
        for before, after in pairs:
            for name in CET_NAMES:
                old, _, _ = available(before, name, index, freq, *YEAR)
                new, _, _ = available(after, name, index, freq, *YEAR)
                assert np.array_equal(old, new), (frame_zone, issue_zone, name)


# ---------------------------------------------------------------------------
# Forecasts: NaiveModel(known_only=True) keys
# ---------------------------------------------------------------------------


class InProcessPool:
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        _worker_init(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, task, *args):
        future = Future()
        try:
            future.set_result(task(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


@pytest.fixture
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)


TEST = ("2025-01-13", "2025-01-26")


def tso_keys(zone: str, freq: str, rules=None) -> set:
    """NaiveModel of `load_forecast_d+{horizon}`, h = 1, 2, issued at noon on the market's clock; each `_d+k` column holds the
    ordinal of the local date its slot is delivered on, so a forecast says which day it read."""
    start = pd.Timestamp("2025-01-10").tz_localize(zone)
    index = pd.date_range(start, start + pd.Timedelta(days=21), freq=freq, inclusive="left")
    ordinal = np.array([day.toordinal() for day in index.tz_localize(None).date])
    frame = pd.DataFrame({"y": np.arange(len(index), dtype=float)}, index=index)
    for k in range(3):
        frame[f"{LOAD}_d+{k}"] = (ordinal + k).astype(float)
    report = (
        ModelPipeline(information_set=InformationSet("12:00", zone, rules=rules), availability="raise")
        .add_model(NaiveModel([f"{LOAD}_d+{{horizon}}"], name="TSO"))
        .run(frame, target="y", freq=freq, test=TEST, horizon=2)
    )
    keys = set()
    for record in report.refs["TSO"].iter_records():
        read = datetime.date.fromordinal(int(record["prediction"]))
        assert read.isoformat() == record["target_date"]
        keys.add((record["run_date"], record["hour"], record["horizon"]))
    return keys


def keys_by_hand(zone: str, freq: str) -> set:
    out = set()
    for day in pd.date_range(*TEST).date:
        for horizon in (1, 2):
            if published(LOAD, day + datetime.timedelta(days=horizon)) <= issued(day, "12:00", zone):
                out |= {(day.isoformat(), period, horizon) for period in range(periods_per_day(freq))}
    return out


@pytest.mark.parametrize(
    ("zone", "freq", "removed", "added"),
    [
        ("Europe/Helsinki", "1h", 14, 0),  # period 0 of D+2 on each of 14 days, forecast before its value existed
        ("Europe/Helsinki", "15min", 56, 0),  # periods 0-3
        ("Europe/Athens", "1h", 14, 0),
        ("Europe/Lisbon", "1h", 0, 14),  # period 23 of D+1, out since 10:00 Brussels, called unpublished
        ("Europe/Warsaw", "1h", 0, 0),
    ],
)
def test_a_naive_forecast_of_the_tso_exists_exactly_where_its_value_was_out(zone, freq, removed, added, single_process):
    after = tso_keys(zone, freq)
    assert after == keys_by_hand(zone, freq)
    before = tso_keys(zone, freq, rules=BEFORE)
    assert len(before - after) == removed
    assert len(after - before) == added
    assert all(horizon == 2 for _, _, horizon in before - after)


# ---------------------------------------------------------------------------
# The rules themselves: the source's, the assumption, a prefixed column, the description
# ---------------------------------------------------------------------------


def test_the_built_in_rules_read_the_data_s_dates_and_publish_in_brussels():
    for name, time in (("load_forecast", "10:00"), ("generation_*_forecast", "18:00")):
        rule = default_rules()[name]
        assert (rule.kind, rule.time, rule.days_before, rule.timezone, rule.day_zone) == (
            "published",
            time,
            1,
            "Europe/Brussels",
            "data",
        )
        assert rule.describe().endswith("(delivery days of the data's local dates)")
        assert Availability.from_dict(rule.to_dict()) == rule
    assumed = ASSUMPTIONS["entsoe_a69_by_10"][1]
    assert (assumed.time, assumed.timezone, assumed.day_zone) == ("10:00", "Europe/Brussels", "data")


def test_an_entsoe_source_s_rule_reads_its_area_s_day_not_the_bound_market_s():
    """A neighbouring area in another market's pipeline keeps its values at their own instants (only A31 is re-stamped at the
    market's midnight), and its TSO's day is its own: Helsinki's, though the pipeline is Warsaw's."""
    source = EntsoeSource("FI", api_key="test-key", type=["load", "generation"])
    DataPipeline(sources=[source], timezone="Europe/Warsaw")
    info = InformationSet().with_source_rules([source])
    for name, time in ((LOAD, "10:00"), (WIND, "18:00")):
        rule = info.resolve(name).rule
        assert (rule.time, rule.timezone, rule.day_zone) == (time, "Europe/Brussels", "Europe/Helsinki")
    # A31 stays in the market's dates, where the source stamps it
    assert info.resolve("load_forecast_daily_max").rule.week_zone == "Europe/Warsaw"
    # Issued at noon Warsaw on a Warsaw frame of Finnish values: Helsinki's 00:00 of D+1 is 23:00 of D in Warsaw, so the
    # Warsaw day dates period 23 of `_d+1` (A65) and of D's own row (A69) a day early, on each of the 21 days.
    first, last = WINDOWS["january"]
    index = frame_index("Europe/Warsaw", "1h", first, last)
    bound = InformationSet("12:00", "Europe/Warsaw").with_source_rules([source])
    built_in = InformationSet("12:00", "Europe/Warsaw")
    for column, horizon in ((LOAD, 1), (WIND, 0)):
        name = f"{column}_d+{horizon}" if horizon else column
        expected = by_hand(column, "Europe/Helsinki", "Europe/Warsaw", "1h", first, last, horizon)
        for info, early in ((bound, 0), (built_in, 21)):
            found, issue, skipped = available(info, name, index, "1h", first, last)
            known = (found <= issue[None, :]) & ~skipped
            assert int((known & ~expected).sum()) == early, (name, early)
            assert not (expected & ~known & ~skipped).any(), name
            if early:
                assert set(np.argwhere(known & ~expected)[:, 0].tolist()) == {23}


def test_an_entsoe_source_without_the_type_or_with_a_prefix_adds_no_rule():
    """`load_forecast` comes with "load", A69 with "generation"; a prefixed column of an EntsoeSource (`fi_load_forecast`) has
    no rule, as `fi_load_actual` and `fi_price` haven't: a rule of yours names it."""
    load = InformationSet().with_source_rules([EntsoeSource("FI", api_key="test-key", type=["load"])])
    assert load.resolve(LOAD).rule.day_zone == "Europe/Helsinki"
    assert load.resolve(WIND).rule.day_zone == "data"  # the built-in rule
    prefixed = InformationSet().with_source_rules(
        [EntsoeSource("FI", api_key="test-key", type=["load", "generation"], prefix="fi")]
    )
    assert prefixed.resolve(f"fi_{LOAD}").rule is None
    assert prefixed.resolve(f"fi_{WIND}").rule is None
    assert prefixed.resolve("fi_load_forecast_daily_max").rule is not None  # A31's, as before


def test_the_a69_assumption_keeps_the_source_s_day_and_moves_the_time_only():
    source = EntsoeSource("FI", api_key="test-key", type=["generation"])
    info = InformationSet("12:00", "Europe/Brussels", assumptions=["entsoe_a69_by_10"]).with_source_rules([source])
    resolved = info.resolve(WIND)
    assert resolved.layer == "assumptions"
    assert (resolved.rule.time, resolved.rule.day_zone) == ("10:00", "Europe/Helsinki")
    assert InformationSet(assumptions=["entsoe_a69_by_10"]).resolve(WIND).rule.day_zone == "data"


def test_a_user_s_published_rule_keeps_the_day_of_its_own_zone():
    """`Availability.published(..., timezone=zone)` without `day_zone` dates the delivery day in `zone`, as it always did."""
    first, last = WINDOWS["january"]
    info = InformationSet("12:00", "Europe/Helsinki", rules={"tso": Availability.published("10:00", 1)})
    known, skipped = known_cells(info, "tso_d+2", "Europe/Helsinki", "1h", first, last)
    assert known[0].all() and not known[1:].any()  # Helsinki's 00:00 dated to Brussels' day before


def test_on_a_utc_frame_issued_in_utc_a_data_day_is_never_earlier_than_brussels():
    """A frame in no market's zone issued on no market's clock: the data's dates would put a CET market's 00:00 on the day
    before (23:00Z), a day early. The rule's own zone is read too, and the later counts: the Brussels day it had before."""
    first, last = WINDOWS["january"]
    for zone in ("Europe/Berlin", "Europe/Helsinki", "Europe/Lisbon"):
        old, _ = known_cells(
            InformationSet("12:00", "UTC", rules=BEFORE), f"{LOAD}_d+1", zone, "1h", first, last, "UTC"
        )
        new, _ = known_cells(InformationSet("12:00", "UTC"), f"{LOAD}_d+1", zone, "1h", first, last, "UTC")
        assert np.array_equal(old, new), zone
    data_only, _ = known_cells(
        InformationSet("12:00", "UTC", rules={LOAD: Availability.published("10:00", 1, timezone="UTC")}),
        f"{LOAD}_d+1",
        "Europe/Berlin",
        "1h",
        first,
        last,
        "UTC",
    )
    assert data_only.sum() > new.sum()  # what the UTC day alone would have called known
