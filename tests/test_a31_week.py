"""The week-ahead load forecast (ENTSO-E A65/A31) follows each market's own Monday-to-Sunday week.

A31 holds one value per local date of the market, and the whole of next week is published on Friday at 10:00 Brussels, two hours
before the 12:00 CET day-ahead gate closure, in every zone. So the built-in rule reads the week in the data's local dates
(`Availability.weekly(..., week_zone="data")`) and the publication instant in Brussels. It used to read the week in Brussels dates:
on an hourly frame of a zone ahead of Brussels the first hour of a Monday is Sunday there, so it was taken as published a week early
(Helsinki: 12 keys of two January weeks forecast before the value existed), and in a zone behind Brussels the last hour of a Sunday
is Monday there, so it was taken as published a week late (Lisbon, Dublin: 14 keys called unpublished). A user's weekly rule keeps
its meaning: its week is its own zone's unless it says `week_zone`.

The truth here is written down from that description alone (`published` below) and compared with every cell the check reads, on
daily, hourly and 15-minute frames over January and both switch weekends, in EU zones (which switch together) and in zones that
switch on other dates or never.
"""

import datetime
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.sources.entsoe import EntsoeSource
from epftoolbox2.information_set import (
    _STEP_MINUTES,
    Availability,
    InformationSet,
    InformationSetError,
    default_rules,
)
from epftoolbox2.models import NaiveModel
from epftoolbox2.models._calendar import periods_per_day
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import DataPipeline, ModelPipeline

COLUMN = "load_forecast_daily_max"
TSO = f"{COLUMN}_d+{{horizon}}"
HORIZON = 9
# Issue days of each window; the frame reaches 12 days past the last, so every target has a row.
WINDOWS = {
    "january": ("2024-01-08", "2024-01-28"),
    "spring": ("2024-03-01", "2024-04-02"),  # New York switches on 10 March, the EU on 31 March
    "autumn": ("2024-10-18", "2024-11-05"),  # the EU on 27 October, New York on 3 November
}


def published(day: datetime.date) -> pd.Timestamp:
    """When A31's value of the local date `day` is out: Friday 10:00 Brussels before the Monday of its week."""
    monday = day - datetime.timedelta(days=day.weekday())
    return pd.Timestamp(f"{monday - datetime.timedelta(days=3)} 10:00", tz="Europe/Brussels")


def issued(day: datetime.date, time: str, zone: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {time}", tz=zone)


def frame_index(zone: str, freq: str, window: str) -> pd.DatetimeIndex:
    """Rows from 2 days before the window's first issue day to 12 after its last, at noon for a daily frame."""
    first = pd.Timestamp(WINDOWS[window][0]) - pd.Timedelta(days=2)
    last = pd.Timestamp(WINDOWS[window][1]) + pd.Timedelta(days=12)
    if freq == "1D":
        return (pd.date_range(first, last, freq="D") + pd.Timedelta(hours=12)).tz_localize(zone)
    return pd.date_range(first.tz_localize(zone), last.tz_localize(zone), freq=freq, inclusive="left")


def known_cells(info: InformationSet, column: str, zone: str, freq: str, window: str, horizon: int) -> tuple:
    """Whether the check calls `column_d+{horizon}` known, as [period, issue day], and which cells the data's clock skips."""
    index = frame_index(zone, freq, window)
    days = pd.date_range(*WINDOWS[window], freq="D")
    clock = info._issue_clock(days, index, freq)
    step = _STEP_MINUTES[freq]
    periods = np.arange(periods_per_day(freq))
    avail = clock.available(info._resolve_for(f"{column}_d+{horizon}", step), periods, step)
    return avail <= clock.issue[None, :], clock.skipped(periods, step), [day.date() for day in days]


def by_hand(days, n_periods: int, horizon: int, issue: str, issue_zone: str, out=published) -> np.ndarray:
    """The truth as [period, issue day]: the value of the local date D + horizon is out by the issue on D. A `_d+k` shift keeps
    the wall clock, so every period of the row reads the same local date."""
    row = [out(day + datetime.timedelta(days=horizon)) <= issued(day, issue, issue_zone) for day in days]
    return np.tile(np.array(row), (n_periods, 1))


# EU zones east of, at and west of Brussels, New York (other switch dates) and Kolkata (no DST, 4.5 hours ahead, issued at 12:00
# there, before Friday's 10:00 Brussels).
ZONES = ["Europe/Helsinki", "Europe/Warsaw", "Europe/Lisbon", "Europe/Dublin", "America/New_York", "Asia/Kolkata"]
FREQS = ["1D", "1h", "15min"]
# Every pair of levels of zone, frequency and window, in 18 of the 54 cells: the window is (zone + frequency) mod 3.
GRID = [(zone, freq, list(WINDOWS)[(z + f) % 3]) for z, zone in enumerate(ZONES) for f, freq in enumerate(FREQS)]


@pytest.mark.parametrize(("zone", "freq", "window"), GRID)
def test_the_built_in_rule_publishes_each_market_s_own_week_on_friday_at_10_brussels(zone, freq, window):
    info = InformationSet("12:00", zone)
    for horizon in range(1, HORIZON + 1):
        known, skipped, days = known_cells(info, COLUMN, zone, freq, window, horizon)
        expected = by_hand(days, known.shape[0], horizon, "12:00", zone)
        wrong = np.argwhere((known != expected) & ~skipped)
        assert not len(wrong), [(days[d], int(p), horizon, bool(known[p, d])) for p, d in wrong[:5]]


@pytest.mark.parametrize("window", list(WINDOWS))
@pytest.mark.parametrize("freq", ["1D", "1h", "15min"])
def test_in_warsaw_the_rule_is_bitwise_the_brussels_week_it_was(freq, window):
    """Warsaw's dates are Brussels' at every instant, so reading the week in the data's dates changes no instant there."""
    before = InformationSet("12:00", "Europe/Warsaw", rules={COLUMN: Availability.weekly("friday", "10:00")})
    after = InformationSet("12:00", "Europe/Warsaw")
    index = frame_index("Europe/Warsaw", freq, window)
    days = pd.date_range(*WINDOWS[window], freq="D")
    step = _STEP_MINUTES[freq]
    periods = np.arange(periods_per_day(freq))
    for horizon in (0, 1, 3, 7, 9):
        name = f"{COLUMN}_d+{horizon}" if horizon else COLUMN
        found = [
            info._issue_clock(days, index, freq).available(info._resolve_for(name, step), periods, step)
            for info in (before, after)
        ]
        assert np.array_equal(found[0], found[1]), name


# ---------------------------------------------------------------------------
# Forecasts: NaiveModel(known_only=True) and the check at "raise"
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


def tso_frame(zone: str, freq: str, first: str = "2024-01-13", days: int = 30) -> pd.DataFrame:
    """Each `_d+k` column holds the ordinal of the local date it is the value of, so a forecast says which date it read."""
    if freq == "1D":
        index = (pd.date_range(first, periods=days, freq="D") + pd.Timedelta(hours=12)).tz_localize(zone)
    else:
        start = pd.Timestamp(first).tz_localize(zone)
        index = pd.date_range(start, start + pd.Timedelta(days=days), freq=freq, inclusive="left")
    ordinal = np.array([day.toordinal() for day in index.tz_localize(None).date])
    frame = pd.DataFrame({"y": np.arange(len(index), dtype=float)}, index=index)
    for k in range(HORIZON + 1):
        frame[f"{COLUMN}_d+{k}"] = (ordinal + k).astype(float)
    return frame


def tso_keys(zone: str, freq: str, issue_days=None, horizons=None) -> set:
    frame = tso_frame(zone, freq)
    report = (
        ModelPipeline(information_set=InformationSet("12:00", zone), availability="raise")
        .add_model(NaiveModel([TSO], name="TSO"))
        .run(
            frame,
            target="y",
            freq=freq,
            test=("2024-01-15", "2024-01-28"),
            issue_days=issue_days,
            **({"horizons": horizons} if horizons else {"horizon": HORIZON}),
        )
    )
    records = list(report.refs["TSO"].iter_records())
    for record in records:  # the value of the target's own local date
        assert record["prediction"] == datetime.date.fromisoformat(record["target_date"]).toordinal()
    return {(record["run_date"], record["hour"], record["horizon"]) for record in records}


def keys_by_hand(zone: str, freq: str, weekdays=None, horizons=range(1, HORIZON + 1)) -> set:
    out = set()
    for day in pd.date_range("2024-01-15", "2024-01-28").date:
        if weekdays is not None and day.weekday() not in weekdays:
            continue
        for horizon in horizons:
            if published(day + datetime.timedelta(days=horizon)) <= issued(day, "12:00", zone):
                out |= {(day.isoformat(), period, horizon) for period in range(periods_per_day(freq))}
    return out


@pytest.mark.parametrize(
    ("zone", "freq"),
    [
        ("Europe/Helsinki", "1h"),  # the look-ahead: 12 keys of period 0 forecast before the value existed
        ("Europe/Helsinki", "1D"),
        ("Europe/Lisbon", "1h"),  # too strict: 14 keys of the last period called unpublished
        ("Europe/Lisbon", "1D"),
        ("Europe/Dublin", "1D"),
        ("Europe/Warsaw", "1h"),
    ],
)
def test_a_naive_forecast_of_the_tso_exists_exactly_where_its_value_was_out(zone, freq, single_process):
    assert tso_keys(zone, freq) == keys_by_hand(zone, freq)


@pytest.mark.parametrize("zone", ["Europe/Lisbon", "Europe/Dublin", "Europe/Helsinki"])
def test_on_fridays_at_noon_the_whole_next_week_is_out_in_every_zone(zone, single_process):
    """The study's form: issued on Fridays for D+3 (Monday) to D+9 (Sunday). Lisbon's and Dublin's D+9 used to be unpublished."""
    keys = tso_keys(zone, "1D", issue_days=["friday"], horizons=range(3, 10))
    assert keys == {(day, 0, h) for day in ("2024-01-19", "2024-01-26") for h in range(3, 10)}


def test_the_helsinki_hour_dated_to_the_week_before_is_refused_at_raise():
    """Issued Monday 2024-01-15 at 12:00 Helsinki, h=7, period 0 is Monday 22 January 00:00 Helsinki, Sunday 23:00 in Brussels:
    the week published Friday 19 January, after the issue. The rule used to date it to the week published on the 12th."""
    index = pd.date_range("2024-01-01", periods=24 * 40, freq="h", tz="Europe/Helsinki")
    info = InformationSet("12:00", "Europe/Helsinki")
    with pytest.raises(InformationSetError, match=r"load_forecast_daily_max_d\+7"):
        info.check({(0, 7): [f"{COLUMN}_d+7"]}, "load_actual", "1h", index, "2024-01-15", "2024-01-15")
    info.check({(0, 6): [f"{COLUMN}_d+6"]}, "load_actual", "1h", index, "2024-01-15", "2024-01-15")  # Sunday 21st: out


@pytest.mark.parametrize("zone", ["Europe/Helsinki", "Europe/Lisbon"])
def test_newest_targets_of_a_weekly_target_read_the_market_s_week(zone):
    """`newest_target="published"` walks up to the newest target out at the issue: on Thursday 18 January the values up to
    Sunday 21st (k = 3), and Monday 22nd only after Friday 19th, in every period. The Brussels week gave Helsinki's period 0
    k = 4 (a look-ahead) and Lisbon's period 23 k = 2."""
    index = pd.date_range("2023-12-01", periods=24 * 70, freq="h", tz=zone)
    days = list(pd.date_range("2024-01-15", "2024-01-28").date)
    found = InformationSet("12:00", zone).newest_targets(COLUMN, "1h", index, days, 7)
    # A weekly value is never out before an older one, so the newest is the last out, capped at horizon - 1.
    newest = [
        max(k for k in range(-1, 7) if published(day + datetime.timedelta(days=k)) <= issued(day, "12:00", zone))
        for day in days
    ]
    assert np.array_equal(found, np.tile(newest, (24, 1)))
    assert found[:, days.index(datetime.date(2024, 1, 18))].tolist() == [3] * 24


# ---------------------------------------------------------------------------
# The publication instant, a user's rule, the source's rule, chains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("zone", "issue", "out"), [("Europe/Helsinki", "09:30", False), ("Europe/Lisbon", "10:30", True)]
)
def test_friday_s_publication_is_10_brussels_whatever_the_data_zone(zone, issue, out):
    """At 09:30 Brussels next week isn't out, in Helsinki too, where it is 10:30; at 10:30 Brussels it is, in Lisbon too, where
    it is 09:30. Publishing at 10:00 on the data's clock would be a look-ahead in Helsinki and too strict in Lisbon."""
    info = InformationSet(issue, "Europe/Brussels")
    for freq in ("1D", "1h"):
        known, skipped, days = known_cells(info, COLUMN, zone, freq, "january", 3)
        on_friday = [day.weekday() == 4 for day in days]
        assert (known[:, on_friday] == out).all(), freq
        expected = by_hand(days, known.shape[0], 3, issue, "Europe/Brussels")
        assert np.array_equal(known | skipped, expected | skipped), freq


@pytest.mark.parametrize("freq", ["1D", "1h"])
def test_a_user_s_weekly_rule_keeps_the_week_of_its_own_zone(freq):
    """`Availability.weekly(..., timezone=zone)` publishes and dates the week in `zone`, as it always did; `week_zone` names
    another zone for the week, the publication staying in `timezone`."""
    helsinki_week = InformationSet(
        "12:00", "Europe/Warsaw", rules={"tso": Availability.weekly("friday", "10:00", "Europe/Helsinki")}
    )
    named = InformationSet(
        "12:00", "Europe/Warsaw", rules={"tso": Availability.weekly("friday", "10:00", week_zone="Europe/Helsinki")}
    )
    brussels_week = InformationSet("12:00", "Europe/Helsinki", rules={"tso": Availability.weekly("friday", "10:00")})

    def out_in_helsinki(day):  # Friday 10:00 Helsinki before the Helsinki week of `day`
        return published(day).tz_localize(None).tz_localize("Europe/Helsinki")

    for window in ("january", "spring", "autumn"):
        for horizon in (3, 7):
            cases = [
                (helsinki_week, "Europe/Warsaw", out_in_helsinki, lambda t, z: t.tz_convert("Europe/Helsinki")),
                (named, "Europe/Warsaw", published, lambda t, z: t.tz_convert("Europe/Helsinki")),
                (brussels_week, "Europe/Helsinki", published, lambda t, z: t.tz_convert("Europe/Brussels")),
            ]
            for info, zone, out, week_clock in cases:
                index = frame_index(zone, freq, window)
                days = pd.date_range(*WINDOWS[window], freq="D")
                clock = info._issue_clock(days, index, freq)
                periods = np.arange(periods_per_day(freq))
                step = _STEP_MINUTES[freq]
                known = clock.available(info._resolve_for(f"tso_d+{horizon}", step), periods, step) <= clock.issue
                skipped = clock.skipped(periods, step)
                expected = np.zeros_like(known)
                for d, day in enumerate(days.date):
                    for p in periods:
                        if skipped[p, d]:
                            continue
                        # The row's delivery instant, its date on the week's clock (the last slot of a daily row's date).
                        wall = pd.Timestamp(day + datetime.timedelta(days=horizon))
                        wall += pd.Timedelta(hours=23) if freq == "1D" else pd.Timedelta(minutes=int(p) * step)
                        instant = wall.tz_localize(zone, ambiguous=False, nonexistent="shift_forward")
                        delivery = week_clock(instant, zone).date()
                        expected[p, d] = out(delivery) <= issued(day.isoformat(), "12:00", zone)
                wrong = np.argwhere((known != expected) & ~skipped)
                assert not len(wrong), (zone, window, horizon, [(str(days[d].date()), int(p)) for p, d in wrong[:4]])


def test_the_source_s_rule_reads_the_bound_market_s_week_on_any_index():
    """An `EntsoeSource` stamps A31 at its bound market's midnight, so its rule reads the week in that zone, even on a UTC
    index, where the data's dates would put Monday's first hour of Warsaw on Sunday. It is published at 10:00 Brussels."""
    source = EntsoeSource("PL", api_key="test-key", type=["load"])
    DataPipeline(sources=[source], timezone="Europe/Warsaw")
    rule = InformationSet().with_source_rules([source]).resolve(COLUMN).rule
    assert (rule.week_zone, rule.timezone, rule.time) == ("Europe/Warsaw", "Europe/Brussels", "10:00")

    info = InformationSet("12:00", "Europe/Warsaw").with_source_rules([source])
    utc = pd.date_range("2024-01-01", periods=24 * 40, freq="h", tz="UTC")
    days = pd.date_range("2024-01-15", "2024-01-18", freq="D")
    clock = info._issue_clock(days, utc, "1h")
    known = clock.available(info._resolve_for(f"{COLUMN}_d+4", 60), np.arange(24), 60) <= clock.issue
    # Issued Wednesday 17 January, + 4 days at 23:00Z is Monday 22 January 00:00 in Warsaw: published Friday 19th, after the
    # issue; at 22:00Z it is still Sunday 21st there, published on the 12th.
    assert not known[23, 2] and known[22, 2]


@pytest.mark.parametrize(("market", "zone"), [("FI", "Europe/Helsinki"), ("PT", "Europe/Lisbon")])
def test_the_source_s_rule_publishes_at_10_brussels(market, zone):
    source = EntsoeSource(market, api_key="test-key", type=["load"])
    DataPipeline(sources=[source], timezone=zone)
    for issue, out in (("09:30", False), ("10:30", True)):
        info = InformationSet(issue, "Europe/Brussels").with_source_rules([source])
        known, _, days = known_cells(info, COLUMN, zone, "1h", "january", 3)
        assert (known[:, [day.weekday() == 4 for day in days]] == out).all(), issue


def test_a_chain_of_weekly_rules_reads_its_periods_on_the_week_s_dates():
    """Each rule of a chain holds for the values whose delivery date is in its period: for a weekly rule a date of its week's
    zone. Helsinki's Monday 22 January 00:00 is Sunday in Brussels, but a date of the second rule here, published Thursday."""
    chain = [
        Availability.weekly("friday", "10:00", week_zone="data").during(valid_until="2024-01-21"),
        Availability.weekly("thursday", "10:00", week_zone="data").during(valid_from="2024-01-22"),
    ]
    info = InformationSet("12:00", "Europe/Helsinki", rules={COLUMN: chain})

    def out(day):
        if day <= datetime.date(2024, 1, 21):
            return published(day)
        return published(day) - pd.Timedelta(days=1)  # Thursday

    for freq in ("1h", "15min", "1D"):
        for horizon in (3, 4, 7):
            known, skipped, days = known_cells(info, COLUMN, "Europe/Helsinki", freq, "january", horizon)
            expected = by_hand(days, known.shape[0], horizon, "12:00", "Europe/Helsinki", out=out)
            wrong = np.argwhere((known != expected) & ~skipped)
            assert not len(wrong), (freq, horizon, [(days[d], int(p)) for p, d in wrong[:4]])


# ---------------------------------------------------------------------------
# The parameter
# ---------------------------------------------------------------------------


def test_week_zone_is_a_weekly_rule_s_and_round_trips():
    rule = Availability.weekly("friday", "10:00", week_zone="data")
    assert rule.to_dict() == {
        "kind": "weekly",
        "weekday": "friday",
        "time": "10:00",
        "timezone": "Europe/Brussels",
        "week_zone": "data",
    }
    assert Availability.from_dict(rule.to_dict()) == rule
    assert "week_zone" not in Availability.weekly().to_dict()  # a file without it reads as it always did
    assert rule.describe() == "published Friday 10:00 Europe/Brussels for the following week of the data's local dates"
    named = Availability.weekly(week_zone="Europe/Lisbon")
    assert named.describe() == "published Friday 10:00 Europe/Brussels for the following week of Europe/Lisbon dates"
    assert default_rules()[COLUMN].week_zone == "data"
    with pytest.raises(ValueError, match="week_zone"):
        Availability("published", time="10:00", week_zone="data")
    with pytest.raises(ValueError, match="time zone"):
        Availability.weekly(week_zone="Mars/Olympus")


def test_a_weekly_rule_with_week_zone_round_trips_through_yaml(tmp_path):
    rule = Availability.weekly("friday", "09:00", "Europe/Lisbon", week_zone="data")
    info = InformationSet("12:00", "Europe/Lisbon", rules={"tso": rule})
    ModelPipeline(information_set=info).add_model(NaiveModel(["tso_d+{horizon}"])).save(tmp_path / "mp.yaml")
    loaded = ModelPipeline.load(tmp_path / "mp.yaml").information_set
    assert loaded.to_dict() == info.to_dict() and loaded.resolve("tso").rule == rule
