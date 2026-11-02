"""`epftoolbox2.models._calendar` on its own: the span of a forecast's rows, the periods a date has on the clock, what is planned.

What a run does with them is in `test_date_windows.py`.
"""

import datetime
from zoneinfo import ZoneInfo

import dateutil.tz
import numpy as np
import pandas as pd
import pytest

try:  # pandas 3 doesn't need it
    import pytz
except ImportError:  # pragma: no cover
    pytz = None

from epftoolbox2._date_utils import zone_name
from epftoolbox2.models._calendar import (
    IssueCalendar,
    clock_periods,
    clock_periods_on,
    periods_per_day,
    skipped_periods,
    training_span,
)

FREQS = {"1h": 1, "15min": 4}


def test_the_span_of_a_forecast():
    assert tuple(training_span(100, 10, 2)) == (88, 97)  # W dates, the newest target on the day before the issue day
    assert tuple(training_span(100, 1, 1)) == (98, 98)
    assert tuple(training_span(100, 10, 2, newest=-5)) == (84, 93)
    first, last = training_span(np.array([100, 110]), 10, 2)
    assert first.tolist() == [88, 98] and last.tolist() == [97, 107]


@pytest.mark.parametrize("newest", [0, 1, 5])
def test_no_row_is_dated_on_the_anchor_or_later_with_any_newest_target(newest):
    assert tuple(training_span(100, 10, 1, newest=newest)) == (90, 99)


@pytest.mark.parametrize(
    ("day", "freq", "zone", "gone"),
    [
        ("2023-03-26", "1h", "Europe/Warsaw", [2]),
        ("2023-03-26", "1h", "Europe/London", [1]),
        ("2023-03-26", "15min", "Europe/Warsaw", [8, 9, 10, 11]),
        ("2023-03-26", "15min", "Europe/London", [4, 5, 6, 7]),
        ("2023-10-29", "1h", "Europe/Warsaw", []),  # the repeated hour is one period, with two rows
        ("2023-10-29", "15min", "Europe/Warsaw", []),
        ("2023-03-12", "1h", "America/Havana", [0]),  # a switch at midnight
        ("2023-10-01", "15min", "Australia/Lord_Howe", [8, 9]),  # half an hour
        ("2023-10-01", "1h", "Australia/Lord_Howe", []),  # its 02:30 to 02:59 exist, so the hour does
        ("2023-04-02", "1h", "Australia/Lord_Howe", []),  # and half an hour back
        ("2024-09-08", "1h", "America/Santiago", [0]),
        ("2023-03-25", "1h", "America/Nuuk", [22]),  # a switch at 22:00
        ("2011-12-30", "1h", "Pacific/Apia", list(range(24))),  # a day the clock skipped
        ("2023-03-26", "1h", "UTC", []),
        ("2023-03-26", "1h", datetime.timezone(datetime.timedelta(hours=1)), []),
        ("2023-03-26", "1h", None, []),
    ],
)
def test_the_clock_has_the_periods_it_has(day, freq, zone, gone):
    assert clock_periods(day, freq, zone) == tuple(p for p in range(periods_per_day(freq)) if p not in gone)


def test_the_clock_reads_a_frame_s_own_zone_and_its_name_alike():
    index = pd.date_range("2023-03-25", periods=2, freq="D", tz="Europe/Warsaw")
    assert clock_periods("2023-03-26", "1h", index.tz) == clock_periods("2023-03-26", "1h", "Europe/Warsaw")


def test_an_unknown_freq_is_refused():
    with pytest.raises(ValueError, match="Unsupported freq"):
        IssueCalendar("30min", None, [1])


def hand_tasks(zone, freq, first_day, days, horizons):
    """The planned keys, one by one through the clock of each key's two days, in the order `run` always made them."""
    tasks = []
    for day in days:
        issue = first_day + datetime.timedelta(days=day)
        for horizon in horizons:
            target = issue + datetime.timedelta(days=horizon)
            on_both = set(clock_periods(issue, freq, zone)) & set(clock_periods(target, freq, zone))
            tasks.extend((period, horizon, day) for period in range(periods_per_day(freq)) if period in on_both)
    return tasks


@pytest.mark.parametrize("freq", list(FREQS))
@pytest.mark.parametrize("zone", ["Europe/Warsaw", "Europe/London", "UTC", None])
@pytest.mark.parametrize("horizons", [(1,), (1, 2, 3), tuple(range(1, 8))], ids=["H1", "H3", "H7"])
def test_tasks_and_count_agree_with_the_clock_key_by_key(freq, zone, horizons):
    calendar = IssueCalendar(freq, zone, horizons)
    first_day = datetime.date(2023, 3, 16)
    days = list(range(20))
    tasks = calendar.tasks(first_day, days)
    assert tasks == hand_tasks(zone, freq, first_day, days, horizons)
    assert calendar.count("2023-03-16", "2023-04-04") == len(tasks)
    gone = set(calendar.unplanned([first_day + datetime.timedelta(days=d) for d in days]))
    planned = {(period, horizon, day) for period, horizon, day in tasks}
    every = {(p, h, d) for d in days for h in horizons for p in range(periods_per_day(freq))}
    assert gone == every - planned


def test_a_hand_count_of_planned_keys_across_a_spring_switch():
    """Issue days S-2 to S+1, horizons 1 and 2. Not planned: S at both horizons, S-1 at h=1, S-2 at h=2."""
    calendar = IssueCalendar("1h", "Europe/Warsaw", [1, 2])
    assert calendar.count("2023-03-24", "2023-03-27") == 4 * 2 * 24 - 4
    assert IssueCalendar("15min", "Europe/Warsaw", [1, 2]).count("2023-03-24", "2023-03-27") == 4 * 2 * 96 - 4 * 4
    assert IssueCalendar("1h", "Europe/Warsaw", [1, 2]).count("2023-10-27", "2023-10-30") == 4 * 2 * 24


def test_a_switch_past_the_last_issue_day_counts_for_the_forecasts_aimed_at_it():
    calendar = IssueCalendar("1h", "Europe/Warsaw", [1, 2])
    assert calendar.count("2023-03-24", "2023-03-25") == 2 * 2 * 24 - 2
    first_day = datetime.date(2023, 3, 24)
    assert {(2, 2, 0), (2, 1, 1)}.isdisjoint(calendar.tasks(first_day, [0, 1]))
    assert (2, 1, 0) in calendar.tasks(first_day, [0, 1])
    assert sorted(calendar.unplanned([first_day, first_day + datetime.timedelta(days=1)])) == [(2, 1, 1), (2, 2, 0)]


def test_tasks_of_days_with_gaps_between_them_as_a_run_with_windows_has():
    calendar = IssueCalendar("1h", "Europe/Warsaw", [1])
    first_day = datetime.date(2023, 3, 20)
    tasks = calendar.tasks(first_day, [0, 6])  # 03-20 and 03-26, the switch itself
    assert tasks == [(p, 1, 0) for p in range(24)] + [(p, 1, 6) for p in range(24) if p != 2]
    assert calendar.tasks(first_day, []) == [] and calendar.unplanned([]) == []
    assert calendar.count("2023-03-20", "2023-03-19") == 0


def test_the_periods_on_several_dates_at_once():
    found = clock_periods_on(
        pd.to_datetime(["2023-03-27", "2023-03-26", "2023-10-29", "2023-03-26"]), "1h", "Europe/Warsaw"
    )
    every = tuple(range(24))
    assert found == {
        pd.Timestamp("2023-03-26"): tuple(p for p in every if p != 2),
        pd.Timestamp("2023-03-27"): every,
        pd.Timestamp("2023-10-29"): every,
    }
    assert (
        len(clock_periods_on(pd.to_datetime(["2023-03-26"]), "15min", "Europe/Warsaw")[pd.Timestamp("2023-03-26")])
        == 92
    )
    assert clock_periods_on(pd.to_datetime(["2023-03-26"]), "1h", None)[pd.Timestamp("2023-03-26")] == every
    assert clock_periods_on([], "1h", "Europe/Warsaw") == {}


@pytest.mark.parametrize(
    "zone",
    ["Europe/Warsaw", dateutil.tz.gettz("Europe/Warsaw"), pytz.timezone("Europe/Warsaw"), ZoneInfo("Europe/Warsaw")],
)
def test_the_periods_a_record_s_dates_skip(zone):
    assert skipped_periods("2023-03-26", "1h", zone) == {2}
    assert skipped_periods(datetime.date(2023, 3, 27), "15min", zone) == frozenset()


@pytest.mark.parametrize(
    ("zone", "name"),
    [
        (ZoneInfo("Europe/Warsaw"), "Europe/Warsaw"),
        (dateutil.tz.gettz("Europe/Warsaw"), "Europe/Warsaw"),  # its str is tzfile('/usr/share/zoneinfo/Europe/Warsaw')
        (datetime.timezone.utc, "UTC"),
        (dateutil.tz.UTC, "UTC"),
        (datetime.timezone(datetime.timedelta(hours=2)), "UTC+02:00"),
        (dateutil.tz.tzoffset(None, 3600), None),
        (None, None),
    ]
    + (
        [(pytz.timezone("Europe/Warsaw"), "Europe/Warsaw"), (pytz.UTC, "UTC"), (pytz.FixedOffset(60), None)]
        if pytz
        else []
    ),
)
def test_the_name_a_manifest_writes_for_a_zone(zone, name):
    assert zone_name(zone) == name
    if name is not None:
        assert (
            pd.Timestamp("2023-03-26 12:00").tz_localize(name).utcoffset()
            == pd.Timestamp("2023-03-26 12:00").tz_localize(zone).utcoffset()
        )
