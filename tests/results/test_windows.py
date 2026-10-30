"""The windows of a run as dates: what `validation=` and `test=` accept, and how `previous_year` cuts a test period."""

import datetime

import pytest

from epftoolbox2.results import RunWindows, Segment
from epftoolbox2.results.windows import add_years, mask_key, to_period, yearly_segments

V = ("2023-01-02", "2023-02-28")
T = ("2023-03-06", "2023-03-31")


def test_two_periods_are_one_fixed_segment():
    windows = RunWindows.resolve(V, T)
    assert windows.periods == [("validation", V), ("test", T)]
    assert windows.mode == "fixed"
    assert windows.segments == (Segment(V, T),)
    assert (windows.first_day, windows.last_day, windows.target_before) == (V[0], T[1], T[0])


@pytest.mark.parametrize(("validation", "test"), [(V, None), (None, T)])
def test_one_window_has_no_segment(validation, test):
    windows = RunWindows.resolve(validation, test)
    assert windows.segments == ()
    assert [name for name, _ in windows.periods] == ["validation" if validation else "test"]


@pytest.mark.parametrize(
    ("validation", "test", "message"),
    [
        (None, None, "No window"),
        (("2023-03-06", "2023-03-07"), T, "must end before"),  # ends on the test start
        (("2023-03-01", "2023-03-20"), T, "must end before"),  # runs into it
        (("2023-02-28", "2023-01-02"), T, "reversed"),
        (V, ("2023-03-31", "2023-03-06"), "reversed"),
        ("previous_month", T, "previous_year"),
        ("previous_year", None, "test=\\(start, end\\)"),
        ("2023-01-02", T, "previous_year"),
        (("2023-01-02",), T, "\\(start, end\\)"),
        (("2023-01-02", "2023-02-28", "2023-03-01"), T, "\\(start, end\\)"),
        (("January", "2023-02-28"), T, "must be a date"),
        (V, "2023-03-06", "\\(start, end\\)"),
    ],
)
def test_what_makes_no_sense_is_refused(validation, test, message):
    with pytest.raises(ValueError, match=message):
        RunWindows.resolve(validation, test)


def test_a_day_of_validation_right_before_the_test_is_fine():
    assert RunWindows.resolve(("2023-01-02", "2023-03-05"), T).validation[1] == "2023-03-05"


def test_dates_are_normalised_whatever_they_come_as():
    import pandas as pd

    assert to_period((datetime.date(2023, 1, 2), pd.Timestamp("2023-02-28")), "test") == V
    assert to_period(["2023-01-02", "2023-02-28"], "test") == V


def test_the_days_of_both_windows_once_and_nothing_between_them():
    days = RunWindows.resolve(("2023-01-30", "2023-02-01"), ("2023-02-03", "2023-02-04")).issue_days()
    assert days == ["2023-01-30", "2023-01-31", "2023-02-01", "2023-02-03", "2023-02-04"]


def test_a_day_belongs_to_its_window_or_to_none():
    windows = RunWindows.resolve(("2023-01-30", "2023-02-01"), ("2023-02-03", "2023-02-04"))
    assert windows.window_of("2023-01-30") == "validation"
    assert windows.window_of("2023-02-01") == "validation"
    assert windows.window_of("2023-02-02") is None
    assert windows.window_of("2023-02-03") == "test"
    assert windows.window_of("2023-02-05") is None


def test_the_windows_survive_a_manifest_as_they_are():
    for windows in (RunWindows.resolve(V, T), RunWindows.resolve(V), RunWindows.resolve("previous_year", T)):
        assert RunWindows.from_dict(windows.to_dict()) == windows


def test_a_test_mask_has_the_key_a_plain_run_gives_it():
    assert mask_key("test", 3, 1) == "3-1"
    assert mask_key("validation", 3, 1) == "validation/3-1"


# ---------------------------------------------------------------------------
# previous_year
# ---------------------------------------------------------------------------


def test_each_segment_has_the_twelve_months_before_it():
    windows = RunWindows.resolve("previous_year", ("2024-01-05", "2025-12-26"))
    assert windows.validation == ("2023-01-05", "2024-01-04")
    assert windows.segments == (
        Segment(("2023-01-05", "2024-01-04"), ("2024-01-05", "2025-01-04")),
        Segment(("2024-01-05", "2025-01-04"), ("2025-01-05", "2025-12-26")),
    )


def test_a_test_period_shorter_than_a_year_is_one_partial_segment_with_a_whole_year_before_it():
    windows = RunWindows.resolve("previous_year", ("2024-06-01", "2024-06-30"))
    assert windows.segments == (Segment(("2023-06-01", "2024-05-31"), ("2024-06-01", "2024-06-30")),)
    assert windows.first_day == "2023-06-01"


def test_a_partial_last_segment():
    segments = yearly_segments(("2022-03-01", "2024-03-02"))
    assert [segment.test for segment in segments] == [
        ("2022-03-01", "2023-02-28"),
        ("2023-03-01", "2024-02-29"),  # 366 days, with the leap day
        ("2024-03-01", "2024-03-02"),
    ]


def test_a_period_of_exactly_twelve_months_has_no_empty_segment_after_it():
    assert [segment.test for segment in yearly_segments(("2023-03-01", "2024-02-29"))] == [("2023-03-01", "2024-02-29")]


def test_the_segments_cover_the_test_period_without_gaps_or_overlaps():
    segments = yearly_segments(("2020-02-29", "2031-07-01"))
    assert segments[0].test[0] == "2020-02-29" and segments[-1].test[1] == "2031-07-01"
    for earlier, later in zip(segments, segments[1:]):
        day_after = datetime.date.fromisoformat(earlier.test[1]) + datetime.timedelta(days=1)
        assert later.test[0] == day_after.isoformat()
        assert later.validation == earlier.test  # the validation of a later segment is the test before it


def test_a_start_on_29_february_does_not_drift():
    """Each boundary counts from the original start: stepping one year at a time would stay on the 28th for good."""
    segments = yearly_segments(("2024-02-29", "2029-01-01"))
    assert [segment.test[0] for segment in segments] == [
        "2024-02-29",
        "2025-02-28",
        "2026-02-28",
        "2027-02-28",
        "2028-02-29",
    ]
    assert segments[0].validation == ("2023-02-28", "2024-02-28")
    assert segments[4].validation == ("2027-02-28", "2028-02-28")


def test_add_years_on_the_leap_day():
    leap = datetime.date(2024, 2, 29)
    assert add_years(leap, 1) == datetime.date(2025, 2, 28)
    assert add_years(leap, 4) == datetime.date(2028, 2, 29)
    assert add_years(leap, -1) == datetime.date(2023, 2, 28)
    assert add_years(leap, -4) == datetime.date(2020, 2, 29)


def test_a_forecast_serves_a_segment_only_if_its_target_day_is_before_the_test():
    segment = Segment(("2023-01-05", "2024-01-04"), ("2024-01-05", "2025-01-04"))
    assert segment.serves("2023-01-05", "2023-01-06")
    assert segment.serves("2024-01-03", "2024-01-04")  # the last target day that counts
    assert not segment.serves("2024-01-03", "2024-01-05")  # for the first test issue day
    assert not segment.serves("2024-01-04", "2024-01-05")
    assert not segment.serves("2023-01-04", "2023-01-05")  # before the validation period
    assert not segment.serves("2024-01-05", "2024-01-05")


# ---------------------------------------------------------------------------
# What a manifest keeps after another run into the same results
# ---------------------------------------------------------------------------


def test_a_window_a_run_does_not_name_stays_recorded():
    both = RunWindows.resolve(V, T)
    assert RunWindows.resolve(V).keeping(both) == both
    assert RunWindows.resolve(None, T).keeping(both) == both
    assert both.keeping(RunWindows.resolve(V)) == both


def test_a_window_a_run_names_replaces_the_older_one():
    both = RunWindows.resolve(V, T)
    other = ("2023-01-02", "2023-03-01")
    assert RunWindows.resolve(other).keeping(both) == RunWindows.resolve(other, T)


def test_what_would_overlap_is_not_kept():
    both = RunWindows.resolve(V, T)
    runs_into_test = ("2023-01-02", "2023-03-10")
    assert RunWindows.resolve(runs_into_test).keeping(both) == RunWindows.resolve(runs_into_test)


def test_what_a_run_would_drop_is_named():
    both = RunWindows.resolve(V, T)
    assert RunWindows.resolve(("2023-04-01", "2023-04-02")).would_drop(both) == ("test", T)
    assert RunWindows.resolve(None, ("2022-11-01", "2022-11-02")).would_drop(both) == ("validation", V)
    assert RunWindows.resolve(V).would_drop(both) is None
    assert RunWindows.resolve(None, T).would_drop(both) is None
    assert RunWindows.resolve(V, T).would_drop(RunWindows.resolve(None, ("2023-03-07", "2023-03-08"))) is None
    assert RunWindows.resolve(None, T).would_drop(RunWindows.resolve(V)) is None


def test_previous_year_stays_with_its_segments_when_it_can():
    test = ("2024-01-05", "2025-12-26")
    yearly = RunWindows.resolve("previous_year", test)
    assert RunWindows.resolve(None, test).keeping(yearly) == yearly
    assert RunWindows.resolve(yearly.validation).keeping(yearly) == yearly
    assert RunWindows.resolve(yearly.validation, test).keeping(yearly) == RunWindows.resolve(yearly.validation, test)
    narrower = ("2024-01-05", "2024-06-30")
    assert RunWindows.resolve(None, narrower).keeping(yearly) == RunWindows.resolve("previous_year", narrower)
    moved = ("2024-03-01", "2024-06-30")
    assert RunWindows.resolve(None, moved).keeping(yearly) == RunWindows.fixed(yearly.validation, moved)
