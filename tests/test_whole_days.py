"""`evaluators._days`: the rule that says which days of a frame of forecasts hold all their periods.

`EconomicEvaluator` (a schedule needs the whole day) and `EvaluationReport.aggregate()` (a daily maximum needs it) both
use it, so these tests pin the rule itself; `tests/test_economic_evaluator.py` pins it through the battery.
"""

import numpy as np
import pytest

from epftoolbox2.evaluators._days import periods_per_hour, whole_day_ids, whole_days


def frame(days, per_hour=1):
    """`day`, `hour` and `sunday` arrays for days given as lists of period ids (the days are Mondays unless a tuple
    `(ids, True)` makes one a Sunday)."""
    day, hour, sunday = [], [], []
    for i, spec in enumerate(days):
        ids, on_sunday = spec if isinstance(spec, tuple) else (spec, False)
        day += [i] * len(ids)
        hour += list(ids)
        sunday.append(on_sunday)
    return np.array(day), np.array(hour), np.array(sunday)


def whole(days, per_hour=1):
    return {periods: found.tolist() for periods, found in whole_days(*frame(days), per_hour)}


CLOCK = list(range(24))
WITHOUT_2 = [h for h in CLOCK if h != 2]


def test_days_of_24_23_and_25_hours_are_whole():
    assert whole([CLOCK, WITHOUT_2, list(range(25))]) == {24: [0], 23: [1], 25: [2]}


@pytest.mark.parametrize("skipped", [1, 2, 3])
def test_a_23_hour_day_may_lack_any_hour_a_clock_change_can_skip(skipped):
    assert whole([[h for h in CLOCK if h != skipped]]) == {23: [0]}


@pytest.mark.parametrize("missing", [0, 4, 12, 23])
def test_a_day_that_lacks_any_other_hour_is_not_whole(missing):
    assert whole([[h for h in CLOCK if h != missing]]) == {}


def test_a_day_numbered_through_as_23_hours_is_whole_only_on_a_sunday():
    through = list(range(23))
    assert whole([through]) == {}
    assert whole([(through, True)]) == {23: [0]}


def test_ids_say_what_a_day_is_not_how_many_there_are():
    """24 ids that are not the hours 0-23, and 25 ids that are not 0-24, are not days."""
    assert whole([list(range(1, 25)), [h for h in range(26) if h != 7]]) == {}


def test_quarter_hour_days_of_92_96_and_100_periods_are_whole():
    clock = list(range(96))
    without_hour_2 = [p for p in clock if not 8 <= p < 12]
    assert whole([clock, without_hour_2, list(range(100))], per_hour=4) == {96: [0], 92: [1], 100: [2]}


def test_24_quarter_hours_are_not_an_hourly_day():
    assert whole([CLOCK], per_hour=4) == {}


def test_days_come_back_in_ascending_order_by_length():
    found = whole_days(*frame([CLOCK, [0, 1], CLOCK, WITHOUT_2, CLOCK]), 1)

    assert [(periods, days.tolist()) for periods, days in found] == [(24, [0, 2, 4]), (23, [3])]


def test_no_rows_is_no_day():
    assert whole_days(np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=bool), 1) == []


def test_the_largest_id_says_whether_periods_are_hours_or_quarter_hours():
    assert periods_per_hour(np.arange(24)) == 1
    assert periods_per_hour(np.arange(25)) == 1  # a 25-hour day numbered through
    assert periods_per_hour(np.arange(96)) == 4
    assert periods_per_hour(np.array([], dtype=int)) == 1


def test_the_templates_hold_the_ids_of_each_length():
    ids = whole_day_ids(1)

    assert sorted(ids) == [23, 24, 25]
    assert len(ids[23]) == 4  # numbered through (Sundays only) and one for each hour a clock change can skip
