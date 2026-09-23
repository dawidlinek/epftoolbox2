"""Calendar-day shifts on local-time indexes (`epftoolbox2._date_utils.shift_calendar_days`)."""

import numpy as np
import pandas as pd
import pytest

from epftoolbox2._date_utils import shift_calendar_days
from tests.synthetic import calendar_shift

TZ = "Europe/Warsaw"


def local(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz=TZ)


# Europe/Warsaw fall-back days: 2023-10-29 and 2024-10-27 (364 days apart); 02:00 occurs twice.
FIRST_0200 = pd.Timestamp("2024-10-27 02:00+02:00").tz_convert(TZ)
SECOND_0200 = pd.Timestamp("2024-10-27 02:00+01:00").tz_convert(TZ)


def shift_one(ts: pd.Timestamp, days: int, **kwargs) -> pd.Timestamp:
    return shift_calendar_days(pd.DatetimeIndex([ts]), days, **kwargs)[0]


def test_spring_switch_keeps_wall_clock_hour():
    assert shift_one(local("2024-03-31 10:00"), -1) == local("2024-03-30 10:00")
    assert shift_one(local("2024-03-30 10:00"), 1) == local("2024-03-31 10:00")
    assert shift_one(local("2024-04-07 03:00"), -7) == local("2024-03-31 03:00")


def test_skipped_spring_hour_gives_nat():
    assert shift_one(local("2024-04-01 02:00"), -1) is pd.NaT
    assert shift_one(local("2024-03-30 02:30"), 1) is pd.NaT


def test_repeated_autumn_hour_falls_back_to_the_single_occurrence():
    assert shift_one(FIRST_0200, -1) == local("2024-10-26 02:00")
    assert shift_one(SECOND_0200, -1) == local("2024-10-26 02:00")
    assert shift_one(SECOND_0200, 1) == local("2024-10-28 02:00")


def test_single_hour_onto_repeated_hour_takes_first_occurrence_or_nat():
    assert shift_one(local("2024-10-26 02:00"), 1) == FIRST_0200
    assert shift_one(local("2024-10-28 02:00"), -1) == FIRST_0200
    assert shift_one(local("2024-10-26 02:00"), 1, ambiguous="NaT") is pd.NaT
    # Unambiguous hours are unaffected by `ambiguous`.
    assert shift_one(local("2024-10-26 03:00"), 1, ambiguous="NaT") == local("2024-10-27 03:00")


def test_repeated_hour_maps_occurrence_to_occurrence():
    first_2023 = pd.Timestamp("2023-10-29 02:00+02:00").tz_convert(TZ)
    second_2023 = pd.Timestamp("2023-10-29 02:00+01:00").tz_convert(TZ)
    assert shift_one(FIRST_0200, -364) == first_2023
    assert shift_one(SECOND_0200, -364) == second_2023
    assert shift_one(SECOND_0200, -364, ambiguous="NaT") == second_2023


@pytest.mark.parametrize("freq", ["1h", "15min"])
@pytest.mark.parametrize("days", [1, 7, -1, -7])
def test_matches_reference_calendar_shift_over_two_years(freq, days):
    index = pd.date_range("2023-01-01", "2025-01-01", freq=freq, tz=TZ, inclusive="left")
    series = pd.Series(np.random.default_rng(0).normal(size=len(index)), index=index)

    ours = series.reindex(shift_calendar_days(index, days)).to_numpy()
    reference = calendar_shift(series, days).to_numpy()

    np.testing.assert_array_equal(ours, reference)


def test_naive_index_shifts_by_whole_days():
    index = pd.date_range("2024-03-30", periods=48, freq="1h")
    assert (shift_calendar_days(index, -1) == index - pd.Timedelta(days=1)).all()


def test_utc_index_shifts_by_utc_days():
    index = pd.date_range("2024-03-30", periods=48, freq="1h", tz="UTC")
    assert (shift_calendar_days(index, 2) == index + pd.Timedelta(days=2)).all()


def test_keeps_timezone_and_length():
    index = pd.date_range("2024-10-26", "2024-10-29", freq="15min", tz=TZ, inclusive="left")
    out = shift_calendar_days(index, -1)
    assert len(out) == len(index)
    assert str(out.tz) == TZ


def test_rejects_unknown_ambiguous_option():
    with pytest.raises(ValueError, match="ambiguous"):
        shift_calendar_days(pd.DatetimeIndex([local("2024-01-01")]), 1, ambiguous="infer")
