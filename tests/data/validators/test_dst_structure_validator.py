import datetime

import pandas as pd
import pytest

from epftoolbox2.data.validators import DSTStructureValidator
from tests.synthetic import make_market_data

TZ = "Europe/Warsaw"
SPRING, FALL = datetime.date(2024, 3, 31), datetime.date(2024, 10, 27)


def local_days(first, last, freq="1h"):
    """Complete local days from `first` to `last` inclusive."""
    start = pd.Timestamp(first, tz=TZ)
    end = pd.Timestamp(last, tz=TZ) + pd.offsets.Day(1)
    return pd.DataFrame({"x": 1.0}, index=pd.date_range(start, end, freq=freq, inclusive="left"))


@pytest.mark.parametrize("freq, per_hour", [("1h", 1), ("15min", 4)])
def test_complete_local_days_across_both_switches_are_valid(freq, per_hour):
    result = DSTStructureValidator().validate(local_days("2024-03-25", "2024-11-03", freq))

    assert result.is_valid, result.errors
    assert result.info["tz"] == TZ
    assert result.info["dst_days"] == [SPRING, FALL]
    counts = result.info["rows_per_day"]
    assert counts[23 * per_hour] == 1 and counts[25 * per_hour] == 1
    stats = result.stats.set_index("date")
    assert stats.loc[SPRING, "expected"] == 23 * per_hour
    assert stats.loc[FALL, "hours"] == 25


def test_synthetic_market_data_has_the_right_structure():
    df = make_market_data(days=150)  # starts at local midnight and spans the spring switch
    assert DSTStructureValidator().validate(df).is_valid


def test_naive_wall_clock_index_is_invalid():
    df = local_days("2024-03-25", "2024-11-03")
    df.index = df.index.tz_localize(None)  # 23 rows on the spring day, 25 in October, all counted as 24-h days

    result = DSTStructureValidator().validate(df)

    assert not result.is_valid
    assert [d["date"] for d in result.info["bad_days"]] == [SPRING, FALL]
    assert "has 23 rows, expected 24" in result.errors[0]


def test_utc_source_frame_checked_in_local_time():
    # Raw source output: UTC index from UTC midnight, so the first local day starts at 01:00 and the last is cut short.
    index = pd.date_range("2024-03-25", "2024-04-05", freq="1h", tz="UTC", inclusive="left")
    df = pd.DataFrame({"x": 1.0}, index=index)

    result = DSTStructureValidator(tz=TZ).validate(df)

    assert result.is_valid, result.errors
    assert len(result.warnings) == 2  # the partial first and last local days
    assert result.info["dst_days"] == [SPRING]

    strict = DSTStructureValidator(tz=TZ, allow_partial_edges=False).validate(df)
    assert not strict.is_valid
    assert len(strict.errors) == 2


def test_missing_hour_on_a_normal_day_is_invalid():
    df = local_days("2024-01-01", "2024-01-05").drop(pd.Timestamp("2024-01-03 12:00", tz=TZ))

    result = DSTStructureValidator().validate(df)

    assert not result.is_valid
    assert result.info["bad_days"] == [{"date": datetime.date(2024, 1, 3), "rows": 23, "expected": 24}]


def test_duplicate_row_is_invalid():
    df = local_days("2024-01-01", "2024-01-05")
    df = pd.concat([df, df.iloc[[30]]]).sort_index()

    assert not DSTStructureValidator().validate(df).is_valid


def test_missing_whole_day_is_invalid():
    df = local_days("2024-01-01", "2024-01-05")
    df = df[df.index.date != datetime.date(2024, 1, 3)]

    result = DSTStructureValidator().validate(df)

    assert not result.is_valid
    assert result.info["bad_days"][0]["rows"] == 0


def test_explicit_freq_that_does_not_match_is_invalid():
    result = DSTStructureValidator(freq="15min").validate(local_days("2024-01-01", "2024-01-03"))

    assert not result.is_valid
    assert "has 24 rows, expected 96" in result.errors[0]


def test_daily_data_is_skipped_with_a_warning():
    df = pd.DataFrame({"x": 1.0}, index=pd.date_range("2024-03-25", periods=20, freq="D", tz=TZ))

    result = DSTStructureValidator().validate(df)

    assert result.is_valid
    assert len(result.warnings) == 1


def test_unknown_time_zone_raises():
    with pytest.raises(ValueError, match="time zone"):
        DSTStructureValidator(tz="Mars/Olympus")
