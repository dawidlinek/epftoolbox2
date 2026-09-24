import pandas as pd
import pytest

from epftoolbox2.data.validators import ContinuityValidator

TZ = "Europe/Warsaw"


def frame(index):
    return pd.DataFrame({"x": 1.0}, index=index)


def hours(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="1h", tz="UTC")


def test_duplicate_timestamp_is_invalid():
    idx = hours(6)
    result = ContinuityValidator("1h").validate(frame(idx.append(idx[[3]]).sort_values()))

    assert not result.is_valid
    assert result.info["duplicate_count"] == 1
    assert result.info["duplicates"] == [idx[3]]
    assert any("duplicate" in e for e in result.errors)


def test_unsorted_index_is_invalid_even_when_steps_stay_within_freq():
    idx = hours(6)
    result = ContinuityValidator("2h").validate(frame(idx[[0, 2, 1, 3, 4, 5]]))

    assert not result.is_valid
    assert result.info["unsorted_count"] == 1
    assert any("not sorted" in e for e in result.errors)


def test_infers_quarter_hours_and_finds_sub_hour_holes():
    q = pd.date_range("2024-01-01", periods=96, freq="15min", tz="UTC")
    result = ContinuityValidator().validate(frame(q.delete([5, 6, 7, 40, 41])))

    assert not result.is_valid
    assert result.info["expected_freq"] == "15min"
    assert result.info["freq_inferred"] is True
    assert result.info["gap_count"] == 2
    assert result.info["missing_periods"] == 5
    assert [g["missing"] for g in result.info["gaps"]] == [3, 2]


def test_explicit_hourly_freq_on_quarter_hour_data_is_invalid():
    q = pd.date_range("2024-01-01", periods=96, freq="15min", tz="UTC")
    result = ContinuityValidator("1h").validate(frame(q))

    assert not result.is_valid
    assert result.info["irregular_steps"] == 95


def test_mixed_resolution_is_invalid():
    hourly = pd.date_range("2025-09-30", "2025-09-30 23:00", freq="1h", tz="UTC")
    quarter = pd.date_range("2025-10-01", periods=24 * 4 * 2, freq="15min", tz="UTC")
    result = ContinuityValidator().validate(frame(hourly.append(quarter)))

    assert result.info["expected_freq"] == "15min"
    assert not result.is_valid
    assert result.info["irregular_steps"] == 0  # 1h steps are whole multiples of 15 min...
    assert result.info["gap_count"] == 24  # ...so each one is three missing quarter-hours


@pytest.mark.parametrize("freq", ["1h", "15min"])
def test_local_index_across_both_dst_switches_is_valid(freq):
    idx = pd.date_range("2024-03-30", "2024-11-01", freq=freq, tz="UTC").tz_convert(TZ)
    result = ContinuityValidator().validate(frame(idx))

    assert result.is_valid, result.errors
    assert result.info["expected_freq"] == freq


def test_naive_wall_clock_index_is_invalid():
    idx = pd.date_range("2024-03-30", "2024-11-01", freq="1h", tz="UTC").tz_convert(TZ).tz_localize(None)
    result = ContinuityValidator("1h").validate(frame(idx))

    assert not result.is_valid
    assert result.info["duplicate_count"] == 1  # the repeated 02:00 in October
    assert result.info["gap_count"] == 1  # the skipped 02:00 in March


def test_daily_local_index_across_dst_is_valid():
    idx = pd.date_range("2024-03-25", "2024-11-05", freq="D", tz=TZ)
    assert ContinuityValidator("1D").validate(frame(idx)).is_valid
    assert ContinuityValidator().validate(frame(idx)).is_valid


def test_gap_messages_are_capped_but_info_is_complete():
    idx = hours(24 * 10).delete(list(range(1, 24 * 10, 4)))
    result = ContinuityValidator("1h").validate(frame(idx))

    assert result.info["gap_count"] == 60
    assert len(result.errors) == 11
    assert result.errors[-1] == "... and 50 more gaps"


def test_single_row_warns():
    result = ContinuityValidator().validate(frame(hours(1)))

    assert result.is_valid
    assert len(result.warnings) == 1


def test_non_fixed_freq_raises():
    with pytest.raises(ValueError, match="freq"):
        ContinuityValidator("MS")
