import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.validators import CoverageValidator
from tests.synthetic import make_market_data

TZ = "Europe/Warsaw"


@pytest.fixture
def hourly():
    index = pd.date_range("2024-01-01", periods=24 * 10, freq="1h", tz="UTC")
    return pd.DataFrame({"price": 50.0 + np.arange(len(index)), "load": 1000.0}, index=index)


def frame(index):
    return pd.DataFrame({"x": 1.0}, index=index)


def hours(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="1h", tz="UTC")


def test_single_column_outage_on_a_continuous_index(hourly):
    # The review's 3-day price hole: the index is complete, so only the column's own values show it.
    hourly.loc[hourly.index[48:120], "price"] = np.nan

    result = CoverageValidator().validate(hourly)

    assert result.info["duplicate_count"] == 0 and result.info["unsorted_count"] == 0
    assert not result.is_valid
    assert len(result.errors) == 1 and "'price'" in result.errors[0]
    assert result.info["coverage"] == {"price": pytest.approx(1 - 72 / 240), "load": 1.0}
    assert result.info["longest_missing_run"] == {"price": 72, "load": 0}
    row = result.stats.set_index("column").loc["price"]
    assert row["longest_missing_start"] == hourly.index[48]
    assert row["longest_missing_end"] == hourly.index[119]
    assert row["missing"] == 72


def test_rows_missing_from_the_index_count_as_missing(hourly):
    thinned = hourly.drop(hourly.index[100:105])
    assert thinned["price"].notna().all()  # a null check on the rows present would see nothing wrong here

    result = CoverageValidator(columns=["price"]).validate(thinned)

    assert not result.is_valid
    assert result.stats.loc[0, "missing"] == 5


def test_requested_range_counts_late_start_and_early_end(hourly):
    hourly.loc[hourly.index[:48], "price"] = np.nan

    own_span = CoverageValidator(columns=["price"]).validate(hourly)
    requested = CoverageValidator(columns=["price"], start="2024-01-01", end="2024-01-11").validate(hourly)
    beyond = CoverageValidator(columns=["load"], start="2024-01-01", end="2024-01-12").validate(hourly)

    assert own_span.is_valid
    assert own_span.info["first_valid"]["price"] == hourly.index[48]
    assert not requested.is_valid
    assert requested.stats.loc[0, "expected"] == 240
    assert requested.stats.loc[0, "missing"] == 48
    assert not beyond.is_valid and beyond.stats.loc[0, "missing"] == 24


def test_end_is_exclusive(hourly):
    result = CoverageValidator(start="2024-01-01", end="2024-01-11").validate(hourly)
    assert result.is_valid, result.errors


def test_lag_columns_are_measured_over_their_own_span():
    df = make_market_data(days=40)  # price_d-1 is empty on the first day, load_d+2 on the last two
    assert CoverageValidator().validate(df).is_valid


def test_min_coverage_and_max_gap(hourly):
    hourly.loc[hourly.index[100:106], "price"] = np.nan  # 6 h, 2.5 %

    relaxed = CoverageValidator(columns=["price"], min_coverage=0.95).validate(hourly)
    gap_limited = CoverageValidator(columns=["price"], min_coverage=0.95, max_gap="3h").validate(hourly)

    assert relaxed.is_valid and len(relaxed.warnings) == 1
    assert not gap_limited.is_valid
    assert "longer than max_gap 3h" in gap_limited.errors[0]


def test_quarter_hours_on_a_local_index_across_dst():
    index = pd.date_range("2024-03-30", "2024-04-02", freq="15min", tz="UTC", inclusive="left").tz_convert(TZ)
    df = pd.DataFrame({"price": 1.0}, index=index)
    df.loc[df.index[200:203], "price"] = np.nan

    result = CoverageValidator(start="2024-03-30", end="2024-04-02").validate(df)

    assert result.stats.loc[0, "expected"] == 3 * 96
    assert result.stats.loc[0, "missing"] == 3
    assert result.stats.loc[0, "longest_missing_start"] == df.index[200]


def test_daily_local_index_across_dst():
    index = pd.date_range("2024-03-25", "2024-04-05", freq="D", tz=TZ)
    df = pd.DataFrame({"x": 1.0}, index=index)
    assert CoverageValidator().validate(df).is_valid
    assert CoverageValidator(start="2024-03-24 23:00", end="2024-04-05 22:00").validate(df).is_valid


def test_step_is_inferred_per_column_on_a_mixed_resolution_frame():
    # Raw source frame: 15-minute load and hourly price share one index.
    index = pd.date_range("2024-10-22", "2024-10-25", freq="15min", tz="UTC", inclusive="left")
    df = pd.DataFrame({"load_actual": 1.0, "price": np.nan}, index=index)
    df.loc[df.index.minute == 0, "price"] = 50.0
    df.loc[pd.Timestamp("2024-10-23 05:00", tz="UTC"), "price"] = np.nan

    result = CoverageValidator().validate(df)
    as_quarter_hours = CoverageValidator(columns=["price"], freq="15min").validate(df)

    assert result.info["freq"] == {"load_actual": "15min", "price": "1h"}
    assert result.info["off_grid"] == {"load_actual": 0, "price": 0}
    assert result.errors == [result.errors[0]] and "'price'" in result.errors[0]
    assert result.stats.set_index("column").loc["price", "missing"] == 1
    assert as_quarter_hours.stats.loc[0, "coverage"] < 0.3


def test_daily_column_in_an_hourly_frame():
    index = pd.date_range("2024-03-25", "2024-04-05", freq="1h", tz=TZ, inclusive="left")
    df = pd.DataFrame({"load": 1.0, "daily_min": np.nan}, index=index)
    df.loc[df.index.hour == 0, "daily_min"] = 7.0  # one value per local midnight, across the spring switch

    result = CoverageValidator().validate(df)

    assert result.is_valid, result.errors
    assert result.info["freq"]["daily_min"] == "1D"


def test_missing_and_empty_columns(hourly):
    hourly["empty"] = np.nan

    result = CoverageValidator(columns=["price", "empty", "nope"]).validate(hourly)

    assert not result.is_valid
    assert result.info["missing_columns"] == ["nope"]
    assert any("'empty' has no values" in e for e in result.errors)


def test_invalid_min_coverage_raises():
    with pytest.raises(ValueError, match="min_coverage"):
        CoverageValidator(min_coverage=1.5)


# Index checks: they hold whichever columns are measured.


@pytest.mark.parametrize("columns", [None, ["x"]])
def test_duplicate_timestamp_is_an_error(columns):
    idx = hours(6)
    result = CoverageValidator(columns=columns, freq="1h").validate(frame(idx.append(idx[[3]]).sort_values()))

    assert not result.is_valid
    assert result.info["duplicate_count"] == 1
    assert result.info["duplicates"] == [idx[3]]
    assert len(result.errors) == 1 and "duplicate" in result.errors[0]
    assert result.stats.loc[0, "missing"] == 0  # every grid point still has a value: only the duplicate is wrong


def test_unsorted_index_is_an_error_even_when_every_value_is_present():
    idx = hours(6)
    result = CoverageValidator(freq="1h").validate(frame(idx[[0, 2, 1, 3, 4, 5]]))

    assert not result.is_valid
    assert result.info["unsorted_count"] == 1
    assert len(result.errors) == 1 and "not sorted" in result.errors[0]
    assert result.stats.loc[0, "missing"] == 0


def test_sorted_unique_index_reports_no_index_problems(hourly):
    result = CoverageValidator().validate(hourly)

    assert result.is_valid
    assert result.info["duplicate_count"] == 0 and result.info["unsorted_count"] == 0
    assert "duplicates" not in result.info


def test_missing_timestamp_with_an_inferred_freq():
    idx = hours(24).delete(5)
    result = CoverageValidator().validate(frame(idx))

    assert not result.is_valid
    assert result.info["freq"] == {"x": "1h"}
    assert result.stats.loc[0, "expected"] == 24 and result.stats.loc[0, "missing"] == 1
    assert result.stats.loc[0, "longest_missing_start"] == hours(24)[5]


def test_infers_quarter_hours_and_finds_sub_hour_holes():
    q = pd.date_range("2024-01-01", periods=96, freq="15min", tz="UTC")
    result = CoverageValidator().validate(frame(q.delete([5, 6, 7, 40, 41])))

    assert not result.is_valid
    assert result.info["freq"] == {"x": "15min"}
    assert result.stats.loc[0, "expected"] == 96
    assert result.stats.loc[0, "missing"] == 5
    assert result.info["longest_missing_run"] == {"x": 3}
    assert result.stats.loc[0, "longest_missing_start"] == q[5]
    assert result.stats.loc[0, "longest_missing_end"] == q[7]


def test_mixed_resolution_is_invalid():
    hourly = pd.date_range("2025-09-30", "2025-09-30 23:00", freq="1h", tz="UTC")
    quarter = pd.date_range("2025-10-01", periods=24 * 4 * 2, freq="15min", tz="UTC")
    result = CoverageValidator().validate(frame(hourly.append(quarter)))

    assert result.info["freq"] == {"x": "15min"}
    assert not result.is_valid
    assert result.stats.loc[0, "missing"] == 24 * 3  # each hourly row is followed by three missing quarter-hours


def test_explicit_hourly_freq_on_quarter_hour_data_is_invalid():
    q = pd.date_range("2024-01-01", periods=96, freq="15min", tz="UTC")
    result = CoverageValidator(freq="1h").validate(frame(q))

    assert not result.is_valid
    assert result.info["off_grid"] == {"x": 72}  # 24 of the 96 values lie on the hourly grid
    assert len(result.errors) == 1 and "between the 1h grid points" in result.errors[0]
    assert result.stats.loc[0, "missing"] == 0


def test_value_between_grid_points_is_an_error_even_when_every_grid_point_has_a_value():
    idx = hours(10).append(hours(10)[[4]] + pd.Timedelta("30min")).sort_values()
    result = CoverageValidator().validate(frame(idx))

    assert not result.is_valid
    assert result.stats.loc[0, "missing"] == 0
    assert result.info["off_grid"] == {"x": 1}
    assert result.stats.loc[0, "off_grid"] == 1


@pytest.mark.parametrize("freq", ["1h", "15min"])
def test_local_index_across_both_dst_switches_is_valid(freq):
    idx = pd.date_range("2024-03-30", "2024-11-01", freq=freq, tz="UTC").tz_convert(TZ)
    result = CoverageValidator().validate(frame(idx))

    assert result.is_valid, result.errors
    assert result.info["freq"] == {"x": freq}
    assert result.info["duplicate_count"] == 0 and result.info["off_grid"] == {"x": 0}
    assert result.stats.loc[0, "missing"] == 0


def test_naive_wall_clock_index_is_invalid():
    idx = pd.date_range("2024-03-30", "2024-11-01", freq="1h", tz="UTC").tz_convert(TZ).tz_localize(None)
    result = CoverageValidator(freq="1h").validate(frame(idx))

    assert not result.is_valid
    assert result.info["duplicate_count"] == 1  # the repeated 02:00 in October
    assert result.stats.loc[0, "missing"] == 1  # the skipped 02:00 in March


def test_daily_local_index_across_both_dst_switches_is_valid():
    idx = pd.date_range("2024-03-25", "2024-11-05", freq="D", tz=TZ)

    for validator in (CoverageValidator(freq="1D"), CoverageValidator()):
        result = validator.validate(frame(idx))
        assert result.is_valid, result.errors
        assert result.stats.loc[0, "expected"] == len(idx)
        assert result.info["off_grid"] == {"x": 0}


def test_missing_day_on_a_dst_switch_is_one_missing_local_day():
    idx = pd.date_range("2024-03-28", "2024-04-03", freq="D", tz=TZ)
    result = CoverageValidator().validate(frame(idx.delete(3)))  # 2024-03-31, the 23-hour day

    assert not result.is_valid
    assert result.stats.loc[0, "missing"] == 1
    assert result.stats.loc[0, "longest_missing_start"] == idx[3]


def test_empty_frame_warns():
    result = CoverageValidator().validate(pd.DataFrame())

    assert result.is_valid
    assert len(result.warnings) == 1


def test_non_datetime_index_is_an_error():
    result = CoverageValidator().validate(pd.DataFrame({"value": [1, 2, 3]}))

    assert not result.is_valid
    assert "DatetimeIndex" in result.errors[0]


def test_single_row_warns():
    result = CoverageValidator().validate(frame(hours(1)))

    assert result.is_valid
    assert len(result.warnings) == 1


def test_non_fixed_freq_raises():
    with pytest.raises(ValueError, match="freq"):
        CoverageValidator(freq="MS")
