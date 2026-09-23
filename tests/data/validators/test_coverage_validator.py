import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.validators import ContinuityValidator, CoverageValidator, NullCheckValidator
from tests.synthetic import make_market_data

TZ = "Europe/Warsaw"


@pytest.fixture
def hourly():
    index = pd.date_range("2024-01-01", periods=24 * 10, freq="1h", tz="UTC")
    return pd.DataFrame({"price": 50.0 + np.arange(len(index)), "load": 1000.0}, index=index)


def test_single_column_outage_on_a_continuous_index(hourly):
    # The review's 3-day price hole: the index is complete, so ContinuityValidator can't see it.
    hourly.loc[hourly.index[48:120], "price"] = np.nan
    assert ContinuityValidator().validate(hourly).is_valid

    result = CoverageValidator().validate(hourly)

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
    assert NullCheckValidator(columns=["price"]).validate(thinned).is_valid

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


def test_missing_and_empty_columns(hourly):
    hourly["empty"] = np.nan

    result = CoverageValidator(columns=["price", "empty", "nope"]).validate(hourly)

    assert not result.is_valid
    assert result.info["missing_columns"] == ["nope"]
    assert any("'empty' has no values" in e for e in result.errors)


def test_invalid_min_coverage_raises():
    with pytest.raises(ValueError, match="min_coverage"):
        CoverageValidator(min_coverage=1.5)
