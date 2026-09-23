import datetime

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.data.validators import FreshnessValidator

TZ = "Europe/Warsaw"


@pytest.fixture
def frame():
    index = pd.date_range("2024-06-01", periods=24 * 5, freq="1h", tz="UTC").tz_convert(TZ)
    df = pd.DataFrame({"price": 1.0, "load_forecast": 2.0, "weather_d+1": 3.0}, index=index)
    df.loc[df.index[-48:], "price"] = np.nan  # the price feed stopped two days before the rest
    return df


def test_column_ending_before_the_frame_is_stale(frame):
    result = FreshnessValidator().validate(frame)

    assert not result.is_valid
    assert result.info["stale_columns"] == ["price"]
    assert result.info["expected_end"] == frame.index[-1]
    assert result.info["last_valid"]["price"] == frame.index[-49]
    assert len(result.errors) == 1 and "'price' ends at" in result.errors[0]
    behind = result.stats.set_index("column")["behind"]
    assert behind["price"] == pd.Timedelta(hours=48)
    assert behind["load_forecast"] == pd.Timedelta(0)


def test_expected_end_and_max_delay(frame):
    end = "2024-06-06 03:00+02:00"  # two hours after the last row

    strict = FreshnessValidator(columns=["load_forecast"], expected_end=end).validate(frame)
    tolerant = FreshnessValidator(columns=["load_forecast"], expected_end=end, max_delay="2h").validate(frame)

    assert not strict.is_valid
    assert tolerant.is_valid


def test_naive_expected_end_is_utc(frame):
    # The last row is 2024-06-05 23:00 UTC.
    assert FreshnessValidator(columns=["load_forecast"], expected_end="2024-06-05 23:00").validate(frame).is_valid
    assert not FreshnessValidator(columns=["load_forecast"], expected_end="2024-06-06 00:00").validate(frame).is_valid


def test_today_keyword():
    today = pd.Timestamp(datetime.date.today().isoformat(), tz="UTC")
    index = pd.date_range(today - pd.Timedelta(days=3), today, freq="1h")
    df = pd.DataFrame({"fresh": 1.0, "stale": 1.0}, index=index)
    df.loc[df.index[-2:], "stale"] = np.nan

    result = FreshnessValidator(expected_end="today").validate(df)

    assert result.info["stale_columns"] == ["stale"]


def test_empty_and_missing_columns(frame):
    frame["empty"] = np.nan

    result = FreshnessValidator(columns=["empty", "nope", "load_forecast"]).validate(frame)

    assert not result.is_valid
    assert result.info["missing_columns"] == ["nope"]
    assert result.info["stale_columns"] == ["empty"]
    assert any("'empty' has no values" in e for e in result.errors)


def test_negative_max_delay_raises():
    with pytest.raises(ValueError, match="max_delay"):
        FreshnessValidator(max_delay="-1h")
