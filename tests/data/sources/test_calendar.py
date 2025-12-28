import pandas as pd
import pytest
from epftoolbox2.data.sources.calendar import CalendarSource, COUNTRY_DATA


class TestCalendarSourceInit:
    def test_init_valid_country(self):
        source = CalendarSource(country="PL")
        assert source.country == "PL"
        assert source.timezone == "Europe/Warsaw"

    def test_init_lowercase_country(self):
        source = CalendarSource(country="pl")
        assert source.country == "PL"

    def test_init_invalid_country(self):
        with pytest.raises(ValueError, match="Unsupported country"):
            CalendarSource(country="XX")

    def test_init_custom_timezone(self):
        source = CalendarSource(country="PL", timezone="UTC")
        assert source.timezone == "UTC"

    def test_init_invalid_holidays_value(self):
        with pytest.raises(ValueError, match="Invalid holidays value"):
            CalendarSource(country="PL", holidays="invalid")

    def test_init_invalid_weekday_value(self):
        with pytest.raises(ValueError, match="Invalid weekday value"):
            CalendarSource(country="PL", weekday="invalid")


class TestCalendarSourceFetch:
    def test_fetch_returns_dataframe(self):
        source = CalendarSource(country="PL")
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        df = source.fetch(start, end)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_fetch_index_is_utc(self):
        source = CalendarSource(country="PL")
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        df = source.fetch(start, end)
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"


class TestHolidayFeatures:
    def test_holidays_binary(self):
        source = CalendarSource(country="PL", holidays="binary", weekday=False)
        start = pd.Timestamp("2024-12-25", tz="UTC")
        end = pd.Timestamp("2024-12-26", tz="UTC")
        df = source.fetch(start, end)
        assert "is_holiday" in df.columns
        assert df["is_holiday"].iloc[0] == 1

    def test_holidays_name(self):
        source = CalendarSource(country="PL", holidays="name", weekday=False)
        start = pd.Timestamp("2024-12-25", tz="UTC")
        end = pd.Timestamp("2024-12-26", tz="UTC")
        df = source.fetch(start, end)
        assert "holiday_name" in df.columns
        assert df["holiday_name"].iloc[0] is not None

    def test_holidays_onehot(self):
        source = CalendarSource(country="PL", holidays="onehot", weekday=False)
        start = pd.Timestamp("2024-12-25", tz="UTC")
        end = pd.Timestamp("2024-12-26", tz="UTC")
        df = source.fetch(start, end)
        assert "is_holiday" in df.columns
        holiday_cols = [c for c in df.columns if c.startswith("holiday_")]
        assert len(holiday_cols) > 0

    def test_holidays_false(self):
        source = CalendarSource(country="PL", holidays=False, weekday=False)
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        df = source.fetch(start, end)
        assert "is_holiday" not in df.columns


class TestWeekdayFeatures:
    def test_weekday_number(self):
        source = CalendarSource(country="PL", holidays=False, weekday="number")
        start = pd.Timestamp("2024-01-01", tz="UTC")  # Monday
        end = pd.Timestamp("2024-01-02", tz="UTC")
        df = source.fetch(start, end)
        assert "weekday" in df.columns
        assert df["weekday"].iloc[0] == 0  # Monday

    def test_weekday_name(self):
        source = CalendarSource(country="PL", holidays=False, weekday="name")
        start = pd.Timestamp("2024-01-01", tz="UTC")  # Monday
        end = pd.Timestamp("2024-01-02", tz="UTC")
        df = source.fetch(start, end)
        assert "weekday_name" in df.columns
        assert df["weekday_name"].iloc[0] == "monday"

    def test_weekday_onehot(self):
        source = CalendarSource(country="PL", holidays=False, weekday="onehot")
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-08", tz="UTC")
        df = source.fetch(start, end)
        weekday_cols = [c for c in df.columns if c.startswith("weekday_")]
        assert len(weekday_cols) == 7


class TestHourFeatures:
    def test_hour_number(self):
        source = CalendarSource(country="PL", holidays=False, weekday=False, hour="number")
        start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        end = pd.Timestamp("2024-01-01 12:00", tz="UTC")
        df = source.fetch(start, end)
        assert "hour" in df.columns

    def test_hour_onehot(self):
        source = CalendarSource(country="PL", holidays=False, weekday=False, hour="onehot")
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        df = source.fetch(start, end)
        hour_cols = [c for c in df.columns if c.startswith("hour_")]
        assert len(hour_cols) == 24


class TestMonthFeatures:
    def test_month_number(self):
        source = CalendarSource(country="PL", holidays=False, weekday=False, month="number")
        start = pd.Timestamp("2024-03-01", tz="UTC")
        end = pd.Timestamp("2024-03-02", tz="UTC")
        df = source.fetch(start, end)
        assert "month" in df.columns
        assert df["month"].iloc[0] == 3

    def test_month_name(self):
        source = CalendarSource(country="PL", holidays=False, weekday=False, month="name")
        start = pd.Timestamp("2024-03-01", tz="UTC")
        end = pd.Timestamp("2024-03-02", tz="UTC")
        df = source.fetch(start, end)
        assert "month_name" in df.columns
        assert df["month_name"].iloc[0] == "march"

    def test_month_onehot(self):
        source = CalendarSource(country="PL", holidays=False, weekday=False, month="onehot")
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-12-31", tz="UTC")
        df = source.fetch(start, end)
        month_cols = [c for c in df.columns if c.startswith("month_")]
        assert len(month_cols) == 12


class TestDaylightFeatures:
    def test_daylight_columns(self):
        source = CalendarSource(country="PL", holidays=False, weekday=False, daylight=True)
        start = pd.Timestamp("2024-06-21", tz="UTC")
        end = pd.Timestamp("2024-06-22", tz="UTC")
        df = source.fetch(start, end)
        assert "sunrise" in df.columns
        assert "sunset" in df.columns
        assert "daylight_hours" in df.columns

    def test_daylight_summer_longer(self):
        source = CalendarSource(country="PL", holidays=False, weekday=False, daylight=True)
        summer = source.fetch(pd.Timestamp("2024-06-21", tz="UTC"), pd.Timestamp("2024-06-22", tz="UTC"))
        winter = source.fetch(pd.Timestamp("2024-12-21", tz="UTC"), pd.Timestamp("2024-12-22", tz="UTC"))
        assert summer["daylight_hours"].iloc[0] > winter["daylight_hours"].iloc[0]


class TestPrefix:
    def test_custom_prefix(self):
        source = CalendarSource(country="PL", holidays="binary", weekday="number", prefix="cal_")
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        df = source.fetch(start, end)
        assert "cal_is_holiday" in df.columns
        assert "cal_weekday" in df.columns


class TestCountryData:
    def test_all_countries_have_required_fields(self):
        for code, data in COUNTRY_DATA.items():
            assert "timezone" in data, f"{code} missing timezone"
            assert "lat" in data, f"{code} missing lat"
            assert "lon" in data, f"{code} missing lon"

    def test_coordinates_in_valid_range(self):
        for code, data in COUNTRY_DATA.items():
            assert -90 <= data["lat"] <= 90, f"{code} lat out of range"
            assert -180 <= data["lon"] <= 180, f"{code} lon out of range"


class TestCacheConfig:
    def test_cache_config_returns_none(self):
        source = CalendarSource(country="PL")
        assert source.get_cache_config() is None
