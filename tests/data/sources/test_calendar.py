import re
from unittest.mock import patch

import pandas as pd
import pytest
import epftoolbox2.data.sources.calendar as calendar_module
from epftoolbox2.data.sources.calendar import CalendarSource, COUNTRY_DATA


class TestCalendarSourceInit:
    def test_init_valid_country(self):
        source = CalendarSource(country="PL")
        assert source.country == "PL"
        # `timezone` stays unset (it may still be inherited from a DataPipeline); `market_timezone` resolves it,
        # falling back to the country's own zone for standalone use.
        assert source.timezone is None
        assert source.market_timezone == "Europe/Warsaw"

    def test_init_lowercase_country(self):
        source = CalendarSource(country="pl")
        assert source.country == "PL"

    def test_init_invalid_country(self):
        with pytest.raises(ValueError, match="Unsupported country"):
            CalendarSource(country="XX")

    def test_init_custom_timezone(self):
        source = CalendarSource(country="PL", timezone="UTC")
        assert source.timezone == "UTC"

    def test_init_default_coordinates(self):
        source = CalendarSource(country="PL")
        assert source.lat == COUNTRY_DATA["PL"]["lat"]
        assert source.lon == COUNTRY_DATA["PL"]["lon"]

    def test_init_override_coordinates(self):
        source = CalendarSource(country="PL", lat=54.3520, lon=18.6466)
        assert source.lat == 54.3520
        assert source.lon == 18.6466

    def test_init_override_lat_only(self):
        source = CalendarSource(country="PL", lat=54.3520)
        assert source.lat == 54.3520
        assert source.lon == COUNTRY_DATA["PL"]["lon"]

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
        weekday_cols = [
            c
            for c in df.columns
            if c.startswith("is_")
            and any(day in c for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])
        ]
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
        hour_cols = [c for c in df.columns if c.startswith("is_") and c[3:].isdigit()]
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
        month_names = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
        month_cols = [c for c in df.columns if c.startswith("is_") and any(m in c for m in month_names)]
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

    def test_daylight_respects_coordinate_override(self):
        # A more northern latitude has longer daylight at the summer solstice.
        warsaw = CalendarSource(country="PL", holidays=False, weekday=False, daylight=True)
        northern = CalendarSource(country="PL", holidays=False, weekday=False, daylight=True, lat=59.3293, lon=18.0686)
        start = pd.Timestamp("2024-06-21", tz="UTC")
        end = pd.Timestamp("2024-06-22", tz="UTC")
        warsaw_df = warsaw.fetch(start, end)
        northern_df = northern.fetch(start, end)
        assert northern_df["daylight_hours"].iloc[0] > warsaw_df["daylight_hours"].iloc[0]


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
    def test_same_config_gives_the_same_key(self):
        a = CalendarSource(country="PL", holidays="binary", weekday="onehot")
        b = CalendarSource(country="PL", holidays="binary", weekday="onehot")
        assert a.get_cache_config() == b.get_cache_config()

    def test_a_parameter_change_gives_a_different_key(self):
        base = CalendarSource(country="PL", holidays="binary")
        changed = CalendarSource(country="PL", holidays="onehot")
        assert base.get_cache_config() != changed.get_cache_config()

    def test_a_holidays_package_version_change_gives_a_different_key(self, monkeypatch):
        source = CalendarSource(country="PL")
        before = source.get_cache_config()
        monkeypatch.setattr(calendar_module.holidays_lib, "__version__", "0.0.0-test")
        after = source.get_cache_config()
        assert before != after
        assert after["holidays_version"] == "0.0.0-test"

    def test_config_includes_schema_version_and_resolved_timezone(self):
        source = CalendarSource(country="DE")
        config = source.get_cache_config()
        assert config["schema_version"] >= 1
        assert config["timezone"] == "Europe/Berlin"

    def test_school_holidays_is_part_of_the_key(self):
        national = CalendarSource(country="PL", school_holidays="national")
        all_regions = CalendarSource(country="PL", school_holidays="all")
        off = CalendarSource(country="PL")
        assert len({str(c.get_cache_config()) for c in (national, all_regions, off)}) == 3

    def test_entries_written_before_unpublished_school_years_were_left_unsettled_are_not_served(self):
        # Version 2 entries may hold NaN rows for school years the API had not published, recorded as covered.
        assert CalendarSource(country="PL", school_holidays="all").get_cache_config()["schema_version"] >= 3

    def test_true_is_the_same_key_as_national(self):
        true = CalendarSource(country="PL", school_holidays=True).get_cache_config()
        assert true == CalendarSource(country="PL", school_holidays="national").get_cache_config()


def _local_day(day: str, tz: str = "Europe/Warsaw") -> tuple:
    """UTC [start, end) covering one local calendar day."""
    start = pd.Timestamp(day, tz=tz)
    end = start + pd.DateOffset(days=1)
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def _flagged_local_dates(df: pd.DataFrame, column: str, tz: str) -> list:
    local = df.tz_convert(tz)
    return sorted({ts.date().isoformat() for ts in local.index[local[column] == 1]})


@pytest.fixture
def c_locale(monkeypatch):
    """A process locale under which the `holidays` package falls back to each country's own language."""
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("LANG", "C")
    monkeypatch.delenv("LANGUAGE", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)


class TestRange:
    def test_end_is_exclusive(self):
        source = CalendarSource(country="PL")
        df = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"))
        assert len(df) == 24
        assert df.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
        assert df.index[-1] == pd.Timestamp("2024-01-01 23:00", tz="UTC")

    def test_naive_bounds_are_utc(self):
        df = CalendarSource(country="PL").fetch(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))
        assert str(df.index.tz) == "UTC"
        assert df.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")

    @pytest.mark.parametrize(
        "day, freq, rows",
        [
            ("2024-03-31", "1h", 23),
            ("2024-10-27", "1h", 25),
            ("2024-03-31", "15min", 92),
            ("2024-10-27", "15min", 100),
            ("2024-06-12", "1h", 24),
        ],
    )
    def test_rows_per_local_day_across_dst(self, day, freq, rows):
        start, end = _local_day(day)
        df = CalendarSource(country="PL", freq=freq).fetch(start, end)
        assert len(df) == rows
        assert set(df.tz_convert("Europe/Warsaw").index.date) == {pd.Timestamp(day).date()}

    def test_invalid_freq_raises(self):
        with pytest.raises(ValueError, match="freq"):
            CalendarSource(country="PL", freq="7min")


class TestHolidayCalendar:
    def test_holiday_follows_local_date_not_utc_date(self):
        # 1 Nov (All Saints' Day) runs from 2024-10-31 23:00 to 2024-11-01 23:00 UTC in Warsaw.
        df = CalendarSource(country="PL", weekday=False).fetch(
            pd.Timestamp("2024-10-31 20:00", tz="UTC"), pd.Timestamp("2024-11-02 02:00", tz="UTC")
        )
        flag = df["is_holiday"]
        assert flag[pd.Timestamp("2024-10-31 22:00", tz="UTC")] == 0
        assert flag[pd.Timestamp("2024-10-31 23:00", tz="UTC")] == 1
        assert flag[pd.Timestamp("2024-11-01 22:00", tz="UTC")] == 1
        assert flag[pd.Timestamp("2024-11-01 23:00", tz="UTC")] == 0

    def test_sweden_does_not_flag_every_sunday(self):
        df = CalendarSource(country="SE", weekday=False).fetch(
            pd.Timestamp("2024-01-01", tz="Europe/Stockholm"), pd.Timestamp("2025-01-01", tz="Europe/Stockholm")
        )
        dates = _flagged_local_dates(df, "is_holiday", "Europe/Stockholm")
        assert "2024-06-02" not in dates  # an ordinary Sunday
        assert "2024-03-31" in dates  # Easter Sunday is still a holiday
        assert "2024-06-22" in dates  # Midsummer Day
        assert len(dates) == 13

    def test_sweden_onehot_has_no_sunday_column(self):
        df = CalendarSource(country="SE", holidays="onehot", weekday=False).fetch(
            pd.Timestamp("2024-06-01", tz="UTC"), pd.Timestamp("2024-07-01", tz="UTC")
        )
        assert not {c.lower() for c in df.columns} & {"holiday_sunday", "holiday_söndag", "holiday_sondag"}

    @pytest.mark.parametrize("region", ["BY", "DE-BY", "by"])
    def test_region_adds_regional_holidays(self, region):
        start, end = _local_day("2024-05-30", "Europe/Berlin")  # Corpus Christi: a holiday in Bavaria only
        national = CalendarSource(country="DE").fetch(start, end)
        bavaria = CalendarSource(country="DE", region=region).fetch(start, end)
        assert national["is_holiday"].eq(0).all()
        assert bavaria["is_holiday"].eq(1).all()

    def test_region_is_normalised(self):
        assert CalendarSource(country="GB", region="gb-eng").region == "ENG"

    def test_gb_england_easter_monday(self):
        start, end = _local_day("2024-04-01", "Europe/London")
        assert CalendarSource(country="GB").fetch(start, end)["is_holiday"].eq(0).all()
        assert CalendarSource(country="GB", region="ENG").fetch(start, end)["is_holiday"].eq(1).all()

    def test_invalid_region_raises(self):
        with pytest.raises(ValueError, match="region"):
            CalendarSource(country="DE", region="XX")

    def test_categories_add_holidays(self):
        start, end = _local_day("2024-06-21", "Europe/Stockholm")  # Midsummer Eve is not a public holiday
        public = CalendarSource(country="SE").fetch(start, end)
        with_bank = CalendarSource(country="SE", categories=["public", "bank"]).fetch(start, end)
        assert public["is_holiday"].eq(0).all()
        assert with_bank["is_holiday"].eq(1).all()

    def test_categories_string_is_accepted(self):
        assert CalendarSource(country="SE", categories="bank").categories == ["bank"]

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="categor"):
            CalendarSource(country="PL", categories=["catholic"])


class TestHolidayNames:
    def test_names_are_english_whatever_the_locale(self, c_locale):
        # Without a pinned language the `holidays` package names holidays after the process locale.
        df = CalendarSource(country="PL", holidays="name", weekday=False).fetch(*_local_day("2024-12-25"))
        assert set(df["holiday_name"]) == {"Christmas Day"}

    def test_onehot_columns_are_sanitised(self, c_locale):
        df = CalendarSource(country="PL", holidays="onehot", weekday=False).fetch(
            pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")
        )
        holiday_cols = [c for c in df.columns if c.startswith("holiday_")]
        assert "holiday_new_years_day" in holiday_cols
        assert "holiday_all_saints_day" in holiday_cols
        assert all(re.fullmatch(r"holiday_[a-z0-9_]+", c) for c in holiday_cols)

    def test_onehot_splits_days_with_two_holidays(self):
        # 1 Jan in Czechia is both New Year's Day and Independent Czech State Restoration Day.
        df = CalendarSource(country="CZ", holidays="onehot", weekday=False).fetch(
            *_local_day("2024-01-01", "Europe/Prague")
        )
        assert df["holiday_new_years_day"].eq(1).all()
        assert df["holiday_independent_czech_state_restoration_day"].eq(1).all()

    def test_onehot_columns_cover_the_whole_year(self):
        # The column set must not depend on which holidays happen to fall inside the requested range.
        df = CalendarSource(country="PL", holidays="onehot", weekday=False).fetch(*_local_day("2024-03-05"))
        assert "holiday_christmas_day" in df.columns
        assert df["holiday_christmas_day"].eq(0).all()
        assert df["is_holiday"].eq(0).all()


class TestSubstituteDaysOff:
    """The `holidays` package names a substitute day off after the day it replaces ("Day off (substituted from
    01/13/2024)" in Hungary and Bulgaria). Without normalising, every such day is its own one-hot column."""

    HU_TZ, BG_TZ = "Europe/Budapest", "Europe/Sofia"

    @staticmethod
    def _range(first: str, last: str, tz: str) -> tuple:
        """UTC [start, end) covering the local days first..last inclusive."""
        return (
            pd.Timestamp(first, tz=tz).tz_convert("UTC"),
            (pd.Timestamp(last, tz=tz) + pd.DateOffset(days=1)).tz_convert("UTC"),
        )

    @staticmethod
    def _holiday_columns(df: pd.DataFrame) -> list:
        return [c for c in df.columns if c.startswith("holiday_")]

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Day off (substituted from 01/13/2024)", "Day off (substituted)"),
            ("Day off (substituted from 13.01.2024)", "Day off (substituted)"),
            ("Day off (substituted from 2024-01-13)", "Day off (substituted)"),
            ("Christmas Day", "Christmas Day"),
            ("Christmas Day (observed)", "Christmas Day (observed)"),
            (
                "75th anniversary of the East German uprising of 1953",
                "75th anniversary of the East German uprising of 1953",
            ),
            ("Boxing Day (substitute day)", "Boxing Day (substitute day)"),
        ],
    )
    def test_only_the_date_of_a_substitute_day_is_dropped(self, name, expected):
        assert calendar_module._holiday_name(name) == expected

    def test_hungary_onehot_has_one_substitute_day_column(self):
        # 2024-08-19 (from 08/03), 2024-12-24 (from 12/07) and 2024-12-27 (from 12/14) are substitute days off.
        source = CalendarSource(country="HU", holidays="onehot", weekday=False)
        df = source.fetch(*self._range("2024-08-01", "2025-01-05", self.HU_TZ))
        columns = self._holiday_columns(df)

        assert "holiday_day_off_substituted" in columns
        assert [c for c in columns if c.startswith("holiday_day_off")] == ["holiday_day_off_substituted"]
        assert not [c for c in columns if re.search(r"\d", c)]
        assert _flagged_local_dates(df, "holiday_day_off_substituted", self.HU_TZ) == [
            "2024-08-19",
            "2024-12-24",
            "2024-12-27",
        ]
        assert _flagged_local_dates(df, "is_holiday", self.HU_TZ) == [
            "2024-08-19",
            "2024-08-20",
            "2024-10-23",
            "2024-11-01",
            "2024-12-24",
            "2024-12-25",
            "2024-12-26",
            "2024-12-27",
            "2025-01-01",
        ]
        assert "holiday_state_foundation_day" in columns  # 20 August is a holiday of its own and keeps its column

    def test_bulgaria_onehot_has_one_substitute_day_column(self):
        source = CalendarSource(country="BG", holidays="onehot", weekday=False)
        df = source.fetch(*self._range("2016-03-01", "2016-10-01", self.BG_TZ))
        columns = self._holiday_columns(df)

        assert [c for c in columns if c.startswith("holiday_day_off")] == ["holiday_day_off_substituted"]
        assert not [c for c in columns if re.search(r"\d", c)]
        assert _flagged_local_dates(df, "holiday_day_off_substituted", self.BG_TZ) == [
            "2016-03-04",
            "2016-05-23",
            "2016-09-05",
            "2016-09-23",
        ]
        assert df["is_holiday"].loc[df["holiday_day_off_substituted"] == 1].eq(1).all()

    @pytest.mark.parametrize("country", ["HU", "BG"])
    def test_column_set_is_fixed_and_has_no_dated_names(self, country):
        # The set covers 1970-2100, so a range without any substitute day has the same columns as one with them.
        source = CalendarSource(country=country, holidays="onehot", weekday=False)
        quiet = source.fetch(*_local_day("2024-03-05", source.market_timezone))
        busy = source.fetch(*self._range("2015-12-28", "2016-03-05", source.market_timezone))

        assert list(quiet.columns) == list(busy.columns)
        assert "holiday_day_off_substituted" in quiet.columns
        assert quiet["holiday_day_off_substituted"].eq(0).all()
        assert not [c for c in quiet.columns if re.search(r"\d", c)]
        assert len(source._onehot_columns()) < 40  # one column per distinct holiday, not one per dated day

    def test_a_substitute_day_after_2024_has_its_column(self):
        # 2026-01-02 replaces Saturday 2026-01-10.
        source = CalendarSource(country="HU", holidays="onehot", weekday=False)
        df = source.fetch(*_local_day("2026-01-02", self.HU_TZ))
        assert df["holiday_day_off_substituted"].eq(1).all()
        assert df["is_holiday"].eq(1).all()

    def test_name_mode_gives_the_same_name_to_every_substitute_day(self):
        source = CalendarSource(country="HU", holidays="name", weekday=False)
        df = source.fetch(*self._range("2024-08-18", "2024-12-27", self.HU_TZ))
        names = df.tz_convert(self.HU_TZ).groupby(df.tz_convert(self.HU_TZ).index.date)["holiday_name"].first()

        assert set(names.dropna()) == {
            "Day off (substituted)",
            "State Foundation Day",
            "National Day",
            "All Saints' Day",
            "Christmas Day",
            "Second Day of Christmas",
        }
        assert names[pd.Timestamp("2024-08-19").date()] == "Day off (substituted)"
        assert names[pd.Timestamp("2024-12-27").date()] == "Day off (substituted)"

    def test_binary_mode_is_unaffected(self):
        source = CalendarSource(country="HU", holidays="binary", weekday=False)
        df = source.fetch(*self._range("2024-08-18", "2024-08-21", self.HU_TZ))
        assert _flagged_local_dates(df, "is_holiday", self.HU_TZ) == ["2024-08-19", "2024-08-20"]

    @pytest.mark.parametrize("country", [c for c in COUNTRY_DATA if c not in ("HU", "BG")])
    def test_countries_without_substitute_days_keep_every_name(self, country):
        source = CalendarSource(country=country, holidays="onehot", weekday=False)
        calendar = calendar_module._build_calendar(
            country, None, None, source._holiday_language(), list(range(1970, 2101))
        )
        raw = sorted({calendar_module._sanitise_name(n) for d in calendar for n in calendar.get_list(d)})

        assert list(source._onehot_columns()) == raw

    def test_cache_entries_of_the_dated_names_are_not_reused(self):
        assert CalendarSource(country="HU", holidays="onehot").get_cache_config()["schema_version"] >= 4


class TestOnehotColumnSet:
    """The one-hot columns belong to the configuration, not to the fetched range: pieces fetched separately and
    concatenated (the source cache) must line up, and so must the predictors a model names."""

    def _source(self, **kwargs):
        return CalendarSource(country="PL", timezone="Europe/Warsaw", holidays="onehot", weekday=False, **kwargs)

    def test_columns_do_not_depend_on_the_years_fetched(self):
        # Poland made Christmas Eve a public holiday from 2025, so its column must exist in 2024 ranges too, not only in 2025 ones.
        in_2024 = self._source().fetch(*_local_day("2024-12-10"))
        in_2025 = self._source().fetch(*_local_day("2025-12-10"))
        spanning = self._source().fetch(pd.Timestamp("2024-12-01", tz="UTC"), pd.Timestamp("2025-02-01", tz="UTC"))
        assert list(in_2024.columns) == list(in_2025.columns) == list(spanning.columns)
        assert "holiday_christmas_eve" in in_2024.columns
        assert in_2024["holiday_christmas_eve"].eq(0).all()
        assert in_2025["holiday_christmas_eve"].eq(0).all()

    def test_christmas_eve_is_flagged_only_when_it_is_a_holiday(self):
        df = self._source().fetch(pd.Timestamp("2024-12-20", tz="UTC"), pd.Timestamp("2026-01-05", tz="UTC"))
        flagged = _flagged_local_dates(df, "holiday_christmas_eve", "Europe/Warsaw")
        assert flagged == ["2025-12-24"]

    def test_columns_are_integers(self):
        df = self._source().fetch(*_local_day("2024-12-10"))
        holiday_columns = [c for c in df.columns if c.startswith("holiday_")]
        assert holiday_columns
        assert all(df[c].dtype.kind == "i" for c in holiday_columns)

    def test_column_set_follows_the_configuration(self):
        bavaria = CalendarSource(country="DE", region="BY", holidays="onehot", weekday=False)
        plain = CalendarSource(country="DE", holidays="onehot", weekday=False)
        day = _local_day("2024-12-10", "Europe/Berlin")
        assert "holiday_corpus_christi" in bavaria.fetch(*day).columns
        assert "holiday_epiphany" in bavaria.fetch(*day).columns
        assert "holiday_epiphany" not in plain.fetch(*day).columns

    def test_prefix_applies_to_every_column(self):
        df = self._source(prefix="cal_").fetch(*_local_day("2024-12-10"))
        assert all(c.startswith("cal_") for c in df.columns)
        assert "cal_holiday_christmas_eve" in df.columns

    def test_range_without_a_grid_point_has_the_same_columns(self):
        start = pd.Timestamp("2024-12-10 00:10", tz="UTC")
        empty = self._source().fetch(start, start + pd.Timedelta(minutes=30))
        assert empty.empty
        assert list(empty.columns) == list(self._source().fetch(*_local_day("2024-12-10")).columns)

    def test_a_holiday_outside_the_covered_years_raises_instead_of_being_dropped(self, monkeypatch):
        monkeypatch.setattr(calendar_module, "ONEHOT_YEARS", (2000, 2010))
        with pytest.raises(ValueError, match="2000"):
            self._source().fetch(*_local_day("2025-12-24"))

    def test_split_run_through_the_cache_equals_an_uncached_run(self, tmp_path, monkeypatch):
        """Cached pieces fetched separately are concatenated: a column missing from one piece must not become NaN."""
        from epftoolbox2.pipelines import DataPipeline

        monkeypatch.chdir(tmp_path)

        def pipeline():
            return DataPipeline(timezone="Europe/Warsaw").add_source(CalendarSource(country="PL", holidays="onehot"))

        fresh = pipeline().run("2024-12-01", "2025-01-31", cache=False)
        pipeline().run("2024-12-01", "2024-12-31", cache=True)
        split = pipeline().run("2024-12-01", "2025-01-31", cache=True)

        assert not split.isna().any().any()
        assert (split.dtypes == fresh.dtypes).all()
        pd.testing.assert_frame_equal(split, fresh, check_freq=False)


class TestPeriodColumn:
    def test_hourly_hour_column_is_hour_of_day(self):
        source = CalendarSource(country="PL", holidays=False, weekday=False, hour="number")
        df = source.fetch(*_local_day("2024-06-12"))
        assert list(df["hour"]) == list(range(24))
        assert "period" not in df.columns

    def test_quarter_hourly_column_is_named_period(self):
        source = CalendarSource(country="PL", holidays=False, weekday=False, hour="number", freq="15min")
        df = source.fetch(*_local_day("2024-06-12"))
        assert "hour" not in df.columns
        assert list(df["period"]) == list(range(96))

    def test_period_uses_wall_clock_on_fall_back_day(self):
        source = CalendarSource(country="PL", holidays=False, weekday=False, hour="number", freq="15min")
        df = source.fetch(*_local_day("2024-10-27"))
        # 02:00-02:45 happen twice, so periods 8-11 appear twice, as in the models' own period index.
        counts = df["period"].value_counts()
        assert len(df) == 100
        assert counts.loc[[8, 9, 10, 11]].eq(2).all()
        assert counts.drop([8, 9, 10, 11]).eq(1).all()

    def test_half_hourly_period(self):
        source = CalendarSource(country="PL", holidays=False, weekday=False, hour="number", freq="30min")
        df = source.fetch(*_local_day("2024-06-12"))
        assert list(df["period"]) == list(range(48))

    def test_quarter_hourly_onehot_names(self):
        source = CalendarSource(country="PL", holidays=False, weekday=False, hour="onehot", freq="15min")
        df = source.fetch(*_local_day("2024-06-12"))
        assert list(df.columns) == [f"is_period_{p}" for p in range(96)]


class TestYamlRoundTrip:
    def test_holiday_parameters_round_trip(self, tmp_path):
        from epftoolbox2.pipelines import DataPipeline

        source = CalendarSource(
            country="DE", region="DE-BY", categories=["public", "catholic"], holidays="onehot", freq="15min"
        )
        path = tmp_path / "dp.yaml"
        DataPipeline(sources=[source]).save(path)
        loaded = DataPipeline.load(path).sources[0]

        assert loaded.region == "BY"
        assert loaded.categories == ["public", "catholic"]
        start, end = _local_day("2024-08-15", "Europe/Berlin")
        pd.testing.assert_frame_equal(source.fetch(start, end), loaded.fetch(start, end))

    def test_lat_lon_round_trip(self, tmp_path):
        from epftoolbox2.pipelines import DataPipeline

        source = CalendarSource(country="PL", daylight=True, lat=54.35, lon=18.65)
        path = tmp_path / "dp.yaml"
        DataPipeline(sources=[source]).save(path)
        loaded = DataPipeline.load(path).sources[0]

        start, end = _local_day("2024-06-21")
        pd.testing.assert_frame_equal(source.fetch(start, end), loaded.fetch(start, end))


class TestDaylightPerDate:
    def test_polar_night_warns(self):
        source = CalendarSource(country="NO", holidays=False, weekday=False, daylight=True, lat=78.2, lon=15.6)
        with patch.object(calendar_module.logger, "warning") as warning:
            df = source.fetch(*_local_day("2024-12-21", "Europe/Oslo"))
        assert df["daylight_hours"].isna().all()
        warning.assert_called_once()
        assert "sunrise" in warning.call_args.args[0]

    def test_daylight_values_unchanged_on_dst_day(self):
        # Per-date computation must give each row of a 25-hour day the value of its local date.
        source = CalendarSource(country="PL", holidays=False, weekday=False, daylight=True)
        df = source.fetch(*_local_day("2024-10-27"))
        assert len(df) == 25
        assert df["daylight_hours"].nunique() == 1
        assert df["sunrise"].iloc[0] == pytest.approx(6 + 26 / 60, abs=0.05)  # 06:26 CET in Warsaw
