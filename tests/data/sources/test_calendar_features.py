"""Opt-in CalendarSource features: bridge days, Christmas period, holiday distance, DST days, regimes."""

import pandas as pd
import pytest

from epftoolbox2.data.sources.calendar import DEFAULT_REGIMES, CalendarSource

BARE = {"holidays": False, "weekday": False}


def _range(first_day: str, last_day: str, tz: str = "Europe/Warsaw") -> tuple:
    """UTC [start, end) covering local days first_day..last_day."""
    start = pd.Timestamp(first_day, tz=tz)
    end = pd.Timestamp(last_day, tz=tz) + pd.DateOffset(days=1)
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def _per_local_date(df: pd.DataFrame, column: str, tz: str = "Europe/Warsaw") -> pd.Series:
    """One value per local date; fails if a feature varies within a local day."""
    local = df[column].tz_convert(tz)
    grouped = local.groupby(local.index.date)
    assert (grouped.nunique() == 1).all(), f"{column} varies within a local day"
    return grouped.first().rename(index=lambda d: d.isoformat())


def _flagged(df: pd.DataFrame, column: str, tz: str = "Europe/Warsaw") -> list:
    per_date = _per_local_date(df, column, tz)
    return list(per_date.index[per_date == 1])


class TestBridgeDays:
    def test_poland_2024(self):
        df = CalendarSource(country="PL", bridge_days=True, **BARE).fetch(*_range("2024-01-01", "2024-12-31"))
        # Thu between two holidays, and Fridays after Thursday holidays.
        assert _flagged(df, "is_bridge_day") == ["2024-05-02", "2024-05-31", "2024-08-16", "2024-12-27"]

    def test_uses_the_configured_holiday_calendar(self):
        tz = "Europe/Berlin"
        national = CalendarSource(country="DE", bridge_days=True, **BARE).fetch(*_range("2024-05-27", "2024-06-02", tz))
        bavaria = CalendarSource(country="DE", subdiv="BY", bridge_days=True, **BARE).fetch(
            *_range("2024-05-27", "2024-06-02", tz)
        )
        assert _flagged(national, "is_bridge_day", tz) == []
        assert _flagged(bavaria, "is_bridge_day", tz) == ["2024-05-31"]  # Friday after Corpus Christi

    def test_neighbours_outside_the_range_are_used(self):
        tz = "Europe/Berlin"
        df = CalendarSource(country="DE", bridge_days=True, **BARE).fetch(*_range("2024-05-10", "2024-05-10", tz))
        assert df["is_bridge_day"].eq(1).all()  # Ascension Day is the day before, the weekend the day after

    def test_does_not_add_holiday_columns(self):
        df = CalendarSource(country="PL", bridge_days=True, **BARE).fetch(*_range("2024-05-01", "2024-05-03"))
        assert list(df.columns) == ["is_bridge_day"]


class TestChristmasPeriod:
    def test_24_december_to_1_january(self):
        df = CalendarSource(country="PL", christmas_period=True, **BARE).fetch(*_range("2024-12-22", "2025-01-03"))
        assert _flagged(df, "is_christmas_period") == [
            "2024-12-24",
            "2024-12-25",
            "2024-12-26",
            "2024-12-27",
            "2024-12-28",
            "2024-12-29",
            "2024-12-30",
            "2024-12-31",
            "2025-01-01",
        ]


class TestHolidayDistance:
    def test_days_to_and_since_holiday(self):
        source = CalendarSource(country="PL", holiday_distance=7, **BARE)
        df = source.fetch(*_range("2024-10-30", "2024-11-12"))
        to = _per_local_date(df, "days_to_holiday")
        since = _per_local_date(df, "days_since_holiday")
        assert to["2024-10-30"] == 2 and since["2024-10-30"] == 7  # previous holiday (15 Aug) is capped
        assert to["2024-11-01"] == 0 and since["2024-11-01"] == 0
        assert to["2024-11-05"] == 6 and since["2024-11-05"] == 4  # between 1 Nov and 11 Nov
        assert to["2024-11-12"] == 7 and since["2024-11-12"] == 1  # next holiday is 25 Dec

    def test_looks_across_the_year_boundary(self):
        df = CalendarSource(country="PL", holiday_distance=7, **BARE).fetch(*_range("2024-12-31", "2024-12-31"))
        assert df["days_to_holiday"].eq(1).all()
        assert df["days_since_holiday"].eq(5).all()

    @pytest.mark.parametrize("value", [True, -1, 400, 2.5])
    def test_invalid_values(self, value):
        with pytest.raises(ValueError, match="holiday_distance"):
            CalendarSource(country="PL", holiday_distance=value)


class TestDstSwitchDay:
    @pytest.mark.parametrize("freq, rows", [("1h", (23, 25)), ("15min", (92, 100))])
    def test_flags_23_and_25_hour_days(self, freq, rows):
        source = CalendarSource(country="PL", dst_switch=True, freq=freq, **BARE)
        spring = source.fetch(*_range("2024-03-30", "2024-04-01"))
        fall = source.fetch(*_range("2024-10-26", "2024-10-28"))
        assert _flagged(spring, "is_dst_switch_day") == ["2024-03-31"]
        assert _flagged(fall, "is_dst_switch_day") == ["2024-10-27"]
        assert spring["is_dst_switch_day"].sum() == rows[0]
        assert fall["is_dst_switch_day"].sum() == rows[1]

    def test_no_switch_in_a_zone_without_dst(self):
        df = CalendarSource(country="PL", timezone="UTC", dst_switch=True, **BARE).fetch(
            *_range("2024-03-30", "2024-04-01", "UTC")
        )
        assert df["is_dst_switch_day"].eq(0).all()


class TestRegimes:
    def test_default_set_for_poland(self):
        source = CalendarSource(country="PL", regimes=True, **BARE)
        df = source.fetch(*_range("2022-06-08", "2022-06-09"))
        assert list(df.columns) == ["regime_core_fb", "regime_sdac_15min_mtu"]
        per_date = _per_local_date(df, "regime_core_fb")
        assert per_date["2022-06-08"] == 0
        assert per_date["2022-06-09"] == 1  # first delivery day of Core flow-based market coupling
        assert df["regime_sdac_15min_mtu"].eq(0).all()

    def test_step_starts_at_local_midnight(self):
        df = CalendarSource(country="DE", regimes=True, **BARE).fetch(
            *_range("2025-09-30", "2025-10-01", "Europe/Berlin")
        )
        flag = df["regime_sdac_15min_mtu"]
        assert flag[pd.Timestamp("2025-09-30 21:00", tz="UTC")] == 0
        assert flag[pd.Timestamp("2025-09-30 22:00", tz="UTC")] == 1  # 2025-10-01 00:00 CEST

    @pytest.mark.parametrize(
        "country, expected",
        [
            ("DE", ["regime_de_at_split", "regime_core_fb", "regime_sdac_15min_mtu"]),
            ("SE", ["regime_nordic_fb", "regime_sdac_15min_mtu"]),
            ("GB", ["regime_brexit_decoupling"]),
            ("IE", ["regime_brexit_decoupling", "regime_sem_30min_mtu"]),
            ("CH", []),
        ],
    )
    def test_defaults_depend_on_country(self, country, expected):
        df = CalendarSource(country=country, regimes=True, **BARE).fetch(*_range("2024-01-01", "2024-01-01"))
        assert list(df.columns) == expected

    def test_defaults_are_documented_dates(self):
        assert DEFAULT_REGIMES["core_fb"]["date"] == "2022-06-09"
        assert DEFAULT_REGIMES["sdac_15min_mtu"]["date"] == "2025-10-01"

    def test_user_regimes_replace_the_defaults(self):
        source = CalendarSource(country="PL", regimes={"new_tariff": "2024-01-02"}, **BARE)
        df = source.fetch(*_range("2024-01-01", "2024-01-02"))
        assert list(df.columns) == ["regime_new_tariff"]
        per_date = _per_local_date(df, "regime_new_tariff")
        assert list(per_date) == [0, 1]

    def test_user_regime_dates_are_normalised(self):
        import datetime

        source = CalendarSource(country="PL", regimes={"x": datetime.date(2024, 1, 2)})
        assert source.regimes == {"x": "2024-01-02"}

    @pytest.mark.parametrize("regimes", [{"bad name": "2024-01-01"}, {"x": "not a date"}, "core_fb", {}])
    def test_invalid_regimes(self, regimes):
        with pytest.raises(ValueError, match="regimes"):
            CalendarSource(country="PL", regimes=regimes)

    def test_string_date_is_open_ended(self):
        source = CalendarSource(country="PL", regimes={"promo": "2024-01-02"}, **BARE)
        df = source.fetch(*_range("2024-01-01", "2024-01-05"))
        assert list(_per_local_date(df, "regime_promo")) == [0, 1, 1, 1, 1]

    def test_tuple_period_is_bounded_and_end_inclusive(self):
        source = CalendarSource(country="PL", regimes={"promo": ("2024-01-02", "2024-01-03")}, **BARE)
        df = source.fetch(*_range("2024-01-01", "2024-01-04"))
        assert list(_per_local_date(df, "regime_promo")) == [0, 1, 1, 0]
        assert source.regimes == {"promo": {"start": "2024-01-02", "end": "2024-01-03"}}

    def test_dict_period_is_bounded_and_end_inclusive(self):
        source = CalendarSource(country="PL", regimes={"promo": {"start": "2024-01-02", "end": "2024-01-03"}}, **BARE)
        df = source.fetch(*_range("2024-01-01", "2024-01-04"))
        assert list(_per_local_date(df, "regime_promo")) == [0, 1, 1, 0]

    def test_dict_period_without_end_is_open_ended(self):
        source = CalendarSource(country="PL", regimes={"promo": {"start": "2024-01-02"}}, **BARE)
        df = source.fetch(*_range("2024-01-01", "2024-01-05"))
        assert list(_per_local_date(df, "regime_promo")) == [0, 1, 1, 1, 1]
        assert source.regimes == {"promo": "2024-01-02"}

    def test_bounded_period_end_inclusive_across_a_dst_switch(self):
        # Autumn switch is 2024-10-27 in Europe/Warsaw; the period ends on the switch day itself.
        source = CalendarSource(country="PL", regimes={"x": ("2024-10-25", "2024-10-27")}, **BARE)
        df = source.fetch(*_range("2024-10-24", "2024-10-28"))
        assert list(_per_local_date(df, "regime_x")) == [0, 1, 1, 1, 0]

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError, match="before its start"):
            CalendarSource(country="PL", regimes={"x": ("2024-01-05", "2024-01-01")})

    @pytest.mark.parametrize(
        "spec",
        [
            ("2024-01-01", "2024-01-02", "2024-01-03"),  # not a pair
            {"start": "2024-01-01", "middle": "2024-01-02"},  # unknown key
            {"end": "2024-01-02"},  # missing start
        ],
    )
    def test_malformed_period_raises(self, spec):
        with pytest.raises(ValueError, match="regimes"):
            CalendarSource(country="PL", regimes={"x": spec})

    def test_round_trip_open_and_bounded_regimes(self, tmp_path):
        from epftoolbox2.pipelines import DataPipeline

        source = CalendarSource(
            country="PL", regimes={"promo": "2024-01-02", "sale": ("2024-06-01", "2024-06-30")}, **BARE
        )
        path = tmp_path / "dp.yaml"
        DataPipeline(sources=[source]).save(path)
        loaded = DataPipeline.load(path).sources[0]

        assert loaded.regimes == {"promo": "2024-01-02", "sale": {"start": "2024-06-01", "end": "2024-06-30"}}
        start, end = _range("2024-01-01", "2024-07-02")
        pd.testing.assert_frame_equal(source.fetch(start, end), loaded.fetch(start, end))


class TestFeaturesYamlRoundTrip:
    def test_round_trip(self, tmp_path):
        from epftoolbox2.pipelines import DataPipeline

        source = CalendarSource(
            country="DE",
            subdiv="BY",
            bridge_days=True,
            christmas_period=True,
            holiday_distance=5,
            dst_switch=True,
            regimes={"core_fb": "2022-06-09", "custom": "2023-01-01"},
            freq="15min",
        )
        path = tmp_path / "dp.yaml"
        DataPipeline(sources=[source]).save(path)
        loaded = DataPipeline.load(path).sources[0]

        start, end = _range("2022-12-30", "2023-01-02", "Europe/Berlin")
        pd.testing.assert_frame_equal(source.fetch(start, end), loaded.fetch(start, end))

    def test_round_trip_default_regimes(self, tmp_path):
        from epftoolbox2.pipelines import DataPipeline

        path = tmp_path / "dp.yaml"
        DataPipeline(sources=[CalendarSource(country="PL", regimes=True)]).save(path)
        assert DataPipeline.load(path).sources[0].regimes is True
