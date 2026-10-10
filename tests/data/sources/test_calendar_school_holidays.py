"""CalendarSource(school_holidays="national"/"all") with the OpenHolidays API mocked by recorded Polish
responses (tests/fixtures/openholidays)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

import epftoolbox2.data.sources._openholidays as openholidays_module
import epftoolbox2.data.sources.calendar as calendar_module
from epftoolbox2.data import cache_manager as cache_manager_module
from epftoolbox2.data.cache_manager import settled_until_of
from epftoolbox2.data.sources import CalendarSource
from epftoolbox2.pipelines import DataPipeline

FIXTURES = Path(__file__).parents[2] / "fixtures" / "openholidays"
PL_ENTRIES = json.loads((FIXTURES / "pl_school_holidays.json").read_text())  # 2019-12-23 .. 2027-02-28
PL_SUBDIVISIONS = json.loads((FIXTURES / "pl_subdivisions.json").read_text())  # 16 top-level voivodeships
TZ = "Europe/Warsaw"
BARE = {"holidays": False, "weekday": False}
# What the API had published before the 2026/27 school year: nothing after the 2026 summer break.
PL_ENTRIES_TO_SUMMER_2026 = [e for e in PL_ENTRIES if e["endDate"] <= "2026-08-31"]


def _response(payload, status=200):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload
    response.raise_for_status.side_effect = requests.HTTPError(f"{status}") if status >= 400 else None
    return response


class FakeApi:
    """Answers like the API: entries overlapping [validFrom, validTo], at most 1095 days per request."""

    def __init__(self, entries=PL_ENTRIES, subdivisions=PL_SUBDIVISIONS):
        self.entries = entries
        self.subdivisions = subdivisions
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if url.endswith("/Subdivisions"):
            return _response(self.subdivisions)
        valid_from, valid_to = pd.Timestamp(params["validFrom"]), pd.Timestamp(params["validTo"])
        if (valid_to - valid_from).days > 1095:
            return _response({"detail": "The maximum date range is 1095 days."}, status=400)
        overlapping = [
            e
            for e in self.entries
            if pd.Timestamp(e["startDate"]) <= valid_to and pd.Timestamp(e["endDate"]) >= valid_from
        ]
        return _response(overlapping)


def _source(api=None, country="PL", school_holidays="national", **kwargs):
    source = CalendarSource(country=country, school_holidays=school_holidays, **BARE, **kwargs)
    source.session = api or FakeApi()
    return source


def _range(first_day: str, last_day: str) -> tuple:
    start = pd.Timestamp(first_day, tz=TZ)
    end = pd.Timestamp(last_day, tz=TZ) + pd.DateOffset(days=1)
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def _per_local_date(df: pd.DataFrame, column: str) -> pd.Series:
    local = df[column].tz_convert(TZ)
    grouped = local.groupby(local.index.date)
    assert (grouped.nunique(dropna=False) == 1).all(), f"{column} varies within a local day"
    return grouped.first().rename(index=lambda d: d.isoformat())


class TestInit:
    def test_true_is_an_alias_for_national(self):
        assert CalendarSource(country="PL", school_holidays=True).school_holidays == "national"

    @pytest.mark.parametrize("value", ["regional", "ALL", "nationwide", 1, ["national"]])
    def test_invalid_value_raises(self, value):
        with pytest.raises(ValueError, match="school_holidays"):
            CalendarSource(country="PL", school_holidays=value)

    def test_school_holidays_off_by_default_and_stays_offline(self):
        source = CalendarSource(country="PL")
        assert source.school_holidays is False
        # No HTTP session is touched: fetch() never calls the OpenHolidays API.
        source.session = MagicMock()
        source.fetch(*_range("2024-01-01", "2024-01-02"))
        source.session.get.assert_not_called()

    def test_school_holidays_and_its_mode_are_part_of_the_cache_key(self):
        national = CalendarSource(country="PL", school_holidays="national").get_cache_config()
        all_regions = CalendarSource(country="PL", school_holidays="all").get_cache_config()
        off = CalendarSource(country="PL").get_cache_config()
        assert national["school_holidays"] == "national"
        assert all_regions["school_holidays"] == "all"
        assert off["school_holidays"] is False
        assert len({str(national), str(all_regions), str(off)}) == 3


class TestNationalFlags:
    def test_country_wide_holidays_only(self):
        df = _source().fetch(*_range("2024-01-01", "2024-12-31"))
        assert list(df.columns) == ["is_school_holiday"]
        flags = _per_local_date(df, "is_school_holiday")
        assert flags["2024-01-20"] == 0  # regional winter break, not country-wide
        assert flags["2024-03-28"] == 1  # spring break
        assert flags["2024-07-15"] == 1  # summer
        assert flags["2024-12-23"] == 1  # Christmas break

    def test_true_behaves_as_national(self):
        national = _source(school_holidays="national").fetch(*_range("2024-01-01", "2024-12-31"))
        via_true = _source(school_holidays=True).fetch(*_range("2024-01-01", "2024-12-31"))
        pd.testing.assert_frame_equal(national, via_true)

    def test_flags_follow_local_dates_not_utc_dates(self):
        # Christmas break (country-wide) starts 23 December; Warsaw is UTC+1 in December.
        df = _source().fetch(*_range("2024-12-22", "2024-12-23"))
        flag = df["is_school_holiday"]
        assert flag[pd.Timestamp("2024-12-22 22:00", tz="UTC")] == 0  # 23:00 on the 22nd in Warsaw
        assert flag[pd.Timestamp("2024-12-22 23:00", tz="UTC")] == 1  # 00:00 on the 23rd

    def test_rows_and_range_follow_calendar_source(self):
        df = _source(freq="15min").fetch(*_range("2024-10-27", "2024-10-27"))
        assert len(df) == 100  # 25-hour day at 15 minutes, [start, end)
        assert str(df.index.tz) == "UTC"

    def test_prefix_applies_to_the_column(self):
        df = _source(prefix="cal_").fetch(*_range("2024-01-15", "2024-01-15"))
        assert list(df.columns) == ["cal_is_school_holiday"]


class TestAllShare:
    def test_staggered_winter_breaks_give_the_expected_share(self):
        # PL-like staggered regional winter breaks: four overlapping groups of Poland's 16 voivodeships.
        df = _source(school_holidays="all").fetch(*_range("2024-01-10", "2024-02-29"))
        assert list(df.columns) == ["school_holiday_share"]
        share = _per_local_date(df, "school_holiday_share")
        assert share["2024-01-14"] == pytest.approx(0.0)  # before any winter break
        assert share["2024-01-20"] == pytest.approx(0.25)  # 4 of 16 voivodeships (one group)
        assert share["2024-01-25"] == pytest.approx(0.375)  # two overlapping groups: 6 of 16
        assert share["2024-02-01"] == pytest.approx(0.4375)  # three overlapping groups: 7 of 16
        assert share["2024-02-20"] == pytest.approx(0.3125)  # last group only: 5 of 16
        assert share["2024-02-26"] == pytest.approx(0.0)  # winter break over everywhere

    def test_nationwide_period_gives_one_in_both_modes(self):
        national = _source(school_holidays="national").fetch(*_range("2024-07-15", "2024-07-15"))
        share = _source(school_holidays="all").fetch(*_range("2024-07-15", "2024-07-15"))
        assert national["is_school_holiday"].eq(1.0).all()
        assert share["school_holiday_share"].eq(1.0).all()

    def test_flags_follow_local_dates_not_utc_dates(self):
        df = _source(school_holidays="all").fetch(*_range("2024-12-22", "2024-12-23"))
        share = df["school_holiday_share"]
        assert share[pd.Timestamp("2024-12-22 22:00", tz="UTC")] == 0
        assert share[pd.Timestamp("2024-12-22 23:00", tz="UTC")] == 1

    def test_prefix_applies_to_the_column(self):
        df = _source(school_holidays="all", prefix="cal_").fetch(*_range("2024-01-15", "2024-01-15"))
        assert list(df.columns) == ["cal_school_holiday_share"]

    def test_fetches_the_region_list_once(self):
        api = FakeApi()
        _source(api=api, school_holidays="all").fetch(*_range("2024-01-15", "2024-01-15"))
        subdivision_calls = [url for url, _ in api.calls if url.endswith("/Subdivisions")]
        assert len(subdivision_calls) == 1

    def test_raises_if_the_country_has_no_subdivisions(self):
        api = FakeApi(subdivisions=[])
        with pytest.raises(ValueError, match="subdivisions"):
            _source(api=api, school_holidays="all").fetch(*_range("2024-01-15", "2024-01-15"))

    def test_nested_children_are_not_rolled_up_into_the_parent(self):
        # A holiday recorded only against a child code (BY-AU) must not count toward its top-level parent
        # (BY) in "all" mode: `top_level_subdivisions` only lists BY, and `daily_flags(..., "BY", ...)` only
        # matches entries naming BY itself (or nationwide ones) - a child-only entry doesn't cascade upward.
        def entry(first, last, code):
            return {"id": first, "startDate": first, "endDate": last, "nationwide": False, "regionalScope": "Local",
                    "temporalScope": "FullDay", "subdivisions": [{"code": code}]}  # fmt: skip

        # BY itself has periods on either side of 5 February, so that day is inside its published history.
        entries = [
            entry("2024-01-08", "2024-01-12", "DE-BY"),
            entry("2024-02-05", "2024-02-05", "DE-BY-AU"),
            entry("2024-03-04", "2024-03-08", "DE-BY"),
        ]
        subdivisions = [{"code": "DE-BY", "children": [{"code": "DE-BY-AU"}]}]
        source = _source(country="DE", school_holidays="all", api=FakeApi(entries=entries, subdivisions=subdivisions))
        with patch.object(openholidays_module, "MAX_GAP_DAYS", 10_000):  # tiny fixture: no coverage check
            df = source.fetch(*_range("2024-02-05", "2024-02-05"))
        assert df["school_holiday_share"].eq(0.0).all()


class TestCoverage:
    def test_dates_before_the_data_starts_are_nan(self):
        # The recorded Polish data starts with the 2019/20 Christmas break (23 December 2019), a country-wide
        # entry, so both modes see the same boundary.
        with patch.object(CalendarSource, "_warn_missing_school_holidays") as warn:
            df = _source().fetch(*_range("2019-11-01", "2020-01-31"))
        flags = _per_local_date(df, "is_school_holiday")
        assert flags[:"2019-12-22"].isna().all()
        assert flags["2019-12-23"] == 1
        assert flags["2020-01-05"] == 0
        warn.assert_called_once()

    def test_all_mode_before_the_data_starts_is_nan(self):
        with patch.object(CalendarSource, "_warn_missing_school_holidays"):
            df = _source(school_holidays="all").fetch(*_range("2019-12-01", "2019-12-31"))
        share = _per_local_date(df, "school_holiday_share")
        assert share[:"2019-12-22"].isna().all()
        assert share["2019-12-23"] == 1

    def test_school_years_not_yet_published_are_nan(self):
        # No country-wide entry is recorded after the 2026 summer break (ends 2026-08-31); the gap beyond
        # it is longer than a year, so it's treated as not-yet-published rather than "no holiday".
        with patch.object(CalendarSource, "_warn_missing_school_holidays"):
            df = _source().fetch(*_range("2026-09-01", "2026-09-05"))
        assert _per_local_date(df, "is_school_holiday").isna().all()

    def test_ordinary_gaps_between_holidays_are_zero(self):
        df = _source().fetch(*_range("2023-09-01", "2023-12-22"))  # no country-wide holidays, data exists
        assert df["is_school_holiday"].eq(0).all()

    def test_warning_names_the_missing_dates(self):
        source = _source()
        with patch.object(calendar_module.logger, "warning") as warning:
            source.fetch(*_range("2019-11-01", "2020-01-31"))
        message = warning.call_args.args[0]
        assert "2019-11-01" in message and "2019-12-22" in message


class TestUnknownIsNaN:
    """A date the API has not published yet is NaN, never 0.0 ("not a holiday")."""

    def test_days_after_the_last_period_are_nan_even_when_the_range_ends_within_a_year(self):
        # No 400-day margin here: the next period (the end of the range) is only 213 days after the summer break.
        days = pd.date_range("2026-08-01", "2027-03-31")
        flags = openholidays_module.daily_flags(PL_ENTRIES_TO_SUMMER_2026, None, days)
        assert (flags[days <= pd.Timestamp("2026-08-31")] == 1.0).all()
        assert np.isnan(flags[days > pd.Timestamp("2026-08-31")]).all()

    def test_a_gap_inside_the_history_is_still_zero(self):
        entries = [
            {"id": "a", "startDate": "2026-01-05", "endDate": "2026-01-09", "nationwide": True},
            {"id": "b", "startDate": "2026-03-02", "endDate": "2026-03-06", "nationwide": True},
        ]
        days = pd.date_range("2026-01-01", "2026-04-30")
        flags = pd.Series(openholidays_module.daily_flags(entries, None, days), index=days)
        assert flags["2026-01-12":"2026-02-27"].eq(0.0).all()  # between two periods: known to be no holiday
        assert flags["2026-03-09":].isna().all()  # after the last one: not published
        assert flags["2026-03-02":"2026-03-06"].eq(1.0).all()

    def test_no_period_at_all_means_every_day_is_unknown(self):
        flags = openholidays_module.daily_flags([], None, pd.date_range("2026-01-01", "2026-01-10"))
        assert np.isnan(flags).all()

    def test_a_half_day_entry_does_not_extend_what_is_published(self):
        half_day = {"id": "h", "startDate": "2026-12-23", "endDate": "2026-12-23", "nationwide": True,
                    "temporalScope": "HalfDay"}  # fmt: skip
        days = pd.date_range("2026-08-01", "2027-01-31")
        flags = openholidays_module.daily_flags(PL_ENTRIES_TO_SUMMER_2026 + [half_day], None, days)
        assert np.isnan(flags[days > pd.Timestamp("2026-08-31")]).all()

    def test_last_published_day_is_per_target(self):
        assert openholidays_module.last_published_day(PL_ENTRIES, None) == pd.Timestamp("2026-08-31")
        assert openholidays_module.last_published_day(PL_ENTRIES, "PL-MA") == pd.Timestamp("2027-02-28")
        assert openholidays_module.last_published_day(PL_ENTRIES, "PL-LB") == pd.Timestamp("2026-08-31")
        assert openholidays_module.last_published_day([], None) is None

    @pytest.mark.parametrize("mode,column", [("national", "is_school_holiday"), ("all", "school_holiday_share")])
    def test_fetch_is_nan_after_the_last_published_period(self, mode, column):
        source = _source(FakeApi(entries=PL_ENTRIES_TO_SUMMER_2026), school_holidays=mode)
        with patch.object(CalendarSource, "_warn_missing_school_holidays"):
            df = source.fetch(*_range("2026-08-25", "2027-03-31"))
        values = _per_local_date(df, column)
        assert values[:"2026-08-31"].eq(1.0).all()
        assert values["2026-09-01":].isna().all()

    def test_all_mode_is_nan_while_any_region_has_not_published(self):
        # The recorded data has winter 2027 breaks for 15 of the 16 voivodeships, but Lubuskie (PL-LB) stops at
        # the summer break: the share needs every region, so it is unknown from there on.
        source = _source(school_holidays="all")
        with patch.object(CalendarSource, "_warn_missing_school_holidays"):
            df = source.fetch(*_range("2027-01-20", "2027-02-10"))
        assert _per_local_date(df, "school_holiday_share").isna().all()


class TestChunking:
    def test_chunks_long_ranges_within_the_api_limit(self):
        api = FakeApi()
        df = _source(api=api, school_holidays="all").fetch(*_range("2020-06-01", "2026-06-01"))
        holiday_calls = [params for url, params in api.calls if url.endswith("/SchoolHolidays")]
        assert len(holiday_calls) >= 3
        for params in holiday_calls:
            assert (pd.Timestamp(params["validTo"]) - pd.Timestamp(params["validFrom"])).days < 1095
        # The first chunk ends on 2022-01-21, inside a regional winter-break entry (14-27 January 2022) that
        # itself spans the chunk boundary: its contribution to the share must survive the split, not be lost
        # or double-counted (`fetch_entries` de-duplicates by id across overlapping chunks).
        assert holiday_calls[0]["validTo"] == "2022-01-21"
        share = _per_local_date(df, "school_holiday_share")
        assert share["2022-01-13"] == pytest.approx(0.0)
        assert share["2022-01-21"] == pytest.approx(0.5625)


class TestHttp:
    def test_retries_server_errors(self, monkeypatch):
        monkeypatch.setattr(openholidays_module.time, "sleep", lambda s: None)
        api = FakeApi()
        responses = iter([_response({}, status=503)])

        def flaky_get(url, **kwargs):
            if url.endswith("/SchoolHolidays"):
                try:
                    return next(responses)
                except StopIteration:
                    pass
            return api.get(url, **kwargs)

        source = _source()
        source.session = MagicMock()
        source.session.get.side_effect = flaky_get
        df = source.fetch(*_range("2024-07-15", "2024-07-15"))
        assert df["is_school_holiday"].eq(1).all()

    def test_gives_up_after_three_attempts(self, monkeypatch):
        monkeypatch.setattr(openholidays_module.time, "sleep", lambda s: None)
        source = _source()
        source.session = MagicMock()
        source.session.get.side_effect = requests.ConnectionError("offline")
        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            source.fetch(*_range("2024-01-15", "2024-01-15"))
        assert source.session.get.call_count == 3

    def test_client_errors_raise_immediately(self):
        source = _source()
        source.session = MagicMock()
        source.session.get.return_value = _response({"detail": "bad"}, status=400)
        with pytest.raises(requests.HTTPError):
            source.fetch(*_range("2024-01-15", "2024-01-15"))
        assert source.session.get.call_count == 1


class TestYamlRoundTrip:
    def test_round_trip_national(self, tmp_path):
        from epftoolbox2.pipelines import DataPipeline

        source = CalendarSource(country="PL", school_holidays=True, prefix="sh_", freq="15min", **BARE)
        path = tmp_path / "dp.yaml"
        DataPipeline(sources=[source]).save(path)
        loaded = DataPipeline.load(path).sources[0]

        assert isinstance(loaded, CalendarSource)
        assert loaded.school_holidays == "national"  # True normalises to "national" before it's ever saved
        loaded.session = FakeApi()
        source.session = FakeApi()
        start, end = _range("2024-01-10", "2024-02-20")
        pd.testing.assert_frame_equal(source.fetch(start, end), loaded.fetch(start, end))

    def test_round_trip_all(self, tmp_path):
        from epftoolbox2.pipelines import DataPipeline

        source = CalendarSource(country="PL", school_holidays="all", prefix="sh_", freq="15min", **BARE)
        path = tmp_path / "dp.yaml"
        DataPipeline(sources=[source]).save(path)
        loaded = DataPipeline.load(path).sources[0]

        assert loaded.school_holidays == "all"
        loaded.session = FakeApi()
        source.session = FakeApi()
        start, end = _range("2024-01-10", "2024-02-20")
        pd.testing.assert_frame_equal(source.fetch(start, end), loaded.fetch(start, end))


def _entry(first: str, last: str, regions=None, name="Winter Holidays") -> dict:
    """A school holiday entry in the API's shape: for the given voivodeships, or nationwide (all 16) without."""
    codes = regions or [s["code"] for s in PL_SUBDIVISIONS]
    return {
        "id": f"{name}-{first}-{regions}", "startDate": first, "endDate": last, "type": "School",
        "name": [{"language": "EN", "text": name}], "nationwide": False,
        "regionalScope": "Regional" if regions else "National", "temporalScope": "FullDay",
        "subdivisions": [{"code": code} for code in codes],
    }  # fmt: skip


# The 2026/27 school year as the API publishes it later: the recorded winter breaks of 15 voivodeships plus the
# missing one (Lubuskie), a nationwide Christmas break and a nationwide spring break reaching 2027-03-31.
PL_ENTRIES_2026_27 = PL_ENTRIES + [
    _entry("2027-02-01", "2027-02-14", ["PL-LB"]),
    _entry("2026-12-23", "2026-12-31", name="Christmas Break"),
    _entry("2027-03-26", "2027-03-31", name="Spring Break"),
]


class TestSettledUntil:
    """The first instant (UTC) whose school holiday value the API has not published yet, as the cache needs it."""

    @staticmethod
    def _declared(source, first_day, last_day):
        return settled_until_of(source.fetch(*_range(first_day, last_day)))

    @pytest.mark.parametrize("mode", ["national", "all"])
    def test_is_local_midnight_after_the_end_of_the_published_data(self, mode):
        source = _source(FakeApi(entries=PL_ENTRIES_TO_SUMMER_2026), school_holidays=mode)
        with patch.object(CalendarSource, "_warn_missing_school_holidays"):
            declared = self._declared(source, "2026-06-01", "2027-03-31")
        assert declared == pd.Timestamp("2026-09-01 00:00", tz=TZ).tz_convert("UTC")
        assert declared == pd.Timestamp("2026-08-31 22:00", tz="UTC")

    def test_follows_the_zone_offset_of_the_day_after_the_last_published_one(self):
        # Published until 27 March 2027, the day before Warsaw moves to summer time: that midnight is still UTC+1.
        entries = PL_ENTRIES_TO_SUMMER_2026 + [_entry("2027-03-20", "2027-03-27", name="Spring Break")]
        source = _source(FakeApi(entries=entries), school_holidays="national")
        with patch.object(CalendarSource, "_warn_missing_school_holidays"):
            declared = self._declared(source, "2027-03-01", "2027-04-10")
        assert declared == pd.Timestamp("2027-03-27 23:00", tz="UTC")

    def test_all_mode_waits_for_the_region_whose_data_stops_first(self):
        # Lubuskie stops at the 2026 summer break while the other 15 voivodeships go on to the winter of 2027.
        source = _source(school_holidays="all")
        with patch.object(CalendarSource, "_warn_missing_school_holidays"):
            declared = self._declared(source, "2026-06-01", "2027-03-31")
        assert declared == pd.Timestamp("2026-08-31 22:00", tz="UTC")

    def test_all_mode_is_settled_up_to_the_end_of_the_nationwide_spring_break(self):
        source = _source(FakeApi(entries=PL_ENTRIES_2026_27), school_holidays="all")
        with patch.object(CalendarSource, "_warn_missing_school_holidays"):
            declared = self._declared(source, "2026-06-01", "2027-06-30")
        assert declared == pd.Timestamp("2027-04-01 00:00", tz=TZ).tz_convert("UTC")  # the spring break ends 31 March

    def test_is_not_declared_when_the_request_ends_inside_the_published_data(self):
        source = _source(FakeApi(entries=PL_ENTRIES_TO_SUMMER_2026))
        assert self._declared(source, "2026-06-01", "2026-08-31") is None
        assert "settled_until" not in source.fetch(*_range("2024-01-01", "2024-02-01")).attrs

    def test_is_not_declared_when_the_end_is_the_first_unknown_instant(self):
        source = _source(FakeApi(entries=PL_ENTRIES_TO_SUMMER_2026))
        assert self._declared(source, "2026-08-01", "2026-08-31") is None  # [.., 2026-09-01 00:00 Warsaw)

    def test_nothing_is_settled_when_the_api_has_no_period_for_the_target(self):
        source = _source(FakeApi(entries=[]), school_holidays="national")
        start, end = _range("2024-01-01", "2024-01-05")
        with patch.object(CalendarSource, "_warn_missing_school_holidays"):
            assert settled_until_of(source.fetch(start, end)) == start

    def test_the_frame_can_still_be_saved_as_parquet(self, tmp_path):
        # pandas writes `attrs` as JSON: a Timestamp in there would make this raise.
        source = _source(FakeApi(entries=PL_ENTRIES_TO_SUMMER_2026))
        df = source.fetch(*_range("2026-06-01", "2027-03-31"))
        df.to_parquet(tmp_path / "calendar.parquet")
        assert pd.read_parquet(tmp_path / "calendar.parquet").attrs == df.attrs

    def test_off_declares_nothing(self):
        source = CalendarSource(country="PL", **BARE)
        assert "settled_until" not in source.fetch(*_range("2026-09-01", "2026-09-05")).attrs


class TestPipelineCache:
    """`DataPipeline.run(cache=True)` records the past school holidays, and never a date the API has not published."""

    FIRST_UNKNOWN = pd.Timestamp("2026-08-31 22:00", tz="UTC")  # 2026-09-01 00:00 in Warsaw
    END = pd.Timestamp("2027-04-01 00:00", tz=TZ).tz_convert("UTC")  # run(.., "2027-03-31") is [.., this)

    @pytest.fixture(autouse=True)
    def cold_cache(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self.set_now = lambda value: monkeypatch.setattr(
            cache_manager_module, "_now", lambda: pd.Timestamp(value, tz="UTC")
        )
        self.set_now("2026-10-10 12:00")  # day 1

    @staticmethod
    def _spied(source):
        calls = []
        fetch = source.fetch
        source.fetch = lambda start, end: calls.append((start, end)) or fetch(start, end)
        return calls

    @staticmethod
    def _run(source, start="2026-06-01", end="2027-03-31", cache=True):
        return DataPipeline(timezone=TZ).add_source(source).run(start, end, cache=cache)

    @staticmethod
    def _recorded():
        ranges = []
        for path in (Path(".cache") / "sources").glob("*/metadata.json"):
            ranges += [
                (pd.Timestamp(r["start"]), pd.Timestamp(r["end"]))
                for r in json.loads(path.read_text())["cached_ranges"]
            ]
        return sorted(ranges)

    @pytest.mark.parametrize("mode,column", [("national", "is_school_holiday"), ("all", "school_holiday_share")])
    def test_a_school_year_published_later_is_fetched_and_the_past_is_not(self, mode, column):
        source = _source(FakeApi(entries=PL_ENTRIES_TO_SUMMER_2026), school_holidays=mode)
        calls = self._spied(source)

        # Day 1: the API knows up to the 2026 summer break, the request runs to March 2027.
        first = self._run(source)
        published = first[column][first.index < self.FIRST_UNKNOWN]
        assert published.notna().all() and published.eq(1.0).any()
        assert first[column][first.index >= self.FIRST_UNKNOWN].isna().all()
        assert len(calls) == 1
        start = pd.Timestamp("2026-06-01", tz=TZ).tz_convert("UTC")
        assert self._recorded() == [(start, self.FIRST_UNKNOWN)]  # the NaN rows are not recorded

        # Day 2: the 2026/27 school year has been published. Only the dates that were unknown are asked for.
        self.set_now("2026-10-11 12:00")
        source.session = FakeApi(entries=PL_ENTRIES_2026_27)
        calls.clear()
        second = self._run(source)
        assert calls == [(self.FIRST_UNKNOWN, self.END)]
        assert second[column].notna().all()
        assert second[column][second.index < self.FIRST_UNKNOWN].equals(first[column][first.index < self.FIRST_UNKNOWN])
        assert second[column].loc["2026-12-24":"2026-12-30"].eq(1.0).all()  # the Christmas break that was unknown

        fresh = self._run(source, cache=False)
        pd.testing.assert_frame_equal(second, fresh, check_freq=False)

    def test_past_dates_are_served_from_the_cache_without_a_request(self):
        source = _source(FakeApi(entries=PL_ENTRIES_2026_27), school_holidays="all")
        self.set_now("2026-10-11 12:00")
        self._run(source)  # day 1, published to 2027-03-31 for all voivodeships but Lubuskie
        source.session = MagicMock()
        source.session.get.side_effect = AssertionError("no request for dates that are cached")
        self._run(source, start="2026-06-01", end="2026-08-31")  # entirely before the first unknown date

    @pytest.mark.parametrize("mode,column", [("national", "is_school_holiday"), ("all", "school_holiday_share")])
    def test_split_runs_equal_a_fresh_fetch_whatever_the_api_knew_when(self, mode, column):
        source = _source(FakeApi(entries=PL_ENTRIES_TO_SUMMER_2026), school_holidays=mode)
        self._run(source, end="2026-09-20")
        self.set_now("2026-10-11 12:00")
        source.session = FakeApi(entries=PL_ENTRIES_2026_27)
        self._run(source, end="2027-01-15")
        self.set_now("2026-10-13 12:00")
        cached = self._run(source)
        fresh = self._run(source, cache=False)
        pd.testing.assert_frame_equal(cached, fresh, check_freq=False)
        assert cached[column].notna().all()

    def test_a_school_year_the_api_still_has_not_published_is_asked_for_again_each_run(self):
        source = _source(FakeApi(entries=PL_ENTRIES_TO_SUMMER_2026))
        calls = self._spied(source)
        self._run(source)
        self.set_now("2026-10-11 12:00")
        again = self._run(source)
        assert calls == [(pd.Timestamp("2026-05-31 22:00", tz="UTC"), self.END), (self.FIRST_UNKNOWN, self.END)]
        assert again["is_school_holiday"][again.index >= self.FIRST_UNKNOWN].isna().all()

    def test_a_range_entirely_after_the_published_data_is_not_recorded(self):
        source = _source(FakeApi(entries=PL_ENTRIES_TO_SUMMER_2026))
        self._run(source, start="2026-09-05", end="2027-03-31")
        assert self._recorded() == []

    def test_the_cache_key_of_a_school_holiday_source_is_its_own(self):
        self._run(_source(FakeApi(entries=PL_ENTRIES_TO_SUMMER_2026), school_holidays="national"), end="2026-07-01")
        self._run(_source(FakeApi(entries=PL_ENTRIES_TO_SUMMER_2026), school_holidays="all"), end="2026-07-01")
        self._run(CalendarSource(country="PL", holidays="binary", weekday=False), end="2026-07-01")
        assert len(list((Path(".cache") / "sources").glob("*/metadata.json"))) == 3

    def test_single_file_cache_is_rebuilt_while_the_api_has_not_published_the_whole_range(self):
        source = _source(FakeApi(entries=PL_ENTRIES_TO_SUMMER_2026))
        self._run(source, cache="calendar/")
        source.session = FakeApi(entries=PL_ENTRIES_2026_27)
        second = self._run(source, cache="calendar/")
        assert second["is_school_holiday"].notna().all()  # the first run's NaN rows were not served again

        source.session = MagicMock()
        source.session.get.side_effect = AssertionError("fully published: served from the file")
        pd.testing.assert_frame_equal(self._run(source, cache="calendar/"), second, check_freq=False)

    def test_without_school_holidays_the_second_run_is_served_from_the_cache(self):
        source = CalendarSource(country="PL", school_holidays=False, holidays="binary", weekday=False)
        calls = self._spied(source)
        pipeline = DataPipeline(timezone=TZ).add_source(source)
        first = pipeline.run("2026-07-01", "2026-07-05", cache=True)
        second = pipeline.run("2026-07-01", "2026-07-05", cache=True)

        assert len(calls) == 1
        pd.testing.assert_frame_equal(first, second, check_freq=False)


@pytest.mark.integration
def test_live_api_poland():
    all_mode = CalendarSource(country="PL", school_holidays="all", **BARE)
    df = all_mode.fetch(*_range("2024-01-15", "2024-01-15"))
    share = df["school_holiday_share"].iloc[0]
    assert 0 < share <= 1  # a regional winter break is active somewhere in Poland on this date
    assert not np.isnan(df["school_holiday_share"]).any()

    national = CalendarSource(country="PL", school_holidays="national", **BARE)
    summer = national.fetch(*_range("2024-07-15", "2024-07-15"))
    assert summer["is_school_holiday"].eq(1.0).all()  # nationwide summer break
