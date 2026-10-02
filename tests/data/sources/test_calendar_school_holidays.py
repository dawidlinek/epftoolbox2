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
from epftoolbox2.data.sources import CalendarSource

FIXTURES = Path(__file__).parents[2] / "fixtures" / "openholidays"
PL_ENTRIES = json.loads((FIXTURES / "pl_school_holidays.json").read_text())  # 2019-12-23 .. 2027-02-28
PL_SUBDIVISIONS = json.loads((FIXTURES / "pl_subdivisions.json").read_text())  # 16 top-level voivodeships
TZ = "Europe/Warsaw"
BARE = {"holidays": False, "weekday": False}


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

    @pytest.mark.parametrize("value", [True, "national", "all"])
    def test_school_holidays_disables_the_source_cache(self, value):
        """The API publishes school years over time, so a cached NaN would never be refetched."""
        assert CalendarSource(country="PL", school_holidays=value).get_cache_config() is None

    def test_the_source_is_cached_when_school_holidays_is_off(self):
        assert CalendarSource(country="PL").get_cache_config() is not None


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
        entries = [
            {"id": "a", "startDate": "2024-02-05", "endDate": "2024-02-05", "nationwide": False,
             "regionalScope": "Local", "temporalScope": "FullDay", "subdivisions": [{"code": "DE-BY-AU"}]},
        ]  # fmt: skip
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


class TestPipelineCache:
    """`DataPipeline.run(cache=True)` must not freeze a school year that the API had not published yet."""

    # The recorded entries, cut off before 2025: every 2026 date is then a gap of more than a year, so NaN.
    UNPUBLISHED = [e for e in PL_ENTRIES if pd.Timestamp(e["endDate"]) < pd.Timestamp("2025-01-01")]

    @pytest.fixture(autouse=True)
    def cold_cache(self, tmp_path, monkeypatch):
        import epftoolbox2.data.cache_manager as cache_manager_module

        monkeypatch.chdir(tmp_path)
        # Far in the future, so the whole range counts as settled and would be recorded in the cache.
        monkeypatch.setattr(cache_manager_module, "_now", lambda: pd.Timestamp("2100-01-01", tz="UTC"))

    @staticmethod
    def _run(source):
        from epftoolbox2.pipelines import DataPipeline

        return DataPipeline(timezone=TZ).add_source(source).run("2026-07-01", "2026-07-05", cache=True)

    @pytest.mark.parametrize("mode,column", [("national", "is_school_holiday"), ("all", "school_holiday_share")])
    def test_a_school_year_published_later_is_picked_up(self, mode, column):
        source = _source(FakeApi(entries=self.UNPUBLISHED), school_holidays=mode)
        first = self._run(source)
        assert first[column].isna().all()

        source.session = FakeApi()  # the API has published the 2026 school year since
        second = self._run(source)
        assert second[column].eq(1.0).all()  # summer holidays

    def test_nothing_is_written_to_the_source_cache(self):
        self._run(_source())
        sources_dir = Path(".cache") / "sources"
        assert not sources_dir.exists() or not any(sources_dir.iterdir())

    def test_without_school_holidays_the_second_run_is_served_from_the_cache(self):
        from epftoolbox2.pipelines import DataPipeline

        source = CalendarSource(country="PL", school_holidays=False, holidays="binary", weekday=False)
        calls = []
        fetch = source.fetch
        source.fetch = lambda start, end: calls.append((start, end)) or fetch(start, end)

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
