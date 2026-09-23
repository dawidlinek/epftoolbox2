"""SchoolHolidaysSource with the OpenHolidays API mocked by recorded Polish responses (tests/fixtures/openholidays)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

import epftoolbox2.data.sources.school_holidays as school_module
from epftoolbox2.data.sources import SchoolHolidaysSource

FIXTURES = Path(__file__).parents[2] / "fixtures" / "openholidays"
PL_ENTRIES = json.loads((FIXTURES / "pl_school_holidays.json").read_text())  # 2019-12-23 .. 2027-02-28
PL_SUBDIVISIONS = json.loads((FIXTURES / "pl_subdivisions.json").read_text())
TZ = "Europe/Warsaw"


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


def _source(api=None, **kwargs):
    source = SchoolHolidaysSource(country="PL", **kwargs)
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
    def test_codes_are_normalised(self):
        source = SchoolHolidaysSource(country="pl", subdivisions="mz, PL-SL")
        assert source.country == "PL"
        assert source.subdivisions == ["PL-MZ", "PL-SL"]
        assert source.timezone == TZ

    def test_weights_dict_defines_subdivisions(self):
        source = SchoolHolidaysSource(country="PL", weights={"MZ": 5.5, "PL-SL": 4.4})
        assert source.subdivisions == ["PL-MZ", "PL-SL"]
        assert source.weights == {"PL-MZ": 5.5, "PL-SL": 4.4}

    def test_weights_list_follows_subdivisions(self):
        source = SchoolHolidaysSource(country="PL", subdivisions=["MZ", "SL"], weights="5.5, 4.4")
        assert source.weights == {"PL-MZ": 5.5, "PL-SL": 4.4}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"subdivisions": ["MZ"], "weights": {"SL": 1.0}},
            {"subdivisions": ["MZ", "SL"], "weights": [1.0]},
            {"weights": {"MZ": 0.0}},
            {"weights": {"MZ": "many"}},
            {"subdivisions": ["MZ", "mz"]},
            {"subdivisions": []},
            {"freq": "7min"},
            {"timezone": "Mars/Base"},
        ],
    )
    def test_invalid_config(self, kwargs):
        with pytest.raises(ValueError):
            SchoolHolidaysSource(country="PL", **kwargs)

    def test_country_without_default_timezone_needs_one(self):
        with pytest.raises(ValueError, match="timezone"):
            SchoolHolidaysSource(country="LU")
        assert SchoolHolidaysSource(country="LU", timezone="Europe/Luxembourg").timezone == "Europe/Luxembourg"

    def test_not_cached(self):
        assert SchoolHolidaysSource(country="PL").get_cache_config() is None


class TestFlags:
    def test_single_subdivision_winter_break(self):
        # Mazovia's 2024 winter break: 15-28 January.
        df = _source(subdivisions="MZ").fetch(*_range("2024-01-13", "2024-01-30"))
        assert list(df.columns) == ["is_school_holiday"]
        flags = _per_local_date(df, "is_school_holiday")
        assert list(flags.index[flags == 1]) == [f"2024-01-{d:02d}" for d in range(15, 29)]

    def test_flags_follow_local_dates_not_utc_dates(self):
        df = _source(subdivisions="MZ").fetch(*_range("2024-01-14", "2024-01-15"))
        flag = df["is_school_holiday"]
        assert flag[pd.Timestamp("2024-01-14 22:00", tz="UTC")] == 0  # 23:00 on the 14th in Warsaw
        assert flag[pd.Timestamp("2024-01-14 23:00", tz="UTC")] == 1  # 00:00 on the 15th

    def test_rows_and_range_follow_calendar_source(self):
        df = _source(subdivisions="MZ", freq="15min").fetch(*_range("2024-10-27", "2024-10-27"))
        assert len(df) == 100  # 25-hour day at 15 minutes, [start, end)
        assert str(df.index.tz) == "UTC"

    def test_country_wide_holidays_only_without_subdivisions(self):
        df = _source().fetch(*_range("2024-01-01", "2024-12-31"))
        flags = _per_local_date(df, "is_school_holiday")
        assert flags["2024-01-20"] == 0  # regional winter breaks are not country-wide
        assert flags["2024-03-28"] == 1  # spring break
        assert flags["2024-07-15"] == 1  # summer
        assert flags["2024-12-23"] == 1  # Christmas break

    def test_several_subdivisions_and_weighted_share(self):
        source = _source(weights={"MZ": 3.0, "SL": 1.0})
        df = source.fetch(*_range("2024-01-20", "2024-02-20"))
        assert list(df.columns) == ["is_school_holiday_pl_mz", "is_school_holiday_pl_sl", "school_holiday_share"]
        share = _per_local_date(df, "school_holiday_share")
        assert share["2024-01-20"] == pytest.approx(0.75)  # Mazovia only
        assert share["2024-02-01"] == pytest.approx(0.0)  # neither
        assert share["2024-02-15"] == pytest.approx(0.25)  # Silesia only (12-25 February)

    def test_chunks_long_ranges_within_the_api_limit(self):
        api = FakeApi()
        df = _source(api=api, subdivisions="MZ").fetch(*_range("2020-06-01", "2026-06-01"))
        holiday_calls = [params for url, params in api.calls if url.endswith("/SchoolHolidays")]
        assert len(holiday_calls) >= 3
        for params in holiday_calls:
            assert (pd.Timestamp(params["validTo"]) - pd.Timestamp(params["validFrom"])).days < 1095
        # The first chunk ends on 2022-01-21, inside Mazovia's winter break (14-27 January 2022).
        assert holiday_calls[0]["validTo"] == "2022-01-21"
        flags = _per_local_date(df, "is_school_holiday")
        assert flags["2022-01-14":"2022-01-27"].eq(1).all()
        assert flags["2022-01-28"] == 0

    def test_unknown_subdivision_raises(self):
        with pytest.raises(ValueError, match="Unknown subdivisions"):
            _source(subdivisions="XX").fetch(*_range("2024-01-01", "2024-01-02"))

    def test_children_inherit_holidays_of_their_parent(self):
        entries = [
            {"id": "a", "startDate": "2024-02-01", "endDate": "2024-02-02", "nationwide": False,
             "regionalScope": "Regional", "temporalScope": "FullDay", "subdivisions": [{"code": "DE-BY"}]},
            {"id": "b", "startDate": "2024-02-05", "endDate": "2024-02-05", "nationwide": False,
             "regionalScope": "Local", "temporalScope": "FullDay", "subdivisions": [{"code": "DE-BY-AU"}]},
        ]  # fmt: skip
        subdivisions = [{"code": "DE-BY", "children": [{"code": "DE-BY-AU"}]}]
        tz = "Europe/Berlin"
        start = pd.Timestamp("2024-02-01", tz=tz).tz_convert("UTC")
        end = pd.Timestamp("2024-02-06", tz=tz).tz_convert("UTC")

        def flagged(code):
            source = SchoolHolidaysSource(country="DE", subdivisions=code)
            source.session = FakeApi(entries=entries, subdivisions=subdivisions)
            with patch.object(school_module, "MAX_GAP_DAYS", 10_000):  # tiny fixture: no coverage check
                local = source.fetch(start, end)["is_school_holiday"].tz_convert(tz)
            return sorted({ts.date().isoformat() for ts in local.index[local == 1]})

        assert flagged("BY") == ["2024-02-01", "2024-02-02"]
        assert flagged("BY-AU") == ["2024-02-01", "2024-02-02", "2024-02-05"]


class TestCoverage:
    def test_dates_before_the_data_starts_are_nan(self):
        # The recorded Polish data starts with the 2019/20 Christmas break (23 December 2019).
        with patch.object(school_module.SchoolHolidaysSource, "_warn_missing") as warn:
            df = _source(subdivisions="MZ").fetch(*_range("2019-11-01", "2020-01-31"))
        flags = _per_local_date(df, "is_school_holiday")
        assert flags[:"2019-12-22"].isna().all()
        assert flags["2019-12-23"] == 1
        assert flags["2020-01-05"] == 0
        warn.assert_called_once()

    def test_school_years_not_yet_published_are_nan(self):
        with patch.object(school_module.SchoolHolidaysSource, "_warn_missing"):
            df = _source(subdivisions="MZ").fetch(*_range("2027-02-01", "2027-06-30"))
        flags = _per_local_date(df, "is_school_holiday")
        # Mazovia's last recorded break is its 2027 winter break (1-14 February); nothing later is published.
        assert flags["2027-02-02"] == 1
        assert flags["2027-02-14"] == 1
        assert flags["2027-02-15":].isna().all()

    def test_ordinary_gaps_between_holidays_are_zero(self):
        df = _source(subdivisions="MZ").fetch(*_range("2023-09-01", "2023-12-22"))  # no holidays, data exists
        assert df["is_school_holiday"].eq(0).all()

    def test_nan_flag_makes_the_share_nan(self):
        with patch.object(school_module.SchoolHolidaysSource, "_warn_missing"):
            df = _source(weights={"MZ": 1.0, "SL": 1.0}).fetch(*_range("2019-12-01", "2019-12-31"))
        share = _per_local_date(df, "school_holiday_share")
        assert share[:"2019-12-22"].isna().all()
        assert share["2019-12-23"] == 1

    def test_warning_names_the_missing_dates(self):
        source = _source(subdivisions="MZ")
        with patch.object(source.logger, "warning") as warning:
            source.fetch(*_range("2019-11-01", "2020-01-31"))
        message = warning.call_args.args[0]
        assert "2019-11-01" in message and "2019-12-22" in message


class TestHttp:
    def test_retries_server_errors(self, monkeypatch):
        monkeypatch.setattr(school_module.time, "sleep", lambda s: None)
        api = FakeApi()
        responses = iter([_response({}, status=503)])

        def flaky_get(url, **kwargs):
            if url.endswith("/SchoolHolidays"):
                try:
                    return next(responses)
                except StopIteration:
                    pass
            return api.get(url, **kwargs)

        source = _source(subdivisions="MZ")
        source.session = MagicMock()
        source.session.get.side_effect = flaky_get
        df = source.fetch(*_range("2024-01-15", "2024-01-15"))
        assert df["is_school_holiday"].eq(1).all()

    def test_gives_up_after_three_attempts(self, monkeypatch):
        monkeypatch.setattr(school_module.time, "sleep", lambda s: None)
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
    def test_round_trip(self, tmp_path):
        from epftoolbox2.pipelines import DataPipeline

        source = SchoolHolidaysSource(country="PL", weights={"MZ": 5.5, "SL": 4.4}, prefix="sh_", freq="15min")
        path = tmp_path / "dp.yaml"
        DataPipeline(sources=[source]).save(path)
        loaded = DataPipeline.load(path).sources[0]

        assert isinstance(loaded, SchoolHolidaysSource)
        assert loaded.subdivisions == ["PL-MZ", "PL-SL"]
        assert loaded.weights == {"PL-MZ": 5.5, "PL-SL": 4.4}
        loaded.session = FakeApi()
        source.session = FakeApi()
        start, end = _range("2024-01-10", "2024-02-20")
        pd.testing.assert_frame_equal(source.fetch(start, end), loaded.fetch(start, end))


@pytest.mark.integration
def test_live_api_poland():
    source = SchoolHolidaysSource(country="PL", subdivisions="MZ")
    df = source.fetch(*_range("2024-01-10", "2024-01-31"))
    flags = _per_local_date(df, "is_school_holiday")
    assert flags["2024-01-15"] == 1
    assert flags["2024-01-12"] == 0
    assert not np.isnan(df["is_school_holiday"]).any()
