import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from epftoolbox2.data.sources._open_meteo_common import (
    OpenMeteoBase,
    parse_hourly,
    parse_locations,
    parse_run,
    run_cutoff,
    shift_local_days,
)

TZ = "Europe/Warsaw"


class _StubSource(OpenMeteoBase):
    """Minimal concrete OpenMeteoBase, for testing the shared behaviour directly."""

    DEFAULT_COLUMNS = ["temperature_2m", "wind_speed_100m"]

    def __init__(self, **kwargs):
        params = dict(
            latitude=52.23,
            longitude=21.01,
            horizon=2,
            model="icon_seamless",
            columns=None,
            prefix="",
            proxy_url=None,
            timezone=TZ,
            api="previous_runs",
            run="00",
            strict=False,
            locations=None,
            keep_locations=False,
        )
        params.update(kwargs)
        self.latitude = params["latitude"]
        self.longitude = params["longitude"]
        self.horizon = params["horizon"]
        self.model = params["model"]
        self.columns = list(params["columns"]) if params["columns"] else list(self.DEFAULT_COLUMNS)
        self.prefix = params["prefix"]
        self.proxy_url = params["proxy_url"]
        self.timezone = params["timezone"]
        self.api = params["api"]
        self.run = params["run"]
        self.strict = params["strict"]
        self.locations = params["locations"]
        self.keep_locations = params["keep_locations"]
        self._setup()
        self._validate_config()

    def fetch(self, start, end):  # pragma: no cover - not exercised here
        raise NotImplementedError

    def get_cache_config(self):
        return {"source_type": "stub", **self._common_cache_config()}

    def _validate_config(self):
        self._validate_common()
        return True


# ---------------------------------------------------------------------------
# Cutoff helpers
# ---------------------------------------------------------------------------


class TestParseRun:
    def test_parses_valid_hours(self):
        assert parse_run("00") == 0
        assert parse_run("09") == 9
        assert parse_run("23") == 23

    @pytest.mark.parametrize("value", ["24", "1", "9", "abc", None, 9, "9:00", "-1"])
    def test_rejects_invalid_run(self, value):
        with pytest.raises(ValueError, match="run"):
            parse_run(value)


class TestRunCutoff:
    def test_resolves_local_day_at_run_hour_to_utc(self):
        # The row (2024-01-10 00:00 UTC = 01:00 CET) belongs to local day D = 2024-01-10; `run` is already a
        # UTC hour, so the cutoff is exactly D's date at that UTC hour, no zone conversion involved.
        rows = pd.DatetimeIndex([pd.Timestamp("2024-01-10 00:00", tz="UTC")])
        cutoff = run_cutoff(rows, TZ, "10")
        assert cutoff[0] == pd.Timestamp("2024-01-10 10:00", tz="UTC")

    def test_needs_no_dst_resolution_since_run_is_already_utc(self):
        # A row that falls in the local DST gap day still resolves cleanly: `run` is a UTC hour, so combining it
        # with D's calendar date never hits an ambiguous/nonexistent local instant.
        rows = pd.DatetimeIndex([pd.Timestamp("2024-03-31 10:00", tz=TZ).tz_convert("UTC")])
        cutoff = run_cutoff(rows, TZ, "00")
        assert cutoff[0] == pd.Timestamp("2024-03-31 00:00", tz="UTC")

    def test_late_run_can_land_on_the_next_utc_calendar_day(self):
        rows = pd.DatetimeIndex([pd.Timestamp("2024-01-10 00:00", tz="UTC")])
        cutoff = run_cutoff(rows, TZ, "23")
        assert cutoff[0] == pd.Timestamp("2024-01-10 23:00", tz="UTC")


class TestShiftLocalDays:
    def test_shifts_by_calendar_days_not_24h(self):
        rows = pd.DatetimeIndex([pd.Timestamp("2024-03-30 10:00", tz=TZ).tz_convert("UTC")])
        shifted = shift_local_days(rows, TZ, 1)
        # Spring-forward day: 23 local hours between 03-30 10:00 and 03-31 10:00, not 24.
        assert shifted[0] == pd.Timestamp("2024-03-31 10:00", tz=TZ)
        assert shifted[0] - rows[0] == pd.Timedelta(hours=23)

    def test_nonexistent_target_hour_is_nat_not_the_next_hour(self):
        # 2024-03-31 02:00 does not exist in Warsaw (spring forward): a row asking for that wall-clock hour
        # gets NaN, not silently the next hour's value under a different row's name.
        rows = pd.DatetimeIndex([pd.Timestamp("2024-03-30 02:00", tz=TZ).tz_convert("UTC")])
        shifted = shift_local_days(rows, TZ, 1)
        assert shifted.isna().all()

    def test_ambiguous_target_hour_takes_the_first_occurrence(self):
        # 2024-10-27 02:00 occurs twice in Warsaw (fall back); the first (CEST, UTC+2) occurrence is used.
        rows = pd.DatetimeIndex([pd.Timestamp("2024-10-26 02:00", tz=TZ).tz_convert("UTC")])
        shifted = shift_local_days(rows, TZ, 1)
        expected = pd.Timestamp("2024-10-27 02:00").tz_localize(TZ, ambiguous=True)
        assert shifted[0] == expected.tz_convert("UTC")
        assert shifted[0].tz_convert(TZ).utcoffset() == pd.Timedelta(hours=2)


# ---------------------------------------------------------------------------
# parse_locations (see OpenMeteoSource's `locations` parameter)
# ---------------------------------------------------------------------------

LOCATIONS_DICT = {
    "warsaw": {"latitude": 52.23, "longitude": 21.01, "weight": 3.0},
    "krakow": {"latitude": 50.06, "longitude": 19.94, "weight": 1.0},
}
LOCATIONS_NORMALISED = [
    {"name": "warsaw", "latitude": 52.23, "longitude": 21.01, "weight": 3.0},
    {"name": "krakow", "latitude": 50.06, "longitude": 19.94, "weight": 1.0},
]


class TestParseLocations:
    def test_parses_dict_of_dicts(self):
        assert parse_locations(LOCATIONS_DICT) == LOCATIONS_NORMALISED

    def test_parses_dict_of_tuples_with_weight(self):
        parsed = parse_locations({"warsaw": (52.23, 21.01, 3.0), "krakow": (50.06, 19.94, 1.0)})
        assert parsed == LOCATIONS_NORMALISED

    def test_parses_dict_of_tuples_without_weight(self):
        parsed = parse_locations({"warsaw": (52.23, 21.01), "krakow": (50.06, 19.94)})
        assert parsed == [
            {"name": "warsaw", "latitude": 52.23, "longitude": 21.01, "weight": None},
            {"name": "krakow", "latitude": 50.06, "longitude": 19.94, "weight": None},
        ]

    def test_parses_compact_string(self):
        parsed = parse_locations("warsaw:52.23:21.01:3.0, krakow:50.06:19.94:1.0")
        assert parsed == LOCATIONS_NORMALISED

    def test_parses_compact_string_without_weight(self):
        parsed = parse_locations("warsaw:52.23:21.01, krakow:50.06:19.94")
        assert [loc["weight"] for loc in parsed] == [None, None]

    def test_dict_missing_latitude_or_longitude_raises(self):
        with pytest.raises(ValueError, match="latitude, longitude and optionally weight"):
            parse_locations({"warsaw": {"latitude": 52.23}})

    def test_dict_with_unknown_key_raises(self):
        with pytest.raises(ValueError, match="latitude, longitude and optionally weight"):
            parse_locations({"warsaw": {"latitude": 52.23, "longitude": 21.01, "extra": 1}})

    def test_tuple_with_wrong_length_raises(self):
        with pytest.raises(ValueError, match=r"\(latitude, longitude\) or \(latitude, longitude, weight\)"):
            parse_locations({"warsaw": (52.23,)})

    def test_spec_of_wrong_type_raises(self):
        with pytest.raises(ValueError, match="dict or a"):
            parse_locations({"warsaw": "52.23,21.01"})

    def test_string_with_wrong_field_count_raises(self):
        with pytest.raises(ValueError, match="name:latitude:longitude"):
            parse_locations("warsaw:52.23")

    def test_string_with_non_numeric_field_raises(self):
        with pytest.raises(ValueError, match="with numbers"):
            parse_locations("warsaw:abc:21.01:3.0")

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="non-empty dict"):
            parse_locations({})
        with pytest.raises(ValueError, match="non-empty dict"):
            parse_locations("")

    def test_duplicate_name_in_compact_string_raises(self):
        with pytest.raises(ValueError, match="duplicate location name"):
            parse_locations("warsaw:52.23:21.01:3.0, warsaw:50.06:19.94:1.0")

    @pytest.mark.parametrize("name", ["has space", "has-dash", "", "accént"])
    def test_invalid_name_characters_raise(self, name):
        with pytest.raises(ValueError, match="letters, digits and underscores"):
            parse_locations({name: (52.23, 21.01)})

    @pytest.mark.parametrize("latitude", [91.0, -91.0])
    def test_invalid_latitude_raises(self, latitude):
        with pytest.raises(ValueError, match="latitude"):
            parse_locations({"warsaw": (latitude, 21.01)})

    @pytest.mark.parametrize("longitude", [181.0, -181.0])
    def test_invalid_longitude_raises(self, longitude):
        with pytest.raises(ValueError, match="longitude"):
            parse_locations({"warsaw": (52.23, longitude)})

    @pytest.mark.parametrize("weight", [0, -1.0, float("nan"), "1.0", True])
    def test_invalid_weight_raises(self, weight):
        with pytest.raises(ValueError, match="weight"):
            parse_locations({"warsaw": (52.23, 21.01, weight)})

    def test_mixed_weights_raise(self):
        with pytest.raises(ValueError, match="every location or for none"):
            parse_locations({"warsaw": (52.23, 21.01, 3.0), "krakow": (50.06, 19.94)})


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseHourly:
    def test_missing_hourly_returns_empty(self):
        assert parse_hourly({}).empty
        assert parse_hourly({"hourly": {}}).empty

    def test_nulls_become_nan(self):
        data = {"hourly": {"time": ["2024-01-01T00:00", "2024-01-01T01:00"], "temperature_2m": [1.0, None]}}
        table = parse_hourly(data)
        assert table["temperature_2m"].tolist()[0] == 1.0
        assert np.isnan(table["temperature_2m"].iloc[1])
        assert str(table.index.tz) == "UTC"


# ---------------------------------------------------------------------------
# OpenMeteoBase behaviour, through a minimal concrete subclass
# ---------------------------------------------------------------------------


class TestValidateCommon:
    @pytest.mark.parametrize("latitude", [91.0, -91.0])
    def test_rejects_invalid_latitude(self, latitude):
        with pytest.raises(ValueError, match="Latitude"):
            _StubSource(latitude=latitude)

    @pytest.mark.parametrize("horizon", [0, -1, 1.5])
    def test_rejects_non_positive_horizon(self, horizon):
        with pytest.raises(ValueError, match="Horizon must be a positive integer"):
            _StubSource(horizon=horizon)

    def test_max_horizon_enforced_when_given(self):
        source = _StubSource(horizon=5)
        with pytest.raises(ValueError, match="Horizon must be at most 3"):
            source._validate_common(max_horizon=3)

    def test_max_horizon_skipped_when_not_given(self):
        source = _StubSource(horizon=100)
        source._validate_common()  # no max_horizon: does not raise

    def test_timezone_is_required(self):
        # A default used to compute the cutoff in the wrong zone for markets outside CET (e.g. London's local
        # day landing on the wrong UTC calendar date), leaking future runs. Construction with no zone is
        # allowed (a DataPipeline may still supply one later), but using the source without one anywhere raises.
        source = _StubSource(timezone=None)
        assert source.timezone is None
        with pytest.raises(ValueError, match="timezone is required"):
            source.get_cache_config()

    def test_invalid_timezone_still_reports_the_bad_value(self):
        with pytest.raises(ValueError, match="Invalid timezone: 'Mars/Olympus'"):
            _StubSource(timezone="Mars/Olympus")

    def test_rejects_invalid_api(self):
        with pytest.raises(ValueError, match="api must be"):
            _StubSource(api="latest_runs")

    def test_rejects_invalid_run(self):
        with pytest.raises(ValueError, match="run"):
            _StubSource(run="9")


class TestCutoffInstants:
    def test_past_cutoffs_are_final(self):
        source = _StubSource(run="10")
        source._now = lambda: pd.Timestamp("2030-01-01", tz="UTC")
        rows = pd.date_range("2024-01-10", periods=3, freq="1h", tz="UTC")
        cutoff, final = source._cutoff_instants(rows)
        assert final.all()
        assert (cutoff <= source._now()).all()

    def test_future_cutoffs_use_now_and_are_not_final(self, caplog):
        source = _StubSource(run="10")
        now = pd.Timestamp("2024-01-10 05:00", tz=TZ).tz_convert("UTC")
        source._now = lambda: now
        rows = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        with caplog.at_level(logging.INFO, logger="epftoolbox2.data.sources.open_meteo"):
            source.logger.propagate = True
            cutoff, final = source._cutoff_instants(rows)
        assert not final.any()
        assert cutoff[0] == now
        assert "cutoff after now" in caplog.text


class TestFinish:
    def _frame(self):
        index = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        return pd.DataFrame({"temperature_2m_d+1": [10.0, 12.0], "wind_speed_100m_d+1": [3.6, 7.2]}, index=index)

    def test_prefix_applied(self):
        source = _StubSource(prefix="pl", horizon=1)
        result = source._finish(self._frame(), np.ones(2, dtype=bool))
        assert list(result.columns) == ["pl_temperature_2m_d+1", "pl_wind_speed_100m_d+1"]

    def test_no_prefix_by_default(self):
        source = _StubSource(horizon=1)
        result = source._finish(self._frame(), np.ones(2, dtype=bool))
        assert list(result.columns) == ["temperature_2m_d+1", "wind_speed_100m_d+1"]

    def test_strict_raises_on_missing_values(self):
        source = _StubSource(horizon=1, strict=True)
        frame = self._frame()
        frame.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="temperature_2m_d\\+1"):
            source._finish(frame, np.ones(2, dtype=bool))

    def test_trailing_unfilled_rows_are_dropped(self):
        source = _StubSource(horizon=1)
        frame = self._frame()
        frame.iloc[1, :] = np.nan
        result = source._finish(frame, np.ones(2, dtype=bool))
        assert len(result) == 1


class TestReportCoverage:
    def test_warns_with_complete_from_and_per_hour_breakdown_when_uneven(self, caplog):
        # wind_speed_100m is missing only at local hour 23 (winter UTC+1: UTC hour 22), the way JMA's
        # previous_day5-7 are missing before 2021 only for the hours that need that bucket: the overall share
        # hides which hours are affected, and the first non-null value would look earlier than the column
        # actually becomes complete.
        source = _StubSource(horizon=1)
        index = pd.date_range("2024-01-10", periods=48, freq="1h", tz="UTC")
        local_hour = index.tz_convert(TZ).hour
        wind = np.where(local_hour == 23, np.nan, 1.0)
        frame = pd.DataFrame({"temperature_2m_d+1": 1.0, "wind_speed_100m_d+1": wind}, index=index)
        with caplog.at_level(logging.WARNING, logger="epftoolbox2.data.sources.open_meteo"):
            source.logger.propagate = True
            source._finish(frame, np.ones(len(index), dtype=bool))
        assert "wind_speed_100m_d+1" in caplog.text
        assert "temperature_2m_d+1:" not in caplog.text
        assert "missing share by local hour" in caplog.text
        assert "23h 100%" in caplog.text

    def test_complete_from_is_none_when_the_column_never_completes(self, caplog):
        source = _StubSource(horizon=1)
        frame = self._frame()
        frame.iloc[-1, 1] = np.nan  # last row of wind_speed_100m_d+1 stays missing
        with caplog.at_level(logging.WARNING, logger="epftoolbox2.data.sources.open_meteo"):
            source.logger.propagate = True
            source._finish(frame, np.ones(2, dtype=bool))
        assert "complete from None" in caplog.text

    def _frame(self):
        index = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        return pd.DataFrame({"temperature_2m_d+1": [10.0, 12.0], "wind_speed_100m_d+1": [3.6, 7.2]}, index=index)


class TestCoverageHelpers:
    def test_first_complete_timestamp_is_one_step_past_the_last_missing_row(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
        series = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0], index=idx)
        assert OpenMeteoBase._first_complete_timestamp(series, dropped_final=0) == idx[3]

    def test_first_complete_timestamp_is_the_start_when_never_missing(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        series = pd.Series([1.0, 2.0, 3.0], index=idx)
        assert OpenMeteoBase._first_complete_timestamp(series, dropped_final=0) == idx[0]

    def test_first_complete_timestamp_is_none_when_the_last_row_is_missing(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        series = pd.Series([1.0, 2.0, np.nan], index=idx)
        assert OpenMeteoBase._first_complete_timestamp(series, dropped_final=0) is None

    def test_first_complete_timestamp_is_none_when_rows_were_dropped_at_the_tail(self):
        # dropped_final rows are all-NaN by construction (see _drop_unfilled_tail), so the column can't be
        # complete beyond the returned data even though every returned row happens to be filled.
        idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        series = pd.Series([1.0, 2.0, 3.0], index=idx)
        assert OpenMeteoBase._first_complete_timestamp(series, dropped_final=2) is None

    def test_hourly_missing_share_is_none_when_even_across_hours(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        series = pd.Series([np.nan, np.nan, np.nan, np.nan], index=idx)
        hours = pd.Index([0, 1, 0, 1])
        assert OpenMeteoBase._hourly_missing_share(series, hours) is None

    def test_hourly_missing_share_reports_uneven_hours(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        series = pd.Series([np.nan, 1.0, np.nan, np.nan], index=idx)
        hours = pd.Index([0, 1, 0, 1])
        share = OpenMeteoBase._hourly_missing_share(series, hours)
        assert share[0] == 1.0
        assert share[1] == 0.5


class TestCommonCacheConfig:
    def test_api_and_run_are_in_the_cache_config(self):
        config = _StubSource().get_cache_config()
        assert config["api"] == "previous_runs"
        assert config["run"] == "00"

    def test_changing_run_changes_the_cache_config(self):
        base = _StubSource().get_cache_config()
        other = _StubSource(run="06").get_cache_config()
        assert base != other

    def test_changing_api_changes_the_cache_config(self):
        base = _StubSource().get_cache_config()
        other = _StubSource(api="single_runs").get_cache_config()
        assert base != other

    def test_columns_are_sorted(self):
        config = _StubSource(columns=["wind_speed_100m", "temperature_2m"]).get_cache_config()
        assert config["columns"] == ["temperature_2m", "wind_speed_100m"]


class TestRequest:
    def _source_with_response(self, payload):
        response = Mock()
        response.status_code = 200
        response.json.return_value = payload
        source = _StubSource()
        source.session = Mock()
        source.session.get.return_value = response
        return source

    def test_returns_data_on_success(self):
        source = self._source_with_response({"hourly": {"time": []}})
        assert source._request("http://example.test", {}) == {"hourly": {"time": []}}

    def test_unavailable_reason_returns_none_without_retrying(self):
        source = self._source_with_response({"error": True, "reason": "requested model run is not available"})
        result = source._request("http://example.test", {}, unavailable="requested model run is not available")
        assert result is None
        assert source.session.get.call_count == 1

    def test_unknown_error_raises(self):
        source = self._source_with_response({"error": True, "reason": "Something else"})
        with pytest.raises(ValueError, match="Open-Meteo API error"):
            source._request("http://example.test", {})

    def test_retries_on_request_exception(self):
        source = _StubSource()
        source.session = Mock()
        source.session.get.side_effect = requests.RequestException("boom")
        with patch("time.sleep"):
            with pytest.raises(requests.RequestException):
                source._request("http://example.test", {})
        assert source.session.get.call_count == source.MAX_RETRIES
