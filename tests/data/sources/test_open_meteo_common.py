import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from epftoolbox2.data.sources._open_meteo_common import (
    OpenMeteoBase,
    add_derived_columns,
    local_issue_times,
    normalise_derived,
    parse_hourly,
    parse_issue_time,
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
            issue_time="10:00",
            publication_delay_h=10,
            derived=None,
            hdd_base_c=18.0,
            cdd_base_c=21.0,
            strict=False,
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
        self.issue_time = params["issue_time"]
        self.publication_delay_h = params["publication_delay_h"]
        self.derived = normalise_derived(params["derived"])
        self.hdd_base_c = params["hdd_base_c"]
        self.cdd_base_c = params["cdd_base_c"]
        self.strict = params["strict"]
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
# Issue-time helpers
# ---------------------------------------------------------------------------


class TestParseIssueTime:
    def test_parses_valid_time(self):
        assert parse_issue_time("10:30") == (10, 30)
        assert parse_issue_time("00:00") == (0, 0)
        assert parse_issue_time("23:59") == (23, 59)

    @pytest.mark.parametrize("value", ["25:00", "10:60", "10", "abc", None, 10.0])
    def test_rejects_invalid_time(self, value):
        with pytest.raises(ValueError, match="issue_time"):
            parse_issue_time(value)


class TestLocalIssueTimes:
    def test_resolves_local_wall_clock_time_to_utc(self):
        rows = pd.DatetimeIndex([pd.Timestamp("2024-01-10 00:00", tz="UTC")])
        issue = local_issue_times(rows, TZ, "10:00")
        assert issue[0] == pd.Timestamp("2024-01-10 09:00", tz="UTC")

    def test_nonexistent_time_resolves_just_before_the_gap(self):
        # 2024-03-31 02:00-03:00 does not exist in Warsaw (spring forward); shift_backward keeps it from raising
        # and lands just before the gap (01:59:59.999999999 CET = 00:59:59.999999999 UTC).
        rows = pd.DatetimeIndex([pd.Timestamp("2024-03-31 00:00", tz=TZ).tz_convert("UTC")])
        issue = local_issue_times(rows, TZ, "02:30")
        assert issue[0] == pd.Timestamp("2024-03-31 00:59:59.999999999", tz="UTC")

    def test_ambiguous_time_resolves_to_the_earlier_occurrence(self):
        # 2024-10-27 02:00-03:00 occurs twice in Warsaw (fall back); the earlier (CEST, UTC+2) one is used.
        rows = pd.DatetimeIndex([pd.Timestamp("2024-10-27 00:00", tz=TZ).tz_convert("UTC")])
        issue = local_issue_times(rows, TZ, "02:30")
        assert issue[0] == pd.Timestamp("2024-10-27 00:30", tz="UTC")


class TestShiftLocalDays:
    def test_shifts_by_calendar_days_not_24h(self):
        rows = pd.DatetimeIndex([pd.Timestamp("2024-03-30 10:00", tz=TZ).tz_convert("UTC")])
        shifted = shift_local_days(rows, TZ, 1)
        # Spring-forward day: 23 local hours between 03-30 10:00 and 03-31 10:00, not 24.
        assert shifted[0] == pd.Timestamp("2024-03-31 10:00", tz=TZ)
        assert shifted[0] - rows[0] == pd.Timedelta(hours=23)

    def test_nonexistent_target_hour_shifts_forward(self):
        rows = pd.DatetimeIndex([pd.Timestamp("2024-03-30 02:00", tz=TZ).tz_convert("UTC")])
        shifted = shift_local_days(rows, TZ, 1)
        assert shifted[0] == pd.Timestamp("2024-03-31 03:00", tz=TZ)


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------


class TestAddDerivedColumns:
    def _frame(self):
        index = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        return pd.DataFrame(
            {
                "temperature_2m_d+1": [10.0, 20.0, np.nan],
                "wind_speed_100m_d+1": [3.6, 7.2, 10.8],
            },
            index=index,
        )

    def test_hdd_is_clipped_at_zero(self):
        df = add_derived_columns(self._frame(), ["hdd"], horizon=1, hdd_base_c=18.0, cdd_base_c=21.0)
        assert df["hdd_d+1"].tolist()[:2] == [8.0, 0.0]
        assert np.isnan(df["hdd_d+1"].iloc[2])

    def test_cdd_is_clipped_at_zero(self):
        df = add_derived_columns(self._frame(), ["cdd"], horizon=1, hdd_base_c=18.0, cdd_base_c=21.0)
        assert df["cdd_d+1"].tolist()[:2] == [0.0, 0.0]

    def test_cdd_positive_above_base(self):
        df = add_derived_columns(self._frame(), ["cdd"], horizon=1, hdd_base_c=18.0, cdd_base_c=5.0)
        assert df["cdd_d+1"].tolist()[:2] == [5.0, 15.0]

    def test_wind_speed_cubed_converts_kmh_to_ms(self):
        df = add_derived_columns(self._frame(), ["wind_speed_100m_cubed"], horizon=1, hdd_base_c=18.0, cdd_base_c=21.0)
        # 3.6 km/h = 1 m/s, 7.2 km/h = 2 m/s, 10.8 km/h = 3 m/s.
        np.testing.assert_allclose(df["wind_speed_100m_cubed_d+1"].tolist(), [1.0, 8.0, 27.0])

    def test_no_derived_names_returns_the_same_frame(self):
        frame = self._frame()
        assert add_derived_columns(frame, [], horizon=1, hdd_base_c=18.0, cdd_base_c=21.0) is frame

    def test_multiple_leads(self):
        df = self._frame()
        df["temperature_2m_d+2"] = [0.0, 30.0, 18.0]
        out = add_derived_columns(df, ["hdd"], horizon=2, hdd_base_c=18.0, cdd_base_c=21.0)
        assert out["hdd_d+2"].tolist() == [18.0, 0.0, 0.0]


class TestNormaliseDerived:
    def test_none_becomes_empty_list(self):
        assert normalise_derived(None) == []

    def test_empty_list_stays_empty(self):
        assert normalise_derived([]) == []

    def test_copies_the_input_list(self):
        source = ["hdd"]
        result = normalise_derived(source)
        result.append("cdd")
        assert source == ["hdd"]


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

    def test_rejects_unknown_derived_name(self):
        with pytest.raises(ValueError, match="derived must be a subset"):
            _StubSource(derived=["not_a_real_column"])

    def test_rejects_derived_missing_its_input_column(self):
        with pytest.raises(ValueError, match="needs 'temperature_2m'"):
            _StubSource(columns=["wind_speed_100m"], derived=["hdd"])

    def test_rejects_non_numeric_base_temperature(self):
        with pytest.raises(ValueError, match="hdd_base_c"):
            _StubSource(hdd_base_c="warm")

    def test_timezone_is_required(self):
        # A default used to compute the issue time in the wrong zone for markets outside CET (e.g. London 23:00
        # on D silently becoming 10:00 on D+1), leaking future runs. It must now be passed explicitly.
        with pytest.raises(ValueError, match="timezone is required"):
            _StubSource(timezone=None)

    def test_invalid_timezone_still_reports_the_bad_value(self):
        with pytest.raises(ValueError, match="Invalid timezone: 'Mars/Olympus'"):
            _StubSource(timezone="Mars/Olympus")


class TestIssueInstants:
    def test_past_issue_times_are_final(self):
        source = _StubSource(issue_time="10:00")
        source._now = lambda: pd.Timestamp("2030-01-01", tz="UTC")
        rows = pd.date_range("2024-01-10", periods=3, freq="1h", tz="UTC")
        issue, final = source._issue_instants(rows)
        assert final.all()
        assert (issue <= source._now()).all()

    def test_future_issue_times_use_now_and_are_not_final(self, caplog):
        source = _StubSource(issue_time="10:00")
        now = pd.Timestamp("2024-01-10 05:00", tz=TZ).tz_convert("UTC")
        source._now = lambda: now
        rows = pd.date_range("2024-01-10", periods=1, freq="1h", tz="UTC")
        with caplog.at_level(logging.INFO, logger="epftoolbox2.data.sources.open_meteo"):
            source.logger.propagate = True
            issue, final = source._issue_instants(rows)
        assert not final.any()
        assert issue[0] == now
        assert "issue time after now" in caplog.text


class TestFinish:
    def _frame(self):
        index = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        return pd.DataFrame({"temperature_2m_d+1": [10.0, 12.0], "wind_speed_100m_d+1": [3.6, 7.2]}, index=index)

    def test_derived_columns_added_before_prefix(self):
        source = _StubSource(prefix="pl", derived=["hdd"], horizon=1)
        result = source._finish(self._frame(), np.ones(2, dtype=bool))
        assert "pl_hdd_d+1" in result.columns

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
    def test_derived_key_omitted_when_unused(self):
        config = _StubSource().get_cache_config()
        assert "derived" not in config
        assert "hdd_base_c" not in config
        assert "cdd_base_c" not in config

    def test_derived_key_present_and_bases_included_only_when_used(self):
        config = _StubSource(derived=["hdd"]).get_cache_config()
        assert config["derived"] == ["hdd"]
        assert config["hdd_base_c"] == 18.0
        assert "cdd_base_c" not in config

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
