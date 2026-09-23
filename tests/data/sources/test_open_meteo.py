import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from epftoolbox2.data.sources.open_meteo import OpenMeteoSource, stored_step_hours
from epftoolbox2.pipelines import DataPipeline

TZ = "Europe/Warsaw"
ISSUE = "10:00"
DELAY_H = 10
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
HOUR = pd.Timedelta(hours=1)

# ---------------------------------------------------------------------------
# Synthetic Previous Runs archive
# ---------------------------------------------------------------------------
# Runs start every 6 h (00/06/12/18 UTC). `previous_dayK` at valid time T holds the latest run initialised at or
# before T - 24K h (Open-Meteo docs; confirmed against the Single Runs API). For a model stored every `step_h`
# hours, Open-Meteo interpolates hourly values from the native points T_{n-1}..T_{n+2} around T, so an off-grid
# hourly value can depend on the run behind T_{n+2}. The archive encodes, per cell:
#   temperature_2m -> latest run init (hours since epoch) among the native points the value depends on
#   wind_speed_10m -> the valid time T (hours since epoch)


def _hours(ts):
    return (ts - EPOCH) / HOUR


def _run_init(point, k):
    return (point - pd.Timedelta(days=k)).floor("6h")


def _latest_init(valid, k, step_h):
    if step_h == 1 or valid.hour % step_h == 0:
        return _run_init(valid, k)
    last_point = valid.floor(f"{step_h}h") + pd.Timedelta(hours=2 * step_h)
    return _run_init(last_point, k)


def make_archive(step_h=1, latest_run=None, null_columns=(), max_bucket=7):
    """Return a fake `session.get` that answers Previous Runs requests from the synthetic archive."""
    calls = []

    def get(url, params=None, timeout=None):
        calls.append(params)
        times = pd.date_range(params["start_date"], params["end_date"] + " 23:00", freq="1h", tz="UTC")
        hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
        for key in params["hourly"]:
            column, k = key.rsplit("_previous_day", 1)
            k = int(k)
            values = []
            for t in times:
                init = _latest_init(t, k, step_h)
                if column in null_columns or k > max_bucket or (latest_run is not None and init > latest_run):
                    values.append(None)
                elif column == "temperature_2m":
                    values.append(float(_hours(init)))
                else:
                    values.append(float(_hours(t)))
            hourly[key] = values
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"hourly": hourly}
        return response

    get.calls = calls
    return get


def make_source(archive=None, **kwargs):
    kwargs.setdefault("columns", ["temperature_2m", "wind_speed_10m"])
    kwargs.setdefault("timezone", TZ)
    source = OpenMeteoSource(latitude=52.23, longitude=21.01, **kwargs)
    source.session = Mock()
    source.session.get.side_effect = archive or make_archive()
    source.console.quiet = True
    return source


def issue_times(index, issue=ISSUE, tz=TZ):
    """Issue instant (UTC) of the local day each row belongs to, computed independently of the source."""
    hh, mm = (int(x) for x in issue.split(":"))
    out = []
    for t in index:
        day = t.tz_convert(tz).date()
        out.append(pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hh, minute=mm, tz=tz))
    return pd.DatetimeIndex(out).tz_convert("UTC")


def from_hours(values):
    return pd.to_datetime(np.atleast_1d(np.asarray(values, dtype=float)) * 3600, unit="s", utc=True)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestOpenMeteoSourceInit:
    def test_init_valid_params(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ)
        assert source.latitude == 52.23
        assert source.longitude == 21.01
        assert source.horizon == 5
        assert source.model == "jma_seamless"
        assert source.prefix == ""
        assert source.vintage == "issue_time"
        assert source.timezone == TZ
        assert source.issue_time == ISSUE
        assert source.publication_delay_h == DELAY_H
        assert source.strict is False
        assert len(source.columns) > 0

    def test_init_custom_model(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ, model="icon_seamless")
        assert source.model == "icon_seamless"

    def test_init_custom_columns(self):
        custom_cols = ["temperature_2m", "rain"]
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ, columns=custom_cols)
        assert source.columns == custom_cols

    def test_columns_do_not_alias_class_default(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ)
        source.columns.append("wind_speed_100m")
        assert "wind_speed_100m" not in OpenMeteoSource.DEFAULT_COLUMNS
        assert "wind_speed_100m" not in OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ).columns

    def test_columns_copy_caller_list(self):
        custom_cols = ["temperature_2m"]
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ, columns=custom_cols)
        source.columns.append("rain")
        assert custom_cols == ["temperature_2m"]

    def test_init_with_prefix(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ, prefix="weather")
        assert source.prefix == "weather"

    @pytest.mark.parametrize("latitude", [91.0, -91.0])
    def test_init_invalid_latitude(self, latitude):
        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            OpenMeteoSource(latitude=latitude, longitude=21.01)

    @pytest.mark.parametrize("longitude", [181.0, -181.0])
    def test_init_invalid_longitude(self, longitude):
        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            OpenMeteoSource(latitude=52.23, longitude=longitude)

    @pytest.mark.parametrize("horizon", [0, -1])
    def test_init_invalid_horizon_not_positive(self, horizon):
        with pytest.raises(ValueError, match="Horizon must be a positive integer"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, horizon=horizon)

    @pytest.mark.parametrize("horizon", [8, 14])
    def test_init_horizon_beyond_previous_day7_is_rejected(self, horizon):
        # The Previous Runs API only serves previous_day1..7; previous_day8+ come back as nulls.
        with pytest.raises(ValueError, match="previous_day"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, horizon=horizon, vintage="valid_time")

    def test_init_horizon_needing_bucket_8_is_rejected(self):
        # Leak-free d+7 needs previous_day8 for most hours, which the API does not serve.
        with pytest.raises(ValueError, match="previous_day8"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ, horizon=7)

    def test_init_legacy_vintage_allows_horizon_7(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ, horizon=7, vintage="valid_time")
        assert source.horizon == 7

    def test_hourly_model_with_short_delay_allows_horizon_6(self):
        # ICON is published about 4 h after initialisation and stored hourly up to 78 h lead.
        source = OpenMeteoSource(
            latitude=52.23, longitude=21.01, timezone=TZ, horizon=6, model="icon_seamless", publication_delay_h=4
        )
        assert source.horizon == 6
        with pytest.raises(ValueError, match="d\\+6 needs previous_day8"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ, horizon=6, model="icon_seamless")

    def test_timezone_is_required(self):
        # A default used to compute the issue time in the wrong zone for markets outside CET (e.g. London 23:00
        # on D silently becoming 10:00 on D+1), leaking future runs. It must now be passed explicitly.
        with pytest.raises(ValueError, match="timezone is required"):
            OpenMeteoSource(latitude=52.23, longitude=21.01)

    def test_invalid_timezone_still_reports_the_bad_value(self):
        with pytest.raises(ValueError, match="Invalid timezone: 'Mars/Olympus'"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Mars/Olympus")

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"timezone": "Mars/Olympus"}, "timezone"),
            ({"issue_time": "25:00"}, "issue_time"),
            ({"issue_time": "10"}, "issue_time"),
            ({"publication_delay_h": -1}, "publication_delay_h"),
            ({"vintage": "latest"}, "vintage"),
            ({"temporal_resolution": "hourly_2"}, "temporal_resolution"),
        ],
    )
    def test_init_rejects_invalid_parameters(self, kwargs, match):
        kwargs.setdefault("timezone", TZ)
        with pytest.raises(ValueError, match=match):
            OpenMeteoSource(latitude=52.23, longitude=21.01, **kwargs)

    def test_requests_enough_buckets_for_the_fallback(self):
        # JMA is stored 6-hourly, so the late hours of D+i need previous_day{i+2}.
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ, horizon=5)
        assert max(source._buckets) == 7
        assert min(source._buckets) >= 1


# ---------------------------------------------------------------------------
# Vintage selection: no run published after the issue time
# ---------------------------------------------------------------------------


class TestIssueTimeVintage:
    START = pd.Timestamp("2024-03-25", tz="UTC")
    END = pd.Timestamp("2024-04-05", tz="UTC")

    @pytest.mark.parametrize("model, step_h", [("icon_seamless", 1), ("jma_seamless", 6), ("ecmwf_ifs04", 3)])
    def test_every_value_comes_from_runs_published_before_issue_time(self, model, step_h):
        source = make_source(make_archive(step_h=step_h), model=model, horizon=3)
        df = source.fetch(self.START, self.END)

        issue = issue_times(df.index)
        for i in range(1, 4):
            init = from_hours(df[f"temperature_2m_d+{i}"])
            assert df[f"temperature_2m_d+{i}"].notna().all()
            late = init + pd.Timedelta(hours=DELAY_H) > issue
            assert not late.any(), f"d+{i}: {late.sum()} rows use runs published after the issue time"

    def test_uses_the_freshest_admissible_bucket(self):
        # Hourly model: d+1 uses previous_day1 where a run initialised as late as T - 24 h would have been
        # published by 10:00 on D, otherwise previous_day2. The source does not assume run cycles, so "as late as
        # T - 24 h" is the bound, not the actual 6-hourly run time.
        source = make_source(make_archive(step_h=1), model="icon_seamless", horizon=1)
        df = source.fetch(self.START, self.END)
        init = from_hours(df["temperature_2m_d+1"])
        valid = from_hours(df["wind_speed_10m_d+1"])
        lead_days = ((valid - init) / pd.Timedelta(days=1)).astype(int)
        assert set(lead_days) == {1, 2}
        issue = issue_times(df.index)
        # Wherever previous_day2 was used, previous_day1 would have been too late.
        used_two = lead_days == 2
        assert (valid[used_two] - pd.Timedelta(days=1) + pd.Timedelta(hours=DELAY_H) > issue[used_two]).all()

    def test_legacy_valid_time_vintage_reproduces_old_mapping_and_warns(self, caplog):
        source = make_source(make_archive(step_h=1), model="icon_seamless", horizon=2, vintage="valid_time")
        with caplog.at_level(logging.WARNING, logger="epftoolbox2.data.sources.open_meteo"):
            source.logger.propagate = True
            df = source.fetch(self.START, self.END)
        assert "look-ahead" in caplog.text
        for i in (1, 2):
            valid = from_hours(df[f"wind_speed_10m_d+{i}"])
            assert (valid == df.index + pd.Timedelta(days=i)).all()
            init = from_hours(df[f"temperature_2m_d+{i}"])
            assert (init == (df.index).floor("6h")).all()
        late = from_hours(df["temperature_2m_d+1"]) + pd.Timedelta(hours=DELAY_H) > issue_times(df.index)
        assert late.any()

    def test_legacy_valid_time_vintage_warns_at_construction(self, caplog):
        # A cache hit never calls fetch(), so the warning must not depend on it: it must fire when the source is
        # built, not only when data is actually downloaded.
        logging.getLogger("epftoolbox2.data.sources.open_meteo").propagate = True
        with caplog.at_level(logging.WARNING, logger="epftoolbox2.data.sources.open_meteo"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ, vintage="valid_time")
        assert "look-ahead" in caplog.text


# ---------------------------------------------------------------------------
# Calendar-day alignment across DST
# ---------------------------------------------------------------------------


class TestDstAlignment:
    @pytest.mark.parametrize(
        "row, expected",
        [
            ("2024-03-30 10:00", "2024-03-31 10:00"),  # spring forward on D+1
            ("2024-03-30 23:00", "2024-03-31 23:00"),
            ("2024-10-26 10:00", "2024-10-27 10:00"),  # fall back on D+1
            ("2024-10-26 00:00", "2024-10-27 00:00"),
            ("2024-03-31 10:00", "2024-04-01 10:00"),  # row on the switch day itself
        ],
    )
    def test_d_plus_1_is_same_local_wall_clock_hour_next_day(self, row, expected):
        source = make_source(make_archive(step_h=1), model="icon_seamless", horizon=1)
        start = pd.Timestamp(row, tz=TZ).normalize().tz_convert("UTC") - pd.Timedelta(days=1)
        df = source.fetch(start, start + pd.Timedelta(days=3))
        valid = from_hours(df.loc[pd.Timestamp(row, tz=TZ), "wind_speed_10m_d+1"])[0]
        assert valid == pd.Timestamp(expected, tz=TZ)

    def test_d_plus_3_across_dst_keeps_wall_clock_hour(self):
        source = make_source(make_archive(step_h=1), model="icon_seamless", horizon=3)
        start = pd.Timestamp("2024-03-26", tz="UTC")
        df = source.fetch(start, start + pd.Timedelta(days=5))
        # Skip the 02:00 row whose target (2024-03-31 02:00) does not exist; it takes 03:00 (tested below).
        df = df[df.index != pd.Timestamp("2024-03-28 02:00", tz=TZ)]
        valid = from_hours(df["wind_speed_10m_d+3"]).tz_convert(TZ)
        local = df.index.tz_convert(TZ)
        assert (valid.hour == local.hour).all()
        assert ((valid.normalize().tz_localize(None) - local.normalize().tz_localize(None)).days == 3).all()

    def test_nonexistent_target_hour_uses_next_hour(self):
        # 2024-03-31 02:00 does not exist in Warsaw; the 02:00 row of 03-30 takes the 03:00 value.
        source = make_source(make_archive(step_h=1), model="icon_seamless", horizon=1)
        df = source.fetch(pd.Timestamp("2024-03-29", tz="UTC"), pd.Timestamp("2024-04-01", tz="UTC"))
        valid = from_hours(df.loc[pd.Timestamp("2024-03-30 02:00", tz=TZ), "wind_speed_10m_d+1"])[0]
        assert valid == pd.Timestamp("2024-03-31 03:00", tz=TZ)


# ---------------------------------------------------------------------------
# Returned range, live runs and coverage checks
# ---------------------------------------------------------------------------


class TestReturnedRange:
    def test_returns_exactly_start_to_end_exclusive(self):
        source = make_source(horizon=2)
        start, end = pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-12", tz="UTC")
        df = source.fetch(start, end)
        assert df.index.min() == start
        assert df.index.max() == end - HOUR
        assert len(df) == 48
        assert str(df.index.tz) == "UTC"

    def test_multi_chunk_fetch_has_no_gaps_and_no_duplicates(self):
        archive = make_archive(step_h=1)
        source = make_source(archive, model="icon_seamless", horizon=2)
        start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-08-01", tz="UTC")
        df = source.fetch(start, end)
        assert len(archive.calls) >= 3
        assert not df.index.duplicated().any()
        assert len(df) == (end - start) / HOUR
        assert df.notna().all().all()

    def test_live_fetch_keeps_all_hours_of_the_issue_day(self):
        # "now" is 08:00 CET on D, before the 10:00 issue time: rows of D are provisional but must be complete
        # (the old now+13h cut dropped 22:00 and 23:00 local).
        source = make_source(horizon=1)
        now = pd.Timestamp("2024-01-11 08:00", tz=TZ).tz_convert("UTC")
        source._now = lambda: now
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-12", tz="UTC"))
        day = df[df.index.tz_convert(TZ).date == pd.Timestamp("2024-01-11").date()]
        assert len(day) == 24
        assert day.notna().all().all()
        init = from_hours(day["temperature_2m_d+1"])
        assert (init + pd.Timedelta(hours=DELAY_H) <= now).all()

    def test_rows_that_cannot_be_filled_are_not_returned(self):
        latest_run = pd.Timestamp("2024-01-11 00:00", tz="UTC")
        source = make_source(make_archive(step_h=1, latest_run=latest_run), model="icon_seamless", horizon=1)
        now = latest_run + pd.Timedelta(hours=DELAY_H)
        source._now = lambda: now
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-20", tz="UTC"))
        assert df.notna().any(axis=1).all()
        assert df.index.max() < pd.Timestamp("2024-01-15", tz="UTC")
        assert (from_hours(df["temperature_2m_d+1"].dropna()) <= latest_run).all()

    def test_warns_about_missing_values_with_share_and_first_valid(self, caplog):
        source = make_source(make_archive(null_columns=("wind_speed_10m",)), horizon=1)
        with caplog.at_level(logging.WARNING, logger="epftoolbox2.data.sources.open_meteo"):
            source.logger.propagate = True
            source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-12", tz="UTC"))
        assert "wind_speed_10m_d+1" in caplog.text
        assert "100.0%" in caplog.text
        assert "temperature_2m_d+1:" not in caplog.text

    def test_strict_raises_on_missing_values(self):
        source = make_source(make_archive(null_columns=("wind_speed_10m",)), horizon=1, strict=True)
        with pytest.raises(ValueError, match="wind_speed_10m_d\\+1"):
            source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-12", tz="UTC"))

    def test_strict_passes_on_complete_data(self):
        source = make_source(horizon=1, strict=True)
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-12", tz="UTC"))
        assert len(df) == 48

    def test_prefix_applied(self):
        source = make_source(horizon=1, prefix="warsaw")
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert list(df.columns) == ["warsaw_temperature_2m_d+1", "warsaw_wind_speed_10m_d+1"]


class TestStoredStepHours:
    def test_ecmwf_ifs04_is_not_matched_by_the_hourly_ecmwf_ifs_prefix(self):
        # ecmwf_ifs04 (ECMWF IFS 0.4 deg) is stored 3-hourly (6-hourly beyond 144 h lead). Its name starts with
        # "ecmwf_ifs", so without its own entry it used to match the hourly ecmwf_ifs (9 km) one instead,
        # understating how far the interpolated hourly value can reach into runs not yet published.
        assert stored_step_hours("ecmwf_ifs04", 1) == 3
        assert stored_step_hours("ecmwf_ifs04", 6) == 6
        assert stored_step_hours("ecmwf_ifs", 1) == 1


# ---------------------------------------------------------------------------
# Timestamps, chunking, requests
# ---------------------------------------------------------------------------


class TestOpenMeteoSourceValidation:
    def test_fetch_validates_timestamps(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ)
        with pytest.raises(ValueError, match="End timestamp.*must be after start timestamp"):
            source.fetch(pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))

    def test_fetch_converts_naive_and_local_timestamps_to_utc(self):
        source = make_source(horizon=1)
        df_naive = source.fetch(pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-11"))
        df_local = source.fetch(
            pd.Timestamp("2024-01-10 01:00", tz="Europe/Warsaw"), pd.Timestamp("2024-01-11 01:00", tz="Europe/Warsaw")
        )
        assert df_naive.index.equals(df_local.index)
        assert str(df_naive.index.tz) == "UTC"


class TestOpenMeteoSourceChunking:
    def test_generate_chunks_single_month(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ)
        start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-15", tz="UTC")
        chunks = source._generate_chunks(start, end, months=3)
        assert chunks == [(start, end)]

    def test_generate_chunks_are_contiguous(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ)
        start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-07-01", tz="UTC")
        chunks = source._generate_chunks(start, end, months=3)
        assert len(chunks) >= 2
        assert chunks[0][0] == start
        assert chunks[-1][1] == end
        for (_, first_end), (second_start, _) in zip(chunks, chunks[1:]):
            assert second_start == first_end

    def test_generate_chunks_validates_months_parameter(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ)
        start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-07-01", tz="UTC")
        for months in (0, -1):
            with pytest.raises(ValueError, match="months parameter must be positive"):
                source._generate_chunks(start, end, months=months)

    def test_request_covers_all_needed_valid_times(self):
        archive = make_archive()
        source = make_source(archive, horizon=2)
        source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-12", tz="UTC"))
        params = archive.calls[0]
        assert params["start_date"] == "2024-01-10"
        assert params["end_date"] >= "2024-01-14"
        assert params["timezone"] == "GMT"
        buckets = {int(key.rsplit("_previous_day", 1)[1]) for key in params["hourly"]}
        assert buckets == set(source._buckets)


class TestOpenMeteoSourceAPIRequests:
    def _source_with_response(self, payload):
        response = Mock()
        response.status_code = 200
        response.json.return_value = payload
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ)
        source.session = Mock()
        source.session.get.return_value = response
        return source

    def test_fetch_chunk_success(self, sample_weather_response):
        source = self._source_with_response(sample_weather_response)
        result = source._fetch_chunk(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC"))
        assert isinstance(result, pd.DataFrame)
        assert "temperature_2m_previous_day1" in result.columns

    def test_minutely_limit_retries_with_capped_backoff(self):
        source = self._source_with_response(
            {"error": True, "reason": "Minutely API request limit exceeded. Please try again in one minute."}
        )
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="Failed to fetch data after"):
                source._fetch_chunk(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"))
        assert mock_sleep.called
        assert max(call.args[0] for call in mock_sleep.call_args_list) <= OpenMeteoSource.BACKOFF_CAP_S

    def test_concurrent_requests_error_backs_off(self):
        source = self._source_with_response({"error": True, "reason": "Too many concurrent requests"})
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="Failed to fetch data after"):
                source._fetch_chunk(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"))
        waits = [call.args[0] for call in mock_sleep.call_args_list]
        assert waits == sorted(waits)
        assert max(waits) <= OpenMeteoSource.BACKOFF_CAP_S

    def test_hourly_limit_raises_instead_of_sleeping_for_hours(self):
        source = self._source_with_response({"error": True, "reason": "Hourly API request limit exceeded."})
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="Hourly API request limit"):
                source._fetch_chunk(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"))
        assert all(call.args[0] <= OpenMeteoSource.BACKOFF_CAP_S for call in mock_sleep.call_args_list)

    def test_unknown_api_error_raises(self):
        source = self._source_with_response({"error": True, "reason": "Cannot initialize previous_day9"})
        with pytest.raises(ValueError, match="Open-Meteo API error"):
            source._fetch_chunk(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"))

    def test_fetch_chunk_retries_on_request_exception(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ)
        source.session = Mock()
        source.session.get.side_effect = requests.RequestException("Network error")
        with patch("time.sleep"):
            with pytest.raises(requests.RequestException, match="Network error"):
                source._fetch_chunk(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"))
        assert source.session.get.call_count == OpenMeteoSource.MAX_RETRIES


class TestOpenMeteoSourceParsing:
    def test_parse_response_keeps_every_bucket_value_in_place(self, sample_weather_response):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ)
        table = source._parse_response(sample_weather_response)
        assert str(table.index.tz) == "UTC"
        assert len(table) == 72
        # The fixture encodes 100 * bucket + position, so any offset error shows up here.
        assert table["temperature_2m_previous_day1"].tolist() == [100.0 + j for j in range(72)]
        assert table["temperature_2m_previous_day2"].iloc[30] == 230.0
        assert table.index[30] == pd.Timestamp("2024-01-02 06:00", tz="UTC")

    def test_parse_response_turns_nulls_into_nan(self, sample_weather_response):
        sample_weather_response["hourly"]["rain_previous_day1"][5] = None
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ)
        table = source._parse_response(sample_weather_response)
        assert np.isnan(table["rain_previous_day1"].iloc[5])

    def test_parse_response_handles_missing_hourly(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ)
        assert source._parse_response({}).empty
        assert source._parse_response({"hourly": {}}).empty

    def test_parse_response_tolerates_short_arrays(self, sample_weather_response_incomplete):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ)
        table = source._parse_response(sample_weather_response_incomplete)
        assert len(table) == 2
        assert np.isnan(table["temperature_2m_previous_day1"].iloc[1])

    def test_legacy_mapping_uses_bucket_i_at_t_plus_i_days(self, sample_weather_response):
        # Legacy vintage: d+i at row t is previous_day{i} at t + 24 h * i (index + 24 * i).
        response = Mock()
        response.json.return_value = sample_weather_response
        source = OpenMeteoSource(
            latitude=52.23,
            longitude=21.01,
            timezone=TZ,
            horizon=2,
            columns=["temperature_2m"],
            vintage="valid_time",
        )
        source.session = Mock()
        source.session.get.return_value = response
        df = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"))
        assert df["temperature_2m_d+1"].tolist() == [100.0 + j + 24 for j in range(24)]
        assert df["temperature_2m_d+2"].tolist() == [200.0 + j + 48 for j in range(24)]


class TestCacheConfigAndSerialization:
    def test_cache_config_has_schema_version_and_vintage_parameters(self):
        config = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ).get_cache_config()
        assert config["schema_version"] >= 2
        for key in ("vintage", "issue_time", "publication_delay_h", "timezone", "horizon", "model", "columns"):
            assert key in config

    @pytest.mark.parametrize(
        "change",
        [
            {"issue_time": "11:30"},
            {"publication_delay_h": 6},
            {"timezone": "Europe/London"},
            {"vintage": "valid_time"},
        ],
    )
    def test_output_affecting_parameters_change_the_cache_config(self, change):
        base = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ).get_cache_config()
        other = OpenMeteoSource(latitude=52.23, longitude=21.01, **{"timezone": TZ, **change}).get_cache_config()
        assert base != other

    def test_strict_does_not_change_the_cache_config(self):
        base = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ).get_cache_config()
        strict = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=TZ, strict=True).get_cache_config()
        assert base == strict

    def test_yaml_round_trip(self, tmp_path):
        source = OpenMeteoSource(
            latitude=52.23,
            longitude=21.01,
            horizon=3,
            model="icon_seamless",
            columns=["temperature_2m"],
            prefix="warsaw",
            timezone="Europe/Berlin",
            issue_time="11:30",
            publication_delay_h=4,
            strict=True,
        )
        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(source).save(path)
        loaded = DataPipeline.load(path).sources[0]
        assert isinstance(loaded, OpenMeteoSource)
        assert loaded.get_cache_config() == source.get_cache_config()
        assert loaded.strict is True


@pytest.mark.integration
class TestOpenMeteoSourceIntegration:
    """Integration tests that hit the real API"""

    def test_fetch_weather_data_warsaw(self):
        source = OpenMeteoSource(latitude=52.2297, longitude=21.0122, timezone=TZ, horizon=3)
        start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC")
        df = source.fetch(start, end)
        assert len(df) == 48
        assert df.index.min() == start and df.index.max() < end

    def test_fetch_weather_data_limited_columns(self):
        source = OpenMeteoSource(
            latitude=52.2297,
            longitude=21.0122,
            timezone=TZ,
            horizon=2,
            columns=["temperature_2m", "wind_speed_10m"],
        )
        df = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"))
        assert sorted(df.columns) == sorted(
            ["temperature_2m_d+1", "temperature_2m_d+2", "wind_speed_10m_d+1", "wind_speed_10m_d+2"]
        )

    def test_fetch_long_period(self):
        source = OpenMeteoSource(
            latitude=52.2297, longitude=21.0122, timezone=TZ, horizon=5, columns=["temperature_2m"]
        )
        df = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-06-01", tz="UTC"))
        assert len(df) == (pd.Timestamp("2024-06-01") - pd.Timestamp("2024-01-01")) / HOUR
