import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from epftoolbox2.data.sources.open_meteo import (
    OpenMeteoSource,
    known_run_hours,
    publication_delay_hours,
    stored_step_hours,
)
from epftoolbox2.pipelines import DataPipeline

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
HOUR = pd.Timedelta(hours=1)

# Spans 2024-03-31 (EU spring-forward) and 2024-10-27 (EU fall-back), for the leak/cutoff tests below.
SPRING = (pd.Timestamp("2024-03-25", tz="UTC"), pd.Timestamp("2024-04-05", tz="UTC"))
FALL = (pd.Timestamp("2024-10-20", tz="UTC"), pd.Timestamp("2024-10-31", tz="UTC"))

# ---------------------------------------------------------------------------
# Synthetic archives: mocked responses that encode each run's own init time in the values
# ---------------------------------------------------------------------------
# Previous Runs API: `previous_dayK` at valid time T holds the latest run initialised at or before T - 24K h
# (Open-Meteo docs). For a model stored every `step_h` hours, Open-Meteo interpolates hourly values from the
# native points around T, so an off-grid hourly value can depend on the run behind a stored point up to two
# steps after T. The archive encodes, per cell:
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


def make_run_archive(missing=()):
    """Return a fake `session.get` that answers Single Runs requests from a synthetic archive.

    `missing`: run inits (UTC Timestamps) the archive does not have. Values encode, per cell: `temperature_2m`
    -> hours since epoch of the run's own init (identifies which run served the cell); other columns -> hours
    since epoch of the valid time (identifies the target alignment).
    """
    missing = set(missing)
    calls = []

    def get(url, params=None, timeout=None):
        calls.append(params)
        run = pd.Timestamp(params["run"], tz="UTC")
        response = Mock()
        response.status_code = 200
        if run in missing:
            response.json.return_value = {
                "error": True,
                "reason": f"The requested model run is not available. Model: {params['models']}, run: {params['run']}Z",
            }
            return response

        n = params["forecast_days"] * 24
        times = pd.date_range(run, periods=n, freq="1h", tz="UTC")
        hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
        for key in params["hourly"]:
            if key == "temperature_2m":
                hourly[key] = [float(_hours(run))] * n
            else:
                hourly[key] = [float(_hours(t)) for t in times]
        response.json.return_value = {"hourly": hourly}
        return response

    get.calls = calls
    return get


def make_source(archive=None, **kwargs):
    kwargs.setdefault("columns", ["temperature_2m", "wind_speed_10m"])
    kwargs.setdefault("timezone", "Europe/Warsaw")
    source = OpenMeteoSource(latitude=52.23, longitude=21.01, **kwargs)
    source.session = Mock()
    source.session.get.side_effect = archive or make_archive()
    source.console.quiet = True
    return source


def cutoff_times(index, run, tz="Europe/Warsaw"):
    """Cutoff instant (UTC) of each row's local day D at `run`:00 UTC — computed independently of the source."""
    hour = int(run)
    out = []
    for t in index:
        day = t.tz_convert(tz).date()
        out.append(pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hour, tz="UTC"))
    return pd.DatetimeIndex(out)


def from_hours(values):
    return pd.to_datetime(np.atleast_1d(np.asarray(values, dtype=float)) * 3600, unit="s", utc=True)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestOpenMeteoSourceInit:
    def test_init_valid_params(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw")
        assert source.latitude == 52.23
        assert source.longitude == 21.01
        assert source.horizon == 5
        assert source.model == "jma_seamless"
        assert source.prefix == ""
        assert source.api == "previous_runs"
        assert source.run == "00"
        assert source.timezone == "Europe/Warsaw"
        assert source.strict is False
        assert len(source.columns) > 0

    def test_init_custom_model(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", model="icon_seamless")
        assert source.model == "icon_seamless"

    def test_init_custom_columns(self):
        custom_cols = ["temperature_2m", "rain"]
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", columns=custom_cols)
        assert source.columns == custom_cols

    def test_columns_do_not_alias_class_default(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw")
        source.columns.append("wind_speed_100m")
        assert "wind_speed_100m" not in OpenMeteoSource.DEFAULT_COLUMNS
        assert (
            "wind_speed_100m" not in OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw").columns
        )

    def test_columns_copy_caller_list(self):
        custom_cols = ["temperature_2m"]
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", columns=custom_cols)
        source.columns.append("rain")
        assert custom_cols == ["temperature_2m"]

    def test_init_with_prefix(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", prefix="weather")
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
    def test_init_previous_runs_horizon_beyond_previous_day7_is_rejected(self, horizon):
        # The Previous Runs API only serves previous_day1..7; previous_day8+ come back as nulls.
        with pytest.raises(ValueError, match="previous_day"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", horizon=horizon)

    def test_init_horizon_needing_bucket_8_is_rejected(self):
        # Leak-free d+6 needs previous_day8 for most hours (default model/run), which the API does not serve.
        with pytest.raises(ValueError, match="previous_day8"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", horizon=6)

    @pytest.mark.parametrize("horizon", [1, 14])
    def test_single_runs_accepts_horizon_up_to_14(self, horizon):
        source = OpenMeteoSource(
            latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", api="single_runs", horizon=horizon
        )
        assert source.horizon == horizon

    def test_single_runs_rejects_horizon_above_14(self):
        with pytest.raises(ValueError, match="Horizon must be at most 14"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", api="single_runs", horizon=15)

    def test_timezone_is_required(self):
        # A default used to compute the cutoff in the wrong zone for markets outside CET (e.g. a market's local
        # day landing on the wrong UTC calendar date), leaking future runs. Construction with no zone is
        # allowed (a DataPipeline may still supply one later), but fetching without one anywhere raises.
        source = OpenMeteoSource(latitude=52.23, longitude=21.01)
        assert source.timezone is None
        with pytest.raises(ValueError, match="timezone is required"):
            source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"))
        with pytest.raises(ValueError, match="timezone is required"):
            source.get_cache_config()

    def test_invalid_timezone_still_reports_the_bad_value(self):
        with pytest.raises(ValueError, match="Invalid timezone: 'Mars/Olympus'"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Mars/Olympus")

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"timezone": "Mars/Olympus"}, "timezone"),
            ({"api": "latest_runs"}, "api"),
            ({"run": "9"}, "run"),
            ({"run": "24"}, "run"),
            ({"temporal_resolution": "hourly_2"}, "temporal_resolution"),
        ],
    )
    def test_init_rejects_invalid_parameters(self, kwargs, match):
        kwargs.setdefault("timezone", "Europe/Warsaw")
        with pytest.raises(ValueError, match=match):
            OpenMeteoSource(latitude=52.23, longitude=21.01, **kwargs)

    def test_single_runs_rejects_a_run_hour_the_model_does_not_have(self):
        with pytest.raises(ValueError, match="jma_seamless.*has no run at '09'"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", api="single_runs", run="09")

    def test_single_runs_accepts_a_run_hour_for_an_unknown_model(self):
        # A model this library has no run-hour table for is not statically checked; whether "run" actually
        # exists is left to the archive's own response (see TestSingleRunsFetch).
        source = OpenMeteoSource(
            latitude=52.23,
            longitude=21.01,
            timezone="Europe/Warsaw",
            api="single_runs",
            run="09",
            model="totally_unknown_model",
        )
        assert source.run == "09"

    def test_requests_enough_buckets_for_the_fallback(self):
        # JMA is stored 6-hourly, so the late hours of D+i need previous_day{i+2}.
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", horizon=5)
        assert max(source._buckets) >= 5
        assert min(source._buckets) >= 1


# ---------------------------------------------------------------------------
# Cutoff correctness: every value comes from a run at or before "D at run:00 UTC"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window", [SPRING, FALL], ids=["spring-dst", "fall-dst"])
@pytest.mark.parametrize("timezone", ["Europe/Warsaw", "Europe/Helsinki"])
@pytest.mark.parametrize("run", ["00", "06", "12"])
@pytest.mark.parametrize("api", ["previous_runs", "single_runs"])
class TestCutoffAcrossApisTimezonesAndRuns:
    def test_every_value_comes_from_a_run_at_or_before_the_cutoff(self, api, run, timezone, window):
        start, end = window
        horizon = 2
        if api == "previous_runs":
            source = make_source(
                make_archive(step_h=1), api=api, run=run, timezone=timezone, horizon=horizon, model="icon_seamless"
            )
        else:
            source = make_source(
                make_run_archive(), api=api, run=run, timezone=timezone, horizon=horizon, model="icon_seamless"
            )
        df = source.fetch(start, end)
        cutoff = cutoff_times(df.index, run, timezone)

        for lead in range(1, horizon + 1):
            column = df[f"temperature_2m_d+{lead}"]
            present = column.notna().to_numpy()
            # Every row's target wall-clock hour exists except (at most) the one spring-forward gap hour, which
            # is correctly NaN and excluded here (see TestDstAlignment for that behaviour specifically).
            assert present.sum() >= len(df) - 1
            init = from_hours(column[present])
            assert (init <= cutoff[present]).all(), f"api={api} run={run} tz={timezone} d+{lead}: a value is late"


class TestCutoffForA3HourlyIconEuLikeModel:
    def test_run_09_is_accepted_and_the_cutoff_still_holds(self):
        # ICON-EU-like models publish more often than every 6 h; run="09" must be accepted (see
        # `known_run_hours("icon_eu")`) and the cutoff arithmetic must not assume a 6-hourly cycle.
        source = make_source(make_run_archive(), api="single_runs", run="09", model="icon_eu", horizon=1)
        df = source.fetch(*SPRING)
        cutoff = cutoff_times(df.index, "09", "Europe/Warsaw")
        present = df["temperature_2m_d+1"].notna().to_numpy()
        init = from_hours(df["temperature_2m_d+1"][present])
        assert (init <= cutoff[present]).all()


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
        start = pd.Timestamp(row, tz="Europe/Warsaw").normalize().tz_convert("UTC") - pd.Timedelta(days=1)
        df = source.fetch(start, start + pd.Timedelta(days=3))
        valid = from_hours(df.loc[pd.Timestamp(row, tz="Europe/Warsaw"), "wind_speed_10m_d+1"])[0]
        assert valid == pd.Timestamp(expected, tz="Europe/Warsaw")

    def test_d_plus_3_across_dst_keeps_wall_clock_hour(self):
        source = make_source(make_archive(step_h=1), model="icon_seamless", horizon=3)
        start = pd.Timestamp("2024-03-26", tz="UTC")
        df = source.fetch(start, start + pd.Timedelta(days=5))
        # Skip the 02:00 row whose target (2024-03-31 02:00) does not exist; it is NaN (tested below).
        df = df[df.index != pd.Timestamp("2024-03-28 02:00", tz="Europe/Warsaw")]
        valid = from_hours(df["wind_speed_10m_d+3"]).tz_convert("Europe/Warsaw")
        local = df.index.tz_convert("Europe/Warsaw")
        assert (valid.hour == local.hour).all()
        assert ((valid.normalize().tz_localize(None) - local.normalize().tz_localize(None)).days == 3).all()

    def test_spring_forward_gap_hour_is_nan_not_the_next_hour(self):
        # The row 2024-03-30 02:00 Warsaw asks for d+1 = 2024-03-31 02:00, which does not exist (spring
        # forward): it must come back NaN, not silently the value for 03:00 under this row's name.
        source = make_source(make_archive(step_h=1), model="icon_seamless", horizon=1)
        df = source.fetch(pd.Timestamp("2024-03-29", tz="UTC"), pd.Timestamp("2024-04-01", tz="UTC"))
        row = df.loc[pd.Timestamp("2024-03-30 02:00", tz="Europe/Warsaw")]
        assert np.isnan(row["wind_speed_10m_d+1"])
        assert np.isnan(row["temperature_2m_d+1"])

    def test_spring_forward_gap_hour_is_nan_under_single_runs_too(self):
        source = make_source(make_run_archive(), api="single_runs", model="icon_seamless", horizon=1)
        df = source.fetch(pd.Timestamp("2024-03-29", tz="UTC"), pd.Timestamp("2024-04-01", tz="UTC"))
        row = df.loc[pd.Timestamp("2024-03-30 02:00", tz="Europe/Warsaw")]
        assert np.isnan(row["temperature_2m_d+1"])


# ---------------------------------------------------------------------------
# Returned range, live runs and coverage checks (api="previous_runs")
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
        # This range crosses the 2024-03-31 spring-forward gap: exactly one row per lead targets the
        # nonexistent wall-clock hour and is correctly NaN (see TestDstAlignment); every other row is filled.
        assert df.notna().all(axis=1).sum() >= len(df) - 2

    def test_live_fetch_keeps_all_hours_of_the_cutoff_day(self):
        # "now" is 08:00 UTC on D, before the "10" run's cutoff (10:00 UTC): rows of D are provisional but must
        # be complete.
        source = make_source(horizon=1, run="10")
        now = pd.Timestamp("2024-01-11 08:00", tz="UTC")
        source._now = lambda: now
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-12", tz="UTC"))
        day = df[df.index.tz_convert("Europe/Warsaw").date == pd.Timestamp("2024-01-11").date()]
        assert len(day) == 24
        assert day.notna().all().all()
        init = from_hours(day["temperature_2m_d+1"])
        assert (init <= now).all()

    def test_rows_that_cannot_be_filled_are_not_returned(self):
        latest_run = pd.Timestamp("2024-01-11 00:00", tz="UTC")
        source = make_source(make_archive(step_h=1, latest_run=latest_run), model="icon_seamless", horizon=1)
        now = latest_run
        source._now = lambda: now
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-20", tz="UTC"))
        assert df.notna().any(axis=1).all()
        assert df.index.max() < pd.Timestamp("2024-01-15", tz="UTC")
        assert (from_hours(df["temperature_2m_d+1"].dropna()) <= latest_run).all()

    def test_warns_about_missing_values_with_share_and_complete_from(self, caplog):
        source = make_source(make_archive(null_columns=("wind_speed_10m",)), horizon=1)
        with caplog.at_level(logging.WARNING, logger="epftoolbox2.data.sources.open_meteo"):
            source.logger.propagate = True
            source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-12", tz="UTC"))
        assert "wind_speed_10m_d+1" in caplog.text
        assert "100.0%" in caplog.text
        assert "complete from None" in caplog.text  # never becomes complete in this window
        assert "temperature_2m_d+1:" not in caplog.text

    def test_warns_with_missing_share_by_local_hour_when_uneven(self, caplog):
        # Some columns are complete for most local hours but missing for a few, the way JMA's previous_day5-7 are
        # missing before 2021 only for the hours that need them. The overall share hides this.
        def get(url, params=None, timeout=None):
            times = pd.date_range(params["start_date"], params["end_date"] + " 23:00", freq="1h", tz="UTC")
            local_hour = times.tz_convert("Europe/Warsaw").hour
            hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
            for key in params["hourly"]:
                column, _ = key.rsplit("_previous_day", 1)
                if column == "wind_speed_10m":
                    values = [None if h in (22, 23) else float(_hours(t)) for h, t in zip(local_hour, times)]
                else:
                    values = [float(_hours(t)) for t in times]
                hourly[key] = values
            response = Mock()
            response.status_code = 200
            response.json.return_value = {"hourly": hourly}
            return response

        source = make_source(get, model="icon_seamless", horizon=1)
        with caplog.at_level(logging.WARNING, logger="epftoolbox2.data.sources.open_meteo"):
            source.logger.propagate = True
            source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-12", tz="UTC"))
        assert "wind_speed_10m_d+1" in caplog.text
        assert "missing share by local hour" in caplog.text
        assert "22h 100%" in caplog.text
        assert "23h 100%" in caplog.text
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


class TestPublicationDelayHours:
    @pytest.mark.parametrize(
        "model, expected",
        [
            ("icon_seamless", 4.0),
            ("icon_d2", 4.0),
            ("gfs_seamless", 5.0),
            ("ecmwf_ifs025", 8.0),
            ("ecmwf_ifs", 8.0),
            ("jma_seamless", 10.0),
        ],
    )
    def test_known_model_families(self, model, expected):
        assert publication_delay_hours(model) == expected

    def test_unknown_model_gets_the_conservative_default(self):
        assert publication_delay_hours("totally_unknown_model") == 10.0


class TestKnownRunHours:
    def test_known_families_report_their_cycle(self):
        assert known_run_hours("jma_seamless") == (0, 6, 12, 18)
        assert known_run_hours("icon_eu") == (0, 3, 6, 9, 12, 15, 18, 21)

    def test_unknown_model_returns_none(self):
        assert known_run_hours("totally_unknown_model") is None


# ---------------------------------------------------------------------------
# api="single_runs": exactly one run per day, no fallback
# ---------------------------------------------------------------------------


class TestSingleRunsFetch:
    def test_returns_exactly_start_to_end_exclusive(self):
        source = make_source(make_run_archive(), api="single_runs", horizon=1)
        start, end = pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC")
        df = source.fetch(start, end)
        assert df.index.min() == start
        assert df.index.max() == end - HOUR
        assert len(df) == 24

    def test_one_request_per_local_day(self):
        archive = make_run_archive()
        source = make_source(archive, api="single_runs", horizon=1)
        start, end = pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-13", tz="UTC")
        source.fetch(start, end)
        rows = pd.date_range(start.ceil("1h"), end, freq="1h", inclusive="left")
        local_days = rows.tz_convert("Europe/Warsaw").normalize().unique()
        assert len(archive.calls) == len(local_days)

    def test_target_valid_time_is_the_same_local_wall_clock_hour(self):
        source = make_source(make_run_archive(), api="single_runs", horizon=1)
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        valid = from_hours(df["wind_speed_10m_d+1"])
        assert (valid == df.index + pd.Timedelta(days=1)).all()

    def test_missing_run_leaves_nan_with_no_fallback_to_an_older_run(self):
        # api="single_runs" requests exactly the run at the cutoff; if the archive doesn't have it, the
        # affected cells are NaN — there is no silent fallback to another run (unlike the old, removed
        # OpenMeteoSingleRunsSource, which tried up to 4 older runs on the same cycle).
        day = pd.Timestamp("2024-01-11", tz="UTC")
        missing_run = day  # run="00" -> cutoff/run instant is exactly local day D's midnight UTC
        archive = make_run_archive(missing={missing_run})
        source = make_source(archive, api="single_runs", horizon=1)
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-13", tz="UTC"))
        day_rows = df[df.index.tz_convert("Europe/Warsaw").date == pd.Timestamp("2024-01-11").date()]
        assert day_rows["temperature_2m_d+1"].isna().all()
        # Only one request was made for that day (no fallback attempts).
        assert sum(1 for call in archive.calls if call["run"] == missing_run.strftime("%Y-%m-%dT%H:%M")) == 1

    def test_strict_raises_on_a_missing_run(self):
        day = pd.Timestamp("2024-01-11", tz="UTC")
        archive = make_run_archive(missing={day})
        source = make_source(archive, api="single_runs", horizon=1, strict=True)
        with pytest.raises(ValueError, match="temperature_2m_d\\+1"):
            source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-13", tz="UTC"))

    def test_prefix_applied(self):
        source = make_source(make_run_archive(), api="single_runs", horizon=1, prefix="warsaw")
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert list(df.columns) == ["warsaw_temperature_2m_d+1", "warsaw_wind_speed_10m_d+1"]

    def test_live_fetch_marks_rows_not_yet_at_the_cutoff_as_provisional(self, caplog):
        source = make_source(make_run_archive(), api="single_runs", horizon=1, run="12", model="icon_seamless")
        now = pd.Timestamp("2024-01-11 08:00", tz="UTC")
        source._now = lambda: now
        with caplog.at_level(logging.INFO, logger="epftoolbox2.data.sources.open_meteo"):
            source.logger.propagate = True
            df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-12", tz="UTC"))
        assert "cutoff after now" in caplog.text
        day = df[df.index.tz_convert("Europe/Warsaw").date == pd.Timestamp("2024-01-11").date()]
        init = from_hours(day["temperature_2m_d+1"])
        assert (init <= now).all()

    def test_fetch_validates_timestamps(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", api="single_runs")
        with pytest.raises(ValueError, match="End timestamp.*must be after start timestamp"):
            source.fetch(pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))


# ---------------------------------------------------------------------------
# Timestamps, chunking, requests (api="previous_runs")
# ---------------------------------------------------------------------------


class TestOpenMeteoSourceValidation:
    def test_fetch_validates_timestamps(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw")
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
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw")
        start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-15", tz="UTC")
        chunks = source._generate_chunks(start, end, months=3)
        assert chunks == [(start, end)]

    def test_generate_chunks_are_contiguous(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw")
        start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-07-01", tz="UTC")
        chunks = source._generate_chunks(start, end, months=3)
        assert len(chunks) >= 2
        assert chunks[0][0] == start
        assert chunks[-1][1] == end
        for (_, first_end), (second_start, _) in zip(chunks, chunks[1:]):
            assert second_start == first_end

    def test_generate_chunks_validates_months_parameter(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw")
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
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw")
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
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw")
        source.session = Mock()
        source.session.get.side_effect = requests.RequestException("Network error")
        with patch("time.sleep"):
            with pytest.raises(requests.RequestException, match="Network error"):
                source._fetch_chunk(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"))
        assert source.session.get.call_count == OpenMeteoSource.MAX_RETRIES


class TestOpenMeteoSourceParsing:
    def test_parse_response_keeps_every_bucket_value_in_place(self, sample_weather_response):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw")
        table = source._parse_response(sample_weather_response)
        assert str(table.index.tz) == "UTC"
        assert len(table) == 72
        # The fixture encodes 100 * bucket + position, so any offset error shows up here.
        assert table["temperature_2m_previous_day1"].tolist() == [100.0 + j for j in range(72)]
        assert table["temperature_2m_previous_day2"].iloc[30] == 230.0
        assert table.index[30] == pd.Timestamp("2024-01-02 06:00", tz="UTC")

    def test_parse_response_turns_nulls_into_nan(self, sample_weather_response):
        sample_weather_response["hourly"]["rain_previous_day1"][5] = None
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw")
        table = source._parse_response(sample_weather_response)
        assert np.isnan(table["rain_previous_day1"].iloc[5])

    def test_parse_response_handles_missing_hourly(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw")
        assert source._parse_response({}).empty
        assert source._parse_response({"hourly": {}}).empty

    def test_parse_response_tolerates_short_arrays(self, sample_weather_response_incomplete):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw")
        table = source._parse_response(sample_weather_response_incomplete)
        assert len(table) == 2
        assert np.isnan(table["temperature_2m_previous_day1"].iloc[1])


class TestCacheConfigAndSerialization:
    def test_cache_config_has_schema_version_and_key_parameters(self):
        config = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw").get_cache_config()
        assert config["schema_version"] >= 3
        for key in ("api", "run", "timezone", "horizon", "model", "columns"):
            assert key in config

    @pytest.mark.parametrize(
        "change",
        [
            {"run": "06"},
            {"api": "single_runs"},
            {"timezone": "Europe/London"},
            {"model": "icon_seamless"},
        ],
    )
    def test_output_affecting_parameters_change_the_cache_config(self, change):
        base = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw").get_cache_config()
        other = OpenMeteoSource(
            latitude=52.23, longitude=21.01, **{"timezone": "Europe/Warsaw", **change}
        ).get_cache_config()
        assert base != other

    def test_strict_does_not_change_the_cache_config(self):
        base = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw").get_cache_config()
        strict = OpenMeteoSource(
            latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", strict=True
        ).get_cache_config()
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
            run="06",
            strict=True,
        )
        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(source).save(path)
        loaded = DataPipeline.load(path).sources[0]
        assert isinstance(loaded, OpenMeteoSource)
        assert loaded.get_cache_config() == source.get_cache_config()
        assert loaded.strict is True

    def test_yaml_round_trip_single_runs(self, tmp_path):
        source = OpenMeteoSource(
            latitude=52.23,
            longitude=21.01,
            horizon=7,
            model="ecmwf_ifs",
            columns=["temperature_2m"],
            prefix="warsaw",
            timezone="Europe/Berlin",
            api="single_runs",
            run="12",
        )
        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(source).save(path)
        loaded = DataPipeline.load(path).sources[0]
        assert isinstance(loaded, OpenMeteoSource)
        assert loaded.get_cache_config() == source.get_cache_config()
        assert loaded.api == "single_runs"


@pytest.mark.integration
class TestOpenMeteoSourceIntegration:
    """Integration tests that hit the real API"""

    def test_fetch_weather_data_warsaw(self):
        source = OpenMeteoSource(latitude=52.2297, longitude=21.0122, timezone="Europe/Warsaw", horizon=3)
        start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC")
        df = source.fetch(start, end)
        assert len(df) == 48
        assert df.index.min() == start and df.index.max() < end

    def test_fetch_weather_data_limited_columns(self):
        source = OpenMeteoSource(
            latitude=52.2297,
            longitude=21.0122,
            timezone="Europe/Warsaw",
            horizon=2,
            columns=["temperature_2m", "wind_speed_10m"],
        )
        df = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"))
        assert sorted(df.columns) == sorted(
            ["temperature_2m_d+1", "temperature_2m_d+2", "wind_speed_10m_d+1", "wind_speed_10m_d+2"]
        )

    def test_fetch_long_period(self):
        source = OpenMeteoSource(
            latitude=52.2297, longitude=21.0122, timezone="Europe/Warsaw", horizon=5, columns=["temperature_2m"]
        )
        df = source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-06-01", tz="UTC"))
        assert len(df) == (pd.Timestamp("2024-06-01") - pd.Timestamp("2024-01-01")) / HOUR

    def test_fetch_weather_data_single_runs(self):
        # The Single Runs archive starts per model: ECMWF IFS from 2024-03-14, JMA (the default) only from 2026-05-13.
        source = OpenMeteoSource(
            latitude=52.2297,
            longitude=21.0122,
            timezone="Europe/Warsaw",
            api="single_runs",
            model="ecmwf_ifs",
            horizon=3,
        )
        start, end = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-01-03", tz="UTC")
        df = source.fetch(start, end)
        assert len(df) == 48
        assert df.index.min() == start and df.index.max() < end
