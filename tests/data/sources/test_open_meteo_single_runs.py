import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from epftoolbox2.data.sources.open_meteo_single_runs import OpenMeteoSingleRunsSource
from epftoolbox2.pipelines import DataPipeline

TZ = "Europe/Warsaw"
ISSUE = "10:00"
DELAY_H = 10
INTERVAL_H = 6
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
HOUR = pd.Timedelta(hours=1)


def _hours(ts):
    return (ts - EPOCH) / HOUR


def from_hours(values):
    return pd.to_datetime(np.atleast_1d(np.asarray(values, dtype=float)) * 3600, unit="s", utc=True)


def issue_times(index, issue=ISSUE, tz=TZ):
    """Issue instant (UTC) of the local day each row belongs to, computed independently of the source."""
    hh, mm = (int(x) for x in issue.split(":"))
    out = []
    for t in index:
        day = t.tz_convert(tz).date()
        out.append(pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hh, minute=mm, tz=tz))
    return pd.DatetimeIndex(out).tz_convert("UTC")


def make_run_archive(missing=(), short_runs=None, null_columns_by_run=None):
    """Return a fake `session.get` that answers Single Runs requests from a synthetic archive.

    `missing`: run inits (UTC Timestamps) that the archive does not have.
    `short_runs`: {run init: max lead in hours} beyond which that run's values are null.
    `null_columns_by_run`: {run init: {column names}} that are always null for that run (a gap in the archive,
        not a short run), so tests can check that it is not silently backfilled from an older run.

    Values encode, per cell: `temperature_2m` -> hours since epoch of the run's own init (identifies which run
    served the cell); other columns -> hours since epoch of the valid time (identifies the target alignment).
    """
    missing = set(missing)
    short_runs = short_runs or {}
    null_columns_by_run = null_columns_by_run or {}
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
        max_lead = short_runs.get(run)
        null_cols = null_columns_by_run.get(run, set())
        hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
        for key in params["hourly"]:
            if key in null_cols:
                hourly[key] = [None] * n
                continue
            values = []
            for t in times:
                if max_lead is not None and (t - run) > pd.Timedelta(hours=max_lead):
                    values.append(None)
                elif key == "temperature_2m":
                    values.append(float(_hours(run)))
                else:
                    values.append(float(_hours(t)))
            hourly[key] = values
        response.json.return_value = {"hourly": hourly}
        return response

    get.calls = calls
    return get


def make_source(archive=None, **kwargs):
    kwargs.setdefault("columns", ["temperature_2m", "wind_speed_100m"])
    kwargs.setdefault("timezone", TZ)
    source = OpenMeteoSingleRunsSource(latitude=52.23, longitude=21.01, **kwargs)
    source.session = Mock()
    source.session.get.side_effect = archive or make_run_archive()
    source.console.quiet = True
    return source


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self):
        source = OpenMeteoSingleRunsSource(latitude=52.23, longitude=21.01, timezone=TZ)
        assert source.model == "ecmwf_ifs"
        assert source.horizon == 7
        assert source.run_interval_h == 6
        assert source.derived == []
        assert source.strict is False

    @pytest.mark.parametrize("horizon", [1, 14])
    def test_accepts_horizon_up_to_14(self, horizon):
        source = OpenMeteoSingleRunsSource(latitude=52.23, longitude=21.01, timezone=TZ, horizon=horizon)
        assert source.horizon == horizon

    def test_rejects_horizon_above_14(self):
        with pytest.raises(ValueError, match="Horizon must be at most 14"):
            OpenMeteoSingleRunsSource(latitude=52.23, longitude=21.01, horizon=15)

    @pytest.mark.parametrize("interval", [0, -6, 5, 7, 24.0, True])
    def test_rejects_run_interval_not_dividing_24(self, interval):
        with pytest.raises(ValueError, match="run_interval_h"):
            OpenMeteoSingleRunsSource(latitude=52.23, longitude=21.01, timezone=TZ, run_interval_h=interval)

    @pytest.mark.parametrize("interval", [1, 2, 3, 4, 6, 8, 12, 24])
    def test_accepts_divisors_of_24(self, interval):
        source = OpenMeteoSingleRunsSource(latitude=52.23, longitude=21.01, timezone=TZ, run_interval_h=interval)
        assert source.run_interval_h == interval

    def test_timezone_is_required(self):
        # A default used to compute the issue time in the wrong zone for markets outside CET, leaking future
        # runs. Construction with no zone is allowed (a DataPipeline may still supply one later), but fetching
        # without one anywhere raises.
        source = OpenMeteoSingleRunsSource(latitude=52.23, longitude=21.01)
        assert source.timezone is None
        with pytest.raises(ValueError, match="timezone is required"):
            source.fetch(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"))

    def test_rejects_invalid_timezone(self):
        with pytest.raises(ValueError, match="timezone"):
            OpenMeteoSingleRunsSource(latitude=52.23, longitude=21.01, timezone="Mars/Olympus")

    def test_rejects_unknown_derived_name(self):
        with pytest.raises(ValueError, match="derived must be a subset"):
            OpenMeteoSingleRunsSource(latitude=52.23, longitude=21.01, timezone=TZ, derived=["not_a_column"])


# ---------------------------------------------------------------------------
# Run selection: no run published after the issue time
# ---------------------------------------------------------------------------


class TestRunSelection:
    START = pd.Timestamp("2024-03-25", tz="UTC")
    END = pd.Timestamp("2024-04-05", tz="UTC")

    def test_run_init_plus_delay_never_exceeds_issue_time(self):
        # Spans a DST switch (2024-03-31 in Warsaw); every row of every lead must be served by a run published
        # (init + publication_delay_h) at or before that row's own issue instant.
        source = make_source(horizon=2)
        df = source.fetch(self.START, self.END)
        issue = issue_times(df.index)
        for lead in (1, 2):
            init = from_hours(df[f"temperature_2m_d+{lead}"])
            assert df[f"temperature_2m_d+{lead}"].notna().all()
            late = init + pd.Timedelta(hours=DELAY_H) > issue
            assert not late.any(), f"d+{lead}: {late.sum()} rows use a run published after the issue time"

    def test_uses_the_run_floored_to_the_cycle(self):
        source = make_source(horizon=1)
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-12", tz="UTC"))
        init = from_hours(df["temperature_2m_d+1"])
        issue = issue_times(df.index)
        expected = (issue - pd.Timedelta(hours=DELAY_H)).floor(f"{INTERVAL_H}h")
        assert (init == expected).all()

    def test_target_valid_time_is_the_same_local_wall_clock_hour(self):
        source = make_source(horizon=1)
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        valid = from_hours(df["wind_speed_100m_d+1"])
        assert (valid == df.index + pd.Timedelta(days=1)).all()

    def test_falls_back_to_older_run_when_ideal_run_is_missing(self):
        row = pd.Timestamp("2024-01-10 12:00", tz="UTC")
        issue = issue_times(pd.DatetimeIndex([row]))[0]
        ideal = (issue - pd.Timedelta(hours=DELAY_H)).floor(f"{INTERVAL_H}h")
        archive = make_run_archive(missing={ideal})
        source = make_source(archive, horizon=1)
        df = source.fetch(row, row + HOUR)
        init = from_hours(df["temperature_2m_d+1"])[0]
        assert init == ideal - pd.Timedelta(hours=INTERVAL_H)
        # The fallback run must still respect the issue-time bound.
        assert init + pd.Timedelta(hours=DELAY_H) <= issue

    def test_gives_up_after_max_fallback_runs_leaves_nan(self):
        row = pd.Timestamp("2024-01-10 12:00", tz="UTC")
        issue = issue_times(pd.DatetimeIndex([row]))[0]
        ideal = (issue - pd.Timedelta(hours=DELAY_H)).floor(f"{INTERVAL_H}h")
        all_missing = {ideal - i * pd.Timedelta(hours=INTERVAL_H) for i in range(0, 6)}
        archive = make_run_archive(missing=all_missing)
        source = make_source(archive, horizon=1)
        df = source.fetch(row, row + HOUR)
        assert df.empty or df["temperature_2m_d+1"].isna().all()

    def test_short_run_falls_back_for_the_lead_it_does_not_cover(self):
        row = pd.Timestamp("2024-01-10 12:00", tz="UTC")
        issue = issue_times(pd.DatetimeIndex([row]))[0]
        ideal = (issue - pd.Timedelta(hours=DELAY_H)).floor(f"{INTERVAL_H}h")
        # The ideal run reaches exactly as far as d+1's target valid time (no DST in this window, so d+i is
        # row + i calendar days), but not d+2's.
        target_1 = row + pd.Timedelta(days=1)
        max_lead_h = int((target_1 - ideal) / HOUR)
        archive = make_run_archive(short_runs={ideal: max_lead_h})
        source = make_source(archive, horizon=2)
        df = source.fetch(row, row + HOUR)
        init_1 = from_hours(df["temperature_2m_d+1"])[0]
        init_2 = from_hours(df["temperature_2m_d+2"])[0]
        assert init_1 == ideal
        assert init_2 == ideal - pd.Timedelta(hours=INTERVAL_H)
        assert df["temperature_2m_d+1"].notna().all()
        assert df["temperature_2m_d+2"].notna().all()

    def test_null_within_a_covered_run_is_not_backfilled_from_an_older_run(self):
        # A gap in one variable of an otherwise-available run is a real gap in the archive: falling back to an
        # older run would hide it behind stale data, so it must stay NaN.
        row = pd.Timestamp("2024-01-10 12:00", tz="UTC")
        issue = issue_times(pd.DatetimeIndex([row]))[0]
        ideal = (issue - pd.Timedelta(hours=DELAY_H)).floor(f"{INTERVAL_H}h")
        archive = make_run_archive(null_columns_by_run={ideal: {"wind_speed_100m"}})
        source = make_source(archive, horizon=1)
        df = source.fetch(row, row + HOUR)
        assert df["temperature_2m_d+1"].notna().all()
        assert df["wind_speed_100m_d+1"].isna().all()

    def test_fallback_stops_once_an_available_run_adds_nothing(self):
        # Every lead is already settled by the ideal run; older runs must not be requested.
        source = make_source(horizon=1)
        source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-10 01:00", tz="UTC"))
        assert source.session.get.call_count == 1


# ---------------------------------------------------------------------------
# Returned range, prefix, derived columns, coverage
# ---------------------------------------------------------------------------


class TestReturnedRange:
    def test_returns_exactly_start_to_end_exclusive(self):
        source = make_source(horizon=1)
        start, end = pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC")
        df = source.fetch(start, end)
        assert df.index.min() == start
        assert df.index.max() == end - HOUR
        assert len(df) == 24
        assert str(df.index.tz) == "UTC"

    def test_prefix_applied(self):
        source = make_source(horizon=1, prefix="warsaw")
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert list(df.columns) == ["warsaw_temperature_2m_d+1", "warsaw_wind_speed_100m_d+1"]

    def test_derived_columns_off_by_default(self):
        source = make_source(horizon=1)
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert "hdd_d+1" not in df.columns

    def test_derived_columns_computed_when_requested(self):
        source = make_source(horizon=1, derived=["hdd", "cdd", "wind_speed_100m_cubed"])
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        for name in ("hdd_d+1", "cdd_d+1", "wind_speed_100m_cubed_d+1"):
            assert name in df.columns

    def test_strict_raises_on_missing_values(self):
        row = pd.Timestamp("2024-01-10 12:00", tz="UTC")
        issue = issue_times(pd.DatetimeIndex([row]))[0]
        ideal = (issue - pd.Timedelta(hours=DELAY_H)).floor(f"{INTERVAL_H}h")
        all_missing = {ideal - i * pd.Timedelta(hours=INTERVAL_H) for i in range(0, 6)}
        archive = make_run_archive(missing=all_missing)
        source = make_source(archive, horizon=1, strict=True)
        with pytest.raises(ValueError, match="temperature_2m_d\\+1"):
            source.fetch(row, row + HOUR)

    def test_live_fetch_marks_rows_not_yet_at_issue_time_as_provisional(self, caplog):
        source = make_source(horizon=1)
        now = pd.Timestamp("2024-01-11 08:00", tz=TZ).tz_convert("UTC")
        source._now = lambda: now
        with caplog.at_level(logging.INFO, logger="epftoolbox2.data.sources.open_meteo"):
            source.logger.propagate = True
            df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-12", tz="UTC"))
        assert "issue time after now" in caplog.text
        day = df[df.index.tz_convert(TZ).date == pd.Timestamp("2024-01-11").date()]
        init = from_hours(day["temperature_2m_d+1"])
        assert (init + pd.Timedelta(hours=DELAY_H) <= now).all()


# ---------------------------------------------------------------------------
# Fetch chunk / request plumbing
# ---------------------------------------------------------------------------


class TestFetchRun:
    def test_unavailable_run_returns_none(self):
        source = make_source(make_run_archive(missing={pd.Timestamp("2024-01-10", tz="UTC")}), horizon=1)
        assert source._fetch_run(pd.Timestamp("2024-01-10", tz="UTC")) is None

    def test_available_run_is_parsed(self):
        source = make_source(horizon=1)
        table = source._fetch_run(pd.Timestamp("2024-01-10", tz="UTC"))
        assert isinstance(table, pd.DataFrame)
        assert "temperature_2m" in table.columns

    def test_retries_on_request_exception(self):
        source = make_source(horizon=1)
        source.session.get.side_effect = requests.RequestException("boom")
        with patch("time.sleep"):
            with pytest.raises(requests.RequestException):
                source._fetch_run(pd.Timestamp("2024-01-10", tz="UTC"))
        assert source.session.get.call_count == OpenMeteoSingleRunsSource.MAX_RETRIES

    def test_fetch_validates_timestamps(self):
        source = OpenMeteoSingleRunsSource(latitude=52.23, longitude=21.01, timezone=TZ)
        with pytest.raises(ValueError, match="End timestamp.*must be after start timestamp"):
            source.fetch(pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))


# ---------------------------------------------------------------------------
# Cache config and serialization
# ---------------------------------------------------------------------------


class TestCacheConfigAndSerialization:
    def test_cache_config_has_schema_version_and_key_parameters(self):
        config = OpenMeteoSingleRunsSource(latitude=52.23, longitude=21.01, timezone=TZ).get_cache_config()
        assert config["schema_version"] >= 1
        for key in ("run_interval_h", "issue_time", "publication_delay_h", "timezone", "horizon", "model", "columns"):
            assert key in config
        assert "derived" not in config

    @pytest.mark.parametrize(
        "change",
        [
            {"issue_time": "11:30"},
            {"publication_delay_h": 6},
            {"run_interval_h": 12},
            {"timezone": "Europe/London"},
            {"derived": ["hdd"]},
        ],
    )
    def test_output_affecting_parameters_change_the_cache_config(self, change):
        base = OpenMeteoSingleRunsSource(latitude=52.23, longitude=21.01, timezone=TZ).get_cache_config()
        other = OpenMeteoSingleRunsSource(
            latitude=52.23, longitude=21.01, **{"timezone": TZ, **change}
        ).get_cache_config()
        assert base != other

    def test_strict_does_not_change_the_cache_config(self):
        base = OpenMeteoSingleRunsSource(latitude=52.23, longitude=21.01, timezone=TZ).get_cache_config()
        strict = OpenMeteoSingleRunsSource(latitude=52.23, longitude=21.01, timezone=TZ, strict=True).get_cache_config()
        assert base == strict

    def test_yaml_round_trip(self, tmp_path):
        source = OpenMeteoSingleRunsSource(
            latitude=52.23,
            longitude=21.01,
            horizon=4,
            model="icon_seamless",
            columns=["temperature_2m"],
            prefix="warsaw",
            timezone="Europe/Berlin",
            issue_time="11:30",
            publication_delay_h=4,
            run_interval_h=3,
            derived=["hdd"],
            strict=True,
        )
        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(source).save(path)
        loaded = DataPipeline.load(path).sources[0]
        assert isinstance(loaded, OpenMeteoSingleRunsSource)
        assert loaded.get_cache_config() == source.get_cache_config()
        assert loaded.strict is True


@pytest.mark.integration
class TestOpenMeteoSingleRunsSourceIntegration:
    """Integration tests that hit the real API"""

    def test_fetch_weather_data_warsaw(self):
        source = OpenMeteoSingleRunsSource(latitude=52.2297, longitude=21.0122, timezone=TZ, horizon=3)
        start, end = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-01-03", tz="UTC")
        df = source.fetch(start, end)
        assert len(df) == 48
        assert df.index.min() == start and df.index.max() < end
