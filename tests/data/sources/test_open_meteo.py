import logging
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from epftoolbox2.data.sources.open_meteo import (
    DEFAULT_SINGLE_RUNS_HORIZON,
    OpenMeteoSource,
    known_run_hours,
    max_previous_day,
    max_previous_runs_horizon,
    previous_run_lengths,
    publication_delay_hours,
    select_buckets,
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
# Previous Runs API (verified against the live API and the Single Runs API, see open_meteo.py): `previous_dayK`
# holds, at each stored point P, the latest run initialised at or before P - 24K h *that reaches P* (the shorter
# 06/18 UTC runs of some models drop out of the deep buckets). A model stored hourly (GFS, ICON, ECMWF IFS 9 km)
# gives an hourly value at T from the point at T. A model stored every `step_h` hours (JMA 6, ECMWF IFS 0.25 3)
# is interpolated when read from the four points around T, so an off-grid hourly value depends on the run behind
# the point up to two steps after T. The archive encodes, per cell:
#   temperature_2m -> latest run init (hours since epoch) among the stored points the value depends on
#   wind_speed_10m -> the valid time T (hours since epoch)
# It is written as plain loops over timestamps, independent of the vectorised `newest_run_init` it tests.

# Every 6-hourly run reaches every point: the default when no model is named.
UNLIMITED_6H_RUNS = {0: 10**6, 6: 10**6, 12: 10**6, 18: 10**6}


def _hours(ts):
    return (ts - EPOCH) / HOUR


# Instants below are integer hours since the epoch, which keeps the plain loops fast enough for the tests.


def _run_init(point, k, lengths):
    """Init of the run `previous_dayK` holds at stored `point`; None if no run within three days reaches it."""
    init = point - 24 * k
    for _ in range(72):
        reach = lengths.get(init % 24)
        if reach is not None and point - init <= reach:
            return init
        init -= 1
    return None


def _latest_init(valid, k, step_h, lengths):
    """Newest init among the stored points the hourly value at `valid` depends on; None if one has no run."""
    if step_h == 1 or valid % step_h == 0:
        points = [valid]
    else:
        first = valid - valid % step_h
        points = [first + n * step_h for n in (-1, 0, 1, 2)]
    inits = [_run_init(point, k, lengths) for point in points]
    return None if None in inits else max(inits)


def archive_facts(model):
    """(stored step, runs by init hour -> length, deepest bucket) the synthetic archive uses for `model`."""
    lengths = previous_run_lengths(model) or {hour: 10**6 for hour in range(24)}
    return stored_step_hours(model, 1), lengths, max_previous_day(model)


def make_archive(step_h=1, latest_run=None, null_columns=(), max_bucket=7, model=None, lengths=None):
    """Return a fake `session.get` that answers Previous Runs requests from the synthetic archive.

    With `model`, its stored step, runs and deepest bucket come from the source's tables (the facts verified
    against the API). Otherwise the runs are `lengths` (init hour -> forecast length, default: 6-hourly runs that
    reach every point), written down by the test itself rather than read from the source's tables, and the given
    `step_h` / `max_bucket`.
    """
    lengths = UNLIMITED_6H_RUNS if lengths is None else lengths
    if model is not None:
        step_h, lengths, max_bucket = archive_facts(model)
    latest = None if latest_run is None else int(_hours(latest_run))
    calls = []

    def get(url, params=None, timeout=None):
        calls.append(params)
        times = pd.date_range(params["start_date"], params["end_date"] + " 23:00", freq="1h", tz="UTC")
        hours = [int(_hours(t)) for t in times]
        hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
        for key in params["hourly"]:
            column, k = key.rsplit("_previous_day", 1)
            k = int(k)
            values = []
            for hour in hours:
                init = None if k > max_bucket else _latest_init(hour, k, step_h, lengths)
                if init is None or column in null_columns or (latest is not None and init > latest):
                    values.append(None)
                else:
                    values.append(float(init if column == "temperature_2m" else hour))
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
        assert source.horizon is None
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

    def test_horizon_above_the_limit_raises_naming_the_limit_and_single_runs(self):
        # jma_seamless at run 00 in Warsaw can serve d+1..d+6 (see TestMaxPreviousRunsHorizon). A full week cannot
        # be served without a run initialised after the cutoff.
        with pytest.raises(ValueError, match=r"horizon=7 is above the largest horizon without look-ahead, 6") as error:
            OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", horizon=7)
        assert "single_runs" in str(error.value)
        assert "horizon<=6" in str(error.value)

    def test_horizon_above_the_limit_raises_for_a_model_with_a_lower_limit(self):
        # ICON has no previous_day7, so d+6 cannot be served and the limit is 5 at run 00.
        with pytest.raises(ValueError, match=r"horizon=6 is above the largest horizon without look-ahead, 5"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", model="icon_seamless", horizon=6)

    def test_horizon_above_the_limit_raises_at_fetch_when_the_zone_comes_from_the_pipeline(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, model="icon_seamless", horizon=6)
        DataPipeline(sources=[source], timezone="Europe/Warsaw")
        with pytest.raises(ValueError, match="above the largest horizon without look-ahead, 5"):
            source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))

    def test_horizon_equal_to_the_limit_is_accepted(self):
        source = OpenMeteoSource(
            latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", model="icon_seamless", horizon=5
        )
        assert source.horizon == 5

    @pytest.mark.parametrize("horizon", [1.5, "5", True])
    def test_horizon_must_be_a_positive_integer_or_none(self, horizon):
        with pytest.raises(ValueError, match="Horizon must be a positive integer"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", horizon=horizon)

    def test_no_horizon_is_leak_free_when_even_d_plus_1_needs_a_later_run(self):
        # ICON-D2 has previous_day1 only, and at run 00 d+1's late hours need previous_day2.
        with pytest.raises(ValueError, match=r"d\+1 cannot be served without look-ahead"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", model="icon_d2")

    def test_later_run_does_not_allow_a_full_week(self):
        # d+7 would need previous_day7 for hours whose run is initialised after D 12:00 UTC.
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", run="06", horizon=6)
        assert source.horizon == 6
        with pytest.raises(ValueError, match="above the largest horizon without look-ahead, 6"):
            OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", run="06", horizon=7)

    def test_single_runs_horizon_none_means_the_modest_default(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", api="single_runs")
        assert source.horizon is None
        assert source._effective_horizon() == DEFAULT_SINGLE_RUNS_HORIZON == 5

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
        # A default zone would compute the cutoff in the wrong zone for markets outside CET (e.g. a market's local
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
        # The late hours of D+5 already need previous_day6, and one spare bucket lets live runs fall back further.
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", horizon=5)
        assert max(source._buckets) == 7
        assert min(source._buckets) == 1

    def test_never_requests_buckets_the_model_does_not_have(self):
        # ICON has no previous_day7: with horizon 5 the buckets stop at 6 instead of adding a spare 7.
        source = OpenMeteoSource(
            latitude=52.23, longitude=21.01, timezone="Europe/Warsaw", model="icon_seamless", horizon=5
        )
        assert max(source._buckets) == 6


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
# The largest horizon without look-ahead (api="previous_runs"), per model, run and market zone
# ---------------------------------------------------------------------------

WARSAW = "Europe/Warsaw"


def oracle_limit(model, timezone, run, days):
    """The largest horizon whose every local hour, on every issue day in `days`, is served leak-free.

    Plain datetime arithmetic and the archive's own run logic, independent of the source's vectorised code: for
    each local hour of issue day D and lead i, the smallest previous_dayK (K >= i, within what the model has) whose
    newest influencing run is at or before D at `run`:00 UTC.
    """
    from datetime import datetime, timezone as utc
    from zoneinfo import ZoneInfo

    zone = ZoneInfo(timezone)
    step_h, lengths, deepest = archive_facts(model)

    def instant(day, hour):
        naive = datetime(day.year, day.month, day.day, hour)
        aware = naive.replace(tzinfo=zone)  # fold=0: the first of an ambiguous hour
        if aware.astimezone(utc.utc).astimezone(zone).replace(tzinfo=None) != naive:
            return None  # the hour does not exist (spring forward)
        return int(_hours(pd.Timestamp(aware.astimezone(utc.utc))))

    limit = 0
    for lead in range(1, 8):
        for day in days:
            cutoff = int(_hours(pd.Timestamp(year=day.year, month=day.month, day=day.day, tz="UTC"))) + int(run)
            for hour in range(24):
                if instant(day, hour) is None:
                    continue
                valid = instant(day + pd.Timedelta(days=lead), hour)
                if valid is None:
                    continue
                inits = (_latest_init(valid, k, step_h, lengths) for k in range(lead, deepest + 1))
                if not any(init is not None and init <= cutoff for init in inits):
                    return limit
        limit = lead
    return limit


# The days around both DST switches of 2024 in each zone, and a quiet day of each offset.
ORACLE_DAYS = {
    "Europe/Warsaw": [pd.Timestamp(d) for d in ("2024-01-15", "2024-03-30", "2024-03-31", "2024-04-01", "2024-07-15")]
    + [pd.Timestamp(d) for d in ("2024-10-26", "2024-10-27", "2024-10-28")],
    "America/New_York": [
        pd.Timestamp(d) for d in ("2024-01-15", "2024-03-09", "2024-03-10", "2024-03-11", "2024-07-15")
    ]
    + [pd.Timestamp(d) for d in ("2024-11-02", "2024-11-03", "2024-11-04")],
}


class TestMaxPreviousRunsHorizon:
    @pytest.mark.parametrize(
        "model, run, expected",
        [
            # previous_dayK at T is the run 24K h older, and the runs are 6-hourly: d+6 is served by previous_day7
            # (the run of D-1 18z at the latest), d+7 never fully.
            ("gfs_seamless", "00", 6),
            ("gfs_seamless", "06", 6),
            ("gfs_seamless", "12", 6),
            ("gfs_seamless", "18", 7),
            # ICON has no previous_day7: d+6 cannot be served before the cutoff reaches D 12:00 UTC (then
            # previous_day6, from the 12z run, covers it).
            ("icon_seamless", "00", 5),
            ("icon_seamless", "06", 5),
            ("icon_seamless", "12", 6),
            ("icon_global", "00", 5),
            # JMA is stored every 6 h and interpolated when read, so its last hours reach into points two steps
            # later. Beyond 132 h only its 00 and 12 UTC runs exist, which keeps d+6 clean at run 00.
            ("jma_seamless", "00", 6),
            ("jma_seamless", "06", 6),
            ("jma_seamless", "12", 6),
            ("jma_seamless", "18", 6),
            # ECMWF IFS 0.25 is stored every 3 h, the 9 km model hourly. Both only have 00 and 12 UTC runs beyond 144 h.
            ("ecmwf_ifs025", "00", 6),
            ("ecmwf_ifs025", "06", 6),
            ("ecmwf_ifs", "00", 6),
            ("ecmwf_ifs", "06", 6),
            ("ecmwf_ifs", "12", 7),
            # A model this library has no table for: hourly runs and a 6 h store, the strictest assumptions.
            ("some_new_model", "00", 5),
            ("some_new_model", "06", 6),
        ],
    )
    def test_limit_in_warsaw(self, model, run, expected):
        assert max_previous_runs_horizon(model, WARSAW, run) == expected

    def test_icon_variants_stop_at_the_deepest_bucket_they_have(self):
        assert max_previous_runs_horizon("icon_eu", WARSAW, "00") == 3
        assert max_previous_runs_horizon("icon_eu", WARSAW, "18") == 4
        assert max_previous_runs_horizon("icon_d2", WARSAW, "00") == 0
        assert max_previous_runs_horizon("icon_d2", WARSAW, "21") == 1

    def test_limit_depends_on_the_market_zone(self):
        # New York's day starts five hours later in UTC: JMA's last hours of D+6 are served by points of D 06z.
        assert max_previous_runs_horizon("jma_seamless", "America/New_York", "00") == 5
        assert max_previous_runs_horizon("jma_seamless", WARSAW, "00") == 6
        assert max_previous_runs_horizon("gfs_seamless", "America/New_York", "00") == 6

    def test_a_full_week_is_never_served_by_the_runs_of_the_morning(self):
        # Whatever the model, d+7 has hours that previous_day7 only holds from a run after D 12:00 UTC.
        for model in ("jma_seamless", "gfs_seamless", "ecmwf_ifs", "ecmwf_ifs025", "icon_seamless", "some_new_model"):
            for run in ("00", "03", "06", "09"):
                assert max_previous_runs_horizon(model, WARSAW, run) <= 6

    @pytest.mark.parametrize("timezone", list(ORACLE_DAYS))
    @pytest.mark.parametrize("run", ["00", "06", "12", "18"])
    @pytest.mark.parametrize(
        "model", ["jma_seamless", "gfs_seamless", "icon_seamless", "ecmwf_ifs025", "ecmwf_ifs", "some_new_model"]
    )
    def test_limit_matches_a_brute_force_check_of_every_local_hour(self, model, run, timezone):
        assert max_previous_runs_horizon(model, timezone, run) == oracle_limit(
            model, timezone, run, ORACLE_DAYS[timezone]
        )


class TestDefaultHorizonIsTheLimit:
    @pytest.mark.parametrize("run", ["00", "06", "12"])
    @pytest.mark.parametrize("model", ["jma_seamless", "gfs_seamless", "icon_seamless", "ecmwf_ifs025"])
    def test_default_horizon_equals_the_limit(self, model, run):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=WARSAW, model=model, run=run)
        limit = max_previous_runs_horizon(model, WARSAW, run)
        assert source.horizon is None
        assert source._effective_horizon() == limit
        assert source.get_cache_config()["horizon"] == limit

    def test_default_fetch_returns_exactly_d_plus_1_to_the_limit(self):
        source = make_source(make_archive(model="icon_seamless"), model="icon_seamless")
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-12", tz="UTC"))
        limit = max_previous_runs_horizon("icon_seamless", WARSAW, "00")
        assert limit == 5
        assert sorted({c.rsplit("_d+", 1)[1] for c in df.columns}, key=int) == [str(i) for i in range(1, limit + 1)]

    def test_default_horizon_fills_every_hour_of_the_last_day(self):
        # The model's missing buckets (ICON's previous_day7) must never be requested: no NaN from a null bucket.
        source = make_source(make_archive(model="icon_seamless"), model="icon_seamless", strict=True)
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-14", tz="UTC"))
        assert df.notna().all().all()

    def test_default_horizon_with_the_zone_taken_from_the_pipeline(self):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, model="icon_seamless", columns=["temperature_2m"])
        DataPipeline(sources=[source], timezone=WARSAW)
        assert source._effective_horizon() == 5
        assert source.get_cache_config()["horizon"] == 5
        source.session = Mock()
        source.session.get.side_effect = make_archive(model="icon_seamless")
        source.console.quiet = True
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-12", tz="UTC"))
        assert "temperature_2m_d+5" in df.columns and "temperature_2m_d+6" not in df.columns

    def test_default_and_explicit_horizon_share_a_cache_entry(self):
        default = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=WARSAW, model="gfs_seamless")
        explicit = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=WARSAW, model="gfs_seamless", horizon=6)
        shorter = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=WARSAW, model="gfs_seamless", horizon=5)
        assert default.get_cache_config() == explicit.get_cache_config()
        assert default.get_cache_config() != shorter.get_cache_config()

    def test_single_runs_default_fetches_five_leads(self):
        source = make_source(make_run_archive(), api="single_runs")
        df = source.fetch(pd.Timestamp("2024-01-10", tz="UTC"), pd.Timestamp("2024-01-11", tz="UTC"))
        assert sorted({c.rsplit("_d+", 1)[1] for c in df.columns}, key=int) == ["1", "2", "3", "4", "5"]

    def test_yaml_round_trip_keeps_the_default(self, tmp_path):
        source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=WARSAW, model="gfs_seamless")
        path = tmp_path / "dp.yaml"
        DataPipeline().add_source(source).save(path)
        loaded = DataPipeline.load(path).sources[0]
        assert loaded.horizon is None
        assert loaded.get_cache_config() == source.get_cache_config()


# Spring and fall switches of New York in 2024 (the Europe/Warsaw ones are SPRING and FALL above).
NY_SPRING = (pd.Timestamp("2024-03-05", tz="UTC"), pd.Timestamp("2024-03-16", tz="UTC"))
NY_FALL = (pd.Timestamp("2024-10-30", tz="UTC"), pd.Timestamp("2024-11-10", tz="UTC"))


@pytest.mark.parametrize(
    "timezone, window",
    [
        ("Europe/Warsaw", SPRING),
        ("Europe/Warsaw", FALL),
        ("America/New_York", NY_SPRING),
        ("America/New_York", NY_FALL),
    ],
    ids=["warsaw-spring", "warsaw-fall", "new-york-spring", "new-york-fall"],
)
@pytest.mark.parametrize("run", ["00", "06"])
@pytest.mark.parametrize("model", ["jma_seamless", "gfs_seamless", "icon_seamless", "ecmwf_ifs025", "ecmwf_ifs"])
class TestLastDayOfTheDefaultHorizonAroundDst:
    def test_every_local_hour_up_to_d_plus_limit_comes_from_a_run_at_or_before_the_cutoff(
        self, model, run, timezone, window
    ):
        source = make_source(make_archive(model=model), model=model, run=run, timezone=timezone)
        df = source.fetch(*window)
        limit = max_previous_runs_horizon(model, timezone, run)
        assert f"temperature_2m_d+{limit + 1}" not in df.columns
        cutoff = cutoff_times(df.index, run, timezone)

        for lead in range(1, limit + 1):
            column = df[f"temperature_2m_d+{lead}"]
            present = column.notna().to_numpy()
            # Every local hour of D+lead exists except (at most) the one spring-forward gap hour, which is NaN.
            assert present.sum() >= len(df) - 1, f"{model} run={run} {timezone}: d+{lead} has holes"
            init = from_hours(column[present])
            assert (init <= cutoff[present]).all(), f"{model} run={run} {timezone}: a value of d+{lead} is late"


class TestTheArchiveWouldCatchALeak:
    def test_one_lead_beyond_the_limit_does_reach_past_the_cutoff(self):
        # The same synthetic archive, read one lead past the limit by hand: some hour is served by a run after the
        # cutoff, so the check above is not vacuous.
        model, run = "gfs_seamless", "00"
        limit = max_previous_runs_horizon(model, WARSAW, run)
        step_h, lengths, deepest = archive_facts(model)
        rows = pd.date_range("2024-03-25", "2024-04-05", freq="1h", tz="UTC")
        cutoff = cutoff_times(rows, run, WARSAW)
        late = 0
        for row, cut in zip(rows, cutoff):
            local = row.tz_convert(WARSAW).tz_localize(None) + pd.Timedelta(days=limit + 1)
            valid = local.tz_localize(WARSAW, ambiguous=True, nonexistent="NaT")
            if pd.isna(valid):
                continue
            inits = [_latest_init(int(_hours(valid)), k, step_h, lengths) for k in range(limit + 1, deepest + 1)]
            late += not any(init <= _hours(cut) for init in inits if init is not None)
        assert late > 0


# Runs behind `icon_seamless` inside the ICON-D2 area (e.g. Berlin), as verified against the Single Runs API in
# 2026-10: `previous_day1` at T holds the newest run of a 3-hourly grid at or before T - 24 h. The 3/9/15/21 UTC
# runs are ICON-D2's alone (48 h), the 00/12 UTC ones reach 180 h and the 06/18 UTC ones 120 h.
ICON_SEAMLESS_D2_AREA_RUNS = {0: 180, 3: 48, 6: 120, 9: 48, 12: 180, 15: 48, 18: 120, 21: 48}


class TestIconSeamlessRunGrid:
    def test_cell_after_a_3_hourly_d2_run_does_not_get_bucket_1(self):
        # Berlin, run="00", issue day 2026-10-04: d+1 at 05:00 CEST is 03:00Z on 10-05. In the D2 area
        # previous_day1 there holds the 03Z run of 10-04, started 3 h after the cutoff (live: 10.2 C, the D2 03Z
        # run, not the 00Z run's 10.8 C). The 06Z global run does not make it better: it is later still.
        valid = pd.date_range("2026-10-05 03:00", periods=3, freq="1h", tz="UTC")
        cutoff = pd.DatetimeIndex([pd.Timestamp("2026-10-04 00:00", tz="UTC")] * len(valid))
        assert select_buckets("icon_seamless", valid, cutoff, 1, 7).tolist() == [2, 2, 2]
        # The cells served by the 00Z run itself stay on bucket 1.
        early = pd.date_range("2026-10-05 00:00", periods=3, freq="1h", tz="UTC")
        assert select_buckets("icon_seamless", early, cutoff, 1, 7).tolist() == [1, 1, 1]

    def test_runs_are_3_hourly_and_only_icon_seamless_has_the_d2_runs(self):
        assert sorted(previous_run_lengths("icon_seamless")) == list(range(0, 24, 3))
        assert previous_run_lengths("icon_seamless")[3] == 48
        # ICON global and ICON-EU were checked live on the same cells: they only have 00/06/12/18 UTC runs.
        assert sorted(previous_run_lengths("icon_global")) == [0, 6, 12, 18]
        assert sorted(previous_run_lengths("icon_eu")) == [0, 6, 12, 18]
        assert sorted(previous_run_lengths("icon_d2")) == list(range(0, 24, 3))

    @pytest.mark.parametrize("run", ["00", "06", "12"])
    def test_no_value_comes_from_a_run_after_the_cutoff_in_the_d2_area(self, run):
        timezone = "Europe/Berlin"
        archive = make_archive(max_bucket=6, lengths=ICON_SEAMLESS_D2_AREA_RUNS)
        source = make_source(archive, model="icon_seamless", run=run, timezone=timezone)
        df = source.fetch(*SPRING)
        limit = max_previous_runs_horizon("icon_seamless", timezone, run)
        assert limit >= 5
        cutoff = cutoff_times(df.index, run, timezone)
        for lead in range(1, limit + 1):
            column = df[f"temperature_2m_d+{lead}"]
            present = column.notna().to_numpy()
            assert present.sum() >= len(df) - 1
            assert (from_hours(column[present]) <= cutoff[present]).all(), f"d+{lead} has a value from a late run"

    def test_the_default_horizon_is_still_5_at_run_00(self):
        assert max_previous_runs_horizon("icon_seamless", "Europe/Berlin", "00") == 5
        assert max_previous_runs_horizon("icon_seamless", WARSAW, "00") == 5


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


class TestPreviousRunsTables:
    @pytest.mark.parametrize("bucket", [1, 4, 7])
    @pytest.mark.parametrize(
        "model, expected",
        [
            # What temporal_resolution="native" returns for each previous_dayK: JMA GSM and ECMWF IFS 0.25 are stored
            # at their own 6 h / 3 h points, GFS, ICON and ECMWF IFS 9 km hourly (verified against the live API).
            ("jma_seamless", 6),
            ("ecmwf_ifs025", 3),
            ("ecmwf_ifs04", 3),
            ("ecmwf_ifs", 1),
            ("icon_seamless", 1),
            ("icon_global", 1),
            ("icon_d2", 1),
            ("gfs_seamless", 1),
            ("gfs_global", 1),
            ("totally_unknown_model", 6),
        ],
    )
    def test_stored_step(self, model, bucket, expected):
        assert stored_step_hours(model, bucket) == expected

    def test_ecmwf_ifs04_is_not_matched_by_the_hourly_ecmwf_ifs_prefix(self):
        # ecmwf_ifs04 (ECMWF IFS 0.4 deg) is stored 3-hourly. Its name starts with "ecmwf_ifs", so without its own
        # entry it would match the hourly ecmwf_ifs (9 km) one instead, understating how far the interpolated
        # hourly value can reach into runs not yet published.
        assert stored_step_hours("ecmwf_ifs04", 1) == 3
        assert stored_step_hours("ecmwf_ifs", 1) == 1

    @pytest.mark.parametrize(
        "model, expected",
        [
            ("jma_seamless", 7),
            ("gfs_seamless", 7),
            ("ecmwf_ifs025", 7),
            ("ecmwf_ifs", 7),
            ("icon_seamless", 6),
            ("icon_global", 6),
            ("icon_eu", 4),
            ("icon_d2", 1),
            ("totally_unknown_model", 7),
        ],
    )
    def test_deepest_bucket(self, model, expected):
        assert max_previous_day(model) == expected

    def test_shorter_runs_of_icon_jma_and_ecmwf_drop_out_of_the_deep_buckets(self):
        # The 06 and 18 UTC runs of these models are shorter than the 00 and 12 UTC ones.
        for model, short in (("icon_global", 120), ("ecmwf_ifs", 144), ("jma_seamless", 132)):
            lengths = previous_run_lengths(model)
            assert lengths[6] == lengths[18] == short < lengths[0] == lengths[12]
        assert set(previous_run_lengths("gfs_seamless").values()) == {384}

    def test_unknown_model_has_no_run_table(self):
        assert previous_run_lengths("totally_unknown_model") is None


class TestPublicationDelayHours:
    @pytest.mark.parametrize(
        "model, expected",
        [
            ("icon_seamless", 4.5),
            ("icon_d2", 4.5),
            ("gfs_seamless", 6.5),
            ("ecmwf_ifs025", 9.0),
            ("ecmwf_ifs", 9.0),
            ("jma_seamless", 9.6),
        ],
    )
    def test_known_model_families(self, model, expected):
        assert publication_delay_hours(model) == expected

    def test_unknown_model_gets_the_conservative_default(self):
        assert publication_delay_hours("totally_unknown_model") >= 10.0

    @pytest.mark.parametrize(
        "model, measured",
        [("icon_seamless", 4.30), ("gfs_seamless", 5.54), ("ecmwf_ifs025", 8.55), ("jma_seamless", 9.53)],
    )
    def test_each_family_is_later_than_the_lag_measured_on_2026_10_11(self, model, measured):
        """The table errs on the late side of Open-Meteo's own metadata."""
        assert publication_delay_hours(model) > measured


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
        # affected cells are NaN — there is no silent fallback to another run.
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
        assert config["schema_version"] >= 5
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
