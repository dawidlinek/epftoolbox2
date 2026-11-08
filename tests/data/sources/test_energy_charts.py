"""EnergyChartsSource: parsing, start-of-period stamps, DST days, the rate limit, the zone licence, the disk cache and the
availability rules. Nothing here touches the network: a fake session answers in the shape of the real API (unix seconds that
mark the start of each period, `end` inclusive), with synthetic values."""

import logging
import time

import pandas as pd
import pytest
import requests

import epftoolbox2.data.sources.energy_charts as ec
from epftoolbox2.data.sources import EnergyChartsSource
from epftoolbox2.data.sources.energy_charts import EnergyChartsApiError, snake_case
from epftoolbox2.information_set import Availability, InformationSet, source_rules
from epftoolbox2.pipelines import DataPipeline

BERLIN = "Europe/Berlin"
SPRING_DAY = (pd.Timestamp("2025-03-29 23:00", tz="UTC"), pd.Timestamp("2025-03-30 22:00", tz="UTC"))  # 23 h
AUTUMN_DAY = (pd.Timestamp("2025-10-25 22:00", tz="UTC"), pd.Timestamp("2025-10-26 23:00", tz="UTC"))  # 25 h
POWER_NAMES = ["Load", "Residual load", "Solar", "Wind onshore", "Fossil brown coal / lignite", "Hydro Run-of-River"]


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, reason="OK"):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.reason = reason
        self.text = "" if payload is None else str(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def stamps(params, step_seconds):
    """The grid points of the real API between `start` and `end` (both unix seconds, `end` inclusive)."""
    first = int(params["start"])
    last = int(params["end"])
    return list(range(first, last + 1, step_seconds))


class FakeApi:
    """Answers like api.energy-charts.info for the endpoints the source reads; `script` is a list of responses given first."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls = []

    def close(self):
        pass

    def get(self, url, params=None, headers=None, timeout=None):
        endpoint = url.rsplit("/", 1)[1]
        self.calls.append((endpoint, dict(params), dict(headers or {})))
        if self.script:
            return self.script.pop(0)
        return FakeResponse(200, self.payload(endpoint, params))

    def payload(self, endpoint, params):
        if endpoint == "public_power_forecast":
            step = 900
            seconds = stamps(params, step)
            vintage = {"day-ahead": 1.0, "intraday": 2.0, "current": 3.0}[params["forecast_type"]]
            kind = {"solar": 10.0, "wind_onshore": 20.0, "wind_offshore": 30.0, "load": 40.0}[params["production_type"]]
            return {
                "unix_seconds": seconds,
                "forecast_values": [kind + vintage / 10 + (s // 900) % 1000 / 1000 for s in seconds],
                "production_type": params["production_type"],
                "forecast_type": params["forecast_type"],
                "deprecated": False,
            }
        if endpoint == "price":
            step = 3600 if params["bzn"] == "CH" else 900
            seconds = stamps(params, step)
            return {
                "license_info": "CC BY 4.0 (creativecommons.org/licenses/by/4.0) from Bundesnetzagentur | SMARD.de",
                "unix_seconds": seconds,
                "price": [50.0 + (s // step) % 100 for s in seconds],
                "unit": "EUR / MWh",
                "deprecated": False,
            }
        if endpoint == "public_power":
            seconds = stamps(params, 900)
            return {
                "unix_seconds": seconds,
                "production_types": [
                    {"name": name, "data": [100.0 * (i + 1) + (s // 900) % 7 for s in seconds]}
                    for i, name in enumerate(POWER_NAMES)
                ],
                "deprecated": False,
            }
        seconds = stamps(params, 900)  # cbet, cbpf
        return {
            "unix_seconds": seconds,
            "countries": [{"name": name, "data": [1.5 for _ in seconds]} for name in ("Austria", "France", "sum")],
            "deprecated": False,
        }


class Clock:
    """A fake monotonic clock whose sleep only advances it."""

    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def fresh_buckets(monkeypatch):
    """The rate limit is process-wide: each test starts with full buckets, and nothing may really sleep."""
    monkeypatch.setattr(ec, "_BUCKETS", {})

    def refuse(seconds):
        raise AssertionError(f"a test slept for {seconds} s of real time")

    monkeypatch.setattr(time, "sleep", refuse)


def make(tmp_path, api=None, clock=None, **kwargs):
    kwargs.setdefault("cache_dir", str(tmp_path / "cache"))
    kwargs.setdefault("prefix", "")
    source = EnergyChartsSource(**kwargs)
    source.session = api if api is not None else FakeApi()
    clock = clock or Clock()
    source._clock = clock
    source._sleep = clock.sleep
    source._now = lambda: pd.Timestamp("2026-11-08 09:00", tz="UTC")
    return source


# ---------------------------------------------------------------------------
# Parsing and stamps
# ---------------------------------------------------------------------------


def test_forecast_columns_are_named_by_quantity_and_vintage(tmp_path):
    source = make(
        tmp_path,
        type=["forecast"],
        forecast_types=["day-ahead", "intraday", "current"],
        production_types=["solar", "load"],
    )
    df = source.fetch(*SPRING_DAY)
    assert set(df.columns) == {
        "solar_forecast_day_ahead",
        "solar_forecast_intraday",
        "solar_forecast_current",
        "load_forecast_day_ahead",
    }
    # the vintages differ, in the order of the fake API's offsets
    assert (df["solar_forecast_current"] > df["solar_forecast_intraday"]).all()
    assert (df["solar_forecast_intraday"] > df["solar_forecast_day_ahead"]).all()


def test_load_is_not_asked_for_other_vintages_than_day_ahead(tmp_path):
    api = FakeApi()
    source = make(tmp_path, api, type=["forecast"], forecast_types=["day-ahead", "intraday"], production_types=["load"])
    df = source.fetch(*SPRING_DAY)
    assert list(df.columns) == ["load_forecast_day_ahead"]
    assert [(c[1]["production_type"], c[1]["forecast_type"]) for c in api.calls] == [("load", "day-ahead")]


def test_load_alone_needs_day_ahead():
    with pytest.raises(ValueError, match="day-ahead"):
        EnergyChartsSource("de", type=["forecast"], forecast_types=["intraday"], production_types=["load"])


def test_timestamps_are_utc_and_mark_the_start_of_the_period(tmp_path):
    api = FakeApi()
    source = make(tmp_path, api, type=["forecast"], production_types=["solar"])
    start = pd.Timestamp("2025-06-10 22:00", tz="UTC")
    df = source.fetch(start, start + pd.Timedelta(hours=1))
    assert df.index.tz is not None and str(df.index.tz) == "UTC"
    assert list(df.index) == [start + pd.Timedelta(minutes=15 * i) for i in range(4)]  # 4 quarter-hours: [start, end)
    endpoint, params, _ = api.calls[0]
    assert endpoint == "public_power_forecast"
    assert params["start"] == str(int(start.timestamp())) and params["end"] == str(int(start.timestamp()) + 3600 - 1)


def test_power_series_get_snake_case_names_and_can_be_selected(tmp_path):
    df = make(tmp_path, type=["power"]).fetch(*SPRING_DAY)
    assert list(df.columns) == [
        "actual_load",
        "actual_residual_load",
        "actual_solar",
        "actual_wind_onshore",
        "actual_fossil_brown_coal_lignite",
        "actual_hydro_run_of_river",
    ]
    picked = make(tmp_path, type=["power"], power="solar, load").fetch(*SPRING_DAY)
    assert list(picked.columns) == ["actual_solar", "actual_load"]
    with pytest.raises(ValueError, match="not in the de power data"):
        make(tmp_path, type=["power"], power=["geothermal"]).fetch(*SPRING_DAY)


def test_snake_case():
    assert snake_case("Fossil brown coal / lignite") == "fossil_brown_coal_lignite"
    assert snake_case("Hydro Run-of-River") == "hydro_run_of_river"


def test_cross_border_flows(tmp_path):
    df = make(tmp_path, type=["cbet", "cbpf"]).fetch(*SPRING_DAY)
    assert set(df.columns) == {f"{kind}_{who}" for kind in ("cbet", "cbpf") for who in ("austria", "france", "sum")}


def test_null_values_become_nan(tmp_path):
    payload = {
        "unix_seconds": [1743289200, 1743290100],
        "forecast_values": [None, 12.5],
        "production_type": "solar",
        "forecast_type": "day-ahead",
        "deprecated": False,
    }
    source = make(tmp_path, FakeApi([FakeResponse(200, payload)]), type=["forecast"], production_types=["solar"])
    df = source.fetch(pd.Timestamp("2025-03-29 23:00", tz="UTC"), pd.Timestamp("2025-03-29 23:30", tz="UTC"))
    assert df["solar_forecast_day_ahead"].isna().tolist() == [True, False]


def test_an_empty_answer_is_an_empty_frame(tmp_path):
    empty = {"unix_seconds": [], "forecast_values": [], "deprecated": False}
    source = make(tmp_path, FakeApi([FakeResponse(200, empty)]), type=["forecast"], production_types=["solar"])
    assert source.fetch(*SPRING_DAY).empty


def test_a_series_that_does_not_match_its_stamps_raises(tmp_path):
    payload = {"unix_seconds": [1743289200, 1743290100], "forecast_values": [1.0], "deprecated": False}
    source = make(tmp_path, FakeApi([FakeResponse(200, payload)]), type=["forecast"], production_types=["solar"])
    with pytest.raises(EnergyChartsApiError, match="values for 2 stamps"):
        source.fetch(*SPRING_DAY)


def test_prefix(tmp_path):
    df = make(tmp_path, type=["price"], prefix="de").fetch(*SPRING_DAY)
    assert list(df.columns) == ["de_price_day_ahead"]


def test_naive_bounds_are_read_as_utc(tmp_path):
    source = make(tmp_path, type=["price"])
    df = source.fetch(pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-01 02:00"))
    assert len(df) == 8 and df.index[0] == pd.Timestamp("2025-06-01", tz="UTC")
    with pytest.raises(ValueError, match="must be after"):
        source.fetch(pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-01"))


def test_sub_range_gives_the_values_of_the_larger_range(tmp_path):
    source = make(tmp_path, type=["forecast", "price"], production_types=["solar"], use_cache=False)
    whole = source.fetch(*SPRING_DAY)
    part_start = SPRING_DAY[0] + pd.Timedelta(hours=7, minutes=15)
    part = source.fetch(part_start, part_start + pd.Timedelta(hours=5))
    pd.testing.assert_frame_equal(part, whole.loc[part.index])


def test_a_long_range_is_fetched_in_chunks_of_a_year_and_joined(tmp_path):
    api = FakeApi()
    source = make(tmp_path, api, type=["price"], bzn="CH", use_cache=False)
    start = pd.Timestamp("2023-12-31 23:00", tz="UTC")
    df = source.fetch(start, start + pd.Timedelta(days=800))
    assert len(api.calls) == 3  # 366 + 366 + 68 days
    assert len(df) == 800 * 24 and df.index.is_monotonic_increasing and not df.index.has_duplicates
    assert df.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()


# ---------------------------------------------------------------------------
# DST: both switches, at 15 min and 1 h
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "day, rows_15min, rows_hourly",
    [(SPRING_DAY, 92, 23), (AUTUMN_DAY, 100, 25)],
    ids=["spring", "autumn"],
)
def test_a_dst_day_has_every_period_the_clock_has(tmp_path, day, rows_15min, rows_hourly):
    fine = make(tmp_path, type=["forecast", "price", "power"], production_types=["solar"]).fetch(*day)
    assert len(fine) == rows_15min and fine.notna().all().all()
    assert fine.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    hourly = make(tmp_path, type=["price"], bzn="CH").fetch(*day)
    assert len(hourly) == rows_hourly
    # the local day: first stamp is local midnight, and the repeated hour appears twice, the skipped one not at all
    local = fine.index.tz_convert(BERLIN)
    assert local[0].hour == 0 and local[0].minute == 0 and local[-1].hour == 23 and local[-1].minute == 45
    assert len(set(local)) == rows_15min


def test_spring_day_skips_two_oclock_and_autumn_day_repeats_it(tmp_path):
    spring = make(tmp_path, type=["price"], bzn="CH").fetch(*SPRING_DAY).index.tz_convert(BERLIN)
    assert 2 not in set(spring.hour)
    autumn = make(tmp_path, type=["price"], bzn="CH").fetch(*AUTUMN_DAY).index.tz_convert(BERLIN)
    assert list(autumn.hour).count(2) == 2


def test_a_pipeline_in_the_market_zone_keeps_both_switch_days(tmp_path):
    from epftoolbox2.data.transformers import TimezoneTransformer

    source = make(tmp_path, type=["price"], bzn="CH")
    pipeline = DataPipeline(timezone=BERLIN).add_source(source).add_transformer(TimezoneTransformer())
    df = pipeline.run("2025-03-29", "2025-03-31")
    assert str(df.index.tz) == BERLIN
    assert df.groupby(df.index.date).size().to_dict()[pd.Timestamp("2025-03-30").date()] == 23
    df = pipeline.run("2025-10-25", "2025-10-27")
    assert df.groupby(df.index.date).size().to_dict()[pd.Timestamp("2025-10-26").date()] == 25


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------


def test_a_429_waits_retry_after_then_retries(tmp_path):
    api = FakeApi(
        [FakeResponse(429, {"detail": "slow down"}, headers={"Retry-After": "37"}, reason="Too Many Requests")]
    )
    clock = Clock()
    source = make(tmp_path, api, clock, type=["price"])
    df = source.fetch(*SPRING_DAY)
    assert len(df) == 92
    assert len(api.calls) == 2
    assert clock.slept and clock.slept[0] == pytest.approx(37.0)
    assert sum(clock.slept) >= 37.0


def test_retry_after_as_an_http_date(tmp_path):
    soon = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=20)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    api = FakeApi([FakeResponse(429, {}, headers={"Retry-After": soon})])
    clock = Clock()
    make(tmp_path, api, clock, type=["price"]).fetch(*SPRING_DAY)
    assert 15 <= clock.slept[0] <= 21


def test_a_429_without_retry_after_waits_a_minute(tmp_path):
    api = FakeApi([FakeResponse(429, {})])
    clock = Clock()
    make(tmp_path, api, clock, type=["price"]).fetch(*SPRING_DAY)
    assert clock.slept[0] == pytest.approx(60.0)


def test_a_retry_after_beyond_the_cap_raises_instead_of_hanging(tmp_path):
    api = FakeApi([FakeResponse(429, {}, headers={"Retry-After": "86400"})])
    with pytest.raises(EnergyChartsApiError, match="Retry-After 86400"):
        make(tmp_path, api, type=["price"]).fetch(*SPRING_DAY)


def test_a_429_that_never_ends_raises_after_the_retries(tmp_path):
    api = FakeApi([FakeResponse(429, {}, headers={"Retry-After": "5"}) for _ in range(20)])
    with pytest.raises(EnergyChartsApiError, match="failed after 7 attempts"):
        make(tmp_path, api, type=["price"]).fetch(*SPRING_DAY)
    assert len(api.calls) == 7


def test_client_side_throttle_spends_the_burst_then_waits_30_seconds_per_request(tmp_path):
    api = FakeApi()
    clock = Clock()
    source = make(
        tmp_path,
        api,
        clock,
        type=["forecast"],
        forecast_types=["day-ahead", "intraday"],
        production_types=["solar", "wind_onshore", "wind_offshore", "load"],
        use_cache=False,
    )
    source.fetch(*SPRING_DAY)
    assert len(api.calls) == 7
    assert sum(clock.slept) == pytest.approx(3 * 30.0)  # burst of 4 free, then one every 30 s (2 a minute)
    assert all(s == pytest.approx(30.0) for s in clock.slept)


def test_price_has_a_burst_of_two(tmp_path):
    api = FakeApi()
    clock = Clock()
    source = make(tmp_path, api, clock, type=["price"], use_cache=False)
    for _ in range(3):
        source.fetch(*SPRING_DAY)
    assert sum(clock.slept) == pytest.approx(30.0)


def test_endpoints_are_throttled_separately(tmp_path):
    clock = Clock()
    source = make(tmp_path, FakeApi(), clock, type=["forecast", "power", "price", "cbet"], production_types=["solar"])
    source.fetch(*SPRING_DAY)
    assert clock.slept == []


def test_the_limit_is_shared_by_sources_in_a_process(tmp_path):
    clock = Clock()
    first = make(tmp_path, FakeApi(), clock, type=["price"], use_cache=False)
    second = make(tmp_path, FakeApi(), clock, type=["price"], bzn="CH", use_cache=False)
    first.fetch(*SPRING_DAY)
    second.fetch(*SPRING_DAY)
    assert clock.slept == []
    first.fetch(*SPRING_DAY)
    assert sum(clock.slept) == pytest.approx(30.0)


def test_the_bucket_refills_with_time(tmp_path):
    clock = Clock()
    source = make(tmp_path, FakeApi(), clock, type=["price"], use_cache=False)
    source.fetch(*SPRING_DAY)
    source.fetch(*SPRING_DAY)
    clock.now += 60.0  # two tokens back
    source.fetch(*SPRING_DAY)
    source.fetch(*SPRING_DAY)
    assert clock.slept == []


def test_requests_carry_a_generic_user_agent_and_no_key(tmp_path):
    api = FakeApi()
    make(tmp_path, api, type=["price"]).fetch(*SPRING_DAY)
    _, params, headers = api.calls[0]
    assert headers["User-Agent"].startswith("epftoolbox2")
    assert "@" not in headers["User-Agent"]
    assert set(params) == {"bzn", "start", "end"}


def test_server_errors_are_retried_and_other_errors_raise(tmp_path):
    clock = Clock()
    api = FakeApi([FakeResponse(503, {}, reason="Service Unavailable")])
    assert len(make(tmp_path, api, clock, type=["price"], use_cache=False).fetch(*SPRING_DAY)) == 92
    assert clock.slept == [pytest.approx(4.0)]

    bad = FakeApi([FakeResponse(400, None, reason="Bad Request")])
    with pytest.raises(EnergyChartsApiError, match="HTTP 400"):
        make(tmp_path, bad, type=["price"], use_cache=False).fetch(*SPRING_DAY)


def test_network_errors_are_retried(tmp_path):
    class Flaky(FakeApi):
        def get(self, *args, **kwargs):
            if not self.calls:
                self.calls.append(("price", {}, {}))
                raise requests.ConnectionError("reset")
            return super().get(*args, **kwargs)

    clock = Clock()
    assert len(make(tmp_path, Flaky(), clock, type=["price"]).fetch(*SPRING_DAY)) == 92
    assert len(clock.slept) == 1


def test_a_body_that_is_not_json_raises(tmp_path):
    api = FakeApi([FakeResponse(200, None)])
    with pytest.raises(EnergyChartsApiError, match="not JSON"):
        make(tmp_path, api, type=["price"]).fetch(*SPRING_DAY)


# ---------------------------------------------------------------------------
# Zone licence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "zone", ["AT", "BE", "CH", "CZ", "DE-LU", "DE-AT-LU", "DK1", "DK2", "FR", "HU", "IT-North", "NL"]
)
def test_zones_under_cc_by_are_accepted(zone):
    assert EnergyChartsSource("de", type=["price"], bzn=zone)._zone == zone


@pytest.mark.parametrize("zone", ["NO2", "PL", "SE4", "SI"])
def test_the_rest_of_the_cc_by_list(zone):
    assert EnergyChartsSource("de", type=["price"], bzn=zone)._zone == zone


@pytest.mark.parametrize("zone", ["RO", "BG", "ES", "PT", "NO1", "SE3", "IT-Sicily", "GR"])
def test_other_zones_are_refused_unless_the_user_opts_in(zone):
    with pytest.raises(ValueError, match="private and internal use"):
        EnergyChartsSource("de", type=["price"], bzn=zone)


def test_opting_in_to_a_restricted_zone_warns(caplog):
    logger = logging.getLogger("epftoolbox2.data.sources.energy_charts")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING):
            source = EnergyChartsSource("ro", type=["price"], allow_restricted_prices=True)
    finally:
        logger.removeHandler(caplog.handler)
    assert source._zone == "RO"
    assert "private and internal use" in caplog.text


def test_the_country_decides_the_zone_only_when_it_has_one():
    assert EnergyChartsSource("de", type=["price"])._zone == "DE-LU"
    assert EnergyChartsSource("pl", type=["price"])._zone == "PL"
    for country in ("dk", "it", "se", "no"):
        with pytest.raises(ValueError, match="several bidding zones"):
            EnergyChartsSource(country, type=["price"])
    assert EnergyChartsSource("dk", type=["price"], bzn="dk2")._zone == "DK2"


def test_restricted_prices_are_not_checked_when_no_price_is_asked():
    EnergyChartsSource("ro", type=["forecast", "power"])  # forecasts and actuals are CC BY 4.0 for any country


def test_unknown_values_are_refused():
    with pytest.raises(ValueError, match="Unknown country"):
        EnergyChartsSource("xx")
    with pytest.raises(ValueError, match="Invalid type"):
        EnergyChartsSource("de", type=["gas"])
    with pytest.raises(ValueError, match="Invalid forecast type"):
        EnergyChartsSource("de", forecast_types=["week-ahead"])
    with pytest.raises(ValueError, match="Invalid production type"):
        EnergyChartsSource("de", production_types=["wind"])
    with pytest.raises(ValueError, match="Unknown bidding zone"):
        EnergyChartsSource("de", type=["price"], bzn="XX")


def test_the_price_request_names_the_api_spelling_of_the_zone(tmp_path):
    api = FakeApi()
    make(tmp_path, api, type=["price"], bzn="de_lu").fetch(*SPRING_DAY)
    assert api.calls[0][1]["bzn"] == "DE-LU"


# ---------------------------------------------------------------------------
# Disk cache and the pipeline cache key
# ---------------------------------------------------------------------------


def test_a_settled_answer_is_read_from_disk_the_second_time(tmp_path):
    api = FakeApi()
    source = make(tmp_path, api, type=["forecast", "price"], production_types=["solar"])
    first = source.fetch(*SPRING_DAY)
    assert len(api.calls) == 2
    again = make(tmp_path, api, type=["forecast", "price"], production_types=["solar"]).fetch(*SPRING_DAY)
    assert len(api.calls) == 2  # nothing new asked, not even by a new source
    pd.testing.assert_frame_equal(first, again)
    assert len(list((tmp_path / "cache").glob("*.json"))) == 2


def test_use_cache_false_asks_every_time_and_writes_nothing(tmp_path):
    api = FakeApi()
    source = make(tmp_path, api, type=["price"], use_cache=False)
    source.fetch(*SPRING_DAY)
    source.fetch(*SPRING_DAY)
    assert len(api.calls) == 2 and not (tmp_path / "cache").exists()


def test_the_last_two_days_are_not_cached(tmp_path):
    api = FakeApi()
    source = make(tmp_path, api, type=["price"])  # now = 2026-11-08 09:00
    recent = pd.Timestamp("2026-11-06 12:00", tz="UTC")
    source.fetch(recent, recent + pd.Timedelta(hours=2))
    source.fetch(recent, recent + pd.Timedelta(hours=2))
    assert len(api.calls) == 2
    old = pd.Timestamp("2026-11-05 12:00", tz="UTC")
    source.fetch(old, old + pd.Timedelta(hours=2))
    source.fetch(old, old + pd.Timedelta(hours=2))
    assert len(api.calls) == 3


def test_a_broken_cache_file_is_a_miss(tmp_path):
    api = FakeApi()
    make(tmp_path, api, type=["price"]).fetch(*SPRING_DAY)
    (path,) = (tmp_path / "cache").glob("*.json")
    path.write_text("{not json")
    assert len(make(tmp_path, api, type=["price"]).fetch(*SPRING_DAY)) == 92
    assert len(api.calls) == 2


def test_default_cache_dir_follows_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert ec.default_cache_dir() == tmp_path / "epftoolbox2" / "energy-charts"


def test_cache_config_has_every_parameter_that_changes_the_output():
    base = EnergyChartsSource("de", type=["forecast", "price"]).get_cache_config()
    assert base["source_type"] == "energy_charts" and base["schema_version"] == ec.SCHEMA_VERSION
    variants = [
        EnergyChartsSource("at", type=["forecast", "price"]),
        EnergyChartsSource("de", type=["forecast"]),
        EnergyChartsSource("de", type=["forecast", "price"], bzn="AT"),
        EnergyChartsSource("de", type=["forecast", "price"], forecast_types=["intraday"]),
        EnergyChartsSource("de", type=["forecast", "price"], production_types=["solar"]),
        EnergyChartsSource("de", type=["forecast", "price"], prefix="de"),
    ]
    configs = [v.get_cache_config() for v in variants]
    assert all(c != base for c in configs) and len({repr(sorted(c.items())) for c in configs}) == len(configs)
    assert EnergyChartsSource("de", type=["price", "forecast"]).get_cache_config() == base  # order is no parameter
    # where the answers are kept does not change them
    assert (
        EnergyChartsSource("de", type=["forecast", "price"], cache_dir="/x", use_cache=False).get_cache_config() == base
    )
    power = EnergyChartsSource("de", type=["power"], power=["solar"]).get_cache_config()
    assert power != EnergyChartsSource("de", type=["power"]).get_cache_config()


def test_the_pipeline_cache_serves_the_second_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    api = FakeApi()
    source = make(tmp_path, api, type=["price"], bzn="CH", use_cache=False)
    pipeline = DataPipeline(timezone=BERLIN).add_source(source)
    first = pipeline.run("2025-03-29", "2025-03-31", cache=True)
    calls = len(api.calls)
    second = pipeline.run("2025-03-29", "2025-03-31", cache=True)
    assert len(api.calls) == calls
    pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# Availability rules
# ---------------------------------------------------------------------------


def EC(*args, **kwargs):
    """A source whose columns are unprefixed, so that the rule tests name them as the API's columns."""
    kwargs.setdefault("prefix", "")
    return EnergyChartsSource(*args, **kwargs)


HOURLY_BERLIN = pd.date_range("2025-03-01", periods=24 * 40, freq="h", tz=BERLIN)


def forecast_info(issue="11:45"):
    source = EC(type=["forecast"], forecast_types=["day-ahead", "intraday", "current"])
    return InformationSet(issue, BERLIN).with_source_rules([source])


def late(names, info=None, start="2025-03-10", end="2025-03-14", horizon=1, index=HOURLY_BERLIN):
    info = info or forecast_info()
    expanded = {(p, h): list(names) for p in range(24) for h in range(1, horizon + 1)}
    found, unknown = info.violations(expanded, "price", "1h", index, start, end)
    return {v.predictor for v in found}, unknown


def test_an_1145_issue_sees_todays_day_ahead_and_intraday_forecasts_but_never_current():
    names = [f"{x}_forecast_{v}" for x in ("solar", "wind_onshore", "wind_offshore") for v in ("day_ahead", "intraday")]
    names += ["load_forecast_day_ahead"]
    assert late(names) == (set(), set())  # published 18:00 on D-1 and 08:00 on D, both before 11:45 on D
    found, _ = late([f"{x}_forecast_current" for x in ("solar", "wind_onshore", "wind_offshore")])
    assert found == {"solar_forecast_current", "wind_onshore_forecast_current", "wind_offshore_forecast_current"}


def test_tomorrows_forecasts_are_late_except_nothing_of_current():
    found, _ = late(
        ["solar_forecast_day_ahead_d+1", "solar_forecast_intraday_d+1", "solar_forecast_current_d+1"], horizon=1
    )
    assert found == {"solar_forecast_day_ahead_d+1", "solar_forecast_intraday_d+1", "solar_forecast_current_d+1"}


def test_current_is_late_at_every_issue_time_of_the_day_it_is_for():
    for issue in ("00:30", "11:45", "23:30"):
        found, _ = late(["wind_onshore_forecast_current"], forecast_info(issue))
        assert found == {"wind_onshore_forecast_current"}, issue


def test_current_is_refused_by_availability_raise():
    from epftoolbox2.information_set import InformationSetError

    expanded = {(p, 1): ["solar_forecast_current"] for p in range(24)}
    with pytest.raises(InformationSetError, match="solar_forecast_current"):
        InformationSet("11:45", BERLIN).check(expanded, "price", "1h", HOURLY_BERLIN, "2025-03-10", "2025-03-12")


def test_vintage_classes():
    rules = source_rules(
        [EC("de", type=["forecast", "power", "price"], forecast_types=["day-ahead", "intraday", "current"])]
    )
    assert rules["solar_forecast_day_ahead"].vintage == "snapshot"
    assert rules["solar_forecast_intraday"].vintage == "snapshot"
    assert rules["solar_forecast_current"].vintage == "latest_revision"
    assert rules["actual_*"].vintage == "latest_revision"
    assert rules["price_day_ahead"].vintage == "snapshot"


def test_rule_values():
    rules = source_rules(
        [EC("de", type=["forecast", "power", "price"], forecast_types=["day-ahead", "intraday", "current"])]
    )
    assert (rules["wind_offshore_forecast_day_ahead"].time, rules["wind_offshore_forecast_day_ahead"].days_before) == (
        "18:00",
        1,
    )
    assert (rules["wind_offshore_forecast_intraday"].time, rules["wind_offshore_forecast_intraday"].days_before) == (
        "08:00",
        0,
    )
    assert rules["actual_*"] == Availability.realised("1h", note=rules["actual_*"].note, vintage="latest_revision")
    assert (rules["price_day_ahead"].time, rules["price_day_ahead"].days_before) == ("13:00", 1)
    assert "load_forecast_intraday" not in rules and "load_forecast_current" not in rules


def test_actuals_are_known_an_hour_after_the_period():
    info = InformationSet("11:45", BERLIN).with_source_rules([EC("de", type=["power"])])
    assert late(["actual_load_d-1", "actual_solar_d-1"], info)[0] == set()
    assert late(["actual_load"], info)[0] == {"actual_load"}  # D's own value is not out at 11:45 on D


def test_prices_follow_the_day_ahead_auction_and_ch_its_own():
    de = InformationSet("11:45", BERLIN).with_source_rules([EC("de", type=["price"])])
    assert late(["price_day_ahead_d-1"], de)[0] == set()
    assert late(["price_day_ahead_d+1"], de)[0] == {"price_day_ahead_d+1"}
    rules = source_rules([EC("de", type=["price"], bzn="CH")])
    assert (rules["price_day_ahead"].time, rules["price_day_ahead"].days_before) == ("12:00", 1)
    # a zone outside SDAC gets no rule, so the check reports it as unknown instead of guessing
    restricted = source_rules([EC("rs", type=["price"], allow_restricted_prices=True)])
    assert "price_day_ahead" not in restricted


def test_rules_follow_the_prefix_and_the_selection():
    rules = source_rules([EC("de", type=["forecast"], production_types=["solar"], prefix="de")])
    assert set(rules) >= {"de_solar_forecast_day_ahead", "de_actual_*"}
    assert "de_wind_onshore_forecast_day_ahead" not in rules and "solar_forecast_day_ahead" not in rules


def test_no_default_column_resolves_to_a_built_in_default_rule():
    """The default prefix keeps every output column off the defaults layer, for any endpoint selection."""
    from itertools import product

    selections = [
        dict(type=["forecast"], forecast_types=["day-ahead", "intraday", "current"]),
        dict(type=["power", "cbet", "cbpf"]),
        dict(type=["price"]),
        dict(type=["price"], bzn="CH"),
        dict(type=["forecast", "power", "price", "cbet", "cbpf"], forecast_types=["day-ahead", "intraday", "current"]),
    ]
    for kwargs, shift in product(selections, ["", "_d-1", "_d+1", "_h-24"]):
        source = EnergyChartsSource("de", cache_dir="unused", **kwargs)
        api = FakeApi()
        source.session = api
        source._clock = Clock()
        source._sleep = source._clock.sleep
        columns = source.fetch(*SPRING_DAY).columns
        assert len(columns) > 0
        for column in columns:
            assert column.startswith("ec_")
            found = InformationSet().resolve(f"{column}{shift}")
            assert found.rule is None, (column, found.layer, found.key)


def test_outside_a_source_aware_run_every_column_is_refused_current_included():
    expanded = {(p, 1): ["ec_solar_forecast_day_ahead", "ec_solar_forecast_current"] for p in range(24)}
    info = InformationSet("11:45", BERLIN)
    with pytest.raises(Exception, match="ec_solar_forecast_current"):
        info.check(expanded, "price", "1h", HOURLY_BERLIN, "2025-03-10", "2025-03-12")


def test_current_is_refused_with_the_source_rules_too():
    from epftoolbox2.information_set import InformationSetError

    source = EnergyChartsSource(
        "de", type=["forecast"], forecast_types=["day-ahead", "current"], production_types=["solar"]
    )
    info = InformationSet("11:45", BERLIN).with_source_rules([source])
    found, unknown = info.violations(
        {(p, 1): ["ec_solar_forecast_day_ahead", "ec_solar_forecast_current"] for p in range(24)},
        "price",
        "1h",
        HOURLY_BERLIN,
        "2025-03-10",
        "2025-03-12",
    )
    assert {v.predictor for v in found} == {"ec_solar_forecast_current"} and not unknown
    with pytest.raises(InformationSetError, match="ec_solar_forecast_current"):
        info.check(
            {(p, 1): ["ec_solar_forecast_current"] for p in range(24)},
            "price",
            "1h",
            HOURLY_BERLIN,
            "2025-03-10",
            "2025-03-12",
        )


def test_the_rule_of_current_survives_a_yaml_round_trip():
    rule = source_rules([EC("de", type=["forecast"], forecast_types=["current"])])["solar_forecast_current"]
    assert Availability.from_dict(rule.to_dict()) == rule
