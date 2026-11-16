"""A refresh with a budget (step 8b): `DataPipeline.run(..., budget=, deadline=, timeout=, retries=, on_failure=)`, the one HTTP
helper every source asks through (`data/sources/_http.py`) and `DataPipeline.last_refresh` (`RefreshRecord`).

Everything runs on a `FakeClock` against fake servers: the real `EntsoeSource` parses prices from a fake ENTSO-E whose answers
take time on the clock (instantly, as long as the request's timeout, or a fixed time whatever the timeout), and the other
sources' loops are driven directly. No test opens a socket (`no_network`) or sleeps for real.

- Without the new arguments a run asks as before: each source's own timeout and retry count (the bitwise check of every
  source's requests, waits and outcomes is `epftoolbox2-research/library_check_step8/feat-refresh-budget/bitwise_requests.py`).
- A source that answers 503 forever ends within `budget` + one `timeout`; a Retry-After longer than the budget isn't waited; a
  request's timeout is cut to the time left; all five retry loops keep to the deadline and take the run's timeout and retries.
- `on_failure="cache"` gives a source that failed, timed out or got no time only its cached settled rows, and records it.
- The record: statuses, counts, seconds, rows, last valid timestamps that match the frame, and no secret anywhere in it.
- The per-source cache lock is held around the fetch, and released by a fetch that times out.
"""

import json
import logging
from unittest.mock import Mock
from urllib.parse import quote, quote_plus

import numpy as np
import pandas as pd
import pytest
import requests
from filelock import FileLock, Timeout

from epftoolbox2.clock import FakeClock, use_clock
from epftoolbox2.data import cache_manager
from epftoolbox2.data.refresh import RefreshRecord, SourceRefresh, SourceTimeout
from epftoolbox2.data.sources import EnergyChartsSource, OpenMeteoSource, PseSource, _http, _openholidays
from epftoolbox2.data.sources import energy_charts as ec
from epftoolbox2.data.sources.base import DataSource
from epftoolbox2.data.sources.entsoe import EntsoeApiError, EntsoeSource
from epftoolbox2.data.transformers import TimezoneTransformer
from epftoolbox2.pipelines import DataPipeline

pytestmark = pytest.mark.usefixtures("no_network")

KEY = "k3y+s3cr3t/==0000"  # a key whose URL-encoded forms differ from it
NOW = "2026-10-21 10:00"  # UTC; the cache records up to 10-19 00:00 UTC (two days before, floored)
START, END = "2026-10-01", "2026-10-22"  # UTC days: [10-01, 10-23)
SETTLED = pd.Timestamp("2026-10-19", tz="UTC")
HOUR = pd.Timedelta("1h")
PRICE_DOC = (
    '<?xml version="1.0" encoding="utf-8"?>\n<Publication_MarketDocument '
    'xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"><mRID>x</mRID><TimeSeries><mRID>1</mRID>'
    "<curveType>A03</curveType><Period><timeInterval><start>{start}</start><end>{end}</end></timeInterval>"
    "<resolution>PT60M</resolution>{points}</Period></TimeSeries></Publication_MarketDocument>"
)


def utc(value) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def price_of(stamp: pd.Timestamp) -> float:
    """The fake market's price: the hours since 2026-01-01 plus a quarter."""
    return (stamp - utc("2026-01-01")) / HOUR + 0.25


def answer(text: str, status: int = 200, headers=None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = text.encode()
    response.headers.update({"Content-Type": "text/xml", **(headers or {})})
    response.reason = {200: "OK", 429: "Too Many Requests", 503: "Service Unavailable"}.get(status, "")
    return response


def price_answer(params) -> requests.Response:
    start = pd.to_datetime(params["periodStart"], format="%Y%m%d%H%M", utc=True)
    end = pd.to_datetime(params["periodEnd"], format="%Y%m%d%H%M", utc=True)
    hours = pd.date_range(start, end, freq="1h", inclusive="left")
    points = "".join(
        f"<Point><position>{n}</position><price.amount>{price_of(stamp)}</price.amount></Point>"
        for n, stamp in enumerate(hours, start=1)
    )
    return answer(PRICE_DOC.format(start=f"{start:%Y-%m-%dT%H:%MZ}", end=f"{end:%Y-%m-%dT%H:%MZ}", points=points))


class Server:
    """A fake ENTSO-E giving day-ahead prices for any period, or broken (`mode`): "503" answers 503, "hang" raises a timeout
    once the request's timeout has passed, "429" asks for `retry_after` seconds, "leak" fails with the key in the message.
    Every answer takes `latency` seconds on the clock, or with `latency=None` the whole timeout the request was given."""

    def __init__(self, clock, mode="ok", latency=0.0, retry_after=None):
        self.clock, self.mode, self.latency, self.retry_after = clock, mode, latency, retry_after
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout, "at": self.clock.now()})
        self.clock.advance(timeout if self.latency is None else self.latency)
        if self.mode == "ok":
            return price_answer(params)
        if self.mode == "503":
            return answer("Service Unavailable", 503)
        if self.mode == "hang":
            raise requests.Timeout(f"read timed out ({timeout} s)")
        if self.mode == "429":
            return answer("Too many requests", 429, {"Retry-After": str(self.retry_after)})
        if self.mode == "leak":
            raise requests.ConnectionError(f"HTTPSConnectionPool: Max retries exceeded (token {KEY})")
        raise AssertionError(self.mode)

    def close(self):
        pass


def entsoe(server, prefix=""):
    source = EntsoeSource("PL", KEY, type=["price"], prefix=prefix)
    source.session = server
    source.console.quiet = True
    return source


@pytest.fixture
def logs():
    """Every message the package's loggers write during the test (they don't propagate to the root logger)."""
    messages = []

    class Keep(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = Keep(level=logging.DEBUG)
    loggers = [
        logging.getLogger(name) for name in list(logging.Logger.manager.loggerDict) if name.startswith("epftoolbox2")
    ]
    for logger in loggers:
        logger.addHandler(handler)
    yield messages
    for logger in loggers:
        logger.removeHandler(handler)


@pytest.fixture
def clock(tmp_path, monkeypatch):
    """A fake clock behind the machine's (the cache settles on it), in a fresh working directory for `.cache/`."""
    monkeypatch.chdir(tmp_path)
    with use_clock(FakeClock(NOW)) as fake:
        yield fake


def expected_prices(start, end) -> pd.Series:
    hours = pd.date_range(start, end, freq="1h", inclusive="left")
    return pd.Series([price_of(stamp) for stamp in hours], index=hours, name="price")


# ---------------------------------------------------------------------------------------------------------------------
# Without the new arguments: what it was
# ---------------------------------------------------------------------------------------------------------------------


def test_without_the_new_arguments_every_request_has_the_sources_own_timeout_and_retries(clock):
    server = Server(clock, mode="503")
    pipeline = DataPipeline(sources=[entsoe(server)])
    with pytest.raises(EntsoeApiError, match="failed after 6 attempts: HTTP 503"):
        pipeline.run(START, END)
    assert [call["timeout"] for call in server.calls] == [EntsoeSource.REQUEST_TIMEOUT] * (EntsoeSource.MAX_RETRIES + 1)
    assert clock.sleeps == [2.0, 4.0, 8.0, 16.0, 32.0]
    [entry] = pipeline.last_refresh.sources
    assert (entry.status, entry.requests, entry.retries, entry.rows) == ("failed", 6, 5, None)
    assert entry.error.startswith("EntsoeApiError: ENTSOE [PL]: A44 ")
    assert pipeline.last_refresh.deadline is None and pipeline.last_refresh.on_failure == "raise"


def test_a_run_without_the_new_arguments_gives_the_frame_it_gave(clock):
    pipeline = DataPipeline(sources=[entsoe(Server(clock))])
    df = pipeline.run(START, END)
    pd.testing.assert_series_equal(df["price"], expected_prices(utc(START), utc("2026-10-23")), check_freq=False)
    record = pipeline.last_refresh
    assert record.ok and record.failed == () and record.cache == "none"
    assert record["EntsoeSource #1"] == SourceRefresh(
        "EntsoeSource #1", "ok", 0.0, 1, 0, None, 528, {"price": utc("2026-10-22 23:00")}
    )


def test_last_refresh_is_none_before_a_run_and_reset_by_a_run_that_fails_before_any_source(clock):
    pipeline = DataPipeline(sources=[entsoe(Server(clock))])
    assert pipeline.last_refresh is None
    pipeline.run(START, END)
    assert pipeline.last_refresh is not None
    with pytest.raises(ValueError, match="must be after"):
        pipeline.run(END, START)
    assert pipeline.last_refresh is None


# ---------------------------------------------------------------------------------------------------------------------
# The budget
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "latency",
    [0.0, None, 60.0],
    ids=["answers-at-once", "hangs-until-its-timeout", "takes-60-s-whatever-its-timeout"],
)
def test_a_source_that_answers_503_forever_ends_within_the_budget_and_one_timeout(clock, latency):
    budget, timeout = 300.0, 60.0
    server = Server(clock, mode="503", latency=latency)
    pipeline = DataPipeline(sources=[entsoe(server)])
    with pytest.raises(SourceTimeout, match="deadline 2026-10-21T10:05:00") as raised:
        pipeline.run(START, END, budget=budget, timeout=timeout, retries=1000)
    assert clock.monotonic() <= budget + timeout
    assert "last error: HTTP 503" in str(raised.value) and raised.value.deadline == utc(NOW) + pd.Timedelta(minutes=5)
    assert raised.value.__context__ is None and raised.value.__cause__ is None
    assert all(call["at"] < raised.value.deadline for call in server.calls)  # no request sent after it
    assert all(call["timeout"] <= timeout for call in server.calls)
    [entry] = pipeline.last_refresh.sources
    assert entry.status == "timeout" and entry.seconds == clock.monotonic() <= budget + timeout
    assert entry.requests == len(server.calls) >= 2 and entry.retries == entry.requests - 1


def test_a_requests_timeout_is_cut_to_the_time_left(clock):
    server = Server(clock, mode="hang", latency=None)
    pipeline = DataPipeline(sources=[entsoe(server)])
    with pytest.raises(SourceTimeout):
        pipeline.run(START, END, budget=100, timeout=60)
    assert [call["timeout"] for call in server.calls] == [60, pytest.approx(100 - 60 - 2)]
    assert clock.monotonic() == pytest.approx(100.0)  # ended at the deadline: 60 s, a wait of 2 s, the 38 s left


def test_no_request_is_sent_once_the_deadline_has_come(clock):
    """Seven months are three requests (ENTSO-E asks three months at a time); each answer takes the whole timeout it was
    given, so the first two spend the budget and the third is never sent."""
    server = Server(clock, latency=None)
    pipeline = DataPipeline(sources=[entsoe(server)])
    with pytest.raises(SourceTimeout, match="no time is left for a request"):
        pipeline.run("2026-01-01", "2026-07-31", budget=90)
    assert [call["timeout"] for call in server.calls] == [60, 30]
    assert pipeline.last_refresh.sources[0].requests == 2 and clock.monotonic() == 90.0


def test_a_retry_after_longer_than_the_budget_is_not_waited(clock):
    server = Server(clock, mode="429", retry_after=100)
    pipeline = DataPipeline(sources=[entsoe(server)])
    with pytest.raises(SourceTimeout, match="the wait before the next attempt of 100 s"):
        pipeline.run(START, END, budget=60)
    assert clock.sleeps == [] and len(server.calls) == 1
    without = Server(clock, mode="429", retry_after=100)
    with pytest.raises(EntsoeApiError):
        DataPipeline(sources=[entsoe(without)]).run(START, END, retries=1)
    assert clock.sleeps == [100.0]  # the same answer without a budget: waited


def test_the_run_takes_the_earlier_of_budget_and_deadline_and_reads_a_naive_deadline_in_its_zone(clock):
    pipeline = DataPipeline(sources=[entsoe(Server(clock))], timezone="Europe/Warsaw")
    pipeline.run(START, END, budget=3600, deadline="2026-10-21 12:30")  # 10:30 UTC (CEST)
    assert pipeline.last_refresh.deadline == utc("2026-10-21 10:30")
    pipeline.run(START, END, budget=60, deadline=utc("2026-10-21 12:00"))
    assert pipeline.last_refresh.deadline == utc(NOW) + pd.Timedelta(seconds=60)
    assert pipeline.last_refresh.started == utc(NOW) and pipeline.last_refresh.budget == 60.0


def test_a_deadline_in_the_past_asks_no_source(clock):
    server = Server(clock)
    pipeline = DataPipeline(sources=[entsoe(server)])
    with pytest.raises(SourceTimeout, match="no time was left for it"):
        pipeline.run(START, END, deadline=utc(NOW) - HOUR)
    assert server.calls == [] and [s.status for s in pipeline.last_refresh.sources] == ["skipped"]


def test_the_runs_timeout_and_retries_replace_the_sources_own(clock):
    server = Server(clock, mode="503")
    pipeline = DataPipeline(sources=[entsoe(server)])
    with pytest.raises(EntsoeApiError, match="failed after 3 attempts"):
        pipeline.run(START, END, timeout=7, retries=2)
    assert [call["timeout"] for call in server.calls] == [7, 7, 7]
    with pytest.raises(EntsoeApiError, match="failed after 1 attempts"):
        pipeline.run(START, END, retries=0)
    entry = pipeline.last_refresh.sources[0]
    assert (entry.requests, entry.retries, pipeline.last_refresh.retries) == (1, 0, 0)


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"budget": -1}, "budget must be non-negative"),
        ({"budget": float("nan")}, "budget must be a finite number"),
        ({"budget": True}, "budget must be a finite number"),
        ({"budget": "60"}, "budget must be a finite number"),
        ({"timeout": 0}, "timeout must be positive"),
        ({"retries": -1}, "retries must be a whole number"),
        ({"retries": 1.5}, "retries must be a whole number"),
        ({"retries": True}, "retries must be a whole number"),
        ({"deadline": "soon"}, "deadline must be an instant"),
        ({"on_failure": "skip"}, "on_failure must be one of"),
        ({"on_failure": "cache"}, "pass cache=True"),
        ({"on_failure": "cache", "cache": "file.parquet"}, "pass cache=True"),
    ],
)
def test_arguments_are_checked_before_any_source_is_asked(clock, arguments, message):
    server = Server(clock)
    with pytest.raises(ValueError, match=message):
        DataPipeline(sources=[entsoe(server)]).run(START, END, **arguments)
    assert server.calls == []


# ---------------------------------------------------------------------------------------------------------------------
# Every source's loop keeps to the deadline and takes the run's settings
# ---------------------------------------------------------------------------------------------------------------------


class Broken:
    """Answers every request the same way (an exception is raised), recording the timeout each was given."""

    def __init__(self, item):
        self.item = item
        self.timeouts = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.timeouts.append(timeout)
        if isinstance(self.item, BaseException):
            raise self.item
        return self.item

    __call__ = get

    def close(self):
        pass


class Reply:
    def __init__(self, status, body=None):
        self.status_code, self.reason, self.text, self.headers, self._body = status, "boom", "boom", {}, body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error")


def _entsoe_request(session):
    source = EntsoeSource("PL", KEY, type=["load"])
    source.session = session
    return lambda: source._api_request({"documentType": "A65"}, utc("2026-10-01"), utc("2026-10-02"))


def _pse_request(session):
    source = PseSource(["kse-load"])
    source.session = session
    return lambda: source._get_json("https://api.raporty.pse.pl/api/kse-load", {}, "kse-load")


def _energy_charts_request(session):
    source = EnergyChartsSource("de", type=["price"], use_cache=False)
    source.session = session
    clock = FakeClock(NOW)  # the rate limit's own time, kept apart from the deadline's clock
    source._clock, source._sleep = clock.monotonic, clock.sleep
    return lambda: source._request("price", {"bzn": "DE-LU"}, "price")


def _open_meteo_request(session):
    source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone="Europe/Warsaw")
    source.session = session
    return lambda: source._request("https://example.invalid/v1", {})


def _openholidays_request(session):
    return lambda: _openholidays.get(session, "SchoolHolidays", {"countryIsoCode": "PL"})


LOOPS = {
    "entsoe": (_entsoe_request, Reply(503)),
    "pse": (_pse_request, Reply(500, {"error": {"message": "boom"}})),
    "energy_charts": (_energy_charts_request, Reply(503)),
    "open_meteo": (_open_meteo_request, requests.ConnectionError("reset")),
    "openholidays": (_openholidays_request, Reply(503)),
}


@pytest.fixture(autouse=True)
def fresh_energy_charts_buckets(monkeypatch):
    monkeypatch.setattr(ec, "_BUCKETS", {})


@pytest.mark.parametrize("loop", sorted(LOOPS))
def test_every_sources_retry_loop_keeps_to_the_deadline(loop):
    make, failure = LOOPS[loop]
    session = Broken(failure)
    clock = FakeClock(NOW)
    with use_clock(clock), _http.limits(deadline=clock.now() + pd.Timedelta(seconds=10), retries=20) as counted:
        with pytest.raises(SourceTimeout, match="no time is left for the wait before the next attempt"):
            make(session)()
    assert clock.monotonic() < 10 and len(session.timeouts) >= 2
    assert counted.requests == len(session.timeouts) and counted.retried == counted.requests - 1
    assert all(timeout <= 10 for timeout in session.timeouts)


@pytest.mark.parametrize("loop", sorted(LOOPS))
def test_every_sources_retry_loop_takes_the_runs_timeout_and_retries(loop):
    make, failure = LOOPS[loop]
    session = Broken(failure)
    with use_clock(FakeClock(NOW)) as clock, _http.limits(timeout=7, retries=1):
        with pytest.raises(Exception) as raised:
            make(session)()
    assert not isinstance(raised.value, SourceTimeout)
    assert session.timeouts == [7, 7] and len(clock.sleeps) == 1
    if loop == "openholidays":  # its own error, raised from the last one
        assert isinstance(raised.value.__cause__, requests.HTTPError) and raised.value.__suppress_context__


def test_energy_charts_does_not_start_a_rate_limit_wait_that_would_end_after_the_deadline(clock):
    """After the burst of four, the fifth forecast request waits 30 s for the rate limit: not with 20 s left."""

    class Api:
        def __init__(self):
            self.calls = 0

        def get(self, url, params=None, headers=None, timeout=None):
            self.calls += 1
            seconds = list(range(int(params["start"]), int(params["end"]) + 1, 900))
            return Reply(200, {"unix_seconds": seconds, "forecast_values": [1.0] * len(seconds)})

        def close(self):
            pass

    source = EnergyChartsSource(
        "de",
        type=["forecast"],
        production_types=["solar", "wind_onshore", "wind_offshore"],
        forecast_types=["day-ahead", "intraday"],
        use_cache=False,
    )
    source.session = api = Api()
    source._clock, source._sleep = clock.monotonic, clock.sleep  # the rate limit on the fake clock, for the test
    pipeline = DataPipeline(sources=[source])
    with pytest.raises(SourceTimeout, match="no time is left for a wait of 30 s"):
        pipeline.run("2026-10-01", "2026-10-01", budget=20)
    assert api.calls == 4 and clock.sleeps == []
    assert pipeline.last_refresh.sources[0].status == "timeout"


# ---------------------------------------------------------------------------------------------------------------------
# on_failure="cache"
# ---------------------------------------------------------------------------------------------------------------------


def _fill_the_cache(clock):
    """A first, healthy run with the cache: it returns the whole range and records the settled part."""
    df = DataPipeline(sources=[entsoe(Server(clock))]).run(START, END, cache=True)
    assert df.index.max() == utc("2026-10-22 23:00")
    return df


@pytest.mark.parametrize("mode, status", [("503", "failed"), ("hang", "timeout")])
def test_a_failed_source_gives_only_its_cached_settled_rows_and_is_recorded(clock, mode, status):
    healthy = _fill_the_cache(clock)
    server = Server(clock, mode=mode, latency=None if mode == "hang" else 0.0)
    pipeline = DataPipeline(sources=[entsoe(server)])
    df = pipeline.run(START, END, cache=True, budget=100, on_failure="cache")
    assert server.calls  # it was asked: the cache had a hole at its tail
    assert df.index.max() < SETTLED  # nothing of the unsettled tail, which the healthy run returned
    pd.testing.assert_frame_equal(df, healthy[healthy.index < SETTLED], check_freq=False)
    [entry] = pipeline.last_refresh.sources
    assert entry.status == status and entry.rows == len(df) == 18 * 24
    assert entry.last_valid == {"price": SETTLED - HOUR}
    assert entry.error.startswith("EntsoeApiError" if status == "failed" else "SourceTimeout")
    assert pipeline.last_refresh.failed == ("EntsoeSource #1",) and not pipeline.last_refresh.ok


def test_no_cached_unsettled_value_is_served_even_when_the_clock_moved_on(clock):
    """Rows the cache recorded were settled when written; a later failed refresh serves exactly them, however far the clock has
    moved, and nothing of the days since."""
    healthy = _fill_the_cache(clock)
    clock.advance(5 * 86400)  # 10-26: 10-19 .. 10-24 have settled since, but the cache never held them
    pipeline = DataPipeline(sources=[entsoe(Server(clock, mode="503"))])
    df = pipeline.run(START, END, cache=True, retries=0, on_failure="cache")
    pd.testing.assert_frame_equal(df, healthy[healthy.index < SETTLED], check_freq=False)


def test_the_other_sources_are_untouched_and_a_source_without_a_cache_gives_nothing(clock):
    class Csv(DataSource):  # no cache of its own, like CsvSource
        def fetch(self, start, end):
            raise FileNotFoundError("prices.csv")

        def _validate_config(self):
            return True

        def get_cache_config(self):
            return None

    _fill_the_cache(clock)
    pipeline = DataPipeline(sources=[entsoe(Server(clock, mode="503")), entsoe(Server(clock), prefix="ok"), Csv()])
    df = pipeline.run(START, END, cache=True, retries=0, on_failure="cache")
    assert list(df.columns) == ["price", "ok_price"]
    assert df["ok_price"].notna().all() and df["price"].last_valid_index() == SETTLED - HOUR
    statuses = {entry.name: (entry.status, entry.rows) for entry in pipeline.last_refresh.sources}
    assert statuses == {"EntsoeSource #1": ("failed", 432), "EntsoeSource #2": ("ok", 528), "Csv #3": ("failed", 0)}
    assert pipeline.last_refresh["Csv #3"].error == "FileNotFoundError: prices.csv"
    assert pipeline.last_refresh["Csv #3"].last_valid == {}


def test_a_source_whose_turn_comes_after_the_deadline_is_skipped_and_gives_its_cached_rows(clock):
    _fill_the_cache(clock)
    first = Server(clock, mode="hang", latency=None)
    second = Server(clock)
    pipeline = DataPipeline(sources=[entsoe(first), entsoe(second, prefix="b")])
    df = pipeline.run(START, END, cache=True, budget=90, on_failure="cache")
    assert [(s.status, s.requests) for s in pipeline.last_refresh.sources] == [("timeout", 2), ("skipped", 0)]
    assert second.calls == [] and list(df.columns) == ["price"]  # the second has no cache yet: no rows
    assert pipeline.last_refresh["EntsoeSource #2"].error.startswith("SourceTimeout: EntsoeSource #2: no time was left")


def test_a_budget_of_zero_serves_the_cache_alone(clock):
    healthy = _fill_the_cache(clock)
    server = Server(clock)
    pipeline = DataPipeline(sources=[entsoe(server)])
    df = pipeline.run(START, END, cache=True, budget=0, on_failure="cache")
    assert server.calls == [] and pipeline.last_refresh.sources[0].status == "skipped"
    pd.testing.assert_frame_equal(df, healthy[healthy.index < SETTLED], check_freq=False)


def test_a_failure_is_warned_about(clock, logs):
    _fill_the_cache(clock)
    DataPipeline(sources=[entsoe(Server(clock, mode="503"))]).run(START, END, cache=True, retries=0, on_failure="cache")
    assert any(message.startswith("Pipeline: EntsoeSource #1 failed (EntsoeApiError: ") for message in logs)


# ---------------------------------------------------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------------------------------------------------


class Ramp(DataSource):
    """Hourly columns, `short` ending three days before `end` and `empty` without a value, from a source that reads no network."""

    def __init__(self, name="ramp", margin_days=0):
        self.name = name
        self.margin_days = margin_days  # rows outside [start, end) too, as ENTSO-E's load and Open-Meteo return them

    def fetch(self, start, end):
        margin = pd.Timedelta(days=self.margin_days)
        hours = pd.date_range(start - margin, end + margin, freq="1h", inclusive="left")
        values = np.arange(len(hours), dtype=float)
        return pd.DataFrame(
            {
                f"{self.name}_long": values,
                f"{self.name}_short": np.where(hours < end - pd.Timedelta(days=3), values, np.nan),
                f"{self.name}_empty": np.nan,
            },
            index=hours,
        )

    def _validate_config(self):
        return True

    def get_cache_config(self):
        return {"source_type": self.name, "schema_version": 1, "margin_days": self.margin_days}


def test_the_record_counts_the_rows_inside_the_range_only(clock):
    pipeline = DataPipeline(sources=[Ramp(margin_days=1)])
    df = pipeline.run(START, END)
    assert pipeline.last_refresh["Ramp #1"].rows == len(df) == 22 * 24


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("timezone", [None, "Europe/Warsaw"])
def test_the_records_last_valid_timestamps_match_the_frame(clock, cache, timezone):
    transformers = [TimezoneTransformer()] if timezone else []
    pipeline = DataPipeline(sources=[entsoe(Server(clock)), Ramp()], transformers=transformers, timezone=timezone)
    df = pipeline.run(START, END, cache=cache, budget=600)
    record = pipeline.last_refresh
    assert set(record.last_valid) == set(df.columns)
    for column in df.columns:
        assert record.last_valid[column] == df[column].last_valid_index(), column  # the same instant, in any zone
    assert record.last_valid["ramp_empty"] is None
    assert record["Ramp #2"].rows == len(df) and record["Ramp #2"].requests == 0
    assert record.cache == ("sources" if cache else "none")


def test_the_record_of_a_failed_source_served_from_the_cache_matches_the_frame_too(clock):
    _fill_the_cache(clock)
    pipeline = DataPipeline(sources=[entsoe(Server(clock, mode="503")), Ramp()])
    df = pipeline.run(START, END, cache=True, retries=0, on_failure="cache")
    for column in df.columns:
        assert pipeline.last_refresh.last_valid[column] == df[column].last_valid_index(), column


def test_the_record_is_json_safe_and_says_what_the_run_asked(clock):
    pipeline = DataPipeline(sources=[entsoe(Server(clock)), Ramp()])
    pipeline.run(START, END, budget=600, timeout=30, retries=2)
    data = json.loads(json.dumps(pipeline.last_refresh.to_dict()))
    assert data["deadline"] == "2026-10-21T10:10:00+00:00" and (data["budget"], data["timeout"], data["retries"]) == (
        600.0,
        30.0,
        2,
    )
    assert data["sources"][0]["last_valid"] == {"price": "2026-10-22T23:00:00+00:00"}
    assert data["sources"][1]["last_valid"]["ramp_empty"] is None and data["ok"] is True
    assert isinstance(pipeline.last_refresh, RefreshRecord)
    with pytest.raises(KeyError, match="no source 'Nope #9'"):
        pipeline.last_refresh["Nope #9"]


def test_a_single_file_cache_hit_asks_no_source_and_says_so(clock):
    pipeline = DataPipeline(sources=[entsoe(Server(clock)), Ramp()])
    pipeline.run("2026-09-01", "2026-09-10", cache="sources.parquet")
    pipeline.sources[0].session = server = Server(clock, mode="503")
    pipeline.run("2026-09-01", "2026-09-10", cache="sources.parquet", budget=0)
    assert server.calls == [] and pipeline.last_refresh.cache == "file"
    assert [(s.status, s.rows, s.last_valid) for s in pipeline.last_refresh.sources] == [("ok", None, None)] * 2


def _secret_forms():
    return [KEY, quote(KEY, safe=""), quote_plus(KEY)]


class Leaky(DataSource):
    """A source with a key that it puts, plainly and URL-encoded, into its error."""

    def __init__(self, api_key):
        self.api_key = api_key

    def fetch(self, start, end):
        raise RuntimeError(f"GET https://api.example/?token={quote_plus(self.api_key)} failed for {self.api_key}")

    def _validate_config(self):
        return True

    def get_cache_config(self):
        return {"source_type": "leaky"}


@pytest.mark.parametrize("case", ["network_error", "timeout_after_network_errors", "own_message", "client_error"])
def test_no_secret_reaches_the_record_or_the_log(clock, logs, case):
    """The redaction tests of `test_entsoe.py` (a network error, a URL error and an HTTP error naming the key) applied to the
    record, to a timeout whose last error names it and to a source of one's own that leaks it, URL-encoded too."""
    if case == "own_message":
        pipeline = DataPipeline(sources=[Leaky(KEY)])
        arguments = {}
    else:
        server = Server(clock, mode="ok" if case == "client_error" else "leak", latency=10.0)
        source = entsoe(server)
        if case == "client_error":
            source.session.get = Mock(side_effect=requests.exceptions.URLRequired(f"bad url (token {KEY})"))
        pipeline = DataPipeline(sources=[source])
        arguments = {"budget": 25} if case == "timeout_after_network_errors" else {}
    pipeline.run(START, END, cache=True, on_failure="cache", **arguments)
    cached = repr(pipeline.last_refresh)
    with pytest.raises(Exception) as raised:
        pipeline.run(START, END, **arguments)
    texts = [cached, repr(pipeline.last_refresh), json.dumps(pipeline.last_refresh.to_dict()), *logs]
    if case != "own_message":  # a source's own error is raised as it is; ENTSO-E's carry no key
        texts.append(str(raised.value))
    assert logs
    for form in _secret_forms():
        assert all(form not in text for text in texts), (case, form)
    assert not pipeline.last_refresh.ok and "***" in " ".join(s.error or "" for s in pipeline.last_refresh.sources)


# ---------------------------------------------------------------------------------------------------------------------
# The per-source cache lock
# ---------------------------------------------------------------------------------------------------------------------


def _held(path) -> bool:
    other = FileLock(str(path))  # another open file: as another process would see it
    try:
        other.acquire(timeout=0)
    except Timeout:
        return True
    other.release()
    return False


def test_the_cache_lock_is_held_around_the_fetch_and_let_go_by_one_that_times_out(clock, tmp_path):
    seen = []

    class Watched(Server):
        def get(self, *args, **kwargs):
            seen.extend(_held(path) for path in tmp_path.glob(".cache/sources/*/.lock"))
            return super().get(*args, **kwargs)

    server = Watched(clock, mode="hang", latency=None)
    pipeline = DataPipeline(sources=[entsoe(server)])
    with pytest.raises(SourceTimeout):
        pipeline.run(START, END, cache=True, budget=100)
    assert seen and all(seen)  # held while it asked
    [lock] = tmp_path.glob(".cache/sources/*/.lock")
    assert not _held(lock)  # and let go by the timeout
    pipeline.run(START, END, cache=True, budget=100, on_failure="cache")
    assert not _held(lock) and pipeline.last_refresh.sources[0].status == "timeout"


def test_a_full_hit_holds_the_lock_only_for_its_reads(clock, tmp_path, monkeypatch):
    _fill_the_cache(clock)
    held = []
    read = cache_manager.CacheManager.read_cached_data

    def watched(self, key, start, end):
        held.append(_held(self._get_cache_path(key) / ".lock"))
        return read(self, key, start, end)

    monkeypatch.setattr(cache_manager.CacheManager, "read_cached_data", watched)
    server = Server(clock)
    df = DataPipeline(sources=[entsoe(server)]).run(START, "2026-10-15", cache=True)
    assert server.calls == [] and len(df) == 15 * 24
    assert held == [False]  # not held across the hit: only the read itself takes it


def test_a_full_hit_whose_file_cannot_be_read_fetches_its_range_again(clock, tmp_path):
    """The hit's second look for holes comes after its read: a file the read finds broken leaves the metadata, and its range
    is fetched under the lock, never served as a hole."""
    healthy = _fill_the_cache(clock)
    [part] = tmp_path.glob(".cache/sources/*/data_*.parquet")
    part.write_bytes(b"not parquet")
    server = Server(clock)
    df = DataPipeline(sources=[entsoe(server)]).run(START, "2026-10-15", cache=True)
    assert len(server.calls) == 1
    pd.testing.assert_frame_equal(df, healthy[healthy.index < utc("2026-10-16")], check_freq=False)


def test_waits_stay_within_each_sources_cap_also_with_more_retries():
    """A Retry-After longer than ENTSO-E's cap waits the cap (120 s), and Open-Meteo's backoff stops at 60 s, also when the
    run asks for more retries than the source's own."""
    entsoe_session = Broken(answer("Too many", 429, {"Retry-After": "600"}))
    with use_clock(FakeClock(NOW)) as clock, _http.limits(retries=1), pytest.raises(EntsoeApiError):
        _entsoe_request(entsoe_session)()
    assert clock.sleeps == [EntsoeSource.MAX_RETRY_WAIT_SECONDS]
    with use_clock(FakeClock(NOW)) as clock, _http.limits(retries=7), pytest.raises(requests.ConnectionError):
        _open_meteo_request(Broken(requests.ConnectionError("reset")))()
    assert clock.sleeps == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
