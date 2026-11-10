"""One clock (`epftoolbox2/_clock.py`, public as `epftoolbox2.clock`): every clock read and every sleep of the package goes
through `current_clock()`.

- The package's source reads no clock but through it (an AST scan), and the scan itself flags every form it looks for.
- `FakeClock`: `sleep(3600)` returns at once and advances an hour; `tick`, `advance`, `use_clock` nesting.
- Under a `FakeClock` follow it: "today" (`resolve_date(value, tz)`, at 23:30 UTC on 10-21 it is 10-22 in Warsaw), the
  per-source cache's unsettled boundary, Retry-After waits (ENTSO-E, PSE, Energy-Charts), Open-Meteo's cut between final and
  provisional rows, manifest and ensemble timestamps, and `forecast_only`'s "today", which is the issue zone's.
- With `SYSTEM` and under a `FakeClock` a run writes the same records; pool workers never read the clock.
"""

import ast
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import epftoolbox2
from epftoolbox2._clock import today, wall_time
from epftoolbox2._date_utils import resolve_date
from epftoolbox2.clock import SYSTEM, Clock, FakeClock, current_clock, use_clock
from epftoolbox2.data import cache_manager
from epftoolbox2.data.sources.base import DataSource
from epftoolbox2.ensembles import AverageEnsemble
from epftoolbox2.information_set import InformationSet
from epftoolbox2.models import OLSModel
from epftoolbox2.pipelines import DataPipeline, ModelPipeline
from epftoolbox2.pipelines import data_pipeline as data_pipeline_module
from tests.synthetic import PREDICTORS, make_market_data

pytestmark = pytest.mark.usefixtures("no_network")

PACKAGE = Path(epftoolbox2.__file__).parent
WARSAW = "Europe/Warsaw"
LATE = "2026-10-21 23:30"  # UTC: 01:30 on 10-22 in Warsaw (CEST)


@pytest.fixture
def no_real_sleep(monkeypatch):
    """A real sleep fails the test."""

    def refuse(seconds):
        raise AssertionError(f"slept {seconds} s of real time")

    monkeypatch.setattr(time, "sleep", refuse)


# ---------------------------------------------------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------------------------------------------------

# time.<name> that reads the clock or waits, wherever it is referenced (called, passed or assigned).
_TIME_READS = {"time", "time_ns", "sleep", "monotonic", "monotonic_ns"}
# time.<name> that reads the clock when called without a time of its own.
_TIME_NOW_IF_BARE = {"gmtime", "localtime", "ctime", "asctime"}
# <owner>.<name>: datetime.now, datetime.datetime.now, date.today, pd.Timestamp.now, Timestamp.utcnow, ...
_NOW_NAMES = {"now", "today", "utcnow"}
_NOW_OWNERS = {"datetime", "date", "Timestamp"}
# A parser given the string "now" or "today": pd.Timestamp("now"), pd.to_datetime("today"), np.datetime64("now").
_PARSERS = {"Timestamp", "to_datetime", "datetime64", "DatetimeIndex"}


def _dotted(node):
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id, *reversed(names)]
    return None


def clock_reads(source: str):
    """(line, what) of every clock read or sleep in `source` that doesn't go through `current_clock()`. `time.perf_counter`
    times compute and is not one (see `_clock.py`)."""
    tree = ast.parse(source)
    modules = {"time"}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.asname or alias.name for alias in node.names if alias.name == "time")
        elif isinstance(node, ast.ImportFrom) and node.module == "time":
            for alias in node.names:
                if alias.name in _TIME_READS | _TIME_NOW_IF_BARE | {"strftime", "*"}:
                    found.append((node.lineno, f"from time import {alias.name}"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = _dotted(node)
            if not chain:
                continue
            if len(chain) == 2 and chain[0] in modules and chain[1] in _TIME_READS:
                found.append((node.lineno, ".".join(chain)))
            elif len(chain) >= 2 and chain[-1] in _NOW_NAMES and chain[-2] in _NOW_OWNERS:
                found.append((node.lineno, ".".join(chain)))
        elif isinstance(node, ast.Call):
            chain = _dotted(node.func)
            if not chain:
                continue
            if len(chain) == 2 and chain[0] in modules:
                bare = not node.args and not node.keywords
                if (chain[1] in _TIME_NOW_IF_BARE and bare) or (chain[1] == "strftime" and len(node.args) < 2):
                    found.append((node.lineno, ".".join(chain) + "()"))
            first = node.args[0] if node.args else None
            if (
                chain[-1] in _PARSERS
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and first.value.strip().lower() in {"now", "today"}
            ):
                found.append((node.lineno, f"{'.'.join(chain)}({first.value!r})"))
    return sorted(found)


def package_files():
    return sorted(path for path in PACKAGE.rglob("*.py") if "__pycache__" not in path.parts)


def test_the_package_reads_no_clock_but_through_current_clock():
    found = {
        str(path.relative_to(PACKAGE)): reads
        for path in package_files()
        if path.name != "_clock.py" and (reads := clock_reads(path.read_text()))
    }
    assert found == {}, f"clock reads outside epftoolbox2/_clock.py (use current_clock()): {found}"


@pytest.mark.parametrize(
    "code",
    [
        "import time\ntime.sleep(1)",
        "import time\nx = time.time()",
        "import time\nt = time.monotonic()",
        "import time\nf = time.sleep",
        "import time as t\nt.sleep(1)",
        "from time import sleep",
        "from time import monotonic as m",
        "import time\ntime.strftime('%Y')",
        "import time\ntime.gmtime()",
        "import time\ntime.localtime()",
        "import pandas as pd\npd.Timestamp.now(tz='UTC')",
        "import pandas as pd\npd.Timestamp.utcnow()",
        "from pandas import Timestamp\nTimestamp.today()",
        "import datetime\ndatetime.datetime.now(datetime.timezone.utc)",
        "from datetime import datetime\ndatetime.now()",
        "from datetime import date\ndate.today()",
        "import datetime\ndatetime.date.today()",
        "import datetime as dt\ndt.datetime.utcnow()",
        "import pandas as pd\npd.Timestamp('now')",
        "import pandas as pd\npd.to_datetime(' Today ')",
        "import numpy as np\nnp.datetime64('now')",
    ],
)
def test_the_scan_flags_every_form_it_looks_for(code):
    assert clock_reads(code), code


@pytest.mark.parametrize(
    "code",
    [
        "from epftoolbox2._clock import current_clock\ncurrent_clock().now()",
        "clock.sleep(1)\nclock.monotonic()",
        "import time\nt = time.perf_counter()",
        "import time\ntime.strftime('%Y', time.gmtime(0))",
        "import datetime\ndatetime.date.fromisoformat('2026-10-22')",
        "import pandas as pd\npd.Timestamp('2026-10-22')",
        "def now():\n    pass\nnow()",
    ],
)
def test_the_scan_passes_what_reads_no_clock(code):
    assert clock_reads(code) == []


def test_only_the_results_store_uses_the_system_clock_itself():
    """`SYSTEM` directly, never faked, only where buffers are stamped in one thread and aged in another (the writer)."""
    users = set()
    for path in package_files():
        if path.name in ("_clock.py", "clock.py"):
            continue
        names = {node.id for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.Name)}
        if "SYSTEM" in names:
            users.add(str(path.relative_to(PACKAGE)))
    assert users == {"results/store.py", "results/parquet_store.py"}


@pytest.mark.parametrize("module", ["models/_worker.py", "models/_engine_worker.py"])
def test_the_pool_workers_import_no_clock(module):
    text = (PACKAGE / module).read_text()
    assert "_clock" not in text and "current_clock" not in text


# ---------------------------------------------------------------------------------------------------------------------
# The clock itself
# ---------------------------------------------------------------------------------------------------------------------


def test_the_system_clock_is_the_default_and_reads_the_machine():
    assert current_clock() is SYSTEM and repr(SYSTEM) == "SYSTEM"
    before = pd.Timestamp.now(tz="UTC")
    now = SYSTEM.now()
    assert str(now.tz) == "UTC" and before <= now <= pd.Timestamp.now(tz="UTC")
    first = SYSTEM.monotonic()
    assert SYSTEM.monotonic() >= first


def test_a_fake_sleep_of_an_hour_returns_at_once_and_advances_an_hour(no_real_sleep):
    clock = FakeClock(LATE)
    started = SYSTEM.monotonic()
    with use_clock(clock):
        current_clock().sleep(3600)
    assert SYSTEM.monotonic() - started < 1.0
    assert clock.now() == pd.Timestamp("2026-10-22 00:30", tz="UTC")
    assert clock.monotonic() == 3600.0
    assert clock.sleeps == [3600]


def test_a_fake_clock_stands_still_until_advanced():
    clock = FakeClock("2026-10-21 23:30:00+02:00")  # an aware start is converted to UTC
    assert clock.start == pd.Timestamp("2026-10-21 21:30", tz="UTC")
    assert clock.now() == clock.now() == clock.start
    clock.advance(90)
    assert clock.now() == clock.start + pd.Timedelta(seconds=90) and clock.sleeps == []
    assert FakeClock(LATE).now() == pd.Timestamp(LATE, tz="UTC")  # a naive start is UTC
    with pytest.raises(ValueError):
        clock.sleep(-1)
    with pytest.raises(ValueError):
        clock.advance(-1)
    with pytest.raises(ValueError):
        FakeClock(None)


def test_a_tick_moves_the_clock_on_every_read():
    clock = FakeClock(LATE, tick=0.5)
    assert clock.now() == pd.Timestamp(LATE, tz="UTC")
    assert clock.monotonic() == 0.5
    assert clock.now() == pd.Timestamp(LATE, tz="UTC") + pd.Timedelta(seconds=1)
    assert "tick=0.5" in repr(clock)


def test_use_clock_nests_and_puts_back_what_was_there():
    outer, inner = FakeClock("2026-01-01"), FakeClock("2026-06-01")
    with use_clock(outer) as given:
        assert given is outer and current_clock() is outer
        with use_clock(inner):
            assert current_clock() is inner
        assert current_clock() is outer
    assert current_clock() is SYSTEM
    with pytest.raises(TypeError, match="now"):
        with use_clock(object()):
            pass
    assert current_clock() is SYSTEM


def test_a_clock_of_ones_own_is_any_class_with_the_three_methods():
    class Frozen(Clock):
        def now(self):
            return pd.Timestamp("2030-01-01 12:00", tz="UTC")

    with use_clock(Frozen()):
        assert resolve_date("today", "UTC") == "2030-01-01"


# ---------------------------------------------------------------------------------------------------------------------
# What follows the clock
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def host_in_new_york():
    """The host's own zone is New York (`TZ`, `time.tzset`) for the test, so that its date at the clock's now is known: at
    23:30 UTC on 10-21 it is 19:30 on 10-21 there, and at 02:00 UTC on 10-22 still 22:00 on 10-21."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is Unix-only")
    before = os.environ.get("TZ")
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = before
        time.tzset()


def test_today_is_a_date_of_the_zone_asked_on_the_clock(host_in_new_york):
    with use_clock(FakeClock(LATE)):
        assert resolve_date("today", WARSAW) == "2026-10-22"
        assert resolve_date("today", "UTC") == "2026-10-21"
        assert resolve_date("now_d-1", WARSAW) == "2026-10-21"
        assert resolve_date("today_d+7", WARSAW) == "2026-10-29"
        assert resolve_date("2024-01-01", WARSAW) == "2024-01-01"
        assert today(WARSAW) == datetime.date(2026, 10, 22)
        assert wall_time(WARSAW) == pd.Timestamp("2026-10-22 01:30", tz=WARSAW)
    with use_clock(FakeClock("2026-10-22 02:00")):  # 22:00 on 10-21 in New York
        assert resolve_date("today", "UTC") == "2026-10-22"
        assert resolve_date("today") == "2026-10-21"  # no zone: the host's date, as date.today() had it
        assert today() == datetime.date(2026, 10, 21)
        assert wall_time() == pd.Timestamp("2026-10-21 22:00")  # naive, the host's wall clock


def test_the_data_pipeline_keywords_count_days_on_the_clock():
    with use_clock(FakeClock(LATE)):
        assert data_pipeline_module._calendar_day("today", WARSAW) == pd.Timestamp("2026-10-22")
        assert data_pipeline_module._calendar_day("today_d+6", "UTC") == pd.Timestamp("2026-10-27")
        assert data_pipeline_module._now(WARSAW) == pd.Timestamp("2026-10-22 01:30", tz=WARSAW)


def test_the_unsettled_boundary_follows_the_clock():
    with use_clock(FakeClock(LATE)):
        assert cache_manager.unsettled_boundary() == pd.Timestamp("2026-10-19", tz="UTC")
        # Floored at Warsaw's midnight of 10-22, two days back.
        assert cache_manager.unsettled_boundary(timezone=WARSAW) == pd.Timestamp("2026-10-19 22:00", tz="UTC")


_SERIES = pd.DataFrame(
    {"price": np.arange(24 * 30, dtype=float)},
    index=pd.date_range("2024-06-01", periods=24 * 30, freq="1h", tz="UTC"),
)


class _Hourly(DataSource):
    def __init__(self):
        self.calls = []

    def fetch(self, start, end):
        self.calls.append((start, end))
        return _SERIES.loc[start : end - pd.Timedelta("1ns")].copy()

    def _validate_config(self):
        return True

    def get_cache_config(self):
        return {"source_type": "hourly"}


def test_the_per_source_cache_refetches_the_tail_the_clock_calls_unsettled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    clock = FakeClock("2024-06-10 09:00")
    with use_clock(clock):
        source = _Hourly()
        DataPipeline().add_source(source).run("2024-06-01", "2024-06-15", cache=True)
        clock.advance(3 * 86400)
        DataPipeline().add_source(source).run("2024-06-01", "2024-06-15", cache=True)
        DataPipeline().add_source(source).run("2024-06-01", "2024-06-15", cache=True)
    assert [start for start, _ in source.calls] == [
        pd.Timestamp("2024-06-01", tz="UTC"),
        pd.Timestamp("2024-06-08", tz="UTC"),  # recorded up to two days before the clock's 06-10, floored
        pd.Timestamp("2024-06-11", tz="UTC"),  # and then up to two days before 06-13
    ]


class _Answer:
    def __init__(self, status, text="", headers=None, body=None):
        self.status_code, self.text, self.reason = status, text, text
        self.headers = headers or {}
        self._body = body

    def json(self):
        return self._body


class _Session:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return self.answers.pop(0)

    def close(self):
        pass


RETRY_AT = "Thu, 22 Oct 2026 00:30:00 GMT"  # an hour after LATE


@pytest.mark.parametrize("module", ["entsoe", "pse", "energy_charts"])
def test_a_retry_after_date_is_read_against_the_clock(module):
    import importlib

    parse = importlib.import_module(f"epftoolbox2.data.sources.{module}")._retry_after_seconds
    with use_clock(FakeClock(LATE)):
        assert parse(RETRY_AT) == 3600.0
    with use_clock(FakeClock("2026-10-22 01:00")):
        assert parse(RETRY_AT) == 0.0  # in the past: no wait
    assert parse("7") == 7.0


def test_entsoe_waits_out_retry_after_on_the_clock(no_real_sleep):
    from epftoolbox2.data.sources.entsoe import EntsoeSource

    source = EntsoeSource("PL", "test-key", type=["load"])
    later = "Wed, 21 Oct 2026 23:31:30 GMT"  # 90 s after the clock's now
    source.session = _Session([_Answer(429, "Too many", {"Retry-After": later}), _Answer(200, "<ok/>")])
    clock = FakeClock(LATE)
    with use_clock(clock):
        text = source._api_request({"documentType": "A65"}, pd.Timestamp("2026-10-20", tz="UTC"), clock.now())
    assert text == "<ok/>" and clock.sleeps == [90.0]
    assert clock.now() == pd.Timestamp(LATE, tz="UTC") + pd.Timedelta(seconds=90)


def test_entsoe_gives_up_after_its_retries_without_a_real_sleep(no_real_sleep):
    from epftoolbox2.data.sources.entsoe import EntsoeSource

    source = EntsoeSource("PL", "test-key", type=["load"])
    source.session = _Session([_Answer(503, "Service Unavailable") for _ in range(EntsoeSource.MAX_RETRIES + 1)])
    clock = FakeClock(LATE)
    with use_clock(clock), pytest.raises(Exception, match="503"):
        source._api_request({"documentType": "A65"}, pd.Timestamp("2026-10-20", tz="UTC"), clock.now())
    backoff = [EntsoeSource.RETRY_BACKOFF_SECONDS * 2**attempt for attempt in range(EntsoeSource.MAX_RETRIES)]
    assert clock.sleeps == [min(wait, EntsoeSource.MAX_RETRY_WAIT_SECONDS) for wait in backoff]
    assert clock.monotonic() == sum(clock.sleeps)


def test_pse_waits_out_retry_after_on_the_clock(no_real_sleep):
    from epftoolbox2.data.sources.pse import PseSource

    source = PseSource(["kse-load"])
    rows = {"value": []}
    source.session = _Session([_Answer(429, "slow down", {"Retry-After": "120"}, {}), _Answer(200, body=rows)])
    clock = FakeClock(LATE)
    with use_clock(clock):
        assert source._get_json("https://api.raporty.pse.pl/api/kse-load", {}, "kse-load") == rows
    assert clock.sleeps == [120.0]


def test_energy_charts_reads_its_clock_its_waits_and_its_rate_limit_on_the_clock(no_real_sleep, monkeypatch):
    from epftoolbox2.data.sources import energy_charts as ec

    monkeypatch.setattr(ec, "_BUCKETS", {})
    source = ec.EnergyChartsSource("de", type=["price"], use_cache=False)
    clock = FakeClock(LATE)
    with use_clock(clock):
        assert source._now() == clock.now() and source._clock() == clock.monotonic() == 0.0
        source._sleep(5)
        assert clock.sleeps == [5]
        # 429 with Retry-After: the bucket is blocked on the fake clock, and the wait is a fake one.
        source.session = _Session(
            [_Answer(429, "rate", {"Retry-After": "30"}), _Answer(200, body={"unix_seconds": [], "price": []})]
        )
        assert source._request("price", {}, "price") == {"unix_seconds": [], "price": []}
    assert sum(clock.sleeps) >= 35  # the 30 s the server asked for, waited out in fake time
    assert all((endpoint, bucket_clock) == ("price", clock) for endpoint, bucket_clock in ec._BUCKETS)


def test_open_meteo_cuts_final_from_provisional_rows_at_the_clock():
    from epftoolbox2.data.sources import OpenMeteoSource

    source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=WARSAW, api="single_runs", run="00")
    rows = pd.date_range("2026-10-21 22:00", periods=6, freq="1h", tz="UTC")  # local 10-22 00:00 to 05:00
    with use_clock(FakeClock(LATE)):
        cutoff, final = source._cutoff_instants(rows)  # 10-22's cutoff, 00:00 UTC, is half an hour away
    assert not final.any() and (cutoff == pd.Timestamp(LATE, tz="UTC")).all()
    with use_clock(FakeClock("2026-10-22 00:00")):
        cutoff, final = source._cutoff_instants(rows)
    assert final.all() and (cutoff == pd.Timestamp("2026-10-22", tz="UTC")).all()


# ---------------------------------------------------------------------------------------------------------------------
# Runs: manifests on the clock, records the same, workers never reading it
# ---------------------------------------------------------------------------------------------------------------------


class _Recording(FakeClock):
    """A fake clock that writes the process id of every read to `log` (a forked worker would see it too)."""

    def __init__(self, start, log: Path):
        super().__init__(start)
        self.log = log

    def _note(self):
        with open(self.log, "a") as handle:
            handle.write(f"{os.getpid()}\n")

    def now(self):
        self._note()
        return super().now()

    def monotonic(self):
        self._note()
        return super().monotonic()

    def sleep(self, seconds):
        self._note()
        super().sleep(seconds)


@pytest.fixture
def market():
    return make_market_data(start="2026-06-01", days=150, tz=WARSAW, horizon=1)


def pipeline(engine="off", information_set=None):
    mp = ModelPipeline(engine=engine, information_set=information_set)
    mp.add_model(OLSModel(PREDICTORS, training_window=28, name="ols"))
    mp.add_model(OLSModel(PREDICTORS[:2], training_window=14, name="ols_small"))
    mp.add_ensemble(AverageEnsemble(["ols", "ols_small"], name="avg"))
    return mp


def sorted_lines(path: Path):
    return sorted(path.read_text().splitlines())


@pytest.mark.parametrize("engine", ["off", "on"])
def test_a_run_on_a_fake_clock_writes_the_same_records_and_its_workers_never_read_it(
    engine, market, tmp_path, monkeypatch
):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    run = dict(data=market, test_start="2026-09-01", test_end="2026-09-04", horizon=1)
    pipeline(engine).run(save_dir=str(tmp_path / "system"), **run)
    log = tmp_path / "reads.log"
    clock = _Recording("2030-01-01 08:00", log)
    with use_clock(clock):
        pipeline(engine).run(save_dir=str(tmp_path / "fake"), **run)

    pids = set(log.read_text().split())
    assert pids == {str(os.getpid())}, "a pool worker read the clock"
    stamp = pd.Timestamp("2030-01-01 08:00", tz="UTC").isoformat()
    for name in ("ols", "ols_small", "avg"):
        assert sorted_lines(tmp_path / "system" / f"{name}.jsonl") == sorted_lines(tmp_path / "fake" / f"{name}.jsonl")
        on_system, on_fake = (
            json.loads((tmp_path / run_dir / f"{name}.manifest.json").read_text()) for run_dir in ("system", "fake")
        )
        assert on_fake["created_at"] == on_fake["updated_at"] == stamp
        assert on_system["created_at"] != stamp
        times = ("created_at", "updated_at")
        assert {k: v for k, v in on_system.items() if k not in times} == {
            k: v for k, v in on_fake.items() if k not in times
        }


def test_parquet_parts_are_named_by_the_clock(market, tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    mp = ModelPipeline().add_model(OLSModel(PREDICTORS, training_window=28, name="ols"))
    with use_clock(FakeClock("2030-01-01 08:00")):
        mp.run(market, "2026-09-01", "2026-09-02", horizon=1, save_dir=str(tmp_path), save_format="parquet")
    parts = [path.name for path in (tmp_path / "ols.parquet").rglob("part-*.parquet")]
    assert parts and all(name.startswith("part-20300101T080000-") for name in parts)


# At 23:30 UTC on 10-21 the host (New York) and UTC are on 10-21, Warsaw on 10-22. The data's last row is 10-28 in Warsaw.
WARSAW_ISSUE = InformationSet("11:45", WARSAW, use_defaults=False, unknown="ignore")


@pytest.mark.parametrize(
    "information_set, expected",
    [
        (WARSAW_ISSUE, "2026-10-22"),
        (InformationSet("11:45", "data", use_defaults=False, unknown="ignore"), "2026-10-22"),  # the frame's zone
        (InformationSet("11:45", "UTC", use_defaults=False, unknown="ignore"), "2026-10-21"),
        (None, "2026-10-21"),  # no issue zone: the host's date, as before
    ],
)
def test_forecast_only_issues_on_the_issue_zones_today(
    information_set, expected, market, tmp_path, monkeypatch, host_in_new_york
):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    with use_clock(FakeClock(LATE)):
        model = OLSModel(PREDICTORS, training_window=28, name="ols")
        ref = model.run(market, horizon=1, forecast_only=True, information_set=information_set)
        assert ref.test_start == ref.test_end == expected
        mp = ModelPipeline(information_set=information_set).add_model(
            OLSModel(PREDICTORS, training_window=28, name="ols")
        )
        report = mp.run(market, horizon=1, forecast_only=True, save_dir=str(tmp_path))
        assert set(report.predictions()["run_date"].astype(str)) == {expected}
        # The report of those results on the same clock reads the same day, also without the frame.
        again = mp.report(str(tmp_path), horizon=1, forecast_only=True)
        assert set(again.predictions()["run_date"].astype(str)) == {expected}


@pytest.mark.parametrize("information_set, expected", [(WARSAW_ISSUE, "2026-10-22"), (None, "2026-10-21")])
def test_a_residual_model_issues_on_the_issue_zones_today(
    information_set, expected, market, monkeypatch, host_in_new_york
):
    from epftoolbox2.models import ResidualModel

    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    base, learner = OLSModel(PREDICTORS, training_window=28), OLSModel(["base_fc_d+{horizon}"], training_window=28)
    model = ResidualModel(base, learner, training_window=28, refit="1D")
    with use_clock(FakeClock(LATE)):
        ref = model.run(market, horizon=1, forecast_only=True, information_set=information_set)
    assert ref.test_start == ref.test_end == expected and ref.produced == 24


def test_forecast_only_over_zones_issues_on_one_day(market, tmp_path, monkeypatch, host_in_new_york):
    """issue_timezone="data" over markets in two zones whose dates differ at the clock's now (Warsaw on 10-22, Reykjavik on
    10-21): the host's date for both."""
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    reykjavik = make_market_data(start="2026-06-01", days=150, tz="Atlantic/Reykjavik", horizon=1)
    info = InformationSet("11:45", "data", use_defaults=False, unknown="ignore")
    mp = ModelPipeline(information_set=info).add_model(OLSModel(PREDICTORS, training_window=28, name="ols"))
    with use_clock(FakeClock(LATE)):
        report = mp.run({"PL": market, "IS": reykjavik}, horizon=1, forecast_only=True)
    for zone in ("PL", "IS"):
        assert set(report.zones[zone].predictions()["run_date"].astype(str)) == {"2026-10-21"}
    with use_clock(FakeClock(LATE)):  # one zone, or zones on one clock: that zone's date
        report = mp.run({"PL": market, "PL2": market}, horizon=1, forecast_only=True)
    assert set(report.zones["PL2"].predictions()["run_date"].astype(str)) == {"2026-10-22"}
