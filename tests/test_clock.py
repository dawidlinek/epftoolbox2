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
import contextlib
import datetime
import json
import os
import threading
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

# What reads the clock or waits wherever it is referenced (called, passed or assigned), by the name it is imported under:
# `from datetime import datetime as dt; dt.now()` is `datetime.datetime.now`.
_ALWAYS = {
    "time.time",
    "time.time_ns",
    "time.sleep",
    "time.monotonic",
    "time.monotonic_ns",
    "time.clock_gettime",
    "time.clock_gettime_ns",
    "datetime.datetime.now",
    "datetime.datetime.today",
    "datetime.datetime.utcnow",
    "datetime.date.today",
    "pandas.Timestamp.now",
    "pandas.Timestamp.today",
    "pandas.Timestamp.utcnow",
    "pandas.Period.now",
    "asyncio.sleep",
    "threading.Timer",
}
# What reads the clock when called without a time of its own (no argument, or None).
_BARE = {
    "time.gmtime",
    "time.localtime",
    "time.ctime",
    "time.asctime",
    "email.utils.formatdate",
    "email.utils.localtime",
}
# A parser given "now" or "today" (as an argument, a keyword, in a list, or through a name bound to it).
_PARSERS = {
    "pandas.Timestamp",
    "pandas.to_datetime",
    "pandas.date_range",
    "pandas.DatetimeIndex",
    "pandas.Period",
    "pandas.period_range",
    "numpy.datetime64",
}
_NOW_WORDS = {"now", "today"}
# Where a bare call's time is given by keyword.
_TIME_KEYWORDS = {"secs", "t", "timeval", "dt"}
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_SCOPES = (*_FUNCTIONS, ast.ClassDef)


def _in_scope(scope):
    """The nodes of `scope` outside its nested functions and classes (whose names it binds, so they are yielded, not
    entered)."""
    todo = list(ast.iter_child_nodes(scope))
    while todo:
        node = todo.pop()
        yield node
        if not isinstance(node, _SCOPES):
            todo.extend(ast.iter_child_nodes(node))


def _targets(node):
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            yield from _targets(element)
    elif isinstance(node, ast.Starred):
        yield from _targets(node.value)


def _bindings(scope):
    """{name: what it is in `scope`}: the dotted name of what an import binds it to, or None for any other binding (an
    argument, an assignment, a loop variable), which shadows an outer import. A comprehension's variable is its own scope's,
    and left out: `time` used next to `[... for time in index]` is still read as the module (the stricter verdict)."""
    out = {}
    if isinstance(scope, _FUNCTIONS):
        arguments = scope.args
        for arg in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg, arguments.kwarg]:
            if arg is not None:
                out[arg.arg] = None
    imported = {}
    for node in _in_scope(scope):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    imported[alias.asname] = alias.name
                else:
                    imported[alias.name.split(".")[0]] = alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                imported[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                out.update(dict.fromkeys(_targets(target)))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            out.update(dict.fromkeys(_targets(node.target)))
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            out.update(dict.fromkeys(_targets(node.optional_vars)))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out[node.name] = None
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = None
        elif isinstance(node, ast.NamedExpr):
            out.update(dict.fromkeys(_targets(node.target)))
    out.update(imported)  # an import and another binding of one name: read as the import (the stricter verdict)
    return out


def _now_strings(tree):
    """Names bound anywhere to the string "now" or "today"."""
    return {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and _is_now(node.value)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _is_now(node, names=frozenset()):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip().lower() in _NOW_WORDS
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(_is_now(element, names) for element in node.elts)
    return isinstance(node, ast.Name) and node.id in names


def _qualified(node, chain):
    """The dotted name `node` (a name or an attribute of one) stands for through the imports of `chain`, else None."""
    attributes = []
    while isinstance(node, ast.Attribute):
        attributes.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    for scope in reversed(chain):
        if node.id in scope:
            root = scope[node.id]
            return None if root is None else ".".join([root, *reversed(attributes)])
    return None


def _timed_wait(call):
    """A call that waits for a time of its own: `x.wait(t)`, `x.wait_for(pred, t)`, a queue's `get(timeout=t)` or
    `put(item, timeout=t)`, `x.join(timeout=t)`, `x.acquire(timeout=t)`, with t other than None and 0. An HTTP request's
    `session.get(url, timeout=t)` bounds the network's answer and is no wait of its own (its first argument isn't `block`)."""
    if not isinstance(call.func, ast.Attribute):
        return False
    name, positional = call.func.attr, call.args
    timeout = next((kw.value for kw in call.keywords if kw.arg == "timeout"), None)
    if name == "wait" and positional:
        timeout = positional[0]
    elif name == "wait_for" and len(positional) > 1:
        timeout = positional[1]
    elif name in {"get", "put"}:
        queue_keywords = {"block", "timeout"} | ({"item"} if name == "put" else set())
        blocks = positional[1:] if name == "put" else positional
        if any(kw.arg not in queue_keywords for kw in call.keywords) or not all(
            isinstance(arg, ast.Constant) and isinstance(arg.value, bool) for arg in blocks[:1]
        ):
            return False
    elif name not in {"wait", "wait_for", "join", "acquire"}:
        return False
    return timeout is not None and not (isinstance(timeout, ast.Constant) and timeout.value in (None, 0))


def clock_reads(source: str):
    """(line, what) of every clock read, sleep or timed wait in `source` that doesn't go through `current_clock()`, by what
    the names are imported as (a local named `time` is no module). `time.perf_counter` times compute and is not one (see
    `_clock.py`). A timed wait is "timed wait: <the call>"."""
    tree = ast.parse(source)
    now_names = frozenset(_now_strings(tree))
    found = []

    def visit(scope, chain):
        chain = [*chain, _bindings(scope)]
        for node in _in_scope(scope):
            if isinstance(node, _SCOPES):
                visit(node, chain)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                found.extend(
                    (node.lineno, f"from {node.module} import {alias.name}")
                    for alias in node.names
                    if f"{node.module}.{alias.name}" in _ALWAYS | _BARE
                )
            elif isinstance(node, (ast.Attribute, ast.Name)) and isinstance(node.ctx, ast.Load):
                if _qualified(node, chain) in _ALWAYS:
                    found.append((node.lineno, ast.unparse(node)))
            if isinstance(node, ast.Call):
                name = _qualified(node.func, chain)
                first = (
                    node.args[0]
                    if node.args
                    else next((kw.value for kw in node.keywords if kw.arg in _TIME_KEYWORDS), None)
                )
                bare = first is None or (isinstance(first, ast.Constant) and first.value is None)
                if (name in _BARE and bare) or (name == "time.strftime" and len(node.args) < 2):
                    found.append((node.lineno, ast.unparse(node)))
                elif name in _PARSERS and any(
                    _is_now(value, now_names) for value in [*node.args, *(kw.value for kw in node.keywords)]
                ):
                    found.append((node.lineno, ast.unparse(node)))
                elif _timed_wait(node):
                    found.append((node.lineno, f"timed wait: {ast.unparse(node)}"))

    visit(tree, [])
    return sorted(set(found))


def package_files():
    return sorted(path for path in PACKAGE.rglob("*.py") if "__pycache__" not in path.parts)


# The timed waits the package makes on real time: the results store's writer thread idles between its batches.
REAL_TIME_WAITS = {
    ("results/store.py", "timed wait: self._write_queue.get(timeout=0.1)"),
    ("results/store.py", "timed wait: self._cond.wait(_TICK_SECONDS)"),
}


def test_the_package_reads_no_clock_but_through_current_clock():
    found = {
        (str(path.relative_to(PACKAGE)), line, what)
        for path in package_files()
        if path.name != "_clock.py"
        for line, what in clock_reads(path.read_text())
    }
    unexpected = sorted(item for item in found if item[::2] not in REAL_TIME_WAITS)
    assert unexpected == [], f"clock reads outside epftoolbox2/_clock.py (use current_clock()): {unexpected}"
    assert {item[::2] for item in found} == REAL_TIME_WAITS  # the list holds no stale entry


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
        # Aliases, and the forms of the code review's probe
        "from datetime import datetime as dt\ndt.now()",
        "from datetime import date as d\nd.today()",
        "from pandas import Timestamp as T\nT.now()",
        "import pandas as pd\npd.Period.now('D')",
        "import pandas as pd\npd.Timestamp(ts_input='now')",
        "import pandas as pd\npd.date_range(end='today', periods=3)",
        "import pandas as pd\npd.to_datetime(['now'])",
        "import pandas as pd\nx = 'now'\npd.Timestamp(x)",
        "import numpy as np\nnp.datetime64('NOW')",
        "import time\ntime.clock_gettime(time.CLOCK_REALTIME)",
        "import time\ntime.localtime(None)",
        "import email.utils\nemail.utils.formatdate()",
        "from email.utils import formatdate\nformatdate(localtime=True)",
        "import time\ndef f():\n    return time.time()",
        "def f():\n    import time\n    return time.sleep(1)",
        "import asyncio\nasyncio.sleep(1)",
        "import threading\nthreading.Timer(5, print)",
        # Timed waits
        "import threading\nthreading.Event().wait(5)",
        "cond.wait(timeout=0.5)",
        "cond.wait_for(ready, 3)",
        "q.get(timeout=1)",
        "q.get(True, timeout=1)",
        "q.put(item, timeout=1)",
        "thread.join(timeout=2)",
        "lock.acquire(timeout=1)",
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
        "from time import perf_counter\nperf_counter()",
        # Names that aren't the modules (the code review's false positives)
        "ts = obj\nts.date.today",
        "time = pd.Timestamp('2026-01-01')\ntime.time()",
        "def f(time):\n    return time.time()",
        "for time in index:\n    time.time()",
        "import time\ndef f(index):\n    for time in index:\n        time.time()",
        "import time\ndef f(time):\n    return time.time()",
        "time.sleep(1)",
        # Waits that wait for no time of their own, and an HTTP request's timeout
        "cond.wait()",
        "lock.acquire(timeout=0)",
        "q.get()",
        "session.get(url, timeout=30)",
        "session.get(url=url, params=params, timeout=30)",
        "', '.join(names)",
    ],
)
def test_the_scan_passes_what_reads_no_clock(code):
    assert clock_reads(code) == []


def test_only_real_time_code_uses_the_system_clock_itself():
    """`SYSTEM` directly, never faked, only where real time is the point: the results store's writer (buffers stamped in one
    thread and aged in another) and its wait for another process's moment on a lock file, and the tuner's waits for its
    workers' processes."""
    users = set()
    for path in package_files():
        if path.name in ("_clock.py", "clock.py"):
            continue
        names = {node.id for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.Name)}
        if "SYSTEM" in names:
            users.add(str(path.relative_to(PACKAGE)))
    assert users == {"results/store.py", "results/parquet_store.py", "tune/bayes.py"}


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


@contextlib.contextmanager
def host_zone(name: str):
    """The host's own zone is `name` inside the block (`TZ`, `time.tzset`), so that its date at the clock's now is known."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is Unix-only")
    before = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = before
        time.tzset()


@pytest.fixture
def host_in_new_york():
    """At 23:30 UTC on 10-21 it is 19:30 on 10-21 in New York, and at 02:00 UTC on 10-22 still 22:00 on 10-21."""
    with host_zone("America/New_York"):
        yield


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

    def raise_for_status(self):
        pass


class _Session:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

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
    assert list(ec._BUCKETS) == [("price", id(clock))]


def test_energy_charts_buckets_go_with_their_clock_and_take_an_unhashable_one(monkeypatch):
    """The rate limit's buckets are kept per clock (its stamps are that clock's seconds) and dropped with it, so a test's fake
    clocks, with their `sleeps`, don't stay referenced; a clock that can't be hashed (a dataclass with `__eq__`) still works."""
    import dataclasses
    import gc
    import weakref

    from epftoolbox2.data.sources import energy_charts as ec

    monkeypatch.setattr(ec, "_BUCKETS", {})
    source = ec.EnergyChartsSource("de", type=["price"], use_cache=False)
    body = {"unix_seconds": [], "price": []}
    clock = FakeClock(LATE)
    alive = weakref.ref(clock)
    with use_clock(clock):
        source.session = _Session([_Answer(200, body=body)])
        source._request("price", {}, "price")
    assert list(ec._BUCKETS) == [("price", id(clock))]
    del clock
    gc.collect()
    assert alive() is None and ec._BUCKETS == {}

    @dataclasses.dataclass
    class Pinned(Clock):  # eq=True: no __hash__
        instant: pd.Timestamp

        def now(self):
            return self.instant

        def monotonic(self):
            return 0.0

        def sleep(self, seconds):
            pass

    with pytest.raises(TypeError):
        hash(Pinned(pd.Timestamp(LATE, tz="UTC")))
    with use_clock(Pinned(pd.Timestamp(LATE, tz="UTC"))):
        source.session = _Session([_Answer(200, body=body)])
        assert source._request("price", {}, "price") == body


def test_open_meteo_backs_off_and_waits_out_its_rate_limit_on_the_clock(no_real_sleep):
    import requests

    from epftoolbox2.data.sources import OpenMeteoSource

    source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=WARSAW)
    limited = {"error": True, "reason": "Minutely API request limit exceeded"}
    source.session = _Session(
        [requests.ConnectionError("reset"), _Answer(200, body=limited), _Answer(200, body={"ok": 1})]
    )
    clock = FakeClock(LATE)
    with use_clock(clock):
        assert source._request("https://example.invalid", {}) == {"ok": 1}
    assert clock.sleeps == [source._backoff(0), source.BACKOFF_CAP_S]


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
    """A fake clock that writes the process and thread of every read to `log` (a forked worker would see it, and so would a
    thread where threads inherit the context)."""

    def __init__(self, start, log: Path):
        super().__init__(start)
        self.log = log

    def _note(self):
        with open(self.log, "a") as handle:
            handle.write(f"{os.getpid()}:{threading.get_ident()}\n")

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

    readers = set(log.read_text().split())
    assert readers == {f"{os.getpid()}:{threading.get_ident()}"}, "a pool worker or another thread read the clock"
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


# At 23:30 UTC on 10-21: 01:30 on 10-22 in Berlin, Paris and Warsaw, 23:30 on 10-21 in Reykjavik.
ZONE_TODAY = {
    "Europe/Berlin": "2026-10-22",
    "Europe/Paris": "2026-10-22",
    WARSAW: "2026-10-22",
    "Atlantic/Reykjavik": "2026-10-21",
}
DATA_ISSUE = InformationSet("11:45", "data", use_defaults=False, unknown="ignore")


def run_dates(report):
    return set(report.predictions()["run_date"].astype(str))


@pytest.mark.parametrize(
    "zones", [{"DE": "Europe/Berlin", "FR": "Europe/Paris"}, {"PL": WARSAW, "IS": "Atlantic/Reykjavik"}]
)
@pytest.mark.parametrize("host", ["UTC", "Asia/Tokyo"])
def test_forecast_only_over_zones_issues_each_zone_on_its_issue_zones_today(zones, host, tmp_path, monkeypatch):
    """issue_timezone="data" over zones: each zone issues on its own zone's today, as a run of its frame alone does and as the
    report of its folder reads it, whatever the host's zone (dates are compared, not zone names: Berlin and Paris agree)."""
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    frames = {zone: make_market_data(start="2026-06-01", days=150, tz=tz, horizon=1) for zone, tz in zones.items()}
    mp = ModelPipeline(information_set=DATA_ISSUE).add_model(OLSModel(PREDICTORS, training_window=28, name="ols"))
    with host_zone(host), use_clock(FakeClock(LATE)):
        panel = mp.run(frames, horizon=1, forecast_only=True, save_dir=str(tmp_path))
        again = mp.report(str(tmp_path), horizon=1, forecast_only=True)
        by_tuple = mp.run(frames, horizon=1, forecast_only=True, test=("today", "today"))
        for zone, tz in zones.items():
            alone = mp.run(frames[zone], horizon=1, forecast_only=True)
            expected = {ZONE_TODAY[tz]}
            assert run_dates(panel.zones[zone]) == run_dates(alone) == expected, zone
            assert run_dates(again.zones[zone]) == run_dates(by_tuple.zones[zone]) == expected, zone


def test_a_pooled_model_over_zones_on_different_days_asks_for_the_dates(monkeypatch, host_in_new_york):
    from epftoolbox2.models import PooledModel

    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    frames = {
        zone: make_market_data(start="2026-06-01", days=150, tz=tz, horizon=1)
        for zone, tz in (("PL", WARSAW), ("IS", "Atlantic/Reykjavik"))
    }
    mp = ModelPipeline(information_set=DATA_ISSUE).add_model(
        PooledModel(OLSModel(PREDICTORS, training_window=28), name="pooled")
    )
    with (
        use_clock(FakeClock(LATE)),
        pytest.raises(ValueError, match=r"PL 2026-10-22, IS 2026-10-21.*pass test_start and test_end"),
    ):
        mp.run(frames, horizon=1, forecast_only=True)


def test_forecast_only_checks_its_issue_days_on_the_issue_zones_today(market, monkeypatch, host_in_new_york):
    """2026-10-22, Warsaw's today at the clock's now, is a Thursday (the host's 10-21 a Wednesday): the pipeline's check of the
    weekdays reads the day the models issue on."""
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    mp = ModelPipeline(information_set=WARSAW_ISSUE).add_model(OLSModel(PREDICTORS, training_window=28, name="ols"))
    with use_clock(FakeClock(LATE)):
        report = mp.run(market, horizon=1, forecast_only=True, issue_days=["thursday"])
        assert set(report.predictions()["run_date"].astype(str)) == {"2026-10-22"}
        with pytest.raises(ValueError, match="(?i)wednesday|weekday|issue day"):
            mp.run(market, horizon=1, forecast_only=True, issue_days=["wednesday"])


def test_today_is_one_day_however_it_is_asked_for(market, tmp_path, monkeypatch, host_in_new_york):
    """The default, test_start/test_end="today" and test=("today", "today") (a Workflow's `test: [today, today]`) give one
    issue day, the issue zone's, in a run and in its report."""
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    mp = ModelPipeline(information_set=WARSAW_ISSUE).add_model(OLSModel(PREDICTORS, training_window=28, name="ols"))
    asked = [{}, {"test_start": "today", "test_end": "today"}, {"test": ("today", "today")}, {"test": ["now", "today"]}]
    with use_clock(FakeClock(LATE)):
        for kwargs in asked:
            assert run_dates(mp.run(market, horizon=1, forecast_only=True, **kwargs)) == {"2026-10-22"}, kwargs
        mp.run(market, horizon=1, forecast_only=True, save_dir=str(tmp_path))
        for kwargs in asked:
            assert run_dates(mp.report(str(tmp_path), horizon=1, forecast_only=True, **kwargs)) == {"2026-10-22"}, (
                kwargs
            )


def test_a_dateutil_zone_issues_on_its_market_s_today(monkeypatch, host_in_new_york):
    """A frame indexed in a dateutil zone (`str` of it doesn't read back) under issue_timezone="data": "today" is a date of
    the zone's name, as `in_data_zone` gives it."""
    import dateutil.tz

    from epftoolbox2.models.base import today_zone

    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    frame = make_market_data(start="2026-06-01", days=150, tz=WARSAW, horizon=1)
    frame.index = frame.index.tz_convert(dateutil.tz.gettz(WARSAW))
    assert today_zone(DATA_ISSUE, frame.index) == WARSAW
    mp = ModelPipeline(information_set=DATA_ISSUE).add_model(OLSModel(PREDICTORS, training_window=28, name="ols"))
    with use_clock(FakeClock(LATE)):
        assert run_dates(mp.run(frame, horizon=1, forecast_only=True)) == {"2026-10-22"}
        ref = OLSModel(PREDICTORS, training_window=28).run(
            frame, horizon=1, forecast_only=True, information_set=DATA_ISSUE
        )
        assert ref.test_start == ref.test_end == "2026-10-22"


def test_a_run_issues_every_model_on_the_day_it_read_once(market, monkeypatch):
    """A clock passing local midnight during a run (every read moves it a second, from 23:59:59.5 in Warsaw) issues every
    model on the day the run read, so the ensemble has its members' forecasts."""
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    clock = FakeClock("2026-10-21 21:59:59.5", tick=1.0)
    with use_clock(clock):
        report = pipeline(information_set=WARSAW_ISSUE).run(market, horizon=1, forecast_only=True)
    predictions = report.predictions()
    assert {model: set(rows["run_date"].astype(str)) for model, rows in predictions.groupby("model")} == {
        "ols": {"2026-10-21"},
        "ols_small": {"2026-10-21"},
        "avg": {"2026-10-21"},
    }
    assert (predictions["model"] == "avg").sum() == 24
    assert clock.now() > pd.Timestamp("2026-10-21 22:00", tz="UTC")  # it did pass Warsaw's midnight


class _Provisional(DataSource):
    """Hourly values up to `until` (the machine's now): the last day is provisional, and `world["revised"]` its later
    revision. What changes is the class's `world` (set by the test), not the instance's, whose attributes the single-file
    cache's fingerprint reads."""

    world: dict = {}

    def __init__(self, until):
        self.until = until

    def fetch(self, start, end):
        self.world["calls"].append((start, end))
        index = pd.date_range(self.until - pd.Timedelta(days=10), self.until, freq="1h", inclusive="left")
        revised = (np.arange(len(index)) >= len(index) - 24) * 1000.0 * self.world["revised"]
        return pd.DataFrame({"load": np.arange(len(index), dtype=float) + revised}, index=index).loc[
            start : end - pd.Timedelta("1ns")
        ]

    def _validate_config(self):
        return True

    def get_cache_config(self):
        return {"source_type": "provisional"}


@pytest.mark.parametrize("cache", [True, "cache.parquet"])
def test_a_clock_ahead_of_the_machine_settles_no_day_the_world_hasnt(cache, tmp_path, monkeypatch):
    """A FakeClock four years ahead: the cache still settles against the machine's now, so a run on the machine's clock a day
    later (its `now` moved on) fetches the last days again and gets their revision, where the fake now would have called
    them settled and kept the provisional values (in the per-source cache, and in a single file until 2030)."""
    monkeypatch.chdir(tmp_path)
    world = {"revised": False, "calls": []}
    monkeypatch.setattr(_Provisional, "world", world)
    until = SYSTEM.now().floor("h")
    first, last = (until - pd.Timedelta(days=9)).strftime("%Y-%m-%d"), until.strftime("%Y-%m-%d")
    with use_clock(FakeClock(until + pd.Timedelta(days=4 * 365))):
        before = DataPipeline().add_source(_Provisional(until)).run(first, last, cache=cache)
    world["revised"] = True
    monkeypatch.setattr(SYSTEM, "now", lambda: until + pd.Timedelta(days=1))  # the machine, a day later
    after = DataPipeline().add_source(_Provisional(until)).run(first, last, cache=cache)
    assert after["load"].iloc[-1] == before["load"].iloc[-1] + 1000.0
    assert len(world["calls"]) == 2


def test_energy_charts_caches_no_answer_the_machine_hasnt_settled(tmp_path):
    from epftoolbox2.data.sources import energy_charts as ec

    source = ec.EnergyChartsSource("de", type=["price"], cache_dir=str(tmp_path))
    real = SYSTEM.now().floor("D")
    body = {"unix_seconds": [], "price": []}
    with use_clock(FakeClock(real + pd.Timedelta(days=4 * 365))):
        source.session = _Session([_Answer(200, body=body), _Answer(200, body=body)])
        source._get("price", {"bzn": "DE-LU"}, real - pd.Timedelta(days=3), real - pd.Timedelta(days=1))  # not settled
        assert list(tmp_path.glob("*.json")) == []
        source._get("price", {"bzn": "DE-LU"}, real - pd.Timedelta(days=30), real - pd.Timedelta(days=20))
        assert len(list(tmp_path.glob("*.json"))) == 1


def test_open_meteo_calls_no_row_final_before_the_machine_does():
    from epftoolbox2.data.sources import OpenMeteoSource

    source = OpenMeteoSource(latitude=52.23, longitude=21.01, timezone=WARSAW, api="single_runs", run="00")
    tomorrow = (SYSTEM.now().tz_convert(WARSAW) + pd.Timedelta(days=1)).normalize()
    rows = pd.date_range(tomorrow, periods=3, freq="1h").tz_convert(
        "UTC"
    )  # their cutoff, tomorrow at 00:00 UTC, is ahead
    with use_clock(FakeClock(SYSTEM.now() + pd.Timedelta(days=4 * 365))):
        cutoff, final = source._cutoff_instants(rows)
    assert not final.any() and (cutoff <= SYSTEM.now()).all()


def test_a_reversed_forecast_only_period_raises_however_it_is_given(market, monkeypatch):
    """ "today" to 10-21 on a clock where today is 10-22 in the issue zone: a reversed period raises, in a run, a model's own
    run, a residual model's, and per zone in a run over zones, where it used to plan no forecast and say nothing."""
    from epftoolbox2.models import ResidualModel

    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    reykjavik = make_market_data(start="2026-06-01", days=150, tz="Atlantic/Reykjavik", horizon=1)
    mp = ModelPipeline(information_set=DATA_ISSUE).add_model(OLSModel(PREDICTORS, training_window=28, name="ols"))
    residual = ResidualModel(
        OLSModel(PREDICTORS, training_window=28), OLSModel(["base_fc_d+{horizon}"], training_window=28), 28, refit="1D"
    )
    reversed_ = [{"test": ("today", "2026-10-21")}, {"test_start": "today", "test_end": "2026-10-21"}]
    with host_zone("UTC"), use_clock(FakeClock(LATE)):
        for kwargs in reversed_:
            with pytest.raises(ValueError, match="reversed"):
                mp.run(market, horizon=1, forecast_only=True, **kwargs)
            with pytest.raises(ValueError, match=r"Zone 'PL'.*reversed"):
                mp.run({"PL": market, "IS": reykjavik}, horizon=1, forecast_only=True, **kwargs)
        for model in (OLSModel(PREDICTORS, training_window=28), residual):
            with pytest.raises(ValueError, match="reversed"):
                model.run(market, "today", "2026-10-21", horizon=1, forecast_only=True, information_set=DATA_ISSUE)


def test_a_period_valid_in_every_zone_runs_though_the_host_reads_it_reversed(monkeypatch):
    """On a UTC host at 23:30 UTC the host's today is 10-21, so ("2026-10-22", "today") reads as reversed there; in Berlin and
    Paris it is one day, 10-22, and the run over the two zones issues on it."""
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    frames = {
        zone: make_market_data(start="2026-06-01", days=150, tz=tz, horizon=1)
        for zone, tz in (("DE", "Europe/Berlin"), ("FR", "Europe/Paris"))
    }
    mp = ModelPipeline(information_set=DATA_ISSUE).add_model(OLSModel(PREDICTORS, training_window=28, name="ols"))
    with host_zone("UTC"), use_clock(FakeClock(LATE)):
        panel = mp.run(frames, horizon=1, forecast_only=True, test=("2026-10-22", "today"))
    assert {zone: run_dates(panel.zones[zone]) for zone in frames} == {"DE": {"2026-10-22"}, "FR": {"2026-10-22"}}


def test_a_report_over_zones_reads_today_once_for_every_zone(tmp_path, monkeypatch):
    """The clock passes 22:00 UTC (midnight in Berlin and Paris) between two zones of one report: both still read the day the
    report read, the day their run issued on."""
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    frames = {
        zone: make_market_data(start="2026-06-01", days=150, tz=tz, horizon=1)
        for zone, tz in (("DE", "Europe/Berlin"), ("FR", "Europe/Paris"))
    }
    mp = ModelPipeline(information_set=DATA_ISSUE).add_model(OLSModel(PREDICTORS, training_window=28, name="ols"))
    with use_clock(FakeClock("2026-10-21 21:59:59")):
        mp.run(frames, horizon=1, forecast_only=True, save_dir=str(tmp_path))
    clock = FakeClock(
        "2026-10-21 21:59:59", tick=2.0
    )  # 23:59:59 in Berlin and Paris, and two seconds later at every read
    with use_clock(clock):
        report = mp.report(str(tmp_path), horizon=1, forecast_only=True)
    assert {zone: run_dates(report.zones[zone]) for zone in frames} == {"DE": {"2026-10-21"}, "FR": {"2026-10-21"}}
    assert {zone: len(report.zones[zone].predictions()) for zone in frames} == {"DE": 24, "FR": 24}


def test_a_residual_model_reads_today_once(market, monkeypatch):
    """A residual model's own run reads "today" for its base and gives the learner that day: midnight passing while the base
    runs (every read moves the clock a second, from 23:59:59.5 in Warsaw) doesn't move the learner to the next day."""
    from epftoolbox2.models import ResidualModel

    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    base, learner = OLSModel(PREDICTORS, training_window=28), OLSModel(["base_fc_d+{horizon}"], training_window=28)
    model = ResidualModel(base, learner, training_window=28, refit="1D")
    clock = FakeClock("2026-10-21 21:59:59.5", tick=1.0)
    with use_clock(clock):
        ref = model.run(market, horizon=1, forecast_only=True, information_set=WARSAW_ISSUE)
    assert ref.test_start == ref.test_end == "2026-10-21" and ref.produced == 24
