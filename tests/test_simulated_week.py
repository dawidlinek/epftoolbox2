"""The simulated week (step 8l, the done criterion of step 8): seven issue days of a live day-ahead pipeline on a fake clock, against
a fake ENTSO-E server answering from a seeded truth by its publication schedule (`tests/live_week.py`), with a late source, a
missing input, an outage, a missed slot held by a hung call, a crash, a failed refit, a stale fit, a revised input, an implausible
value, a dry run, deliveries and a late call. Each call's status, exit code and revision are the operations spec's table ("The
simulated week"); the final revisions equal a backtest of the same pipeline over the archive the server holds at the end of the
week, bit for bit.

The week runs once per module (`week`), on one process (the worker pools run in process), under `use_clock(FakeClock)`: no socket,
no real sleep. `test_a_second_pass_gives_the_same_bytes` runs it again into a fresh directory, and
`test_the_week_in_parquet_gives_the_same_records` with `save_format="parquet"`.

The columns waited for are the issue rows' `load_forecast_d+1` and `price_exaa_d+1` (the spec's table names the source columns,
`load_forecast` and `price_exaa`): `wait_for` reads a column's cells on the issue day's rows, where `load_forecast` is the
forecast of the issue day itself, out since the day before.
"""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import hashlib
import json
import logging
import os
import time
from collections import Counter
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
import yaml
from threadpoolctl import threadpool_limits

from epftoolbox2 import cli
from epftoolbox2.clock import FakeClock, use_clock
from epftoolbox2.data.sources import CalendarSource, EntsoeSource
from epftoolbox2.data.transformers import LagTransformer, TimezoneTransformer
from epftoolbox2.ensembles import FallbackEnsemble
from epftoolbox2.models import NaiveModel, OLSModel
from epftoolbox2.models import _engine_worker, _fits
from epftoolbox2.models._fits import FitTooEarlyError, fit_dir, read_fit
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.models.context import list_hash
from epftoolbox2.pipelines import DataPipeline, ModelPipeline
from epftoolbox2.pipelines._issue_records import read_snapshot
from epftoolbox2.pipelines._live import data_span
from epftoolbox2.results import live as live_module
from epftoolbox2.results.checks import ForecastChecks
from epftoolbox2.results.live import LiveStore
from epftoolbox2.results.store import read_records
from tests.live_week import (
    BASE_PREDICTORS,
    KEY,
    WEEK,
    WEEKDAY_COLUMNS,
    ZONE,
    FakeEntsoe,
    ParquetReads,
    QuietProgress,
    archive,
    local,
    main_predictors,
)

pytestmark = pytest.mark.usefixtures("no_network")

WED, THU, FRI, SAT, SUN, MON, TUE = WEEK
DEADLINE = "11:55"
C1 = {"ensure": True, "wait_for": {"load_forecast_d+1": "10:45"}}
C2 = {"wait_for": {"price_exaa_d+1": "11:40"}}
MODELS = ["main", "base", "naive"]
NAMES = [*MODELS, "live"]
SETUP_AT = (datetime.date(2026, 10, 18), "22:00")  # Sunday: the stored fits of the 10-19 anchor
MONITOR_AT = (datetime.date(2026, 10, 28), "18:00")
EXAA = "price_exaa_d+1"
A65 = "load_forecast_d+1"
# What a live record and its backtest twin may differ in: the record's place in its run, the live keys, and the actual, which
# the backtest's archive holds and no forecast issued at 11:45 can (the target day's prices come out at 12:57).
NOT_COMPARED = ("day_in_test", "revision", "issued_at", "mode", "snapshot", "actual")

# The table: per issue day, the revision each policy reads, and what the first revision lacked against the backtest. A call is
# unchanged only when its records are identical in full (decision 2 as Dawid settled it on 2026-11-17), so Friday's and Tuesday's
# 11:20 calls write r2, where the spec's table had them unchanged: on Friday main's skips now read the list with EXAA, on Tuesday
# main's check_failed detail names another failing value. Each r2 is the backtest's record.
FINAL = {WED: 2, THU: 2, FRI: 2, SAT: 2, SUN: 3, MON: 2, TUE: 2}
DELIVERED = {WED: 2, MON: 2}
FIRST_DIFFERS = {  # the names whose first revision isn't the backtest's record, and why
    WED: {"main", "live"},  # no EXAA at 10:15
    THU: {"main", "live"},  # no EXAA at 10:15
    FRI: {"main"},  # 10:45, before EXAA: main's skips carry the list without it
    SAT: {"main", "base", "live"},  # the outage: no price of Friday or Saturday in the cache
    SUN: {"main", "live"},  # no EXAA at 10:15
    MON: {"main", "base", "live"},  # the 10-19 fit (stale) and A65 before its revision
    TUE: {"main"},  # no EXAA at 10:16: main's check_failed skips carry the other list and failing value
}


# --- the pipelines --------------------------------------------------------------------------------------------------------------


def data_pipeline() -> DataPipeline:
    return (
        DataPipeline(timezone=ZONE)
        .add_source(EntsoeSource("DE_LU", api_key=KEY, type=["price", "load"], price_exaa=True, prefer_15min=True))
        .add_source(CalendarSource("DE", holidays=False, weekday="onehot", freq="15min"))
        .add_transformer(TimezoneTransformer())
        .add_transformer(LagTransformer(columns=["price"], lags=[1, 2, 6], freq="day"))
        .add_transformer(
            LagTransformer(columns=["load_forecast", "price_exaa", *WEEKDAY_COLUMNS], lags=[-1], freq="day")
        )
    )


def model_pipeline(dp: DataPipeline) -> ModelPipeline:
    mp = (
        ModelPipeline(issue_time="11:45", issue_timezone=ZONE, checks=ForecastChecks(range=(-500, 1000)))
        .add_model(OLSModel(main_predictors, training_window=56, refit="7D", name="main"))
        .add_model(OLSModel(BASE_PREDICTORS, training_window=56, refit="7D", name="base"))
        .add_model(NaiveModel(["price_d-6"], name="naive", known_only=True))
        .add_ensemble(FallbackEnsemble(["main", "base", "naive"], name="live"))
    )
    # What `Workflow` does for a live call, so the CLI's pipeline and this one are one pipeline (one config digest).
    mp.information_set = mp.information_set.with_source_rules([*dp.sources, *dp.transformers])
    return mp


class InProcessPool:
    """Stands in for the worker pools: the week runs on one process, and the instrumented `_predict_fitted` sees every row."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        if initializer is _engine_worker._engine_init:
            _engine_worker._engine_load(*initargs)
        else:
            _worker_init(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, task, *args):
        future = Future()
        try:
            future.set_result(task(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


@contextlib.contextmanager
def quiet():
    """The package's INFO lines off for the week (about a thousand through rich's handler, more than half its time); warnings stay."""
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


def prepare(mp: pytest.MonkeyPatch, root: Path) -> None:
    mp.chdir(root)  # the per-source caches go to ./.cache
    mp.setenv("MAX_PROCESSES", "1")
    mp.setenv("THREADS_PER_PROCESS", "1")
    mp.setenv("ENTSOE_API_KEY", KEY)  # the saved data pipeline names it as ${ENTSOE_API_KEY}
    for module in ("epftoolbox2.models.base", "epftoolbox2.models._engine", "epftoolbox2.models._fits"):
        mp.setattr(f"{module}.ProcessPoolExecutor", InProcessPool)
    mp.setattr("epftoolbox2.data.sources.entsoe.Progress", QuietProgress)
    mp.setattr("epftoolbox2.data.cache_manager.pd.read_parquet", ParquetReads(pd.read_parquet))


def tree(root: Path) -> Dict[str, Any]:
    """Every file under `root` with its bytes' hash and modification time, and every directory: what a dry run must not change."""
    out = {}
    for path in sorted(root.rglob("*")):
        key = str(path.relative_to(root))
        out[key] = "dir" if path.is_dir() else (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    return out


def keyed(records, day: Optional[datetime.date] = None) -> Dict[tuple, str]:
    """(run_date, hour, horizon) -> the record as JSON text without what a live record and its backtest twin may differ in."""
    iso = None if day is None else day.isoformat()
    return {
        (r["run_date"], int(r["hour"]), int(r["horizon"])): json.dumps(
            {k: v for k, v in r.items() if k not in NOT_COMPARED}, sort_keys=True, default=str
        )
        for r in records
        if iso is None or r["run_date"] == iso
    }


def lines(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- the week ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Call:
    """What one call of the week gave."""

    label: str
    at: pd.Timestamp
    ended: pd.Timestamp
    exit_code: int
    result: Any = field(default=None, repr=False)
    seconds: float = 0.0
    before: Any = field(
        default=None, repr=False
    )  # the live directory before the call, where the table says it writes nothing
    after: Any = field(default=None, repr=False)

    @property
    def status(self) -> Optional[str]:
        return getattr(self.result, "status", None)

    @property
    def revision(self) -> Optional[int]:
        return getattr(self.result, "revision", None)


class Week:
    """The simulated week in `root`: the live directory `root/live`, the per-source caches under `root/.cache`, the workflow YAML
    the CLI reads, the fake server and the clock. `run()` makes every call of the table in order, one after another."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch, save_format: str = "jsonl"):
        self.root = root
        self.live = root / "live"
        self.save_format = save_format
        self.monkeypatch = monkeypatch
        self.clock = FakeClock(local(*SETUP_AT))
        self.fake = FakeEntsoe().install(monkeypatch)
        self.calls: Dict[str, Call] = {}
        self.predicted_rows = 0
        self.nan_rows = 0
        self._instrument()
        self.experiment = self._write_experiment()
        self.receipt = root / "receipt.json"
        self.receipt.write_text('{"accepted": true}\n')

    # -- set-up ------------------------------------------------------------------------------------------------------------------

    def _instrument(self) -> None:
        """Every row any OLS forecast reads is checked for a NaN (and counted), whichever path forecasts it: the engine's
        forecast and the model path both end in `_predict_fitted`."""
        predict = OLSModel._predict_fitted
        week = self

        def instrumented(model, fitted, x):
            week.predicted_rows += 1
            if not np.isfinite(np.asarray(x, dtype=float)).all():
                week.nan_rows += 1
                raise AssertionError(f"{model.name}: a NaN input reached _predict_fitted")
            return predict(model, fitted, x)

        self.monkeypatch.setattr(OLSModel, "_predict_fitted", instrumented)

    def _write_experiment(self) -> Path:
        dp = data_pipeline()
        dp.save(self.root / "data_pipeline.yaml")
        model_pipeline(dp).save(self.root / "model_pipeline.yaml")
        config = {
            "data_pipeline": {"path": "data_pipeline.yaml", "cache": True},
            "model_pipeline": {
                "path": "model_pipeline.yaml",
                "target": "price",
                "horizon": 1,
                "freq": "15min",
                **({} if self.save_format == "jsonl" else {"save_format": self.save_format}),
            },
            "live": {
                "save_dir": "live",
                "deadline": DEADLINE,
                "poll": "5min",
                "budget": "120s",
                "timeout": "30s",
                "on_failure": "cache",
            },
        }
        path = self.root / "experiment.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    # -- the clock and the calls ---------------------------------------------------------------------------------------------

    def move_to(self, day: datetime.date, hhmm: str) -> pd.Timestamp:
        """Move the clock to `hhmm` of `day`; the previous call must have ended before it."""
        target = local(day, hhmm)
        now = self.clock.now()
        assert now <= target, f"the previous call ended at {now.tz_convert(ZONE)}, after {target.tz_convert(ZONE)}"
        self.clock.advance((target - now).total_seconds())
        return target

    def _note(self, label: str, at: pd.Timestamp, exit_code: int, result: Any = None, started: float = 0.0) -> Call:
        assert label not in self.calls, label
        call = Call(label, at, self.clock.now(), exit_code, result, time.perf_counter() - started if started else 0.0)
        self.calls[label] = call
        return call

    def options(self) -> Dict[str, Any]:
        return dict(
            deadline=DEADLINE,
            poll="5min",
            budget="120s",
            timeout="30s",
            on_failure="cache",
            target="price",
            horizon=1,
            freq="15min",
            save_format=self.save_format,
        )

    def forecast(self, label: str, day: datetime.date, hhmm: str, **kwargs) -> Call:
        at = self.move_to(day, hhmm)
        started = time.perf_counter()
        dp = data_pipeline()
        result = model_pipeline(dp).forecast(dp, day, self.live, **{**self.options(), **kwargs})
        return self._note(label, at, result.exit_code, result, started)

    def cli(self, label: str, day: datetime.date, hhmm: str, *argv: str) -> Call:
        at = self.move_to(day, hhmm)
        started = time.perf_counter()
        code = cli.main([argv[0], str(self.experiment), *argv[1:]], clock=self.clock)
        return self._note(label, at, code, None, started)

    def refit(self, label: str, day: datetime.date, hhmm: str, issue: datetime.date) -> Call:
        at = self.move_to(day, hhmm)
        started = time.perf_counter()
        dp = data_pipeline()
        mp = model_pipeline(dp)
        frame = dp.run(*data_span(mp.models, issue, [1], dp), cache=True, budget=120, timeout=30, on_failure="cache")
        result = mp.refit(frame, self.live, issue=issue, if_due=True, horizon=1, freq="15min")
        return self._note(label, at, 0, result, started)

    def hold_lock(self) -> int:
        """`live.lock` held through a second open file description, as a hung call in another process holds it."""
        fd = os.open(self.live / "live.lock", os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd

    # -- the table ------------------------------------------------------------------------------------------------------------

    def run(self) -> "Week":
        mp = self.monkeypatch
        with use_clock(self.clock):
            self.refit("setup refit", *SETUP_AT, issue=datetime.date(2026, 10, 19))

            # Wednesday: nothing happens.
            self.forecast("wed C1", WED, "08:45", **C1)
            self.forecast("wed C2", WED, "11:20", **C2)
            at = self.move_to(WED, "11:30")
            line = model_pipeline(data_pipeline()).mark_delivered(self.live, WED, 2, self.receipt)
            self._note("wed deliver r2", at, 0, line)

            # Thursday: EXAA for Friday comes at 11:37 (late source).
            self.forecast("thu C1", THU, "08:45", **C1)
            self.forecast("thu C2", THU, "11:20", **C2)

            # Friday: the load forecast for Saturday never comes (missing input).
            self.forecast("fri C1", FRI, "08:45", **C1)
            self.forecast("fri C2", FRI, "11:20", **C2)

            # Saturday: every request 503 after 20 s from 08:00 to 11:25 (outage).
            self.forecast("sat C1", SAT, "08:45", **C1)
            self.sat_export = self.calls["sat C1"].result.export(self.root / "out" / "sat.csv", grid="products")
            self.forecast("sat C2", SAT, "11:20", **C2)

            # Sunday, a 25-hour day: no 08:45 call, and a hung call holds the lock until 09:40 (missed slot).
            self.move_to(SUN, "08:45")
            fd = self.hold_lock()
            before = tree(self.live)
            self.cli("sun busy 09:30", SUN, "09:30", "forecast", "--issue", SUN.isoformat(), "--ensure")
            self.calls["sun busy 09:30"].before, self.calls["sun busy 09:30"].after = before, tree(self.live)
            self.move_to(SUN, "09:40")
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            self.forecast("sun C1 09:45", SUN, "09:45", **C1)
            self.forecast("sun ensure 10:30", SUN, "10:30", ensure=True)
            opened = live_module.LiveRevision.open

            def crash(revision, name, *args, **kwargs):
                if name == "base":  # after main's records, before the commit
                    raise OSError("injected crash: the machine went down")
                return opened(revision, name, *args, **kwargs)

            mp.setattr(live_module.LiveRevision, "open", crash)
            self.cli(
                "sun C2 crash", SUN, "11:20", "forecast", "--issue", SUN.isoformat(), "--wait-for", f"{EXAA}=11:40"
            )
            mp.setattr(live_module.LiveRevision, "open", opened)
            self.forecast("sun ensure 11:30", SUN, "11:30", ensure=True)
            self.forecast("sun rerun 11:35", SUN, "11:35", **C2)
            run_fits = _fits.run_fits

            def failing(*args, **kwargs):
                raise MemoryError("injected failure: the refit ran out of memory")

            mp.setattr(_fits, "run_fits", failing)
            self.cli("sun refit 22:00", SUN, "22:00", "refit", "--if-due", "--issue", "tomorrow")
            mp.setattr(_fits, "run_fits", run_fits)
            self.calls["sun refit 22:00"].after = {name: _fits.stored_anchors(self.live, name) for name in MODELS[:2]}

            # Monday: no fit of the 10-26 anchor at 08:45 (stale fit); Tuesday's A65 is revised at 11:05 (revised input).
            self.forecast("mon C1", MON, "08:45", **C1)
            self.refit("mon refit 11:00", MON, "11:00", issue=MON)
            self.forecast("mon C2", MON, "11:20", **C2)
            before = tree(self.live)
            self.forecast("mon dry run 11:50", MON, "11:50", dry_run=True)
            self.calls["mon dry run 11:50"].before, self.calls["mon dry run 11:50"].after = before, tree(self.live)
            day, receipt = MON.isoformat(), str(self.receipt)
            deliver = ("mark-delivered", "--issue", day, "--receipt", receipt, "--revision")
            self.cli("mon deliver r2", MON, "11:52", *deliver, "2")
            self.cli("mon deliver r3", MON, "11:53", *deliver, "3")
            self.cli("mon deliver r2 dry", MON, "11:53", *deliver, "2", "--dry-run")
            before = tree(self.live)
            self.cli("mon C2 late", MON, "11:58", "forecast", "--issue", day, "--wait-for", f"{EXAA}=11:40")
            self.calls["mon C2 late"].before, self.calls["mon C2 late"].after = before, tree(self.live)

            # Tuesday: one A65 value for Wednesday is 1e6 (implausible input).
            self.forecast("tue C1", TUE, "08:45", **C1)
            self.forecast("tue C2", TUE, "11:20", **C2)

            # Wednesday 18:00: the monitor, through the CLI (the exit code) and the Python API (the tables).
            alerts = self.root / "alerts.yaml"
            alerts.write_text(yaml.safe_dump([{"delivered_below": 1.0}]))
            self.cli("monitor", *MONITOR_AT, "monitor", "--windows", "7", "--alert", str(alerts))
            dp = data_pipeline()
            actuals = dp.run(WED.isoformat(), "2026-10-30", cache=True)
            monitored = model_pipeline(dp)
            self.monitor = monitored.monitor(
                actuals, self.live, windows=[7], alerts=[{"delivered_below": 1.0}], cutoff=DEADLINE
            )
            self.monitor_first = monitored.monitor(actuals, self.live, windows=[7], policy="first", cutoff=DEADLINE)
        return self

    # -- reading the live directory -------------------------------------------------------------------------------------------

    def store(self) -> LiveStore:
        return LiveStore(self.live)

    def path(self, name: str) -> Path:
        parquet = self.save_format == "parquet" and name in MODELS
        return self.live / f"{name}{'.parquet' if parquet else '.jsonl'}"

    def visible(self, name: str, policy: str = "latest") -> List[Dict[str, Any]]:
        return list(read_records(self.path(name), statuses=None, committed=self.store().visible(policy, "deadline")))

    def revision(self, name: str, day: datetime.date, revision: int) -> List[Dict[str, Any]]:
        iso = day.isoformat()
        only = {iso: {revision: self.store().committed(iso)[iso][revision]}}
        return [r for r in read_records(self.path(name), statuses=None, committed=only) if r["run_date"] == iso]

    def snapshot_dir(self, day: datetime.date, revision: int) -> Path:
        return self.live / "issues" / day.isoformat() / f"r{revision}"

    def json(self, day: datetime.date, revision: int, name: str) -> Any:
        return json.loads((self.snapshot_dir(day, revision) / name).read_text())

    def summary(self) -> Dict[str, Any]:
        """What two runs of the week must give alike: every model's and ensemble's records of each committed revision (sorted JSON
        lines; the raw sorted lines of each JSONL file too), every snapshot file's bytes, and every call's exit code. Read once."""
        if getattr(self, "_summary", None) is not None:
            return self._summary
        started = time.perf_counter()
        records = {}
        for day, revisions in sorted(self.store().committed().items()):
            for number in sorted(revisions):
                for name in NAMES:
                    rows = self.revision(name, datetime.date.fromisoformat(day), number)
                    records[(name, day, number)] = sorted(json.dumps(r, sort_keys=True, default=str) for r in rows)
        files = {
            str(path.relative_to(self.live)): sorted(path.read_text().splitlines())
            for path in sorted(self.live.glob("*.jsonl"))
            if path.name in {f"{name}.jsonl" for name in NAMES}
        }
        snapshots = {
            str(path.relative_to(self.live)): path.read_bytes()
            for path in sorted((self.live / "issues").rglob("*"))
            if path.is_file()
        }
        self._summary = {
            "records": records,
            "files": files,
            "snapshots": snapshots,
            "exit_codes": [(label, call.exit_code) for label, call in self.calls.items()],
        }
        self.summary_seconds = time.perf_counter() - started
        return self._summary


def run_week(root: Path, save_format: str = "jsonl") -> Week:
    root.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp, threadpool_limits(limits=1), quiet():
        prepare(mp, root)
        started = time.perf_counter()
        week = Week(root, mp, save_format).run()
        week.seconds = time.perf_counter() - started
    return week


def backtest(root: Path) -> Path:
    """The backtest twin: `run()` of the same pipeline over the archive the server holds at the end of the week (Saturday's missing
    load forecast, Tuesday's revision and Wednesday's bad value in it), test 2026-10-21 to 2026-10-27, fetched afresh."""
    folder = root / "backtest"
    with pytest.MonkeyPatch.context() as mp, threadpool_limits(limits=1), quiet():
        prepare(mp, root)
        FakeEntsoe().install(mp)
        with use_clock(FakeClock(local(*MONITOR_AT))):
            dp = data_pipeline()
            frame = dp.run("2026-06-15", "2026-10-30")
            model_pipeline(dp).run(
                frame, WED.isoformat(), TUE.isoformat(), target="price", horizon=1, freq="15min", save_dir=str(folder)
            )
    return folder


@pytest.fixture(scope="module")
def week(tmp_path_factory) -> Week:
    """The week and its backtest twin, run once for the module's assertions."""
    out = run_week(tmp_path_factory.mktemp("week"))
    started = time.perf_counter()
    out.backtest = backtest(out.root)
    out.backtest_seconds = time.perf_counter() - started
    return out


def once(records) -> Dict[tuple, str]:
    """`keyed` of a day's records, which must be its 96 keys once each: a duplicate key can't hide behind the mapping."""
    records = list(records)
    out = keyed(records)
    assert len(records) == len(out) == 96, (len(records), len(out))
    return out


def twin(week: Week, name: str, day: datetime.date) -> Dict[tuple, str]:
    return once(r for r in lines(week.backtest / f"{name}.jsonl") if r["run_date"] == day.isoformat())


def equals_twin(week: Week, day: datetime.date, revision: int, names=NAMES) -> Dict[str, bool]:
    return {name: once(week.revision(name, day, revision)) == twin(week, name, day) for name in names}


LISTS = {
    "main": [[*BASE_PREDICTORS, A65], [*BASE_PREDICTORS, A65, EXAA]],
    "base": [BASE_PREDICTORS],
}
NAMES_OF = {list_hash(names): names for lists in LISTS.values() for names in lists}


def lists_of(week: Week, day: datetime.date, revision: int, model: str = "main") -> List[List[str]]:
    """The predictor lists `model` read in a revision, by the list hashes its snapshot records."""
    snap = read_snapshot(week.snapshot_dir(day, revision) / "inputs.parquet")
    hashes = {entry["list_hash"] for entry in snap.lists if entry["model"] == model}
    return [NAMES_OF[hashed] for hashed in sorted(hashes)]


def rungs(records) -> Counter:
    return Counter(int(r["rung"]) for r in records if r.get("status", "ok") == "ok")


def polls(result) -> List[Dict[str, Any]]:
    return list(result.waits)


def berlin(stamp: Any) -> pd.Timestamp:
    return pd.Timestamp(stamp).tz_convert(ZONE)


def between(stamp: Any, day: datetime.date, first: str, last: str) -> bool:
    return local(day, first) <= pd.Timestamp(stamp) <= local(day, last)


# --- the table, row by row ------------------------------------------------------------------------------------------------------


def test_the_calls_run_one_after_another_and_give_the_table_s_status_exit_code_and_revision(week):
    table = [
        ("setup refit", None, 0, None),
        ("wed C1", "ok", 0, 1),
        ("wed C2", "ok", 0, 2),
        ("wed deliver r2", None, 0, None),
        ("thu C1", "ok", 0, 1),
        ("thu C2", "ok", 0, 2),
        ("fri C1", "fallback", 10, 1),
        ("fri C2", "fallback", 10, 2),  # the spec's table: unchanged (r1); see FINAL
        ("sat C1", "fallback", 10, 1),
        ("sat C2", "ok", 0, 2),
        ("sun busy 09:30", None, 75, None),
        ("sun C1 09:45", "ok", 0, 1),
        ("sun ensure 10:30", "ok", 0, 1),
        ("sun C2 crash", None, 1, None),
        ("sun ensure 11:30", "ok", 0, 1),
        ("sun rerun 11:35", "ok", 0, 3),
        ("sun refit 22:00", None, 1, None),
        ("mon C1", "ok", 0, 1),
        ("mon refit 11:00", None, 0, None),
        ("mon C2", "ok", 0, 2),
        ("mon dry run 11:50", "ok", 0, 2),  # unchanged against r2: nothing a commit would write
        ("mon deliver r2", None, 0, None),
        ("mon deliver r3", None, 1, None),
        ("mon deliver r2 dry", None, 0, None),
        ("mon C2 late", None, 30, None),
        ("tue C1", "fallback", 10, 1),
        ("tue C2", "fallback", 10, 2),  # the spec's table: unchanged (r1); see FINAL
        ("monitor", None, 3, None),
    ]
    got = [(label, call.status, call.exit_code, call.revision) for label, call in week.calls.items()]
    assert got == table
    ends = [(call.at, call.ended) for call in week.calls.values()]
    assert all(end <= next_start for (_, end), (next_start, _) in zip(ends, ends[1:]))  # one after another


def test_wednesday_r1_waits_for_a65_until_the_10_15_poll_and_reads_no_exaa(week):
    # The spec's "r1 at 10:15" is 10:16:35 here: each poll sleeps 5 minutes from the end of its refresh, and a refresh takes 5 s
    # on the clock (five requests, 1 s each), so the polls drift by 5 s each: 08:45:05, 08:50:10, ..., 10:11:30, 10:16:35.
    # Likewise Thursday's "about 11:41" is 11:40:05 and Sunday's 10:15 is 10:15:35.
    result = week.calls["wed C1"].result
    waits = polls(result)
    assert waits[-1]["outcome"] == "in" and between(waits[-1]["at"], WED, "10:12", "10:20")
    assert all(A65 in poll["lacking"] for poll in waits[:-1]) and between(waits[-2]["at"], WED, "10:05", "10:12")
    assert between(result.call["finished"], WED, "10:12", "10:20")
    assert lists_of(week, WED, 1) and all(EXAA not in names for names in lists_of(week, WED, 1))
    assert week.json(WED, 1, "waits.json")["polls"] == jsonable(waits)


def jsonable(value):
    return json.loads(json.dumps(value, default=str))


def test_wednesday_r2_reads_exaa_and_every_model_equals_the_backtest(week):
    assert all(EXAA in names for names in lists_of(week, WED, 2))
    assert equals_twin(week, WED, 2) == dict.fromkeys(NAMES, True)
    # main's list with EXAA wasn't in the 10-19 fit (stored on Sunday night, before EXAA of Monday was out): its units were
    # fitted now, and fits.json says what they read beside the stored fit's sha256.
    entry = week.json(WED, 2, "fits.json")["models"]["main"]["2026-10-19"]
    assert entry["source"] == "stored" and entry["inline_units"] == 96 and len(entry["inline_rows_sha256"]) == 64
    assert "inline_rows_sha256" not in week.json(WED, 1, "fits.json")["models"]["main"]["2026-10-19"]


def test_wednesday_r2_is_delivered(week):
    line = week.calls["wed deliver r2"].result
    assert (line["issue"], line["revision"]) == (WED.isoformat(), 2)
    assert line["receipt_sha256"] == hashlib.sha256(week.receipt.read_bytes()).hexdigest()


def test_thursday_r2_waits_twenty_minutes_for_the_late_exaa_and_equals_the_backtest(week):
    result = week.calls["thu C2"].result
    waits = polls(result)
    assert between(waits[0]["at"], THU, "11:20", "11:21") and EXAA in waits[0]["lacking"]
    assert waits[-1]["outcome"] == "in" and between(waits[-1]["at"], THU, "11:37", "11:42")
    assert (berlin(waits[-1]["at"]) - berlin(waits[0]["at"])) >= pd.Timedelta(minutes=19)
    assert between(result.call["finished"], THU, "11:40", "11:42")  # "r2 at about 11:41"
    assert all(EXAA in names for names in lists_of(week, THU, 2))
    assert equals_twin(week, THU, 2) == dict.fromkeys(NAMES, True)


def test_friday_gives_up_on_a65_at_10_45_and_base_serves(week):
    result = week.calls["fri C1"].result
    waits = polls(result)
    assert waits[-1]["outcome"] == "gave up" and waits[-1]["gave_up"] == [A65]
    assert between(waits[-1]["at"], FRI, "10:45", "10:46")
    assert (
        result.models["main"]["skipped"] == {"missing_test_predictors": 96} and result.models["main"]["forecasts"] == 0
    )
    assert rungs(result.records["live"]) == {1: 96}
    assert {r["member"] for r in result.records["live"]} == {"base"}
    quality = week.json(FRI, 1, "quality.json")
    assert quality["columns"][A65]["missing"] == 96 and quality["columns"][A65]["issue_rows"] == 96


def test_friday_r2_records_main_s_skips_with_the_list_that_reads_exaa(week):
    """EXAA was out at 11:08, so at 11:20 main's list has it; main is still skipped on every key (no A65), base still serves. The
    records differ from r1 in main's list alone, and that is a revision: r2's skips are the backtest's, list and all."""
    result = week.calls["fri C2"].result
    assert (result.unchanged, result.revision) == (False, 2)
    assert polls(result)[-1]["outcome"] == "in"
    without, with_exaa = (list_hash(names) for names in LISTS["main"])
    first, second = (week.revision("main", FRI, number) for number in (1, 2))
    assert {r["list_hash"] for r in first} == {without} and {r["list_hash"] for r in second} == {with_exaa}
    assert {r["status"] for r in second} == {"missing_test_predictors"}
    for name in ("base", "naive", "live"):
        assert keyed(week.revision(name, FRI, 1)) == keyed(week.revision(name, FRI, 2))

    def without_the_list(rows):
        return once({k: v for k, v in r.items() if k not in ("list_hash", "predictor_list")} for r in rows)

    assert without_the_list(first) == without_the_list(second)  # main's list alone differs
    assert equals_twin(week, FRI, 2) == dict.fromkeys(NAMES, True)


def test_saturday_r1_in_the_outage_is_naive_s_from_the_settled_cache_and_exports_100_products(week):
    result = week.calls["sat C1"].result
    assert rungs(result.records["live"]) == {2: 96} and {r["member"] for r in result.records["live"]} == {"naive"}
    for model in ("main", "base"):
        assert result.models[model]["skipped"] == {"missing_test_predictors": 96}
    waits = polls(result)
    assert waits[-1]["outcome"] == "gave up" and between(waits[-1]["at"], SAT, "10:45", "10:48")
    # Every poll's refresh: ENTSO-E out of time within the 120 s budget, the calendar fine. The spec's table says "failed"; the
    # refresh record says `timeout`, its status for a source that ran out of the refresh's time (a 503 on each attempt, 20 s
    # each, then a wait or a request the budget no longer fits), and `failed` for one that refused or gave up within it.
    for poll in waits:
        assert poll["refresh"] == {"EntsoeSource #1": "timeout", "CalendarSource #2": "ok"}
    refresh = week.json(SAT, 1, "refresh.json")
    entsoe = refresh["sources"][0]
    assert entsoe["status"] == "timeout" and entsoe["seconds"] <= 120 and refresh["seconds"] <= 120
    # The cache gave the rows settled when it was last written (Friday 11:20, two days back to local midnight): up to Tuesday.
    settled = local(datetime.date(2026, 10, 21), "00:00") - pd.Timedelta(minutes=15)
    assert {berlin(stamp) for stamp in entsoe["last_valid"].values()} == {settled}
    assert not result.checks["failed"]
    # The server: 503 after 20 s during the outage, a timeout where the 20 s didn't fit what was left of the budget, then 200.
    answers = Counter(
        (at < local(SAT, "11:25"), status)
        for at, _, _, status in week.fake.requests
        if at >= local(SAT, "08:00") and at < local(SUN, "00:00")
    )
    assert answers[(True, 503)] and answers[(True, "timeout")] and not answers[(True, 200)] and answers[(False, 200)]
    table = pd.read_csv(week.sat_export)
    assert len(table) == 100 and (table["target_date"] == SUN.isoformat()).all()
    assert table["prediction"].notna().all() and table["occurrence"].max() == 2


def test_saturday_r2_after_the_outage_is_rung_0_and_equals_the_backtest(week):
    result = week.calls["sat C2"].result
    first, second = polls(result)[:2]
    assert first["refresh"]["EntsoeSource #1"] == "timeout" and between(first["at"], SAT, "11:21", "11:23")
    assert second["refresh"]["EntsoeSource #1"] == "ok" and second["outcome"] == "in"
    assert between(result.call["finished"], SAT, "11:25", "11:30")
    assert rungs(result.records["live"]) == {0: 96}
    assert equals_twin(week, SAT, 2) == dict.fromkeys(NAMES, True)


def test_sunday_s_catch_up_finds_the_lock_held_and_writes_nothing(week):
    call = week.calls["sun busy 09:30"]
    assert call.exit_code == 75 and call.before == call.after
    sunday = [line for line in week.store().calls() if line["issue"] == SUN.isoformat()]
    assert all(pd.Timestamp(line["started"]) >= local(SUN, "09:45") for line in sunday)  # no line of the 09:30 call


def test_sunday_r1_then_ensure_is_a_noop(week):
    assert between(week.calls["sun C1 09:45"].result.call["finished"], SUN, "10:12", "10:20")
    noop = week.calls["sun ensure 10:30"].result
    assert noop.noop and noop.revision == 1 and noop.call is None


def test_sunday_s_crashed_r2_stays_on_file_uncommitted_and_unseen(week):
    store = week.store()
    lines_of_day = [line for line in store.calls() if line["issue"] == SUN.isoformat()]
    crashed = [line for line in lines_of_day if line.get("revision") == 2]
    assert [line["status"] for line in crashed] == ["begun", "error"] and not any(line["committed"] for line in crashed)
    assert "injected crash" in crashed[-1]["error"]
    assert any(r["revision"] == 2 for r in read_records_raw(week, "main", SUN))  # main's records were written
    assert not any(r["revision"] == 2 for r in read_records_raw(week, "base", SUN))
    assert week.calls["sun ensure 11:30"].result.noop and week.calls["sun ensure 11:30"].revision == 1


def read_records_raw(week: Week, name: str, day: datetime.date) -> List[Dict[str, Any]]:
    """Every line of a JSONL file for the day, committed or not."""
    if not week.path(name).name.endswith(".jsonl"):
        pytest.skip("raw lines of a Parquet directory")
    return [r for r in lines(week.path(name)) if r["run_date"] == day.isoformat()]


def test_sunday_r3_is_the_one_readers_see_and_equals_the_backtest(week):
    assert sorted(week.store().committed(SUN)[SUN.isoformat()]) == [1, 3]
    for name in NAMES:
        assert {r["revision"] for r in week.visible(name) if r["run_date"] == SUN.isoformat()} == {3}
    assert all(EXAA in names for names in lists_of(week, SUN, 3))
    assert equals_twin(week, SUN, 3) == dict.fromkeys(NAMES, True)


def test_sunday_s_failed_refit_stores_no_fit_of_the_10_26_anchor(week):
    anchors = week.calls["sun refit 22:00"].after
    assert anchors == {name: [datetime.date(2026, 10, 19)] for name in ("main", "base")}
    served = week.calls["mon C1"].result.models
    for model in ("main", "base"):
        assert served[model]["fits"]["2026-10-26"]["source"] == "previous"
    refit = week.calls["mon refit 11:00"].result
    assert refit.fitted == {"main": ["2026-10-26"], "base": ["2026-10-26"]}


def test_monday_r1_is_served_from_the_10_19_fit_seven_days_old_as_by_hand(week):
    result = week.calls["mon C1"].result
    for model in ("main", "base"):
        entry = result.models[model]["fits"]["2026-10-26"]
        assert (entry["used"], entry["fit_age_days"]) == ("2026-10-19", 7)
        assert {r["refit_anchor"] for r in result.records[model] if r["status"] == "ok"} == {"2026-10-19"}
    # By hand: each forecast from its unit of the stored 10-19 fit and the issue row the revision's snapshot holds.
    snap = read_snapshot(week.snapshot_dir(MON, 1) / "inputs.parquet")
    pipeline = model_pipeline(data_pipeline())
    for model in pipeline.models[:2]:
        stored = read_fit(fit_dir(week.live, model.name, datetime.date(2026, 10, 19)), model)
        for record in week.revision(model.name, MON, 1):
            key = (record["hour"], 1, record["list_hash"], 0)
            row = snap.inputs[NAMES_OF[record["list_hash"]]].to_numpy(dtype=float)[[record["hour"]]]
            prediction, _ = stored.units[key].forecasts(model, row)[0]
            assert prediction == record["prediction"], (model.name, record["hour"])


def test_monday_refit_on_saturday_noon_is_refused_until_sunday_s_prices_are_out(tmp_path):
    """The refit of the 10-26 anchor trains on prices up to Sunday 10-25, out by their rule at 13:00 on Saturday (the server's
    12:57): at 12:00 it is refused, at 13:05 it fits."""
    with pytest.MonkeyPatch.context() as mp, threadpool_limits(limits=1), quiet():
        prepare(mp, tmp_path)
        FakeEntsoe().install(mp)
        clock = FakeClock(local(SAT, "12:00"))
        with use_clock(clock):
            dp = data_pipeline()
            mp_ = model_pipeline(dp)
            frame = dp.run(*data_span(mp_.models, MON, [1], dp), cache=True)
            with pytest.raises(FitTooEarlyError, match="2026-10-24 11:00 UTC"):
                mp_.refit(frame, tmp_path / "live", issue=MON, horizon=1, freq="15min")
            assert not (tmp_path / "live" / "fits").exists()
            clock.advance(65 * 60)
            frame = dp.run(*data_span(mp_.models, MON, [1], dp), cache=True)
            fitted = mp_.refit(frame, tmp_path / "live", issue=MON, horizon=1, freq="15min")
    assert fitted.fitted == {"main": ["2026-10-26"], "base": ["2026-10-26"]}


def test_monday_r2_has_the_new_fit_the_revised_a65_and_exaa_and_equals_the_backtest(week):
    result = week.calls["mon C2"].result
    for model in ("main", "base"):
        assert result.models[model]["fits"]["2026-10-26"]["source"] == "stored"
    first = read_snapshot(week.snapshot_dir(MON, 1) / "inputs.parquet")
    second = read_snapshot(week.snapshot_dir(MON, 2) / "inputs.parquet")
    assert first.hash != second.hash
    assert (second.inputs[A65] - first.inputs[A65] == 1500).all()  # the revision moved every cell of Tuesday
    assert all(EXAA in names for names in lists_of(week, MON, 2))
    assert equals_twin(week, MON, 2) == dict.fromkeys(NAMES, True)


def test_monday_s_dry_run_changes_no_file(week):
    call = week.calls["mon dry run 11:50"]
    assert call.result.dry_run and call.result.unchanged and call.before == call.after


def test_monday_s_deliveries_record_r2_once(week):
    assert [week.calls[label].exit_code for label in ("mon deliver r2", "mon deliver r3", "mon deliver r2 dry")] == [
        0,
        1,
        0,
    ]
    assert [line["revision"] for line in week.store().deliveries(MON)] == [2]


def test_monday_s_call_after_the_deadline_is_late_and_writes_its_call_line_alone(week):
    call = week.calls["mon C2 late"]
    changed = [name for name in call.after if call.after[name] != call.before.get(name)]
    assert changed == ["issues.jsonl"]
    line = [c for c in week.store().calls() if c["issue"] == MON.isoformat()][-1]
    assert (line["status"], line["exit_code"], line["revision"], line["committed"]) == ("late", 30, None, False)


def test_tuesday_s_implausible_a65_fails_main_s_range_check_and_base_serves(week):
    result = week.calls["tue C1"].result
    assert {r["status"] for r in result.records["main"]} == {"check_failed"}
    assert rungs(result.records["live"]) == {1: 96} and {r["member"] for r in result.records["live"]} == {"base"}
    checks = week.json(TUE, 1, "checks.json")
    assert [(f["member"], sorted(f["failed"])) for f in checks["failed"]] == [("main", ["range"])]
    assert checks["failed"][0]["failed"]["range"].startswith("1 outside [-500, 1000] (56 at ")
    assert week.json(TUE, 1, "quality.json")["columns"][A65]["missing"] == 0


def test_tuesday_r2_still_falls_back_and_equals_the_backtest_with_the_same_checks(week):
    """The spec's table has Tuesday's C2 unchanged. Every forecast is (base serves at rung 1 again), but main's check_failed
    records name the failing value, and with EXAA in its list main forecasts that quarter-hour otherwise: a new detail, so r2."""
    result = week.calls["tue C2"].result
    assert (result.status, result.unchanged, result.revision) == ("fallback", False, 2)
    first, second = (week.revision("main", TUE, number) for number in (1, 2))
    assert [r["detail"] for r in first] != [r["detail"] for r in second]
    assert {r["status"] for r in second} == {"check_failed"} and all(EXAA in names for names in lists_of(week, TUE, 2))
    for name in ("base", "naive", "live"):
        assert keyed(week.revision(name, TUE, 1)) == keyed(week.revision(name, TUE, 2))
    assert {r["status"] for r in lines(week.backtest / "main.jsonl") if r["run_date"] == TUE.isoformat()} == {
        "check_failed"
    }
    assert equals_twin(week, TUE, 2) == dict.fromkeys(NAMES, True)


def test_the_monitor_scores_the_week_and_alerts_on_deliveries(week):
    report = week.monitor
    live = report.tables().set_index("member").loc["live"]
    assert (live["issue_days"], live["committed_days"], live["delivered_days"]) == (7, 7, 2)
    assert live["participation"] == 1.0 and live["delivered"] == pytest.approx(2 / 7)
    assert live["forecasts"] == 668 and live["unscored"] == 4  # 672 keys: the repeated hour's four have no actual
    # Input-quality days are days whose quality.json, in the policy's revision, records a missing or imputed cell or a source that
    # failed or timed out (Dawid, 2026-11-17). Implausible values are the checks' (checks.json, `check_failed`). The spec's table
    # says Fri, Sat, Tue: Saturday's last revision refreshed after the outage, and Tuesday's 1e6 is a value, not a missing cell.
    assert problem_days(report) == [FRI.isoformat()] and live["quality_days"] == 1 and live["quality_records"] == 7
    assert problem_days(week.monitor_first) == [FRI.isoformat(), SAT.isoformat()]
    first = week.monitor_first.quality().assign(issue=lambda q: q["issue"].astype(str).str[:10]).set_index("issue")
    assert first.loc[SAT.isoformat(), "failed_sources"] == "EntsoeSource #1"  # timed out on every poll
    latest = report.quality().assign(issue=lambda q: q["issue"].astype(str).str[:10]).set_index("issue")
    assert not latest.loc[TUE.isoformat(), "problem"] and live["check_failed"] == 0
    assert report.tables().set_index("member").loc["main", "check_failed_days"] == 1  # Tuesday's 1e6, in the checks
    assert [alert.rule for alert in report.alerts] == ["delivered_below"]
    assert rung_counts(report) == {0: 5, 1: 2, 2: 0}
    assert rung_counts(week.monitor_first) == {0: 4, 1: 2, 2: 1}


def problem_days(report) -> List[str]:
    quality = report.quality()
    return sorted(str(day)[:10] for day in quality.loc[quality["problem"].astype(bool), "issue"])


def rung_counts(report) -> Dict[int, int]:
    table = report.rungs()
    table = table[(table["member"] == "live") & (table["policy"] == report.policy) & (table["rung"] >= 0)]
    return {int(rung): int(days) for rung, days in zip(table["rung"], table["days"])}


# --- the assertions beyond the table --------------------------------------------------------------------------------------------


def test_every_committed_revision_read_back_is_what_its_call_returned(week):
    for call in week.calls.values():
        result = call.result
        if result is None or not hasattr(result, "records") or result.revision is None:
            continue
        if result.noop or result.unchanged or result.dry_run:
            continue
        day = datetime.date.fromisoformat(result.issue)
        for name in NAMES:
            on_file = [
                {k: v for k, v in r.items() if k not in ("revision", "issued_at", "mode", "snapshot")}
                for r in week.revision(name, day, result.revision)
            ]
            assert keyed(on_file) == keyed(result.records[name]), (call.label, name)
        frame = result.frame()
        assert len(frame) == 96 and (frame["run_date"] == result.issue).all()


def test_each_policy_reads_the_table_s_revisions(week):
    expected = {"latest": FINAL, "first": dict.fromkeys(WEEK, 1), "delivered": DELIVERED}
    for policy, revisions in expected.items():
        for name in NAMES:
            seen = {}
            for record in week.visible(name, policy):
                seen.setdefault(record["run_date"], set()).add(record["revision"])
            assert seen == {day.isoformat(): {number} for day, number in revisions.items()}, (policy, name)


def test_no_call_committed_after_its_deadline_and_numbers_are_never_reused(week):
    store = week.store()
    for day, revisions in store.committed().items():
        for commit in revisions.values():
            assert commit.finished <= commit.deadline == local(datetime.date.fromisoformat(day), DEADLINE)
    begun = {}  # every revision number a call took, in the order taken: committed, crashed, or none (late, unchanged lines)
    for line in store.calls():
        if line["status"] == "begun":
            begun.setdefault(line["issue"], []).append(line["revision"])
    assert begun == {day.isoformat(): list(range(1, (3 if day == SUN else FINAL[day]) + 1)) for day in WEEK}
    committed = Counter((line["issue"], line["revision"]) for line in store.calls() if line.get("committed"))
    assert set(committed.values()) == {1}


def test_no_key_mixes_revisions(week):
    for name in NAMES:
        for policy in ("latest", "first", "delivered"):
            by_day = {}
            for record in week.visible(name, policy):
                by_day.setdefault(record["run_date"], set()).add(record["revision"])
            assert all(len(numbers) == 1 for numbers in by_day.values()), (name, policy)


def test_no_live_record_has_an_actual(week):
    """At 11:45 no target day's price is out (12:57): every live record's actual is missing, whatever its revision."""
    for day, revisions in week.store().committed().items():
        for number in revisions:
            for name in NAMES:
                actuals = [r.get("actual") for r in week.revision(name, datetime.date.fromisoformat(day), number)]
                assert actuals and all(a is None or a != a for a in actuals), (name, day, number)


def test_no_model_received_a_nan_input(week):
    assert week.nan_rows == 0 and week.predicted_rows > 0


def test_the_final_revisions_equal_the_backtest_and_the_first_differ_only_where_the_table_says(week):
    for day in WEEK:
        assert equals_twin(week, day, FINAL[day]) == dict.fromkeys(NAMES, True), day
        first = equals_twin(week, day, 1)
        assert {name for name, same in first.items() if not same} == FIRST_DIFFERS[day], day
    # Friday and Tuesday through base, statuses and skip details included.
    for day, status in ((FRI, "missing_test_predictors"), (TUE, "check_failed")):
        assert {r["status"] for r in week.revision("main", day, FINAL[day])} == {status}
        assert {r["member"] for r in week.revision("live", day, FINAL[day])} == {"base"}


def test_every_quality_record_names_a65_s_last_valid_timestamp(week):
    for day, revisions in week.store().committed().items():
        issue = datetime.date.fromisoformat(day)
        for number in revisions:
            quality = week.json(issue, number, "quality.json")
            refresh = week.json(issue, number, "refresh.json")
            entsoe = refresh["sources"][0]
            last = pd.Timestamp(quality["sources"]["EntsoeSource #1"]["last_valid"]["load_forecast"])
            assert last == pd.Timestamp(entsoe["last_valid"]["load_forecast"])
            if entsoe["status"] == "ok":  # what the server held when it was asked, within the range asked for
                held = archive(week.fake, pd.Timestamp(refresh["started"]))["load_forecast"].dropna()
                held = held[held.index < pd.Timestamp(refresh["end"])]
                assert last == held.index[-1], (day, number)
            expected = {
                (FRI, 1): local(FRI, "23:45"),
                (FRI, 2): local(FRI, "23:45"),
                (SAT, 1): local(datetime.date(2026, 10, 20), "23:45"),
            }.get((issue, number), local(issue + datetime.timedelta(days=1), "23:45"))
            assert last == expected, (day, number)


def test_a_second_pass_in_parquet_gives_the_same_bytes_and_records(week, tmp_path):
    """The whole week again into a fresh directory, with `save_format="parquet"` (the models' records in Parquet directories,
    the ensemble's in JSONL as always): every exit code, every snapshot file's bytes, the ensemble's JSONL lines (sorted), and
    every model's and the ensemble's records of each committed revision as sorted JSON lines are the first pass's.

    One pass answers both of the spec's checks, determinism (a second run) and format equivalence (Parquet), since a third pass
    would put the module past 60 s (each pass is about 21 s on one process). So it checks the sorted records and the snapshot
    bytes, not the JSONL bytes of the models' files of a second JSONL run."""
    again = run_week(tmp_path / "again", save_format="parquet")
    assert (again.live / "main.parquet").is_dir() and not (again.live / "main.jsonl").exists()
    first, second = week.summary(), again.summary()
    assert first["exit_codes"] == second["exit_codes"]
    assert first["records"].keys() == second["records"].keys()
    assert len(first["records"]) == 4 * 14  # 14 committed revisions, four names each
    differ = [key for key in first["records"] if first["records"][key] != second["records"][key]]
    assert not differ
    assert second["files"] == {"live.jsonl": first["files"]["live.jsonl"]}
    assert first["snapshots"].keys() == second["snapshots"].keys()
    differ = [name for name in first["snapshots"] if first["snapshots"][name] != second["snapshots"][name]]
    assert not differ
