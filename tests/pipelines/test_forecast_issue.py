"""`ModelPipeline.forecast()` (step 8e): one live call, each branch of its flow alone on a small frame, and the identity it rests
on: a live forecast over a frame equal to the backtest's is the backtest's record to the bit.

The identity is checked per kind of list (fixed, a predictor function, a matching list, a list of training windows), per refit
(None, "7D", "once") and engine (off, on), each against a backtest of the same pipeline over the same frame: every record but
`day_in_test` (its place in its run) and the live keys, compared as JSON text, skips included, and an ensemble's too. Then the
branches: busy, ensure (noop), late (before and after serving, and the commit's backstop), unchanged, dry run, wait and give-up,
a crash between models, the live `published` of decision 7, missing inputs, fallback and skipped statuses, checks, the export
grids, `run(forecast_only=True, save_dir=...)`, a late `ctx.value()` at "raise", and `verify(through="serve")`.
"""

import datetime
import hashlib
import json
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from epftoolbox2.clock import FakeClock, use_clock
from epftoolbox2.ensembles import AverageEnsemble, FallbackEnsemble
from epftoolbox2.information_set import Availability, InformationSet, InformationSetError
from epftoolbox2.models import HistGBModel, LEARModel, NaiveModel, OLSModel
from epftoolbox2.models import _engine_worker
from epftoolbox2.models._fits import FitTooEarlyError
from epftoolbox2.models.context import Use
from epftoolbox2.models.residual import ResidualModel
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline, _live
from epftoolbox2.data.sources import CsvSource
from epftoolbox2.data.sources.base import DataSource
from epftoolbox2.data.transformers import LambdaTransformer, RollingStatsTransformer, TimezoneTransformer
from epftoolbox2.pipelines import DataPipeline
from epftoolbox2.pipelines._issue_records import read_snapshot
from epftoolbox2.results.checks import ForecastChecks
from epftoolbox2.results.live import LiveStore
from epftoolbox2.results.manifest import ManifestMismatchError
from epftoolbox2.results.store import LiveDirectoryError
from epftoolbox2.results.store import read_records
from tests.synthetic import PREDICTORS, make_market_data

pytestmark = pytest.mark.usefixtures("no_network")

RULES = {
    "load": Availability.published("10:00", 10),
    "wind": Availability.published("10:00", 10),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published("13:00", 1),
    "extra": Availability.published("11:15", 1),
    "late": Availability.published("12:30", 1),
}
INFO = InformationSet("11:45", "Europe/Warsaw", rules=RULES)
DAYS = ["2023-03-06", "2023-03-07", "2023-03-08"]  # a Monday to a Wednesday: a "7D" block from Monday 6 March
ISSUE = "2023-03-07"
AT = pd.Timestamp("2023-03-07 10:30", tz="UTC")  # 11:30 in Warsaw: before the 11:45 issue time and the 11:55 deadline


class InProcessPool:
    """Stands in for the worker pools, whose start is most of these tests' time."""

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


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    for module in ("epftoolbox2.models.base", "epftoolbox2.models._engine", "epftoolbox2.models._fits"):
        monkeypatch.setattr(f"{module}.ProcessPoolExecutor", InProcessPool)
    with threadpool_limits(limits=1):
        yield


def market(days=60, freq="1h", start="2023-01-20"):
    data = make_market_data(start=start, days=days, freq=freq, horizon=2, noise=2.0)
    rng = np.random.default_rng(1)
    for h in (1, 2):
        data[f"extra_d+{h}"] = rng.normal(0, 10, len(data))
        data[f"late_d+{h}"] = rng.normal(0, 10, len(data))
    return data


DATA = market()


def pipeline(*models, ensembles=(), engine="off", info=INFO, **kwargs):
    out = ModelPipeline(information_set=info, engine=engine, **kwargs)
    for model in models:
        out.add_model(model)
    for ensemble in ensembles:
        out.add_ensemble(ensemble)
    return out


def ols(name="a", predictors=PREDICTORS, window=21, **kwargs):
    return OLSModel(predictors, training_window=window, name=name, **kwargs)


def keyed(records, day=None):
    """(run_date, hour, horizon) -> the record as JSON text without `day_in_test` and the live keys."""
    drop = ("day_in_test", "revision", "issued_at", "mode", "snapshot")
    return {
        (r["run_date"], r["hour"], r["horizon"]): json.dumps(
            {k: v for k, v in r.items() if k not in drop}, sort_keys=True
        )
        for r in records
        if day is None or r["run_date"] == day
    }


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def backtest(make, tmp_path, days=DAYS, data=DATA, horizon=2):
    folder = tmp_path / "backtest"
    make().run(data, days[0], days[-1], horizon=horizon, save_dir=str(folder))
    return folder


def snapshot_of(folder):
    """Every file under `folder` with its bytes and modification time."""
    return {
        str(path.relative_to(folder)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


# --- live equals backtest ------------------------------------------------------------------------------------------------------


def by_load(ctx):
    """A list chosen by the issue day: what a function's plan gives on the live frame must be the backtest's."""
    return PREDICTORS if ctx.issue_day.day % 2 else PREDICTORS[:2]


def matching(ctx):
    return Use(PREDICTORS if ctx.target_day.weekday() < 5 else PREDICTORS[:2], train_on="matching")


KINDS = {
    "fixed": lambda refit, **kw: ols(refit=refit, **kw),
    "function": lambda refit, **kw: OLSModel(by_load, training_window=21, name="a", refit=refit, **kw),
    "matching": lambda refit, **kw: OLSModel(matching, training_window=28, name="a", refit=refit, **kw),
    "windows": lambda refit, **kw: LEARModel(
        PREDICTORS, training_window=[14, 21], criterion="bic", name="a", refit=refit, **kw
    ),
}
CASES = [
    ("fixed", None, "off"),
    ("fixed", "7D", "on"),
    ("function", None, "on"),
    ("function", "7D", "off"),
    ("matching", None, "off"),
    ("windows", None, "on"),
    ("windows", "7D", "off"),
    ("fixed", "once", "off"),
    ("fixed", "once", "on"),
]


@pytest.mark.parametrize("kind, refit, engine", CASES, ids=[f"{k}-{r}-{e}" for k, r, e in CASES])
def test_a_live_forecast_over_the_backtest_s_frame_is_the_backtest_s_record(kind, refit, engine, tmp_path):
    once = {"refit_from": DAYS[0]} if refit == "once" else {}
    if kind == "matching" and refit is not None:
        pytest.skip("a matching list runs with refit=None")

    def make():
        model = KINDS[kind](refit, **once)
        nested = ols("b", PREDICTORS[:2], 14)
        return pipeline(model, nested, ensembles=[FallbackEnsemble(["a", "b"], name="fb")], engine=engine)

    folder = backtest(make, tmp_path)
    live = tmp_path / "live"
    for day in DAYS:
        live_pipeline = make()
        if refit is not None:
            live_pipeline.refit(DATA, live, issue=day, horizon=2)
        result = live_pipeline.forecast(DATA, day, live, horizon=2)
        assert result.status == "ok", (day, result.models)
        for name in ("a", "b", "fb"):
            expected = keyed(lines(folder / f"{name}.jsonl"), day)
            got = keyed(result.records[name])
            assert set(got) == set(expected), name
            differ = [key for key in expected if got[key] != expected[key]]
            assert not differ, f"{name} on {day}: {len(differ)} of {len(expected)} differ, e.g. {differ[:2]}"
            on_file = keyed(lines(live / f"{name}.jsonl"), day)
            assert on_file == got, name  # what the revision wrote is what the call returned
        if refit is not None:
            assert all(entry["source"] == "stored" for entry in result.models["a"]["fits"].values())


def test_a_refit_model_without_a_stored_fit_serves_the_newest_older_one_or_fits_inline(tmp_path):
    def make():
        return pipeline(ols(refit="7D"))

    folder = backtest(make, tmp_path)
    live = tmp_path / "live"
    make().refit(DATA, live, issue="2023-02-27", horizon=2)  # the block before
    older = make().forecast(DATA, ISSUE, live, horizon=2)
    assert older.models["a"]["fits"]["2023-03-06"]["source"] == "previous"
    assert older.models["a"]["fits"]["2023-03-06"]["fit_age_days"] == 8
    inline = make().forecast(DATA, ISSUE, None, horizon=2, fit_missing="fit")
    assert keyed(inline.records["a"]) == keyed(lines(folder / "a.jsonl"), ISSUE)
    skipped = make().forecast(DATA, ISSUE, None, horizon=2, fit_missing="skip")
    assert skipped.status == "skipped" and skipped.missing == 48
    assert {r["status"] for r in skipped.records["a"]} == {"no_stored_fit"}


# --- the branches ----------------------------------------------------------------------------------------------------------


def simple():
    return pipeline(ols(), ols("b", PREDICTORS[:2], 14), ensembles=[FallbackEnsemble(["a", "b"], name="fb")])


def test_a_call_finding_the_lock_held_is_busy_and_writes_nothing(tmp_path):
    live = tmp_path / "live"
    simple().forecast(DATA, ISSUE, live, horizon=2)
    before = snapshot_of(live)
    with LiveStore(live).hold():
        result = simple().forecast(DATA, ISSUE, live, horizon=2)
    assert (result.status, result.exit_code, result.revision) == ("busy", 75, None)
    assert snapshot_of(live) == before


def test_ensure_is_a_noop_on_an_issue_already_committed_by_its_deadline(tmp_path):
    live = tmp_path / "live"
    with use_clock(FakeClock(AT)):
        first = simple().forecast(DATA, ISSUE, live, horizon=2, deadline="11:55")
        before = snapshot_of(live)
        again = simple().forecast(DATA, ISSUE, live, horizon=2, deadline="11:55", ensure=True)
    assert (first.status, first.revision) == ("ok", 1)
    assert (again.status, again.exit_code, again.revision, again.noop) == ("ok", 0, 1, True)
    assert snapshot_of(live) == before
    other = simple().forecast(DATA, "2023-03-08", live, horizon=2, ensure=True)  # another day isn't done
    assert (other.noop, other.revision) == (False, 1)


def test_a_call_past_the_deadline_is_late_and_writes_its_call_line_alone(tmp_path):
    live = tmp_path / "live"
    with use_clock(FakeClock(AT)):
        simple().forecast(DATA, ISSUE, live, horizon=2)
    before = snapshot_of(live)
    with use_clock(FakeClock(pd.Timestamp("2023-03-07 11:00", tz="UTC"))):  # 12:00 in Warsaw
        late = simple().forecast(DATA, ISSUE, live, horizon=2, deadline="11:55")
    assert (late.status, late.exit_code, late.revision) == ("late", 30, None)
    after = snapshot_of(live)
    assert set(after) == set(before)
    assert [name for name in after if after[name] != before[name]] == ["issues.jsonl"]
    line = lines(live / "issues.jsonl")[-1]
    assert (line["status"], line["exit_code"], line["revision"], line["committed"]) == ("late", 30, None, False)


def test_a_call_whose_serving_runs_past_the_deadline_commits_nothing(tmp_path, monkeypatch):
    live = tmp_path / "live"
    clock = FakeClock(AT)
    serve_models = _live.serve_models

    def slow(*args, **kwargs):
        out = serve_models(*args, **kwargs)
        clock.sleep(30 * 60)  # 12:00 in Warsaw
        return out

    monkeypatch.setattr(_live, "serve_models", slow)
    with use_clock(clock):
        late = simple().forecast(DATA, ISSUE, live, horizon=2, deadline="11:55")
    assert (late.status, late.exit_code, late.revision) == ("late", 30, None)
    assert late.records["fb"]  # computed, not written
    store = LiveStore(live)
    assert store.committed() == {}
    assert not (live / "issues").exists() and not (live / "a.jsonl").exists()


def test_a_commit_after_the_deadline_is_refused_and_its_revision_never_seen(tmp_path, monkeypatch):
    live = tmp_path / "live"
    clock = FakeClock(AT)
    write_json = _live.records_io.write_json

    def slow(directory, name, content):
        if name == _live.FITS_FILE:
            clock.sleep(30 * 60)
        return write_json(directory, name, content)

    monkeypatch.setattr(_live.records_io, "write_json", slow)
    with use_clock(clock):
        late = simple().forecast(DATA, ISSUE, live, horizon=2, deadline="11:55")
    assert (late.status, late.exit_code) == ("late", 30)
    assert LiveStore(live).committed() == {}
    assert LiveStore(live).next_revision(ISSUE) == 2  # its number is taken


def test_an_unchanged_rerun_writes_no_revision_and_a_changed_input_writes_the_next(tmp_path):
    live = tmp_path / "live"
    first = simple().forecast(DATA, ISSUE, live, horizon=2)
    records = (live / "a.jsonl").read_bytes()
    again = simple().forecast(DATA, ISSUE, live, horizon=2)
    assert (again.status, again.revision, again.unchanged) == ("ok", 1, True)
    assert (live / "a.jsonl").read_bytes() == records
    assert sorted(LiveStore(live).committed()[ISSUE]) == [1]
    assert lines(live / "issues.jsonl")[-1]["unchanged"] is True
    changed = DATA.copy()
    row = changed.index[changed.index.normalize() == pd.Timestamp(ISSUE, tz="Europe/Warsaw")][5]
    changed.loc[row, "load_d+1"] += 50.0
    third = simple().forecast(changed, ISSUE, live, horizon=2)
    assert (third.revision, third.unchanged) == (2, False)
    assert first.snapshot != third.snapshot
    seen = LiveStore(live).visible("latest", None)
    read = list(read_records(live / "a.jsonl", committed=seen))
    assert {r["revision"] for r in read if r["run_date"] == ISSUE} == {2}


def test_a_dry_run_writes_nothing_anywhere(tmp_path):
    live = tmp_path / "live"
    fresh = simple().forecast(DATA, ISSUE, live, horizon=2, dry_run=True)
    assert (fresh.status, fresh.exit_code, fresh.dry_run, fresh.revision) == ("ok", 0, True, 1)
    assert not live.exists()
    simple().forecast(DATA, ISSUE, live, horizon=2)
    before = snapshot_of(live)
    changed = DATA.copy()
    row = changed.index[changed.index.normalize() == pd.Timestamp(ISSUE, tz="Europe/Warsaw")][5]
    changed.loc[row, "load_d+1"] += 50.0
    dry = simple().forecast(changed, ISSUE, live, horizon=2, dry_run=True)
    assert (dry.status, dry.revision, dry.unchanged) == ("ok", 2, False)
    assert snapshot_of(live) == before
    with use_clock(FakeClock(pd.Timestamp("2023-03-07 11:00", tz="UTC"))):
        late = simple().forecast(DATA, ISSUE, live, horizon=2, deadline="11:55", dry_run=True)
    assert (late.status, late.exit_code) == ("late", 30)
    assert snapshot_of(live) == before


class FakeDataPipeline:
    """A data pipeline whose `load_d+1` cells of the issue rows arrive at `arrives` on the clock; each `run` is a refresh."""

    def __init__(self, data, arrives, column="load_d+1"):
        self.data, self.arrives, self.column = data, arrives, column
        self.timezone = "Europe/Warsaw"
        self.calls = []
        self.last_refresh = None

    def run(self, start, end, **options):
        from epftoolbox2._clock import current_clock

        now = current_clock().now()
        self.calls.append((now, start, end, options))
        frame = self.data.copy()
        if self.arrives is None or now < self.arrives:
            rows = frame.index.normalize() == pd.Timestamp(ISSUE, tz="Europe/Warsaw")
            frame.loc[rows, self.column] = np.nan
        return frame


def test_wait_for_polls_on_the_clock_until_the_column_is_in(tmp_path):
    source = FakeDataPipeline(DATA, arrives=pd.Timestamp("2023-03-07 09:12", tz="UTC"))
    clock = FakeClock(pd.Timestamp("2023-03-07 08:50", tz="UTC"))
    with use_clock(clock):
        result = simple().forecast(
            source, ISSUE, tmp_path / "live", horizon=2, wait_for={"load_d+1": "10:45"}, poll="10min", deadline="11:55"
        )
    assert result.status == "ok"
    assert [poll["at"][11:16] for poll in result.waits] == ["08:50", "09:00", "09:10", "09:20"]
    assert result.waits[-1]["outcome"] == "in" and result.waits[0]["lacking"] == {"load_d+1": 24}
    assert clock.now() == pd.Timestamp("2023-03-07 09:20", tz="UTC")
    start, end = source.calls[0][1:3]
    assert start <= "2023-02-06" and end == "2023-03-10"  # the fits' rows back from the issue day, the targets after it
    written = json.loads((tmp_path / "live" / "issues" / ISSUE / "r1" / "waits.json").read_text())
    assert len(written["polls"]) == 4


def test_wait_for_gives_up_at_its_time_and_the_missing_inputs_are_skips(tmp_path):
    source = FakeDataPipeline(DATA, arrives=None)
    clock = FakeClock(pd.Timestamp("2023-03-07 09:00", tz="UTC"))
    no_load = ["wind_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]
    ladder = pipeline(ols(), ols("b", no_load, 14), ensembles=[FallbackEnsemble(["a", "b"], name="fb")])
    with use_clock(clock):
        result = ladder.forecast(
            source, ISSUE, tmp_path / "live", horizon=2, wait_for={"load_d+1": "10:45"}, poll="20min"
        )
    assert result.waits[-1]["outcome"] == "gave up" and result.waits[-1]["gave_up"] == ["load_d+1"]
    assert result.waits[-1]["at"].startswith("2023-03-07T09:45")  # 10:45 in Warsaw: the last sleep ends there
    # `a` reads load_d+1: its horizon-1 forecasts are skips; `b` has no load, so the fallback serves horizon 1 from rung 1.
    assert result.models["a"]["skipped"] == {"missing_test_predictors": 24}
    assert (result.status, result.exit_code) == ("fallback", 10)
    assert result.rungs == {"2023-03-08": 1, "2023-03-09": 0}


def test_wait_for_rules_waits_for_each_input_until_its_publication_plus_grace(tmp_path):
    source = FakeDataPipeline(DATA, arrives=None, column="extra_d+1")
    clock = FakeClock(pd.Timestamp("2023-03-07 09:30", tz="UTC"))
    p = pipeline(OLSModel(exaa_when_out, training_window=21, name="a"))
    with use_clock(clock):
        result = p.forecast(source, ISSUE, None, horizon=2, wait_for="rules", poll="15min", grace="10min")
    # `extra` is out at 11:15 Warsaw (10:15 UTC) by its rule: with 10 minutes of grace it is given up at 10:25 UTC. Load and
    # wind are out ten days before, and `late` after the issue time: neither is waited for.
    assert [poll["at"][11:16] for poll in result.waits] == ["09:30", "09:45", "10:00", "10:15", "10:25"]
    assert result.waits[-1]["gave_up"] == ["extra_d+1"]
    assert clock.now() == pd.Timestamp("2023-03-07 10:25", tz="UTC")
    on_time = FakeDataPipeline(DATA, arrives=pd.Timestamp("2023-03-07 10:05", tz="UTC"), column="extra_d+1")
    with use_clock(FakeClock(pd.Timestamp("2023-03-07 09:30", tz="UTC"))):
        served = p.forecast(on_time, ISSUE, None, horizon=2, wait_for="rules", poll="15min", grace="10min")
    assert served.waits[-1]["outcome"] == "in" and len(served.waits) == 4


def test_a_crash_between_models_leaves_an_invisible_revision_whose_number_is_never_used_again(tmp_path, monkeypatch):
    live = tmp_path / "live"
    from epftoolbox2.results import live as live_module

    opened = live_module.LiveRevision.open
    count = {"n": 0}

    def crash(self, model, *args, **kwargs):
        count["n"] += 1
        if count["n"] == 2:
            raise OSError("disk full")
        return opened(self, model, *args, **kwargs)

    monkeypatch.setattr(live_module.LiveRevision, "open", crash)
    with pytest.raises(OSError, match="disk full"):
        simple().forecast(DATA, ISSUE, live, horizon=2)
    monkeypatch.setattr(live_module.LiveRevision, "open", opened)
    line = lines(live / "issues.jsonl")[-1]
    assert (line["status"], line["exit_code"], line["revision"], line["committed"]) == ("error", 1, 1, False)
    assert "disk full" in line["error"]
    assert LiveStore(live).committed() == {}
    again = simple().forecast(DATA, ISSUE, live, horizon=2)
    assert (again.revision, again.unchanged) == (2, False)
    seen = LiveStore(live).visible("latest", None)
    assert {r["revision"] for r in read_records(live / "a.jsonl", committed=seen)} == {2}


def exaa_when_out(ctx):
    """The first rung of a fallback ladder: `extra` (EXAA, out 11:15) when it is published, else without it."""
    return [*PREDICTORS, "extra_d+{horizon}"] if ctx.published("extra_d+{horizon}") else PREDICTORS


def late_when_out(ctx):
    """`late` is out at 12:30, after the 11:45 issue time: no list may ever use it, live or not."""
    return [*PREDICTORS, "late_d+{horizon}"] if ctx.published("late_d+{horizon}") else PREDICTORS


def test_a_revision_before_an_input_is_out_uses_the_list_without_it_and_the_last_the_backtest_s(tmp_path):
    def make():
        return pipeline(OLSModel(exaa_when_out, training_window=21, name="a"))

    folder = backtest(make, tmp_path)
    early = DATA.copy()
    rows = early.index.normalize() == pd.Timestamp(ISSUE, tz="Europe/Warsaw")
    early.loc[rows, ["extra_d+1", "extra_d+2"]] = np.nan  # 11:10: EXAA isn't out
    live = tmp_path / "live"
    with use_clock(FakeClock(pd.Timestamp("2023-03-07 10:10", tz="UTC"))):
        first = make().forecast(early, ISSUE, live, horizon=2)
    with use_clock(FakeClock(pd.Timestamp("2023-03-07 10:20", tz="UTC"))):
        second = make().forecast(DATA, ISSUE, live, horizon=2)
    hashes = {r["list_hash"] for r in first.records["a"]}
    assert hashes != {r["list_hash"] for r in second.records["a"]}
    assert (first.revision, second.revision) == (1, 2)
    assert keyed(second.records["a"]) == keyed(lines(folder / "a.jsonl"), ISSUE)
    # A column present in the frame now but out only after the issue time by its rule is never used.
    never = pipeline(OLSModel(late_when_out, training_window=21, name="a")).forecast(DATA, ISSUE, None, horizon=2)
    fixed = pipeline(ols()).forecast(DATA, ISSUE, None, horizon=2)
    assert {r["list_hash"] for r in never.records["a"]} == {r["list_hash"] for r in fixed.records["a"]}


def test_an_issue_row_with_a_missing_input_is_a_skip_and_never_reaches_a_model(tmp_path, monkeypatch):
    gappy = DATA.copy()
    row = gappy.index[gappy.index.normalize() == pd.Timestamp(ISSUE, tz="Europe/Warsaw")][7]
    gappy.loc[row, "wind_d+2"] = np.nan
    gappy.loc[row, "price_d-1"] = np.nan
    seen = []
    predict = OLSModel._predict_fitted

    def instrumented(self, fitted, x):
        seen.append(np.isfinite(np.asarray(x)).all())
        return predict(self, fitted, x)

    monkeypatch.setattr(OLSModel, "_predict_fitted", instrumented)
    p = pipeline(ols(refit="7D"), NaiveModel(["price_d-1"], name="naive"))
    live = tmp_path / "live"
    p.refit(gappy, live, issue=ISSUE, horizon=2)
    result = p.forecast(gappy, ISSUE, live, horizon=2)
    assert seen and all(seen)
    skips = [r for r in result.records["a"] if r["status"] != "ok"]
    assert [(r["hour"], r["horizon"], r["status"]) for r in skips] == [(7, 2, "missing_test_predictors")]
    naive = [r for r in result.records["naive"] if r["status"] != "ok"]
    assert {(r["hour"], r["status"]) for r in naive} == {(7, "missing_test_predictors")}
    assert result.status == "skipped" and result.delivered == "naive" and result.missing == 2


def test_checks_turn_a_failing_member_day_into_check_failed_and_the_ladder_moves_on(tmp_path):
    # A naive forecast of a weekday's `is_weekend` is one value all day: `not_constant` fails each of its days, so the ladder
    # serves them from the OLS model, and the checks run before the ensembles.
    flat = NaiveModel(["is_weekend_d+{horizon}"], name="flat")
    ladder = [FallbackEnsemble(["flat", "b"], name="fb")]
    plain = pipeline(flat, ols("b", PREDICTORS[:2], 14), ensembles=ladder).forecast(DATA, ISSUE, None, horizon=2)
    assert plain.status == "ok" and set(plain.rungs.values()) == {0}
    flat = NaiveModel(["is_weekend_d+{horizon}"], name="flat")
    checked = pipeline(flat, ols("b", PREDICTORS[:2], 14), ensembles=ladder, checks=ForecastChecks()).forecast(
        DATA, ISSUE, tmp_path / "live", horizon=2
    )
    assert {r["status"] for r in checked.records["flat"]} == {"check_failed"}
    assert {(f["member"], f["horizon"]) for f in checked.checks["failed"]} == {("flat", 1), ("flat", 2)}
    assert (checked.status, checked.exit_code) == ("fallback", 10)
    assert checked.rungs == {"2023-03-08": 1, "2023-03-09": 1}
    written = json.loads((tmp_path / "live" / "issues" / ISSUE / "r1" / "checks.json").read_text())
    assert written == json.loads(json.dumps(checked.checks))


def test_the_revision_keeps_its_inputs_quality_and_the_fits_it_was_made_from(tmp_path):
    p = pipeline(ols(refit="7D"), ols("b", PREDICTORS[:2], 14))
    live = tmp_path / "live"
    p.refit(DATA, live, issue=ISSUE, horizon=2, models=["a"])
    result = p.forecast(DATA, ISSUE, live, horizon=2)
    folder = live / "issues" / ISSUE / "r1"
    assert sorted(path.name for path in folder.iterdir()) == [
        "checks.json",
        "fits.json",
        "inputs.parquet",
        "quality.json",
        "refresh.json",
        "waits.json",
    ]
    snap = read_snapshot(folder / "inputs.parquet")
    assert snap.hash == result.snapshot
    assert {r["snapshot"] for r in lines(live / "a.jsonl")} == {snap.hash}
    assert set(snap.inputs.columns) == {
        f"{c}_d+{h}" for c in ("load", "wind", "is_weekend", "is_weekday") for h in (1, 2)
    }
    fits = json.loads((folder / "fits.json").read_text())["models"]
    stored = fits["a"]["2023-03-06"]
    assert stored["source"] == "stored" and len(stored["fit_json_sha256"]) == 64
    fit_json = live / "fits" / "a" / "2023-03-06" / "fit.json"
    assert stored["fit_json_sha256"] == hashlib.sha256(fit_json.read_bytes()).hexdigest()
    inline = fits["b"][ISSUE]
    assert (
        inline["source"] == "inline" and inline["rows"]["through"] == "2023-03-06" and len(inline["rows_sha256"]) == 64
    )
    quality = json.loads((folder / "quality.json").read_text())
    assert quality["columns"]["load_d+1"]["missing"] == 0
    assert json.loads((live / "a.manifest.json").read_text())["mode"] == "live"


def test_an_inline_fit_s_identity_moves_with_a_training_cell_it_reads_alone(tmp_path):
    def identity(frame):
        result = pipeline(ols("b", PREDICTORS[:2], 14)).forecast(frame, ISSUE, None, horizon=2)
        return result.models["b"]["fits"][ISSUE]["rows_sha256"]

    revised = DATA.copy()
    row = revised.index[revised.index.normalize() == pd.Timestamp("2023-03-01", tz="Europe/Warsaw")][3]
    revised.loc[row, "price"] += 5.0  # a target of b's span
    unread = DATA.copy()
    unread.loc[row, "is_weekend_d+1"] = 1.0 - unread.loc[row, "is_weekend_d+1"]  # a column b doesn't read
    assert identity(DATA) == identity(DATA.copy())
    assert identity(revised) != identity(DATA)
    assert identity(unread) == identity(DATA)


def test_the_export_writes_every_market_product_and_the_repeated_hour_twice(tmp_path):
    data = market(days=40, freq="15min", start="2023-09-25")
    p = pipeline(ols(window=14))
    result = p.forecast(data, "2023-10-28", None, horizon=1, freq="15min")
    products = result.table("products")
    periods = result.table("periods")
    assert len(periods) == 96 and len(products) == 100
    repeated = products[products["slot"].isin([8, 9, 10, 11])]
    assert list(repeated["occurrence"]) == [1, 1, 1, 1, 2, 2, 2, 2]
    assert products["start_utc"].is_unique
    first, second = repeated.iloc[0], repeated.iloc[4]
    assert first["prediction"] == second["prediction"]
    assert (pd.Timestamp(second["start_utc"]) - pd.Timestamp(first["start_utc"])) == pd.Timedelta(hours=1)
    path = result.export(tmp_path / "out.csv")
    assert len(pd.read_csv(path)) == 100
    path = result.export(tmp_path / "out.json", format="json", grid="periods")
    assert len(json.loads(path.read_text())) == 96


def test_run_forecast_only_into_a_save_dir_raises_and_points_to_forecast(tmp_path):
    with pytest.raises(ValueError, match="forecast\\(\\)"):
        simple().run(DATA, horizon=1, forecast_only=True, save_dir=str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()
    simple().run(DATA, ISSUE, ISSUE, horizon=1, forecast_only=True)  # without save_dir it works as before


def reads_late(ctx):
    return PREDICTORS if ctx.value("late_d+{horizon}") > -1e9 else PREDICTORS[:2]


def test_a_late_value_at_raise_still_raises_live_and_the_call_says_so(tmp_path):
    p = pipeline(OLSModel(reads_late, training_window=21, name="a"), availability="raise")
    live = tmp_path / "live"
    with pytest.raises(InformationSetError, match="late_d"):
        p.forecast(DATA, ISSUE, live, horizon=2)
    line = lines(live / "issues.jsonl")[-1]
    assert (line["status"], line["exit_code"], line["committed"]) == ("error", 1, False)
    assert "InformationSetError" in line["error"]


def test_a_top_ensemble_must_be_frozen_first(tmp_path):
    p = pipeline(ols(), ols("b", PREDICTORS[:2], 14), ensembles=[AverageEnsemble(["a", "b"], top=1, name="top")])
    with pytest.raises(ValueError, match="frozen"):
        p.forecast(DATA, ISSUE, tmp_path / "live", horizon=2)
    assert not (tmp_path / "live").exists()


def test_the_issue_is_today_in_the_issue_zone_on_the_clock(tmp_path):
    with use_clock(FakeClock(pd.Timestamp("2023-03-06 23:30", tz="UTC"))):  # 00:30 on 7 March in Warsaw
        result = simple().forecast(DATA, "today", None, horizon=2)
    assert result.issue == ISSUE
    assert {r["run_date"] for r in result.records["a"]} == {ISSUE}


@pytest.mark.parametrize("refit", [None, "7D"])
def test_verify_through_the_serve_path_is_ok_with_its_control(refit):
    p = pipeline(ols(refit=refit), engine="on")
    report = p.verify(DATA, "2023-03-01", "2023-03-08", horizon=2, days=2, through="serve")
    assert report.ok, report.table()
    leaky = pipeline(OLSModel([*PREDICTORS, "late_d+{horizon}"], training_window=21, name="a", refit=refit))
    assert not leaky.verify(DATA, "2023-03-01", "2023-03-08", horizon=2, days=2, through="serve").ok


def test_each_call_keeps_its_own_deadline_and_the_directory_its_issue_zone(tmp_path):
    """The deadline is each call's (an earlier one tomorrow is no other pipeline): its call line and its revision's commit
    line hold it, and readers cut off at "deadline" read it there. `live.json` holds the issue zone, compared on every call."""
    live = tmp_path / "live"
    with use_clock(FakeClock(AT)):
        simple().forecast(DATA, ISSUE, live, horizon=2, deadline="11:55")
        changed = DATA.copy()
        changed.loc[
            changed.index[changed.index.normalize() == pd.Timestamp(ISSUE, tz="Europe/Warsaw")][5], "load_d+1"
        ] += 9
        second = simple().forecast(changed, ISSUE, live, horizon=2, deadline="11:40")
    assert second.revision == 2
    identity = json.loads((live / "live.json").read_text())
    assert identity["issue_timezone"] == "Europe/Warsaw" and "deadline" not in identity
    commits = LiveStore(live).committed()[ISSUE]
    assert commits[1].deadline == pd.Timestamp("2023-03-07 10:55", tz="UTC")
    assert commits[2].deadline == pd.Timestamp("2023-03-07 10:40", tz="UTC")
    seen = LiveStore(live).visible("latest", "deadline")
    assert sorted(seen[ISSUE]) == [1, 2]
    other = ModelPipeline(issue_time="11:45", issue_timezone="Europe/Berlin").add_model(ols())
    with pytest.raises(LiveDirectoryError, match="Europe/Warsaw"):
        other.forecast(DATA, ISSUE, live, horizon=2, dry_run=True)


def test_a_dry_run_refuses_what_the_real_call_would(tmp_path):
    live = tmp_path / "live"
    simple().forecast(DATA, ISSUE, live, horizon=2)
    before = snapshot_of(live)
    narrower = pipeline(
        ols(window=14), ols("b", PREDICTORS[:2], 14), ensembles=[FallbackEnsemble(["a", "b"], name="fb")]
    )
    with pytest.raises(ManifestMismatchError, match="training_window"):
        narrower.forecast(DATA, ISSUE, live, horizon=2, dry_run=True)
    assert snapshot_of(live) == before
    with pytest.raises(ManifestMismatchError, match="training_window"):
        narrower.forecast(DATA, ISSUE, live, horizon=2)


# --- the data range a refresh asks for ---------------------------------------------------------------------------------------


def csv_pipeline(raw, tmp_path, *transformers):
    """A real data pipeline over a CSV of `raw` (UTC), which returns exactly the range it is asked for, transformed."""
    path = tmp_path / "market.csv"
    raw.tz_convert("UTC").rename_axis("datetime").reset_index().to_csv(path, index=False)
    return DataPipeline(
        sources=[CsvSource(str(path))],
        transformers=[TimezoneTransformer("Europe/Warsaw"), *transformers],
        timezone="Europe/Warsaw",
    )


ROLLING = RollingStatsTransformer("load", window_days=60, stat="mean", lag_days=1)
ROLLED = {**RULES, ROLLING.output_column: Availability.static()}


def test_the_range_covers_a_rolling_window_reaching_far_behind_the_oldest_training_row(tmp_path):
    raw = market(days=200, start="2022-10-01")
    data = csv_pipeline(raw, tmp_path, ROLLING)
    info = InformationSet("11:45", "Europe/Warsaw", rules=ROLLED)

    def make():
        return pipeline(ols(predictors=[*PREDICTORS, ROLLING.output_column], window=56), info=info)

    whole = data.run("2022-10-01", "2023-04-15")
    folder = tmp_path / "backtest"
    make().run(whole, ISSUE, ISSUE, horizon=2, save_dir=str(folder))
    result = make().forecast(data, ISSUE, tmp_path / "live", horizon=2, cache=False)
    # The rolling mean is a running sum, so its last bits depend on where the frame starts: equal within 1e-9, not to the bit.
    # A range that cuts the window leaves the oldest training rows without it, and those fits lose rows (differences of 1e-2).
    expected = {(r["hour"], r["horizon"]): r["prediction"] for r in lines(folder / "a.jsonl")}
    got = {(r["hour"], r["horizon"]): r["prediction"] for r in result.records["a"] if r["status"] == "ok"}
    assert result.status == "ok" and len(expected) == 48 and set(got) == set(expected)
    worst = max(abs(got[key] - expected[key]) for key in expected)
    assert worst < 1e-9, worst


def test_a_transformer_that_can_t_say_its_lookback_warns_and_span_sets_the_range(tmp_path):
    raw = market(days=200, start="2022-10-01")
    double = LambdaTransformer(columns={"twice": "load * 2"})
    data = csv_pipeline(raw, tmp_path, double)
    with pytest.warns(UserWarning, match="LambdaTransformer can't say how far back it reads"):
        pipeline(ols()).forecast(data, ISSUE, None, horizon=2, cache=False)
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        pipeline(ols()).forecast(data, ISSUE, None, horizon=2, cache=False, span=("2022-10-01", "2023-03-10"))


def test_the_range_covers_an_older_stored_fit_served_in_place_of_a_missing_one(tmp_path):
    raw = market(days=330, start="2022-06-01")
    data = csv_pipeline(raw, tmp_path)
    whole = data.run("2022-06-01", "2023-04-15")

    def make():
        return pipeline(ols(refit="3MS", window=56))

    live = tmp_path / "live"
    make().refit(whole, live, issue="2022-10-01", horizon=2)  # October's fit; January's is missing
    with _no_stale_warning():
        result = make().forecast(data, ISSUE, live, horizon=2, cache=False)
    entry = result.models["a"]["fits"]["2023-01-01"]
    assert entry["source"] == "previous" and entry["used"] == "2022-10-01" and "stale" not in entry


class _no_stale_warning:
    def __enter__(self):
        import warnings as _warnings

        self._catch = _warnings.catch_warnings(record=True)
        self.seen = self._catch.__enter__()
        _warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        self._catch.__exit__(*exc)
        stale = [str(w.message) for w in self.seen if "holds otherwise" in str(w.message)]
        assert not stale, stale
        return False


FIRST_RUNGS = {
    "ols": lambda: OLSModel([*PREDICTORS, "extra_d+{horizon}"], training_window=21, name="first"),
    "hgb": lambda: HistGBModel([*PREDICTORS, "extra_d+{horizon}"], training_window=21, max_iter=20, name="first"),
    "naive": lambda: NaiveModel(["extra_d+{horizon}"], name="first"),
}


@pytest.mark.parametrize("family", sorted(FIRST_RUNGS))
def test_a_first_rung_with_no_forecast_at_all_falls_back_whatever_its_family(family, tmp_path):
    """Before EXAA is out every issue row lacks it: the first rung has no forecast, through its stored or inline fits (OLS) or
    its own run of the day (HistGB, Naive), and the ladder serves the day from the second, never a failed call."""
    early = DATA.copy()
    early.loc[early.index.normalize() == pd.Timestamp(ISSUE, tz="Europe/Warsaw"), ["extra_d+1", "extra_d+2"]] = np.nan
    p = pipeline(FIRST_RUNGS[family](), ols("second"), ensembles=[FallbackEnsemble(["first", "second"], name="fb")])
    result = p.forecast(early, ISSUE, tmp_path / "live", horizon=2)
    assert (result.status, result.exit_code, result.revision) == ("fallback", 10, 1)
    assert result.rungs == {"2023-03-08": 1, "2023-03-09": 1}
    assert sum(result.models["first"]["skipped"].values()) == 48 and result.models["first"]["forecasts"] == 0
    assert len(lines(tmp_path / "live" / "first.jsonl")) == 48  # its skip records are the revision's


def residual(base, warmup):
    learner = OLSModel(PREDICTORS[:2], training_window=14, name="l")
    return ResidualModel(base, learner, training_window=14, refit="7D", warmup=warmup, name="res")


@pytest.mark.parametrize("warmup", [7, None])
def test_a_residual_model_whose_forecasts_depend_on_the_backtest_s_start_is_refused(warmup, tmp_path):
    p = pipeline(ols("a"), residual(ols("a"), warmup))
    with pytest.raises(ValueError, match="dated warm-up"):
        p.forecast(DATA, ISSUE, tmp_path / "live", horizon=2)
    assert not (tmp_path / "live").exists()


@pytest.mark.parametrize("named", [True, False], ids=["named-base", "base-object"])
def test_a_residual_model_with_a_dated_warmup_is_its_backtest_s_record_live(named, tmp_path):
    raw = market(days=200, start="2022-10-01")
    data = csv_pipeline(raw, tmp_path)
    whole = data.run("2022-10-01", "2023-04-15")

    def make():
        # The base's forecasts from 15 January: with its 21-day window it reads rows from late December, which the range
        # of a refresh must hold although the residual model's own window is 14 days.
        return pipeline(ols("a"), residual("a" if named else ols("a"), "2023-01-15"))

    folder = tmp_path / "backtest"
    make().run(whole, "2023-03-01", "2023-03-09", horizon=2, save_dir=str(folder))
    result = make().forecast(data, ISSUE, tmp_path / "live", horizon=2, cache=False)
    assert result.status == "ok"
    assert keyed(result.records["res"]) == keyed(lines(folder / "res.jsonl"), ISSUE)
    start, _ = _live.data_span(make().models, datetime.date(2023, 3, 7), [1, 2])
    assert start <= "2022-12-23"  # 15 January less the base's window and horizon


EARLY_FITS = {
    "hgb": lambda: HistGBModel(
        PREDICTORS, training_window=21, refit="1D", max_iter=20, name="g"
    ),  # pooled periods, own run
    "ols": lambda: OLSModel(PREDICTORS, training_window=21, name="g"),  # the stored fits' path, fitted inline
}


def too_early_frame():
    early = DATA.copy()
    early.loc[early.index >= pd.Timestamp("2023-03-07", tz="Europe/Warsaw"), "price"] = np.nan
    return early


@pytest.mark.parametrize("family", sorted(EARLY_FITS))
def test_a_call_before_the_newest_training_target_is_out_is_refused_whatever_the_path(family, tmp_path):
    """Issue 8 March, called at 11:00 on 6 March (Warsaw): the price of 7 March, the newest target a fit of the 8th trains on,
    is out at 13:00 that day. A fit made now would train on one target less than the backtest's: refused, nothing committed."""
    live = tmp_path / "live"
    with use_clock(FakeClock(pd.Timestamp("2023-03-06 10:00", tz="UTC"))):
        with pytest.raises(FitTooEarlyError, match="2023-03-07"):
            pipeline(EARLY_FITS[family]()).forecast(too_early_frame(), "2023-03-08", live, horizon=2)
    assert LiveStore(live).committed() == {}


def test_a_model_that_fits_nothing_has_no_training_target_to_wait_for(tmp_path):
    with use_clock(FakeClock(pd.Timestamp("2023-03-06 10:00", tz="UTC"))):
        result = pipeline(NaiveModel(["load_d+{horizon}"], name="g")).forecast(
            too_early_frame(), "2023-03-08", None, horizon=2
        )
    assert result.status == "ok"


def test_waits_snapshot_and_quality_read_the_issue_rows_the_models_read_in_the_frame_s_zone(tmp_path):
    """A frame in Helsinki issued in Warsaw: the models take the issue day's rows by Helsinki dates, an hour ahead of Warsaw's.
    The missing load_d+1 at Helsinki's midnight is an issue row of theirs (Warsaw's 23:00 of the day before): the wait, the
    quality record and the snapshot count it, and both zones are recorded."""
    data = make_market_data(start="2023-01-20", days=60, tz="Europe/Helsinki", horizon=2, noise=2.0)
    info = InformationSet(
        "11:45", "Europe/Warsaw", rules={k: RULES[k] for k in ("load", "wind", "is_weekend", "is_weekday", "price")}
    )
    data.loc[pd.Timestamp(ISSUE + " 00:00", tz="Europe/Helsinki"), "load_d+1"] = np.nan

    class Source:
        timezone, last_refresh, transformers = "Europe/Helsinki", None, []

        def run(self, start, end, **options):
            return data

    live = tmp_path / "live"
    with use_clock(FakeClock(pd.Timestamp(ISSUE + " 08:00", tz="UTC"))):
        result = pipeline(ols(), info=info).forecast(
            Source(), ISSUE, live, horizon=2, wait_for={"load_d+1": "10:45"}, poll="20min"
        )
    assert result.models["a"]["skipped"] == {"missing_test_predictors": 1}
    assert result.waits[0]["lacking"] == {"load_d+1": 1} and result.waits[-1]["outcome"] == "gave up"
    folder = live / "issues" / ISSUE / "r1"
    quality = json.loads((folder / "quality.json").read_text())
    assert quality["columns"]["load_d+1"]["missing"] == 1
    assert (quality["issue_timezone"], quality["data_timezone"]) == ("Europe/Warsaw", "Europe/Helsinki")
    inputs = read_snapshot(folder / "inputs.parquet").inputs.index.tz_convert("Europe/Helsinki")
    assert (str(inputs.min()), len(inputs)) == ("2023-03-07 00:00:00+02:00", 24)
    commit = LiveStore(live).committed()[ISSUE][1].line
    assert (commit["issue_timezone"], commit["data_timezone"]) == ("Europe/Warsaw", "Europe/Helsinki")


def test_the_export_writes_every_planned_target_day_even_one_without_records():
    """A missing input on one issue row fails `a`'s horizon-2 day (`periods`), so the average has no record of that target day
    at all: its 24 products are still written, with status "missing", as the skipped status says."""
    gappy = DATA.copy()
    row = gappy.index[gappy.index.normalize() == pd.Timestamp(ISSUE, tz="Europe/Warsaw")][7]
    gappy.loc[row, "wind_d+2"] = np.nan
    p = pipeline(
        ols("a"),
        NaiveModel(["price_d-1"], name="naive"),
        ensembles=[AverageEnsemble(["a", "naive"], name="avg")],
        checks=ForecastChecks(),
    )
    result = p.forecast(gappy, ISSUE, None, horizon=2)
    assert (result.status, result.missing, result.planned) == ("skipped", 24, 48)
    assert not [r for r in result.records["avg"] if r["target_date"] == "2023-03-09"]
    products = result.table("products")
    assert len(products) == 48 and sorted(products["target_date"].unique()) == ["2023-03-08", "2023-03-09"]
    assert (products[products["target_date"] == "2023-03-09"]["status"] == "missing").all()
    periods = result.table("periods")
    assert len(periods) == 48 and periods["prediction"].isna().sum() == 24


@pytest.mark.parametrize(
    "freq, days, issue, horizons, products, periods",
    [
        (
            "1h",
            75,
            "2023-03-25",
            [1],
            23,
            23,
        ),  # the spring day: 23 hours, the skipped one neither a product nor a period
        ("1h", 60, ISSUE, [1, 2, 3], 72, 72),  # hourly, several horizons: each target day's 24 products
        ("1D", 60, ISSUE, [1, 2], 2, 2),  # one product a day, from its local midnight
    ],
    ids=["spring-1h", "hourly-three-horizons", "daily"],
)
def test_the_export_grids_on_each_clock(freq, days, issue, horizons, products, periods):
    data = make_market_data(start="2023-01-20", days=days, horizon=3, noise=2.0)
    if freq == "1D":
        data = data.at_time("12:00")
    result = pipeline(ols(window=14)).forecast(data, issue, None, horizons=horizons, freq=freq)
    table = result.table("products")
    assert (len(table), len(result.table("periods"))) == (products, periods)
    assert (table["status"] == "ok").all() and table["occurrence"].eq(1).all()
    for horizon, day in table.groupby("horizon"):
        assert day["start_utc"].is_unique
        target = datetime.date.fromisoformat(issue) + datetime.timedelta(days=int(horizon))
        assert (day["target_date"] == target.isoformat()).all()
        first = pd.Timestamp(day["start_utc"].iloc[0]).tz_convert("Europe/Warsaw")
        assert (first.date(), first.hour, first.minute) == (target, 0, 0)
    if freq == "1h" and issue == "2023-03-25":
        assert 2 not in set(table["slot"])


class FrameSource(DataSource):
    """A cached source over a frame: each fetch is the frame's rows of [start, end) in UTC, counted."""

    def __init__(self, frame):
        self._frame = frame.tz_convert("UTC")
        self._fetches = []  # private: a source's public attributes are its cache fingerprint

    @property
    def fetches(self):
        return self._fetches

    def _validate_config(self) -> bool:
        return True

    def get_cache_config(self):
        return {"frame": "market", "columns": list(self._frame.columns)}

    def fetch(self, start, end):
        self._fetches.append((start, end))
        return self._frame[(self._frame.index >= start) & (self._frame.index < end)]


def test_a_dry_run_reads_the_sources_caches_and_writes_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = FrameSource(market(days=200, start="2022-10-01"))
    data = DataPipeline(sources=[source], transformers=[TimezoneTransformer("Europe/Warsaw")], timezone="Europe/Warsaw")
    dry = pipeline(ols()).forecast(data, ISSUE, None, horizon=2, dry_run=True)
    assert dry.status == "ok" and source.fetches and not (tmp_path / ".cache").exists()
    pipeline(ols()).forecast(data, ISSUE, None, horizon=2)  # a call that writes: the cache is made
    cached = snapshot_of(tmp_path / ".cache")
    assert cached
    fetched = len(source.fetches)
    again = pipeline(ols()).forecast(data, ISSUE, None, horizon=2, dry_run=True)
    assert len(source.fetches) == fetched  # read from the cache
    later = pipeline(ols()).forecast(data, "2023-03-09", None, horizon=2, dry_run=True)  # partly beyond it: fetched
    assert len(source.fetches) > fetched and later.status == "ok" and again.status == "ok"
    assert snapshot_of(tmp_path / ".cache") == cached


def test_a_file_cache_read_without_writes_makes_nothing(tmp_path):
    source = FrameSource(market(days=60, start="2023-01-20"))
    data = DataPipeline(sources=[source], transformers=[TimezoneTransformer("Europe/Warsaw")], timezone="Europe/Warsaw")
    path = tmp_path / "cache" / "frame.parquet"
    fresh = data.run("2023-02-01", "2023-03-01", cache=str(path), cache_writes=False)
    assert not (tmp_path / "cache").exists() and len(source.fetches) == 1
    data.run("2023-02-01", "2023-03-01", cache=str(path))
    written = snapshot_of(tmp_path / "cache")
    again = data.run("2023-02-01", "2023-03-01", cache=str(path), cache_writes=False)
    assert len(source.fetches) == 2 and snapshot_of(tmp_path / "cache") == written
    pd.testing.assert_frame_equal(fresh, again, check_freq=False)
