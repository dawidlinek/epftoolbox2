"""The live commands of `epf` (step 8i): `epf forecast`, `refit`, `mark-delivered`, `status`, `monitor` and `snapshot`, in process
through `cli.main(argv, clock=...)`, and the `live:` section of the workflow YAML they read.

An experiment is a workflow YAML over a CSV of synthetic market data (no network), with two OLS models and a fallback
ensemble, issued 11:45 Europe/Warsaw on 2023-03-07. Every call runs on a `FakeClock`: sleeps pass at once.
"""

import hashlib
import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from epftoolbox2 import cli
from epftoolbox2.clock import FakeClock
from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import _engine_worker
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline, Workflow
from epftoolbox2.results.forecast_result import ForecastResult
from epftoolbox2.results.live import LiveStore
from tests.synthetic import make_market_data

pytestmark = pytest.mark.usefixtures("no_network")

RULES = {
    "load": Availability.published("10:00", 10),
    "wind": Availability.published("10:00", 10),
    "is_weekend": Availability.static(),
    "is_weekday": Availability.static(),
    "price": Availability.published("13:00", 1),
    "extra": Availability.published("11:15", 1),
}
INFO = InformationSet("11:45", "Europe/Warsaw", rules=RULES)
PREDICTORS = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]
ISSUE = "2023-03-07"
BEFORE_DEADLINE = "2023-03-07T11:30+01:00"  # 11:30 in Warsaw: before the 11:45 issue time and the 11:55 deadline
AFTER_DEADLINE = "2023-03-07T11:58+01:00"


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


LIVE = {
    "save_dir": "live",
    "deadline": "11:55",
    "poll": "5min",
    "budget": "120s",
    "timeout": "30s",
    "retries": 2,
    "on_failure": "cache",
    "deliver": "live",
    "export": {"path": "out/{issue}.csv", "format": "csv", "grid": "products"},
}


def market_csv(path, days=60, blank=()):
    """The market data as a CSV; the columns in `blank` are empty on the issue day's rows (an input that never came)."""
    data = make_market_data(start="2023-01-20", days=days, horizon=2, noise=2.0)
    for column in blank:
        data.loc[data.index.strftime("%Y-%m-%d") == ISSUE, column] = np.nan
    data.tz_convert("UTC").rename_axis("datetime").reset_index().to_csv(path, index=False)


def model(name, predictors=PREDICTORS):
    return {"class": "OLSModel", "params": {"predictors": predictors, "training_window": 21, "name": name}}


def experiment_config(tmp_path, live=LIVE, blank=(), **model_pipeline):
    market_csv(tmp_path / "market.csv", blank=blank)
    return {
        "data_pipeline": {
            "cache": False,
            "sources": [{"class": "CsvSource", "params": {"file_path": str(tmp_path / "market.csv")}}],
            "transformers": [{"class": "TimezoneTransformer", "params": {"target_tz": "Europe/Warsaw"}}],
        },
        "model_pipeline": {
            "target": "price",
            "horizon": 2,
            "information_set": INFO.to_dict(),
            "engine": "off",
            "models": [model("main"), model("base", ["is_weekend_d+{horizon}", "is_weekday_d+{horizon}"])],
            "ensembles": [{"class": "FallbackEnsemble", "params": {"members": ["main", "base"], "name": "live"}}],
            **model_pipeline,
        },
        **({} if live is None else {"live": live}),
    }


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the sources' caches go to ./.cache


@pytest.fixture
def experiment(tmp_path):
    """The path of a workflow YAML with a live section, over a CSV in `tmp_path`."""
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment_config(tmp_path)))
    return path


def experiment_with(tmp_path, **kwargs):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment_config(tmp_path, **kwargs)))
    return path


# --- the live: section ------------------------------------------------------------------------------------------------------


def test_a_workflow_with_a_live_section_needs_no_data_range(experiment):
    workflow = Workflow.load(experiment)
    assert workflow.data_start is None and workflow.data_end is None
    assert workflow.live.save_dir == "live" and workflow.live.deadline == "11:55"
    assert workflow.live.export == LIVE["export"]


def test_the_live_section_survives_a_save_and_a_load(experiment, tmp_path):
    first = Workflow.load(experiment)
    again = tmp_path / "again.yaml"
    first.save(again)
    second = Workflow.load(again)
    assert second.live == first.live
    assert yaml.safe_load(again.read_text())["live"] == LIVE
    assert second.to_dict() == first.to_dict()
    assert "start" not in second.to_dict()["data_pipeline"]


def test_a_workflow_without_live_still_needs_its_data_range(tmp_path):
    config = experiment_config(tmp_path, live=None)
    path = tmp_path / "plain.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(KeyError, match="start"):
        Workflow.load(path)


def test_a_live_only_workflow_refuses_a_backtest(experiment):
    with pytest.raises(ValueError, match="live-only"):
        Workflow.load(experiment).run()


@pytest.mark.parametrize(
    "live, message",
    [
        ({"deadline": "11:55"}, "needs save_dir"),
        ({"save_dir": "live", "deadlin": "11:55"}, "unknown key"),
        ({"save_dir": "live", "deadline": 715}, "quote it"),
        ({"save_dir": "live", "wait_for": {"load": 645}}, "quote it"),
        ({"save_dir": "live", "wait_for": "later"}, "wait_for"),
        ({"save_dir": "live", "retries": -1}, "retries"),
        ({"save_dir": "live", "export": {"format": "csv"}}, "export.path"),
        ({"save_dir": "live", "export": {"path": "x", "format": "xlsx"}}, "export.format"),
        ({"save_dir": "live", "export": {"path": "x", "grid": "days"}}, "export.grid"),
        ({"save_dir": "live", "export": {"path": "x", "mode": "a"}}, "unknown key"),
    ],
)
def test_a_mistake_in_the_live_section_is_refused_when_the_file_is_loaded(tmp_path, live, message):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(experiment_config(tmp_path, live=live)))
    with pytest.raises(ValueError, match=message):
        Workflow.load(path)


def test_forecast_only_into_a_save_dir_is_still_refused_next_to_a_live_section(tmp_path):
    config = experiment_config(tmp_path, forecast_only=True, save_dir="backtest")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="forecast_only"):
        Workflow.load(path)


# --- epf forecast ------------------------------------------------------------------------------------------------------------


def forecast(experiment, *extra, at=BEFORE_DEADLINE, clock=None):
    argv = ["forecast", str(experiment), "--issue", ISSUE, *extra]
    if clock is None:
        argv += ["--as-of", at]
    return cli.main(argv, clock=clock)


def tree(root):
    """Every file under `root` with its bytes' hash and modification time, and every directory: what a dry run must not change."""
    root = Path(root)
    out = {}
    for path in sorted(root.rglob("*")):
        key = str(path.relative_to(root))
        if path.is_dir():
            out[key] = "dir"
        else:
            out[key] = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    return out


def issue_lines(live, name):
    return sorted((Path(live) / f"{name}.jsonl").read_text().splitlines())


def test_forecast_gives_exit_0_and_a_revision_when_every_day_comes_from_the_first_rung(experiment, tmp_path, capsys):
    assert forecast(experiment) == 0
    store = LiveStore(tmp_path / "live")
    assert sorted(store.committed()[ISSUE]) == [1]
    assert "status ok, exit 0, revision 1" in capsys.readouterr().out
    assert [line["status"] for line in store.calls() if line["committed"]] == ["ok"]


def test_forecast_gives_exit_10_when_a_lower_rung_serves(tmp_path):
    path = experiment_with(tmp_path, blank=["load_d+1", "load_d+2"])
    assert forecast(path) == 10
    commits = [line for line in LiveStore(tmp_path / "live").calls() if line["committed"]]
    assert [(line["status"], line["exit_code"]) for line in commits] == [("fallback", 10)]


def test_forecast_gives_exit_20_when_a_planned_key_has_no_forecast(tmp_path):
    path = experiment_with(tmp_path, blank=["load_d+1", "load_d+2", "is_weekend_d+1", "is_weekend_d+2"])
    assert forecast(path) == 20


def test_forecast_gives_exit_30_after_the_deadline_and_writes_no_revision(experiment, tmp_path):
    assert forecast(experiment, at=AFTER_DEADLINE) == 30
    store = LiveStore(tmp_path / "live")
    assert store.committed() == {}
    assert [(line["status"], line["committed"]) for line in store.calls()] == [("late", False)]
    assert not (tmp_path / "out").exists()


def test_forecast_gives_exit_75_while_another_call_holds_the_live_lock(experiment, tmp_path):
    with LiveStore(tmp_path / "live").hold():
        assert forecast(experiment) == 75
    assert LiveStore(tmp_path / "live").committed() == {}
    assert not (tmp_path / "out").exists()


def test_forecast_gives_exit_1_when_the_call_raises(tmp_path, capsys):
    path = experiment_with(tmp_path, live={**LIVE, "deliver": "nosuch"})
    assert forecast(path) == 1
    assert "nosuch" in capsys.readouterr().out
    assert forecast(tmp_path / "missing.yaml") == 1


def test_a_usage_mistake_exits_with_2(experiment):
    for argv in (["--issue", "someday"], ["--as-of", "2023-03-07T11:30"], ["--wait-for", "load"], ["--refresh", "now"]):
        with pytest.raises(SystemExit) as stop:
            cli.main(["forecast", str(experiment), *argv])
        assert stop.value.code == 2


def test_ensure_is_a_noop_with_exit_0_once_a_revision_is_committed(experiment, tmp_path):
    assert forecast(experiment, "--ensure") == 0
    before = tree(tmp_path / "live")
    assert forecast(experiment, "--ensure") == 0
    assert tree(tmp_path / "live") == before
    assert sorted(LiveStore(tmp_path / "live").committed()[ISSUE]) == [1]


def test_an_unchanged_rerun_writes_no_revision_and_exits_0(experiment, tmp_path):
    assert forecast(experiment) == 0
    assert forecast(experiment) == 0
    assert sorted(LiveStore(tmp_path / "live").committed()[ISSUE]) == [1]


def test_a_forecast_exports_the_delivered_model_as_the_live_section_says(experiment, tmp_path):
    assert forecast(experiment) == 0
    exported = pd.read_csv(tmp_path / "out" / f"{ISSUE}.csv")
    assert len(exported) == 48 and exported["status"].eq("ok").all() and exported["prediction"].notna().all()


# --- --as-of is a FakeClock ------------------------------------------------------------------------------------------


def test_as_of_runs_as_main_does_on_a_fake_clock_given_to_it(tmp_path):
    outcomes = []
    for index, give_clock in enumerate((False, True)):
        folder = tmp_path / f"run{index}"
        folder.mkdir()
        path = experiment_with(folder)
        start = pd.Timestamp(BEFORE_DEADLINE)
        code = forecast(path, clock=FakeClock(start) if give_clock else None)
        live = folder / "live"
        calls = [{k: v for k, v in line.items() if k != "call"} for line in LiveStore(live).calls()]
        outcomes.append(
            (
                code,
                calls,
                issue_lines(live, "main"),
                issue_lines(live, "live"),
                (folder / "out" / f"{ISSUE}.csv").read_bytes(),
            )
        )
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] == 0 and outcomes[0][2]


def test_as_of_and_a_clock_given_to_main_are_refused_together(experiment):
    with pytest.raises(SystemExit) as stop:
        cli.main(["forecast", str(experiment), "--as-of", BEFORE_DEADLINE], clock=FakeClock(BEFORE_DEADLINE))
    assert stop.value.code == 2


def test_a_wait_passes_at_once_on_the_fake_clock(tmp_path):
    path = experiment_with(tmp_path, blank=["load_d+1"], live={**LIVE, "wait_for": {"load_d+1": "11:00"}})
    clock = FakeClock("2023-03-07T10:00+01:00")
    assert forecast(path, clock=clock) == 10
    assert set(clock.sleeps) == {300.0} and len(clock.sleeps) == 12  # 5 minutes, 10:00 to 11:00, no real second
    r1 = tmp_path / "live" / "issues" / ISSUE / "r1"
    polls = json.loads((r1 / "waits.json").read_text())["polls"]
    assert [poll["at"] for poll in polls][:3] == [
        "2023-03-07T09:00:00+00:00",
        "2023-03-07T09:05:00+00:00",
        "2023-03-07T09:10:00+00:00",
    ]  # every 5 minutes from 10:00 Warsaw, on the fake clock
    assert len(polls) == 13 and polls[-1]["at"] == "2023-03-07T10:00:00+00:00"  # it gave up at 11:00


# --- --dry-run writes nothing --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "blank, at, code",
    [((), BEFORE_DEADLINE, 0), (("load_d+1", "load_d+2"), BEFORE_DEADLINE, 10), ((), AFTER_DEADLINE, 30)],
    ids=["ok", "fallback", "late"],
)
def test_dry_run_gives_the_exit_code_the_call_would_give_and_writes_nothing(tmp_path, blank, at, code):
    path = experiment_with(tmp_path, blank=blank)
    assert forecast(path, "--dry-run", at=at) == code
    assert not (tmp_path / "live").exists() and not (tmp_path / "out").exists()
    assert forecast(path, at=at) == code


def test_dry_run_changes_no_file_of_a_live_directory_that_has_a_revision(experiment, tmp_path):
    assert forecast(experiment) == 0
    before = tree(tmp_path)
    assert forecast(experiment, "--dry-run", at="2023-03-07T11:40+01:00") == 0
    assert tree(tmp_path) == before


# --- the live: keys reach forecast() unchanged -------------------------------------------------------------------------------


@pytest.fixture
def spy(monkeypatch):
    calls = []

    def stub(self, data, issue="today", save_dir=None, **kwargs):
        calls.append({"issue": issue, "save_dir": save_dir, **kwargs})
        return ForecastResult(issue="2023-03-07", status="ok", exit_code=0, noop=True)

    monkeypatch.setattr(ModelPipeline, "forecast", stub)
    return calls


def test_the_live_keys_reach_forecast_unchanged(experiment, tmp_path, spy):
    live = dict(LIVE, wait_for={"load_d+1": "10:45"}, poll="5min", grace="20min", fit_missing="fit")
    path = experiment_with(tmp_path, live=live)
    assert forecast(path) == 0
    (call,) = spy
    for key in (
        "deadline",
        "wait_for",
        "poll",
        "budget",
        "timeout",
        "retries",
        "on_failure",
        "grace",
        "fit_missing",
        "deliver",
    ):
        assert call[key] == live[key], key
    assert call["save_dir"] == tmp_path / "live" and call["issue"] == "2023-03-07"
    assert (call["target"], call["horizon"], call["freq"], call["save_format"]) == ("price", 2, "1h", "jsonl")


def test_a_key_the_section_does_not_set_is_left_to_forecast_s_default(tmp_path, spy):
    path = experiment_with(tmp_path, live={"save_dir": "live"})
    assert forecast(path) == 0
    (call,) = spy
    assert not {"deadline", "wait_for", "poll", "budget", "timeout", "retries", "on_failure", "deliver"} & set(call)
    assert call["ensure"] is False and call["refresh"] == "auto" and call["dry_run"] is False


def test_the_flags_replace_the_section_s_keys_for_one_call(experiment, tmp_path, spy):
    path = experiment_with(tmp_path, live={**LIVE, "wait_for": {"load_d+1": "10:45"}})
    code = forecast(
        path,
        "--deadline",
        "12:10",
        "--wait-for",
        "wind_d+1=10:30",
        "--wait-for",
        "load_d+1=10:20",
        "--ensure",
        "--refresh",
        "never",
    )
    assert code == 0
    (call,) = spy
    assert call["deadline"] == "12:10" and call["wait_for"] == {"wind_d+1": "10:30", "load_d+1": "10:20"}
    assert call["ensure"] is True and call["refresh"] == "never"
    forecast(path, "--wait-for", "rules")
    assert spy[-1]["wait_for"] == "rules"
