"""`tune.BayesSearch`: proposals and successive halving on the validation period alone, returning ordinary models.

The tests that count promotions or share a journal between processes score trials with a cheap stand-in for the model run
(`_score`): they test the search, not a model. The others run OLS (a window and a subset of predictors), and a torch model
where torch is installed.
"""

import json
import math
import os
import subprocess
import sys
import textwrap
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from epftoolbox2._extras import MissingExtraError
from epftoolbox2.evaluators import MAEEvaluator
from epftoolbox2.models import OLSModel
from epftoolbox2.models._calendar import IssueCalendar
from epftoolbox2.models._worker import _worker_init
from epftoolbox2.pipelines import ModelPipeline
from epftoolbox2.results.manifest import manifest_path
from epftoolbox2.tune import BayesSearch, SearchResult, Subset
from epftoolbox2.tune import bayes
from tests.synthetic import make_market_data

ROOT = Path(__file__).resolve().parents[1]
V = ("2023-03-20", "2023-04-02")
T = ("2023-04-10", "2023-04-12")
FIXED = ["is_weekend_d+{horizon}", "is_weekday_d+{horizon}"]


class InProcessPool:
    """The worker pool in the test's own process: what a trial's run costs is mostly the pool's start."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
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
    pytest.importorskip("optuna")
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    monkeypatch.setenv(bayes.ENV_POLL, "0.05")
    monkeypatch.delenv(bayes.ENV_WORKERS, raising=False)
    monkeypatch.delenv(bayes.ENV_REQUEUE, raising=False)
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)
    with threadpool_limits(limits=1):  # what a pool's worker runs on: a fit's bits depend on its BLAS threads
        yield


@pytest.fixture(scope="module")
def df():
    return make_market_data(start="2023-01-01", days=110, horizon=2, noise=5.0, seed=3)


def make(params):
    """OLS with a window and a subset of the drivers."""
    names = [f"{column}_d+{{horizon}}" for column in params["drivers"]] + FIXED
    return OLSModel(names, training_window=params["window"], name="ols")


SPACE = {"window": [14, 28, 42], "drivers": Subset(["load", "wind"])}


def search(**kwargs):
    arguments = dict(n_trials=6, seed=0, rungs=[{"every_nth_day": 4}, {"every_nth_day": 1}], keep=0.5)
    arguments.update(kwargs)
    return BayesSearch(make, SPACE, **arguments)


# ---------------------------------------------------------------------------------------------------------------------
# What a trial may read
# ---------------------------------------------------------------------------------------------------------------------


def test_no_trial_reads_past_the_validation_end(df, monkeypatch):
    """Rows after the validation end poisoned with NaN: the same table as the clean frame, no NaN anywhere, and every trial's
    frame ends on the validation end."""
    poisoned = df.copy()
    after = poisoned.index.tz_localize(None).normalize() > pd.Timestamp(V[1])
    poisoned.loc[after, :] = np.nan
    seen = []
    real = ModelPipeline.run

    def spy(self, data, *args, **kwargs):
        seen.append(data.index.tz_localize(None).normalize().max())
        return real(self, data, *args, **kwargs)

    monkeypatch.setattr(ModelPipeline, "run", spy)
    clean = search().run(df, horizon=2, validation=V).table()
    dirty = search().run(poisoned, horizon=2, validation=V).table()
    pd.testing.assert_frame_equal(clean, dirty)
    assert set(seen) == {pd.Timestamp(V[1])}
    assert clean["validation_mae"].notna().all() and np.isfinite(clean["validation_mae"]).all()


def test_the_targets_after_the_validation_end_are_never_scored(df):
    """The price of the H days after the validation end (the test period's) poisoned with a huge finite value: the table is the
    clean frame's, since the forecasts for those days are made and not scored."""
    poisoned = df.copy()
    day = poisoned.index.tz_localize(None).normalize()
    test_days = (day > pd.Timestamp(V[1])) & (day <= pd.Timestamp(V[1]) + pd.Timedelta(days=2))
    poisoned.loc[test_days, "price"] = 1e6
    clean = search().run(df, horizon=2, validation=V)
    dirty = search().run(poisoned, horizon=2, validation=V)
    pd.testing.assert_frame_equal(clean.table(), dirty.table())
    assert clean.table()["validation_mae"].max() < 100
    assert clean.record["search"]["data_through"] == V[1]


def test_the_cut_is_on_local_days():
    frame = make_market_data(start="2023-03-20", days=10, horizon=1)
    cut = bayes.cut_frame(frame, "2023-03-26")
    local = cut.index.tz_localize(None)
    assert local.max() == pd.Timestamp("2023-03-26 23:00") and local.min() == pd.Timestamp("2023-03-20")
    assert len(cut) == 7 * 24 - 1  # 26 March has 23 hours in Warsaw


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        (dict(validation=V, test=T), ValueError, "never reads the test period"),
        (dict(validation="previous_year"), ValueError, "one search per"),
        (dict(validation=None), ValueError, "validation=\\(start, end\\)"),
    ],
)
def test_what_a_search_refuses(df, kwargs, error, message):
    with pytest.raises(error, match=message):
        search().run(df, horizon=1, **kwargs)


def test_a_dict_of_zones_is_refused(df):
    with pytest.raises(TypeError, match="one frame"):
        search().run({"PL": df}, horizon=1, validation=V)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(sampler="grid"), "sampler"),
        (dict(n_trials=0), "n_trials"),
        (dict(keep=0), "keep"),
        (dict(keep=1.5), "keep"),
        (dict(rungs=[]), "rungs"),
        (dict(rungs=[{"every_nth_day": 0}]), "every_nth_day"),
        (dict(rungs=[{"every": 2}]), "every_nth_day"),
        (dict(rungs=[{"every_nth_day": 2, "seeds": 0}]), "seeds"),
    ],
)
def test_settings_are_checked_when_built(kwargs, message):
    with pytest.raises(ValueError, match=message):
        search(**kwargs)


@pytest.mark.parametrize(
    "space, message",
    [
        ({}, "non-empty"),
        ({"a": []}, "at least one"),
        ({"a": (1.0, 1.0)}, "low < high"),
        ({"a": (0.0, 1.0, "log")}, "low > 0"),
        ({"a": (0.0, 1.0, "linear")}, "range is"),
        ({"a[0]": [1, 2]}, "without"),
        ({"a": "abc"}, "list of choices"),
        ({"a": [object()]}, "must be None"),
    ],
)
def test_a_space_is_checked_when_built(space, message):
    with pytest.raises((ValueError, TypeError), match=message):
        BayesSearch(make, space)


def test_rung_seeds_raise_for_a_model_without_seeds_before_the_dear_rungs(df, monkeypatch):
    """OLS has no seeds: a later rung's seeds fail the first trial before it runs, not after rung 0 is spent."""
    runs = []
    monkeypatch.setattr(ModelPipeline, "run", lambda self, *a, **k: runs.append(1))
    with pytest.raises(ValueError, match="rungs\\[1\\] sets seeds=3, and OLSModel has no seeds parameter"):
        search(rungs=[{"every_nth_day": 4}, {"every_nth_day": 1, "seeds": 3}]).run(df, horizon=1, validation=V)
    assert runs == []


# ---------------------------------------------------------------------------------------------------------------------
# One worker: the same table twice; OLS windows and subsets
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sampler", ["tpe", "random"])
def test_a_fixed_seed_with_one_worker_gives_the_same_table_twice(df, sampler):
    first = search(sampler=sampler, n_trials=12).run(df, horizon=1, validation=V)
    second = search(sampler=sampler, n_trials=12).run(df, horizon=1, validation=V)
    pd.testing.assert_frame_equal(first.table(), second.table())
    assert first.record == second.record
    other = search(sampler=sampler, n_trials=12, seed=1).run(df, horizon=1, validation=V)
    assert not first.table()[["window", "drivers"]].equals(other.table()[["window", "drivers"]])


def test_each_score_is_the_models_validation_mae_on_the_rungs_days(df):
    result = search(n_trials=4).run(df, horizon=1, validation=V)
    table = result.table()
    for row in table.itertuples():
        model = make({"window": row.window, "drivers": row.drivers})
        for r, every in ((0, 4), (1, 1)):
            mae = getattr(row, f"mae_r{r}")
            if np.isnan(mae):
                continue
            pipeline = ModelPipeline()
            pipeline.add_model(model)
            report = pipeline.run(df, horizon=1, validation=V, validation_every=every)
            records = list(report.refs["ols"].iter_records())
            scored = [r_ for r_ in records if r_["target_date"] <= V[1]]  # a later target is the test period's
            assert mae == math.fsum(abs(r_["prediction"] - r_["actual"]) for r_ in scored) / len(scored)  # exact
            assert getattr(row, f"forecasts_r{r}") == len(scored)
    calendar = IssueCalendar("1h", df.index.tz, (1,))
    # every day, over the spring switch, but for 3 April
    assert table["forecasts_r1"].dropna().unique().tolist() == [calendar.count(*V) - 24]
    assert table["forecasts_r0"].dropna().unique().tolist() == [calendar.count(*V, 4)]  # 20, 24, 28 March and 1 April
    assert calendar.count(*V, 4) == 4 * 24 and calendar.count(*V) < 14 * 24


def test_a_trial_that_skips_a_key_gets_no_score(df):
    """A NaN in the wind on one validation issue day: every trial that reads the wind skips it and fails, with the reason,
    so it isn't compared on fewer, easier keys."""
    holed = df.copy()
    day = holed.index.tz_localize(None).normalize() == pd.Timestamp("2023-03-24")
    holed.loc[day & (holed.index.hour == 5), "wind_d+1"] = np.nan
    result = BayesSearch(
        make, {"window": [28], "drivers": Subset(["load", "wind"])}, sampler="random", n_trials=8, seed=0
    ).run(holed, horizon=1, validation=V)
    table = result.table()
    reads_wind = table["drivers"].map(lambda drivers: "wind" in drivers)
    assert reads_wind.any() and (~reads_wind).any()
    assert table.loc[reads_wind, "validation_mae"].isna().all()
    assert table.loc[reads_wind, "error"].str.contains("missing_test_predictors").all()
    assert table.loc[~reads_wind, "validation_mae"].notna().all()
    assert set(result.ranking()) == set(table.loc[~reads_wind, "trial"])


def test_a_subset_under_its_min_size_fails_without_a_run(df, monkeypatch):
    runs = []
    real = ModelPipeline.run
    monkeypatch.setattr(ModelPipeline, "run", lambda self, *a, **k: runs.append(1) or real(self, *a, **k))
    result = BayesSearch(
        make, {"window": [28], "drivers": Subset(["load", "wind"], min_size=2)}, sampler="random", n_trials=10, seed=0
    ).run(df, horizon=1, validation=V)
    table = result.table()
    small = table["drivers"].map(len) < 2
    assert small.any() and table.loc[small, "error"].str.contains("min_size=2").all()
    assert len(runs) == (~small).sum()


class FakeReport:
    """What `score_ref` reads of a run's validation report."""

    def __init__(self, records, planned, skipped=None):
        self._records = records
        self._counts = {"model": "m", "planned": planned, "produced": len(records), "missing": planned - len(records)}
        self._counts.update(skipped or {})

    def forecast_counts(self):
        return pd.DataFrame([self._counts])

    @property
    def refs(self):
        records = self._records

        class Ref:
            def iter_records(self, cols=None):
                return iter(records)

        return {"m": Ref()}


def test_a_score_counts_every_planned_key_but_those_with_no_row():
    records = [{"prediction": 1.0, "actual": 2.0}, {"prediction": 3.0, "actual": float("nan")}]
    assert bayes.score_ref(FakeReport(records, 3, {"missing_test_row": 1}), "m") == bayes._Score(1.0, 1, 3, None)
    lacking = bayes.score_ref(FakeReport(records, 3, {"missing_test_predictors": 1}), "m")
    assert lacking.mae is None and "missing_test_predictors" in lacking.error
    nan = bayes.score_ref(FakeReport([{"prediction": float("nan"), "actual": 1.0}], 1), "m")
    assert nan.mae is None and "not finite" in nan.error
    assert (
        bayes.score_ref(FakeReport([{"prediction": 1.0, "actual": float("nan")}], 1), "m").error
        == "no forecast has an actual"
    )


def test_a_score_is_the_same_whatever_order_the_pool_finished_in():
    rng = np.random.default_rng(1)
    records = [{"prediction": float(p), "actual": float(a)} for p, a in rng.normal(50, 30, (2000, 2))]
    scores = set()
    for seed in range(20):
        order = np.random.default_rng(seed).permutation(len(records))
        scores.add(bayes.score_ref(FakeReport([records[i] for i in order], len(records)), "m").mae)
    assert len(scores) == 1


# ---------------------------------------------------------------------------------------------------------------------
# Successive halving
# ---------------------------------------------------------------------------------------------------------------------


def fake_score(self, values, r, run):
    """A cheap stand-in for a trial's run: a smooth function of the settings, worse on cheap rungs."""
    value = (values["x"] - 0.3) ** 2 + 0.01 * values["k"] + 0.1 / (r + 1)
    return bayes._Score(value, 10 * (r + 1), 10 * (r + 1), None)


def fake_search(**kwargs):
    arguments = dict(
        space={"x": (0.0, 1.0), "k": [0, 1, 2, 3]},
        n_trials=100,
        seed=0,
        rungs=[{"every_nth_day": 8}, {"every_nth_day": 2}, {"every_nth_day": 1}],
        keep=1 / 3,
    )
    arguments.update(kwargs)
    return BayesSearch(make, **arguments)


@pytest.mark.parametrize("sampler", ["tpe", "random"])
def test_100_trials_keeping_a_third_promote_33_then_11(df, monkeypatch, sampler):
    monkeypatch.setattr(BayesSearch, "_score", fake_score)
    result = fake_search(sampler=sampler).run(df, horizon=1, validation=V)
    table = result.table()
    assert len(table) == 100 and table["mae_r0"].notna().all()
    assert table["mae_r1"].notna().sum() == 33 and table["mae_r2"].notna().sum() == 11
    first, second = result.record["promoted"]
    assert first == table.sort_values(["mae_r0", "trial"])["trial"].tolist()[:33]
    assert second == table.dropna(subset=["mae_r1"]).sort_values(["mae_r1", "trial"])["trial"].tolist()[:11]
    assert result.ranking()[:11] == table.dropna(subset=["mae_r2"]).sort_values(["mae_r2", "trial"])["trial"].tolist()
    assert len(result.finalists()) == 11
    assert table["rung"].value_counts().to_dict() == {0: 67, 1: 22, 2: 11}


def test_tpe_proposes_better_settings_than_its_random_start(df, monkeypatch):
    monkeypatch.setattr(BayesSearch, "_score", fake_score)
    table = fake_search(rungs=None, n_trials=60).run(df, horizon=1, validation=V).table()
    assert table["mae_r0"].iloc[30:].median() < table["mae_r0"].iloc[:10].median()


def test_a_failed_trial_counts_as_tried_and_is_not_promoted(df, monkeypatch):
    def failing(self, values, r, run):
        if values["k"] == 0:
            return bayes._Score(None, 3, 10, "forecast 3 of 10 planned keys (skipped: {'fit_error': 7})")
        return fake_score(self, values, r, run)

    monkeypatch.setattr(BayesSearch, "_score", failing)
    result = fake_search(n_trials=30, rungs=[{"every_nth_day": 4}, {"every_nth_day": 1}], keep=0.5).run(
        df, horizon=1, validation=V
    )
    table = result.table()
    assert table["mae_r1"].notna().sum() == min(15, (table["k"] != 0).sum())
    assert table.loc[table["k"] == 0, "error"].str.contains("fit_error").all()
    assert not set(result.record["promoted"][0]) & set(table.loc[table["k"] == 0, "trial"])


def test_an_exception_in_a_trial_stops_the_search_and_marks_the_trial(df, monkeypatch, tmp_path):
    def broken(self, values, r, run):
        raise RuntimeError("planted")

    monkeypatch.setattr(BayesSearch, "_score", broken)
    journal = tmp_path / "s.journal"
    with pytest.raises(RuntimeError, match="planted"):
        fake_search(n_trials=5, storage=journal).run(df, horizon=1, validation=V)
    import optuna

    study = optuna.load_study(
        study_name=bayes.STUDY, storage=bayes.BayesSearch(make, SPACE, storage=journal)._storage()
    )
    trials = study.get_trials()
    assert [t.state.name for t in trials] == ["FAIL"] and "planted" in trials[0].user_attrs["error"]


# ---------------------------------------------------------------------------------------------------------------------
# Storage: resume, other settings, two processes, requeue
# ---------------------------------------------------------------------------------------------------------------------


def test_a_rerun_on_its_journal_resumes_and_gives_the_same_result(df, monkeypatch, tmp_path):
    monkeypatch.setattr(BayesSearch, "_score", fake_score)
    journal = tmp_path / "s.journal"
    first = fake_search(n_trials=20, storage=journal).run(df, horizon=1, validation=V)
    calls = []
    monkeypatch.setattr(BayesSearch, "_score", lambda *a: calls.append(1) or fake_score(*a))
    again = fake_search(n_trials=20, storage=journal).run(df, horizon=1, validation=V)
    assert calls == [] and again.record == first.record
    pd.testing.assert_frame_equal(again.table(), fake_search(n_trials=20).run(df, horizon=1, validation=V).table())


def test_a_resume_with_another_make_is_refused(df, tmp_path):
    """A search stopped by an error and resumed after `make` was edited would rank trials of two models: refused, while a
    resume with the same `make` goes on and best() of its saved result needs that make."""
    journal = tmp_path / "s.journal"
    calls = {"n": 0}

    def crashing(params):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("worker crashed")
        return make(params)

    def edited(params):
        return OLSModel(["load_d+{horizon}"] + FIXED, training_window=params["window"], name="ols")

    settings = dict(n_trials=4, seed=0, rungs=search().rungs, keep=0.5, storage=journal, sampler="random")
    with pytest.raises(RuntimeError, match="worker crashed"):
        BayesSearch(crashing, SPACE, **settings).run(df, horizon=1, validation=V)
    with pytest.raises(ValueError, match="scored with another model than make builds now"):
        BayesSearch(edited, SPACE, **settings).run(df, horizon=1, validation=V)
    result = search(n_trials=4, storage=journal, sampler="random").run(df, horizon=1, validation=V)
    trials = result.record["trials"]
    assert all(trial["model"] for trial in trials if trial["rungs"][0]["state"] == "complete")
    assert [trial["model"] for trial in trials if "worker crashed" in str(trial["rungs"][0]["error"])] == [None]
    result.save(tmp_path / "s.yaml")
    loaded = SearchResult.load(tmp_path / "s.yaml")
    assert loaded.best(1, make=make)[0].name == "ols_bayes_00"
    with pytest.raises(ValueError, match="make builds another model"):
        loaded.best(1, make=edited)


@pytest.mark.parametrize(
    "change, message",
    [
        (dict(n_trials=21), "n_trials"),
        (dict(seed=1), "seed"),
        (dict(space={"x": (0.0, 2.0), "k": [0, 1, 2, 3]}), "space"),
        (dict(keep=0.5), "keep"),
    ],
)
def test_a_journal_of_another_search_is_refused(df, monkeypatch, tmp_path, change, message):
    monkeypatch.setattr(BayesSearch, "_score", fake_score)
    journal = tmp_path / "s.journal"
    fake_search(n_trials=20, storage=journal).run(df, horizon=1, validation=V)
    with pytest.raises(ValueError, match=f"other settings or data \\({message}"):
        fake_search(**{"n_trials": 20, "storage": journal, **change}).run(df, horizon=1, validation=V)


def test_a_journal_of_other_data_is_refused(df, monkeypatch, tmp_path):
    monkeypatch.setattr(BayesSearch, "_score", fake_score)
    journal = tmp_path / "s.journal"
    fake_search(n_trials=5, storage=journal).run(df, horizon=1, validation=V)
    changed = df.copy()
    changed.iloc[100, 0] += 1.0
    with pytest.raises(ValueError, match="data_hash"):
        fake_search(n_trials=5, storage=journal).run(changed, horizon=1, validation=V)


def test_several_workers_need_a_storage(df, monkeypatch):
    monkeypatch.setenv(bayes.ENV_WORKERS, "2")
    with pytest.raises(ValueError, match="storage="):
        search().run(df, horizon=1, validation=V)


WORKER = """
import json, os, sys, time
sys.path.insert(0, {root!r})
from tests.test_tune import fake_score, fake_search, V
from tests.synthetic import make_market_data
from epftoolbox2.tune import BayesSearch, bayes

log = {log!r}
joined = []

def logged(self, values, r, run):
    if not joined:  # wait, once, until both workers are in: each then takes trials, however late the other started
        with open(log + ".started", "a") as f:
            f.write(str(os.getpid()) + "\\n")
        deadline = time.time() + 120
        while len(set(open(log + ".started").read().split())) < 2 and time.time() < deadline:
            time.sleep(0.05)
        joined.append(True)
    with open(log, "a") as f:
        f.write(json.dumps({{"pid": os.getpid(), "rung": r, "values": values}}) + "\\n")
    time.sleep(0.3)
    return fake_score(self, values, r, run)

BayesSearch._score = logged
df = make_market_data(start="2023-01-01", days=110, horizon=2, noise=5.0, seed=3)
result = fake_search(n_trials={n}, storage={journal!r}).run(df, horizon=1, validation=V)
result.save({out!r} + "." + os.environ["EPF_TUNE_WORKER"] + ".yaml")
"""


def run_cli(*args, env=None, timeout=300):
    return subprocess.run(
        [sys.executable, "-c", "from epftoolbox2.cli import main; main()", *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def worker_env(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith("EPF_TUNE")}
    env.update({bayes.ENV_POLL: "0.05", "MAX_PROCESSES": "1", "THREADS_PER_PROCESS": "1", "PYTHONPATH": str(ROOT)})
    return env


def test_two_processes_on_one_journal_finish_every_trial_once(tmp_path):
    """`epf tune --workers 2`: both workers take trials, each (rung, trial) is evaluated once, and both write one result."""
    log, journal, out = tmp_path / "calls.jsonl", tmp_path / "s.journal", tmp_path / "result"
    script = tmp_path / "search.py"
    script.write_text(
        textwrap.dedent(WORKER.format(root=str(ROOT), log=str(log), n=24, journal=str(journal), out=str(out)))
    )
    done = run_cli("tune", str(script), "--workers", "2", env=worker_env(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len({call["pid"] for call in calls}) == 2  # both took trials
    first, second = (SearchResult.load(tmp_path / f"result.{i}.yaml") for i in (0, 1))
    assert first.record == second.record
    table = first.table()
    assert len(table) == 24 and table["mae_r0"].notna().all()
    assert table["mae_r1"].notna().sum() == 8 and table["mae_r2"].notna().sum() == 2
    by_rung = {r: [call for call in calls if call["rung"] == r] for r in (0, 1, 2)}
    assert [len(by_rung[r]) for r in (0, 1, 2)] == [24, 8, 2]  # each trial of each rung once
    for r in (1, 2):
        evaluated = sorted(json.dumps(call["values"], sort_keys=True) for call in by_rung[r])
        promoted = first.record["promoted"][r - 1]
        expected = sorted(json.dumps(first.params(number), sort_keys=True) for number in promoted)
        assert evaluated == expected


def test_requeue_marks_a_dead_workers_trial_and_a_rerun_redoes_its_slot(df, monkeypatch, tmp_path):
    monkeypatch.setattr(BayesSearch, "_score", fake_score)
    journal = tmp_path / "s.journal"
    searcher = fake_search(n_trials=6, storage=journal, rungs=None)
    # A worker killed during trial 2: the trials before it finished, it stays running.
    import optuna

    calls = {"n": 0}

    def dies(self, values, r, run):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return fake_score(self, values, r, run)

    monkeypatch.setattr(BayesSearch, "_score", dies)
    with pytest.raises(KeyboardInterrupt):
        searcher.run(df, horizon=1, validation=V)
    storage = searcher._storage()
    study = optuna.load_study(study_name=bayes.STUDY, storage=storage)
    interrupted = [t for t in study.get_trials() if t.user_attrs.get("interrupted")]
    assert len(interrupted) == 1  # a KeyboardInterrupt marks its trial itself
    # A trial left running by a process that was killed outright: what --requeue is for.
    stuck = study.ask()
    monkeypatch.setenv(bayes.ENV_REQUEUE, "1")
    with pytest.raises(SystemExit):
        fake_search(n_trials=6, storage=journal, rungs=None).run(df, horizon=1, validation=V)
    monkeypatch.delenv(bayes.ENV_REQUEUE)
    assert (
        optuna.load_study(study_name=bayes.STUDY, storage=storage).get_trials()[stuck.number].user_attrs["interrupted"]
    )
    monkeypatch.setattr(BayesSearch, "_score", fake_score)
    result = fake_search(n_trials=6, storage=journal, rungs=None).run(df, horizon=1, validation=V)
    table = result.table()
    assert len(table) == 6 and table["mae_r0"].notna().all()
    assert stuck.number not in table["trial"].tolist() and interrupted[0].number not in table["trial"].tolist()


def test_epf_tune_requeue_runs_the_script_once_to_requeue(tmp_path):
    journal, out = tmp_path / "s.journal", tmp_path / "result"
    script = tmp_path / "search.py"
    script.write_text(
        textwrap.dedent(
            WORKER.format(root=str(ROOT), log=str(tmp_path / "log"), n=4, journal=str(journal), out=str(out))
        )
    )
    done = run_cli("tune", str(script), "--requeue", env=worker_env(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Requeued 0 interrupted trial(s)" in done.stdout
    assert not (tmp_path / "result.0.yaml").exists()


def test_epf_tune_refuses_a_missing_script(tmp_path):
    done = run_cli("tune", str(tmp_path / "nothing.py"), env=worker_env(tmp_path))
    assert done.returncode == 2 and "No such script" in done.stdout


# ---------------------------------------------------------------------------------------------------------------------
# Warm start, the result, best() and provenance
# ---------------------------------------------------------------------------------------------------------------------


def test_warm_start_enqueues_last_years_best(df, monkeypatch, tmp_path):
    monkeypatch.setattr(BayesSearch, "_score", fake_score)
    last_year = fake_search(n_trials=30).run(df, horizon=1, validation=V)
    last_year.save(tmp_path / "last.yaml")
    finalists = last_year.finalists()
    assert len(finalists) == 3
    for warm in (tmp_path / "last.yaml", last_year):
        this_year = fake_search(n_trials=30, seed=7, warm_start=warm).run(df, horizon=1, validation=V)
        table = this_year.table()
        assert table["warm_start"].tolist() == [True] * 3 + [False] * 27
        assert [this_year.params(n) for n in range(3)] == [last_year.params(n) for n in finalists]
        assert this_year.record["search"]["warm_start"] == str(tmp_path / "last.yaml")


def test_warm_start_leaves_out_what_the_space_cant_hold(df, monkeypatch, tmp_path):
    monkeypatch.setattr(BayesSearch, "_score", fake_score)
    last_year = fake_search(n_trials=30).run(df, horizon=1, validation=V)
    finalists = [last_year.params(n) for n in last_year.finalists()]
    xs = sorted(params["x"] for params in finalists)
    assert xs[0] < xs[-1]
    cut = (xs[0] + xs[-1]) / 2  # a space that holds some finalists' x and not the others'
    with pytest.warns(UserWarning, match="left out"):
        this_year = fake_search(n_trials=10, space={"x": (0.0, cut), "k": [0, 1, 2, 3]}, warm_start=last_year).run(
            df, horizon=1, validation=V
        )
    held = [params for params in finalists if params["x"] <= cut]
    assert 0 < len(held) < len(finalists)
    assert [this_year.params(n) for n in range(len(held))] == held
    assert this_year.table()["warm_start"].sum() == len(held)


def test_best_gives_ordinary_models_named_after_the_search(df, tmp_path):
    result = search(n_trials=6).run(df, horizon=1, validation=V)
    models = result.best(2)
    assert [model.name for model in models] == ["ols_bayes_00", "ols_bayes_01"]
    assert [type(model) for model in models] == [OLSModel, OLSModel]
    first = result.params(result.ranking()[0])
    assert models[0].training_window == first["window"]
    assert models[0].predictors == [f"{c}_d+{{horizon}}" for c in first["drivers"]] + FIXED
    assert result.record["chosen"] == [
        {"k": 2, "seeds": None, "models": [{"name": m.name, "trial": n} for m, n in zip(models, result.ranking())]}
    ]
    with pytest.raises(ValueError, match="best\\(7\\)"):
        result.best(7)
    with pytest.raises(ValueError, match="no seeds parameter"):
        result.best(1, seeds=4)


def test_save_holds_the_space_every_trial_and_the_choice_and_loads_back(df, tmp_path):
    result = search(n_trials=6).run(df, horizon=1, validation=V)
    models = result.best(1)
    assert models[0]._search["file"] is None
    result.save(tmp_path / "s.yaml")
    assert models[0]._search["file"] == str(tmp_path / "s.yaml")
    loaded = SearchResult.load(tmp_path / "s.yaml")
    assert loaded.record == result.record
    pd.testing.assert_frame_equal(loaded.table(), result.table())
    saved = loaded.record
    assert saved["search"]["space"] == {
        "window": {"choice": [14, 28, 42]},
        "drivers": {"subset": ["load", "wind"], "min_size": 0},
    }
    assert saved["search"]["validation"] == list(V) and saved["search"]["data_through"] == V[1]
    assert [trial["trial"] for trial in saved["trials"]] == list(range(6))
    assert all(
        set(rung) >= {"rung", "every_nth_day", "validation_mae", "forecasts"}
        for t in saved["trials"]
        for rung in t["rungs"]
    )
    assert saved["chosen"][0]["models"][0]["name"] == "ols_bayes_00"
    with pytest.raises(ValueError, match="pass make="):
        loaded.best(1)
    again = loaded.best(1, make=make)
    assert again[0].training_window == models[0].training_window
    assert len(SearchResult.load(tmp_path / "s.yaml").record["chosen"]) == 2  # a best() after save goes into the file


def test_a_chosen_models_manifest_and_report_say_where_it_was_chosen(df, tmp_path):
    result = search(n_trials=6).run(df, horizon=1, validation=V)
    result.save(tmp_path / "s.yaml")
    models = result.best(2)
    pipeline = ModelPipeline()
    for model in models:
        pipeline.add_model(model)
    pipeline.add_model(OLSModel(["load_d+{horizon}"] + FIXED, training_window=28, name="plain"))
    pipeline.add_evaluator(MAEEvaluator())
    report = pipeline.run(df, horizon=1, validation=V, test=T, save_dir=str(tmp_path / "run"))
    manifest = json.loads(manifest_path(tmp_path / "run" / "ols_bayes_00.jsonl").read_text())
    assert manifest["search"]["file"] == str(tmp_path / "s.yaml") and manifest["search"]["trial"] == result.ranking()[0]
    assert "search" not in json.loads(manifest_path(tmp_path / "run" / "plain.jsonl").read_text())
    for shown in (report, report.validation, pipeline.report(tmp_path / "run", horizon=1)):
        choices = shown.choices()
        assert choices["chosen"].tolist() == [["ols_bayes_00"], ["ols_bayes_01"]]
        assert choices["ensemble"].tolist() == ["ols_bayes"] * 2 and choices["candidates"].tolist() == [6, 6]
        summary = report.summary().set_index("model")
        for row in choices.itertuples():
            assert row.test_mae == pytest.approx(summary.loc[row.chosen[0], "MAE"], abs=1e-12)
            assert row.gap == pytest.approx(row.test_mae - row.validation_mae, abs=1e-12)
            assert (row.validation_start, row.validation_end, row.test_start, row.test_end) == (*V, *T)
    assert report.provenance().empty  # no card: nothing new in the licence table
    # every report of the run shows the same rows: of one window, of fewer horizons, aggregated
    whole = report.choices()
    for other in (
        pipeline.report(tmp_path / "run", horizon=1, validation=V),
        pipeline.report(tmp_path / "run", horizon=1, test=T),
        report.aggregate("1D", agg="mean"),
    ):
        pd.testing.assert_frame_equal(other.choices(), whole)
    assert whole.attrs["search"]["ols_bayes_00"]["trial"] == result.ranking()[0]


def test_a_chosen_model_is_never_tested_on_the_days_it_was_chosen_on(df, monkeypatch):
    """A test period starting on or before the search's validation end raises before any fit; one after it runs."""
    result = search(n_trials=4).run(df, horizon=1, validation=V)
    models = result.best(1)
    assert models[0]._search["data_through"] == V[1]
    fits = []
    real = OLSModel.run
    monkeypatch.setattr(OLSModel, "run", lambda self, *a, **k: fits.append(1) or real(self, *a, **k))
    for test in ((V[1], "2023-04-05"), ("2023-03-30", "2023-04-05")):
        with pytest.raises(ValueError, match="must not be scored on days its settings were chosen on"):
            ModelPipeline().add_model(models[0]).run(df, horizon=1, test=test)
        with pytest.raises(ValueError, match="chose it on data through 2023-04-02"):
            ModelPipeline().add_model(models[0]).run({"PL": df}, horizon=1, test=test)
    with pytest.raises(ValueError, match="must not be scored"):
        ModelPipeline().add_model(models[0]).run(
            df, horizon=1, validation=("2023-03-01", "2023-03-20"), test=(V[1], T[1])
        )
    assert fits == []
    ModelPipeline().add_model(models[0]).run(df, horizon=1, test=("2023-04-03", "2023-04-05"))
    ModelPipeline().add_model(models[0]).run(df, horizon=1, validation=V)  # no test: nothing to refuse
    assert len(fits) == 2


def test_ensemble_choices_show_their_candidates_and_gap(df):
    pipeline = ModelPipeline()
    for window in (14, 28, 42):
        pipeline.add_model(OLSModel(["load_d+{horizon}"] + FIXED, training_window=window, name=f"w{window}"))
    from epftoolbox2.ensembles import AverageEnsemble

    pipeline.add_ensemble(AverageEnsemble(["w14", "w28", "w42"], top=2, name="top2"))
    pipeline.add_evaluator(MAEEvaluator())
    report = pipeline.run(df, horizon=1, validation=V, test=T)
    row = report.choices().iloc[0]
    assert row["candidates"] == 3
    assert row["test_mae"] == pytest.approx(report.summary().set_index("model").loc["top2", "MAE"], abs=1e-12)
    assert row["gap"] == pytest.approx(row["test_mae"] - row["validation_mae"], abs=1e-12)
    pd.testing.assert_frame_equal(report.validation.choices(), report.choices())
    alone = pipeline.run(df, horizon=1, validation=V)  # no test window: nothing to compare with yet
    assert math.isnan(alone.choices().iloc[0]["test_mae"]) and math.isnan(alone.choices().iloc[0]["gap"])


# ---------------------------------------------------------------------------------------------------------------------
# Imports and the extra
# ---------------------------------------------------------------------------------------------------------------------


def test_importing_the_package_or_the_module_imports_no_optuna():
    code = "import sys, epftoolbox2, epftoolbox2.tune; sys.exit('optuna' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], cwd=str(ROOT)).returncode == 0


def test_without_optuna_a_search_names_the_extra(df, monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)  # an import of it raises ImportError
    with pytest.raises(MissingExtraError, match=r"epftoolbox2\[tune\]"):
        search().run(df, horizon=1, validation=V)


# ---------------------------------------------------------------------------------------------------------------------
# A torch model: rung seeds in place of the model's
# ---------------------------------------------------------------------------------------------------------------------


def test_rung_seeds_replace_a_torch_models_seeds(df, monkeypatch):
    pytest.importorskip("torch")
    from epftoolbox2.models import MLPHourlyModel

    monkeypatch.setenv("TORCH_STACKS", "off")  # each model on its own, through the pool above
    seen = []
    real = ModelPipeline.run

    def spy(self, data, *args, **kwargs):
        seen.append((kwargs["validation_every"], self.models[0].seeds))
        return real(self, data, *args, **kwargs)

    monkeypatch.setattr(ModelPipeline, "run", spy)

    def make_mlp(params):
        return MLPHourlyModel(
            ["load_d+{horizon}", "wind_d+{horizon}"] + FIXED,
            hidden=[params["hidden"]],
            lr=params["lr"],
            training_window=21,
            refit="7D",
            max_epochs=3,
            seeds=5,
            device="cpu",
            name="mlp",
        )

    result = BayesSearch(
        make_mlp,
        {"hidden": [4, 8], "lr": (1e-3, 1e-2, "log")},
        sampler="random",
        n_trials=2,
        rungs=[{"every_nth_day": 7, "seeds": 1}, {"every_nth_day": 7, "seeds": 2}],
        keep=0.5,
    ).run(df, horizon=1, validation=("2023-03-27", "2023-04-02"))
    assert seen == [(7, 1), (7, 1), (7, 2)]
    models = result.best(1, seeds=3)
    assert models[0].seeds == 3 and models[0].name == "mlp_bayes_00"
    assert result.best(1)[0].seeds == 5  # the model as make builds it
