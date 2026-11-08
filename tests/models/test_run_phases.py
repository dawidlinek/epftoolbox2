"""`BaseModel.run` as its three phases, and the worker's task functions with explicit inputs (step 5g).

`run` is `_prepare_run` (windows, plan, checks, period frames, masks, store, manifest, pending tasks, the skips decided before any
fit), `_execute` (one pool of `_run_tasks`, its records written through a sink) and `_finish_run` (counts and the ref). The engine
of step 5h calls the phases itself, for many models and one pool, and runs the worker functions with each model's own inputs
(`_worker.TaskInputs`) in place of the module globals a pool loads. These tests hold each piece to the plain run, record for
record: the composition by hand, two models' inputs in one process, period frames read once for two models, and a pipeline that
checks each model once and hands the check to its run. Warnings of a run still point at the caller's line.
"""

import itertools
import json
import warnings
from collections import Counter
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epftoolbox2.information_set import Availability, InformationSet
from epftoolbox2.models import NaiveModel, OLSModel, RidgeModel
from epftoolbox2.models import _worker as worker
from epftoolbox2.models.base import BaseModel, _Checked, _MemorySink
from epftoolbox2.pipelines import ModelPipeline
from tests.synthetic import PREDICTORS, make_market_data

TEST = ("2023-03-20", "2023-03-29")


class InProcessPool:
    """Stands in for the worker pool, whose start is most of the time these runs would take."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        worker._worker_init(*initargs)

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
    monkeypatch.setattr("epftoolbox2.models.base.ProcessPoolExecutor", InProcessPool)


@pytest.fixture(scope="module")
def market():
    data = make_market_data(days=100, noise=5.0, seed=31, horizon=2)
    data["aux"] = np.random.default_rng(32).normal(0, 1, len(data))
    issue = (data.index.date == pd.Timestamp("2023-03-24").date()) & (data.index.hour == 4)
    data.loc[issue, "load_d+1"] = np.nan  # a missing predictor on an issue row: a skip decided in the worker
    return data.drop(index=data.index[(data.index.date == pd.Timestamp("2023-03-25").date()) & (data.index.hour == 9)])


def texts(records):
    return sorted(json.dumps(record, sort_keys=True) for record in records)


def lines(path):
    return sorted(line for line in Path(path).read_text().splitlines() if line.strip())


def models():
    return [
        OLSModel(PREDICTORS, 28, name="ols"),
        OLSModel([*PREDICTORS, "aux"], [14, 28], binary_mask="fit", name="list_fit"),
        RidgeModel(PREDICTORS, [14, 28], refit="7D", name="ridge_refit"),
        NaiveModel(["price_d-1"], name="naive"),
    ]


def prepared(model, data, save_to=None, **kwargs):
    arguments = dict(
        data=data,
        test_start=TEST[0],
        test_end=TEST[1],
        target="price",
        horizon=2,
        freq="1h",
        save_to=save_to,
        forecast_only=False,
        information_set=None,
        allow_manifest_mismatch=False,
        require_manifest=False,
        plan=None,
        availability="raise",
        warn_listing=True,
        windows=None,
        frame_hashes=None,
        cut_rules=None,
        horizons=None,
        issue_days=None,
    )
    return model._prepare_run(**{**arguments, **kwargs})


@pytest.mark.parametrize("index", range(4), ids=[m.name for m in models()])
def test_run_is_its_three_phases(market, tmp_path, index):
    """The phases called by hand give the run's records, its file line for line and its ref's counts."""
    plain = models()[index].run(market, *TEST, horizon=2)
    model = models()[index]
    state = prepared(model, market)
    try:
        model._execute(state)
        ref = model._finish_run(state)
    finally:
        state.close()
    assert (
        texts(ref._results) == texts(plain._results) and ref.skipped == plain.skipped and ref.produced == plain.produced
    )

    saved = tmp_path / "plain" / "m.jsonl"
    models()[index].run(market, *TEST, horizon=2, save_to=str(saved))
    by_hand = tmp_path / "phases" / "m.jsonl"
    model = models()[index]
    state = prepared(model, market, save_to=str(by_hand))
    try:
        model._execute(state)
        model._finish_run(state)
    finally:
        state.close()
    assert lines(by_hand) == lines(saved)


def test_a_sink_gets_every_record_of_the_pool(market):
    """`_execute(state, sink)` writes each record through the sink it is given, in place of the run's list."""
    model = models()[2]
    state = prepared(model, market)
    received = []
    model._execute(state, _MemorySink(received))
    state.close()
    assert texts(received) == texts(models()[2].run(market, *TEST, horizon=2)._results)
    assert state.in_memory == []


def inputs_of(model, data, **kwargs):
    """A model's `TaskInputs` and its tasks by pool job, as `_run_tasks` builds them, without a pool."""
    state = prepared(model, data, **kwargs)
    hour_arrays = model._build_hour_arrays(max(state.horizons))
    expanded, config = model._worker_inputs(state, 1)
    jobs = model._jobs(state.tasks, state.first_day)
    model._cleanup()
    state.close()
    return worker.task_inputs(hour_arrays, expanded, config, type(model), model._model_kwargs), jobs


def test_two_models_inputs_in_one_process_give_each_model_s_records(market):
    """The worker functions read only the inputs they are given: two models' jobs, interleaved in one process whose globals
    hold a third model's, give each model's own records and skips."""
    first, second = models()[0], models()[2]
    first_inputs, first_jobs = inputs_of(first, market)
    second_inputs, second_jobs = inputs_of(second, market)
    third, _ = inputs_of(models()[1], market)
    worker._worker_init(third.hour_arrays, third.expanded, third.config, OLSModel, {})
    got = {"ols": ([], Counter()), "ridge_refit": ([], Counter())}
    first = [("ols", worker._run_day, tasks, first_inputs) for tasks in first_jobs.values()]
    second = [("ridge_refit", worker._run_block, tasks, second_inputs) for tasks in second_jobs.values()]
    for job in itertools.chain.from_iterable(itertools.zip_longest(first, second)):
        if job is not None:
            name, run, tasks, inputs = job
            results, skipped = run(tasks, inputs)
            got[name][0].extend(results)
            got[name][1].update(reason.partition(": ")[0] for _, reason in skipped)
    for name, model in (("ols", models()[0]), ("ridge_refit", models()[2])):
        ref = model.run(market, *TEST, horizon=2)
        assert texts(got[name][0]) == texts(ref._results), name
        # A missing issue row is decided in the parent (`_prepare_run`), before any task reaches a worker.
        assert got[name][1] == Counter({k: n for k, n in ref.skipped.items() if k != "missing_test_row"}), name
        assert ref.skipped["missing_test_row"] and ref.skipped["missing_test_predictors"], name


def test_period_frames_read_once_serve_two_models(market, tmp_path):
    """`_prepare_run(periods=...)` with frames read for the union of two models' columns gives each model's own run, byte for
    byte; frames of another frame, horizon or freq, or without a column, are refused."""
    reader = models()[0]
    reader._set_freq("1h")
    expanded = [name.replace("{horizon}", str(h)) for h in (1, 2) for name in PREDICTORS]
    union = ["price", *expanded, "aux", "price_d-1"]
    periods = reader._load_periods(market, union, 2, "price")
    reader._cleanup()
    for index in range(4):
        expected = tmp_path / "own" / f"{index}.jsonl"
        models()[index].run(market, *TEST, horizon=2, save_to=str(expected))
        shared = tmp_path / "shared" / f"{index}.jsonl"
        model = models()[index]
        state = prepared(model, market, save_to=str(shared), periods=periods)
        try:
            model._execute(state)
            model._finish_run(state)
        finally:
            state.close()
        assert lines(shared) == lines(expected), index
        provenance = ("package_version", "git_sha", "created_at", "updated_at", "record_schema")
        shared_manifest, own_manifest = (
            {
                k: v
                for k, v in json.loads(path.with_name(f"{index}.manifest.json").read_text()).items()
                if k not in provenance
            }
            for path in (shared, expected)
        )
        assert shared_manifest == own_manifest, index
    for wrong, match in (
        (dict(data=market.copy()), "another frame"),
        (dict(horizon=1), "another horizon"),
    ):
        with pytest.raises(ValueError, match=match):
            prepared(models()[0], **{"data": market, **wrong}, periods=periods)
    narrow = reader._load_periods(market, ["price", *expanded], 2, "price")
    reader._cleanup()
    with pytest.raises(ValueError, match="don't hold"):
        prepared(models()[1], market, periods=narrow)


def test_a_pipeline_checks_each_model_once_and_its_run_uses_that_check(market, monkeypatch, tmp_path):
    """`ModelPipeline.run` checks every model before any runs and hands each run its check: one `_check_inputs` per model, where
    there were two, and the same files. A plain run into results made with windows is a run of the test window alone, which
    its check wasn't for: that run checks again."""
    calls = []
    original = BaseModel._check_inputs

    def counted(self, *args, **kwargs):
        calls.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(BaseModel, "_check_inputs", counted)
    pipeline = ModelPipeline()
    for model in models():
        pipeline.add_model(model)
    pipeline.run(market, *TEST, horizon=2, save_dir=str(tmp_path / "plain"))
    assert calls == [model.name for model in models()]

    calls.clear()
    windows = dict(validation=("2023-03-10", "2023-03-15"), test=TEST)
    pipeline.run(market, horizon=2, save_dir=str(tmp_path / "windows"), **windows)
    assert calls == [model.name for model in models()]
    calls.clear()
    pipeline.run(market, *TEST, horizon=2, save_dir=str(tmp_path / "windows"))
    assert calls == [model.name for model in models()] * 2


def test_a_run_checked_by_its_pipeline_gives_the_same_ref_as_one_checked_again(market, monkeypatch):
    """The availability listing a run records comes from the pipeline's check: the same as the run's own check gives."""
    info = InformationSet("11:45", "Europe/Brussels", rules={"load": Availability.published("10:00", 10)})
    once = ModelPipeline(information_set=info).add_model(OLSModel(PREDICTORS, 28, name="m"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = once.run(market, *TEST, horizon=2)
        listing = report.refs["m"].availability_listing
        monkeypatch.setattr(_Checked, "is_for", lambda self, run: False)  # every run checks again, as it used to
        again = (
            ModelPipeline(information_set=info)
            .add_model(OLSModel(PREDICTORS, 28, name="m"))
            .run(market, *TEST, horizon=2)
        )
    assert listing and listing == again.refs["m"].availability_listing
    assert texts(report.refs["m"]._results) == texts(again.refs["m"]._results)


def midnight_daily():
    hourly = make_market_data(days=90, noise=5.0, seed=33, horizon=1)
    hourly["load_h-1"] = hourly["load"].shift(1)
    daily = hourly.at_time("12:00").copy()
    return daily.set_axis(daily.index.normalize())


def caught(run):
    with warnings.catch_warnings(record=True) as found:
        warnings.simplefilter("always")
        run()
    return [w for w in found if issubclass(w.category, UserWarning)]


def test_a_run_s_warnings_point_at_its_caller(market, tmp_path):
    """Warnings raised in the phases (a listing, an empty window, a cut tail, a midnight stamp) are attributed to the line that
    called `run`, as they were when `run` was one function."""
    info = InformationSet("11:45", "Europe/Brussels", rules={"load": Availability.published("10:00", 10)})
    gappy = market.copy()
    gappy.loc[gappy.index < pd.Timestamp("2023-03-12", tz=gappy.index.tz), "wind_d+1"] = np.nan
    from epftoolbox2.results.windows import RunWindows

    windows = RunWindows.fixed(("2023-03-05", "2023-03-07"), TEST)
    path = tmp_path / "cut" / "m.jsonl"
    OLSModel(PREDICTORS, 28, name="m").run(market, *TEST, horizon=1, save_to=str(path))
    with open(path, "ab") as f:
        f.write(b'{"run_date": "2023-03-2')

    runs = {
        "not known": lambda: OLSModel(PREDICTORS, 28, name="m").run(
            market, *TEST, horizon=2, information_set=info, availability="warn"
        ),
        "no forecasts in the validation window": lambda: OLSModel(PREDICTORS, 3, name="m").run(
            gappy, windows=windows, horizon=1, save_to=str(tmp_path / "empty" / "m.jsonl")
        ),
        "cut off its last": lambda: OLSModel(PREDICTORS, 28, name="m").run(market, *TEST, horizon=1, save_to=str(path)),
        "stamped at 00:00": lambda: OLSModel(["load_d+1", "load_h-1"], 21, name="m").run(
            midnight_daily(), *TEST, horizon=1, freq="1D"
        ),
    }
    for text, run in runs.items():
        found = [w for w in caught(run) if text in str(w.message)]
        assert len(found) == 1 and found[0].filename == __file__, (
            text,
            [(w.filename, str(w.message)[:60]) for w in found],
        )


def quarter_after_hourly():
    """15-minute rows from 1 April 2023, hourly ones before: a run reading back before April at freq "15min" is refused."""
    quarter = make_market_data(days=300, freq="15min", horizon=1)
    return quarter[(quarter.index >= pd.Timestamp("2023-04-01", tz=quarter.index.tz)) | (quarter.index.minute == 0)]


def test_a_check_from_before_a_setting_changed_is_no_check_of_the_run():
    """A check is of one model in one state: a window widened after it, back onto rows the 15-minute run can't train on, makes
    the run check itself and refuse, as a run without the check does."""
    data, issue = quarter_after_hourly(), "2023-09-01"
    model = OLSModel(PREDICTORS, training_window=45, name="ols")
    checked = model._checked_inputs(data, "price", 1, "15min", None, issue, issue)
    model.training_window = 200
    with pytest.raises(ValueError, match="coarser"):
        model.run(data, issue, issue, horizon=1, freq="15min", checked=checked)


SETTINGS = {
    "training_window": lambda m: setattr(m, "training_window", 56),
    "training_window list": lambda m: setattr(m, "training_window", [28, 56]),
    "predictors": lambda m: m.predictors.append("price_d-1"),
    "refit": lambda m: setattr(m, "refit", "7D"),
    "binary_mask": lambda m: setattr(m, "binary_mask", "fit"),
    "scaler": lambda m: setattr(m, "scaler", "asinh"),
    "name": lambda m: setattr(m, "name", "other"),
    "estimator option": lambda m: setattr(m, "fit_intercept", True),
}


@pytest.mark.parametrize("change", list(SETTINGS))
def test_a_check_is_of_one_model_in_one_state(market, change):
    """`_Checked.is_for` holds a check to the model it was made for, as it was then (`_check_settings`)."""
    model = OLSModel(list(PREDICTORS), 28, name="m")
    checked = model._checked_inputs(market, "price", 2, "1h", None, *TEST)

    def run(of):
        return _Checked.run_of(of, market, "price", 2, None, "1h", None, *TEST, None, "raise", None, None, None)

    assert checked.is_for(run(model))
    assert not checked.is_for(run(OLSModel(list(PREDICTORS), 28, name="m")))  # its twin, another object
    SETTINGS[change](model)
    assert not checked.is_for(run(model))


def test_another_model_s_check_is_ignored(market, monkeypatch):
    """A check handed to another model's run is no check of it: the run checks its own inputs, and its own check it uses."""
    calls = []
    original = BaseModel._check_inputs

    def counted(self, *args, **kwargs):
        calls.append(self.name)
        return original(self, *args, **kwargs)

    first, second = OLSModel(PREDICTORS, 28, name="a"), OLSModel(PREDICTORS, 56, name="b")
    of_first = first._checked_inputs(market, "price", 2, "1h", None, *TEST)
    of_second = second._checked_inputs(market, "price", 2, "1h", None, *TEST)
    monkeypatch.setattr(BaseModel, "_check_inputs", counted)
    ignored = second.run(market, *TEST, horizon=2, checked=of_first)
    assert calls == ["b"]
    used = second.run(market, *TEST, horizon=2, checked=of_second)
    assert calls == ["b"] and texts(used._results) == texts(ignored._results)


def test_a_task_never_fits_with_the_process_s_model(market, monkeypatch):
    """A task runs with its inputs whole, or the pool's whole: inputs without a model raise, whatever model the process's
    globals hold, and `_fit_member` has no model of its own to fall back on."""
    import inspect

    plain, plain_jobs = inputs_of(models()[0], market)
    refit, refit_jobs = inputs_of(models()[2], market)
    other, _ = inputs_of(models()[1], market)
    worker._worker_init(other.hour_arrays, other.expanded, other.config, OLSModel, {})  # the process's model
    day_tasks, block_tasks = next(iter(plain_jobs.values())), next(iter(refit_jobs.values()))
    for run, tasks, inputs in (
        (worker._run_day, day_tasks, plain),
        (worker._run_block, block_tasks, refit),
        (lambda tasks, inputs: worker._fit_one_numpy(*tasks[0], inputs), day_tasks, plain),
        (
            lambda tasks, inputs: worker._fit_unit(worker._units_of(inputs.config, tasks[0])[0], tasks, inputs),
            block_tasks,
            refit,
        ),
    ):
        assert run(tasks, inputs)  # whole: its own records
        with pytest.raises(ValueError, match="no model"):
            run(tasks, inputs._replace(model=None))
    assert inspect.signature(worker._fit_member).parameters["model"].default is inspect.Parameter.empty
    monkeypatch.setattr(worker, "_W_MODEL", None)  # a process no `_worker_init` set up
    with pytest.raises(ValueError, match="no model"):
        worker._run_day(day_tasks)
