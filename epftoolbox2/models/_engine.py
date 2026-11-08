"""Engine groups (step 5h): `ModelPipeline(engine="on")` runs the models that can be fitted from shared statistics together.

A **group** is the engine-capable models of a run (`BaseModel._engine_capable`) with one refit block rule (`_refit_block`:
None, a `"kD"`/`"kMS"` rule, or the coarsest of `"auto"`'s) that read their period frames as BaseModel does, in list order;
a model whose class or instance reads them otherwise (`_frame_readers`) is a group of its own. A group is cut into **chunks**
of at most `ENGINE_MAX_MODELS` models (20,000) and `ENGINE_MAX_COLUMNS` union columns (4,096), environment variables like
`MAX_PROCESSES`, read when a group runs. Per chunk: one `_load_periods` over the union of the chunk's columns, each model's
`_prepare_run` on those period frames (its masks, store and manifest, pending tasks and parent-side skips, as its own `run`
has them), one set of period arrays over the union (`base.hour_arrays_of`), and one pool (`_engine_worker._engine_init`, the
native-thread limit of `_worker._pool_init`). A **job** is (job day, period), the job day the issue day without refit or the
block's anchor with it, and holds every pending task of that day or block and period of every model of the chunk, so a task's
window-list members and a unit's tasks are in one job. Each model's records go to its own store through one `StoreWriter` for
the group (or to its in-memory list), one progress bar per group, and each model's `_finish_run` (skip counts, the shortfall
warning, its ref) and its store's close once its chunk has run.

Nothing a model stores depends on its group, chunk, the union or the job: every task runs on its model's own inputs through the
model path's functions (`_engine_worker.run_job`), and a fit reads its own columns by name, the same rows (dates) whatever else
the union holds. A model whose fits come from shared statistics (`_engine_worker.fit_path_of`: OLS, Ridge and LEAR with
`scaler="standard"`) has each fit made from its span's exact sums, which no other model can move by a bit; its manifest records
`fit_path` (compared on resume) and, after its chunk, what those fits did (`results.manifest.record_engine`, provenance). Every
other model's records are engine off's to the bit.
"""

import datetime
import gc
import inspect
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from rich.progress import Progress

from ..results.manifest import manifest_path, record_engine
from ..results.ref import ModelResultRef
from ..results.store import StoreWriter
from ._calendar import resolve_horizons
from ._engine_worker import EngineCounts, EngineTaskFailed, _engine_init, fit_path_of, run_job
from .base import _FRAME_READERS, _WINDOWS_MAX_WORKERS, BaseModel, _MemorySink, _RunState, hour_arrays_of

ENGINE_OFF = "off"
ENGINE_ON = "on"
ENGINE_MODES = (ENGINE_OFF, ENGINE_ON)
# YAML 1.1 (PyYAML) reads an unquoted `on` / `off` as a bool: what each means (`engine_from_yaml`).
_YAML_BOOLS = {True: ENGINE_ON, False: ENGINE_OFF}

# The frame readers of a model that reads its period frames as BaseModel does (`BaseModel._frame_readers`).
_BASE_READERS = tuple(BaseModel for _ in _FRAME_READERS)

# A chunk's limits, and the environment variables that set them (read when a group runs).
ENGINE_MAX_MODELS = 20_000
ENGINE_MAX_COLUMNS = 4_096
MAX_MODELS_VAR = "ENGINE_MAX_MODELS"
MAX_COLUMNS_VAR = "ENGINE_MAX_COLUMNS"

# `BaseModel.run`'s parameters after `self`, and its defaults, for a `_prepare_run` called with what a `run` call names.
_RUN_PARAMETERS = tuple(inspect.signature(BaseModel.run).parameters)[1:]
_RUN_DEFAULTS = {
    name: parameter.default
    for name, parameter in inspect.signature(BaseModel.run).parameters.items()
    if parameter.default is not inspect.Parameter.empty
}


def engine_from_yaml(engine: Any) -> Any:
    """`engine` as a YAML file gives it: `engine: on` unquoted is read as True and `off` as False (YAML 1.1), which mean "on"
    and "off"; anything else as it is, for `check_engine`."""
    return _YAML_BOOLS[engine] if isinstance(engine, bool) else engine


def check_engine(engine: Any) -> str:
    """`engine` if it is one of `ENGINE_MODES`, else ValueError."""
    if engine not in ENGINE_MODES:
        raise ValueError(f"engine must be one of {list(ENGINE_MODES)}, got {engine!r}")
    return engine


def engine_limits() -> Tuple[int, int]:
    """(the most models, the most union columns) of a chunk: `ENGINE_MAX_MODELS` and `ENGINE_MAX_COLUMNS`, or the defaults."""
    limits = []
    for variable, default in ((MAX_MODELS_VAR, ENGINE_MAX_MODELS), (MAX_COLUMNS_VAR, ENGINE_MAX_COLUMNS)):
        raw = os.environ.get(variable)
        try:
            value = default if raw is None else int(raw)
        except ValueError:
            value = 0
        if value < 1:
            raise ValueError(f"{variable} must be a whole number from 1, got {raw!r}")
        limits.append(value)
    return limits[0], limits[1]


def engine_groups(models: Sequence[BaseModel]) -> List[List[BaseModel]]:
    """The engine groups of these (engine-capable) models: one per refit block rule (`_refit_block`) of the models that read
    their period frames as BaseModel does, and one group of its own for each model whose class or instance reads them
    otherwise (`_frame_readers`: a `_preprocess` of its own, say), which reads its own frames as its own run does: such a
    method can depend on the instance (a lag it is given), so two instances of one class can't share frames. In the order of
    each group's first model, its models in list order."""
    groups: Dict[Tuple[Optional[str], Optional[int]], List[BaseModel]] = {}
    for model in models:
        own = None if model._frame_readers() == _BASE_READERS else id(model)
        groups.setdefault((model._refit_block, own), []).append(model)
    return list(groups.values())


def engine_chunks(
    models: Sequence[BaseModel],
    columns: Optional[Dict[str, Sequence[str]]],
    max_models: int,
    max_columns: int,
) -> List[List[BaseModel]]:
    """A group cut into chunks in list order: a model starts a new chunk when the current one has `max_models` models, or when
    its columns (`columns[name]`, its predictors) would take the chunk's union past `max_columns`. A model with more columns
    than that is a chunk alone. No bit depends on where the cuts fall."""
    chunks: List[List[BaseModel]] = []
    current: List[BaseModel] = []
    union: set = set()
    for model in models:
        own = set(columns[model.name]) if columns is not None else set()
        if current and (len(current) >= max_models or len(union | own) > max_columns):
            chunks.append(current)
            current, union = [], set()
        current.append(model)
        union |= own
    if current:
        chunks.append(current)
    return chunks


def run_arguments(**given: Any) -> Dict[str, Any]:
    """`_prepare_run`'s arguments for a run called with `given` (what `BaseModel.run` takes), the others at `run`'s defaults."""
    unknown = set(given) - set(_RUN_PARAMETERS)
    if unknown:
        raise TypeError(f"run() got unexpected keyword arguments {sorted(unknown)}")
    return {**_RUN_DEFAULTS, **given}


def run_group(
    models: Sequence[BaseModel],
    arguments: Dict[str, Dict[str, Any]],
    columns: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, ModelResultRef]:
    """Run one engine group: `models` (one refit block rule, engine-capable), each with its `run` arguments (`arguments[name]`,
    from `run_arguments`; one frame, target, horizons and freq for all), and return each model's ref by name. A model's
    arguments are taken out of `arguments` when its chunk starts, so the caller's dict holds no model's check past its chunk.

    `columns[name]` is each model's predictors (its check's `used_predictors`), which a chunk's period frames are read for once;
    None only for a group of one model, which reads its own (`verify`'s probes, `run_alone`). Called where a `run` would be
    (`ModelPipeline.run`), so each model's `_prepare_run` and `_finish_run` warn at the line their `run` would.

    Each chunk is finished before the next starts: its pool, its stores flushed, each model's `_finish_run` (skip counts, the
    shortfall warning, its ref) and its stores closed, so what a model holds for its run (its tasks, its store's done keys)
    lives as long as its chunk. A task that raises stops the group as a model's `run` stops: no job not yet started runs, what
    came back is in the stores, and a RuntimeError names the model, the day or block and the period; anything else that stops
    the loop (a store's failed write, an interrupt) stops it the same way, and a later run resumes. The
    first error a chunk's `_finish_run` raises (no forecast at all) is raised once the chunk's other models finished, and the
    later chunks don't run, as later models don't with engine off."""
    if not models:
        return {}
    if columns is None and len(models) > 1:
        raise ValueError("an engine group of more than one model needs each model's columns")
    if len(models) > 1 and any(model._frame_readers() != _BASE_READERS for model in models):
        raise ValueError("a model that reads its own period frames is an engine group of its own (`engine_groups`)")
    first = arguments[models[0].name]
    data, target, freq = first["data"], first["target"], first["freq"]
    horizon, _ = resolve_horizons(first["horizon"], first["horizons"], 7)
    chunks = engine_chunks(models, columns, *engine_limits())
    writer = StoreWriter() if any(arguments[model.name]["save_to"] for model in models) else None
    refs: Dict[str, ModelResultRef] = {}
    try:
        with Progress() as progress:
            bar = _Bar(progress, progress.add_task(f"[cyan]{_label(models)}", total=0))
            for chunk in chunks:
                prepared: List[Tuple[BaseModel, _RunState]] = []
                try:
                    periods = None
                    if columns is not None:
                        union = dict.fromkeys(name for model in chunk for name in columns[model.name])
                        chunk[0]._set_freq(freq)
                        periods = chunk[0]._load_periods(data, list(dict.fromkeys([target, *union])), horizon, target)
                    for model in chunk:
                        state = model._prepare_run(
                            **arguments.pop(model.name), periods=periods, writer=writer, fit_path=fit_path_of(model)
                        )
                        prepared.append((model, state))
                    periods = None  # the models hold the frames until `_release`
                    counts = _run_chunk(prepared, bar)
                except BaseException:
                    _release(prepared, writer, chunk, raising=False)
                    _close(prepared, raising=False)
                    raise
                _release(prepared, writer, chunk)
                error: Optional[BaseException] = None
                for model, state in prepared:
                    try:
                        if state.store and counts.get(model.name):
                            record_engine(manifest_path(state.store.path), state.manifest, counts[model.name].to_dict())
                        refs[model.name] = model._finish_run(state)
                    except Exception as exc:  # every model's counts and warnings, then the first failure
                        error = error or exc
                _close(prepared, raising=error is None)
                prepared.clear()
                state = None  # nothing of the chunk's runs outlives it but the refs
                if error is not None:
                    raise error
    except BaseException:
        _close_writer(writer, raising=False)
        raise
    _close_writer(writer)
    return refs


def _label(models: Sequence[BaseModel]) -> str:
    rule = models[0]._refit_block
    who = models[0].name if len(models) == 1 else f"engine: {len(models)} models"
    return who if rule is None or len(models) == 1 else f"{who}, refit {rule}"


class _Bar:
    """A group's one progress bar, counting tasks: each chunk adds its planned tasks to the total, those already done or skipped
    before any fit at once, and each job its tasks as it comes back."""

    def __init__(self, progress: Progress, task: Any):
        self.progress = progress
        self.task = task
        self.total = 0

    def add(self, planned: int, done: int) -> None:
        self.total += planned
        self.progress.update(self.task, total=self.total)
        self.advance(done)

    def advance(self, n: int) -> None:
        if n:
            self.progress.advance(self.task, n)


def _run_chunk(prepared: List[Tuple[BaseModel, _RunState]], bar: _Bar) -> Dict[str, EngineCounts]:
    """One chunk's pool: the union arrays, each model's worker inputs (`_worker_inputs`, explicit config, with its fit path
    `config["fit_path"]` where its fits come from shared statistics, `_engine_worker.fit_path_of`), the jobs, and each job's
    records into each model's sink as it comes back (`_collect`). Returns, by model name, what its fits from shared statistics
    did (`EngineCounts`), for the manifest's provenance."""
    pending = [(model, state) for model, state in prepared if state.tasks]
    bar.add(
        sum(len(state.all_tasks) for _, state in prepared),
        sum(len(state.all_tasks) - len(state.tasks) for _, state in prepared),
    )
    counts: Dict[str, EngineCounts] = {}
    if not pending:
        return counts
    first, state0 = prepared[0]
    union = list(dict.fromkeys(name for model, _ in pending for name in model._used_predictors))
    hour_arrays = hour_arrays_of(
        first._hour_data, first._hour_days, first._target, max(state0.horizons), union, first._periods_per_day
    )
    threads, n_processes = first._executor_config()
    inputs = []
    jobs: Dict[Tuple[Any, int], Dict[int, List[tuple]]] = {}
    for place, (model, state) in enumerate(pending):
        expanded, config = model._worker_inputs(state, threads)
        fit_path = fit_path_of(model)
        if fit_path is not None:
            config = {**config, "fit_path": fit_path}
        inputs.append((model.name, expanded, config, type(model), model._model_kwargs))
        for job, tasks in model._jobs(state.tasks, state.first_day).items():
            for task in tasks:
                jobs.setdefault((job, task[0]), {}).setdefault(place, []).append(task)
    sinks = [state.store if state.store else _MemorySink(state.in_memory) for _, state in pending]
    n_processes = min(n_processes, len(jobs))
    if sys.platform == "win32":
        n_processes = min(n_processes, _WINDOWS_MAX_WORKERS)
    refit = first._refit_block is not None

    mp_context = multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn")
    with ProcessPoolExecutor(
        max_workers=n_processes, mp_context=mp_context, initializer=_engine_init, initargs=(hour_arrays, inputs)
    ) as pool:
        try:
            futures = {pool.submit(run_job, list(job.items())): key for key, job in jobs.items()}
            for future in as_completed(futures):
                # A job's records leave with its future (`as_completed` keeps none it has given), so the parent holds one
                # job's records at a time, not the chunk's.
                key = futures.pop(future)
                job = jobs.pop(key)
                try:
                    outcomes = future.result()
                except Exception as exc:
                    raise _job_failed(exc, key, job, pending, refit) from exc
                for place, results, skipped, fitted in outcomes:
                    model, state = pending[place]
                    model._collect(state, results, skipped, sinks[place])
                    if fitted is not None:
                        counts.setdefault(model.name, EngineCounts()).add(fitted)
                bar.advance(sum(len(tasks) for tasks in job.values()))
        except BaseException:
            # Whatever stops the loop, a task, a store's failed write that `_collect` raises or an interrupt: no job not yet
            # started runs (leaving the `with` would wait for all of them). Results already saved stay in the stores, so a
            # fixed rerun resumes from here.
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    return counts


def _job_failed(
    exc: BaseException,
    key: Tuple[Any, int],
    job: Dict[int, List[tuple]],
    pending: List[Tuple[BaseModel, _RunState]],
    refit: bool,
) -> RuntimeError:
    """The error of a job that raised: the failing model's name (or the job's models', for a pool that broke), its day or
    refit block (from that model's run) and the period."""
    day, period = key
    place = exc.model if isinstance(exc, EngineTaskFailed) else next(iter(job))
    what = (
        f"the refit block from {day}"
        if refit
        else f"run date {pending[place][1].first_day + datetime.timedelta(days=day)}"
    )
    if isinstance(exc, EngineTaskFailed):
        return RuntimeError(f"{pending[place][0].name}: forecasting failed for {what}, period {period}: {exc}")
    names = ", ".join(pending[at][0].name for at in job)
    return RuntimeError(f"forecasting failed for {what}, period {period} (models {names}): {exc}")


def _release(
    prepared: List[Tuple[BaseModel, _RunState]],
    writer: Optional[StoreWriter],
    chunk: Sequence[BaseModel],
    raising: bool = True,
) -> None:
    """After a chunk's pool, however it ended: every store's records written (`StoreWriter.flush`, or each store's own), every
    model's frames let go, and one `gc.collect()` for the chunk (a model's own run collects once per model). A failed write
    raises unless `raising` is False (the chunk is raising already; the store raises it again when it closes)."""
    try:
        if writer is not None:
            writer.flush()
        else:
            for _, state in prepared:
                if state.store:
                    state.store.flush()
    except BaseException:
        if raising:
            raise
    finally:
        for model in chunk:
            model._release()
        gc.collect()


def _close(states: List[Tuple[BaseModel, _RunState]], raising: bool = True) -> None:
    """Close these models' stores (releasing their locks); the first failure is raised after all, unless `raising` is False
    (the group is raising already)."""
    error: Optional[BaseException] = None
    for _, state in states:
        try:
            state.close()
        except BaseException as exc:
            error = error or exc
    if error is not None and raising:
        raise error


def _close_writer(writer: Optional[StoreWriter], raising: bool = True) -> None:
    """Close the group's writer (its thread and its lock files), raising what it raises unless `raising` is False."""
    if writer is None:
        return
    try:
        writer.close()
    except BaseException:
        if raising:
            raise


class EngineAlone:
    """A model whose `run` goes through the engine as a group of one (`run_group`), as `ModelPipeline(engine="on")` runs it:
    what `verify` and the leakage probes run when the pipeline's engine is on (decision 25 of the step 5 spec). Everything else
    is the model's (attributes are read through). A run the engine doesn't take (a `train_on="matching"` plan, `forecast_only`)
    is the model's own `run`."""

    def __init__(self, model: BaseModel):
        object.__setattr__(self, "_model", model)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_model"), name)

    def run(self, data: pd.DataFrame, *args: Any, **kwargs: Any) -> ModelResultRef:
        return run_alone(self._model, data, *args, **kwargs)


def run_alone(model: BaseModel, data: pd.DataFrame, *args: Any, **kwargs: Any) -> ModelResultRef:
    """`model.run(data, ...)` through the engine as a group of one: its ref, records and files are what a group would give it.
    A run the model can't take there (`_engine_capable` with the run's plan and `forecast_only`) is its own `run`, and so is
    a predictor function's run given no plan: the plan its run would make is what decides (a `train_on="matching"` list), and
    it isn't made until then (the probes always pass theirs)."""
    given = {**dict(zip(_RUN_PARAMETERS[1:], args)), **kwargs}  # positional ones after `data`
    plan = given.get("plan")
    if (model._ctx_fn is not None and plan is None) or not model._engine_capable(
        plan, given.get("forecast_only", False)
    ):
        return model.run(data, *args, **kwargs)
    return run_group([model], {model.name: run_arguments(data=data, **given)})[model.name]
