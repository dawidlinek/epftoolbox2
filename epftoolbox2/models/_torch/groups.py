"""Torch stacks (step 6m, plan sections 3.11 to 3.13): the networks of an `add_model` loop trained together on one device.

Speed only: every forecast is the one its model gives alone, to the bit. `ModelPipeline.run` runs the torch models of a run
that share a **stack key** (`stack_key`) as one group (`stack_groups`, `run_stack_group`), after the engine groups and before
the other models; a model whose key no other model of the run has runs alone, as every model does with `TORCH_STACKS=off`.

**Who stacks.** A `TorchModel` (the MLP presets too) whose fits are vmap stacks on its device (`stack="vmap"`, or `"auto"` on a
GPU: a loop's arithmetic is not a stack's, so a CPU's default loop runs alone), that reads its period frames and fits as
`TorchModel` does (no method of its own or of its instance in `_KEPT`), in a run that isn't `forecast_only`. The key: the
module's import path (or a session class's token key) and hash, `module_params` without `dropout` (the class's defaults
filled in), `output`, the training window(s), `refit`, the loss, batch size, epochs, patience, early-stopping share, weight
decay, the device, `tf32`, the stack size (`pad_to`) and the named blocks' sizes.

**What runs together.** A group runs like an engine group (`models/_engine.py`): chunks of `ENGINE_MAX_MODELS` models and
`ENGINE_MAX_COLUMNS` union columns, one read of the period frames per chunk, each model's `_prepare_run` on them (its store on
the group's one `StoreWriter`), one set of period arrays over the union, one pool of `TorchModel._executor_config()`'s processes
(one on a GPU). A **job** holds fit units of every model (the tasks of one with `refit=None`; a block's whole with a list of
training windows, whose task records need every window's unit), in the order block, unit, model (`_pieces`, `_jobs`), at least
`TORCH_STACK_MEMBERS` members (by default the stack size, up to four stacks, fewer where that leaves processes idle) and at
most `MAX_JOB_ITEMS` units. In a job each unit (or task) runs the worker's own function (`_run_unit`, `_run_task`) on its
model's own inputs, in a thread of its own, up to the model's fit: `TorchModel._fit_rows` hands the call to the job's session
(`stack_session`) and waits. Once every unit waits or is done, the session trains the waiting calls' members together in
**stacks** of the key's size (`_train`): members of one structure (`_structure`: the module's layers, their attributes but a
`nets.Dropout`'s rate, parameter and buffer shapes) and of the same shapes (`trainer._Shapes`: nominal rows, early-stopping
rows, steps per epoch, batch), by learning rate (a stack trains until its last member stops), each with its own rows (its
unit's arrays, gathered from the stack's blocks), seed streams, learning rate and dropout rate, padded with dummies; then the
units go on, and their predictions are served the same way, each in its stack (`_predict`). Records go to each model's store
as the job comes back.

**Why the bits hold.** A member's numbers depend on its own inputs, key, learning rate, buffers and on the stack's shapes alone
(`trainer` module docstring): on the CPU each slot's products are its own (`nets._per_slot_linear`, one thread), on a GPU one
batched product of a fixed shape computes every slot alike, so neither the other slots nor their order move a bit, and the
shapes are the member's own call's (`trainer.prepare_members`). The unit runs the worker's code, so its rows, scaling, seeds,
records and skips are those of the model's own run. A call the trainer refuses gets the error its own run would raise
(`prepare_members`'s), and a stack that raises is trained again call by call, as each model alone (`_train`'s solo calls). A
diverged member is `failed` in its own slot: its fit's forecasts are fit errors, as alone, and no other member moves.

Nothing stored names the group, its jobs or stacks; the manifest's runtime fields are the model's own (`stack`, `pad_to`).
"""

import contextvars
import copy
import datetime
import hashlib
import inspect
import json
import math
import multiprocessing
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import astuple, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rich.progress import Progress

from ..._extras import has_extra
from ...results.ref import ModelResultRef
from ...results.store import StoreWriter
from .._calendar import resolve_horizons
from .._engine import _BASE_READERS, _Bar, _close, _close_writer, _release, engine_chunks, engine_limits, mask_readers
from .._worker import (
    _POOLS_SPAWN,
    TaskFailed,
    TaskInputs,
    _limit_native_threads,
    _limit_openmp_in_task_thread,
    _run_task,
    _run_unit,
    block_records,
    block_units,
    pool_native_threads,
    task_inputs,
)
from ..base import _MASK_READERS, _WINDOWS_MAX_WORKERS, BaseModel, _MemorySink, _RunState, hour_arrays_of

__all__ = ["stack_key", "stack_groups", "run_stack_group", "run_stack_job", "stack_session", "TORCH_STACKS_VAR"]

# "off" runs every torch model on its own; "on" (the default) stacks the models that share a key.
TORCH_STACKS_VAR = "TORCH_STACKS"
STACKS_MODES = ("on", "off")
# The members a job holds at least (a whole refit block more if the last one passes it); by default `_job_members`.
STACK_MEMBERS_VAR = "TORCH_STACK_MEMBERS"
# The fit units (or tasks) of one job at most, each a thread in the worker while it runs: a bound on what a job holds at once.
MAX_JOB_ITEMS = 1024

# What a stacked model runs as `TorchModel` and `BaseModel` define it: the group does `run`'s part, and the stack replaces the
# trainer's call in `_fit_rows` and its predictions in `_members_out`.
_KEPT = (
    "run",
    "_prepare_run",
    "_execute",
    "_run_tasks",
    "_finish_run",
    "_build_hour_arrays",
    "_jobs",
    "_collect",
    "_worker_inputs",
    "_worker_model",
    "_executor_config",
    "_fit_rows",
    "_members_out",
    "_fit_days",
    "_predict_days",
    "_fit_model",
    "_predict_fitted",
    "_fit_predict",
    "_factory",
)
# A module's attributes that `_structure` doesn't compare: its parameters, buffers and submodules are compared by name, shape
# and dtype, and a `nets.Dropout`'s `p` is its `rate` buffer's value, which each slot holds for itself.
_MODULE_INTERNALS = ("_parameters", "_buffers", "_modules")


# --- which models stack ------------------------------------------------------------------------------------------------


def stacks_mode() -> str:
    """`TORCH_STACKS`: "on" (the default) or "off", read when a run runs."""
    raw = os.environ.get(TORCH_STACKS_VAR, "on").strip().lower()
    if raw not in STACKS_MODES:
        raise ValueError(f"{TORCH_STACKS_VAR} must be one of {list(STACKS_MODES)}, got {raw!r}")
    return raw


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def stack_key(model: BaseModel, forecast_only: bool = False) -> Optional[tuple]:
    """The key of the stacks `model` trains in with other models of a run (module docstring), or None for a model that runs
    alone: not a torch model, a loop on its device, a method in `_KEPT` of its own class or instance, frames it reads itself,
    a `forecast_only` run, or a device that doesn't resolve (its own run says why)."""
    from ..torch_model import TorchModel

    if forecast_only or not isinstance(model, TorchModel) or not has_extra("torch"):
        return None
    if model._frame_readers() != _BASE_READERS:
        return None
    if any(type(model)._owner_of(name) not in (TorchModel, BaseModel) or name in vars(model) for name in _KEPT):
        return None
    from .runtime import resolve_device, resolve_pad, resolve_stack

    try:
        device = resolve_device(model.device)
    except (RuntimeError, ValueError):
        return None
    if resolve_stack(model.stack, device) != "vmap":
        return None
    return (
        model._module_ref(),
        model._module_hash(),
        _canonical(_module_params(model)),
        model.output,
        _canonical(model.training_window),
        model.refit,
        model.loss,
        model.batch_size,
        model.max_epochs,
        model.patience,
        model.early_stopping,
        model.weight_decay,
        str(device),
        bool(model.tf32),
        resolve_pad("vmap", model.pad_to),
        _canonical(model._block_sizes()),
    )


def _module_params(model: Any) -> Dict[str, Any]:
    """The module's keyword arguments as it is built, its defaults filled in (`MLPDailyModel`'s `activation="relu"` is
    `TorchModel(module=MLP)`'s default), without `dropout`, which members of one stack may differ in."""
    params = dict(model._module_spec()[1])
    try:
        signature = inspect.signature(model._module_class())
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        for name, parameter in list(signature.parameters.items())[2:]:
            if parameter.default is not parameter.empty and parameter.kind in (
                parameter.POSITIONAL_OR_KEYWORD,
                parameter.KEYWORD_ONLY,
            ):
                params.setdefault(name, parameter.default)
    params.pop("dropout", None)
    return params


def stack_groups(models: Sequence[BaseModel], forecast_only: bool = False) -> List[List[BaseModel]]:
    """The groups of these models that train together: the models of one stack key, in list order, for each key at least two
    models have, in the order of each group's first model. None with `TORCH_STACKS=off`."""
    if stacks_mode() == "off":
        return []
    groups: Dict[tuple, List[BaseModel]] = {}
    for model in models:
        key = stack_key(model, forecast_only)
        if key is not None:
            groups.setdefault(key, []).append(model)
    return [group for group in groups.values() if len(group) > 1]


# --- the parent: a group's chunks, jobs and records ----------------------------------------------------------------------


def run_stack_group(
    models: Sequence[BaseModel], arguments: Dict[str, Dict[str, Any]], columns: Dict[str, Sequence[str]]
) -> Dict[str, ModelResultRef]:
    """Run one group of `stack_groups`, each model with its `run` arguments (`arguments[name]`, `_engine.run_arguments`; one
    frame, target, horizons and freq for all) and its predictors (`columns[name]`), and return each model's ref by name. As
    `_engine.run_group`: called where a model's `run` would be, so `_prepare_run`'s and `_finish_run`'s warnings point where
    theirs do; each chunk finished (its stores flushed, each model's `_finish_run`, its stores closed) before the next starts; a
    task that raises stops the group, with what came back saved, and a RuntimeError naming the model and its blocks."""
    if not models:
        return {}
    first = arguments[models[0].name]
    data, target, freq = first["data"], first["target"], first["freq"]
    horizon, _ = resolve_horizons(first["horizon"], first["horizons"], 7)
    chunks = engine_chunks(models, columns, *engine_limits())
    writer = StoreWriter() if any(arguments[model.name]["save_to"] for model in models) else None
    refs: Dict[str, ModelResultRef] = {}
    try:
        with Progress() as progress:
            bar = _Bar(progress, progress.add_task(f"[cyan]torch stacks: {len(models)} models", total=0))
            for chunk in chunks:
                prepared: List[Tuple[BaseModel, _RunState]] = []
                try:
                    union = dict.fromkeys(name for model in chunk for name in columns[model.name])
                    chunk[0]._set_freq(freq)
                    periods = chunk[0]._load_periods(data, list(dict.fromkeys([target, *union])), horizon, target)
                    periods.memo[_MASK_READERS] = mask_readers(chunk, columns)
                    for model in chunk:
                        state = model._prepare_run(**arguments.pop(model.name), periods=periods, writer=writer)
                        prepared.append((model, state))
                    periods = None  # the models hold the frames until `_release`
                    _run_chunk(prepared, bar)
                except BaseException:
                    _release(prepared, writer, chunk, raising=False)
                    _close(prepared, raising=False)
                    raise
                _release(prepared, writer, chunk)
                error: Optional[BaseException] = None
                for model, state in prepared:
                    try:
                        refs[model.name] = model._finish_run(state)
                    except Exception as exc:  # every model's counts and warnings, then the first failure
                        error = error or exc
                _close(prepared, raising=error is None)
                prepared.clear()
                state = None
                if error is not None:
                    raise error
    except BaseException:
        _close_writer(writer, raising=False)
        raise
    _close_writer(writer)
    return refs


def _pieces(config: Dict, tasks: List[tuple], seeds: int) -> List[Tuple[tuple, List[tuple], int, int]]:
    """A model's tasks of one block (or issue day) as the parts a job may hold apart: (rank, tasks, members, items) each. A fit
    unit's tasks with `refit` (with a list of training windows the block's whole: a task's record needs every window's unit in
    its job), each task with `refit=None` (its windows fit inside it); members are the fits' seeds, the rank `_rank`'s, so a
    job holds a unit of every model before the next unit."""
    if config.get("refit") is None:
        windows = len(config.get("members") or [None])
        return [(_rank(task, None), [task], windows * seeds, 1) for task in tasks]
    units = block_units(config, tasks)
    if config.get("members") is not None:
        first = next(iter(units))
        return [(_rank(first, []), list(tasks), len(units) * seeds, len(units))]
    return [(_rank(unit, unit_tasks), unit_tasks, seeds, 1) for unit, unit_tasks in units.items()]


def _job_members(total: int, pad: int, processes: int) -> int:
    """The members a job holds at least: `TORCH_STACK_MEMBERS`, else one stack, up to four where the run has enough of them
    for every process."""
    raw = os.environ.get(STACK_MEMBERS_VAR)
    if raw is not None:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value < 1:
            raise ValueError(f"{STACK_MEMBERS_VAR} must be a whole number from 1, got {raw!r}")
        return value
    return max(pad, min(4 * pad, math.ceil(total / max(1, processes))))


def _jobs(
    pieces: List[Tuple[Any, tuple, int, List[tuple], int, int]], target: int
) -> List[Tuple[Any, Any, Dict[int, List[tuple]]]]:
    """The pieces (block, rank, model's place, tasks, members, items), in that order, cut into jobs of at least `target`
    members and at most `MAX_JOB_ITEMS` items (fit units, or tasks with `refit=None`), a piece more than that a job of its own:
    (first block, last block, {model's place: its tasks}) each."""
    jobs: List[Tuple[Any, Any, Dict[int, List[tuple]]]] = []
    current: Dict[int, List[tuple]] = {}
    members = items = 0
    start = last = None
    for key, _, place, tasks, weight, size in pieces:
        if current and items + size > MAX_JOB_ITEMS:
            jobs.append((start, last, current))
            current, members, items = {}, 0, 0
        if not current:
            start = key
        current.setdefault(place, []).extend(tasks)
        members += weight
        items += size
        last = key
        if members >= target:
            jobs.append((start, last, current))
            current, members, items = {}, 0, 0
    if current:
        jobs.append((start, last, current))
    return jobs


def _run_chunk(prepared: List[Tuple[BaseModel, _RunState]], bar: _Bar) -> None:
    """One chunk's pool: the union arrays, each model's worker inputs (`_worker_inputs`: its torch threads, its module built
    and checked once here), the jobs, and each job's records into each model's sink as it comes back (`_collect`)."""
    pending = [(model, state) for model, state in prepared if state.tasks]
    bar.add(
        sum(len(state.all_tasks) for _, state in prepared),
        sum(len(state.all_tasks) - len(state.tasks) for _, state in prepared),
    )
    if not pending:
        return
    first, state0 = prepared[0]
    union = list(dict.fromkeys(name for model, _ in pending for name in model._used_predictors))
    hour_arrays = hour_arrays_of(
        first._hour_data, first._hour_days, first._target, max(state0.horizons), union, first._periods_per_day
    )
    threads, n_processes = first._executor_config()
    from .runtime import resolve_pad

    pad = resolve_pad("vmap", first.pad_to)
    inputs = []
    pieces: List[Tuple[Any, tuple, int, List[tuple], int, int]] = []
    for place, (model, state) in enumerate(pending):
        expanded, config = model._worker_inputs(state, threads)
        inputs.append((model.name, expanded, config, *model._worker_model()))
        for key, tasks in model._jobs(state.tasks, state.first_day).items():
            pieces.extend((key, rank, place, *rest) for rank, *rest in _pieces(config, tasks, model.seeds))
    pieces.sort(key=lambda piece: piece[:3])  # by block, then unit, then model
    jobs = _jobs(pieces, _job_members(sum(piece[4] for piece in pieces), pad, n_processes))
    pieces = None
    sinks = [state.store if state.store else _MemorySink(state.in_memory) for _, state in pending]
    n_processes = min(n_processes, len(jobs))
    if sys.platform == "win32":
        n_processes = min(n_processes, _WINDOWS_MAX_WORKERS)
    refit = first._refit_block is not None

    mp_context = multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn")
    with (
        pool_native_threads(),  # the workers inherit one OpenBLAS thread, so they start none
        ProcessPoolExecutor(
            max_workers=n_processes, mp_context=mp_context, initializer=_stack_init, initargs=(hour_arrays, inputs)
        ) as pool,
    ):
        try:
            futures = {pool.submit(run_stack_job, list(job.items())): index for index, (_, _, job) in enumerate(jobs)}
            for future in as_completed(futures):
                index = futures.pop(future)  # its records leave with it: the parent holds one job's at a time
                start, last, job = jobs[index]
                jobs[index] = None
                try:
                    outcomes = future.result()
                except Exception as exc:
                    raise _job_failed(exc, start, last, job, pending, refit) from exc
                for place, results, skipped in outcomes:
                    model, state = pending[place]
                    model._collect(state, results, skipped, sinks[place])
                bar.advance(sum(len(tasks) for tasks in job.values()))
        except BaseException:
            # As `_engine._run_chunk`: whatever stops the loop, no job not yet started runs; what came back is saved.
            pool.shutdown(wait=True, cancel_futures=True)
            raise


def _job_failed(
    exc: BaseException,
    start: Any,
    last: Any,
    job: Dict[int, List[tuple]],
    pending: List[Tuple[BaseModel, _RunState]],
    refit: bool,
) -> RuntimeError:
    """The error of a job that raised: the failing model's name (or the job's models', for a pool that broke) and its blocks."""
    if refit:
        what = f"the refit block from {start}" if start == last else f"the refit blocks from {start} to {last}"
    else:
        day = pending[next(iter(job))][1].first_day
        first, end = day + datetime.timedelta(days=start), day + datetime.timedelta(days=last)
        what = f"run date {first}" if start == last else f"run dates {first} to {end}"
    if isinstance(exc, StackTaskFailed):
        return RuntimeError(f"{pending[exc.model][0].name}: forecasting failed for {what}: {exc}")
    names = ", ".join(pending[place][0].name for place in job)
    return RuntimeError(f"forecasting failed for {what} (models {names}): {exc}")


# --- a worker: a job's units, their calls and the stacks ------------------------------------------------------------------

# A pool worker's models: (name, `TaskInputs`) by place, on the chunk's one set of arrays (`_stack_init`), and their modules'
# structures by (place, inputs, outputs) (`_structure_of`).
_S_MODELS: List[Tuple[str, TaskInputs]] = []
_S_STRUCTURES: Dict[tuple, tuple] = {}


class StackTaskFailed(TaskFailed):
    """A unit (or task) of one model of a stack job raised: `model` is its place in the chunk, `message` the worker's
    `TaskFailed` text (the unit and the traceback)."""

    def __init__(self, model: int, message: str):
        super().__init__(model, message)
        self.model = model
        self.message = message

    def __str__(self) -> str:
        return self.message


def _stack_init(hour_arrays: Dict, models: Sequence[tuple]) -> None:
    """Initialiser of a stack pool: torch pinned to one thread around the native-thread limits, as `_worker._pool_init` does
    for a torch model's pool, then every model's inputs."""
    from ..torch_model import THREADS_PER_FIT
    from .runtime import pin_torch_threads

    pin_torch_threads(THREADS_PER_FIT, forked=not _POOLS_SPAWN)
    _limit_native_threads()
    pin_torch_threads(THREADS_PER_FIT, forked=not _POOLS_SPAWN)
    _stack_load(hour_arrays, models)


def _stack_load(hour_arrays: Dict, models: Sequence[tuple]) -> None:
    """Load a chunk's models into this process: each rebuilt as a pool's worker rebuilds it (`_worker.task_inputs`)."""
    global _S_MODELS
    _S_STRUCTURES.clear()
    _S_MODELS = [
        (name, task_inputs(hour_arrays, expanded, config, model_cls, model_kwargs))
        for name, expanded, config, model_cls, model_kwargs in models
    ]


def run_stack_job(
    job: Sequence[Tuple[int, List[tuple]]], models: Optional[Sequence[Tuple[str, TaskInputs]]] = None
) -> List[Tuple[int, List[Dict], List[Tuple[tuple, str]]]]:
    """Run one stack job: (model's place in the chunk, its tasks of the job's blocks) for each model with some, on `models` (the
    pool's if None). Returns (place, records, [(task, skip reason)]) per model, as `_run_block` (or `_run_day`) returns them for
    that model's tasks. A unit that raised raises `StackTaskFailed` naming its model, once every unit of the job ran."""
    models = _S_MODELS if models is None else models
    items: List[Tuple[int, Any, Optional[List[tuple]]]] = []  # (place, unit or task, the unit's tasks or None)
    blocks: Dict[int, bool] = {}
    for place, tasks in job:
        cfg = models[place][1].config
        blocks[place] = cfg.get("refit") is not None
        if blocks[place]:
            items.extend((place, unit, unit_tasks) for unit, unit_tasks in block_units(cfg, tasks).items())
        else:
            items.extend((place, task, None) for task in tasks)
    ran = _Session(models).run(items)
    for (place, _, _), (ok, outcome) in zip(items, ran):
        if not ok:
            if isinstance(outcome, TaskFailed):
                raise StackTaskFailed(place, str(outcome)) from None
            raise outcome
    done: Dict[int, list] = {place: [] for place, _ in job}
    for (place, what, _), (_, outcome) in zip(items, ran):
        done[place].append((what, outcome))
    out = []
    for place, outcomes in done.items():
        if blocks[place]:
            results, skipped = block_records(models[place][1].config, [outcome for _, outcome in outcomes])
        else:
            results = [outcome for _, outcome in outcomes if not isinstance(outcome, str)]
            skipped = [(task, outcome) for task, outcome in outcomes if isinstance(outcome, str)]
        out.append((place, results, skipped))
    return out


# The session of the unit running in this thread (`_Ticket`), None outside a stack job: what `TorchModel._fit_rows` asks.
_SESSION: contextvars.ContextVar = contextvars.ContextVar("epftoolbox2_torch_stack", default=None)


def stack_session() -> Optional["_Ticket"]:
    """The stack job's session of the unit running in this thread (`_Ticket.fit`), None outside a stack job."""
    return _SESSION.get()


class _Aborted(BaseException):
    """What a unit's call gets when its job's session failed: a BaseException, so the worker's handlers of a fit's errors
    (`except ValueError`, `except Exception`) let it through to the session, which raises the failure itself."""


_FIT, _PREDICT = "fit", "predict"


class _Request:
    """A unit's call to its session, waiting in its thread until the session served it: a fit (`payload` the call) or a
    prediction (`payload` (the fit's `_StackedResult`, the rows)); `order` sorts a round's fits (`_Ticket.order`)."""

    __slots__ = ("kind", "order", "payload", "result", "error", "done", "solo")

    def __init__(self, kind: str, order: tuple, payload: Any):
        self.kind = kind
        self.order = order
        self.payload = payload
        self.result: Any = None
        self.error: Optional[BaseException] = None
        self.done = False
        self.solo = False


class _Ticket:
    """One unit's way to its session, set in its thread's context: its model's place, the item, its rank and its calls so far.
    A round's fits are stacked in the order of (rank, place, item, call): the same unit (anchor, horizon, period, window) of
    every model one after another, so a stack holds the models' fits of one unit, whose rows (and with one list, values) are
    shared. Any order gives the same bits; this one shares the most."""

    def __init__(self, session: "_Session", place: int, item: int, rank: tuple):
        self.session = session
        self.place = place
        self.item = item
        self.rank = rank
        self.calls = 0

    @property
    def order(self) -> tuple:
        return (self.rank, self.place, self.item, self.calls)

    def fit(self, model, factory, x, y, y_mask, members, settings) -> Tuple["_StackedResult", Any]:
        """(the call's result, its device) once the session trained its members: what `train_members` would have returned
        alone, its predictions served by the session."""
        self.calls += 1
        payload = dict(
            place=self.place,
            model=model,
            factory=factory,
            x=x,
            y=y,
            y_mask=y_mask,
            members=list(members),
            settings=settings,
        )
        return self.session.ask(_Request(_FIT, self.order, payload))


class _StackedResult:
    """A call's members in their stacks (`slots`: (stack, slot) per member), or a call trained alone (`solo`, the trainer's
    `TrainResult`): what `TorchModel` reads of a `TrainResult` (`predict`, `failed`, `best_epoch`, `stats`), its predictions
    asked of the session."""

    stacked = True

    def __init__(self, session, slots, failed, best_epoch, stats, n_inputs, n_outputs, device, dtype, tf32, solo=None):
        self.session = session
        self.slots = slots
        self.failed = failed
        self.best_epoch = best_epoch
        self.stats = stats
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.device = device
        self.dtype = dtype
        self.tf32 = tf32
        self.solo = solo

    @property
    def n_members(self) -> int:
        return len(self.failed)

    def predict(self, rows: np.ndarray) -> np.ndarray:
        ticket = stack_session()
        return self.session.ask(_Request(_PREDICT, () if ticket is None else ticket.order, (self, rows)))


class _Session:
    """A stack job in one worker: each item (a fit unit, or a task with `refit=None`) runs in a thread of its own until it asks
    for a fit or a prediction (`ask`); once every running item waits, this thread (the worker's own, the only one that runs
    torch) serves the round (`_serve`) and lets them go on, until every item is done."""

    def __init__(self, models: Sequence[Tuple[str, TaskInputs]]):
        self.models = models
        self.lock = threading.Condition()
        self.running = 0
        self.blocked = 0
        self.queue: List[_Request] = []
        self.failure: Optional[BaseException] = None

    def ask(self, request: _Request) -> Any:
        """In an item's thread: wait until the session served `request`, then its result (or raise its error)."""
        with self.lock:
            if self.failure is not None:
                raise _Aborted()
            self.queue.append(request)
            self.blocked += 1
            self.lock.notify_all()
            while not request.done:
                self.lock.wait()
        if request.error is not None:
            raise request.error
        return request.result

    def run(self, items: List[Tuple[int, Any, Optional[List[tuple]]]]) -> List[Tuple[bool, Any]]:
        """(True, the item's outcome) or (False, what it raised), per item in order; a failure of the session itself raised."""
        outcomes: List[Tuple[bool, Any]] = [(False, None)] * len(items)

        def body(index: int) -> None:
            place, what, unit_tasks = items[index]
            inputs = self.models[place][1]
            _SESSION.set(_Ticket(self, place, index, _rank(what, unit_tasks)))
            try:
                _limit_openmp_in_task_thread()
                if unit_tasks is None:
                    outcomes[index] = (True, _run_task(what, inputs))
                else:
                    outcomes[index] = (True, _run_unit(what, unit_tasks, inputs))
            except BaseException as exc:  # a TaskFailed of the worker's, or `_Aborted`
                outcomes[index] = (False, exc)
            finally:
                with self.lock:
                    self.running -= 1
                    self.lock.notify_all()

        threads = [threading.Thread(target=body, args=(index,), daemon=True) for index in range(len(items))]
        with self.lock:
            self.running = len(threads)
        for thread in threads:
            thread.start()
        try:
            while True:
                with self.lock:
                    while self.running > self.blocked:
                        self.lock.wait()
                    if self.running == 0:
                        break
                    batch, self.queue = self.queue, []
                try:
                    if self.failure is None:
                        self._serve(batch)
                except BaseException as exc:
                    self.failure = exc
                self._release(batch)
        except BaseException as exc:  # an interrupt while waiting: no item may wait for a round that won't come
            self.failure = self.failure or exc
            raise
        finally:
            with self.lock:
                waiting, self.queue = self.queue, []
            self._release(waiting)
            for thread in threads:
                thread.join()
        if self.failure is not None:
            raise self.failure
        return outcomes

    def _release(self, batch: List[_Request]) -> None:
        """Let the items of `batch` go on; once the session failed, each with `_Aborted` (its job raises the failure)."""
        with self.lock:
            for request in batch:
                if self.failure is not None:
                    request.result, request.error = None, _Aborted()
                request.done = True
            self.blocked -= len(batch)
            self.lock.notify_all()

    def _serve(self, batch: List[_Request]) -> None:
        fits = sorted((r for r in batch if r.kind == _FIT), key=lambda r: r.order)
        predicts = [r for r in batch if r.kind == _PREDICT]
        if fits:
            _train(self, fits)
        if predicts:
            _predict(predicts)


def _rank(what: Any, unit_tasks: Optional[List[tuple]]) -> tuple:
    """Where a unit's (or a task's) fits go in a round's stacks (`_Ticket`): by anchor, horizon, period, window and list for a
    unit, by issue day, horizon and period for a task."""
    if unit_tasks is None:
        period, horizon, day = what
        return (day, horizon, period)
    return (
        what.anchor.toordinal(),
        what.horizon,
        -1 if what.period is None else what.period,
        what.member,
        -1 if what.list_id is None else what.list_id,
    )


def _error(exc: BaseException) -> BaseException:
    """A copy of `exc` for one unit's thread to raise (several units can get the same stack's error)."""
    try:
        return copy.copy(exc)
    except Exception:
        return exc


def _describe(value: Any) -> Any:
    """A module attribute's value as `_structure` compares it."""
    import torch

    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous().numpy().tobytes()
        return ("tensor", tuple(value.shape), str(value.dtype), hashlib.blake2b(data, digest_size=16).hexdigest())
    if isinstance(value, dict):
        return ("dict", tuple(sorted((repr(k), _describe(v)) for k, v in value.items())))
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted(repr(v) for v in value)))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_describe(v) for v in value))
    return repr(value)


def _structure(module: Any) -> tuple:
    """What members must share to train in one stack: every layer's name, class and attributes (but its parameters, buffers and
    submodules, and a `nets.Dropout`'s `p`, which its `rate` buffer holds per slot), and every parameter's and buffer's name,
    shape, dtype and whether it trains. vmap runs one module's code for every slot, the first member's: anything of a layer
    but its tensors would be the first member's for every slot."""
    from ... import nets

    out: List[Any] = []
    for name, sub in module.named_modules():
        state = []
        for attr, value in sorted(vars(sub).items()):
            if attr in _MODULE_INTERNALS or (isinstance(sub, nets.Dropout) and attr == "p"):
                continue
            state.append((attr, _describe(value)))
        out.append(("module", name, type(sub).__module__, type(sub).__qualname__, tuple(state)))
    for name, param in module.named_parameters():
        out.append(("parameter", name, tuple(param.shape), str(param.dtype), bool(param.requires_grad)))
    for name, buffer in module.named_buffers():
        out.append(("buffer", name, tuple(buffer.shape), str(buffer.dtype)))
    return tuple(out)


def _structure_of(place: int, factory: Any, n_inputs: int, n_outputs: int) -> tuple:
    """`_structure` of a model's module for this many inputs and outputs, built once per chunk in this process on the CPU (a
    generator of its own, one thread), as `TorchModel._check_buildable` builds it."""
    key = (place, n_inputs, n_outputs)
    found = _S_STRUCTURES.get(key)
    if found is None:
        import torch

        from .runtime import torch_scope

        with torch_scope("cpu", tf32=False, threads=1), torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            found = _S_STRUCTURES[key] = _structure(factory())
    return found


def _digest(prepared: Any) -> bytes:
    """A call's arrays, as a key: calls with the same values share their block of a stack's rows."""
    h = hashlib.blake2b(digest_size=16)
    for array in (prepared.x, prepared.y, prepared.mask):
        h.update(repr((array.shape, str(array.dtype))).encode())
        h.update(np.ascontiguousarray(array).tobytes())
    return h.digest()


def _settings_key(settings: Dict[str, Any]) -> tuple:
    """The settings of a call every member of a stack shares: all but the learning rate (each member's own) and those the
    stack's key and shapes hold already (stack, size, nominal rows)."""
    blocks = settings.get("blocks")
    return (
        settings["loss"],
        settings["max_epochs"],
        settings["patience"],
        settings["batch_size"],
        settings["early_stopping"],
        settings["weight_decay"],
        settings["dtype"],
        None if blocks is None else tuple((name, sl.start, sl.stop) for name, sl in blocks.items()),
    )


def _train(session: _Session, requests: List[_Request]) -> None:
    """Train a round's fit requests: each call planned as its own `train_members` would plan it (`prepare_members`; a refusal
    is its error), its members in stacks of the calls of one structure, shapes and settings, in order, padded with dummies;
    a call whose stack raised, or that couldn't be built, trained alone (`train_members`, its own errors)."""
    import torch

    from ..torch_model import DTYPE, THREADS_PER_FIT
    from . import trainer
    from .runtime import torch_scope

    keyed: Dict[tuple, List[Tuple[_Request, Any]]] = {}
    for request in requests:
        call = request.payload
        model = call["model"]
        try:
            prepared = trainer.prepare_members(
                call["x"], call["y"], call["y_mask"], call["members"], device=model.device, **call["settings"]
            )
        except Exception as exc:  # what the call's own `train_members` would raise
            request.error = exc
            continue
        call["prepared"] = prepared
        if prepared.kind != "vmap" or call["settings"]["dtype"] != DTYPE:
            request.solo = True
            continue
        try:
            structure = _structure_of(call["place"], call["factory"], prepared.x.shape[1], prepared.y.shape[1])
        except Exception:
            request.solo = True  # its own fit raises what building it raises
            continue
        key = (
            structure,
            astuple(prepared.shapes),
            prepared.pad,
            str(prepared.device),
            bool(model.tf32),
            _settings_key(call["settings"]),
        )
        keyed.setdefault(key, []).append((request, prepared))

    for key, entries in keyed.items():
        pad, tf32 = key[2], key[4]
        # By learning rate first: a stack trains until its last member stops, and members of one rate tend to stop together.
        entries.sort(key=lambda entry: (float(entry[0].payload["settings"]["lr"]), entry[0].order))
        slots = [(request, prepared, i) for request, prepared in entries for i in range(len(prepared.members))]
        for request, prepared in entries:
            n = len(prepared.members)
            request.result = dict(
                slots=[None] * n,
                failed=np.zeros(n, dtype=bool),
                best_epoch=np.zeros(n, dtype=np.int64),
                epochs=np.zeros(n, dtype=np.int64),
                best_val=np.full(n, math.nan),
                seconds=0.0,
                stacks=set(),
            )
        for start in range(0, len(slots), pad):
            chunk = [slot for slot in slots[start : start + pad] if not slot[0].solo]
            if not chunk:
                continue
            try:
                fit, seconds = _train_stack(chunk, pad, tf32, torch, trainer, torch_scope, THREADS_PER_FIT)
            except Exception:
                for request, _, _ in chunk:  # trained alone below: its own errors, if it has any, are its own
                    request.solo = True
                continue
            for j, (request, _, i) in enumerate(chunk):
                part = request.result
                part["slots"][i] = (fit, j)
                part["failed"][i] = bool(fit.failed[j])
                part["best_epoch"][i] = int(fit.best_epoch[j])
                part["epochs"][i] = int(fit.epochs[j])
                part["best_val"][i] = float(fit.best_val[j])
                if id(fit) not in part["stacks"]:
                    part["stacks"].add(id(fit))
                    part["seconds"] += seconds

    for request in requests:
        if request.error is not None:
            continue
        call = request.payload
        model, prepared = call["model"], call["prepared"]
        dtype = prepared.settings["dtype"]
        if request.solo:
            try:
                with torch_scope(model.device, tf32=model.tf32, threads=THREADS_PER_FIT) as device:
                    solo = trainer.train_members(
                        call["factory"],
                        call["x"],
                        call["y"],
                        call["y_mask"],
                        call["members"],
                        device=device,
                        **call["settings"],
                    )
            except Exception as exc:
                request.error = exc
                request.result = None
                continue
            stats = solo.stats
            result = _StackedResult(
                session,
                None,
                solo.failed,
                solo.best_epoch,
                stats,
                prepared.x.shape[1],
                prepared.y.shape[1],
                device,
                dtype,
                bool(model.tf32),
                solo=solo,
            )
        else:
            part = request.result
            shapes = prepared.shapes
            stats = {
                "epochs": part["epochs"],
                "best_val": part["best_val"],
                "stack": "vmap",
                "pad_to": prepared.pad,
                "stacks": len(part["stacks"]),
                "steps_per_epoch": shapes.steps,
                "val_rows": shapes.val_rows,
                "nominal_rows": shapes.nominal,
                "train_seconds": part["seconds"],
            }
            device = prepared.device
            result = _StackedResult(
                session,
                part["slots"],
                part["failed"],
                part["best_epoch"],
                stats,
                prepared.x.shape[1],
                prepared.y.shape[1],
                device,
                dtype,
                bool(model.tf32),
            )
        request.result = (result, device)
        call.pop("prepared", None)


def _train_stack(chunk, pad, tf32, torch, trainer, torch_scope, threads):
    """One stack of `chunk`'s members, (request, prepared call, member) each, on the rows of their calls' arrays (each set of
    values once): the trained `_StackFit` and its seconds."""
    offsets: Dict[int, int] = {}
    by_digest: Dict[bytes, int] = {}
    xs, ys, masks = [], [], []
    total = 0
    for request, prepared, _ in chunk:
        if id(request) in offsets:
            continue
        digest = request.payload.get("digest")
        if digest is None:
            digest = request.payload["digest"] = _digest(prepared)
        if digest in by_digest:
            offsets[id(request)] = by_digest[digest]
            continue
        offsets[id(request)] = by_digest[digest] = total
        xs.append(prepared.x)
        ys.append(prepared.y)
        masks.append(prepared.mask)
        total += len(prepared.x)
    members = []
    for request, prepared, i in chunk:
        member, offset = prepared.members[i], offsets[id(request)]
        members.append(replace(member, fit=member.fit + offset, val=member.val + offset))
    factories = [request.payload["factory"] for request, _, _ in chunk]
    first = chunk[0][1]
    settings, shapes = first.settings, first.shapes
    started = time.perf_counter()
    with torch_scope(first.device, tf32=tf32, threads=threads) as device:
        data = trainer.tensors(np.concatenate(xs), np.concatenate(ys), np.concatenate(masks), device, settings["dtype"])
        with torch.enable_grad(), trainer._stack_threads(device):
            fit = trainer._StackFit(factories, members, pad, data, shapes, settings)
        trainer._sync(device)
    return fit, time.perf_counter() - started


def _predict(requests: List[_Request]) -> None:
    """Serve a round's predictions: a call's members each from its own slot, on its own rows, every stack once per number of
    rows with each slot's member's rows (a slot no call asks for gets a copy, discarded), in the scope of its fit; a call
    trained alone by its own `TrainResult.predict`."""
    import torch

    from ..torch_model import THREADS_PER_FIT
    from .runtime import torch_scope

    stacks: Dict[Tuple[int, int], List[Any]] = {}
    for request in requests:
        result, rows = request.payload
        rows = np.asarray(rows, dtype=np.float64)
        n, k = result.n_members, result.n_inputs
        if rows.ndim == 2:
            rows = np.broadcast_to(rows, (n, *rows.shape))
        if rows.ndim != 3 or rows.shape[0] != n or rows.shape[2] != k:
            request.error = ValueError(f"predict takes (rows, {k}) or ({n}, rows, {k}), got {rows.shape}")
            continue
        if result.solo is not None:
            try:
                with torch_scope(result.device, tf32=result.tf32, threads=THREADS_PER_FIT):
                    request.result = result.solo.predict(rows)
            except Exception as exc:
                request.error = exc
            continue
        out = np.empty((n, rows.shape[1], result.n_outputs), dtype=np.float64)
        request.result = out
        for i, (fit, j) in enumerate(result.slots):
            entry = stacks.setdefault((id(fit), rows.shape[1]), [fit, result, []])
            entry[2].append((request, out, i, j, rows[i]))
    for fit, result, assigned in stacks.values():
        taken = [j for _, _, _, j, _ in assigned]
        if len(set(taken)) != len(taken):
            raise RuntimeError("a stack's slot was asked for twice in one round")
        block = np.empty((fit.slots, *assigned[0][4].shape), dtype=np.float64)
        block[:] = assigned[0][4]
        for _, _, _, j, rows in assigned:
            block[j] = rows
        try:
            with torch_scope(result.device, tf32=result.tf32, threads=THREADS_PER_FIT) as device:
                values = fit.run_slots(torch.as_tensor(block, dtype=result.dtype, device=device))
        except Exception as exc:
            for request, _, _, _, _ in assigned:
                request.error, request.result = _error(exc), None
            continue
        for _, out, i, j, _ in assigned:
            out[i] = values[j]
    for request in requests:
        if request.error is None and request.result is not None:
            result = request.payload[0]
            if result.solo is None:
                request.result[result.failed] = np.nan
