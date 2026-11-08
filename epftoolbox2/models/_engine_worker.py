"""The worker side of an engine group (`models/_engine.py`): one pool for every model of a chunk, each job one (day or refit
block, period) holding the pending tasks of every model of the chunk on that day or block and period.

Each task runs on its model's **own** inputs (`_worker.TaskInputs`: the chunk's union arrays, which a fit reads by column name,
and the model's expanded lists, config and rebuilt model), through the model path's functions: `_run_task` for a refit=None
task, `_run_unit` for each unit of a refit block (`_worker.block_units`), whose records `_worker.block_records` makes as
`_run_block` does. So a model's records are what its own `run` gives, to the bit, whatever else is in the job: step 5h runs no
fit from shared statistics (`_fit_stats` is 5i's).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from ._worker import (
    TaskFailed,
    TaskInputs,
    _limit_native_threads,
    _limit_openmp_in_task_thread,
    _run_task,
    _run_unit,
    block_records,
    block_units,
    task_inputs,
)

# What `_engine_init` loaded into a pool's worker: the chunk's models' inputs, by their place in the chunk.
_E_MODELS: List["ModelInputs"] = []


class ModelInputs(NamedTuple):
    """One model of a chunk in a worker: its name (for errors) and its `TaskInputs`."""

    name: str
    inputs: TaskInputs


class EngineTaskFailed(TaskFailed):
    """A task of one model of an engine job raised: `model` is the model's place in the chunk, `message` the model path's
    `TaskFailed` text (the task and the worker's traceback)."""

    def __init__(self, model: int, message: str):
        super().__init__(model, message)
        self.model = model
        self.message = message

    def __str__(self) -> str:
        return self.message


def model_inputs(hour_arrays: Dict, models: Sequence[tuple]) -> List[ModelInputs]:
    """Each model's inputs from (name, expanded lists, config, class, `_model_kwargs`), the model rebuilt as a pool's worker
    rebuilds it (`_worker.task_inputs`), all on the chunk's one set of arrays."""
    return [
        ModelInputs(name, task_inputs(hour_arrays, expanded, config, model_cls, model_kwargs))
        for name, expanded, config, model_cls, model_kwargs in models
    ]


def _engine_init(hour_arrays: Dict, models: Sequence[tuple]) -> None:
    """Initialiser of an engine pool: native threads limited as `_worker._pool_init` limits them, then every model's inputs."""
    _limit_native_threads()
    _engine_load(hour_arrays, models)


def _engine_load(hour_arrays: Dict, models: Sequence[tuple]) -> None:
    """Load a chunk's models' inputs into this process (`_engine_init` without the thread limit, for a caller's process)."""
    global _E_MODELS
    _E_MODELS = model_inputs(hour_arrays, models)


Outcome = Tuple[int, List[Dict], List[Tuple[tuple, str]]]


def run_job(job: Sequence[Tuple[int, List[tuple]]], models: Optional[Sequence[ModelInputs]] = None) -> List[Outcome]:
    """Run one engine job: (model's place in the chunk, its tasks of the job's day or block and period) for each model with
    some, on `models` (the pool's if None). Returns (place, records, [(task, skip reason)]) per model, as `_run_day` or
    `_run_block` returns them for that model's tasks. With `THREADS_PER_PROCESS` above 1, the job's tasks (refit=None) or units
    (refit) of every model run on that many threads. A task that raises raises `EngineTaskFailed` naming its model."""
    models = _E_MODELS if models is None else models
    items: List[Tuple[int, tuple, Optional[List[tuple]]]] = []  # (place, task or unit, the unit's tasks or None)
    blocks: Dict[int, bool] = {}
    for place, tasks in job:
        cfg = models[place].inputs.config
        blocks[place] = cfg.get("refit") is not None
        if blocks[place]:
            items.extend((place, unit, unit_tasks) for unit, unit_tasks in block_units(cfg, tasks).items())
        else:
            items.extend((place, task, None) for task in tasks)
    threads = models[job[0][0]].inputs.config["threads_per_process"] if job else 1

    def run(item: Tuple[int, tuple, Optional[List[tuple]]]) -> Tuple[int, tuple, Union[Dict, str, list]]:
        place, what, unit_tasks = item
        inputs = models[place].inputs
        try:
            if unit_tasks is None:
                return place, what, _run_task(what, inputs)
            return place, what, _run_unit(what, unit_tasks, inputs)
        except TaskFailed as exc:
            raise EngineTaskFailed(place, str(exc)) from None

    if len(items) <= 1 or threads == 1:
        ran = [run(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=threads, initializer=_limit_openmp_in_task_thread) as pool:
            futures = [pool.submit(run, item) for item in items]
            ran = [future.result() for future in as_completed(futures)]

    # Per model, in the order its tasks or units came back, as `_run_day` and `_run_block` collect them.
    done: Dict[int, list] = {place: [] for place, _ in job}
    for place, what, outcome in ran:
        done[place].append((what, outcome))
    out: List[Outcome] = []
    for place, outcomes in done.items():
        if blocks[place]:
            results, skipped = block_records(models[place].inputs.config, [outcome for _, outcome in outcomes])
        else:
            results = [outcome for _, outcome in outcomes if not isinstance(outcome, str)]
            skipped = [(task, outcome) for task, outcome in outcomes if isinstance(outcome, str)]
        out.append((place, results, skipped))
    return out
