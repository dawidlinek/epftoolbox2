"""The worker side of an engine group (`models/_engine.py`): one pool for every model of a chunk, each job one (day or refit
block, period) holding the pending tasks of every model of the chunk on that day or block and period.

Each task runs on its model's **own** inputs (`_worker.TaskInputs`: the chunk's union arrays, which a fit reads by column name,
and the model's expanded lists, config and rebuilt model), through the model path's functions: `_run_task` for a refit=None
task, `_run_unit` for each unit of a refit block (`_worker.block_units`), whose records `_worker.block_records` makes as
`_run_block` does. Those functions select every fit's rows, mask and skips, and build every record.

**Fits from shared statistics** (step 5i). A model whose config names a fit path (`config["fit_path"]`, `fit_path_of`: its
`_ENGINE_NEEDS` asks for no CV blocks and its scaler is exactly `StandardScaler`, so OLS, Ridge and LEAR with
`scaler="standard"`) hands each fit to `EngineFits` once the model path has selected it (`_worker.StatsFit`), in place of the
scaler and `_fit_predict` / `_fit_model`:

- per **span** (period, horizon and the period's rows of `training_span`, before any filter: the same rows, by date, in any
  group), one `_sums.SpanSums` over the columns the job's fits of that span read (`_job_spans`), built when its first fit asks
  and let go after its last; one per task thread, never shared between threads;
- per fit, `SpanSums.over(ρ, its columns)` (ρ the model path's finite rows; a downdate when at most `DOWNDATE_SHARE` (τ) of the
  span's rows are left out, else a direct sum: the same bits either way), `_stats.standard_stats` with the fit's mask and
  `scale_binary`, `model._fit_stats`, and each issue row through `params.transform`, `model._predict_fitted` (one row at a time)
  and `params.inverse`.

A fit with fewer than `_stats.MIN_ROWS` rows in ρ, one whose sums `standard_stats` can't scale (`StatsFallback`: cancellation,
overflow), one `_fit_stats` declines (None: OLS's rank and conditioning guards, Ridge's n <= p and rounding budget, an estimator
that isn't the one the statistics reproduce) or raises a ValueError on is **handed back**: the model path fits it there, on the
same values, to the bit. Each hand-back is decided from the fit's own rows and span, so it is the same in any group.

Every element of the sums is a function of the two columns' values in ρ and of the span (exact integer slices, `_sums`), and
everything after them is the fit's own, so a model's records are the same bits alone, in any group, order or chunking, at any
process, thread or BLAS thread count, and whatever τ. Nothing stored names the engine but the manifest's `fit_path` (compared on
resume) and its provenance counts (`EngineCounts`, never compared).
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from ..scalers.registry import make_scaler
from ..scalers.standard import StandardScaler
from ._stats import MIN_ROWS, StatsFallback, standard_stats
from ._sums import SpanSums
from ._worker import (
    StatsFit,
    TaskFailed,
    TaskInputs,
    _limit_native_threads,
    _limit_openmp_in_task_thread,
    _run_task,
    _run_unit,
    block_records,
    block_units,
    issue_row,
    member_rows,
    task_inputs,
    task_list,
    task_newest,
    task_place,
    unit_rows,
)

# The numeric path of the fits from shared statistics, as a manifest records it (`fit_path`, compared on resume): a later
# release that changes their numbers gets another name. None runs every fit of a group on the model path (5h's plumbing, which
# `tests/models/test_engine_groups.py` checks to the bit against engine off).
ENGINE_FIT_PATH: Optional[str] = "engine-1"

# τ: a fit's sums are the span's totals minus the rows ρ leaves out when those are at most this share of the span's rows, else
# summed over ρ directly. Both give the same bits (`_sums`); τ decides speed alone.
DOWNDATE_SHARE = 0.5

# An issue row with a scaled column more than this many training standard deviations from its training mean is counted
# (`EngineCounts.far_issue_rows`): an extrapolation multiplies a coefficient's last bits, which no fit-time guard can see.
FAR_ISSUE_Z = 100.0

# Why a fit went back to the model path (`EngineCounts.fallbacks`): fewer than `MIN_ROWS` rows in ρ; `StatsFallback`'s reasons
# ("cancellation", "overflow"); `_fit_stats` returned None; `_fit_stats` raised a ValueError.
FEW_ROWS = "few_rows"
DECLINED = "declined"
ERROR = "error"

# What `_engine_init` loaded into a pool's worker: the chunk's models' inputs, by their place in the chunk.
_E_MODELS: List["ModelInputs"] = []


def fit_path_of(model) -> Optional[str]:
    """The numeric path of a model's fits in an engine group (`ENGINE_FIT_PATH`), or None where they are the model path's: a
    model whose `_ENGINE_NEEDS` asks for CV blocks (LassoCV, LEAR `"cv"`; or that declares none) or whose scaler isn't exactly
    `StandardScaler` (a VST) keeps the model path in this release."""
    if ENGINE_FIT_PATH is None:
        return None
    needs = getattr(model, "_ENGINE_NEEDS", None)
    if needs is None or set(needs) - {"rows"}:
        return None
    try:
        scaler = make_scaler(getattr(model, "scaler", "standard"))
    except (TypeError, ValueError):
        return None
    return ENGINE_FIT_PATH if type(scaler) is StandardScaler else None


class ModelInputs(NamedTuple):
    """One model of a chunk in a worker: its name (for errors), its `TaskInputs`, and with fits from shared statistics
    (`config["fit_path"]`) its scaler's `scale_binary`, else None (every fit on the model path)."""

    name: str
    inputs: TaskInputs
    scale_binary: Optional[bool] = None


class EngineTaskFailed(TaskFailed):
    """A task of one model of an engine job raised: `model` is the model's place in the chunk, `message` the model path's
    `TaskFailed` text (the task and the worker's traceback)."""

    def __init__(self, model: int, message: str):
        super().__init__(model, message)
        self.model = model
        self.message = message

    def __str__(self) -> str:
        return self.message


class EngineCounts:
    """What one model's fits from shared statistics did, for the manifest's provenance (never compared): `fits` made from the
    statistics, `fallbacks` handed back to the model path by reason, and of the issue rows they forecast the largest |z| of a
    scaled column (`max_issue_z`, training standard deviations from the training mean) and how many had one above `FAR_ISSUE_Z`
    (`far_issue_rows`)."""

    __slots__ = ("fits", "fallbacks", "max_issue_z", "far_issue_rows")

    def __init__(self):
        self.fits = 0
        self.fallbacks: Counter = Counter()
        self.max_issue_z = 0.0
        self.far_issue_rows = 0

    def __bool__(self) -> bool:
        return bool(self.fits or self.fallbacks)

    def add(self, other: "EngineCounts") -> None:
        self.fits += other.fits
        self.fallbacks.update(other.fallbacks)
        self.max_issue_z = max(self.max_issue_z, other.max_issue_z)
        self.far_issue_rows += other.far_issue_rows

    def issue(self, z: float) -> None:
        self.max_issue_z = max(self.max_issue_z, z)
        if z > FAR_ISSUE_Z:
            self.far_issue_rows += 1

    def to_dict(self) -> Dict:
        return {
            "fits": int(self.fits),
            "fallbacks": {reason: int(self.fallbacks[reason]) for reason in sorted(self.fallbacks)},
            "max_issue_z": float(self.max_issue_z),
            "far_issue_rows": int(self.far_issue_rows),
        }


def model_inputs(hour_arrays: Dict, models: Sequence[tuple]) -> List[ModelInputs]:
    """Each model's inputs from (name, expanded lists, config, class, `_model_kwargs`), the model rebuilt as a pool's worker
    rebuilds it (`_worker.task_inputs`), all on the chunk's one set of arrays."""
    out = []
    for name, expanded, config, model_cls, model_kwargs in models:
        scale_binary = None
        if config.get("fit_path") is not None:
            scale_binary = bool(make_scaler(config.get("scaler", "standard")).scale_binary)
        out.append(ModelInputs(name, task_inputs(hour_arrays, expanded, config, model_cls, model_kwargs), scale_binary))
    return out


def _engine_init(hour_arrays: Dict, models: Sequence[tuple]) -> None:
    """Initialiser of an engine pool: native threads limited as `_worker._pool_init` limits them, then every model's inputs."""
    _limit_native_threads()
    _engine_load(hour_arrays, models)


def _engine_load(hour_arrays: Dict, models: Sequence[tuple]) -> None:
    """Load a chunk's models' inputs into this process (`_engine_init` without the thread limit, for a caller's process)."""
    global _E_MODELS
    _E_MODELS = model_inputs(hour_arrays, models)


# --- spans ------------------------------------------------------------------------------------------------------------

SpanKey = Tuple[
    int, int, int, int
]  # (period, horizon, first row, end row) of the period's arrays: a span's rows, by date


def _item_spans(what: tuple, unit_tasks: Optional[List[tuple]], inputs: TaskInputs) -> List[Tuple[SpanKey, List[int]]]:
    """The spans of a task's (refit=None: each member's) or a unit's fits, with the columns each reads, by the model path's own
    selection (`_fit_one_numpy`, `_fit_unit`): what a job's `SpanSums` must hold. A task with no issue row has none."""
    cfg = inputs.config
    if unit_tasks is None:
        hour, hz, day_in_test = what
        arrays = inputs.hour_arrays[hour]
        days = arrays["days"]
        day = cfg["offset"] + day_in_test
        if issue_row(days, day) is None:
            return []
        _, _, position = task_place(cfg, day_in_test)
        newest = task_newest(cfg, hour, position)
        _, predictors, _ = task_list(cfg, inputs.expanded, hour, hz, position)
        members = cfg.get("members")
        windows = [cfg["training_window"]] if members is None else members
        spans = [member_rows(days, day, member, hz, newest) for member in windows]
    else:
        hour, hz = what.period, what.horizon
        arrays = inputs.hour_arrays[hour]
        spans = [unit_rows(cfg, arrays["days"], what)[0]]
        predictors = inputs.expanded[(hour, hz) if what.list_id is None else what.list_id]
    columns = [arrays["col_index"][name] for name in predictors]
    return [((hour, hz, rows.start, rows.stop), columns) for rows in spans if rows.stop > rows.start]


class _Span:
    """One span's `SpanSums` over the columns its job's fits read (their places in the period's arrays, ascending)."""

    __slots__ = ("sums", "columns", "n")

    def __init__(self, arrays: Dict, key: SpanKey, columns: np.ndarray):
        _, hz, start, stop = key
        self.columns = columns
        self.n = stop - start
        self.sums = SpanSums(arrays["x"][start:stop][:, columns], arrays["y"][start:stop, hz - 1].copy())

    def local(self, columns: Sequence[int]) -> np.ndarray:
        """The places of a fit's columns among the span's; a column the job's plan didn't give the span is a bug, not a fit."""
        wanted = np.asarray(columns, dtype=np.int64)
        at = np.searchsorted(self.columns, wanted)
        if wanted.size and (at.max() >= self.columns.size or not np.array_equal(self.columns[at], wanted)):
            raise RuntimeError(
                f"a fit reads columns {wanted.tolist()} its span's sums don't hold {self.columns.tolist()}"
            )
        return at


class _Spans:
    """The spans of one task thread: each span's `_Span` built when a fit first asks for it, from the job's plan (`columns`, the
    columns each span's fits read), and let go once the thread ran the last item that reads it (`done`)."""

    def __init__(self, hour_arrays: Dict, columns: Dict[SpanKey, np.ndarray], uses: Counter):
        self.hour_arrays = hour_arrays
        self.columns = columns
        self.uses = uses
        self.built: Dict[SpanKey, _Span] = {}

    def get(self, period: int, horizon: int, rows: slice) -> _Span:
        key = (period, horizon, rows.start, rows.stop)
        span = self.built.get(key)
        if span is None:
            if key not in self.columns:
                raise RuntimeError(f"a fit of span {key} that its job's plan doesn't have")
            span = self.built[key] = _Span(self.hour_arrays[period], key, self.columns[key])
        return span

    def done(self, keys) -> None:
        for key in keys:
            self.uses[key] -= 1
            if self.uses[key] <= 0:
                self.built.pop(key, None)


class EngineFits:
    """A model's fits from shared statistics in one task thread: the `engine` its tasks' `_fit_member` and units' `_fit_unit`
    call with each fit (`_worker.StatsFit`), returning (prediction in target units, coefficients) per issue row, or None to hand
    the fit back to the model path (see the module)."""

    def __init__(self, spans: _Spans, model, scale_binary: bool, counts: EngineCounts):
        self.spans = spans
        self.model = model
        self.scale_binary = scale_binary
        self.counts = counts

    def _back(self, reason: str) -> None:
        self.counts.fallbacks[reason] += 1
        return None

    def __call__(self, fit: StatsFit) -> Optional[List[Tuple[float, list]]]:
        r = fit.x.shape[0]
        if r < MIN_ROWS:
            return self._back(FEW_ROWS)
        span = self.spans.get(fit.period, fit.horizon, fit.rows)
        columns = span.local(fit.columns)
        method = "downdate" if span.n - r <= DOWNDATE_SHARE * span.n else "direct"
        raw = span.sums.over(fit.finite, columns, method=method)
        try:
            stats, params = standard_stats(raw, (fit.x, fit.y), fit.mask, self.scale_binary)
        except StatsFallback as exc:
            return self._back(exc.reason)
        try:
            fitted = self.model._fit_stats(stats)
        except ValueError:
            return self._back(ERROR)
        if fitted is None:
            return self._back(DECLINED)
        varying = params.scaled & ~params.constant
        out = []
        for row in range(fit.test.shape[0]):
            # One row at a time, as the model path forecasts (decision 16 of the step 3 spec).
            z = params.transform(fit.test[row : row + 1])
            self.counts.issue(float(np.abs(z[0, varying]).max()) if varying.any() else 0.0)
            prediction, coefs = self.model._predict_fitted(fitted, z)
            out.append((params.inverse(float(prediction)), coefs))
        self.counts.fits += 1
        return out


def _job_spans(
    items: Sequence[Tuple[int, tuple, Optional[List[tuple]]]], models: Sequence[ModelInputs]
) -> Tuple[List[List[SpanKey]], Dict[SpanKey, np.ndarray]]:
    """Per item the spans its engine fits read (none for a model on the model path), and per span the columns of all of them."""
    keys: List[List[SpanKey]] = []
    columns: Dict[SpanKey, set] = {}
    for place, what, unit_tasks in items:
        if models[place].scale_binary is None:
            keys.append([])
            continue
        spans = _item_spans(what, unit_tasks, models[place].inputs)
        keys.append(list(dict.fromkeys(key for key, _ in spans)))
        for key, cols in spans:
            columns.setdefault(key, set()).update(cols)
    return keys, {key: np.array(sorted(cols), dtype=np.int64) for key, cols in columns.items()}


def _batches(keys: List[List[SpanKey]], threads: int) -> List[List[int]]:
    """The items cut into at most `threads` batches, each run on one thread with its own spans: the items that read the same
    spans together, the largest such sets first onto the least loaded batch (ties: the earlier batch); each batch in item
    order."""
    together: Dict[tuple, List[int]] = {}
    for i, item_keys in enumerate(keys):
        together.setdefault(tuple(sorted(item_keys)), []).append(i)
    batches: List[List[int]] = [[] for _ in range(threads)]
    for _, indices in sorted(together.items(), key=lambda kv: (-len(kv[1]), kv[1][0])):
        least = min(range(threads), key=lambda b: (len(batches[b]), b))
        batches[least].extend(indices)
    return [sorted(batch) for batch in batches if batch]


Outcome = Tuple[int, List[Dict], List[Tuple[tuple, str]], Optional[EngineCounts]]


def run_job(job: Sequence[Tuple[int, List[tuple]]], models: Optional[Sequence[ModelInputs]] = None) -> List[Outcome]:
    """Run one engine job: (model's place in the chunk, its tasks of the job's day or block and period) for each model with
    some, on `models` (the pool's if None). Returns (place, records, [(task, skip reason)], its `EngineCounts` or None) per
    model, as `_run_day` or `_run_block` returns them for that model's tasks, its tasks or units in their order in the job.
    With `THREADS_PER_PROCESS` above 1 the job's tasks (refit=None) or units (refit) of every model run on that many threads,
    those that read the same spans on one (`_batches`), each thread with its own spans. A task that raises raises
    `EngineTaskFailed` naming its model."""
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
    keys, columns = _job_spans(items, models)
    arrays = {id(models[place].inputs.hour_arrays) for place, _ in job if models[place].scale_binary is not None}
    if len(arrays) > 1:
        raise RuntimeError("the models of an engine job with fits from shared statistics must share one set of arrays")
    hour_arrays = models[job[0][0]].inputs.hour_arrays if job else {}

    def run(batch: List[int]) -> Tuple[List[Tuple[int, Union[Dict, str, list]]], Dict[int, EngineCounts]]:
        spans = _Spans(hour_arrays, columns, Counter(key for i in batch for key in keys[i]))
        counts: Dict[int, EngineCounts] = {}
        fitters: Dict[int, EngineFits] = {}
        out = []
        for i in batch:
            place, what, unit_tasks = items[i]
            model = models[place]
            engine = None
            if model.scale_binary is not None:
                engine = fitters.get(place)
                if engine is None:
                    counts[place] = EngineCounts()
                    engine = fitters[place] = EngineFits(spans, model.inputs.model, model.scale_binary, counts[place])
            try:
                if unit_tasks is None:
                    outcome = _run_task(what, model.inputs, engine)
                else:
                    outcome = _run_unit(what, unit_tasks, model.inputs, engine)
            except TaskFailed as exc:
                raise EngineTaskFailed(place, str(exc)) from None
            spans.done(keys[i])
            out.append((i, outcome))
        return out, counts

    if len(items) <= 1 or threads == 1:
        parts = [run(list(range(len(items))))]
    else:
        with ThreadPoolExecutor(max_workers=threads, initializer=_limit_openmp_in_task_thread) as pool:
            futures = [pool.submit(run, batch) for batch in _batches(keys, threads)]
            parts = [future.result() for future in as_completed(futures)]
    ran = sorted(((i, outcome) for part, _ in parts for i, outcome in part), key=lambda done: done[0])
    totals: Dict[int, EngineCounts] = {}
    for _, counts in parts:
        for place, part_counts in counts.items():
            totals.setdefault(place, EngineCounts()).add(part_counts)

    # Per model, in the order of its tasks or units in the job, as `_run_day` and `_run_block` collect them on one thread.
    done: Dict[int, list] = {place: [] for place, _ in job}
    for i, outcome in ran:
        place, what, _ = items[i]
        done[place].append((what, outcome))
    out: List[Outcome] = []
    for place, outcomes in done.items():
        if blocks[place]:
            results, skipped = block_records(models[place].inputs.config, [outcome for _, outcome in outcomes])
        else:
            results = [outcome for _, outcome in outcomes if not isinstance(outcome, str)]
            skipped = [(task, outcome) for task, outcome in outcomes if isinstance(outcome, str)]
        out.append((place, results, skipped, totals.get(place)))
    return out
