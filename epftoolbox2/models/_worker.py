from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import contextlib
import contextvars
import datetime
import hashlib
import math
import os
import sys
import threading
import traceback

import numpy as np
from threadpoolctl import ThreadpoolController

from ..results.store import with_status
from ..scalers.registry import make_scaler
from ..scalers.standard import StandardScaler
from ..scalers.target import MultiTargetAsinh
from ._calendar import (
    MASK_FIT,
    FitUnit,
    block_days,
    member_window,
    nominal_block_length,
    refit_anchor,
    training_span,
)
from ._row_target import (
    ZERO_WHEN,
    has_offset,
    issue_offset,
    learner_targets,
    scale_column,
    scaled_back,
    usable_scale,
    zero_outcome,
)
from ._units import DAY, PER_PERIOD, PERIOD, POOLED, day_arrays, kind_of, pooled_values

# What `_worker_init` loads into a pool's worker: the default `TaskInputs` of every task function (`_pool_inputs`).
_W_HOUR_ARRAYS: Dict = {}
_W_EXPANDED_PREDS: Dict = {}
_W_CONFIG: Dict = {}
_W_MODEL = None  # built with refit=None: its `_fit_predict`, or a refit's `_fit_model` and `_predict_fitted`

# Native thread pools (BLAS, OpenMP) used by one task. The pool already runs MAX_PROCESSES x THREADS_PER_PROCESS
# tasks at once, so more threads per task only oversubscribe the cores (an OLS run took 4-40x longer that way).
NATIVE_THREADS_PER_TASK = 1
# Read by these libraries when they load, so they cover a library a model imports after the pool started.
NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_W_THREADPOOLS: Optional[ThreadpoolController] = None  # set by `_limit_native_threads`, in a pool's worker only
# `pool_native_threads` in the process that starts the pools: how many holds are open, and what to put back after the last.
_HOLD_LOCK = threading.Lock()
_HOLDS = 0
_HELD_COUNTS: List[Tuple[Any, int]] = []  # (threadpoolctl library controller, the thread count it had)
_HELD_ENV: Dict[str, Optional[str]] = {}  # the thread variables as they were, under spawn
# Whether the pools spawn their workers rather than fork them (`BaseModel._run_tasks`, `_engine._run_chunk`).
_POOLS_SPAWN = sys.platform == "win32"

# Reasons a task produces no forecast without being an error. They are counted per model and reported.
SKIP_TOO_FEW_ROWS = "too_few_rows"
SKIP_NO_FINITE_TRAINING_ROWS = "no_finite_training_rows"
SKIP_MISSING_TEST_PREDICTORS = "missing_test_predictors"
SKIP_FIT_ERROR = "fit_error"
SKIP_NO_MATCHING_DAYS = "no_matching_days"
# The issue day has no row for this period (a gap in the data); without the row there is nothing to forecast from.
SKIP_MISSING_TEST_ROW = "missing_test_row"
# A model that forecasts a published value (`NaiveModel`, `known_only`) whose value its rule puts after the issue time: the
# forecast didn't exist yet. Decided in the parent, before any task runs (`BaseModel._unpublished`).
SKIP_NOT_PUBLISHED = "not_published"
# A model with `normalise_by` whose issue row's scale is zero or negative: nothing to multiply the forecast by
# (`models._row_target`); a missing one is `missing_test_predictors`.
SKIP_NON_POSITIVE_SCALE = "non_positive_scale"

# In `PredictorPlan.day_lists`, next to list ids: a day no matching forecast of that (period, horizon) reads, and a day
# the function returned a list on that no forecast uses, which matches nothing.
NOT_PLANNED = -1
OTHER_LIST = -2


class TaskFailed(RuntimeError):
    """An unexpected exception inside a worker task; carries the task identity and the worker's traceback."""


class FitContext(NamedTuple):
    """What a model's fit call (`_fit_predict`, `_fit_model`, `_fit_days`) is fitting, for a model whose fit draws random
    numbers (`TorchModel`: its seeds are `_calendar.unit_seed` of the unit, its shapes from the window): the unit (a refit
    unit's own `FitUnit`; a fit of one forecast, refit=None, is `FitUnit(period, horizon, issue date, list id, member index)`),
    the member's training window as the model has it (an int or `"expanding"`), the hash of the unit's predictor list
    (`config["list_hashes"]`, a predictor function's; None for a fixed list), which, unlike the list id, a plan numbering its
    lists by first appearance gives the same whatever days the run holds, and the day id of each training row the call gets,
    in row order (a period's repeated autumn hour gives its date two rows)."""

    unit: FitUnit
    window: Any
    list_hash: Optional[str] = None
    row_days: Optional[np.ndarray] = None


# The fit in progress in this thread (`fit_scope`), None outside one: a context variable, so each task thread of a pool's
# process (`THREADS_PER_PROCESS`) sees its own.
_FIT: contextvars.ContextVar = contextvars.ContextVar("epftoolbox2_fit", default=None)


def current_fit() -> Optional[FitContext]:
    """The `FitContext` of the fit call in progress in this thread, None outside one."""
    return _FIT.get()


@contextlib.contextmanager
def fit_scope(unit: FitUnit, window: Any, list_hash: Optional[str] = None, row_days: Optional[np.ndarray] = None):
    """Make `FitContext(unit, window, list_hash, row_days)` the fit in progress around a model's fit call, and put back what
    was there after."""
    token = _FIT.set(FitContext(unit, window, list_hash, row_days))
    try:
        yield
    finally:
        _FIT.reset(token)


def _unit_list_hash(cfg: Dict, list_id: Optional[int]) -> Optional[str]:
    """The hash of a predictor function's list `list_id` (`config["list_hashes"]`), None for a fixed list."""
    return None if list_id is None else cfg["list_hashes"][list_id]


def _window_of(cfg: Dict, unit: FitUnit) -> Any:
    """A refit unit's training window as the model has it: the model's, or with a list of them the unit's member's."""
    members = cfg.get("members")
    return cfg["training_window"] if members is None else members[unit.member]


def _limit_native_threads() -> None:
    """Limit BLAS and OpenMP to NATIVE_THREADS_PER_TASK threads in this worker process.

    Only for a pool worker: the limits are never restored, and the caller's process must keep its own.
    """
    global _W_THREADPOOLS
    for name in NATIVE_THREAD_ENV_VARS:
        os.environ[name] = str(NATIVE_THREADS_PER_TASK)
    # Libraries already loaded ignore the variables: under fork that is everything the parent imported, under
    # spawn whatever importing this module loaded (numpy's BLAS at least). threadpoolctl sets them directly, and only where
    # the count differs: an OpenBLAS with its own thread pool comes with the count already (`pool_native_threads`), and
    # setting it again in a forked child would start that pool's threads.
    _W_THREADPOOLS = ThreadpoolController()
    for lib in _W_THREADPOOLS.lib_controllers:
        if lib.num_threads != NATIVE_THREADS_PER_TASK:
            lib.set_num_threads(NATIVE_THREADS_PER_TASK)


def _has_own_thread_pool(lib) -> bool:
    """Whether a threadpoolctl library controller is an OpenBLAS that runs its own pool of threads (not OpenMP's)."""
    return lib.internal_api == "openblas" and getattr(lib, "threading_layer", None) not in ("openmp", "disabled")


@contextlib.contextmanager
def pool_native_threads():
    """Hold this process's OpenBLAS at NATIVE_THREADS_PER_TASK threads while worker pools start from it, and put back the
    counts it had when the outermost hold closes: nested and concurrent holds share one.

    OpenBLAS stops its thread pool before every fork, and setting its thread count in the child starts the pool again: one
    thread per CPU it counted when it loaded, each spinning for its thread timeout (about 0.1 s) before it sleeps, in every
    worker of every pool. A worker that inherits the count needs no set (`_limit_native_threads`), so it starts none. Putting
    the count back starts this process's own pool again, once per hold: hold it around a whole run, not around each pool.
    Under spawn the thread variables are set too, for the workers to load their libraries at that count; under fork they
    would reach only a library this process loads during the hold, which would then keep it. While held, BLAS calls of this
    process, in any thread, run on NATIVE_THREADS_PER_TASK threads.
    """
    global _HOLDS
    with _HOLD_LOCK:
        if _HOLDS == 0:
            _hold_native_threads()
        _HOLDS += 1
    try:
        yield
    finally:
        with _HOLD_LOCK:
            _HOLDS -= 1
            if _HOLDS == 0:
                _release_native_threads()


def _hold_native_threads() -> None:
    for lib in ThreadpoolController().lib_controllers:
        if _has_own_thread_pool(lib):
            count = lib.num_threads
            if count != NATIVE_THREADS_PER_TASK:
                lib.set_num_threads(NATIVE_THREADS_PER_TASK)
                _HELD_COUNTS.append((lib, count))
    if _POOLS_SPAWN:
        for name in NATIVE_THREAD_ENV_VARS:
            _HELD_ENV[name] = os.environ.get(name)
            os.environ[name] = str(NATIVE_THREADS_PER_TASK)


def _release_native_threads() -> None:
    while _HELD_COUNTS:
        lib, count = _HELD_COUNTS.pop()
        lib.set_num_threads(count)
    for name, value in _HELD_ENV.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    _HELD_ENV.clear()


def _limit_openmp_in_task_thread() -> None:
    """The OpenMP limit is per thread (the BLAS ones are per process), so each task thread sets its own."""
    if _W_THREADPOOLS is not None:
        _W_THREADPOOLS.limit(limits=NATIVE_THREADS_PER_TASK, user_api="openmp")


class TaskInputs(NamedTuple):
    """What the task functions of this module run one model's tasks with: the period arrays (`BaseModel._build_hour_arrays`:
    `x`, `y`, `days`, `timestamps`, `col_index` per period; a fit reads its columns by name through `col_index`, so arrays with
    more columns give the same records), the expanded lists (keyed (period, horizon), or list id with a predictor function), the
    model's config (`BaseModel._worker_inputs`) and the model, rebuilt as `model_cls(predictors=[], **model_kwargs)` with
    refit=None whatever the model's (`task_inputs`): what a refit needs travels in `config["refit"]`. Every task function takes
    one; None is the pool's (`_pool_inputs`), what `_worker_init` loaded."""

    hour_arrays: Dict
    expanded: Dict
    config: Dict
    model: Any


def task_inputs(hour_arrays, expanded_preds, config, model_cls, model_kwargs) -> TaskInputs:
    """The inputs of one model's tasks, the model rebuilt from its class and `_model_kwargs` as a worker rebuilds it."""
    return TaskInputs(hour_arrays, expanded_preds, config, model_cls(predictors=[], **model_kwargs))


def _pool_init(hour_arrays, expanded_preds, config, model_cls, model_kwargs):
    """Initialiser of the worker pool: limit native threads, then load the data and the model.

    A torch family's config sets `torch_threads`: torch's intra-op threads (OpenMP and its bundled MKL) are pinned to it and its
    inter-op threads to 1 (`_torch.runtime.pin_torch_threads`) before the BLAS and OpenMP limits, and again after them (which
    set torch's OpenMP to one thread too). A thread count changes a torch fit's numbers, a forked worker inherits the parent's
    count whatever the environment says, and its bundled MKL left at the parent's count hangs on the first product after the
    parent's parallel work. More than one thread is refused where the pool forks (`check_torch_threads`). Without the setting
    nothing imports torch."""
    torch_threads = config.get("torch_threads")
    if torch_threads:
        from ._torch.runtime import pin_torch_threads

        pin_torch_threads(torch_threads, forked=not _POOLS_SPAWN)
    _limit_native_threads()
    if torch_threads:
        pin_torch_threads(torch_threads, forked=not _POOLS_SPAWN)
    _worker_init(hour_arrays, expanded_preds, config, model_cls, model_kwargs)


def _worker_init(hour_arrays, expanded_preds, config, model_cls, model_kwargs):
    """Load one model's inputs into this process's module globals, the default of every task function (`_pool_inputs`)."""
    global _W_HOUR_ARRAYS, _W_EXPANDED_PREDS, _W_CONFIG, _W_MODEL
    _W_HOUR_ARRAYS = hour_arrays
    _W_EXPANDED_PREDS = expanded_preds
    _W_CONFIG = config
    _W_MODEL = task_inputs(hour_arrays, expanded_preds, config, model_cls, model_kwargs).model


def _pool_inputs() -> TaskInputs:
    """The inputs `_worker_init` loaded into this process: what a task function runs with when it is given none."""
    return TaskInputs(_W_HOUR_ARRAYS, _W_EXPANDED_PREDS, _W_CONFIG, _W_MODEL)


def _inputs_of(inputs: Optional[TaskInputs]) -> TaskInputs:
    """The inputs a task function runs with: `inputs` whole, or without them the pool's, all four of what `_worker_init` loaded,
    never some of each. A ValueError if they have no model (no `_worker_init` in this process, or inputs built without one):
    a fit never falls back to another model."""
    resolved = _pool_inputs() if inputs is None else inputs
    if resolved.model is None:
        raise ValueError(
            "a task's inputs have no model: pass TaskInputs with one (task_inputs), or run it in a process _worker_init set up"
        )
    return resolved


def training_rows(days: np.ndarray, day: int, training_window: int, hz: int, newest: int = -1) -> slice:
    """The rows the task (`hz`, `day`) may train on, in a period whose rows are on `days`, which are sorted: those whose date is in
    `training_span`, `newest` being the day of the newest target relative to `day`. Rows are chosen by date, since a date can hold
    two of them or none (see `epftoolbox2.models._calendar`), so a doubled autumn row has both occurrences in or out, each with the
    same target."""
    # newest_target="d-1" takes today's path, call for call.
    first, last = (
        training_span(day, training_window, hz) if newest == -1 else training_span(day, training_window, hz, newest)
    )
    return slice(int(np.searchsorted(days, first, side="left")), int(np.searchsorted(days, last, side="right")))


def issue_row(days: np.ndarray, day: int) -> Optional[int]:
    """The row a forecast issued on `day` is made from: the last one of that date, so the second occurrence of a repeated autumn
    hour. None if the date has no row in this period."""
    row = int(np.searchsorted(days, day, side="right")) - 1
    return row if row >= 0 and days[row] == day else None


def training_row_days(days: np.ndarray, day: int, training_window: int, hz: int, newest: int = -1) -> np.ndarray:
    """The day id of each training row of the task (`hz`, `day`) in a period whose rows are on `days`, before any filter."""
    if newest == -1:
        return days[training_rows(days, day, training_window, hz)]
    return days[training_rows(days, day, training_window, hz, newest)]


def matching_rows(row_days: np.ndarray, lists: np.ndarray, start: int, list_id: int) -> np.ndarray:
    """Which of these rows got list `list_id`, where `lists` has an entry per calendar day from day id `start` on. A doubled
    autumn row follows its date. A row the plan didn't evaluate would quietly match nothing, so it raises."""
    row_lists = lists[row_days - start]
    if (row_lists == NOT_PLANNED).any():
        raise RuntimeError(
            "A training row of a train_on='matching' forecast is on a day its plan has no list for: "
            f"{(row_days[row_lists == NOT_PLANNED] - start).tolist()[:3]} day(s) from the start of the plan's days"
        )
    return row_lists == list_id


def matching_digest(row_days: np.ndarray, first_ordinal: int) -> str:
    """A short hash of a set of days, as dates: day ids depend on where the frame starts, the dates don't."""
    days = np.unique(row_days).astype("<i8") + first_ordinal
    return hashlib.sha256(days.tobytes()).hexdigest()[:12]


def task_place(cfg: Dict, day_in_test: int) -> Tuple[Optional[str], int, int]:
    """(window, days since its start, place among the issue days) of a task's day, `day_in_test` counting from the run's first
    day: `config["run_days"]` in a run with windows or `issue_days`, else (None, the day, the day)."""
    run_days = cfg.get("run_days")
    return (None, day_in_test, day_in_test) if run_days is None else run_days[day_in_test]


def task_newest(cfg: Dict, hour: int, position: int) -> int:
    """The newest target of a task of a refit=None model, relative to its issue day, by its place among the issue days: -1 with
    `newest_target="d-1"` (no `config["newest"]`)."""
    newest_targets = cfg.get("newest")
    return -1 if newest_targets is None else int(newest_targets[hour][position])


def task_list(cfg: Dict, expanded: Dict, hour: int, hz: int, position: int) -> Tuple[Optional[int], List[str], tuple]:
    """(list id, the list's names, its masks' key) of the task (`hour`, `hz`) at its place among the issue days. A fixed list is
    keyed by (period, horizon), with no id. With a predictor function of `ctx` (see epftoolbox2.models.context) each task has its
    own list (`config["task_lists"]`), and each list its own binary-column mask per (period, horizon). The mask of a matching
    list is that list's too, not one per set of days, while the scaler is fitted on what the filter leaves."""
    task_lists = cfg.get("task_lists")
    if task_lists is None:
        return None, expanded[(hour, hz)], (hour, hz)
    list_id = int(task_lists[(hour, hz)][position])
    return list_id, expanded[list_id], (list_id, hour, hz)


def task_mask(cfg: Dict, window: Optional[str], mask_of: tuple, member: Any = None) -> Optional[np.ndarray]:
    """The parent's binary-column mask of a refit=None fit: of the task's list (`mask_of`, from `task_list`) in its window, and
    with a list of training windows of its member. None with `binary_mask="fit"` (`config["binary_mask"]`): the fit reads its own
    from its span's rows (`_fit_member`)."""
    if cfg.get("binary_mask") == MASK_FIT:
        return None
    masks = cfg["scalable_masks"] if window is None else cfg["scalable_masks"][window]
    return masks[mask_of if member is None else (*mask_of, member)]


def member_rows(days: np.ndarray, day: int, member: Any, hz: int, newest: int) -> slice:
    """The rows one training window's fit of the task (`hz`, day id `day`) trains on, in a period whose rows are on `days`, before
    any filter: those of `training_span` with the member's window (`member_window`: `"expanding"` from the frame's first date) and
    the task's newest target (-1 with `newest_target="d-1"`, which takes today's path, call for call)."""
    span = member_window(member, day)
    return training_rows(days, day, span, hz) if newest == -1 else training_rows(days, day, span, hz, newest)


def unit_rows(cfg: Dict, days: np.ndarray, unit: FitUnit) -> Tuple[slice, int]:
    """(the rows, the newest target relative to the anchor) of a refit unit, in a period whose rows are on `days`, before any
    filter: those of a forecast issued on the anchor, `training_span(anchor, W, h, newest)`, W the model's window or with a list of
    them the unit's member's, the newest target read on the anchor (a target published between the anchor and the issue day
    doesn't reach the fit), -1 with `newest_target="d-1"`."""
    refit = cfg["refit"]
    anchor = unit.anchor.toordinal() - refit["first_ordinal"]
    members = cfg.get("members")
    span = member_window(cfg["training_window"] if members is None else members[unit.member], anchor)
    newest_targets = refit["newest"]
    if newest_targets is None:
        return training_rows(days, anchor, span, unit.horizon), -1
    newest = int(newest_targets.at(unit.period, unit.anchor.toordinal()))
    return training_rows(days, anchor, span, unit.horizon, newest), newest


class StatsFit(NamedTuple):
    """One fit as the model path has selected it, for a fit from shared statistics (`engine` of `_fit_member` and `_fit_unit`,
    which the engine's worker passes, `_engine_worker.EngineFits`): its span, the period's rows `rows` of `training_span` (a
    slice, before any filter) at `horizon`, read by every fit of that span; `columns`, the fit's columns' places in the period's
    arrays, in the list's order; `finite`, which of the span's rows the fit keeps (ρ: target and every column finite); `x` and
    `y`, ρ's values; the scaler's `mask`; `test`, the raw issue rows it forecasts (k rows of its columns, all finite)."""

    period: int
    horizon: int
    rows: slice
    columns: List[int]
    finite: np.ndarray
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    test: np.ndarray


# A fit from shared statistics: (prediction in target units, coefficients) per issue row of `StatsFit.test`, None for a row the
# model path must forecast, or None where the model path must fit the whole fit (a hand-back), which it then does on the same
# rows.
StatsFitter = Any


def finite_rows(train_x: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    """Which training rows a fit keeps: those whose target and every predictor are finite (NaN or inf would fail the fit)."""
    rows = np.isfinite(train_y)
    if train_x.size:
        rows &= np.isfinite(train_x).all(axis=1)
    return rows


def forecast_record(
    run_date: datetime.date,
    hour: int,
    hz: int,
    day_in_window: int,
    prediction: Optional[float],
    actual: float,
    coefficients: list,
    freq: str,
    list_hash: str,
    window: Optional[str] = None,
    predictor_list: Optional[str] = None,
    newest_target_day: Optional[str] = None,
    refit_anchor: Optional[str] = None,
    extra: Optional[Dict] = None,
    detail: Optional[str] = None,
) -> Dict:
    """The record of a forecast, keys in their order: the key fields, `prediction`, `actual` (the issue row's target),
    `coefficients`, `freq`; a refit unit's `refit_anchor`; `newest_target_day` with `newest_target="published"`; a predictor
    function's `predictor_list`; a matching list's `extra` (`training_days`, `matching_days`); `window` in a run with windows;
    then D4's keys (`with_status`): `status` "ok", `list_hash`, `refit_anchor` (the issue day where the record has none: each
    forecast of a refit=None model is its own fit), `revision`, and `detail` where one is given (`"zero_when"`: a forecast of 0
    made without a fit, `_row_target`). `_combined` makes a list of training windows' from it."""
    record = {
        "run_date": run_date.isoformat(),
        "target_date": (run_date + datetime.timedelta(days=hz)).isoformat(),
        "run_weekday": run_date.weekday(),
        "hour": hour,
        "horizon": hz,
        "day_in_test": day_in_window,
        "prediction": prediction,
        "actual": actual,
        "coefficients": coefficients,
        "freq": freq,
    }
    if refit_anchor is not None:
        record["refit_anchor"] = refit_anchor
    if newest_target_day is not None:
        record["newest_target_day"] = newest_target_day
    if predictor_list is not None:
        record["predictor_list"] = predictor_list
    if extra:
        record.update(extra)
    if window is not None:
        record["window"] = window
    if detail is not None:
        return with_status(record, list_hash=list_hash, refit_anchor=record["run_date"], detail=detail)
    return with_status(record, list_hash=list_hash, refit_anchor=record["run_date"])


def _fit_one_numpy(
    hour: int, hz: int, day_in_test: int, inputs: Optional[TaskInputs] = None, engine: Optional[StatsFitter] = None
) -> Union[Dict, str]:
    """Fit and forecast one task, with `inputs` (the pool's if None). `engine` fits each member from shared statistics where it
    can (`_fit_member`); None is the model path.

    Returns the result record, or a skip reason (one of the SKIP_* names, optionally followed by ": detail")
    when no forecast is possible.

    `day_in_test` counts days from the run's first day. In a run with validation `run_days` says what that day is: its
    window, the days since the start of that window (which the record's `day_in_test` holds) and its place in the plan
    of a predictor function, which has no days between the windows. Each window has its own binary-column masks. Nothing else
    depends on the day's window, so a forecast is what a plain run over its window alone gives. With `binary_mask="fit"`
    (`config["binary_mask"]`) there are none: each member's fit reads its mask from its own span's rows (`_fit_member`).

    A model with a list of training windows (`config["members"]`) fits each member on its own rows, with its own mask and
    scaler, and the forecast is their mean in target units (`_combined`), made only if every member's is: else the task is
    skipped with the first failing member's reason, tagged with its window. Member i's forecast is that of the model with
    `training_window` its window, to the bit. With one window the body runs once, as it always did.
    """
    inputs = _inputs_of(inputs)
    arrays = inputs.hour_arrays[hour]
    cfg = inputs.config
    day = cfg["offset"] + day_in_test
    window, day_in_window, position = task_place(cfg, day_in_test)

    days = arrays["days"]
    # A model with newest_target="published" has the day of its newest target per issue day, read from the target's rule.
    newest = task_newest(cfg, hour, position)
    test = issue_row(days, day)
    if test is None:
        return SKIP_MISSING_TEST_ROW
    list_id, predictors, mask_of = task_list(cfg, inputs.expanded, hour, hz, position)

    def fit(member, scalable_mask):
        return _fit_member(
            arrays, cfg, hour, hz, day, member, newest, test, list_id, predictors, scalable_mask, inputs.model, engine
        )

    members = cfg.get("members")
    if members is None:  # one training window: masks keyed as they always were
        fitted = fit(cfg["training_window"], task_mask(cfg, window, mask_of))
        if isinstance(fitted, str):
            return fitted
        prediction, coefs, extra, detail = fitted
    else:
        outcomes = []
        for member in members:
            fitted = fit(member, task_mask(cfg, window, mask_of, member))
            if isinstance(fitted, str):
                return _tagged(fitted, member)
            outcomes.append(fitted)
        # `zero_when` reads the issue row alone, so it says the same for every member.
        prediction, coefs, extra, detail = None, [], {}, outcomes[0][3]

    run_date = datetime.date.fromisoformat(cfg["test_start"]) + datetime.timedelta(days=day_in_test)
    record = forecast_record(
        run_date,
        hour,
        hz,
        day_in_window,
        prediction,
        float(arrays["y"][test, hz - 1]),
        coefs,
        cfg.get("freq", "1h"),
        _list_hash_of(cfg, inputs.expanded, hour, hz, list_id),
        window=window,
        predictor_list=None if list_id is None else cfg["list_hashes"][list_id],
        # The newest target date of the training span, D + min(k, h-1); a row is never dated `day` or later (`training_span`).
        # Not necessarily a target that trained: it can be a slot the clock skips or a missing value, whose row drops.
        newest_target_day=None
        if cfg.get("newest") is None
        else (run_date + datetime.timedelta(days=min(newest, hz - 1))).isoformat(),
        extra=extra,
        detail=detail,
    )
    if members is not None:
        return _combined(record, [prediction for prediction, _, _, _ in outcomes])
    return record


def _scale_skip(value: float) -> Optional[str]:
    """The skip reason of an issue row whose `normalise_by` value is `value`: `missing_test_predictors` for a missing (or
    infinite) one, `non_positive_scale` for one at or below 0; None for a usable one."""
    if not np.isfinite(value):
        return SKIP_MISSING_TEST_PREDICTORS
    if value <= 0:
        return SKIP_NON_POSITIVE_SCALE
    return None


def _list_hash_of(cfg: Dict, expanded: Dict, hour: int, hz: int, list_id: Optional[int]) -> str:
    """The `list_hash` of a record: a predictor function's list's (`config["list_hashes"]`, what `predictor_list` holds), or a
    fixed list's of its (period, horizon), the hash a function returning that list gets (`config["fixed_list_hashes"]`, or from
    the list itself, in `expanded`, for a worker set up without them)."""
    if list_id is not None:
        return cfg["list_hashes"][list_id]
    fixed = cfg.get("fixed_list_hashes")
    if fixed is None:
        from .context import list_hash  # it imports this module

        return list_hash(expanded[(hour, hz)])
    return fixed[(hour, hz)]


def _fit_member(
    arrays: Dict,
    cfg: Dict,
    hour: int,
    hz: int,
    day: int,
    member,
    newest: int,
    test: int,
    list_id,
    predictors,
    scalable_mask,
    model,
    engine: Optional[StatsFitter] = None,
) -> Union[Tuple[float, list, Dict, Optional[str]], str]:
    """One training window's forecast of the task (`hour`, `hz`, day id `day`) issued from row `test`: (the prediction in target
    units, the coefficients, a matching list's record fields, the record's `detail`), or a skip reason. Its rows are
    `member_rows`: those of `training_span` with the member's window (`member_window`: `"expanding"` from the frame's first date),
    with the task's newest target. `model` fits it (`_fit_predict`): the task's own, from its inputs, never one of the process's.

    A per-row target (`config["row_target"]`, `models._row_target`) divides each training row's target by its `normalise_by`
    value, drops a row whose value is missing or not positive with the rows of missing values, and multiplies the forecast by
    the issue row's value, then clips it to `bounds`. After the skips of the training rows and of the issue row's inputs,
    `zero_when`'s 1 on the issue row is a forecast of 0 with no fit (detail `"zero_when"`) and a missing value
    `missing_test_predictors`; then an issue row's scale that is missing is `missing_test_predictors` and one not positive
    `non_positive_scale`. An offset (`ShapeModel`) subtracts each training row's from its target (a row it doesn't keep, or
    whose offset is missing, drops with the rows of missing values) and adds the issue row's to the forecast, a missing one
    being `missing_test_predictors`. Without one every step is as it was.

    `scalable_mask` None (`binary_mask="fit"`) is read here from those rows in the list's columns, before the matching and the
    finite filters, as `_fit_unit` reads a refit unit's: which columns are binary is then a function of this fit's span alone.

    `engine` (the engine's, `_engine_worker.EngineFits`) is handed the fit as selected here (`StatsFit`: the rows, filters, mask
    and skips are this function's) and fits it from shared statistics; where it hands the fit back (None), the scaler and
    `_fit_predict` below fit it, on the same values, as without it. `_fit_predict` runs in `fit_scope` of
    `FitUnit(hour, hz, the issue date, list_id, the member's place among config["members"])`, `member`, the list's hash and the
    training rows' day ids."""
    days = arrays["days"]
    train = member_rows(days, day, member, hz, newest)

    col_idx = [arrays["col_index"][c] for c in predictors]
    train_x = arrays["x"][train][:, col_idx]
    train_y = arrays["y"][train][:, hz - 1]
    row_target = cfg.get("row_target")
    if has_offset(row_target):
        # The learner's target, y less each row's offset, NaN on a row the offset's `keep` drops (`models._row_target`).
        train_y = learner_targets(arrays, row_target, hz, train, train_y)

    # No row dated inside the span, so nothing to train on for this (hour, horizon, day).
    if train_x.shape[0] == 0:
        return SKIP_TOO_FEW_ROWS
    if scalable_mask is None:
        # Before any filter: a filter only drops rows, and the mask is the span's, as a refit unit's is (`_fit_unit`).
        scalable_mask = StandardScaler.get_scalable_mask(train_x)

    test_x = arrays["x"][test : test + 1][:, col_idx]
    # A per-row target's scale: each training row's `normalise_by` value (None without one).
    scale_at = scale_column(arrays, row_target, hz)
    train_s = None if scale_at is None else arrays["x"][train, scale_at]

    # train_on="matching" filters what the span holds, so the set can't reach past it. The plan evaluated the function on these
    # days, so nothing about it is computed here.
    matching = cfg.get("matching")
    matching_days = None
    train_days = None
    if matching is not None and list_id in matching["ids"]:
        train_days = days[train]
        same = matching_rows(train_days, matching["lists"][(hour, hz)], matching["start"], list_id)
        train_x, train_y, train_days = train_x[same], train_y[same], train_days[same]
        if train_s is not None:
            train_s = train_s[same]
        if train_days.size == 0:
            return SKIP_NO_MATCHING_DAYS
        matching_days = matching_digest(train_days, matching["first_ordinal"])

    # Drop rows with NaN/inf in either predictors or target to avoid model fit failures.
    finite = finite_rows(train_x, train_y)
    if train_s is not None:
        finite &= usable_scale(train_s)  # a row with no positive scale drops out, as one with a missing input
    train_x = train_x[finite]
    train_y = train_y[finite]
    if train_s is not None:
        train_y = train_y / train_s[finite]  # the learner's target, (y - 0) / s of each row

    if train_x.shape[0] == 0:
        return SKIP_NO_FINITE_TRAINING_ROWS

    if not np.isfinite(test_x).all():
        return SKIP_MISSING_TEST_PREDICTORS

    extra = {}
    if matching_days is not None:
        extra["training_days"] = int(np.unique(train_days[finite]).size)
        extra["matching_days"] = matching_days
    issue_scale = None
    offset = None
    if row_target is not None:
        zero = zero_outcome(arrays, row_target, hz, test)
        if zero is None:
            return SKIP_MISSING_TEST_PREDICTORS
        if zero:
            return 0.0, [], extra, ZERO_WHEN  # no fit: `zero_when` says the forecast is 0, whatever its scale
        if scale_at is not None:
            issue_scale = float(arrays["x"][test, scale_at])
            skipped = _scale_skip(issue_scale)
            if skipped is not None:
                return skipped
        offset = issue_offset(arrays, row_target, hz, test)
        if offset is not None and not np.isfinite(offset):
            return SKIP_MISSING_TEST_PREDICTORS  # nothing to add the learner's forecast to
        if engine is not None:
            raise RuntimeError("a model with a per-row target runs on the model path: no engine fits it")
    if engine is not None:
        fitted = engine(StatsFit(hour, hz, train, col_idx, finite, train_x, train_y, scalable_mask, test_x))
        if fitted is not None and fitted[0] is not None:
            ((prediction, coefs),) = fitted
            return prediction, coefs, extra, None

    scaler = make_scaler(cfg.get("scaler", "standard"))
    train_x, train_y, test_x = scaler.fit_transform(
        train_x.copy(),
        train_y.copy(),
        test_x.copy(),
        scalable_mask=scalable_mask,
        columns=predictors,
    )
    issue_date = datetime.date.fromisoformat(cfg["test_start"]) + datetime.timedelta(days=day - cfg["offset"])
    members = cfg.get("members")
    member_index = 0 if members is None else list(members).index(member)  # a list's windows are distinct
    row_days = (days[train] if train_days is None else train_days)[finite]
    unit = FitUnit(hour, hz, issue_date, list_id, member_index)
    try:
        with fit_scope(unit, member, _unit_list_hash(cfg, list_id), row_days):
            pred, coefs = model._fit_predict(train_x, train_y, test_x)
    except ValueError as exc:
        # scikit-learn raises ValueError on degenerate input (e.g. fewer rows than CV folds); counted, not fatal.
        return f"{SKIP_FIT_ERROR}: {exc}"
    if row_target is not None:
        return scaled_back(row_target, scaler.inverse(float(pred)), issue_scale, offset), coefs, extra, None
    return scaler.inverse(float(pred)), coefs, extra, None


def member_mean(values: Any) -> float:
    """The mean of a forecast's members (seeds) in target units: `math.fsum` divided by their number, exactly rounded so their
    order can't change a bit; with an infinite member (an overflow, as `verify`'s control can cause) numpy's mean, NaN or
    infinite, where `fsum` would raise."""
    values = [float(value) for value in np.ravel(values)]
    try:
        return math.fsum(values) / len(values)
    except (ValueError, OverflowError):
        return float(np.mean(values))


def _tagged(reason: str, member) -> str:
    """A member's skip reason, with its window after the reason (`"no_finite_training_rows: window 56"`), before any detail. A
    missing issue row is the key's, whatever the window, and stays as it is."""
    if reason == SKIP_MISSING_TEST_ROW:
        return reason
    name, sep, detail = reason.partition(": ")
    return f"{name}: window {member}" + (f": {detail}" if sep else "")


def _combined(record: Dict, predictions: List[float]) -> Dict:
    """The record of a model with a list of training windows, from its first member's: the prediction is the members' mean in
    target units, `math.fsum` divided by their number (exactly rounded, so their order can't change a bit), `member_predictions`
    holds theirs in member order, and the coefficients are empty (decision 11 of the step 4 spec)."""
    out = {}
    for key, value in record.items():
        if key == "prediction":
            out["prediction"] = math.fsum(predictions) / len(predictions)
            out["member_predictions"] = [float(prediction) for prediction in predictions]
        elif key == "coefficients":
            out["coefficients"] = []
        else:
            out[key] = value
    return out


def _row_value(hour: int, hz: int, day_in_test: int, inputs: Optional[TaskInputs] = None) -> Union[Dict, str]:
    """Forecast one task of a model that fits nothing (`NaiveModel`): its one predictor's value on the row the forecast is
    issued from, the last row dated on the issue day, as it stands. No training row, scaler or mask is read, so a task needs
    no history. Returns the record, with no coefficients, or a skip reason, as `_fit_one_numpy` does. With `bounds`
    (`config["row_target"]`, the only per-row target such a model takes) the value is clipped to them."""
    inputs = _inputs_of(inputs)
    arrays = inputs.hour_arrays[hour]
    cfg = inputs.config
    day = cfg["offset"] + day_in_test
    window, day_in_window, position = task_place(cfg, day_in_test)
    test = issue_row(arrays["days"], day)
    if test is None:
        return SKIP_MISSING_TEST_ROW
    list_id, (name,), _ = task_list(cfg, inputs.expanded, hour, hz, position)
    value = float(arrays["x"][test, arrays["col_index"][name]])
    if not np.isfinite(value):
        return SKIP_MISSING_TEST_PREDICTORS
    row_target = cfg.get("row_target")
    if row_target is not None:
        if row_target["scale"] is not None or row_target["zero"] is not None or has_offset(row_target):
            # Refused before any task runs.
            raise RuntimeError("a model that fits nothing takes bounds alone of a per-row target")
        value = scaled_back(row_target, value, None)

    run_date = datetime.date.fromisoformat(cfg["test_start"]) + datetime.timedelta(days=day_in_test)
    return forecast_record(
        run_date,
        hour,
        hz,
        day_in_window,
        value,
        float(arrays["y"][test, hz - 1]),
        [],
        cfg.get("freq", "1h"),
        _list_hash_of(cfg, inputs.expanded, hour, hz, list_id),
        window=window,
        predictor_list=None if list_id is None else cfg["list_hashes"][list_id],
    )


def skip_record(
    run_date: datetime.date,
    hour: int,
    hz: int,
    day_in_test: int,
    actual: Optional[float],
    freq: str,
    window: Optional[str],
    outcome: str,
    list_hash: str,
    refit_anchor: str,
) -> Dict:
    """The record of a key that has no forecast (a skip record): the key fields a forecast's record has, in its order,
    `prediction` null, `actual` the issue row's target (null without the row), `coefficients` empty, then the skip reason as
    `status`, the forecast's `list_hash` and `refit_anchor`, `revision`, and `detail`, the text after the reason's ": " (such as
    "window 56"), when there is one. `BaseModel.run` writes them, for skips decided in the parent and in a worker alike."""
    status, _, detail = outcome.partition(": ")
    record = {
        "run_date": run_date.isoformat(),
        "target_date": (run_date + datetime.timedelta(days=hz)).isoformat(),
        "run_weekday": run_date.weekday(),
        "hour": hour,
        "horizon": hz,
        "day_in_test": day_in_test,
        "prediction": None,
        "actual": actual,
        "coefficients": [],
        "freq": freq,
    }
    if window is not None:
        record["window"] = window
    return with_status(record, status, list_hash, refit_anchor, detail or None)


def _run_task(
    task: tuple, inputs: Optional[TaskInputs] = None, engine: Optional[StatsFitter] = None
) -> Union[Dict, str]:
    inputs = _inputs_of(inputs)
    try:
        # A model that fits nothing has `config["fits"]` False and forecasts from the issue row alone.
        return _fit_one_numpy(*task, inputs, engine) if inputs.config.get("fits", True) else _row_value(*task, inputs)
    except Exception as exc:
        hour, hz, day_in_test = task
        raise TaskFailed(
            f"task (period={hour}, horizon={hz}, day_in_test={day_in_test}) raised {exc!r}\n{traceback.format_exc()}"
        ) from None


def _run_day(tasks: list, inputs: Optional[TaskInputs] = None) -> Tuple[List[Dict], List[Tuple[tuple, str]]]:
    """Run one test day's tasks, with `inputs` (the pool's if None). Returns (results, [(task, skip reason)]); unexpected errors
    raise TaskFailed."""
    inputs = _inputs_of(inputs)
    threads = inputs.config["threads_per_process"]
    results, skipped = [], []

    def collect(task, outcome):
        if isinstance(outcome, str):
            skipped.append((task, outcome))
        else:
            results.append(outcome)

    if len(tasks) <= 1 or threads == 1:
        for t in tasks:
            collect(t, _run_task(t, inputs))
        return results, skipped

    with ThreadPoolExecutor(max_workers=threads, initializer=_limit_openmp_in_task_thread) as pool:
        futures = {pool.submit(_run_task, t, inputs): t for t in tasks}
        for future in as_completed(futures):
            collect(futures[future], future.result())
    return results, skipped


def _unit_of(cfg: Dict, task: tuple, member: int = 0) -> FitUnit:
    """The fit a task of a model with `refit` forecasts from: its period and horizon, the anchor of its issue day under its
    member's rule (`config["refit"]["rules"]`, each member's with `"auto"`), its list, and the index of its training window among
    the model's (0 with one window). A day or pooled unit (`config["units"]`) fits every period at once: its period is None."""
    hour, hz, day_in_test = task
    task_lists = cfg.get("task_lists")
    list_id = None if task_lists is None else int(task_lists[(hour, hz)][task_place(cfg, day_in_test)[2]])
    run_date = datetime.date.fromisoformat(cfg["test_start"]) + datetime.timedelta(days=day_in_test)
    period = hour if cfg.get("units") is None else None
    return FitUnit(period, hz, refit_anchor(run_date, cfg["refit"]["rules"][member]), list_id, member)


def _units_of(cfg: Dict, task: tuple) -> List[FitUnit]:
    """The units a task forecasts from: one, or with a list of training windows one per member, in member order."""
    members = cfg.get("members")
    return [_unit_of(cfg, task)] if members is None else [_unit_of(cfg, task, member) for member in range(len(members))]


def _fit_unit(
    unit: FitUnit, tasks: List[tuple], inputs: Optional[TaskInputs] = None, engine: Optional[StatsFitter] = None
) -> List[Tuple[tuple, Union[Dict, str]]]:
    """Fit one unit of a model with `refit` and forecast each of `tasks` from it, one row at a time, with `inputs` (the pool's
    if None). Returns (task, record or skip reason) of each. `engine` fits the unit from shared statistics where it can, as in
    `_fit_member`: the rows, mask and skips are this function's, and a fit it hands back is fitted here as without it.

    The rows are those of a forecast issued on the anchor: `training_span(anchor, W, h, newest)` with the anchor's newest target,
    filtered as `_fit_one_numpy` filters a task's. Which columns are binary is read from these rows, before any filter, so a
    forecast's scaling depends on its anchor alone, never on where the run starts (the mask of a forecast with `refit=None` is
    read on the window's first day). The scaler is fitted on the training rows and transforms the test rows of `tasks` together,
    value by value, so a forecast is the same whichever other forecasts of its unit are run with it: one cut short and resumed
    equals a fresh one.

    A model whose class defines `_predict_block`, with `block_predict` True (`block_calls`, `config["block_predict"]`), predicts
    in one call the issue rows
    of every day of the unit's block that the period has with finite inputs, the run's tasks or not, padded to the rule's
    nominal length with copies of the first (`_block_forecasts`), and each task's forecast is its own row's: still the same
    whichever others are run with it, since the call's rows and shape don't depend on them. Tasks without a row or with a
    missing input are skipped as one at a time.

    A per-row target (`models._row_target`) is applied as `_fit_member` applies it: the training rows' targets divided by their
    scale, each forecast's `zero_when` (a forecast of 0, or a missing value's skip) and then its scale's skips (missing or not
    positive) decided per task, the others fitted together, scaled back by their own issue row's value and clipped; a unit
    whose forecasts are all 0 makes no fit, so a fit that would fail leaves them 0. An offset as in `_fit_member`: the training
    targets less each row's, each forecast plus its own issue row's.
    """
    inputs = _inputs_of(inputs)
    cfg = inputs.config
    model = inputs.model
    hour, hz = unit.period, unit.horizon
    arrays = inputs.hour_arrays[hour]
    days = arrays["days"]
    # The unit's training window: the model's, or with a list of them its member's; its newest target read on the anchor.
    train, newest = unit_rows(cfg, days, unit)

    predictors = inputs.expanded[(hour, hz) if unit.list_id is None else unit.list_id]
    col_idx = [arrays["col_index"][c] for c in predictors]
    train_x = arrays["x"][train][:, col_idx]
    train_y = arrays["y"][train][:, hz - 1]
    scalable_mask = StandardScaler.get_scalable_mask(train_x)
    row_target = cfg.get("row_target")
    if has_offset(row_target):
        # y less each row's offset, as `_fit_member`'s.
        train_y = learner_targets(arrays, row_target, hz, train, train_y)
    scale_at = scale_column(arrays, row_target, hz)
    train_s = None if scale_at is None else arrays["x"][train, scale_at]

    outcomes: List[Tuple[tuple, Union[Dict, str]]] = []
    issued = []
    for task in tasks:
        test = issue_row(days, cfg["offset"] + task[2])
        if test is None:
            outcomes.append((task, SKIP_MISSING_TEST_ROW))
        else:
            issued.append((task, test))

    def skip_all(reason: str) -> List[Tuple[tuple, Union[Dict, str]]]:
        return outcomes + [(task, reason) for task, _ in issued]

    if train_x.shape[0] == 0:
        return skip_all(SKIP_TOO_FEW_ROWS)

    matching = cfg.get("matching")
    matching_days = None
    train_days = None
    if matching is not None and unit.list_id in matching["ids"]:
        train_days = days[train]
        same = matching_rows(train_days, matching["lists"][(hour, hz)], matching["start"], unit.list_id)
        train_x, train_y, train_days = train_x[same], train_y[same], train_days[same]
        if train_s is not None:
            train_s = train_s[same]
        if train_days.size == 0:
            return skip_all(SKIP_NO_MATCHING_DAYS)
        matching_days = matching_digest(train_days, matching["first_ordinal"])

    finite = finite_rows(train_x, train_y)
    if train_s is not None:
        finite &= usable_scale(train_s)  # a row with no positive scale drops out, as one with a missing input
    train_x = train_x[finite]
    train_y = train_y[finite]
    if train_s is not None:
        train_y = train_y / train_s[finite]  # the learner's target, (y - 0) / s of each row
    if train_x.shape[0] == 0:
        return skip_all(SKIP_NO_FINITE_TRAINING_ROWS)

    forecast = []
    zeros = []  # (task, issue row) of the forecasts `zero_when` makes 0
    scales = {}  # issue row -> its scale, with `normalise_by`
    offsets = {}  # issue row -> its offset, with one
    for task, test in issued:
        if not np.isfinite(arrays["x"][test, col_idx]).all():
            outcomes.append((task, SKIP_MISSING_TEST_PREDICTORS))
        elif row_target is None:
            forecast.append((task, test))
        else:
            zero = zero_outcome(arrays, row_target, hz, test)
            if zero is None:
                outcomes.append((task, SKIP_MISSING_TEST_PREDICTORS))
                continue
            if zero:  # a forecast of 0, whatever its scale
                zeros.append((task, test))
                continue
            if scale_at is not None:
                scales[test] = float(arrays["x"][test, scale_at])
                skipped = _scale_skip(scales[test])
                if skipped is not None:
                    outcomes.append((task, skipped))
                    continue
            offset = issue_offset(arrays, row_target, hz, test)
            if offset is not None:
                if not np.isfinite(offset):
                    outcomes.append((task, SKIP_MISSING_TEST_PREDICTORS))
                    continue
                offsets[test] = offset
            forecast.append((task, test))
    extra = None
    if matching_days is not None:
        extra = {"training_days": int(np.unique(train_days[finite]).size), "matching_days": matching_days}

    def record_of(task: tuple, test: int, prediction: float, coefs: list, detail: Optional[str] = None) -> Dict:
        window, day_in_window, _ = task_place(cfg, task[2])
        run_date = datetime.date.fromisoformat(cfg["test_start"]) + datetime.timedelta(days=task[2])
        return forecast_record(
            run_date,
            hour,
            hz,
            day_in_window,
            prediction,
            float(arrays["y"][test, hz - 1]),
            coefs,
            cfg.get("freq", "1h"),
            _list_hash_of(cfg, inputs.expanded, hour, hz, unit.list_id),
            window=window,
            predictor_list=None if unit.list_id is None else cfg["list_hashes"][unit.list_id],
            # The newest target date of the unit's span, anchor + min(k, h-1), as `_fit_one_numpy` writes it for its day.
            newest_target_day=None
            if cfg["refit"]["newest"] is None
            else (unit.anchor + datetime.timedelta(days=min(newest, hz - 1))).isoformat(),
            refit_anchor=unit.anchor.isoformat(),
            extra=extra,
            detail=detail,
        )

    # No fit: `zero_when` says these forecasts are 0.
    outcomes.extend((task, record_of(task, test, 0.0, [], ZERO_WHEN)) for task, test in zeros)
    if not forecast:
        return outcomes

    test_rows = arrays["x"][[test for _, test in forecast]][:, col_idx]
    predicted = None
    if cfg.get("block_predict", False):
        # The block's rows in one call, each forecast from its own row (`_block_forecasts`); never the engine's. The parent's
        # model decided (`block_calls`).
        predicted = _block_forecasts(
            cfg, unit, arrays, col_idx, predictors, train_x, train_y, scalable_mask, forecast, model
        )
        if isinstance(predicted, str):  # the fit refused: every forecast's fit_error, as one at a time
            return outcomes + [(task, predicted) for task, _ in forecast]
    elif engine is not None:
        if row_target is not None:
            raise RuntimeError("a model with a per-row target runs on the model path: no engine fits it")
        predicted = engine(StatsFit(hour, hz, train, col_idx, finite, train_x, train_y, scalable_mask, test_rows))
    if predicted is None or any(row is None for row in predicted):
        # The whole unit, or the rows the engine left to the model path (each row's forecast is its own: scaled value by
        # value, one row at a time), on the model path; a fit it refuses is every row's fit_error, as without the engine.
        from_engine = predicted
        scaler = make_scaler(cfg.get("scaler", "standard"))
        train_x, train_y, test_x = scaler.fit_transform(
            train_x.copy(),
            train_y.copy(),
            test_rows,
            scalable_mask=scalable_mask,
            columns=predictors,
        )
        row_days = (days[train] if train_days is None else train_days)[finite]
        try:
            with fit_scope(unit, _window_of(cfg, unit), _unit_list_hash(cfg, unit.list_id), row_days):
                fitted = model._fit_model(train_x, train_y)
        except ValueError as exc:
            return outcomes + [(task, f"{SKIP_FIT_ERROR}: {exc}") for task, _ in forecast]
        predicted = []
        for row in range(len(forecast)):
            if from_engine is not None and from_engine[row] is not None:
                predicted.append(from_engine[row])
                continue
            # One row at a time (decision 16 of the step 3 spec): a product over several rows can differ in the last bits.
            pred, coefs = model._predict_fitted(fitted, test_x[row : row + 1])
            predicted.append((scaler.inverse(float(pred)), flat_coefficients(coefs, model, "_predict_fitted")))

    for (task, test), (prediction, coefs) in zip(forecast, predicted):
        if row_target is not None:
            prediction = scaled_back(row_target, prediction, scales.get(test), offsets.get(test))
        outcomes.append((task, record_of(task, test, prediction, coefs)))
    return outcomes


def block_calls(model) -> bool:
    """Whether `model`'s refit units predict their blocks in one call (`_fit_unit`, see `BaseModel`): it has `refit`,
    `block_predict` is True, and its class defines `_predict_block` in the class that defines `_predict_fitted`. A subclass
    that refits with an estimator of its own (`_fit_predict`, `_fit_model`, `_predict_fitted`) but inherits `_predict_block`
    would have its parent's block predictions made from its own fit, and one that overrides `_predict_block` alone would
    forecast otherwise than its `refit=None` run: each predicts one row at a time through `_predict_fitted` instead, as every
    other model. Read on the parent's model, which hands the answer to the workers in `config["block_predict"]`: a worker's
    model is rebuilt with refit=None from `_model_kwargs`, which may leave `block_predict` out."""
    cls = type(model)
    if getattr(cls, "_predict_block", None) is None or getattr(model, "refit", None) is None:
        return False
    if getattr(model, "_unit_kind", PERIOD) != PERIOD:
        return False  # a day or pooled unit always predicts its block in one call, on its own path (`_run_unit`)
    owners = {
        next(klass for klass in cls.__mro__ if name in vars(klass)) for name in ("_predict_block", "_predict_fitted")
    }
    return len(owners) == 1 and bool(getattr(model, "block_predict", False))


def block_rows(cfg: Dict, unit: FitUnit, days: np.ndarray, x: np.ndarray, col_idx: List[int]) -> List[int]:
    """The rows of a unit's block call before padding, in a period whose rows are on `days` with values `x`: the issue row
    (`issue_row`, so the second of a repeated autumn hour) of each day of the unit's block under its member's rule
    (`block_days`) that the period has a row on with every input of `col_idx` finite, in date order. Every such day, whichever
    of them the run forecasts: a call holds the same rows alone, in the full block, after a resume and whatever `test_start`
    or `issue_days`, and only the frame's end or a missing input leaves a day out. Its inputs only: no target of the block's
    days enters a call."""
    refit = cfg["refit"]
    rows = []
    for day in block_days(unit.anchor, refit["rules"][unit.member]):
        row = issue_row(days, day.toordinal() - refit["first_ordinal"])
        if row is not None and np.isfinite(x[row, col_idx]).all():
            rows.append(row)
    return rows


def _block_forecasts(
    cfg: Dict,
    unit: FitUnit,
    arrays: Dict,
    col_idx: List[int],
    predictors: List[str],
    train_x: np.ndarray,
    train_y: np.ndarray,
    scalable_mask: np.ndarray,
    forecast: List[Tuple[tuple, int]],
    model,
) -> Union[List[Tuple[float, list]], str]:
    """(prediction in target units, coefficients) of each of `forecast`'s (task, issue row), in its order, from one fit of the
    unit (`_fit_model`) and one call of `model._predict_block` over its block's rows (`block_rows`), padded to the rule's nominal
    length (`nominal_block_length`) with copies of the first, never another day's row: one shape for every call of the rule,
    whatever the frame, the run or a resume holds. The scaler is fitted on the training rows and transforms the call's rows
    together, value by value, as one at a time. A fit that raises ValueError is a `fit_error` reason, returned for every task."""
    rows = block_rows(cfg, unit, arrays["days"], arrays["x"], col_idx)
    length = nominal_block_length(cfg["refit"]["rules"][unit.member])
    if not rows or len(rows) > length:
        raise RuntimeError(f"a block call of {len(rows)} rows for a nominal length of {length}")
    call = rows + [rows[0]] * (length - len(rows))
    place = {row: i for i, row in enumerate(rows)}
    scaler = make_scaler(cfg.get("scaler", "standard"))
    train_x, train_y, call_x = scaler.fit_transform(
        train_x.copy(),
        train_y.copy(),
        arrays["x"][call][:, col_idx],
        scalable_mask=scalable_mask,
        columns=predictors,
    )
    try:
        with fit_scope(unit, _window_of(cfg, unit), _unit_list_hash(cfg, unit.list_id)):
            fitted = model._fit_model(train_x, train_y)
    except ValueError as exc:
        return f"{SKIP_FIT_ERROR}: {exc}"
    predictions, coefficients = model._predict_block(fitted, call_x)
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != (length,) or len(coefficients) != length:
        raise RuntimeError(
            f"{type(model).__name__}._predict_block returned {predictions.shape} predictions and {len(coefficients)} "
            f"coefficient lists for a call of {length} rows: one of each per row"
        )
    return [
        (
            scaler.inverse(float(predictions[place[test]])),
            flat_coefficients(coefficients[place[test]], model, "_predict_block"),
        )
        for _, test in forecast
    ]


_REAL = (int, float, np.integer, np.floating)


def flat_coefficients(coefs: Any, model: Any, method: str) -> list:
    """One forecast's coefficients as its record holds them, from what `model.<method>` returned for its row: a list of numbers,
    as every core model returns, unchanged; a tuple of them, or a 1-D numeric numpy array, as a list, numpy scalars as Python
    numbers, so the record writes as JSON. Anything else (a 2-D array, nested lists, None, strings, bools) is a TypeError
    naming the model, where a results file would otherwise fail later with a bare one."""
    if type(coefs) is list and all(type(value) is float for value in coefs):
        return coefs
    if isinstance(coefs, np.ndarray):
        if coefs.ndim == 1 and coefs.dtype.kind in "iuf":
            return coefs.tolist()
    elif isinstance(coefs, (list, tuple)) and all(
        isinstance(value, _REAL) and not isinstance(value, (bool, np.bool_)) for value in coefs
    ):
        return [value if type(value) in (int, float) else value.item() for value in coefs]
    raise TypeError(
        f"{type(model).__name__}.{method} returned coefficients {_described(coefs)} for a forecast: give a flat list of "
        "numbers (one per column, or [] for none) or a 1-D numeric array"
    )


def _described(coefs: Any) -> str:
    if isinstance(coefs, np.ndarray):
        return f"as a {coefs.ndim}-D {coefs.dtype} array of shape {coefs.shape}"
    return f"of type {type(coefs).__name__}: {coefs!r:.80}"


def _run_unit(
    unit: FitUnit, tasks: List[tuple], inputs: Optional[TaskInputs] = None, engine: Optional[StatsFitter] = None
) -> List[Tuple[tuple, Union[Dict, str]]]:
    """(task, record or skip reason, member) of each of a unit's tasks, from the fit of its kind (`config["units"]`): a period
    unit's `_fit_unit`, a day unit's `_fit_day_unit`, a pooled unit's `_fit_pooled_unit`. Any exception is a TaskFailed naming
    the unit."""
    try:
        kind = kind_of(_inputs_of(inputs).config)
        if kind == DAY:
            outcomes = _fit_day_unit(unit, tasks, inputs)
        elif kind == POOLED:
            outcomes = _fit_pooled_unit(unit, tasks, inputs)
        else:
            outcomes = _fit_unit(unit, tasks, inputs, engine)
        return [(task, outcome, unit.member) for task, outcome in outcomes]
    except Exception as exc:
        raise TaskFailed(
            f"fit unit (period={unit.period}, horizon={unit.horizon}, anchor={unit.anchor}, list={unit.list_id}, "
            f"member={unit.member}) of tasks {[task[2] for task in tasks]} raised {exc!r}\n{traceback.format_exc()}"
        ) from None


def _run_block(tasks: list, inputs: Optional[TaskInputs] = None) -> Tuple[List[Dict], List[Tuple[tuple, str]]]:
    """Run the tasks of one refit block, those whose issue days share an anchor, with `inputs` (the pool's if None): each unit's
    fit once, then its forecasts. Returns what `_run_day` does. Nothing carries over between units: each fits from scratch on its
    own rows.

    With a list of training windows a task has a unit per member, all in its block, and its record is made after every one of
    them ran: the members' mean if each forecast it, else the first failing member's reason (in member order), tagged with its
    window, as `_fit_one_numpy` does. With `refit="auto"` the block is one of the coarsest member's rule (`config["refit"]["rule"]`)
    and a finer member has a unit per anchor of its own in it; the record's `refit_anchor` is the block's, the oldest fit the
    forecast uses, and its `newest_target_day` the newest of its members'."""
    inputs = _inputs_of(inputs)
    cfg = inputs.config
    units = block_units(cfg, tasks)
    threads = cfg["threads_per_process"]
    if len(units) <= 1 or threads == 1:
        ran = [_run_unit(unit, unit_tasks, inputs) for unit, unit_tasks in units.items()]
    else:
        with ThreadPoolExecutor(max_workers=threads, initializer=_limit_openmp_in_task_thread) as pool:
            futures = [pool.submit(_run_unit, unit, unit_tasks, inputs) for unit, unit_tasks in units.items()]
            ran = [future.result() for future in as_completed(futures)]
    return block_records(cfg, ran)


def block_units(cfg: Dict, tasks: List[tuple]) -> Dict[FitUnit, List[tuple]]:
    """The units of a refit block's tasks (`_units_of`: one per task, or one per member), each with the tasks it forecasts, in
    the order of `tasks`. A unit's tasks are all of its period (of every period for a day or pooled unit, whose period is None),
    so a job that holds a block's tasks of one period (or of every period) holds each of its units whole."""
    units: Dict[FitUnit, List[tuple]] = {}
    for task in tasks:
        for unit in _units_of(cfg, task):
            units.setdefault(unit, []).append(task)
    return units


def block_records(
    cfg: Dict, ran: List[List[Tuple[tuple, Union[Dict, str], int]]]
) -> Tuple[List[Dict], List[Tuple[tuple, str]]]:
    """What `_run_block` returns, from what its units gave (`_run_unit` of each, in the order they came back): a task's record,
    or with a list of training windows the members' combination once every member's unit ran (`_combined`), else the first
    failing member's reason, tagged with its window."""
    members = cfg.get("members")
    by_task: Dict[tuple, Dict[int, Union[Dict, str]]] = {}
    results, skipped = [], []
    for outcomes in ran:
        for task, outcome, member in outcomes:
            if members is not None:
                by_task.setdefault(task, {})[member] = outcome
            elif isinstance(outcome, str):
                skipped.append((task, outcome))
            else:
                results.append(outcome)
    for task, outcomes in by_task.items():
        ordered = [outcomes[member] for member in range(len(members))]
        failed = next(((member, outcome) for member, outcome in enumerate(ordered) if isinstance(outcome, str)), None)
        if failed is not None:
            skipped.append((task, _tagged(failed[1], members[failed[0]])))
        else:
            results.append(
                _combined(_across_members(ordered, cfg["refit"]["rule"]), [record["prediction"] for record in ordered])
            )
    return results, skipped


def _across_members(records: List[Dict], block_rule: str) -> Dict:
    """The first member's record of a task of a refit model with a list of windows, with what differs between its members when
    their rules do (`refit="auto"`): `refit_anchor`, the block's anchor, which is the coarsest member's, and `newest_target_day`,
    the newest of the members'. With one rule for every member these are every member's, so the record is the first's."""
    first = records[0]
    anchor = refit_anchor(first["run_date"], block_rule).isoformat()
    if anchor == first["refit_anchor"] and all(record["refit_anchor"] == anchor for record in records):
        return first
    out = dict(first, refit_anchor=anchor)
    if "newest_target_day" in first:
        out["newest_target_day"] = max(record["newest_target_day"] for record in records)
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# Day and pooled units (`models._units`): one fit of every period of a day at once
# ---------------------------------------------------------------------------------------------------------------------------


def unit_list(cfg: Dict, expanded: Dict, unit: FitUnit) -> List[str]:
    """The names a day or pooled unit fits on: with a predictor function the plan's list `unit.list_id`, the only place a
    worker reads them from (nn_api found a day model given an empty list when lists came from two places), else the fixed list
    of its horizon, which is every period's (`_units.check_lists_by_period`)."""
    if cfg.get("task_lists") is not None:
        if unit.list_id is None:
            raise RuntimeError(f"a unit of a predictor function's plan has no list: {unit}")
        return list(expanded[unit.list_id])
    return list(expanded[(0, unit.horizon)])


def _unit_width(cfg: Dict, unit: FitUnit, anchor: int) -> int:
    """The int window of `training_span` of a unit anchored on day id `anchor`: the model's, or its member's."""
    members = cfg.get("members")
    return member_window(cfg["training_window"] if members is None else members[unit.member], anchor)


def _unit_newest(cfg: Dict, unit: FitUnit, periods: int) -> Optional[np.ndarray]:
    """k of each period on the unit's anchor, the newest target relative to it known by its issue time
    (`newest_target="published"`, read on the anchors, `config["refit"]["newest"]`); None with "d-1"."""
    newest_targets = cfg["refit"]["newest"]
    if newest_targets is None:
        return None
    ordinal = unit.anchor.toordinal()
    return np.array([int(newest_targets.at(period, ordinal)) for period in range(periods)], dtype=np.int64)


def _issued(inputs: TaskInputs, tasks: List[tuple]) -> Tuple[List[Tuple[tuple, str]], List[Tuple[tuple, int]]]:
    """(the tasks with no issue row, as `missing_test_row`; (task, its period's issue row) of the others), in task order."""
    outcomes, issued = [], []
    for task in tasks:
        test = issue_row(inputs.hour_arrays[task[0]]["days"], inputs.config["offset"] + task[2])
        if test is None:
            outcomes.append((task, SKIP_MISSING_TEST_ROW))
        else:
            issued.append((task, test))
    return outcomes, issued


def _unit_record(
    inputs: TaskInputs,
    unit: FitUnit,
    task: tuple,
    test: int,
    prediction: float,
    coefs: list,
    newest: Any,
    detail: Optional[str] = None,
) -> Dict:
    """The record of one forecast of a day or pooled unit, as `_fit_unit` writes a period unit's: the key, the prediction in
    target units, the `actual` of the period's issue row, the coefficients, `refit_anchor`, with `newest_target="published"`
    the newest target day of the period's slots, anchor + min(k, h - 1), and a `detail` where one is given (`"zero_when"`)."""
    cfg = inputs.config
    period, hz, day_in_test = task
    window, day_in_window, _ = task_place(cfg, day_in_test)
    run_date = datetime.date.fromisoformat(cfg["test_start"]) + datetime.timedelta(days=day_in_test)
    return forecast_record(
        run_date,
        period,
        hz,
        day_in_window,
        prediction,
        float(inputs.hour_arrays[period]["y"][test, hz - 1]),
        coefs,
        cfg.get("freq", "1h"),
        _list_hash_of(cfg, inputs.expanded, period, hz, unit.list_id),
        window=window,
        predictor_list=None if unit.list_id is None else cfg["list_hashes"][unit.list_id],
        newest_target_day=None
        if newest is None
        else (unit.anchor + datetime.timedelta(days=min(int(newest[period]), hz - 1))).isoformat(),
        refit_anchor=unit.anchor.isoformat(),
        detail=detail,
    )


def _fit_day_unit(
    unit: FitUnit, tasks: List[tuple], inputs: Optional[TaskInputs] = None
) -> List[Tuple[tuple, Union[Dict, str]]]:
    """Fit one day unit (`FitUnit(None, h, anchor, list, member)`) and forecast each of `tasks`, every period of its horizon
    issued in its block, from one call. Returns (task, record or skip reason) of each.

    Rows are the local dates of `training_span(anchor, W, h)`. Under `newest_target="published"` each period p has the W dates of
    its own period unit, `training_span(anchor, W, h, k(p))` with k(p) read on the anchor, and the rows are every date of any of
    them: a slot of p outside its own span is masked, so p trains on the dates its period unit would (the hour a spring switch
    skips counts as known, so its k can be another period's plus one, and a span of the largest k would cost every other
    period a date). A slot with no target is masked too (`_units.day_arrays`: the hour a spring switch skips, the repeated
    autumn hour, a missing value). X is each date's value of the list's day-level columns; a date with a missing input, or with no target to train on,
    drops out. The binary-column mask is read from the span's dates before any filter, the scaler (`config["scaler"]`) fitted on
    the training dates and applied to the call's rows value by value, and the targets go through `MultiTargetAsinh`, each
    period's Invariant fitted on its own training slots.

    One `_fit_days(x, y, y_mask)` and one `_predict_days(fitted, rows)` over the issue dates of every day of the block that the
    frame has with finite inputs, the run's tasks or not, in date order, padded to the rule's nominal length with copies of the
    first: the call has one shape and the same rows whatever the run, a resume, `test_start` or `issue_days` hold. Each forecast
    of (p, h, D) is column p of D's row, mapped back with p's Invariant; a model of several members (`TorchModel`'s seeds)
    returns (members, rows, P), and the forecast is the members' mean after each is mapped back (`math.fsum`). A task without its period's row on D is
    `missing_test_row`; a period with no training target `no_finite_training_rows`; a day with a missing input
    `missing_test_predictors`; a fit raising ValueError is each forecast's `fit_error`."""
    inputs = _inputs_of(inputs)
    cfg, model = inputs.config, inputs.model
    if cfg.get("row_target") is not None:  # refused before any fit (`BaseModel._check_row_target_support`)
        raise RuntimeError("a day unit takes no per-row target (normalise_by, bounds, zero_when)")
    periods = cfg["units"]["periods"]
    hz = unit.horizon
    refit = cfg["refit"]
    first_ordinal = refit["first_ordinal"]
    anchor = unit.anchor.toordinal() - first_ordinal
    newest = _unit_newest(cfg, unit, periods)
    width = _unit_width(cfg, unit, anchor)
    if newest is None:
        first, last = training_span(anchor, width, hz)
    else:
        # Each period's own span, the W dates of its period unit; the rows are every date of any of them.
        firsts, lasts = training_span(anchor, width, hz, newest)
        first, last = firsts.min(), lasts.max()
    first, last = int(first), int(last)
    predictors = unit_list(cfg, inputs.expanded, unit)
    has_row, train_x, train_y = day_arrays(inputs.hour_arrays, periods, predictors, first, last, hz)

    outcomes, issued = _issued(inputs, tasks)

    def skip_all(reason: str) -> List[Tuple[tuple, Union[Dict, str]]]:
        return outcomes + [(task, reason) for task, _ in issued]

    if not has_row.any():
        return skip_all(SKIP_TOO_FEW_ROWS)
    # Which columns are binary: from the span's dates, before any filter, as a period unit reads it from its rows.
    scalable_mask = StandardScaler.get_scalable_mask(train_x[has_row])
    valid = np.isfinite(train_y)
    if newest is not None:
        dates = np.arange(first, last + 1, dtype=np.int64)[:, None]
        valid &= (dates >= firsts[None, :]) & (dates <= lasts[None, :])
    keep = has_row & valid.any(axis=1)
    if train_x.shape[1]:
        keep &= np.isfinite(train_x).all(axis=1)
    if not keep.any():
        return skip_all(SKIP_NO_FINITE_TRAINING_ROWS)
    train_x, train_y, valid = train_x[keep], train_y[keep], valid[keep]
    trained = valid.any(axis=0)

    rule = refit["rules"][unit.member]
    block = block_days(unit.anchor, rule)
    block_first = block[0].toordinal() - first_ordinal
    has_day, block_x, _ = day_arrays(inputs.hour_arrays, periods, predictors, block_first, block_first + len(block) - 1)
    in_call = has_day & (np.isfinite(block_x).all(axis=1) if block_x.shape[1] else True)

    forecast = []
    for task, test in issued:
        place = cfg["offset"] + task[2] - block_first
        if not trained[task[0]]:
            outcomes.append((task, SKIP_NO_FINITE_TRAINING_ROWS))
        elif not in_call[place]:
            outcomes.append((task, SKIP_MISSING_TEST_PREDICTORS))
        else:
            forecast.append((task, test, place))
    if not forecast:
        return outcomes

    rows = [int(day) for day in np.flatnonzero(in_call)]
    length = nominal_block_length(rule)
    if len(rows) > length:
        raise RuntimeError(f"a block call of {len(rows)} days for a nominal length of {length}")
    call = rows + [rows[0]] * (length - len(rows))
    position = {row: i for i, row in enumerate(rows)}
    scaler = make_scaler(cfg.get("scaler", "standard"))
    # The scaler's inputs only: the targets go through each period's Invariant below.
    fit_x, _, call_x = scaler.fit_transform(
        train_x.copy(), np.zeros(len(train_x)), block_x[call].copy(), scalable_mask=scalable_mask, columns=predictors
    )
    target = MultiTargetAsinh().fit(train_y, valid)
    fit_y = np.where(valid, target.transform(train_y), np.nan)
    row_days = np.arange(first, last + 1, dtype=np.int64)[keep]  # one row per date
    try:
        with fit_scope(unit, _window_of(cfg, unit), _unit_list_hash(cfg, unit.list_id), row_days):
            fitted = model._fit_days(fit_x, fit_y, valid.copy())
    except ValueError as exc:
        return outcomes + [(task, f"{SKIP_FIT_ERROR}: {exc}") for task, _, _ in forecast]
    predicted = np.asarray(model._predict_days(fitted, call_x), dtype=np.float64)
    # (rows, P), or (members, rows, P) for a fit of several members (seeds) whose forecast is their mean in target units.
    if predicted.shape[-2:] != (length, periods) or predicted.ndim not in (2, 3) or not predicted.shape[0]:
        raise RuntimeError(
            f"{type(model).__name__}._predict_days returned an array of shape {predicted.shape} for a call of {length} rows "
            f"and {periods} periods: one row of {periods} per row of the call (and per member, before them)"
        )
    for task, test, place in forecast:
        period = task[0]
        if predicted.ndim == 3:
            # Each member mapped back with the period's Invariant, then their mean, exactly rounded (`_combined`'s rule).
            prediction = member_mean(target.inverse(predicted[:, position[place], period], period))
        else:
            prediction = float(target.inverse(predicted[position[place], period], period))
        outcomes.append((task, _unit_record(inputs, unit, task, test, prediction, [], newest)))
    return outcomes


def _fit_pooled_unit(
    unit: FitUnit, tasks: List[tuple], inputs: Optional[TaskInputs] = None
) -> List[Tuple[tuple, Union[Dict, str]]]:
    """Fit one pooled unit (`FitUnit(None, h, anchor, list, member)`) and forecast each of `tasks`, every period of its horizon
    issued in its block, from one call. Returns (task, record or skip reason) of each.

    Rows are every period's rows of `training_span(anchor, W, h, k(p))`, each period's as its period unit reads them (both rows
    of a repeated autumn hour, k(p) on the anchor under `newest_target="published"`), stacked by date, then period. Each
    predictor is read on its period's own row, and the virtual `period` is the row's period. Rows with a missing input or
    target drop out; the binary-column mask is read from the stacked rows before any filter, and the scaler
    (`config["scaler"]`) is fitted on the kept rows. The targets go through `MultiTargetAsinh`: each period's own Invariant
    (`target_transform="per_period"`) or one over every row (`"pooled"`).

    One `_fit_model` and one `_predict_block` over the issue row (the second of a repeated hour) of each period on each day of
    the block that the frame has with finite inputs, the run's tasks or not, by date then period, padded to the rule's nominal
    length times the periods of a day with copies of the first: one shape and the same rows whatever the run holds. A forecast is
    its own row's, mapped back with its period's Invariant (or the pooled one). Skips as for a day unit.

    A per-row target (`models._row_target`) as in `_fit_unit`: each stacked row's target divided by its own period row's
    `normalise_by` value (a row without a positive one drops out), each forecast's scale and `zero_when` read on its own issue row,
    a forecast mapped back with its period's Invariant, then multiplied by its issue row's scale and clipped; a unit whose
    forecasts are all 0 makes no fit. An offset as in `_fit_member`: each stacked row's target less its own period row's offset
    (a row it doesn't keep drops out), each forecast plus its own issue row's."""
    inputs = _inputs_of(inputs)
    cfg, model = inputs.config, inputs.model
    units = cfg["units"]
    periods = units["periods"]
    hz = unit.horizon
    refit = cfg["refit"]
    first_ordinal = refit["first_ordinal"]
    anchor = unit.anchor.toordinal() - first_ordinal
    newest = _unit_newest(cfg, unit, periods)
    width = _unit_width(cfg, unit, anchor)
    predictors = unit_list(cfg, inputs.expanded, unit)
    row_target = cfg.get("row_target")

    # The stacked rows exist once (W = 728 days of 96 periods and 200 columns are 107 MB a copy): a first pass reads each
    # period's rows, which of them are kept (every input and the target finite) and which columns are binary, and lets go of
    # them; a second places each period's kept rows straight into the stack. Which columns are binary is read from every row
    # before any filter, as a period unit reads it: a column is binary in the stack exactly when it is in each period's rows.
    spans, kept_rows, keys = [], [], []
    scalable_mask = np.zeros(len(predictors), dtype=bool)
    total = 0
    for period in range(periods):
        arrays = inputs.hour_arrays[period]
        days = arrays["days"]
        if newest is None:
            rows = training_rows(days, anchor, width, hz)
        else:
            rows = training_rows(days, anchor, width, hz, int(newest[period]))
        values = pooled_values(arrays, predictors, rows, period)
        targets = arrays["y"][rows][:, hz - 1]
        if has_offset(row_target):
            targets = learner_targets(arrays, row_target, hz, rows, targets)  # y less each row's offset
        kept = finite_rows(values, targets)
        scale_at = scale_column(arrays, row_target, hz)
        if scale_at is not None:
            kept &= usable_scale(arrays["x"][rows, scale_at])  # a row with no positive scale drops out
        scalable_mask |= StandardScaler.get_scalable_mask(values)
        total += len(values)
        del values
        spans.append(rows)
        kept_rows.append(kept)
        keys.append(days[rows][kept].astype(np.int64) * periods + period)

    outcomes, issued = _issued(inputs, tasks)

    def skip_all(reason: str) -> List[Tuple[tuple, Union[Dict, str]]]:
        return outcomes + [(task, reason) for task, _ in issued]

    if total == 0:
        return skip_all(SKIP_TOO_FEW_ROWS)
    keys = np.concatenate(keys)
    if not len(keys):
        return skip_all(SKIP_NO_FINITE_TRAINING_ROWS)
    # By date, then period; a repeated hour's two rows in their order (a stable sort).
    order = np.argsort(keys, kind="stable")
    position = np.empty(len(keys), dtype=np.int64)
    position[order] = np.arange(len(keys))
    train_x = np.empty((len(keys), len(predictors)))
    train_y = np.empty(len(keys))
    start = 0
    for period, (rows, kept) in enumerate(zip(spans, kept_rows)):
        arrays = inputs.hour_arrays[period]
        at = position[start : start + int(kept.sum())]
        train_x[at] = pooled_values(arrays, predictors, rows, period)[kept]
        if has_offset(row_target):
            train_y[at] = learner_targets(arrays, row_target, hz, rows, arrays["y"][rows][:, hz - 1])[kept]
        else:
            train_y[at] = arrays["y"][rows][kept, hz - 1]
        scale_at = scale_column(arrays, row_target, hz)
        if scale_at is not None:
            train_y[at] = train_y[at] / arrays["x"][rows, scale_at][kept]  # (y - 0) / s of each row
        start += len(at)
    groups = (keys[order] % periods).astype(np.int64)
    per_period = units["target_transform"] == PER_PERIOD
    # A period with no training row is skipped whatever the transform, so the forecast keys don't depend on it.
    trained = np.isin(np.arange(periods), groups)

    rule = refit["rules"][unit.member]
    call_rows: List[Tuple[int, int]] = []  # (period, issue row) of each block day's period the frame has, inputs finite
    values: List[np.ndarray] = []
    for day in block_days(unit.anchor, rule):
        day_id = day.toordinal() - first_ordinal
        for period in range(periods):
            arrays = inputs.hour_arrays[period]
            row = issue_row(arrays["days"], day_id)
            if row is None:
                continue
            found = pooled_values(arrays, predictors, [row], period)
            if np.isfinite(found).all():
                call_rows.append((period, row))
                values.append(found)

    place = {key: i for i, key in enumerate(call_rows)}
    forecast = []
    scales = {}  # (period, issue row) -> its scale, with `normalise_by`
    offsets = {}  # (period, issue row) -> its offset, with one
    for task, test in issued:
        if not trained[task[0]]:
            outcomes.append((task, SKIP_NO_FINITE_TRAINING_ROWS))
        elif (task[0], test) not in place:
            outcomes.append((task, SKIP_MISSING_TEST_PREDICTORS))
        elif row_target is None:
            forecast.append((task, test))
        else:
            own = inputs.hour_arrays[task[0]]
            zero = zero_outcome(own, row_target, hz, test)
            if zero is None:
                outcomes.append((task, SKIP_MISSING_TEST_PREDICTORS))
                continue
            if zero:  # no fit: `zero_when` says the forecast is 0, whatever its scale
                outcomes.append((task, _unit_record(inputs, unit, task, test, 0.0, [], newest, ZERO_WHEN)))
                continue
            scale_at = scale_column(own, row_target, hz)
            if scale_at is not None:
                scales[(task[0], test)] = float(own["x"][test, scale_at])
                skipped = _scale_skip(scales[(task[0], test)])
                if skipped is not None:
                    outcomes.append((task, skipped))
                    continue
            offset = issue_offset(own, row_target, hz, test)
            if offset is not None:
                if not np.isfinite(offset):
                    outcomes.append((task, SKIP_MISSING_TEST_PREDICTORS))
                    continue
                offsets[(task[0], test)] = offset
            forecast.append((task, test))
    if not forecast:
        return outcomes

    length = nominal_block_length(rule) * periods
    if len(call_rows) > length:
        raise RuntimeError(f"a block call of {len(call_rows)} rows for a nominal length of {length}")
    call = np.vstack(values + [values[0]] * (length - len(values)))
    scaler = make_scaler(cfg.get("scaler", "standard"))
    # Every scaler copies its input, which isn't read again: let go of it as soon as the scaled copy exists.
    fit_x, _, call_x = scaler.fit_transform(
        train_x, np.zeros(len(train_x)), call, scalable_mask=scalable_mask, columns=predictors
    )
    del train_x
    target_groups = groups if per_period else np.zeros(len(groups), dtype=np.int64)
    target = MultiTargetAsinh().fit_groups(train_y, target_groups, periods if per_period else 1)
    try:
        with fit_scope(unit, _window_of(cfg, unit), _unit_list_hash(cfg, unit.list_id)):
            fitted = model._fit_model(fit_x, target.transform(train_y, target_groups))
    except ValueError as exc:
        return outcomes + [(task, f"{SKIP_FIT_ERROR}: {exc}") for task, _ in forecast]
    predictions, coefficients = model._predict_block(fitted, call_x)
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != (length,) or len(coefficients) != length:
        raise RuntimeError(
            f"{type(model).__name__}._predict_block returned {predictions.shape} predictions and {len(coefficients)} "
            f"coefficient lists for a call of {length} rows: one of each per row"
        )
    for task, test in forecast:
        i = place[(task[0], test)]
        prediction = float(target.inverse(predictions[i], task[0] if per_period else 0))
        if row_target is not None:
            prediction = scaled_back(row_target, prediction, scales.get((task[0], test)), offsets.get((task[0], test)))
        coefs = flat_coefficients(coefficients[i], model, "_predict_block")
        outcomes.append((task, _unit_record(inputs, unit, task, test, prediction, coefs, newest)))
    return outcomes
