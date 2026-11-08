from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import contextlib
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
from ._calendar import MASK_FIT, FitUnit, member_window, refit_anchor, training_span

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
# Whether the pools spawn their workers rather than fork them (`BaseModel._run_tasks`, `_engine.run_group`).
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

# In `PredictorPlan.day_lists`, next to list ids: a day no matching forecast of that (period, horizon) reads, and a day
# the function returned a list on that no forecast uses, which matches nothing.
NOT_PLANNED = -1
OTHER_LIST = -2


class TaskFailed(RuntimeError):
    """An unexpected exception inside a worker task; carries the task identity and the worker's traceback."""


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
    """Initialiser of the worker pool: limit native threads, then load the data and the model."""
    _limit_native_threads()
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
) -> Dict:
    """The record of a forecast, keys in their order: the key fields, `prediction`, `actual` (the issue row's target),
    `coefficients`, `freq`; a refit unit's `refit_anchor`; `newest_target_day` with `newest_target="published"`; a predictor
    function's `predictor_list`; a matching list's `extra` (`training_days`, `matching_days`); `window` in a run with windows;
    then D4's keys (`with_status`): `status` "ok", `list_hash`, `refit_anchor` (the issue day where the record has none: each
    forecast of a refit=None model is its own fit) and `revision`. `_combined` makes a list of training windows' from it."""
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
        prediction, coefs, extra = fitted
    else:
        outcomes = []
        for member in members:
            fitted = fit(member, task_mask(cfg, window, mask_of, member))
            if isinstance(fitted, str):
                return _tagged(fitted, member)
            outcomes.append(fitted)
        prediction, coefs, extra = None, [], {}

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
    )
    if members is not None:
        return _combined(record, [prediction for prediction, _, _ in outcomes])
    return record


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
) -> Union[Tuple[float, list, Dict], str]:
    """One training window's forecast of the task (`hour`, `hz`, day id `day`) issued from row `test`: (the prediction in target
    units, the coefficients, a matching list's record fields), or a skip reason. Its rows are `member_rows`: those of
    `training_span` with the member's window (`member_window`: `"expanding"` from the frame's first date), with the task's newest
    target. `model` fits it (`_fit_predict`): the task's own, from its inputs, never one of the process's.

    `scalable_mask` None (`binary_mask="fit"`) is read here from those rows in the list's columns, before the matching and the
    finite filters, as `_fit_unit` reads a refit unit's: which columns are binary is then a function of this fit's span alone.

    `engine` (the engine's, `_engine_worker.EngineFits`) is handed the fit as selected here (`StatsFit`: the rows, filters, mask
    and skips are this function's) and fits it from shared statistics; where it hands the fit back (None), the scaler and
    `_fit_predict` below fit it, on the same values, as without it."""
    days = arrays["days"]
    train = member_rows(days, day, member, hz, newest)

    col_idx = [arrays["col_index"][c] for c in predictors]
    train_x = arrays["x"][train][:, col_idx]
    train_y = arrays["y"][train][:, hz - 1]

    # No row dated inside the span, so nothing to train on for this (hour, horizon, day).
    if train_x.shape[0] == 0:
        return SKIP_TOO_FEW_ROWS
    if scalable_mask is None:
        # Before any filter: a filter only drops rows, and the mask is the span's, as a refit unit's is (`_fit_unit`).
        scalable_mask = StandardScaler.get_scalable_mask(train_x)

    test_x = arrays["x"][test : test + 1][:, col_idx]

    # train_on="matching" filters what the span holds, so the set can't reach past it. The plan evaluated the function on these
    # days, so nothing about it is computed here.
    matching = cfg.get("matching")
    matching_days = None
    train_days = None
    if matching is not None and list_id in matching["ids"]:
        train_days = days[train]
        same = matching_rows(train_days, matching["lists"][(hour, hz)], matching["start"], list_id)
        train_x, train_y, train_days = train_x[same], train_y[same], train_days[same]
        if train_days.size == 0:
            return SKIP_NO_MATCHING_DAYS
        matching_days = matching_digest(train_days, matching["first_ordinal"])

    # Drop rows with NaN/inf in either predictors or target to avoid model fit failures.
    finite = finite_rows(train_x, train_y)
    train_x = train_x[finite]
    train_y = train_y[finite]

    if train_x.shape[0] == 0:
        return SKIP_NO_FINITE_TRAINING_ROWS

    if not np.isfinite(test_x).all():
        return SKIP_MISSING_TEST_PREDICTORS

    extra = {}
    if matching_days is not None:
        extra["training_days"] = int(np.unique(train_days[finite]).size)
        extra["matching_days"] = matching_days
    if engine is not None:
        fitted = engine(StatsFit(hour, hz, train, col_idx, finite, train_x, train_y, scalable_mask, test_x))
        if fitted is not None and fitted[0] is not None:
            ((prediction, coefs),) = fitted
            return prediction, coefs, extra

    scaler = make_scaler(cfg.get("scaler", "standard"))
    train_x, train_y, test_x = scaler.fit_transform(
        train_x.copy(),
        train_y.copy(),
        test_x.copy(),
        scalable_mask=scalable_mask,
        columns=predictors,
    )
    try:
        pred, coefs = model._fit_predict(train_x, train_y, test_x)
    except ValueError as exc:
        # scikit-learn raises ValueError on degenerate input (e.g. fewer rows than CV folds); counted, not fatal.
        return f"{SKIP_FIT_ERROR}: {exc}"
    return scaler.inverse(float(pred)), coefs, extra


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
    no history. Returns the record, with no coefficients, or a skip reason, as `_fit_one_numpy` does."""
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
    the model's (0 with one window)."""
    hour, hz, day_in_test = task
    task_lists = cfg.get("task_lists")
    list_id = None if task_lists is None else int(task_lists[(hour, hz)][task_place(cfg, day_in_test)[2]])
    run_date = datetime.date.fromisoformat(cfg["test_start"]) + datetime.timedelta(days=day_in_test)
    return FitUnit(hour, hz, refit_anchor(run_date, cfg["refit"]["rules"][member]), list_id, member)


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
        if train_days.size == 0:
            return skip_all(SKIP_NO_MATCHING_DAYS)
        matching_days = matching_digest(train_days, matching["first_ordinal"])

    finite = finite_rows(train_x, train_y)
    train_x = train_x[finite]
    train_y = train_y[finite]
    if train_x.shape[0] == 0:
        return skip_all(SKIP_NO_FINITE_TRAINING_ROWS)

    forecast = []
    for task, test in issued:
        if np.isfinite(arrays["x"][test, col_idx]).all():
            forecast.append((task, test))
        else:
            outcomes.append((task, SKIP_MISSING_TEST_PREDICTORS))
    if not forecast:
        return outcomes

    test_rows = arrays["x"][[test for _, test in forecast]][:, col_idx]
    predicted = None
    if engine is not None:
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
        try:
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
            predicted.append((scaler.inverse(float(pred)), coefs))

    extra = None
    if matching_days is not None:
        extra = {"training_days": int(np.unique(train_days[finite]).size), "matching_days": matching_days}
    for (task, test), (prediction, coefs) in zip(forecast, predicted):
        window, day_in_window, _ = task_place(cfg, task[2])
        run_date = datetime.date.fromisoformat(cfg["test_start"]) + datetime.timedelta(days=task[2])
        record = forecast_record(
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
        )
        outcomes.append((task, record))
    return outcomes


def _run_unit(
    unit: FitUnit, tasks: List[tuple], inputs: Optional[TaskInputs] = None, engine: Optional[StatsFitter] = None
) -> List[Tuple[tuple, Union[Dict, str]]]:
    try:
        return [(task, outcome, unit.member) for task, outcome in _fit_unit(unit, tasks, inputs, engine)]
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
    the order of `tasks`. A unit's tasks are all of its period, so a job that holds a block's tasks of one period holds each of
    its units whole."""
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
