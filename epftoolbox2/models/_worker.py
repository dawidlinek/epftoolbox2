from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union
import datetime
import hashlib
import os
import traceback

import numpy as np
from threadpoolctl import ThreadpoolController

from ..scalers.registry import make_scaler
from ._calendar import training_span

_W_HOUR_ARRAYS: Dict = {}
_W_EXPANDED_PREDS: Dict = {}
_W_CONFIG: Dict = {}
_W_FIT_PREDICT = None

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
_W_THREADPOOLS: Optional[ThreadpoolController] = None
_W_NATIVE_LIMITS = None  # held for the worker's lifetime

# Reasons a task produces no forecast without being an error. They are counted per model and reported.
SKIP_TOO_FEW_ROWS = "too_few_rows"
SKIP_NO_FINITE_TRAINING_ROWS = "no_finite_training_rows"
SKIP_MISSING_TEST_PREDICTORS = "missing_test_predictors"
SKIP_FIT_ERROR = "fit_error"
SKIP_NO_MATCHING_DAYS = "no_matching_days"
# The issue day has no row for this period (a gap in the data); without the row there is nothing to forecast from.
SKIP_MISSING_TEST_ROW = "missing_test_row"

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
    global _W_THREADPOOLS, _W_NATIVE_LIMITS
    for name in NATIVE_THREAD_ENV_VARS:
        os.environ[name] = str(NATIVE_THREADS_PER_TASK)
    # Libraries already loaded ignore the variables: under fork that is everything the parent imported, under
    # spawn whatever importing this module loaded (numpy's BLAS at least). threadpoolctl sets them directly.
    _W_THREADPOOLS = ThreadpoolController()
    _W_NATIVE_LIMITS = _W_THREADPOOLS.limit(limits=NATIVE_THREADS_PER_TASK)


def _limit_openmp_in_task_thread() -> None:
    """The OpenMP limit is per thread (the BLAS ones are per process), so each task thread sets its own."""
    if _W_THREADPOOLS is not None:
        _W_THREADPOOLS.limit(limits=NATIVE_THREADS_PER_TASK, user_api="openmp")


def _pool_init(hour_arrays, expanded_preds, config, model_cls, model_kwargs):
    """Initialiser of the worker pool: limit native threads, then load the data and the model."""
    _limit_native_threads()
    _worker_init(hour_arrays, expanded_preds, config, model_cls, model_kwargs)


def _worker_init(hour_arrays, expanded_preds, config, model_cls, model_kwargs):
    global _W_HOUR_ARRAYS, _W_EXPANDED_PREDS, _W_CONFIG, _W_FIT_PREDICT
    _W_HOUR_ARRAYS = hour_arrays
    _W_EXPANDED_PREDS = expanded_preds
    _W_CONFIG = config
    _W_FIT_PREDICT = model_cls(predictors=[], **model_kwargs)._fit_predict


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


def _fit_one_numpy(hour: int, hz: int, day_in_test: int) -> Union[Dict, str]:
    """Fit and forecast one task.

    Returns the result record, or a skip reason (one of the SKIP_* names, optionally followed by ": detail")
    when no forecast is possible.

    `day_in_test` counts days from the run's first day. In a run with validation `run_days` says what that day is: its
    window, the days since the start of that window (which the record's `day_in_test` holds) and its place in the plan
    of a predictor function, which has no days between the windows. Each window has its own binary-column masks. Nothing else
    depends on the day's window, so a forecast is what a plain run over its window alone gives.
    """
    arrays = _W_HOUR_ARRAYS[hour]
    cfg = _W_CONFIG
    day = cfg["offset"] + day_in_test
    run_days = cfg.get("run_days")
    window, day_in_window, position = (None, day_in_test, day_in_test) if run_days is None else run_days[day_in_test]
    scalable_masks = cfg["scalable_masks"] if window is None else cfg["scalable_masks"][window]

    days = arrays["days"]
    # A model with newest_target="published" has the day of its newest target per issue day, read from the target's rule.
    newest_targets = cfg.get("newest")
    if newest_targets is None:
        newest = -1
        train = training_rows(days, day, cfg["training_window"], hz)
    else:
        newest = int(newest_targets[hour][position])
        train = training_rows(days, day, cfg["training_window"], hz, newest)
    test = issue_row(days, day)
    if test is None:
        return SKIP_MISSING_TEST_ROW

    # A fixed list is keyed by (period, horizon). With a predictor function of `ctx` (see epftoolbox2.models.context)
    # each task has its own list, and each list its own binary-column mask per (period, horizon). The mask of a matching
    # list is that list's too, not one per set of days, while the scaler below is fitted on what the filter leaves.
    task_lists = cfg.get("task_lists")
    if task_lists is None:
        list_id = None
        predictors = _W_EXPANDED_PREDS[(hour, hz)]
        scalable_mask = scalable_masks[(hour, hz)]
    else:
        list_id = int(task_lists[(hour, hz)][position])
        predictors = _W_EXPANDED_PREDS[list_id]
        scalable_mask = scalable_masks[(list_id, hour, hz)]

    col_idx = [arrays["col_index"][c] for c in predictors]
    train_x = arrays["x"][train][:, col_idx]
    train_y = arrays["y"][train][:, hz - 1]

    # No row dated inside the span, so nothing to train on for this (hour, horizon, day).
    if train_x.shape[0] == 0:
        return SKIP_TOO_FEW_ROWS

    test_x = arrays["x"][test : test + 1][:, col_idx]
    actual = float(arrays["y"][test, hz - 1])

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
    finite_rows = np.isfinite(train_y)
    if train_x.size:
        finite_rows &= np.isfinite(train_x).all(axis=1)
    train_x = train_x[finite_rows]
    train_y = train_y[finite_rows]

    if train_x.shape[0] == 0:
        return SKIP_NO_FINITE_TRAINING_ROWS

    if not np.isfinite(test_x).all():
        return SKIP_MISSING_TEST_PREDICTORS

    run_date = datetime.date.fromisoformat(cfg["test_start"]) + datetime.timedelta(days=day_in_test)
    run_date_str = run_date.isoformat()
    target_date_str = (run_date + datetime.timedelta(days=hz)).isoformat()

    scaler = make_scaler(cfg.get("scaler", "standard"))
    train_x, train_y, test_x = scaler.fit_transform(
        train_x.copy(),
        train_y.copy(),
        test_x.copy(),
        scalable_mask=scalable_mask,
        columns=predictors,
    )
    try:
        pred, coefs = _W_FIT_PREDICT(train_x, train_y, test_x)
    except ValueError as exc:
        # scikit-learn raises ValueError on degenerate input (e.g. fewer rows than CV folds); counted, not fatal.
        return f"{SKIP_FIT_ERROR}: {exc}"

    record = {
        "run_date": run_date_str,
        "target_date": target_date_str,
        "run_weekday": run_date.weekday(),
        "hour": hour,
        "horizon": hz,
        "day_in_test": day_in_window,
        "prediction": scaler.inverse(float(pred)),
        "actual": actual,
        "coefficients": coefs,
        "freq": cfg.get("freq", "1h"),
    }
    if newest_targets is not None:
        # The date of the newest target the span holds; a row is never dated `day` or later (`training_span`).
        record["newest_target_day"] = (run_date + datetime.timedelta(days=min(newest, hz - 1))).isoformat()
    if list_id is not None:
        record["predictor_list"] = cfg["list_hashes"][list_id]
    if matching_days is not None:
        record["training_days"] = int(np.unique(train_days[finite_rows]).size)
        record["matching_days"] = matching_days
    if window is not None:
        record["window"] = window
    return record


def _run_task(task: tuple) -> Union[Dict, str]:
    try:
        return _fit_one_numpy(*task)
    except Exception as exc:
        hour, hz, day_in_test = task
        raise TaskFailed(
            f"task (period={hour}, horizon={hz}, day_in_test={day_in_test}) raised {exc!r}\n{traceback.format_exc()}"
        ) from None


def _run_day(tasks: list) -> Tuple[List[Dict], List[Tuple[tuple, str]]]:
    """Run one test day's tasks. Returns (results, [(task, skip reason)]); unexpected errors raise TaskFailed."""
    threads = _W_CONFIG["threads_per_process"]
    results, skipped = [], []

    def collect(task, outcome):
        if isinstance(outcome, str):
            skipped.append((task, outcome))
        else:
            results.append(outcome)

    if len(tasks) <= 1 or threads == 1:
        for t in tasks:
            collect(t, _run_task(t))
        return results, skipped

    with ThreadPoolExecutor(max_workers=threads, initializer=_limit_openmp_in_task_thread) as pool:
        futures = {pool.submit(_run_task, t): t for t in tasks}
        for future in as_completed(futures):
            collect(futures[future], future.result())
    return results, skipped
