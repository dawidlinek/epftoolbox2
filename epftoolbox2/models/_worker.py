from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union
import datetime
import os
import traceback

import numpy as np
from threadpoolctl import ThreadpoolController

from ..scalers.registry import make_scaler

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


def _fit_one_numpy(hour: int, hz: int, day_in_test: int) -> Union[Dict, str]:
    """Fit and forecast one task.

    Returns the result record, or a skip reason (one of the SKIP_* names, optionally followed by ": detail")
    when no forecast is possible.
    """
    arrays = _W_HOUR_ARRAYS[hour]
    cfg = _W_CONFIG
    day = cfg["offset"] + day_in_test

    days = arrays["days"]
    day_min = day - cfg["training_window"] - hz
    mask = (days >= day_min) & (days <= day)

    # A fixed list is keyed by (period, horizon). With a predictor function of `ctx` (see epftoolbox2.models.context)
    # each task has its own list, and each list its own binary-column mask per (period, horizon).
    task_lists = cfg.get("task_lists")
    if task_lists is None:
        list_id = None
        predictors = _W_EXPANDED_PREDS[(hour, hz)]
        scalable_mask = cfg["scalable_masks"][(hour, hz)]
    else:
        list_id = int(task_lists[(hour, hz)][day_in_test])
        predictors = _W_EXPANDED_PREDS[list_id]
        scalable_mask = cfg["scalable_masks"][(list_id, hour, hz)]

    col_idx = [arrays["col_index"][c] for c in predictors]
    x_full = arrays["x"][mask][:, col_idx]
    y_full = arrays["y"][mask][:, hz - 1]

    # Not enough samples to build train/test slices for this (hour, horizon, day).
    if x_full.shape[0] <= (1 + hz):
        return SKIP_TOO_FEW_ROWS

    train_x = x_full[: -(1 + hz)]
    train_y = y_full[: -(1 + hz)]
    test_x = x_full[-1:]
    actual = float(arrays["y"][mask][-1, hz - 1])

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
        "day_in_test": day_in_test,
        "prediction": scaler.inverse(float(pred)),
        "actual": actual,
        "coefficients": coefs,
        "freq": cfg.get("freq", "1h"),
    }
    if list_id is not None:
        record["predictor_list"] = cfg["list_hashes"][list_id]
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
