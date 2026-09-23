from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Union
import datetime
import os
import traceback

import numpy as np

from ..scalers.standard import StandardScaler

_W_HOUR_ARRAYS: Dict = {}
_W_EXPANDED_PREDS: Dict = {}
_W_CONFIG: Dict = {}
_W_FIT_PREDICT = None

# Reasons a task produces no forecast without being an error. They are counted per model and reported.
SKIP_TOO_FEW_ROWS = "too_few_rows"
SKIP_NO_FINITE_TRAINING_ROWS = "no_finite_training_rows"
SKIP_MISSING_TEST_PREDICTORS = "missing_test_predictors"
SKIP_FIT_ERROR = "fit_error"


class TaskFailed(RuntimeError):
    """An unexpected exception inside a worker task; carries the task identity and the worker's traceback."""


def _worker_init(hour_arrays, expanded_preds, config, model_cls, model_kwargs):
    global _W_HOUR_ARRAYS, _W_EXPANDED_PREDS, _W_CONFIG, _W_FIT_PREDICT
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
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

    col_idx = [arrays["col_index"][c] for c in _W_EXPANDED_PREDS[(hour, hz)]]
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

    scaler = StandardScaler()
    train_x, train_y, test_x = scaler.fit_transform(
        train_x.copy(),
        train_y.copy(),
        test_x.copy(),
        scalable_mask=cfg["scalable_masks"][(hour, hz)],
    )
    try:
        pred, coefs = _W_FIT_PREDICT(train_x, train_y, test_x)
    except ValueError as exc:
        # scikit-learn raises ValueError on degenerate input (e.g. fewer rows than CV folds); counted, not fatal.
        return f"{SKIP_FIT_ERROR}: {exc}"

    return {
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

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {pool.submit(_run_task, t): t for t in tasks}
        for future in as_completed(futures):
            collect(futures[future], future.result())
    return results, skipped
