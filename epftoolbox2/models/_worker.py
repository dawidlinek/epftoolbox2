from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict
import datetime
import os

import numpy as np

from ..scalers.standard import StandardScaler

_W_HOUR_ARRAYS: Dict = {}
_W_EXPANDED_PREDS: Dict = {}
_W_CONFIG: Dict = {}
_W_FIT_PREDICT = None


def _worker_init(hour_arrays, expanded_preds, config, model_cls, model_kwargs):
    global _W_HOUR_ARRAYS, _W_EXPANDED_PREDS, _W_CONFIG, _W_FIT_PREDICT
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _W_HOUR_ARRAYS = hour_arrays
    _W_EXPANDED_PREDS = expanded_preds
    _W_CONFIG = config
    _W_FIT_PREDICT = model_cls(predictors=[], **model_kwargs)._fit_predict


def _fit_one_numpy(hour: int, hz: int, day_in_test: int) -> Dict:
    arrays = _W_HOUR_ARRAYS[hour]
    cfg = _W_CONFIG
    day = cfg["offset"] + day_in_test

    days = arrays["days"]
    day_min = day - cfg["training_window"] - hz
    mask = (days >= day_min) & (days <= day)

    col_idx = [arrays["col_index"][c] for c in _W_EXPANDED_PREDS[(hour, hz)]]
    x_full = arrays["x"][mask][:, col_idx]
    y_full = arrays["y"][mask][:, hz - 1]
    train_x = x_full[: -(1 + hz)]
    train_y = y_full[: -(1 + hz)]
    test_x  = x_full[-1:]
    actual  = float(arrays["y"][mask][-1, hz - 1])

    ts = arrays["timestamps"][mask][-1]
    run_date_str    = str(ts.astype("datetime64[D]"))
    target_date_str = str((ts + np.timedelta64(hz, "D")).astype("datetime64[D]"))
    run_weekday     = datetime.date.fromisoformat(run_date_str).weekday()

    scaler = StandardScaler()
    train_x, train_y, test_x = scaler.fit_transform(
        train_x.copy(), train_y.copy(), test_x.copy(),
        scalable_mask=cfg["scalable_masks"][(hour, hz)],
    )
    pred, coefs = _W_FIT_PREDICT(train_x, train_y, test_x)

    return {
        "run_date":     run_date_str,
        "target_date":  target_date_str,
        "run_weekday":  run_weekday,
        "hour":         hour,
        "horizon":      hz,
        "day_in_test":  day_in_test,
        "prediction":   scaler.inverse(float(pred)),
        "actual":       actual,
        "coefficients": coefs,
    }


def _run_day(tasks: list) -> list:
    threads = _W_CONFIG["threads_per_process"]
    if len(tasks) <= 1 or threads == 1:
        return [_fit_one_numpy(*t) for t in tasks]

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {pool.submit(_fit_one_numpy, *t): t for t in tasks}
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                t = futures[future]
                print(f"Error on task (hour={t[0]}, hz={t[1]}, day={t[2]}): {e}")
        return results
