---
paths:
  - "epftoolbox2/models/**"
  - "epftoolbox2/scalers/**"
  - "tests/models/**"
  - "tests/test_forecast_only.py"
---

# Models: how a run works and what must not change by accident

## Run structure (`BaseModel.run` → `_worker.py`)
- `_preprocess` adds `{target}_d+{h}` targets by shifting **calendar days in local time** (not 24h), then `day` ids from local dates and a `hour` period id (0–23, or 0–95 for `freq="15min"`).
- One fit per task `(period, horizon, run_day)`. Task identity `(hour, horizon, day_in_test)` is also the resume key in `ResultStore`; changing it (or the JSONL record keys) breaks resuming from existing result files.
- Workers are a `ProcessPoolExecutor` (fork on Linux, spawn on Windows) initialised once with numpy arrays. The model is **rebuilt in each worker** as `model_cls(predictors=[], **self._model_kwargs)`. A new hyperparameter that is not returned from `_model_kwargs` silently falls back to its default inside workers — see `LassoCVModel` for the pattern.
- `_fit_predict(train_x, train_y, test_x)` receives standardised data and returns `(prediction_in_scaled_space, coefficients)`; the worker inverse-transforms the prediction.

## Information set (the leakage boundary)
In `_fit_one_numpy`, rows cover days `[day - training_window - hz, day]`; the last row is the forecast issued on `day`, and training drops the last `1 + hz` rows, so the newest training target is the price of `day - 1`. This is deliberately conservative (it does not assume the day-`day` auction result is known). Moving this boundary changes every published number — only do it when asked, and say so explicitly.

## Scaling
`StandardScaler` is fitted per task on training rows only. Columns that are binary 0/1 are not scaled; the mask is computed once per `(period, horizon)` from the **first** test day's window (`_precompute_scalable_masks`) and reused for all days. The target is standardised, which is why `OLSModel` uses `fit_intercept=False` (centering plus one-hot dummies carry the level).

## Parallelism and determinism
- `MAX_PROCESSES` and `THREADS_PER_PROCESS` env vars control workers. In tests set both to `1` with `monkeypatch.setenv` — faster and deterministic.
- Models must be deterministic for a given input: stochastic models need a fixed `random_state` exposed as a parameter. `tests/test_golden_metrics.py` pins current numbers.

## Known issues (don't fix as a side effect of other work)
- Forecasts disappear silently: `_fit_one_numpy` returns `None` on NaN predictors, too few rows, or `ValueError` from the fit; a crashing day is only `print`ed in `BaseModel.run`. `ModelResultRef.count` reports planned tasks, not produced ones.
- `_worker_init` sets `OMP_NUM_THREADS`/`MKL_NUM_THREADS` after fork, when BLAS is already initialised in the parent — probably no effect. `threadpoolctl` (installed with scikit-learn) is the reliable way.
