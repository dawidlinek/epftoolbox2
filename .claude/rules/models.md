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
- One fit per task `(period, horizon, run_day)`. Task identity for the worker is `(hour, horizon, day_in_test)`, but `ResultStore` resumes on `(hour, horizon, run_date)` — the absolute calendar date each record already carries — so a `test_start` change between runs into the same directory can't make an unrelated date look already done. `BaseModel.run(save_to=...)` also writes/checks a manifest (`epftoolbox2.results.manifest`) recording the rest of what defines a forecast (predictors and their per-`(period, horizon)` order, target, freq, training window, model params, the data used up to `test_end` and its time zone); resuming into results made with a different one of those raises `ManifestMismatchError` unless overridden. The information-set settings are recorded but not compared (they gate predictors, not what a fitted model predicts). Don't rename the JSONL record keys — old files must keep loading.
- Workers are a `ProcessPoolExecutor` (fork on Linux, spawn on Windows) initialised once with numpy arrays. The model is **rebuilt in each worker** as `model_cls(predictors=[], **self._model_kwargs)`. A new hyperparameter that is not returned from `_model_kwargs` silently falls back to its default inside workers — see `LassoCVModel` for the pattern.
- `_fit_predict(train_x, train_y, test_x)` receives standardised data and returns `(prediction_in_scaled_space, coefficients)`; the worker inverse-transforms the prediction.

## Information set (the leakage boundary)
In `_fit_one_numpy`, rows cover days `[day - training_window - hz, day]`; the last row is the forecast issued on `day`, and training drops the last `1 + hz` rows, so the newest training target is the price of `day - 1`. This is deliberately conservative (it does not assume the day-`day` auction result is known). Moving this boundary changes every published number — only do it when asked, and say so explicitly.

Whether a *predictor* is available — including a raw, unshifted (lag 0) column, such as `price` used to
forecast a different target — is a separate question, decided by `epftoolbox2.information_set` comparing
each column's publication time to `issue_time` (e.g. a day-ahead price is published ~13:00 D-1, so it may be
used from then on, same-day or older, but never for the day it prices). This convention is independent of
the training-target boundary above: don't conflate the two, and don't move either as a side effect of the
other.

## Scaling
`StandardScaler` is fitted per task on training rows only. Columns that are binary 0/1 are not scaled; the mask is computed once per `(period, horizon)` from the **first** test day's window (`_precompute_scalable_masks`) and reused for all days. The target is standardised, which is why `OLSModel` uses `fit_intercept=False` (centering plus one-hot dummies carry the level).

## Parallelism and determinism
- `MAX_PROCESSES` and `THREADS_PER_PROCESS` env vars control workers. In tests set both to `1` with `monkeypatch.setenv` — faster and deterministic.
- Models must be deterministic for a given input: stochastic models need a fixed `random_state` exposed as a parameter. `tests/test_golden_metrics.py` pins current numbers.

## Known issues (don't fix as a side effect of other work)
- The training cut assumes exactly one row per day in each period's frame, and DST breaks that for the switch hour (hour 2 in Europe/Warsaw). Both cases are pinned in `tests/test_known_issues.py`:
  - **Fall-back:** the doubled hour makes "drop the last `1 + hz` rows" drop one day too few. On the next `hz` run days, the hour-2 forecasts train on a target dated on or after the issue day, i.e. look-ahead.
  - **Spring:** the missing hour has no test row on the switch day. The hour-2 forecast is built from the previous day's row but labelled with the switch day's dates, so its `actual` belongs to a different target; for `hz=1` it's dropped.
- Forecasts disappear silently: `_fit_one_numpy` returns `None` on NaN predictors, too few rows, or `ValueError` from the fit; a crashing day is only `print`ed in `BaseModel.run`. `ModelResultRef.count` reports planned tasks, not produced ones.
- `_worker_init` sets `OMP_NUM_THREADS`/`MKL_NUM_THREADS` after fork, when BLAS is already initialised in the parent — probably no effect. `threadpoolctl` (installed with scikit-learn) is the reliable way.
