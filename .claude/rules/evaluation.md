---
paths:
  - "epftoolbox2/evaluators/**"
  - "epftoolbox2/results/**"
  - "epftoolbox2/exporters/**"
  - "tests/test_evaluation.py"
  - "tests/test_rmse_rmae_csv.py"
  - "tests/results/**"
---

# Evaluation, results, exporters

- `Evaluator.compute(df, **kwargs)` gets one model's rows for one group plus `model_dfs`: every model's DataFrame for the same group. The frames are **keyed**: `run_date`, `target_date` (datetime64), `hour`, `horizon` (int), `prediction`, `actual` (`FORECAST_KEYS` in `evaluators/base.py`). A key that no record of a model carries is left out of its frame (records passed as plain lists may lack them), so evaluators that need keys must check. Relative metrics (rMAE) read their benchmark from `model_dfs` and must align on the keys. Existing evaluators ignore the extra columns.
- Every `EvaluationReport` path builds its frames with `_model_frames()` (one pass per model, compact dtypes: these can hold millions of rows) and cuts groups with `_grouped_table()`: `summary()`, `compute_all()` (single pass, used by exporters) and `_compute_grouped()` (the `by_*` methods) all go through them. A change in grouping or evaluator calling convention goes there; `tests/test_report_consistency.py` checks that the paths agree and that evaluators receive keyed frames from each.
- Records are written in the order tasks finish, which varies between runs of the same study, and float sums depend on the order of their terms. `_model_frames()` therefore sorts each model's rows by `RECORD_ORDER` (the forecast key `run_date, target_date, horizon, hour`, then `prediction`, `actual`, `day_in_test` for ties) before any evaluator sees them; `iter_details()` and `predictions()` use the same order. Anything that reads records for a number must go through that, not through file order. `tests/results/test_report_order.py` shuffles records and demands bitwise-identical tables and details and identical exporter files (the CSV and terminal checks are regression guards: they print no unrounded sums).
- Exporters only format what the report computed; they must not compute metrics themselves.
- Result records (JSONL) have keys `run_date, target_date, run_weekday, hour, horizon, day_in_test, prediction, actual, coefficients, freq`. Old result files must keep loading; add keys, don't rename them.

## Comparing models (research correctness)
Two models are only comparable on the **same forecast keys** `(run_date, target_date, hour, horizon)`. Models with different predictors can produce different key sets because workers skip tasks with missing inputs. Report how many forecasts each model contributed whenever results are compared.

`rMAEEvaluator` does this: both MAEs are taken on the keys the model and the benchmark share (finite prediction and actual on both sides), NaN if they share none, `ValueError` if the key columns are missing, null or duplicated. The `forecasts` column of `summary()` is still each model's own count.

In the EPF literature (Lago et al. 2021, the original epftoolbox) rMAE is usually relative to a naive seasonal benchmark, and significance is checked with Diebold–Mariano or Giacomini–White tests. Neither test exists here yet.
