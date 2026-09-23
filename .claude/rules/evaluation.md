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

- `Evaluator.compute(df, **kwargs)` gets one model's rows for one group (`prediction`, `actual`) plus `model_dfs`: every model's DataFrame for the same group. Relative metrics (rMAE) read their benchmark from `model_dfs`.
- Grouped metrics are computed in **two** places: `EvaluationReport.compute_all()` (single pass, used by exporters) and `_compute_grouped()` (the `by_*` methods). A change in grouping or evaluator calling convention must be made in both; `tests/test_report_consistency.py` checks that they agree.
- Exporters only format what the report computed; they must not compute metrics themselves.
- Result records (JSONL) have keys `run_date, target_date, run_weekday, hour, horizon, day_in_test, prediction, actual, coefficients, freq`. Old result files must keep loading; add keys, don't rename them.

## Comparing models (research correctness)
Two models are only comparable on the **same forecast keys** `(run_date, hour, horizon)`. Models with different predictors can produce different key sets because workers skip tasks with missing inputs. Report how many forecasts each model contributed whenever results are compared.

In the EPF literature (Lago et al. 2021, the original epftoolbox) rMAE is usually relative to a naive seasonal benchmark, and significance is checked with Diebold–Mariano or Giacomini–White tests. Neither test exists here yet.

## Known issues (don't fix as a side effect of other work)
- `rMAEEvaluator` divides MAEs taken from each model's own row set, not aligned on keys (`report.py` builds per-model DataFrames; `rmae.py` compares their means). Pinned by an `xfail(strict=True)` test in `tests/test_known_issues.py`.
