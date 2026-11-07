"""Pin current metric values on deterministic synthetic data.

A change in these numbers means model fitting, feature alignment or evaluation changed. That may be
intended (a fix) — then regenerate with `UPDATE_GOLDEN=1 uv run pytest tests/test_golden_metrics.py`
and say in the commit message why the numbers moved. The test window includes the spring DST switch
(2023-03-26) on purpose: a change to the dates of a forecast shows here.
"""

import json
import os
from pathlib import Path

import pytest

from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator, rMAEEvaluator
from epftoolbox2.models import LassoCVModel, LEARModel, OLSModel, RidgeModel
from epftoolbox2.pipelines import ModelPipeline
from tests.synthetic import PREDICTORS, make_market_data

GOLDEN_FILE = Path(__file__).parent / "golden" / "synthetic_metrics.json"
# The models of the forecasting layer (step 4) are pinned in a file of their own, so the file above stays as it was: Ridge and LEAR.
LEAR_RIDGE_GOLDEN_FILE = Path(__file__).parent / "golden" / "lear_ridge_metrics.json"

# OLS is exact up to float rounding. LassoCV stops at a convergence tolerance and the scikit-learn version differs
# between environments (the lock has 1.8; the 3.15 and newest-dependencies CI runs resolve newer ones), so it gets
# more room.
RTOL = {"OLS": 1e-9, "LassoCV": 1e-6}
# Ridge is a closed form (scikit-learn's leave-one-out `RidgeCV`), exact up to float rounding like OLS. LEAR's kernels run in numpy
# or numba, which agree within 1e-9 with the same kinks, not to the bit (decision 17 of the step 4 spec): the golden holds on either.
LEAR_RIDGE_RTOL = {"Ridge": 1e-9, "LEAR": 1e-9, "LEAR_lago": 1e-9}
# LEAR without `is_weekday`: with an intercept it is minus `is_weekend` after centring, an exact twin, which the fit drops anyway
# (`_lars.without_twins`); the inputs are as they were pinned.
LEAR_INPUTS = ["load_d+{horizon}", "wind_d+{horizon}", "is_weekend_d+{horizon}", "price_d-1"]


def compute_metrics() -> dict:
    df = make_market_data(noise=5.0, seed=42)
    predictors = PREDICTORS + ["price_d-1"]
    pipeline = (
        ModelPipeline()
        .add_model(OLSModel(predictors, training_window=45, name="OLS"))
        .add_model(LassoCVModel(predictors, training_window=45, cv=3, name="LassoCV"))
        .add_evaluator(MAEEvaluator())
        .add_evaluator(RMSEEvaluator())
        .add_evaluator(rMAEEvaluator(base_model="OLS"))
    )
    report = pipeline.run(df, test_start="2023-03-25", test_end="2023-03-27", target="price", horizon=2)
    tables = report.compute_all()

    metrics = {name: {"forecasts": len(details)} for name, details in report.iter_details()}
    for row in tables["summary"].to_dict("records"):
        metrics[row["model"]].update({k: float(row[k]) for k in ("MAE", "RMSE", "rMAE")})
    for row in tables["horizon"].to_dict("records"):
        metrics[row["model"]][f"MAE_h{row['horizon']}"] = float(row["MAE"])
    return metrics


def test_metrics_match_golden(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    actual = compute_metrics()

    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN_FILE.parent.mkdir(exist_ok=True)
        GOLDEN_FILE.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"golden values written to {GOLDEN_FILE}")

    expected = json.loads(GOLDEN_FILE.read_text())
    assert actual.keys() == expected.keys()
    for model, values in expected.items():
        assert actual[model]["forecasts"] == values["forecasts"], f"{model}: number of forecasts changed"
        for key, value in values.items():
            if key != "forecasts":
                assert actual[model][key] == pytest.approx(value, rel=RTOL[model]), f"{model} {key}"


def compute_lear_ridge_metrics() -> dict:
    """The golden test's data and window, with the models of step 4 alone, no benchmark to scale against: Ridge at W 45, LEAR with
    its defaults but the windows [30, 45] (BIC, the exact refit, asinh) and with `lago()`'s estimator (AIC, Lago's coordinate
    descent, the Invariant scaler)."""
    df = make_market_data(noise=5.0, seed=42)
    invariant = {"name": "asinh", "params": {"standardise": False}}
    pipeline = (
        ModelPipeline()
        .add_model(RidgeModel(PREDICTORS + ["price_d-1"], training_window=45, name="Ridge"))
        .add_model(LEARModel(LEAR_INPUTS, training_window=[30, 45], name="LEAR"))
        .add_model(LEARModel(LEAR_INPUTS, [30, 45], criterion="aic", solver="cd", scaler=invariant, name="LEAR_lago"))
        .add_evaluator(MAEEvaluator())
        .add_evaluator(RMSEEvaluator())
    )
    report = pipeline.run(df, test_start="2023-03-25", test_end="2023-03-27", target="price", horizon=2)
    tables = report.compute_all()

    metrics = {name: {"forecasts": len(details)} for name, details in report.iter_details()}
    for row in tables["summary"].to_dict("records"):
        metrics[row["model"]].update({k: float(row[k]) for k in ("MAE", "RMSE")})
    for row in tables["horizon"].to_dict("records"):
        metrics[row["model"]][f"MAE_h{row['horizon']}"] = float(row["MAE"])
    return metrics


def test_lear_ridge_metrics_match_golden(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    actual = compute_lear_ridge_metrics()

    if os.environ.get("UPDATE_GOLDEN") == "1":
        LEAR_RIDGE_GOLDEN_FILE.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"golden values written to {LEAR_RIDGE_GOLDEN_FILE}")

    expected = json.loads(LEAR_RIDGE_GOLDEN_FILE.read_text())
    assert actual.keys() == expected.keys()
    for model, values in expected.items():
        assert actual[model]["forecasts"] == values["forecasts"], f"{model}: number of forecasts changed"
        for key, value in values.items():
            if key != "forecasts":
                assert actual[model][key] == pytest.approx(value, rel=LEAR_RIDGE_RTOL[model]), f"{model} {key}"
