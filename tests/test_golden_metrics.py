"""Pin current metric values on deterministic synthetic data.

A change in these numbers means model fitting, feature alignment or evaluation changed. That may be
intended (a fix) — then regenerate with `UPDATE_GOLDEN=1 uv run pytest tests/test_golden_metrics.py`
and say in the commit message why the numbers moved. The test window includes the spring DST switch
(2023-03-26) on purpose, so fixing the DST issues in tests/test_known_issues.py will move them.
"""

import json
import os
from pathlib import Path

import pytest

from epftoolbox2.evaluators import MAEEvaluator, RMSEEvaluator, rMAEEvaluator
from epftoolbox2.models import LassoCVModel, OLSModel
from epftoolbox2.pipelines import ModelPipeline
from tests.synthetic import PREDICTORS, make_market_data

GOLDEN_FILE = Path(__file__).parent / "golden" / "synthetic_metrics.json"

# OLS is exact up to float rounding. LassoCV stops at a convergence tolerance and CI resolves a
# different scikit-learn per Python version (1.7 on 3.10, 1.8 on 3.11+), so it gets more room.
RTOL = {"OLS": 1e-9, "LassoCV": 1e-6}


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
