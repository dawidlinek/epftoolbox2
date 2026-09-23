"""ModelPipeline.report(): planned-vs-produced accounting and the missing-forecasts warning."""

import json
import warnings

import pytest

from epftoolbox2.models import OLSModel
from epftoolbox2.pipelines import ModelPipeline


def _row(hour, horizon, run_date):
    return {
        "run_date": run_date,
        "target_date": run_date,
        "run_weekday": 0,
        "hour": hour,
        "horizon": horizon,
        "day_in_test": 0,
        "prediction": 1.0,
        "actual": 1.0,
        "coefficients": [],
        "freq": "1h",
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_report_warns_when_a_model_has_missing_forecasts(tmp_path):
    """One test day, horizon=1, freq=1h: 24 forecasts are planned per model."""
    pipeline = (
        ModelPipeline().add_model(OLSModel(["price_d-1"], name="Full")).add_model(OLSModel(["price_d-1"], name="Short"))
    )
    _write_jsonl(tmp_path / "full.jsonl", [_row(h, 1, "2024-01-10") for h in range(24)])
    _write_jsonl(tmp_path / "short.jsonl", [_row(h, 1, "2024-01-10") for h in range(10)])  # 14 missing

    with pytest.warns(UserWarning, match="Short") as record:
        report = pipeline.report(tmp_path, test_start="2024-01-10", test_end="2024-01-10", horizon=1, freq="1h")

    message = str(record[0].message)
    assert "Full" not in message  # the complete model is not named
    assert "10" in message and "24" in message and "14" in message  # produced, planned, missing
    assert "different forecast sets" in message
    assert set(report.refs) == {"Full", "Short"}


def test_report_does_not_warn_when_every_model_is_complete(tmp_path):
    pipeline = ModelPipeline().add_model(OLSModel(["price_d-1"], name="Full"))
    _write_jsonl(tmp_path / "full.jsonl", [_row(h, 1, "2024-01-10") for h in range(24)])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipeline.report(tmp_path, test_start="2024-01-10", test_end="2024-01-10", horizon=1, freq="1h")
