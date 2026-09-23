"""The model's `scaler` reaches the worker, is fitted on training rows only, and every scaler runs end to end."""

import numpy as np
import pytest

from epftoolbox2.models import LassoCVModel, OLSModel, _worker
from epftoolbox2.models.base import BaseModel
from tests.synthetic import PREDICTORS, make_market_data

SCALERS = ["standard", "asinh", "npit", {"name": "asinh", "params": {"apply_to": ["price_d-*"]}}]
CAPTURED: list = []


class RecordingModel(BaseModel):
    """Records the scaled training target and returns the mean of the scaled training target."""

    def _fit_predict(self, train_x, train_y, test_x):
        CAPTURED.append(train_y.copy())
        return float(train_y.mean()), []


@pytest.fixture(autouse=True)
def clean_capture():
    CAPTURED.clear()
    yield
    CAPTURED.clear()


def init_worker(y: np.ndarray, scaler, offset: int = 60, training_window: int = 30):
    days = np.arange(len(y))
    x = np.column_stack([days.astype(float), np.cos(days)])
    hour_arrays = {0: {"x": x, "y": y.reshape(-1, 1), "days": days, "timestamps": days, "col_index": {"a": 0, "b": 1}}}
    config = {
        "offset": offset,
        "test_start": "2024-01-01",
        "training_window": training_window,
        "scalable_masks": {(0, 1): np.array([True, True])},
        "scaler": scaler,
        "threads_per_process": 1,
        "freq": "1h",
    }
    _worker._worker_init(hour_arrays, {(0, 1): ["a", "b"]}, config, RecordingModel, {})


@pytest.mark.parametrize("scaler", ["asinh", "npit"])
def test_worker_uses_the_configured_scaler(scaler):
    y = np.random.default_rng(0).normal(50, 20, 100)
    y[::9] = 400.0
    init_worker(y, "standard")
    standard = _worker._fit_one_numpy(0, 1, day_in_test=5)
    init_worker(y, scaler)
    vst = _worker._fit_one_numpy(0, 1, day_in_test=5)

    # The same scaled-space prediction (the mean) maps back differently: the VST's centre is not the mean.
    train_y = y[65 - 30 - 1 : 65 - 1]
    assert standard["prediction"] == pytest.approx(train_y.mean())
    assert vst["prediction"] != pytest.approx(train_y.mean(), rel=1e-3)
    assert train_y.min() <= vst["prediction"] <= train_y.max()


@pytest.mark.parametrize("scaler", ["standard", "asinh", "npit"])
def test_prices_from_the_issue_day_on_do_not_reach_the_scaler(scaler):
    y = np.random.default_rng(1).normal(50, 20, 100)
    init_worker(y, scaler)
    first = _worker._fit_one_numpy(0, 1, day_in_test=5)

    # Row d's target y[d] is the price of day d+1. Row 65 is the issue day D, so training targets end at y[63]
    # (the price of D-1); y[64:] are prices of D and later and must not move the fitted scaler or the forecast.
    scrambled = y.copy()
    scrambled[64:] = np.random.default_rng(2).normal(500, 300, len(y) - 64)
    init_worker(scrambled, scaler)
    second = _worker._fit_one_numpy(0, 1, day_in_test=5)

    np.testing.assert_array_equal(CAPTURED[0], CAPTURED[1])
    assert first["prediction"] == second["prediction"]
    assert first["actual"] != second["actual"]


@pytest.fixture
def single_process(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")


@pytest.mark.parametrize("model_cls", [OLSModel, LassoCVModel])
def test_every_scaler_runs_on_synthetic_data(model_cls, single_process):
    df = make_market_data(noise=5.0, seed=42)
    predictors = PREDICTORS + ["price_d-1"]
    kwargs = {"cv": 3} if model_cls is LassoCVModel else {}

    results = {}
    for scaler in SCALERS:
        model = model_cls(predictors, training_window=45, scaler=scaler, **kwargs)
        ref = model.run(df, test_start="2023-03-25", test_end="2023-03-31", target="price", horizon=2)
        results[str(scaler)] = {(r["hour"], r["horizon"], r["day_in_test"]): r for r in ref._results}

    keys = [set(r) for r in results.values()]
    assert all(k == keys[0] for k in keys), "scalers must forecast the same keys"
    assert len(keys[0]) > 0

    standard = results["standard"]
    for name, forecasts in results.items():
        predictions = np.array([f["prediction"] for f in forecasts.values()])
        errors = np.array([f["prediction"] - f["actual"] for f in forecasts.values()])
        errors = errors[np.isfinite(errors)]
        assert np.isfinite(predictions).all(), name
        assert np.abs(errors).mean() < 8.0, name  # noise sd is 5; the price itself has sd ~20
        if name != "standard":
            assert any(forecasts[k]["prediction"] != standard[k]["prediction"] for k in forecasts), name
