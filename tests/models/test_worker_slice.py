"""The training/test slice in _worker._fit_one_numpy, tested in-process (the pool path hides it from coverage).

The slice is the information-set boundary: for a forecast issued on `day` at horizon `hz`, training
rows must have targets dated at most day-1, and the test row must be `day` itself.
"""

import numpy as np
import pytest

from epftoolbox2.models import _worker
from epftoolbox2.models.base import BaseModel

CAPTURED: list[dict] = []


class RecordingModel(BaseModel):
    """Records what the worker passes to the model and predicts 0."""

    def _fit_predict(self, train_x, train_y, test_x):
        CAPTURED.append({"train_x": train_x.copy(), "test_x": test_x.copy()})
        return 0.0, []


def init_worker(n_days: int, training_window: int, horizon: int, offset: int, threads: int = 1):
    days = np.arange(n_days)
    # Column 0 is the row's own day id; target column h-1 holds the target's day id (day + h).
    x = days.reshape(-1, 1).astype(float)
    y = np.stack([days + h for h in range(1, horizon + 1)], axis=1).astype(float)
    hour_arrays = {0: {"x": x, "y": y, "days": days, "timestamps": days, "col_index": {"day": 0}}}
    expanded = {(0, hz): ["day"] for hz in range(1, horizon + 1)}
    config = {
        "offset": offset,
        "test_start": "2024-01-01",
        "training_window": training_window,
        "scalable_masks": {(0, hz): np.array([False]) for hz in range(1, horizon + 1)},  # keep raw day ids
        "threads_per_process": threads,
        "freq": "1h",
    }
    _worker._worker_init(hour_arrays, expanded, config, RecordingModel, {})


@pytest.fixture(autouse=True)
def clean_capture():
    CAPTURED.clear()
    yield
    CAPTURED.clear()


@pytest.mark.parametrize("hz", [1, 2, 3])
def test_training_targets_end_the_day_before_the_forecast_is_issued(hz):
    init_worker(n_days=100, training_window=30, horizon=3, offset=60)

    result = _worker._fit_one_numpy(0, hz, day_in_test=5)

    issue_day = 65
    train_days = CAPTURED[0]["train_x"][:, 0]
    assert CAPTURED[0]["test_x"][0, 0] == issue_day
    assert train_days.max() + hz == issue_day - 1  # newest training target is dated issue_day - 1
    assert train_days.min() == issue_day - 30 - hz
    assert len(train_days) == 30
    assert result["actual"] == issue_day + hz  # the actual belongs to the labelled target
    assert result["run_date"] == "2024-01-06" and result["horizon"] == hz


def test_threaded_day_gives_the_same_results_as_sequential():
    tasks = [(0, hz, 3) for hz in (1, 2, 3)]

    init_worker(n_days=100, training_window=30, horizon=3, offset=60, threads=1)
    sequential, skipped_sequential = _worker._run_day(tasks)
    init_worker(n_days=100, training_window=30, horizon=3, offset=60, threads=4)
    threaded, skipped_threaded = _worker._run_day(tasks)

    by_horizon = lambda results: sorted(results, key=lambda r: r["horizon"])  # noqa: E731
    assert by_horizon(sequential) == by_horizon(threaded)
    assert [r["horizon"] for r in by_horizon(threaded)] == [1, 2, 3]
    assert skipped_sequential == skipped_threaded == []


def test_skipped_tasks_report_their_reason():
    init_worker(n_days=100, training_window=30, horizon=3, offset=60)
    # Day 0 has no earlier rows: nothing to train on.
    results, skipped = _worker._run_day([(0, 1, -60), (0, 1, 5)])
    assert [r["day_in_test"] for r in results] == [5]
    assert skipped == [((0, 1, -60), _worker.SKIP_TOO_FEW_ROWS)]
