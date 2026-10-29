"""Native thread pools (BLAS, OpenMP) in the model worker pool, and forecasts that don't depend on the pool's size.

The pool runs MAX_PROCESSES x THREADS_PER_PROCESS tasks at once, so each task must run its BLAS and OpenMP code on
one thread; otherwise every task starts one thread per core and the run oversubscribes the machine. Environment
variables set inside a forked worker come too late for that (the libraries read them when they load), so the
worker limits the loaded libraries with threadpoolctl. The caller's own limits must stay as they were.
"""

import os

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from epftoolbox2.models import LassoCVModel, OLSModel, _worker
from epftoolbox2.models import base
from epftoolbox2.models.base import BaseModel
from tests.synthetic import PREDICTORS, make_market_data


def thread_counts():
    return {(lib["user_api"], lib["filepath"]): lib["num_threads"] for lib in threadpool_info()}


@pytest.fixture
def blas_loaded():
    if not any(api == "blas" for api, _ in thread_counts()):
        pytest.skip("threadpoolctl finds no BLAS library to limit")


class ThreadProbeModel(BaseModel):
    """Predicts 0 and reports, as its coefficients, the thread limit of every native library in the task's thread."""

    def _fit_predict(self, train_x, train_y, test_x):
        return 0.0, [float(n) for n in thread_counts().values()]


@pytest.mark.parametrize("threads_per_process", [1, 2])
def test_tasks_run_blas_and_openmp_on_one_thread_and_the_caller_keeps_its_limits(
    blas_loaded, monkeypatch, threads_per_process
):
    # Two task threads per process exercise the OpenMP limit, which is per thread, not per process.
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", str(threads_per_process))
    df = make_market_data(days=40)

    # A forked worker inherits the caller's limits; with 2 here, a worker that does not limit itself reports 2.
    with threadpool_limits(limits=2):
        before = thread_counts()
        ref = ThreadProbeModel(PREDICTORS, training_window=14).run(
            df, test_start="2023-02-01", test_end="2023-02-02", horizon=1
        )
        assert thread_counts() == before

    assert len(ref._results) == 48
    reported = {n for record in ref._results for n in record["coefficients"]}
    assert reported == {float(_worker.NATIVE_THREADS_PER_TASK)}


def test_worker_init_in_the_callers_process_leaves_its_limits_alone(blas_loaded):
    """Tests (and any in-process use) call _worker_init directly; only the pool initialiser limits threads."""
    before = thread_counts()
    hour_arrays = {0: {"x": np.zeros((3, 1)), "y": np.zeros((3, 1)), "days": np.arange(3), "col_index": {"a": 0}}}
    config = {"threads_per_process": 2, "scalable_masks": {}, "offset": 0}
    _worker._worker_init(hour_arrays, {(0, 1): ["a"]}, config, ThreadProbeModel, {})
    assert _worker._W_THREADPOOLS is None
    assert thread_counts() == before


def run_records(model_cls, monkeypatch, processes, threads, **params):
    monkeypatch.setenv("MAX_PROCESSES", str(processes))
    monkeypatch.setenv("THREADS_PER_PROCESS", str(threads))
    df = make_market_data(days=80, noise=5.0, seed=7)
    model = model_cls(PREDICTORS + ["price_d-1"], training_window=30, **params)
    ref = model.run(df, test_start="2023-03-10", test_end="2023-03-11", target="price", horizon=1)
    return {(r["run_date"], r["hour"], r["horizon"]): (r["prediction"], r["coefficients"]) for r in ref._results}


@pytest.mark.parametrize(("model_cls", "params"), [(OLSModel, {}), (LassoCVModel, {"cv": 3})], ids=["OLS", "LassoCV"])
def test_forecasts_are_bitwise_identical_with_one_or_two_processes(monkeypatch, model_cls, params):
    """Each task is fitted alone, so how tasks are spread over processes and threads can't change a forecast: not
    even in the last bit, which is why this compares with == rather than a tolerance.

    Both runs inherit the same BLAS limits, so this does not check the limit itself; the probe test above does.
    """
    one = run_records(model_cls, monkeypatch, processes=1, threads=1, **params)
    two = run_records(model_cls, monkeypatch, processes=2, threads=2, **params)

    assert len(one) == 24 * 2  # periods x run days: two days, so each of the two processes can take one
    assert one == two


def test_default_is_one_task_thread_per_process_and_one_process_per_usable_cpu(monkeypatch):
    monkeypatch.delenv("MAX_PROCESSES", raising=False)
    monkeypatch.delenv("THREADS_PER_PROCESS", raising=False)
    monkeypatch.setattr(base, "_usable_cpu_count", lambda: 8)
    model = OLSModel(PREDICTORS)
    assert model._executor_config() == (1, 8)

    monkeypatch.setenv("THREADS_PER_PROCESS", "4")
    assert model._executor_config() == (4, 2)
    monkeypatch.setenv("MAX_PROCESSES", "3")
    assert model._executor_config() == (4, 3)


def test_usable_cpu_count_is_at_most_the_machine_count():
    assert 1 <= base._usable_cpu_count() <= (os.cpu_count() or 1)
