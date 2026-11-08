"""No native helper threads in a model pool's workers, engine off and on.

OpenBLAS's fork handler stops its thread pool in the parent before every fork, and the worker's thread limit (threadpoolctl's
`openblas_set_num_threads`) starts it again in the child: one thread per CPU the library counted when it loaded, for numpy's
OpenBLAS and for scipy's, each spinning for OpenBLAS's thread timeout before it sleeps, in every worker of every pool (62 per
worker on 32 CPUs, and a fit in a fresh worker cost 5-50 times its wall time in CPU). The worker stops them
(`_worker._stop_openblas_threads`). These tests count a worker's threads instead of timing it, since a spinning helper is a
thread that exists. The `helpers_start` fixture first shows, in a forked process limited without the stop, that this
environment's OpenBLAS starts helpers there; without that there is nothing to catch, and the tests skip. Linux only: they read
/proc.
"""

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
import scipy.linalg
from threadpoolctl import threadpool_info, threadpool_limits

from epftoolbox2.models import OLSModel, _engine_worker, _worker
from epftoolbox2.models.base import BaseModel
from epftoolbox2.pipelines import ModelPipeline
from tests.synthetic import PREDICTORS, make_market_data

pytestmark = pytest.mark.skipif(
    not os.path.isdir("/proc/self/task"), reason="counts a process's threads in /proc (Linux)"
)

# A stopped thread's /proc entry can outlive its join by a moment, so the first count in a process waits up to this long for
# the count to come down. A helper that was never stopped sleeps on, so without the stop the count never comes down.
SETTLE_SECONDS = 2.0
_settled_in = set()


def os_threads(allowed: int = 1) -> int:
    """This process's OS threads, the first time in a process after waiting up to SETTLE_SECONDS for at most `allowed`."""
    deadline = time.monotonic() + (0.0 if os.getpid() in _settled_in else SETTLE_SECONDS)
    _settled_in.add(os.getpid())
    while True:
        threads = len(os.listdir("/proc/self/task"))
        if threads <= allowed or time.monotonic() >= deadline:
            return threads
        time.sleep(0.01)


def in_forked_process(fn, *args):
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("fork")) as pool:
        return pool.submit(fn, *args).result()


def blas_work():
    """Bits of a few BLAS and LAPACK calls large enough for OpenBLAS to thread them at more than one thread."""
    a = np.random.default_rng(3).normal(size=(300, 300))
    return [(a @ a).tobytes(), scipy.linalg.lstsq(a, a[:, 0])[0].tobytes(), np.linalg.solve(a, a[:, 1]).tobytes()]


def limited(stop: bool):
    """In a forked process: the pool worker's thread limit (with the stop, or without it as before the fix); then its threads,
    its libraries' limits, the bits of `blas_work` and its threads after that work."""
    if not stop:
        _worker._stop_openblas_threads = lambda controller: None
    _worker._limit_native_threads()
    threads = os_threads()
    limits = {lib["num_threads"] for lib in threadpool_info()}
    work = blas_work()
    return threads, limits, work, len(os.listdir("/proc/self/task"))


@pytest.fixture(scope="module")
def helpers_start():
    if not any(
        lib["internal_api"] == "openblas" and lib.get("threading_layer") == "pthreads" for lib in threadpool_info()
    ):
        pytest.skip("no OpenBLAS with its own thread pool is loaded")
    threads, _, _, _ = in_forked_process(limited, False)
    if threads <= 1:
        pytest.skip(
            "this OpenBLAS starts no helper threads in a forked process (one CPU, or a thread variable set at start)"
        )


def test_the_worker_limit_leaves_one_thread_and_the_same_numbers(helpers_start):
    threads, limits, work, threads_after_work = in_forked_process(limited, True)
    assert threads == 1
    assert limits == {_worker.NATIVE_THREADS_PER_TASK}
    assert threads_after_work == 1  # a BLAS call at one thread does not start the pool again
    with threadpool_limits(limits=1):
        assert work == blas_work()  # the stop changes no bit: the same calls at one thread in this process


class ThreadCountModel(BaseModel):
    """Predicts 0 and reports, as its one coefficient, the OS threads of the process its task runs in."""

    allowed = 1

    def _fit_predict(self, train_x, train_y, test_x):
        return 0.0, [float(os_threads(self.allowed))]


@pytest.mark.parametrize("threads_per_process", [1, 2])
def test_pool_workers_run_on_their_task_threads_alone(helpers_start, monkeypatch, threads_per_process):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", str(threads_per_process))
    # The main thread, and with more than one task thread per process the job's task threads.
    allowed = 1 if threads_per_process == 1 else 1 + threads_per_process
    monkeypatch.setattr(ThreadCountModel, "allowed", allowed)
    df = make_market_data(days=40)
    ref = ThreadCountModel(PREDICTORS, training_window=14).run(
        df, test_start="2023-02-01", test_end="2023-02-02", horizon=1
    )
    assert len(ref._results) == 48
    assert max(record["coefficients"][0] for record in ref._results) <= allowed


@pytest.mark.parametrize("threads_per_process", [1, 2])
def test_engine_workers_run_on_their_task_threads_alone(helpers_start, monkeypatch, tmp_path, threads_per_process):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", str(threads_per_process))
    allowed = 1 if threads_per_process == 1 else 1 + threads_per_process
    run_task = _engine_worker._run_task

    def counting(*args, **kwargs):  # forked into the engine's worker, where `run_job` finds it
        outcome = run_task(*args, **kwargs)
        with open(tmp_path / f"{os.getpid()}.threads", "a") as f:
            f.write(f"{os_threads(allowed)}\n")
        return outcome

    monkeypatch.setattr(_engine_worker, "_run_task", counting)
    df = make_market_data(days=40)
    pipeline = ModelPipeline(engine="on").add_model(OLSModel(PREDICTORS, training_window=14))
    pipeline.run(df, test_start="2023-02-01", test_end="2023-02-02", horizon=1)

    files = list(tmp_path.glob("*.threads"))
    assert files and all(path.stem != str(os.getpid()) for path in files)  # the tasks ran in a pool's worker
    counts = [int(line) for path in files for line in path.read_text().split()]
    assert len(counts) == 48
    assert max(counts) <= allowed
