"""No native helper threads in a model pool's workers, engine off and on, and the caller's thread counts put back.

OpenBLAS (with its own thread pool, the "pthreads" layer) stops that pool before every fork, and setting its thread count
in the child starts it again: one thread per CPU it counted when it loaded, for numpy's OpenBLAS and for scipy's, each
spinning for OpenBLAS's thread timeout before it sleeps, in every worker of every pool (62 per worker on 32 CPUs, and a fit
in a fresh worker cost 5-50 times its wall time in CPU). So the process that starts the pools holds its OpenBLAS at one
thread while they exist (`_worker.pool_native_threads`, once per `ModelPipeline.run`), the workers inherit that count and set
nothing on it, and the caller gets its counts back when the run's pools are done. Under spawn the thread variables do the
same for the libraries a worker loads.

These tests count threads instead of timing anything, since a spinning helper is a thread that exists. The `helpers_start`
fixture first sets each library to one thread in a forked process with no hold, to see which start threads there; with
none there is nothing to catch, and the tests skip, as they do, naming it, for a library that starts threads but isn't an
OpenBLAS with its own pool (which the hold doesn't cover). Linux only: they read /proc.
"""

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
import scipy.linalg
from threadpoolctl import ThreadpoolController, threadpool_info, threadpool_limits

from epftoolbox2.models import OLSModel, _engine_worker, _worker
from epftoolbox2.models.base import BaseModel
from epftoolbox2.pipelines import ModelPipeline
from tests.synthetic import PREDICTORS, make_market_data

pytestmark = pytest.mark.skipif(
    not os.path.isdir("/proc/self/task"), reason="counts a process's threads in /proc (Linux)"
)

# A thread's /proc entry can outlive its join by a moment, so the first count in a process waits up to this long for the
# count to come down. A helper that was started sleeps on, so with helpers the count never comes down.
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


def in_process(fn, *args, method="fork"):
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context(method)) as pool:
        return pool.submit(fn, *args).result()


def described(lib) -> str:
    return f"{lib.internal_api} ({getattr(lib, 'threading_layer', '?')}) {os.path.basename(lib.filepath)} {lib.version}"


def threads_started_by_each_library():
    """In a forked process, with no hold: each library set to one thread, and the threads that started."""
    started = []
    for lib in ThreadpoolController().lib_controllers:
        before = len(os.listdir("/proc/self/task"))
        if lib.num_threads != 1:
            lib.set_num_threads(1)
        started.append((described(lib), _worker._has_own_thread_pool(lib), len(os.listdir("/proc/self/task")) - before))
    return started


@pytest.fixture(scope="module")
def helpers_start():
    started = in_process(threads_started_by_each_library)
    if not any(n for _, _, n in started):
        names = "; ".join(name for name, _, _ in started) or "no native library"
        pytest.skip(f"no library starts threads in a forked process when set to one thread ({names})")
    uncovered = [name for name, own_pool, n in started if n and not own_pool]
    if uncovered:
        pytest.skip(f"these start threads in a forked process and are not an OpenBLAS with its own pool: {uncovered}")


def blas_work():
    """Bits of a few BLAS and LAPACK calls large enough for OpenBLAS to thread them at more than one thread."""
    a = np.random.default_rng(3).normal(size=(300, 300))
    return [(a @ a).tobytes(), scipy.linalg.lstsq(a, a[:, 0])[0].tobytes(), np.linalg.solve(a, a[:, 1]).tobytes()]


def limited():
    """In a pool's worker: its thread limit, then its threads, its libraries' limits, the bits of `blas_work` and its
    threads after that work."""
    _worker._limit_native_threads()
    threads = os_threads()
    limits = {lib["num_threads"] for lib in threadpool_info()}
    work = blas_work()
    return threads, limits, work, len(os.listdir("/proc/self/task"))


def test_a_worker_forked_under_the_hold_has_one_thread_and_the_same_numbers(helpers_start):
    with _worker.pool_native_threads():
        threads, limits, work, threads_after_work = in_process(limited)
    assert threads == 1
    assert limits == {_worker.NATIVE_THREADS_PER_TASK}
    assert threads_after_work == 1  # a BLAS call at one thread starts no pool
    with threadpool_limits(limits=1):
        assert work == blas_work()  # the same calls at one thread in this process


def own_pool_counts():
    return [lib.num_threads for lib in ThreadpoolController().lib_controllers if _worker._has_own_thread_pool(lib)]


def test_the_hold_puts_back_the_callers_counts_once_the_outermost_closes():
    if not own_pool_counts():
        pytest.skip("no OpenBLAS with its own thread pool is loaded")
    names = _worker.NATIVE_THREAD_ENV_VARS
    environment = {name: os.environ.get(name) for name in names}
    with threadpool_limits(limits=2):
        before = {lib["filepath"]: lib["num_threads"] for lib in threadpool_info()}
        with _worker.pool_native_threads():
            assert set(own_pool_counts()) == {_worker.NATIVE_THREADS_PER_TASK}
            with _worker.pool_native_threads():
                pass
            assert set(own_pool_counts()) == {_worker.NATIVE_THREADS_PER_TASK}  # the inner hold puts nothing back
            others = {
                lib.filepath: lib.num_threads
                for lib in ThreadpoolController().lib_controllers
                if not _worker._has_own_thread_pool(lib)
            }
            assert others == {path: n for path, n in before.items() if path in others}  # OpenMP, MKL, ...: untouched
            assert {name: os.environ.get(name) for name in names} == environment  # under fork, no variable is set
        assert {lib["filepath"]: lib["num_threads"] for lib in threadpool_info()} == before


def test_a_pipeline_run_holds_once_for_all_its_pools(monkeypatch):
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    calls = []
    hold, release = _worker._hold_native_threads, _worker._release_native_threads
    monkeypatch.setattr(_worker, "_hold_native_threads", lambda: (calls.append("hold"), hold())[1])
    monkeypatch.setattr(_worker, "_release_native_threads", lambda: (calls.append("release"), release())[1])
    df = make_market_data(days=40)
    pipeline = ModelPipeline()
    for name in ("one", "two", "three"):  # three pools
        pipeline.add_model(OLSModel(PREDICTORS, training_window=14, name=name))
    with threadpool_limits(limits=2):
        before = threadpool_info()
        pipeline.run(df, test_start="2023-02-01", test_end="2023-02-01", horizon=1)
        assert threadpool_info() == before
    assert calls == ["hold", "release"]


def threads_and_counts_after_import():
    """In a spawned process: its threads, once their number has held for 0.2 s, and each library's (api, layer, count),
    numpy, scipy and scikit-learn being loaded by unpickling this function, which imports this module."""
    threads = [len(os.listdir("/proc/self/task"))]
    while len(threads) < 3 or threads[-1] != threads[-3]:
        time.sleep(0.1)
        threads.append(len(os.listdir("/proc/self/task")))
    return threads[-1], [
        (lib["internal_api"], lib.get("threading_layer"), lib["num_threads"]) for lib in threadpool_info()
    ]


def test_spawned_workers_load_openblas_at_one_thread_under_the_hold(monkeypatch):
    """Where pools spawn their workers (Windows), the hold sets the thread variables, which a worker's OpenBLAS reads when it
    loads. Checked with a spawned process standing in for one, the hold told the pools spawn: it has every thread an unheld
    one has (an allocator's background thread, say) but OpenBLAS's helpers, one fewer than each library's count there."""
    unheld, unheld_libs = in_process(threads_and_counts_after_import, method="spawn")
    helpers = sum(n - 1 for api, layer, n in unheld_libs if api == "openblas" and layer not in ("openmp", "disabled"))
    if not helpers:
        pytest.skip(
            f"a spawned process's OpenBLAS starts no threads here (one CPU, or a thread variable set): {unheld_libs}"
        )
    monkeypatch.setattr(_worker, "_POOLS_SPAWN", True)
    environment = {name: os.environ.get(name) for name in _worker.NATIVE_THREAD_ENV_VARS}
    with _worker.pool_native_threads():
        held, held_libs = in_process(threads_and_counts_after_import, method="spawn")
    assert held == unheld - helpers
    assert {n for _, _, n in held_libs} == {_worker.NATIVE_THREADS_PER_TASK}
    assert {name: os.environ.get(name) for name in _worker.NATIVE_THREAD_ENV_VARS} == environment


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
