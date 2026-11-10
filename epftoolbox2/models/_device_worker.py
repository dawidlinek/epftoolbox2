"""A spawned process that hosts a run's GPU work, so the process that runs the pipeline never initialises CUDA.

A process that has initialised CUDA can't fork a worker that uses CUDA ("Cannot re-initialize CUDA in forked subprocess"), and
once it has, every later worker pool must spawn instead of fork (`_worker.pool_context`, the safety net), which loses whatever a
session changed on a class after importing it and can't run a class defined in `__main__` or a notebook. So a family whose
calls run on a GPU in the pipeline's own process (a foundation model; an in-context model run in this process) hosts its
objects in one spawned child per process instead:

- `shared().host(factory) -> key` builds an object in the child (`factory`, a picklable callable: a module-level function or a
  `functools.partial` of one, with picklable arguments), and returns the key it is kept under;
- `call(key, method, *args, **kwargs)` runs a method of it in the child and returns the result (arguments and results are
  pickled: numpy arrays, plain data, paths of files both processes see);
- `drop(key)` lets it go.

The child starts at the first `host`, is shared by every family of the process and lives as long as it does; it imports what
its factories import (torch, the family's package) and nothing else. An exception in the child is raised here; a child that
died (out of memory) is replaced at the next `host`, and the call that found it dead raises `RuntimeError` naming the cause.
"""

from __future__ import annotations

import itertools
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Dict, Optional

# The objects the child hosts, by key (in the child only).
_HOSTED: Dict[str, Any] = {}


def _host(key: str, factory: Callable[[], Any]) -> int:
    _HOSTED[key] = factory()
    return os.getpid()


def _call(key: str, method: str, args: tuple, kwargs: dict) -> Any:
    return getattr(_HOSTED[key], method)(*args, **kwargs)


def _drop(key: str) -> None:
    _HOSTED.pop(key, None)


class DeviceProcess:
    """One spawned child that hosts objects and runs their methods (see the module's docstring). `pid` is the child's."""

    def __init__(self) -> None:
        self._pool: Optional[ProcessPoolExecutor] = None
        self._lock = threading.Lock()
        self._keys = itertools.count()
        self.pid: Optional[int] = None

    def _run(self, function: Callable, *args: Any) -> Any:
        with self._lock:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
            pool = self._pool
        try:
            return pool.submit(function, *args).result()
        except BrokenProcessPool as error:
            with self._lock:
                self._pool = None
                self.pid = None
            raise RuntimeError(
                "the process that hosts the GPU work died (out of memory on the device, or killed); what it held is gone, "
                "and the next model starts a new one"
            ) from error

    def host(self, factory: Callable[[], Any]) -> str:
        key = f"hosted-{next(self._keys)}"
        self.pid = self._run(_host, key, factory)
        return key

    def call(self, key: str, method: str, *args: Any, **kwargs: Any) -> Any:
        return self._run(_call, key, method, args, kwargs)

    def drop(self, key: str) -> None:
        if self._pool is not None:
            try:
                self._run(_drop, key)
            except RuntimeError:
                pass

    def close(self) -> None:
        with self._lock:
            pool, self._pool, self.pid = self._pool, None, None
        if pool is not None:
            pool.shutdown(wait=True)


_SHARED: Optional[DeviceProcess] = None
_SHARED_LOCK = threading.Lock()


def shared() -> DeviceProcess:
    """This process's one `DeviceProcess`, made at the first call."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = DeviceProcess()
        return _SHARED


__all__ = ["DeviceProcess", "shared"]
