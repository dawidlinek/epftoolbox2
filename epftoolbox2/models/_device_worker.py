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
- `drop(key)` lets it go; `holds(key)` says whether the child that hosts it is still the one running.

The child starts at the first `host`, is shared by every family of the process and lives as long as it does; it imports what
its factories import (torch, the family's package) and nothing else. An exception in the child is raised here, with the child's
traceback as a note. Its life:

- It ends with the process that started it, in the middle of a call too: on Linux the kernel tells it (`PR_SET_PDEATHSIG`,
  checked against `os.getppid()`, so a thread of the parent that ends doesn't stop it), elsewhere a thread of the child watches
  the parent; an idle child also ends when its pipe closes. A SIGKILLed pipeline leaves no child holding a GPU.
- A child that dies (host RAM, a crash in native code, a kill) is replaced at the next `host`: the call that found it dead raises
  `HostDied` naming the cause from its exit code, and every key of the dead child is gone (`holds` is False, `call` raises
  `HostDied`), so its users host their objects again.
- A process forked from this one (a later model's pool worker) starts a child of its own when it hosts anything: the parent's
  child and pipe are never used from a fork (`os.register_at_fork`).

Spawning re-runs the main script in the child unless it is guarded, so `host` refuses to start the child from a script without
an `if __name__ == "__main__":` block (`_check_main_guard`), before spawning.
"""

from __future__ import annotations

import ast
import itertools
import multiprocessing
import os
import signal
import sys
import threading
import traceback
from typing import Any, Callable, Dict, Optional

# The objects the child hosts, by key (in the child only).
_HOSTED: Dict[str, Any] = {}
_PR_SET_PDEATHSIG = 1  # linux/prctl.h


class HostDied(RuntimeError):
    """The process that hosts the GPU work died, or an object it hosted is gone with it."""


def _host(key: str, factory: Callable[[], Any]) -> int:
    _HOSTED[key] = factory()
    return os.getpid()


def _call(key: str, method: str, args: tuple, kwargs: dict) -> Any:
    return getattr(_HOSTED[key], method)(*args, **kwargs)


def _drop(key: str) -> None:
    _HOSTED.pop(key, None)


# --- the child --------------------------------------------------------------------------------------------------------


def _await_parent_signal(parent: int) -> None:
    """The kernel's word that the thread of the parent that started the child ended (`PR_SET_PDEATHSIG`): the child ends if
    that was the parent process itself, not one of its threads. A thread of its own, so a long call on the device doesn't hold
    it up."""
    while True:
        signal.sigwait({signal.SIGUSR1})
        if os.getppid() != parent:
            os._exit(0)


def _poll_parent(parent: int) -> None:
    """Where the kernel doesn't tell it: end the child once its parent is gone (its sentinel, else every second)."""
    from multiprocessing.connection import wait

    started_by = multiprocessing.parent_process()
    sentinels = [started_by.sentinel] if started_by is not None else []
    idle = threading.Event()
    while True:
        ready = wait(sentinels, timeout=1.0) if sentinels else idle.wait(1.0)
        if sys.platform == "win32" and ready:  # the parent's handle: signalled when it exits
            os._exit(0)
        if os.getppid() != parent:
            os._exit(0)
        if ready:  # the parent let the pipe go but runs: watch its pid alone
            sentinels = []


def _watch_parent(parent: int, pdeathsig: bool) -> None:
    """Make the child end with `parent`, even in the middle of a call: on Linux (`pdeathsig`) the kernel signals it
    (`PR_SET_PDEATHSIG`, SIGUSR1, blocked in every thread but one that waits for it), elsewhere a thread watches the parent."""
    armed = False
    if pdeathsig and sys.platform.startswith("linux"):
        try:
            import ctypes

            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})  # threads started later inherit the mask
            threading.Thread(
                target=_await_parent_signal, args=(parent,), daemon=True, name="epftoolbox2-parent-death"
            ).start()
            armed = ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, int(signal.SIGUSR1), 0, 0, 0) == 0
        except (OSError, AttributeError, ValueError):
            armed = False
    if not armed:
        threading.Thread(target=_poll_parent, args=(parent,), daemon=True, name="epftoolbox2-parent-watch").start()
    if os.getppid() != parent:  # it died before the watch was set
        os._exit(0)


def _portable(error: BaseException) -> BaseException:
    """`error` with the child's traceback as a note, or a RuntimeError of its text where it can't be pickled."""
    import pickle

    note = "in the GPU host process (pid {}):\n{}".format(os.getpid(), "".join(traceback.format_exception(error)))
    try:
        error.add_note(note)
        pickle.dumps(error)
        return error
    except Exception:
        return RuntimeError(f"{type(error).__name__}: {error}\n{note}")


def _serve(connection: Any, parent: int, pdeathsig: bool) -> None:
    """The child's loop: run each (function, args) the parent sends and send back (ok, result or error), until it sends None
    or its end of the pipe closes."""
    _watch_parent(parent, pdeathsig)
    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            return
        except BaseException as error:  # arguments it can't unpickle (a class it can't import): the parent hears why
            connection.send((False, _portable(error)))
            continue
        if message is None:
            return
        function, args = message
        try:
            reply = (True, function(*args))
        except BaseException as error:
            reply = (False, _portable(error))
        try:
            connection.send(reply)
        except Exception as error:  # a result it can't pickle
            connection.send(
                (False, _portable(RuntimeError(f"the result of {function.__name__} can't be sent: {error}")))
            )


# --- the parent -------------------------------------------------------------------------------------------------------


def _check_main_guard() -> None:
    """Refuse to spawn from a main script without an `if __name__ == "__main__":` block: spawning re-runs the script's body
    in the child (as `__mp_main__`), which would run the pipeline there again and fail as it starts its own child."""
    from multiprocessing import spawn

    prepared = spawn.get_preparation_data("epftoolbox2-gpu-host")
    name, path = prepared.get("init_main_from_name"), prepared.get("init_main_from_path")
    if name is not None:
        if name == "__main__" or name.endswith(".__main__"):
            return  # `python -m package`: its __main__ isn't re-run
        import importlib.util

        try:
            found = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            return
        path = getattr(found, "origin", None)
    if not path or not str(path).endswith(".py") or not os.path.isfile(path):
        return  # an interactive session, a notebook, `python -c`: nothing is re-run
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(path))
    except (OSError, SyntaxError, ValueError):
        return
    for node in tree.body:
        if isinstance(node, ast.If) and any(
            isinstance(part, ast.Name) and part.id == "__name__" for part in ast.walk(node.test)
        ):
            return
    raise RuntimeError(
        f'{path} runs GPU work without an `if __name__ == "__main__":` block: the GPU host process is spawned, and spawning '
        "re-runs the main script's body in it, which would run the pipeline there again. Put the script's work under "
        '`if __name__ == "__main__":` (or in a function called there).'
    )


def _death(pid: Optional[int], code: Optional[int]) -> str:
    """What a dead child's exit code says about why it died."""
    if code is None:
        cause = "stopped answering"
    elif code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        if name == "SIGKILL":
            cause = (
                "was killed (SIGKILL): most often by the kernel's out-of-memory killer when the machine's RAM ran out "
                "(`dmesg`, or the job's memory use, tells), else by a kill"
            )
        elif name in ("SIGSEGV", "SIGBUS", "SIGABRT", "SIGILL", "SIGFPE"):
            cause = f"crashed ({name}) in native code: torch, CUDA, the driver or the family's package"
        else:
            cause = f"was ended by {name}"
    else:
        cause = f"exited with code {code}: an error as it started or ran, which it printed on stderr"
    return (
        f"the process that hosts the GPU work (pid {pid}) {cause}. What it held is gone; the next model starts a new one. "
        "Its own message, if it printed one, is on stderr above. A device out-of-memory error is raised as itself, not as "
        "this; a child that can't start (a model class defined in `__main__` of `python -c`, a notebook or an interactive "
        'session, or a script without an `if __name__ == "__main__":` block) also ends like this.'
    )


# Bumped in a forked child of this process (`os.register_at_fork`): a `DeviceProcess` made before the fork is not the fork's.
_FORK_EPOCH = 0


class DeviceProcess:
    """One spawned child that hosts objects and runs their methods (see the module's docstring). `pid` is the child's, None
    before it starts or after it died; `generation` counts the children started, and a key names the one that hosts it."""

    def __init__(self, pdeathsig: bool = True) -> None:
        self._pdeathsig = pdeathsig
        self._process: Any = None
        self._connection: Any = None
        self._lock = threading.Lock()
        self._keys = itertools.count()
        self._epoch = _FORK_EPOCH
        self._last_death: Optional[str] = None
        self.environ: Dict[str, str] = {}  # the environment the running child started with (it read CUDA's then)
        self.generation = 0
        self.pid: Optional[int] = None

    def _forget(self) -> None:
        """Let go of the child without a word to it (it is another process's, or dead): close this side's end of the pipe."""
        connection, self._connection, self._process, self.pid = self._connection, None, None, None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def _settle(self) -> None:
        """With the lock: forget a child made before this process was forked (the parent's) or one that has died."""
        if self._epoch != _FORK_EPOCH:
            self._epoch = _FORK_EPOCH
            self._forget()
            self.generation += 1
        elif self._process is not None and not self._process.is_alive():
            self._process.join(timeout=10)
            self._last_death = _death(self._process.pid, self._process.exitcode)
            self._forget()

    def _start(self) -> None:
        _check_main_guard()
        context = multiprocessing.get_context("spawn")
        ours, theirs = context.Pipe(duplex=True)
        process = context.Process(
            target=_serve, args=(theirs, os.getpid(), self._pdeathsig), name="epftoolbox2-gpu-host", daemon=True
        )
        self.environ = dict(os.environ)
        process.start()
        theirs.close()
        self._process, self._connection, self.pid = process, ours, process.pid
        self.generation += 1

    def _exchange(self, function: Callable, args: tuple) -> Any:
        """With the lock and a running child: send one call and wait for its reply."""
        process = self._process
        try:
            self._connection.send((function, args))
            ok, value = self._connection.recv()
        except (EOFError, OSError) as error:
            process.join(timeout=10)
            self._last_death = _death(process.pid, process.exitcode)
            self._forget()
            raise HostDied(self._last_death) from error
        if not ok:
            raise value
        return value

    def _ours(self, key: str) -> bool:
        return self._process is not None and key.startswith(f"hosted-{self.generation}-")

    def holds(self, key: str) -> bool:
        """Whether `key` was hosted by the child running now, and that child is alive."""
        with self._lock:
            self._settle()
            return self._ours(key)

    def host(self, factory: Callable[[], Any]) -> str:
        with self._lock:
            self._settle()
            if self._process is None:
                self._start()
            key = f"hosted-{self.generation}-{next(self._keys)}"
            self._exchange(_host, (key, factory))
        return key

    def call(self, key: str, method: str, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self._settle()
            if not self._ours(key):
                raise HostDied(
                    f"{key} was hosted by a process that hosts the GPU work and has died or was replaced: host it again"
                    + (f" ({self._last_death})" if self._last_death else "")
                )
            return self._exchange(_call, (key, method, args, kwargs))

    def drop(self, key: str) -> None:
        with self._lock:
            self._settle()
            if self._ours(key):
                try:
                    self._exchange(_drop, (key,))
                except HostDied:
                    pass

    def close(self) -> None:
        with self._lock:
            self._settle()
            if self._process is not None:
                try:
                    self._connection.send(None)
                except OSError:
                    pass
                self._process.join(timeout=10)
            self._forget()


_SHARED: Optional[DeviceProcess] = None
_SHARED_LOCK = threading.Lock()


def shared() -> DeviceProcess:
    """This process's one `DeviceProcess`, made at the first call."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = DeviceProcess()
        return _SHARED


def _after_fork_in_child() -> None:
    """In a process forked from this one: the parent's child and pipe are the parent's (writing to that pipe would mix the
    two processes' calls, and a thread of the parent waiting on it doesn't exist here), so forget them and start afresh."""
    global _SHARED, _SHARED_LOCK, _FORK_EPOCH
    _FORK_EPOCH += 1
    if _SHARED is not None:
        _SHARED._lock = threading.Lock()  # a parent thread may have held it at the fork
        _SHARED._forget()
    _SHARED = None
    _SHARED_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


__all__ = ["DeviceProcess", "HostDied", "shared"]
