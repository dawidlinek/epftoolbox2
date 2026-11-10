"""The GPU host process (`models._device_worker`): it ends with a parent that is killed, even in the middle of a call; a death
is named from its exit code and its keys are gone with it; a script without a `__main__` guard is refused before spawning; a
forked process starts a host of its own. Objects hosted here are the standard library's (`threading.Event`, `dict`), so these
run without torch or a GPU.
"""

import faulthandler
import functools
import multiprocessing
import operator
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from epftoolbox2.models import _device_worker
from epftoolbox2.models._device_worker import DeviceProcess, HostDied

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals, fork and /proc")


def gone(pid):
    """Whether process `pid` has ended (a zombie waiting for its reaper counts)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].startswith("Z")
    except (OSError, IndexError):
        return False


def wait_until(condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.05)
    return condition()


@pytest.fixture
def host():
    process = DeviceProcess()
    yield process
    process.close()


BUSY_PARENT = """
import os
import threading

if __name__ == "__main__":
    from epftoolbox2.models._device_worker import DeviceProcess

    process = DeviceProcess(pdeathsig={pdeathsig})
    key = process.host(threading.Event)
    threading.Thread(target=process.call, args=(key, "wait", 120), daemon=True).start()
    print(os.getpid(), process.pid, flush=True)
    threading.Event().wait(120)
"""


@pytest.mark.parametrize("pdeathsig", [True, False], ids=["pdeathsig", "polling"])
def test_the_host_ends_with_a_killed_parent_even_in_the_middle_of_a_call(tmp_path, pdeathsig):
    """A pipeline SIGKILLed while its GPU host runs a two-minute call: the host ends within seconds (the kernel's signal on
    Linux, a watching thread elsewhere), instead of finishing the call as an orphan holding the GPU."""
    script = tmp_path / "busy_parent.py"
    script.write_text(textwrap.dedent(BUSY_PARENT.format(pdeathsig=pdeathsig)))
    parent = subprocess.Popen([sys.executable, str(script)], cwd=ROOT, stdout=subprocess.PIPE, text=True)
    child = None
    try:
        parent_pid, child = map(int, parent.stdout.readline().split())
        time.sleep(1.0)  # the call has reached the child
        assert not gone(child)
        os.kill(parent_pid, signal.SIGKILL)
        parent.wait(10)
        assert wait_until(lambda: gone(child)), "the GPU host outlived its SIGKILLed parent"
    finally:
        parent.kill()
        if child is not None and not gone(child):
            os.kill(child, signal.SIGKILL)


def test_a_death_is_named_from_its_exit_code_and_its_keys_go_with_it(host):
    """A SIGKILL (the OOM killer's way), a crash and an exit each say so; a call on a key of the dead child raises `HostDied`,
    never a KeyError, and the next `host` starts a new child. An exception in the child is raised as itself."""
    key = host.host(threading.Event)
    first = host.pid
    os.kill(first, signal.SIGKILL)
    assert wait_until(lambda: not host.holds(key))
    with pytest.raises(HostDied, match=r"killed \(SIGKILL\): most often by the kernel's out-of-memory killer"):
        host.call(key, "is_set")
    with pytest.raises(HostDied, match="stderr above"):
        host.call(key, "is_set")
    with pytest.raises(HostDied, match=r"crashed \(SIGSEGV\) in native code"):
        host.host(faulthandler._sigsegv)
    with pytest.raises(HostDied, match="exited with code 3"):
        host.host(functools.partial(os._exit, 3))
    again = host.host(threading.Event)
    assert host.pid not in (None, first) and host.call(again, "is_set") is False and host.generation == 4
    with pytest.raises(ZeroDivisionError) as raised:
        host.host(functools.partial(operator.truediv, 1, 0))
    assert any("in the GPU host process" in note for note in raised.value.__notes__)
    assert host.call(again, "is_set") is False  # an error is not a death


UNGUARDED = """
import threading

from epftoolbox2.models._device_worker import DeviceProcess

DeviceProcess().host(threading.Event)
print("hosted", flush=True)
"""
GUARDED = """
import threading

if __name__ == "__main__":
    from epftoolbox2.models._device_worker import DeviceProcess

    DeviceProcess().host(threading.Event)
    print("hosted", flush=True)
"""


def test_a_script_without_a_main_guard_is_refused_before_spawning(tmp_path):
    """Spawning re-runs the main script's body in the child: a script without `if __name__ == "__main__":` would run its
    pipeline there again. It is refused with a message saying so, before any child starts; the guarded script hosts."""
    unguarded = tmp_path / "unguarded.py"
    unguarded.write_text(textwrap.dedent(UNGUARDED))
    done = subprocess.run([sys.executable, str(unguarded)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode != 0 and "hosted" not in done.stdout
    assert 'without an `if __name__ == "__main__":` block' in done.stderr, done.stderr[-2000:]
    guarded = tmp_path / "guarded.py"
    guarded.write_text(textwrap.dedent(GUARDED))
    done = subprocess.run([sys.executable, str(guarded)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and "hosted" in done.stdout, done.stderr[-2000:]


def _host_in_a_fork(queue):
    signal.alarm(60)  # a hang ends the fork (exit code -14)
    try:
        process = _device_worker.shared()
        key = process.host(threading.Event)
        queue.put(("hosted", process.pid, process.call(key, "is_set")))
        process.close()
    except BaseException as error:
        queue.put(("error", type(error).__name__, str(error)[:300]))


def test_a_forked_process_starts_a_host_of_its_own(monkeypatch):
    """A process forked from one whose host runs (a later model's pool worker) never uses the parent's child or pipe: it starts
    its own, and the parent's still answers."""
    monkeypatch.setattr(_device_worker, "_SHARED", None)
    parent = _device_worker.shared()
    try:
        key = parent.host(threading.Event)
        context = multiprocessing.get_context("fork")
        queue = context.Queue()
        worker = context.Process(target=_host_in_a_fork, args=(queue,))
        worker.start()
        worker.join(90)
        if worker.is_alive():
            worker.kill()
        said = queue.get(timeout=10)
        assert said[0] == "hosted", said
        assert said[1] not in (None, parent.pid) and said[2] is False
        assert worker.exitcode == 0
        assert parent.call(key, "is_set") is False
    finally:
        parent.close()
