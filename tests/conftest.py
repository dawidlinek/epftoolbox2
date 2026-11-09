import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def entsoe_api_key():
    """
    Get ENTSOE API key from environment variable.

    To run integration tests, set: ENTSOE_API_KEY=your-key-here
    """
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        pytest.skip("ENTSOE_API_KEY environment variable not set")
    return api_key


# Variables that Workflow.run() sets on the whole process.
PROCESS_ENV = (
    "PYTHON_GIL",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLAS_NUM_THREADS",
)


@pytest.fixture(autouse=True)
def restore_process_env():
    """Give every test the process environment it started with.

    `Workflow.run()` sets these for the whole process, so a test that runs a workflow would otherwise hand its settings
    to whichever test the same process runs next: with a random order, or with pytest-xdist reusing a worker for the
    next test, that is a dependence on what ran before.
    """
    names = (*PROCESS_ENV, "MAX_PROCESSES", "THREADS_PER_PROCESS")
    before = {name: os.environ.get(name) for name in names}
    yield
    for name, value in before.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture(autouse=True)
def no_kept_tables():
    """Start every test with none of the tables the information set keeps for the whole process (the verdicts `violations`
    reached, `_VERDICTS`, and the newest training targets, `_NEWEST_TABLES`): with pytest-xdist reusing a worker, a test would
    otherwise read what an earlier test computed, and one that counts what is computed would count less. The same for the
    estimator checks the engine's fits keep (`models._stats._ESTIMATOR_CHECKS`)."""
    from epftoolbox2 import information_set
    from epftoolbox2.models import _stats

    for tables, lock in (
        (information_set._VERDICTS, information_set._VERDICTS_LOCK),
        (information_set._NEWEST_TABLES, information_set._NEWEST_TABLES_LOCK),
    ):
        with lock:
            tables.clear()
    _stats._ESTIMATOR_CHECKS.clear()


@pytest.fixture(autouse=True)
def no_kept_hashes():
    """Start every test with no weight file hashes kept (`models._weights._HASHES`, kept per process by path, inode, size and times):
    a worker reused for another test must hash its files itself."""
    from epftoolbox2.models import _weights

    _weights._HASHES.clear()


class NetworkAttempts:
    """The attempts `no_network` refused, in this process and in every process forked from it: each is a line appended to
    `log`, so a worker's attempt reaches the test even when the worker caught the error. Iterates over (what, detail)."""

    def __init__(self, log):
        self.log = log
        self.log.write_text("")

    def add(self, what: str, detail) -> None:
        line = f"{what}\t{detail!r}\t{os.getpid()}\n".encode()
        handle = os.open(self.log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        try:
            os.write(handle, line)  # one write in append mode: lines of several processes don't interleave
        finally:
            os.close(handle)

    def read(self):
        return [tuple(line.split("\t")[:2]) for line in self.log.read_text().splitlines()]

    def clear(self) -> None:
        self.log.write_text("")

    def __iter__(self):
        return iter(self.read())

    def __len__(self) -> int:
        return len(self.read())

    def __eq__(self, other) -> bool:
        return self.read() == list(other)


@pytest.fixture
def no_network(monkeypatch, tmp_path_factory):
    """Fail the test on any attempt to reach a network: `connect`, `connect_ex`, `sendto` or `sendmsg` on a socket that isn't
    a Unix one (a process pool's own sockets are), `socket.create_connection`, or a name or address lookup (`getaddrinfo`,
    `gethostbyname`, `gethostbyname_ex`, `gethostbyaddr`, `getnameinfo`). Each attempt raises OSError and is logged to a file
    (`NetworkAttempts`), and the test fails at teardown if there was one, also when the code under test caught the error and
    when it was made in a forked worker, which inherits the guard. Yields the attempts (a test of the guard clears them)."""
    import socket

    attempts = NetworkAttempts(tmp_path_factory.mktemp("network") / "attempts.log")
    real = {name: getattr(socket.socket, name) for name in ("connect", "connect_ex", "sendto", "sendmsg")}

    def refuse(what, detail):
        attempts.add(what, detail)
        raise OSError(f"network access in a test that forbids it: {what} {detail!r}")

    def guarded(name):
        def call(self, *args, **kwargs):
            if self.family == socket.AF_UNIX:
                return real[name](self, *args, **kwargs)
            return refuse(name, args[-1] if name in ("sendto", "sendmsg") and args else (args[0] if args else None))

        return call

    for name in real:
        monkeypatch.setattr(socket.socket, name, guarded(name))
    monkeypatch.setattr(socket, "create_connection", lambda address, *a, **k: refuse("create_connection", address))
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo"):
        monkeypatch.setattr(socket, name, lambda host, *a, _name=name, **k: refuse(_name, host))
    yield attempts
    assert not attempts.read(), f"the test reached for the network: {attempts.read()}"


@pytest.fixture
def keyless_live_dir(tmp_path, monkeypatch):
    """The directory a live test runs in, for an example whose sources need no key (`examples/lear_lago.py`: the benchmark
    datasets come from Zenodo). `live_dir` without the key.

    It is the working directory, so the source cache (`.cache/sources`) and the files written land there. One
    process and one thread run the models. The examples and Workflow.run() set MAX_PROCESSES, THREADS_PER_PROCESS
    and a few more variables for the whole process, so each is registered with monkeypatch first to be restored
    on teardown.
    """
    monkeypatch.chdir(tmp_path)
    for name in PROCESS_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    return tmp_path


@pytest.fixture
def live_dir(keyless_live_dir, monkeypatch, entsoe_api_key):
    """The directory a live test (examples, documentation quick start) runs in. Skips without ENTSOE_API_KEY.

    `keyless_live_dir` with the key in the environment.
    """
    monkeypatch.setenv("ENTSOE_API_KEY", entsoe_api_key)
    return keyless_live_dir


@pytest.fixture
def information_set_checks(monkeypatch):
    """Record every issue-time check that finds nothing, so a test can show the check ran and passed.

    `InformationSet.check` (availability="raise") raises InformationSetError on a predictor that is not known at issue
    time, so a recorded call means it did not. `InformationSet.late_uses` (availability="warn") returns such predictors
    and any column without a rule, so it is recorded when it returns none.
    """
    from epftoolbox2.information_set import InformationSet

    calls = []
    original_check = InformationSet.check
    original_late_uses = InformationSet.late_uses

    def recording_check(self, expanded, *args, **kwargs):
        original_check(self, expanded, *args, **kwargs)
        calls.append((kwargs.get("model_name"), self.issue_time, self.issue_timezone))

    def recording_late_uses(self, expanded, *args, **kwargs):
        found = original_late_uses(self, expanded, *args, **kwargs)
        if not found:
            calls.append((kwargs.get("model_name"), self.issue_time, self.issue_timezone))
        return found

    monkeypatch.setattr(InformationSet, "check", recording_check)
    monkeypatch.setattr(InformationSet, "late_uses", recording_late_uses)
    return calls
