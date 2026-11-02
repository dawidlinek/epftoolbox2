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


# Variables that the examples (at import), Workflow.run() and the documentation set on the whole process.
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


@pytest.fixture
def live_dir(tmp_path, monkeypatch, entsoe_api_key):
    """The directory a live test (examples, documentation quick start) runs in. Skips without ENTSOE_API_KEY.

    It is the working directory, so the source cache (`.cache/sources`) and the files written land there. One
    process and one thread run the models. The examples and Workflow.run() set MAX_PROCESSES, THREADS_PER_PROCESS
    and a few more variables for the whole process, so each is registered with monkeypatch first to be restored
    on teardown.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENTSOE_API_KEY", entsoe_api_key)
    for name in PROCESS_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAX_PROCESSES", "1")
    monkeypatch.setenv("THREADS_PER_PROCESS", "1")
    return tmp_path


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
