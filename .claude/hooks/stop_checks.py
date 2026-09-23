#!/usr/bin/env python3
"""Stop hook: before Claude finishes, lint and run the fast test suite if Python code changed.

Runs only when the content of the project's Python files (plus pyproject.toml / uv.lock) differs
from the last state that passed, so question-only turns stay fast. A failure is sent back to Claude
with exit code 2, once per turn (``stop_hook_active`` prevents a loop).

Disable for a session with EPF_SKIP_STOP_CHECKS=1.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

TAIL_LINES = 60


def fingerprint(project: Path) -> str:
    files = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py", "pyproject.toml", "uv.lock"],
        cwd=project,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    digest = hashlib.sha256()
    for name in sorted(f for f in files if f):
        path = project / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def state_file(project: Path) -> Path:
    git_dir = subprocess.run(
        ["git", "rev-parse", "--absolute-git-dir"], cwd=project, capture_output=True, text=True, check=True
    ).stdout.strip()
    return Path(git_dir) / "epf-stop-checks.sha256"


def tail(text: str) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-TAIL_LINES:])


def main() -> int:
    payload = json.load(sys.stdin)
    if payload.get("stop_hook_active") or os.environ.get("EPF_SKIP_STOP_CHECKS") == "1":
        return 0

    project = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()).resolve()
    try:
        current = fingerprint(project)
        marker = state_file(project)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0  # not a git checkout; nothing sensible to compare against

    if marker.is_file() and marker.read_text().strip() == current:
        return 0

    problems = []
    uv = ["uv", "run", "--frozen", "--quiet"]
    lint = subprocess.run(
        [*uv, "ruff", "check", "--output-format", "concise", "."], cwd=project, capture_output=True, text=True
    )
    if lint.returncode != 0:
        problems.append(f"ruff check failed:\n{tail(lint.stdout)}")
    fmt = subprocess.run([*uv, "ruff", "format", "--check", "."], cwd=project, capture_output=True, text=True)
    if fmt.returncode != 0:
        problems.append(f"ruff format --check failed (run `uv run ruff format .`):\n{tail(fmt.stdout)}")
    tests = subprocess.run(
        [*uv, "pytest", "-m", "not integration", "-q", "-x", "--tb=short", "-o", "addopts=", "-p", "no:cacheprovider"],
        cwd=project,
        capture_output=True,
        text=True,
    )
    if tests.returncode != 0:
        problems.append(f"pytest failed:\n{tail(tests.stdout + tests.stderr)}")

    if problems:
        print(
            "Checks failed before finishing — fix them or explain why they fail:\n\n" + "\n\n".join(problems),
            file=sys.stderr,
        )
        return 2

    marker.write_text(current)
    return 0


if __name__ == "__main__":
    sys.exit(main())
