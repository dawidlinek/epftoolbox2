#!/usr/bin/env python3
"""PostToolUse hook: format an edited Python file and report lint problems back to Claude.

Exit code 2 sends stderr to Claude as feedback (the edit itself has already happened).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Rules that fire while a change is half-written (an import added one edit before its first use).
# The Stop hook runs the full rule set, so nothing is lost by skipping them here.
TRANSIENT_RULES = "F401,F841"


def main() -> int:
    payload = json.load(sys.stdin)
    file_path = (payload.get("tool_input") or {}).get("file_path") or ""
    if not file_path.endswith(".py") or not os.path.isfile(file_path):
        return 0

    project = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()).resolve()
    try:
        Path(file_path).resolve().relative_to(project)
    except ValueError:
        return 0  # not a project file

    ruff = ["uv", "run", "--frozen", "--quiet", "ruff"]
    subprocess.run([*ruff, "format", "--quiet", file_path], cwd=project, capture_output=True)
    lint = subprocess.run(
        [*ruff, "check", "--output-format", "concise", "--extend-ignore", TRANSIENT_RULES, file_path],
        cwd=project,
        capture_output=True,
        text=True,
    )
    if lint.returncode != 0:
        print(f"ruff reported problems after editing {file_path}:\n{lint.stdout.strip()}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
