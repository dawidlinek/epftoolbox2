"""Stub packages that log any attempt to import them, for the tests that some code imports no extra.

A check of `sys.modules` after the code ran can pass without checking anything: an import that failed (the package isn't
installed) leaves no entry, so a guarded `try: import t0 except ImportError: ...` at module level goes unseen. `run_without`
runs the code in a subprocess with a stub of each named package first on the path: importing one appends its name to a log and
raises `ImportError`, so every attempt is seen, installed package or not. `importlib.util.find_spec` (`has_extra`) finds a
stub without running it, as it would find the real package.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


def run_without(code: str, names: Sequence[str], tmp_path: Path) -> Tuple[subprocess.CompletedProcess, List[str]]:
    """(the finished subprocess that ran `code`, the stubbed packages it tried to import, in order). The subprocess runs in
    `tmp_path` with this interpreter and a stub of each of `names` ahead of everything else on the path."""
    stubs = tmp_path / "import-stubs"
    log = tmp_path / "imports.log"
    for name in names:
        package = stubs / name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text(
            "import os\n"
            "with open(os.environ['EPFTOOLBOX2_IMPORT_LOG'], 'a') as log:\n"
            f"    log.write({name!r} + '\\n')\n"
            f"raise ImportError({name!r} + ' is a stub of the import test: something tried to import it')\n"
        )
    path = [str(stubs), *([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path), "EPFTOOLBOX2_IMPORT_LOG": str(log)}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, env=env)
    return result, log.read_text().split() if log.exists() else []
