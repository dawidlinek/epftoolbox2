"""The layout of the test suite itself.

`tests/` has no `__init__.py` files, so pytest imports each test file by its basename: two files of one name (in different
folders) make "import file mismatch" and the second is never collected, silently, in a full run, while each runs fine alone.
"""

from collections import defaultdict
from pathlib import Path

TESTS = Path(__file__).parent


def test_no_two_test_files_share_a_basename():
    by_name = defaultdict(list)
    for path in TESTS.rglob("test_*.py"):
        by_name[path.name].append(str(path.relative_to(TESTS)))
    shared = {name: sorted(paths) for name, paths in by_name.items() if len(paths) > 1}
    assert not shared, f"test files sharing a basename (only one of each is collected): {shared}"
