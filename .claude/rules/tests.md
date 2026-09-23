---
paths:
  - "tests/**"
---

# Tests

- No network in unit tests; mock the HTTP session. Real-API tests are `@pytest.mark.integration` and are excluded by `-m "not integration"`.
- Anything that runs a model: `monkeypatch.setenv("MAX_PROCESSES", "1")` and `monkeypatch.setenv("THREADS_PER_PROCESS", "1")`.
- Prefer synthetic data with a known answer (`tests/synthetic.py`) over hand-built frames: a generator with a seed, known coefficients and a local-time index spanning a DST switch.
- `tests/test_golden_metrics.py` pins current metric values on synthetic data. Refresh with `UPDATE_GOLDEN=1 uv run pytest tests/test_golden_metrics.py` only when a numeric change is intended, and say in the commit message why the numbers moved.
- Known methodology issues are pinned in `tests/test_known_issues.py` with `xfail(strict=True)`. Fixing one makes its test pass, which strict mode reports as a failure: remove the marker and move the test next to the code it covers.
- CSV fixtures are allowed under `tests/fixtures/` (everything else `*.csv` is gitignored).
