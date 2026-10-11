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
- `tests/test_examples.py` runs each `examples/*.py` end to end through its `main(...)` on a short window against the live ENTSO-E, Open-Meteo and calendar sources. Every test there carries both `integration` (so `-m "not integration"` skips it) and `examples`, and uses the shared `entsoe_api_key` fixture (in `tests/conftest.py`), so it skips without a key. Run it with `ENTSOE_API_KEY=... uv run pytest -m examples -q` and put the key on the command line only. Each test works in `tmp_path` (the working directory, so the source cache and the output files land there) with one process and one thread.
- An example's `main(...)` takes the date ranges it uses and an output directory for the files it writes, and its defaults are the full run the script documents. Keep test-only code out of the examples: they are documentation first. When an example changes, update its test in the same commit.
- An example test asserts outcomes: the data frame (non-empty, key columns, no all-NaN column), `report.forecast_counts()` (no missing forecasts outside a DST window, identical forecast keys across models), that `InformationSet.check` ran where the example sets an issue time, and the files written.
