# epftoolbox2

Electricity price forecasting library (a rewrite of Lago et al.'s epftoolbox): download market data, build features, run rolling-window forecasts, evaluate. It is used for research, so a methodological error — look-ahead, DST misalignment, models compared on different samples — is worse than a crash: it produces wrong numbers that look fine.

## Commands
- Fast tests (~10 s, no network): `uv run pytest -m "not integration" -q`
- Lint / format: `uv run ruff check .` and `uv run ruff format .` (line length 120)
- Integration tests need `ENTSOE_API_KEY` and hit real APIs; don't run them unless asked.
- `.python-version` pins free-threaded 3.14t locally, but CI runs 3.10–3.14: no syntax newer than 3.10 in the package.
- Dependencies live only in `pyproject.toml`. After changing them run `uv lock`; CI fails on a stale lock (`uv lock --check`).

## Hooks you will see
- Editing a `.py` file formats it with ruff and reports lint errors back (unused imports are ignored mid-edit).
- Before finishing a turn in which Python files changed, `ruff check`, `ruff format --check` and the fast tests run; failures come back as feedback. The user can disable this with `EPF_SKIP_STOP_CHECKS=1`.
- `git commit` is blocked when changes contain a real-looking API key. `DataPipeline.save()` writes `api_key` as a `${ENV_VAR}` reference by default; the hook still matters because `save(path, include_secrets=True)` is an explicit, documented opt-out into plain text, and pipeline YAMLs are tracked.

## How the pieces fit
`DataPipeline` (sources → transformers → validators, UTC in, cached per source) produces one DataFrame. `ModelPipeline` fits one model per (period of day, horizon, run day) in a process pool, stores results as JSONL (resumable), then evaluators and exporters run over an `EvaluationReport`. `Workflow` runs both from YAML; `environment.model_index` exists for SLURM array jobs. Component classes must be registered in `COMPONENT_REGISTRY` (`pipelines/data_pipeline.py`, `pipelines/model_pipeline.py`) to load from YAML.

Column suffixes are relative to the row's local date D: `x_d-k` is x on D-k, `x_d+h` is x on D+h as known on D (targets are `{target}_d+{h}`). Predictors may contain `{horizon}`, or be callables `(horizon)` / `(horizon, hour)`; callables can't be saved to YAML.

Path-scoped rules in `.claude/rules/` load automatically for models, data, evaluation, docs and tests; they hold the invariants and known issues for each area.

## Research invariants
- A forecast issued on day D may use only information available at issue time. Training targets currently end at D-1; don't move that boundary unless asked.
- Local calendar days, not 24-hour offsets, define "same hour yesterday" (DST days have 23 or 25 hours).
- Compare models only on identical forecast keys, and report how many forecasts each model produced.
- Same data and config must give the same numbers. `tests/test_golden_metrics.py` pins them; update the golden file only for an intended change and explain it.
- Known methodology issues are pinned by `xfail(strict=True)` tests in `tests/test_known_issues.py`. Don't fix them as a side effect of other work; fix them as their own change.

## Changing public API
- New source / transformer / validator / model / evaluator / exporter: use the `add-component` skill; it touches about eight places.
- Any change to public classes, parameters or defaults must reach README.md, `llms.txt` and `docs/`. Use the `docs-sync` agent before finishing, or run `uv run python .claude/skills/docs-sync/scripts/check_docs.py` to see drift.
- Changes under `models/`, `data/transformers/`, `evaluators/` or `results/`: have the `forecast-methodology-reviewer` agent review the diff.
- Measuring the numeric effect of a change on real data: `experiment-check` skill.

## Git
- One branch per change (`feat/…`, `fix/…`, `chore/…`, `docs/…`), conventional commit prefixes (see `git log`). Never push without being asked.
- The version is set in `pyproject.toml` and `epftoolbox2/__init__.py`; the `release` skill bumps both.
- The bulk ruff format commit is listed in `.git-blame-ignore-revs`.
