---
name: add-component
description: Checklist for adding or renaming a public epftoolbox2 component — data source, transformer, validator, model, evaluator or exporter — so that it is importable, loadable from YAML, offered in the TUI, tested and documented. Use whenever such a class is created or renamed.
---

# Adding a component

A component is finished only when all of these places agree. Work through them in order; `check_docs.py` verifies most of them at the end.

| # | Where | What |
|---|---|---|
| 1 | `epftoolbox2/<package>/<module>.py` | Subclass the base: `DataSource`, `Transformer`, `Validator`, `BaseModel`, `Evaluator` or `Exporter`. Store every constructor parameter as a **same-named public attribute**; internal state starts with `_`. YAML serialization depends on both. |
| 2 | `epftoolbox2/<package>/__init__.py` | Import it and add it to `__all__`. |
| 3 | `COMPONENT_REGISTRY` | `pipelines/data_pipeline.py` for sources/transformers/validators, `pipelines/model_pipeline.py` for models/evaluators/exporters: `"ClassName": "epftoolbox2.<package>.<module>"`. |
| 4 | `epftoolbox2/tui.py` | Add it to the matching `_SOURCES` … `_EXPORTERS` list and add a `PARAM_SCHEMAS["ClassName"]` entry. Field `name`s must equal constructor parameter names. |
| 5 | `tests/` | Behaviour tests plus a YAML round trip (save → load → same parameters), following `tests/pipelines/test_serialization.py`. |
| 6 | `docs/src/content/docs/<category>/<kebab-name>.md` | New page modelled on a sibling; the sidebar picks it up automatically. |
| 7 | `llms.txt` | `## ClassName` section in the right category, listing every parameter with its default. |
| 8 | `README.md` | Key Features list. |
| 9 | `examples/` | Only if the component changes the typical workflow. |

## Category specifics
- **Source.** `fetch(start, end)` returns a UTC-indexed DataFrame. Implement `_validate_config()`. `get_cache_config()` must return `"source_type"` plus every parameter that affects the output. Tests mock HTTP; a real-API test is marked `@pytest.mark.integration`. `_serialize_component` skips attributes named `session`, `console`, `logger`, `lat` and `lon`.
- **Transformer.** `transform(df) -> df`. Never drop or reorder rows silently. Time shifts should follow local calendar days: use `shift_calendar_days` from `epftoolbox2/_date_utils.py` (semantics in `.claude/rules/data.md`).
- **Validator.** `validate(df) -> ValidationResult`. Use errors for data that can't be used and warnings for data that is suspicious.
- **Model.** Implement `_fit_predict(train_x, train_y, test_x) -> (prediction, coefficients)` on standardised data. Every hyperparameter must be in `__init__` **and** returned by `_model_kwargs`, because workers rebuild the model from `_model_kwargs`. Stochastic models take a `random_state`. `predictors` stays the first parameter. Add the model to `tests/test_golden_metrics.py` so its numbers are pinned.
- **Evaluator.** Set a class attribute `name` and implement `compute(df, **kwargs) -> float`. Relative metrics read the benchmark from `kwargs["model_dfs"]` and must compare on aligned forecast keys.
- **Exporter.** `export(report)` consumes `report.compute_all()` or `report.iter_details()` and never computes metrics itself.

## Finish
1. `uv run pytest -m "not integration" -q`
2. `uv run python .claude/skills/docs-sync/scripts/check_docs.py`: no errors and no registry warnings for the new class.
3. For models, transformers and evaluators, have the `forecast-methodology-reviewer` agent review the diff.
