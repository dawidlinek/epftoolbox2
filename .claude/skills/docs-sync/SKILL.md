---
name: docs-sync
description: Bring README.md, llms.txt, docs/ and examples/ in line with the current epftoolbox2 code. Use after changing public classes, constructor parameters, defaults, output columns, EvaluationReport methods, CLI flags or YAML formats; when asked to update, sync or check the documentation; or when check_docs.py reports drift.
---

# Sync documentation with the code

The code is the source of truth. It is documented in four places, all hand-written:

| Surface | Role |
|---|---|
| `docs/src/content/docs/<category>/*.md(x)` | Starlight site; one page per component, plus pipeline, workflow and reference pages |
| `llms.txt` | Single-file API reference for LLM users: `# CATEGORY` → `## ClassName` → full constructor example, **Parameters:** list |
| `README.md` | Key Features list and quick start |
| `examples/*.py` | Runnable end-to-end scripts (need an ENTSO-E key; not run in CI) |

## 1. Find the drift

**Mechanical drift** — run the checker; it imports the package and compares it with every surface:

```bash
uv run python .claude/skills/docs-sync/scripts/check_docs.py              # findings, exit 1 on errors
uv run python .claude/skills/docs-sync/scripts/check_docs.py --inventory  # every component: params, defaults, llms.txt lines, docs page
```

It covers registry/TUI gaps, missing sections or pages, undocumented parameters, `(default: X)` mismatches in llms.txt, broken `from epftoolbox2 … import` lines, documented calls with keyword arguments the constructor doesn't accept, and `report.<method>()` calls that don't exist.

**Semantic drift** — the checker can't see changed *behaviour*. Look at what changed in the code (default scope: `git diff main...HEAD -- epftoolbox2/`, plus uncommitted changes) and read the changed public code in full. Watch for:
- changed output columns of a source or transformer
- changed meaning of a parameter, even when its name and default stay the same
- new or changed `EvaluationReport` methods → `model-pipeline/results.md`, llms.txt "EVALUATION REPORT"
- `cli.py` flags → `getting-started/cli.mdx`; `Workflow` YAML keys → `workflow/overview.mdx`, llms.txt "WORKFLOW"; caching → the two `caching.mdx` pages

## 2. Update every surface the change touches

| Change | Update |
|---|---|
| New or changed constructor parameter | docs page parameter table and example; llms.txt constructor example (every parameter, comment `# Optional. … (default: X)` with X exactly as the code's default) and **Parameters:** list; serialization YAML examples if the component appears there |
| New component | new page `docs/src/content/docs/<category>/<kebab-name>.md` modelled on a sibling page (frontmatter `title`/`description`, example, parameter table, output, notes); llms.txt `## ClassName` section in its category; README Key Features |
| Removed or renamed parameter / class | remove it everywhere, including examples and YAML snippets |
| Output columns | the page's output section and llms.txt "Output columns" |

Rules:
- Every statement must be traceable to code you read. Don't document behaviour from memory or from other docs.
- Match the style of the surrounding page. `.mdx` pages import Starlight components at the top (`import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';`).
- Categories under `docs/src/content/docs/` with `autogenerate` in `docs/astro.config.mjs` (data-sources, transformers, validators, models, evaluators, exporters) pick up new pages automatically; anything else needs a sidebar entry.
- Don't touch `docs/.astro/` (generated).
- Findings in the `registry` group that point at `epftoolbox2/tui.py` or `COMPONENT_REGISTRY` are code gaps, not doc gaps: report them, don't fix them as part of a docs sync.

## 3. Verify and report

1. Re-run `check_docs.py` until it reports no errors. Resolve warnings or say why they stay (e.g. a documented default that is resolved at runtime, like `columns=None` → `DEFAULT_COLUMNS`).
2. If `docs/node_modules` exists and pages changed structurally, build the site: `cd docs && npm run build`. Otherwise say the build wasn't run.
3. Report: each file changed and what changed in it, drift fixed, anything left and why.
