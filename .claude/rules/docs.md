---
paths:
  - "docs/src/**"
  - "llms.txt"
  - "README.md"
  - "examples/**"
---

# Documentation surfaces

Docs are hand-written; the code is the source of truth. Read the class signature before editing a docs page, and never document a parameter or default from memory.

- `docs/src/content/docs/<category>/<name>.md(x)` — Astro Starlight site. The categories `data-sources`, `transformers`, `validators`, `models`, `evaluators`, `exporters` use `autogenerate` in `docs/astro.config.mjs`, so a new page appears in the sidebar automatically; every other section needs a manual sidebar entry there. `.mdx` pages may use Starlight components (`<Tabs>`, `<Aside>`) and Mermaid diagrams.
- `llms.txt` — single-file API reference for LLM users: `# CATEGORY` → `## ClassName` → a constructor example listing **all** parameters with defaults as comments. Keep the parameter lists complete.
- `README.md` — feature list and quick start only; keep the "Key Features" list complete.
- `examples/` — runnable scripts; they need `ENTSOE_API_KEY` and network access, so they are not run in CI.
- `docs/.astro/` is generated — don't edit it.

`uv run python .claude/skills/docs-sync/scripts/check_docs.py` reports drift (missing classes/params, broken imports in code blocks, registry/TUI gaps). Use the `docs-sync` skill or agent to fix it.
