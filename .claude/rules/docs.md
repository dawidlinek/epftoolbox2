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
- `llms.txt` — short entry point for LLM agents in the llmstxt.org format: H1, `>` summary, about 15 essential rules, then H2 sections that mirror the docs sidebar with one `- [Title](docs/src/content/docs/<path>): one-line description` bullet per docs page, and an `## Optional` section. It is an index, not a reference. Every docs page must be listed and every link must resolve (`check_docs.py` fails otherwise), so a new, renamed or removed page changes `llms.txt` too. Don't copy parameter lists or page content into it.
- `README.md` — feature list and quick start only; keep the "Key Features" list complete.
- `examples/` — runnable scripts; they need `ENTSOE_API_KEY` and network access, so they are not run in CI. Each one has a `main(...)` with its date ranges and output directory as parameters (defaults = the full run), and `tests/test_examples.py` runs them on a short window: `ENTSOE_API_KEY=... uv run pytest -m examples`. A change to an example's behaviour or output files changes its test too.
- Code blocks — every ```python block in `README.md` and `docs/src/content/docs/` is checked by `tests/test_docs_code.py` (fast suite): it compiles, its `epftoolbox2` imports run and every call to an epftoolbox2 class, function or method is bound to the real signature, so a renamed parameter fails. Give each block its imports, and fix the block rather than skipping it. A block that is deliberately not Python opts out with the fence ```` ```python nocheck ````. The `reference/api.md` stubs must list every real parameter, in order, with the real defaults. The quick start blocks also run live with `ENTSOE_API_KEY=... uv run pytest -m examples`.
- `docs/.astro/` is generated — don't edit it.

`uv run python .claude/skills/docs-sync/scripts/check_docs.py` reports drift (missing classes/params, docs pages missing from `llms.txt`, dead links in it, broken imports in code blocks, registry/TUI gaps). Use the `docs-sync` skill or agent to fix it.
