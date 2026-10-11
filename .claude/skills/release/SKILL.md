---
name: release
description: Prepare an epftoolbox2 release — bump the version in both places, refresh uv.lock, draft release notes from the commits, and commit locally. Only when the user asks for a release or a version bump.
disable-model-invocation: true
argument-hint: "[patch|minor|major|X.Y.Z]"
---

# Release

Requested version: `$ARGUMENTS`. If it is empty, ask which bump the user wants and propose one. Use semantic versioning: `patch` for fixes, `minor` for new components or parameters, `major` for breaking changes to the Python API, the YAML format or the result files.

## 1. Preconditions
- The working tree is clean. Check which branch you are on and confirm it with the user if it isn't `main`.
- These pass:
  - `uv run pytest -m "not integration" -q`
  - `uv run ruff check .`
  - `uv run ruff format --check .`
  - `uv run python .claude/skills/docs-sync/scripts/check_docs.py` (no errors)

  If any fails, stop and report it.

## 2. Bump
The version lives in two places, and both must change:
- `version = "X.Y.Z"` in `pyproject.toml`
- `__version__ = "X.Y.Z"` in `epftoolbox2/__init__.py`

Then run `uv lock` and check that they agree:

```bash
uv run python -c "import epftoolbox2, importlib.metadata as m; assert epftoolbox2.__version__ == m.version('epftoolbox2'), (epftoolbox2.__version__, m.version('epftoolbox2')); print(epftoolbox2.__version__)"
```

## 3. Release notes
The repo has no tags. The previous release is the last commit whose subject starts with `chore: bump version to`:

```bash
prev=$(git log --format=%H --grep='^chore: bump version to' -n1)
git log "$prev"..HEAD --no-merges --format='%s'
```

Group the subjects by prefix (feat, fix, docs, test, chore, ci, style). Put user-visible changes first, and add a "Breaking changes" section if there are any. Show the notes to the user. There is no CHANGELOG file; only create one if the user asks.

## 4. Commit
Commit only `pyproject.toml`, `epftoolbox2/__init__.py` and `uv.lock`, with the message `chore: bump version to X.Y.Z` (the repo's convention). Create a `vX.Y.Z` tag only if the user wants tags.

Never push, and never publish to PyPI yourself. Finish by telling the user the commands:

```bash
git push origin <branch> --follow-tags
uv build
uv publish
```
