---
name: experiment-check
description: Measure whether a code change alters forecasts or metrics on a realistic epftoolbox2 run. Snapshot a Workflow on the baseline (e.g. main, in a git worktree) and on the current code, then compare them on identical forecast keys. Use for refactors that must not change numbers, to quantify a methodology fix, or when the user asks how a change affects results.
---

# Experiment check

The unit tests and `tests/test_golden_metrics.py` use small synthetic data. This skill answers a different question: what does this change do to results on real data?

## 1. Choose a small experiment with frozen data
- Start from a Workflow YAML such as `experiment.yaml`. Keep it small: one or two models, a test window of 1–4 weeks, a short horizon. A run should take minutes.
- Freeze the input data so both runs see identical data and no API is called. Set `environment.cache_path` to an **absolute** path of a CSV file; `cache_path` is resolved against the working directory, not the YAML. The first run fetches the data and writes the CSV; later runs read it. Keep these files outside the repo, e.g. in `~/epf-experiments/`.
- Never put a real ENTSO-E key into a tracked YAML. Keep experiment YAMLs outside the repo.
- Heavy runs belong on the shared servers or WCSS, not this machine. Check their load first, and ask the user before launching anything there.

## 2. Snapshot the baseline and the candidate

```bash
git worktree add ../epftoolbox2-baseline main   # or whichever commit to compare against
(cd ../epftoolbox2-baseline && uv run python "$PWD/.claude/skills/experiment-check/scripts/snapshot.py" /abs/path/workflow.yaml --out /abs/path/runs/baseline)
uv run python .claude/skills/experiment-check/scripts/snapshot.py /abs/path/workflow.yaml --out /abs/path/runs/candidate
```

Two things to know:
- `$PWD` expands before the `cd`, so the baseline runs this checkout's script against the worktree's environment. The script imports `epftoolbox2` from whichever project it runs in.
- `snapshot.json` records which `epftoolbox2` file ran and at which commit. Check both before trusting a comparison.

Pipeline paths inside the workflow YAML resolve relative to the YAML, so both runs share the same pipeline files. `--model-index N` runs a single model.

## 3. Compare

```bash
uv run python .claude/skills/experiment-check/scripts/compare.py /abs/path/runs/baseline /abs/path/runs/candidate
```

Metrics use only the forecast keys that both runs share. The `only baseline` and `only candidate` columns count forecasts that appeared or disappeared, which is where silent drops show up. The script exits with 1 when anything differs beyond `--atol` (default 1e-9).

## 4. Report
- **Refactor meant to be neutral:** the result must be "identical". Any difference is a finding. Find where it concentrates by grouping the JSONL rows by date, hour and horizon; DST days and the start of the test window are the usual suspects.
- **Methodology change:** report ΔMAE per model, the changes in key counts, and where the differences concentrate.
- Clean up with `git worktree remove ../epftoolbox2-baseline`.
