#!/usr/bin/env python3
"""Run a Workflow YAML and save every forecast plus summary metrics, for comparison with compare.py.

epftoolbox2 is imported from the *current environment*, so run this from the checkout you want to
measure (e.g. a git worktree of main). The YAML's exporters are disabled so the run leaves no
Excel/CSV side outputs.

Usage
  uv run python .claude/skills/experiment-check/scripts/snapshot.py WORKFLOW.yaml --out DIR [--model-index N]

Writes DIR/snapshot.json (provenance + per-model summary) and DIR/<model>.jsonl (one forecast per line).
"""

import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path

KEY_COLUMNS = ["run_date", "target_date", "hour", "horizon", "prediction", "actual"]


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def metrics(rows: list[dict]) -> dict:
    errors = [
        r["prediction"] - r["actual"]
        for r in rows
        if r.get("actual") is not None and not math.isnan(r["actual"]) and not math.isnan(r["prediction"])
    ]
    if not errors:
        return {"evaluated": 0, "mae": None, "rmse": None}
    return {
        "evaluated": len(errors),
        "mae": sum(abs(e) for e in errors) / len(errors),
        "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-index", type=int, default=None, help="run only this model from the pipeline")
    args = parser.parse_args()

    import epftoolbox2
    from epftoolbox2.pipelines.workflow import Workflow

    workflow = Workflow.load(args.workflow)
    workflow._mp_inline["exporters"] = []  # inline override: no side outputs
    if args.model_index is not None:
        workflow.model_index = args.model_index

    started = time.time()
    report = workflow.run()
    elapsed = time.time() - started

    args.out.mkdir(parents=True, exist_ok=True)
    models = {}
    for name, details in report.iter_details():
        rows = [{k: row[k] for k in KEY_COLUMNS if k in row} for row in details.to_dict("records")]
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", name) + ".jsonl"
        with open(args.out / filename, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        models[name] = {"file": filename, "forecasts": len(rows), "planned": report.refs[name].count, **metrics(rows)}

    snapshot = {
        "workflow": str(args.workflow.resolve()),
        "epftoolbox2": {"version": epftoolbox2.__version__, "file": epftoolbox2.__file__},
        "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "epftoolbox2"))},
        "elapsed_s": round(elapsed, 1),
        "models": models,
    }
    (args.out / "snapshot.json").write_text(json.dumps(snapshot, indent=2))

    print(f"Snapshot written to {args.out} (commit {snapshot['git']['commit'][:8]}, {elapsed:.1f}s)")
    print("| model | forecasts / planned | MAE | RMSE |")
    print("|---|---|---|---|")
    for name, m in models.items():
        mae = f"{m['mae']:.4f}" if m["mae"] is not None else "–"
        rmse = f"{m['rmse']:.4f}" if m["rmse"] is not None else "–"
        print(f"| {name} | {m['forecasts']} / {m['planned']} | {mae} | {rmse} |")


if __name__ == "__main__":
    main()
