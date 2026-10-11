#!/usr/bin/env python3
"""Compare two snapshots from snapshot.py on identical forecast keys (run_date, hour, horizon).

Metrics are computed only on the keys both snapshots share, so a model that silently produced fewer
forecasts can't look better or worse because of it; the key-set difference is reported separately.

Usage
  uv run python .claude/skills/experiment-check/scripts/compare.py BASELINE_DIR CANDIDATE_DIR [--atol 1e-9]

Exit code 1 when key sets differ or any shared forecast differs by more than --atol.
"""

import argparse
import json
import math
import sys
from pathlib import Path


def load(snapshot_dir: Path) -> tuple[dict, dict[str, dict]]:
    meta = json.loads((snapshot_dir / "snapshot.json").read_text())
    models = {}
    for name, info in meta["models"].items():
        rows = {}
        with open(snapshot_dir / info["file"]) as f:
            for line in f:
                r = json.loads(line)
                rows[(r["run_date"], r["hour"], r["horizon"])] = (r["prediction"], r.get("actual"))
        models[name] = rows
    return meta, models


def mae(pairs: list[tuple[float, float | None]]) -> float | None:
    errors = [abs(p - a) for p, a in pairs if a is not None and not math.isnan(a)]
    return sum(errors) / len(errors) if errors else None


def fmt(x: float | None, digits: int = 4) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-9, help="max allowed |Δprediction| on shared keys")
    args = parser.parse_args()

    meta_a, models_a = load(args.baseline)
    meta_b, models_b = load(args.candidate)
    print(f"baseline:  {meta_a['git']['commit'][:8]}{' (dirty)' if meta_a['git']['dirty'] else ''}  {args.baseline}")
    print(f"candidate: {meta_b['git']['commit'][:8]}{' (dirty)' if meta_b['git']['dirty'] else ''}  {args.candidate}\n")
    print(
        "| model | baseline n | candidate n | shared | only baseline | only candidate | MAE baseline | MAE candidate | ΔMAE | max abs Δpred | preds > atol |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")

    different = False
    for name in sorted(set(models_a) | set(models_b)):
        a, b = models_a.get(name, {}), models_b.get(name, {})
        shared = sorted(set(a) & set(b))
        deltas = [abs(a[k][0] - b[k][0]) for k in shared]
        over = sum(d > args.atol for d in deltas)
        mae_a = mae([a[k] for k in shared])
        mae_b = mae([b[k] for k in shared])
        delta_mae = None if mae_a is None or mae_b is None else mae_b - mae_a
        only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
        different |= bool(only_a or only_b or over)
        print(
            f"| {name} | {len(a)} | {len(b)} | {len(shared)} | {only_a} | {only_b} | {fmt(mae_a)} | {fmt(mae_b)} "
            f"| {fmt(delta_mae)} | {fmt(max(deltas) if deltas else None, 6)} | {over} |"
        )

    print("\nResult:", "DIFFERENT" if different else f"identical within atol={args.atol}")
    return 1 if different else 0


if __name__ == "__main__":
    sys.exit(main())
