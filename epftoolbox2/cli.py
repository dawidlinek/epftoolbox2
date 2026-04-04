import argparse
import sys
from pathlib import Path

from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    from .pipelines.workflow import Workflow

    wf = Workflow.load(args.yaml)

    # CLI flags override YAML values; set on the Workflow object so _apply_environment()
    # picks them up (rather than setting env vars here and having them overwritten).
    if args.processes is not None:
        wf.max_processes = args.processes
    if args.threads is not None:
        wf.threads_per_process = args.threads
    if args.model_index is not None:
        wf.model_index = args.model_index

    if args.dry_run:
        wf._build_data_pipeline()
        wf._build_model_pipeline()
        console.print("[green]Dry run OK — pipelines loaded successfully.[/green]")
        return

    wf.run()


def cmd_data(args: argparse.Namespace) -> None:
    import pandas as pd
    from .pipelines.data_pipeline import DataPipeline

    dp = DataPipeline.load(args.yaml)

    # args.cache is: None (not given) → False, True (--cache alone) → True, str (--cache path) → str
    cache = args.cache if args.cache is not None else False
    df = dp.run(start=args.start, end=args.end, cache=cache)

    if df.empty:
        console.print("[yellow]Warning: pipeline returned an empty DataFrame.[/yellow]")

    output = Path(args.output)
    df.to_csv(output)
    console.print(f"[green]Saved {len(df)} rows →[/green] {output}")


def cmd_model(args: argparse.Namespace) -> None:
    import pandas as pd
    from .pipelines.model_pipeline import ModelPipeline

    data_path = Path(args.data)
    if not data_path.exists():
        console.print(f"[red]Data file not found: {data_path}[/red]")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)

    mp = ModelPipeline.load(args.yaml)
    mp.run(
        data=df,
        test_start=args.test_start,
        test_end=args.test_end,
        target=args.target,
        horizon=args.horizon,
        save_dir=args.save_dir,
        forecast_only=args.forecast_only,
    )


def cmd_validate(args: argparse.Namespace) -> None:
    import yaml as _yaml

    path = Path(args.yaml)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        sys.exit(1)

    with open(path) as f:
        raw = _yaml.safe_load(f)

    if raw is None:
        console.print("[red]Empty or invalid YAML.[/red]")
        sys.exit(1)

    top_keys = set(raw.keys())
    try:
        if "data_pipeline" in top_keys or "model_pipeline" in top_keys:
            from .pipelines.workflow import Workflow
            wf = Workflow.load(path)
            wf._build_data_pipeline()
            wf._build_model_pipeline()
            console.print(f"[green]Workflow YAML valid:[/green] {path}")
        elif "sources" in top_keys or "transformers" in top_keys or "validators" in top_keys:
            from .pipelines.data_pipeline import DataPipeline
            DataPipeline.load(path)
            console.print(f"[green]DataPipeline YAML valid:[/green] {path}")
        elif "models" in top_keys or "evaluators" in top_keys or "exporters" in top_keys:
            from .pipelines.model_pipeline import ModelPipeline
            ModelPipeline.load(path)
            console.print(f"[green]ModelPipeline YAML valid:[/green] {path}")
        else:
            console.print("[yellow]Unrecognised YAML structure — could not determine pipeline type.[/yellow]")
            sys.exit(1)
    except Exception as exc:
        console.print(f"[red]Validation failed:[/red] {exc}")
        sys.exit(1)


_EXPERIMENT_TEMPLATE = """\
data_pipeline:
  path: data_pipeline.yaml
  start: "2022-01-01"
  end: "2024-01-01"
  cache: true

model_pipeline:
  path: model_pipeline.yaml
  test_start: "2023-01-01"
  test_end: "2024-01-01"
  target: price
  horizon: 7

# environment:
#   max_processes: 4
#   threads_per_process: 2
#   model_index: 0
"""

_DATA_PIPELINE_TEMPLATE = """\
sources:
  - class: CsvSource
    params:
      file_path: data/prices.csv

transformers:
  - class: TimezoneTransformer
    params:
      target_tz: Europe/Warsaw

validators:
  - class: NullCheckValidator
    params:
      columns: [price]
"""

_MODEL_PIPELINE_TEMPLATE = """\
models:
  - class: OLSModel
    params:
      predictors: [price_d-1]
      training_window: 365
      name: OLS

evaluators:
  - class: MAEEvaluator
    params: {}

exporters:
  - class: TerminalExporter
    params: {}
"""


def cmd_init(args: argparse.Namespace) -> None:
    files = {
        "experiment.yaml": _EXPERIMENT_TEMPLATE,
        "data_pipeline.yaml": _DATA_PIPELINE_TEMPLATE,
        "model_pipeline.yaml": _MODEL_PIPELINE_TEMPLATE,
    }
    for name, content in files.items():
        p = Path(name)
        if p.exists() and not args.force:
            console.print(f"[yellow]Skipped (already exists):[/yellow] {name}  — use --force to overwrite")
            continue
        p.write_text(content)
        console.print(f"[green]Created:[/green] {name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="epf",
        description="EPF Toolbox 2 — electricity price forecasting CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=False, metavar="<command>")

    # -- run --
    run_p = sub.add_parser("run", help="Run a full workflow experiment")
    run_p.add_argument("yaml", metavar="experiment.yaml", help="Path to workflow YAML")
    run_p.add_argument("--model-index", type=int, metavar="N", help="Override environment.model_index (SLURM array)")
    run_p.add_argument("--processes", type=int, metavar="N", help="Override max_processes")
    run_p.add_argument("--threads", type=int, metavar="N", help="Override threads_per_process")
    run_p.add_argument("--dry-run", action="store_true", help="Build pipelines but skip execution")

    # -- data --
    data_p = sub.add_parser("data", help="Run a data pipeline and save the result to CSV")
    data_p.add_argument("yaml", metavar="data_pipeline.yaml", help="Path to DataPipeline YAML")
    data_p.add_argument("--start", required=True, metavar="DATE", help="Fetch start date (e.g. 2022-01-01)")
    data_p.add_argument("--end", required=True, metavar="DATE", help="Fetch end date (e.g. 2024-01-01)")
    data_p.add_argument("--output", required=True, metavar="FILE", help="Output CSV path (e.g. data.csv)")
    data_p.add_argument(
        "--cache", nargs="?", const=True, metavar="PATH",
        help="Enable caching. Without a value uses automatic per-source cache (.cache/sources/). "
             "With a value (--cache my.csv) caches all merged sources to that CSV file.",
    )

    # -- model --
    model_p = sub.add_parser("model", help="Run a model pipeline on an existing CSV dataset")
    model_p.add_argument("yaml", metavar="model_pipeline.yaml", help="Path to ModelPipeline YAML")
    model_p.add_argument("--data", required=True, metavar="FILE", help="Input CSV file (from epf data)")
    model_p.add_argument("--test-start", required=True, metavar="DATE", help="Test period start date")
    model_p.add_argument("--test-end", required=True, metavar="DATE", help="Test period end date")
    model_p.add_argument("--target", default="price", metavar="COL", help="Target column name (default: price)")
    model_p.add_argument("--horizon", type=int, default=7, metavar="N", help="Forecast horizon in hours (default: 7)")
    model_p.add_argument("--save-dir", metavar="DIR", help="Directory to save per-model prediction JSONL files")
    model_p.add_argument("--forecast-only", action="store_true", help="Skip evaluation and exporters")

    # -- validate --
    val_p = sub.add_parser("validate", help="Validate a pipeline or workflow YAML")
    val_p.add_argument("yaml", metavar="pipeline.yaml", help="Path to YAML file to validate")

    # -- init --
    init_p = sub.add_parser("init", help="Scaffold template YAML files in the current directory")
    init_p.add_argument("--force", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()

    if args.cmd is None:
        from .tui import run_interactive
        run_interactive()
        return

    dispatch = {
        "run": cmd_run,
        "data": cmd_data,
        "model": cmd_model,
        "validate": cmd_validate,
        "init": cmd_init,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
