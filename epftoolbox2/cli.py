import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape

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


def cmd_report(args: argparse.Namespace) -> None:
    from .pipelines.workflow import Workflow

    wf = Workflow.load(args.yaml)
    report = wf.report(allow_missing=args.allow_missing)
    counts = report.forecast_counts()
    console.print(counts.to_string(index=False))
    choices = report.choices()
    if not choices.empty:
        console.print(choices.to_string(index=False), markup=False, highlight=False, soft_wrap=True)


def cmd_data(args: argparse.Namespace) -> None:
    from ._frame_csv import save_frame_csv
    from .pipelines.data_pipeline import DataPipeline

    dp = DataPipeline.load(args.yaml)

    # args.cache is: None (not given) → False, True (--cache alone) → True, str (--cache path) → str
    cache = args.cache if args.cache is not None else False
    df = dp.run(start=args.start, end=args.end, cache=cache)

    if df.empty:
        console.print("[yellow]Warning: pipeline returned an empty DataFrame.[/yellow]")

    output = Path(args.output)
    save_frame_csv(df, output)
    console.print(f"[green]Saved {len(df)} rows →[/green] {output}")


def _load_data(args: argparse.Namespace):
    """The frame `--data` and `--tz` name; exits with code 1 if it can't be read."""
    from ._frame_csv import load_frame_csv

    data_path = Path(args.data)
    if not data_path.exists():
        console.print(f"[red]Data file not found: {data_path}[/red]")
        sys.exit(1)

    try:
        return load_frame_csv(data_path, tz=args.tz)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)


def _load_model_pipeline(args: argparse.Namespace):
    """The ModelPipeline YAML, with `--issue-time`, `--issue-timezone` and `--availability` applied to it."""
    from .pipelines.model_pipeline import ModelPipeline

    mp = ModelPipeline.load(args.yaml)
    if args.issue_time or args.issue_timezone:
        from .information_set import InformationSet

        settings = mp.information_set.to_dict() if mp.information_set is not None else {}
        if args.issue_time:
            settings["issue_time"] = args.issue_time
        if args.issue_timezone:
            settings["issue_timezone"] = args.issue_timezone
        mp.information_set = InformationSet.from_dict(settings)
    if getattr(args, "availability", None):
        try:
            mp.availability = args.availability
        except ValueError as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")
            sys.exit(1)
    return mp


def cmd_model(args: argparse.Namespace) -> None:
    from .results.windows import RunWindows, to_period

    validation = args.validation
    if validation is not None and len(validation) == 1:
        validation = validation[0]  # previous_year; anything else is refused below
    try:
        if args.test_start is None and args.test_end is None and validation is None:
            raise ValueError("No window to forecast: pass --test-start and --test-end, --validation, or both")
        test = None
        if args.test_start is not None or args.test_end is not None:
            test = to_period((args.test_start, args.test_end), "test")
        if validation is not None:
            RunWindows.resolve(validation, test)
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        sys.exit(1)
    df = _load_data(args)
    mp = _load_model_pipeline(args)
    mp.run(
        data=df,
        test_start=args.test_start,
        test_end=args.test_end,
        validation=validation,
        target=args.target,
        horizon=args.horizon,
        freq=args.freq,
        save_dir=args.save_dir,
        forecast_only=args.forecast_only,
        allow_manifest_mismatch=args.allow_manifest_mismatch,
        require_manifest=args.require_manifest,
    )


def cmd_verify(args: argparse.Namespace) -> None:
    df = _load_data(args)
    mp = _load_model_pipeline(args)
    try:
        report = mp.verify(
            data=df,
            test_start=args.test_start,
            test_end=args.test_end,
            target=args.target,
            horizon=args.horizon,
            freq=args.freq,
            days=args.days,
            seed=args.seed,
            models=args.models,
            positive_control=args.positive_control,
        )
    except (ValueError, RuntimeError) as exc:  # ValueError includes InformationSetError
        console.print(f"[red]Verification failed:[/red] {escape(str(exc))}")
        sys.exit(1)

    plain = {"markup": False, "highlight": False, "soft_wrap": True}
    console.print(f"Issue days: {', '.join(report.issue_days)}", **plain)
    console.print(report.table().to_string(index=False), **plain)
    for problem in report.problems():
        console.print(problem, **plain)
    if not report.ok:
        console.print("[red]Verification failed.[/red]")
        sys.exit(1)
    console.print("[green]Verification passed.[/green]")


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
#   threads_per_process: 1
#   model_index: 0   # SLURM array: one model per task; needs model_pipeline.save_dir, then `epf report`
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
  - class: CoverageValidator
    params:
      columns: [price]
      min_coverage: 1.0
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


def build_parser() -> argparse.ArgumentParser:
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

    # -- report --
    report_p = sub.add_parser(
        "report", help="Merge per-model results of a workflow (e.g. after a SLURM array) and run evaluators/exporters"
    )
    report_p.add_argument(
        "yaml", metavar="experiment.yaml", help="Path to workflow YAML (with model_pipeline.save_dir)"
    )
    report_p.add_argument(
        "--allow-missing", action="store_true", help="Report the models that have results instead of failing"
    )

    # -- data --
    data_p = sub.add_parser("data", help="Run a data pipeline and save the result to CSV")
    data_p.add_argument("yaml", metavar="data_pipeline.yaml", help="Path to DataPipeline YAML")
    data_p.add_argument("--start", required=True, metavar="DATE", help="Fetch start date (e.g. 2022-01-01)")
    data_p.add_argument("--end", required=True, metavar="DATE", help="Fetch end date (e.g. 2024-01-01)")
    data_p.add_argument("--output", required=True, metavar="FILE", help="Output CSV path (e.g. data.csv)")
    data_p.add_argument(
        "--cache",
        nargs="?",
        const=True,
        metavar="PATH",
        help="Enable caching. Without a value uses automatic per-source cache (.cache/sources/). "
        "With a value (--cache my.csv) caches all merged sources to that CSV file.",
    )

    # -- model --
    model_p = sub.add_parser("model", help="Run a model pipeline on an existing CSV dataset")
    model_p.add_argument("yaml", metavar="model_pipeline.yaml", help="Path to ModelPipeline YAML")
    model_p.add_argument("--data", required=True, metavar="FILE", help="Input CSV file (from epf data)")
    model_p.add_argument("--test-start", metavar="DATE", help="Test period start date")
    model_p.add_argument("--test-end", metavar="DATE", help="Test period end date")
    model_p.add_argument(
        "--validation",
        nargs="+",
        metavar="PERIOD",
        help=(
            "Validation period: START END, or previous_year (each 12 months of the test period served by the 12 months "
            "before them). Without --test-start and --test-end only these days are forecast."
        ),
    )
    model_p.add_argument("--target", default="price", metavar="COL", help="Target column name (default: price)")
    model_p.add_argument("--horizon", type=int, default=7, metavar="N", help="Forecast horizon in days (default: 7)")
    model_p.add_argument(
        "--freq",
        default="1h",
        choices=["1h", "15min"],
        help="Resolution of the data and forecasts: one model per hour or per quarter-hour (default: 1h)",
    )
    model_p.add_argument(
        "--tz",
        metavar="ZONE",
        help="Local time zone of the data (e.g. Europe/Warsaw). Files from `epf data` name it in their header; "
        "other files with UTC offsets need it",
    )
    model_p.add_argument(
        "--issue-time",
        metavar="HH:MM",
        help="Check every predictor against this issue time before fitting (information set; overrides the YAML)",
    )
    model_p.add_argument(
        "--issue-timezone",
        metavar="ZONE",
        help="Zone of --issue-time (default: the YAML's, else Europe/Brussels). Without --issue-time the YAML's issue "
        "time is kept, or 11:45 if the YAML has no information set",
    )
    model_p.add_argument(
        "--availability",
        choices=("none", "warn", "raise"),  # information_set.AVAILABILITY_LEVELS, without importing pandas for --help
        help="What a predictor known after the issue time does: warn lists it and goes on (the default with an issue "
        "time), raise refuses it before fitting, none checks nothing (overrides the YAML)",
    )
    model_p.add_argument("--save-dir", metavar="DIR", help="Directory to save per-model prediction JSONL files")
    model_p.add_argument("--forecast-only", action="store_true", help="Skip evaluation and exporters")
    model_p.add_argument(
        "--allow-manifest-mismatch",
        action="store_true",
        help="Resume even if --save-dir's manifest disagrees with this run's configuration (overwrites it)",
    )
    model_p.add_argument(
        "--require-manifest",
        action="store_true",
        help="Refuse to resume from a --save-dir that has results but no manifest, instead of warning",
    )

    # -- verify --
    verify_p = sub.add_parser(
        "verify", help="Check on sampled issue days that no forecast uses a value unknown when it is issued"
    )
    verify_p.add_argument("yaml", metavar="pipeline.yaml", help="Path to ModelPipeline YAML")
    verify_p.add_argument("--data", required=True, metavar="FILE", help="Input CSV file (from epf data)")
    verify_p.add_argument("--test-start", required=True, metavar="DATE", help="First issue day to sample from")
    verify_p.add_argument("--test-end", required=True, metavar="DATE", help="Last issue day to sample from")
    verify_p.add_argument("--target", default="price", metavar="COL", help="Target column name (default: price)")
    verify_p.add_argument("--horizon", type=int, default=1, metavar="N", help="Forecast horizon in days (default: 1)")
    verify_p.add_argument(
        "--freq",
        default="1h",
        choices=["1h", "15min"],
        help="Resolution of the data and forecasts: one model per hour or per quarter-hour (default: 1h)",
    )
    verify_p.add_argument(
        "--tz",
        metavar="ZONE",
        help="Local time zone of the data (e.g. Europe/Warsaw). Files from `epf data` name it in their header; "
        "other files with UTC offsets need it",
    )
    verify_p.add_argument(
        "--issue-time",
        metavar="HH:MM",
        help="Issue time to verify against (overrides the YAML's information set, or adds one if it has none)",
    )
    verify_p.add_argument(
        "--issue-timezone",
        metavar="ZONE",
        help="Zone of --issue-time (default: the YAML's, else Europe/Brussels). Without --issue-time the YAML's issue "
        "time is kept, or 11:45 if the YAML has no information set",
    )
    verify_p.add_argument("--days", type=int, default=5, metavar="N", help="Issue days per model (default: 5)")
    verify_p.add_argument("--seed", type=int, default=0, metavar="N", help="Picks the days and the noise (default: 0)")
    verify_p.add_argument("--models", nargs="+", metavar="NAME", help="Names of the models to verify (default: all)")
    verify_p.add_argument(
        "--no-positive-control",
        dest="positive_control",
        action="store_false",
        help="Skip the check that perturbing what is known moves the forecasts; a pass then only says nothing moved",
    )

    # -- validate --
    val_p = sub.add_parser("validate", help="Validate a pipeline or workflow YAML")
    val_p.add_argument("yaml", metavar="pipeline.yaml", help="Path to YAML file to validate")

    # -- init --
    init_p = sub.add_parser("init", help="Scaffold template YAML files in the current directory")
    init_p.add_argument("--force", action="store_true", help="Overwrite existing files")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd is None:
        from .tui import run_interactive

        run_interactive()
        return

    dispatch = {
        "run": cmd_run,
        "report": cmd_report,
        "data": cmd_data,
        "model": cmd_model,
        "verify": cmd_verify,
        "validate": cmd_validate,
        "init": cmd_init,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
