import argparse
import contextlib
import sys
from pathlib import Path
from typing import List, Optional

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


def cmd_array(args: argparse.Namespace) -> None:
    from .pipelines.workflow import Workflow

    # `sbatch --array=$(epf array experiment.yaml)` takes all of stdout, so that holds the range alone: logs go to stderr.
    errors = Console(stderr=True)
    with contextlib.redirect_stdout(sys.stderr):
        try:
            workflow = Workflow.load(args.yaml)
            size = workflow.array_size()
        except ValueError as exc:
            errors.print(f"[red]{escape(str(exc))}[/red]")
            sys.exit(1)
        errors.print(f"{size} array task(s), {workflow.models_per_task or 1} model(s) per task", markup=False)
    print(f"0-{size - 1}")


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


def horizon_list(text: str) -> tuple:
    """`--horizons`: `3-9` (every one from 3 to 9), `1,3,7`, or both kinds joined by commas (`1-3,7`)."""
    found = []
    for part in text.split(","):
        first, dash, last = part.strip().partition("-")
        try:
            found.extend(range(int(first), int(last) + 1) if dash else [int(first)])
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{text!r} is not a list of horizons: write 3-9, 1,3,7 or 1-3,7 (whole numbers of days)"
            ) from None
        if dash and int(last) < int(first):
            raise argparse.ArgumentTypeError(f"{part.strip()!r} runs backwards: write {last}-{first}")
    from .models._calendar import resolve_horizons

    try:
        return resolve_horizons(None, found, 1)[1]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _add_horizon_arguments(parser: argparse.ArgumentParser, default: int) -> None:
    """`--horizon N` (every horizon from 1 to N) or `--horizons 3-9` / `1,3,7`, not both."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--horizon",
        type=int,
        metavar="N",
        help=f"Forecast horizon in days: every horizon from 1 to N (default: {default})",
    )
    group.add_argument(
        "--horizons",
        type=horizon_list,
        metavar="LIST",
        help="Only these horizons, in days: 3-9, 1,3,7 or 1-3,7 (in place of --horizon)",
    )


def issue_day_list(text: str) -> Optional[List[str]]:
    """`--issue-days`: weekday names joined by commas, in any case (`friday` or `Monday,Thursday`); all seven are every day."""
    from .models._calendar import issue_day_names, resolve_issue_days

    try:
        return issue_day_names(resolve_issue_days([part.strip() for part in text.split(",")]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _add_issue_days_argument(parser: argparse.ArgumentParser, what: str) -> None:
    parser.add_argument(
        "--issue-days",
        type=issue_day_list,
        metavar="DAYS",
        help=f"{what} on these weekdays only, e.g. friday or monday,thursday (default: every day)",
    )


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
        horizons=args.horizons,
        issue_days=args.issue_days,
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
            horizons=args.horizons,
            issue_days=args.issue_days,
            freq=args.freq,
            days=args.days,
            seed=args.seed,
            models=args.models,
            positive_control=args.positive_control,
            device=args.device,
            co_batch=args.co_batch,
            rerun=args.rerun,
        )
    except (ValueError, RuntimeError) as exc:  # ValueError includes InformationSetError
        console.print(f"[red]Verification failed:[/red] {escape(str(exc))}")
        sys.exit(1)

    plain = {"markup": False, "highlight": False, "soft_wrap": True}
    console.print(f"Issue days: {', '.join(report.issue_days)}", **plain)
    if report.devices:
        console.print(f"On device {args.device}: {', '.join(report.devices)}", **plain)
    console.print(report.table().to_string(index=False), **plain)
    for problem in report.problems():
        console.print(problem, **plain)
    if not report.ok:
        console.print("[red]Verification failed.[/red]")
        sys.exit(1)
    console.print("[green]Verification passed.[/green]")


def cmd_tune(args: argparse.Namespace) -> None:
    """Run a search script (`epftoolbox2.tune.BayesSearch` with `storage=`) in `--workers` processes that share its trials."""
    import os
    import subprocess

    from .models.base import _usable_cpu_count
    from .tune.bayes import ENV_REQUEUE, ENV_WORKER, ENV_WORKERS

    script = Path(args.script)
    if not script.is_file():
        console.print(f"[red]No such script:[/red] {escape(str(script))}")
        sys.exit(2)
    if args.workers < 1:
        console.print("[red]--workers must be at least 1[/red]")
        sys.exit(2)
    command = [sys.executable, str(script)]
    base = dict(os.environ)
    if args.requeue:
        # One process, before any worker runs: the search marks the trials left running as interrupted and exits.
        result = subprocess.run(command, env={**base, ENV_REQUEUE: "1", ENV_WORKERS: "1", ENV_WORKER: "0"})
        sys.exit(result.returncode)
    # The workers' model runs share the CPUs: each pool gets a share, unless MAX_PROCESSES says otherwise.
    if "MAX_PROCESSES" not in base:
        base["MAX_PROCESSES"] = str(max(1, _usable_cpu_count() // args.workers))
    workers = [
        subprocess.Popen(command, env={**base, ENV_WORKERS: str(args.workers), ENV_WORKER: str(index)})
        for index in range(args.workers)
    ]
    codes = [worker.wait() for worker in workers]
    failed = [index for index, code in enumerate(codes) if code != 0]
    plain = {"markup": False, "highlight": False, "soft_wrap": True}
    if failed:
        console.print(f"Worker(s) {failed} of {args.workers} failed (exit codes {[codes[i] for i in failed]})", **plain)
        sys.exit(1)
    console.print(f"{args.workers} worker(s) finished {script}", **plain)


def cmd_results(args: argparse.Namespace) -> None:
    """`epf results convert|compact|verify SAVE_DIR`: what is on disk in a save_dir, in either results format."""
    from .results import parquet_store
    from .results.store import ResultsLockedError

    plain = {"markup": False, "highlight": False, "soft_wrap": True}
    save_dir = Path(args.save_dir)
    if not save_dir.is_dir():
        console.print(f"[red]Not a directory: {escape(str(save_dir))}[/red]")
        sys.exit(1)
    try:
        if args.results_cmd == "convert":
            done = parquet_store.convert(save_dir, args.to, model=args.model)
            for stem, n in done:
                console.print(f"{stem}: {n} records converted to {args.to}", **plain)
            if not done:
                console.print(
                    f"Nothing to convert: every model's results in {save_dir} are {args.to} already.", **plain
                )
        elif args.results_cmd == "compact":
            summary = parquet_store.compact(save_dir, model=args.model)
            for stem, counts in summary.items():
                console.print(
                    f"{stem}: {counts['parts_merged']} parts merged into {counts['parts_written']}, "
                    f"{counts['files_removed']} files removed",
                    **plain,
                )
            if not summary:
                console.print(f"No Parquet results in {save_dir}.", **plain)
        else:
            findings = parquet_store.verify(save_dir, model=args.model)
            for finding in findings:
                console.print(str(finding), **plain)
            errors = [f for f in findings if f.level == "error"]
            if errors:
                console.print(f"[red]{len(errors)} error(s).[/red]")
                sys.exit(1)
            console.print(f"[green]Results verified[/green] ({len(findings)} note(s)).")
    except (ValueError, FileNotFoundError, ResultsLockedError, RuntimeError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        sys.exit(1)


def _carded_classes(name: str) -> List[tuple]:
    """(how `epf weights` names it, class) for `name`: a model class registered in COMPONENT_REGISTRY, `module:Class`, or
    `all` (every registered class whose card has weights). SystemExit for a name that is none of them."""
    import importlib

    from .models._cards import class_path
    from .pipelines.model_pipeline import COMPONENT_REGISTRY

    registry = COMPONENT_REGISTRY["models"]
    if name == "all":
        found = [getattr(importlib.import_module(module), cls) for cls, module in registry.items()]
        return [(class_path(cls), cls) for cls in found if getattr(getattr(cls, "_card", None), "weights", None)]
    if ":" in name:
        module, _, qualname = name.partition(":")
        try:
            target: object = importlib.import_module(module)
            for part in qualname.split("."):
                target = getattr(target, part)
        except (ImportError, AttributeError) as exc:
            console.print(f"[red]Can't import {escape(name)}: {escape(str(exc))}[/red]")
            sys.exit(1)
        return [(name, target)]
    if name not in registry:
        console.print(
            f"[red]Unknown model class {escape(name)!r}.[/red] Give a registered model ({', '.join(sorted(registry))}), "
            "`module:Class` for one of your own, or `all`."
        )
        sys.exit(1)
    return [(name, getattr(importlib.import_module(registry[name]), name))]


def _card_of(label: str, cls: type, size: Optional[str]):
    """The card of `cls` to fetch: its `_card`, or with `--size` the one of `_card_variants` named so."""
    variants = getattr(cls, "_card_variants", None) or {}
    if size is not None:
        if size not in variants:
            sizes = ", ".join(sorted(variants)) if variants else "none (it has one card)"
            console.print(f"[red]{escape(label)} has no size {escape(size)!r}; its sizes: {escape(sizes)}[/red]")
            sys.exit(1)
        return f"{label} --size {size}", variants[size]
    return label, getattr(cls, "_card", None)


def cmd_weights(args: argparse.Namespace) -> None:
    """`epf weights list|fetch|verify`: the model weights a run reads, which only this command downloads."""
    from .models import _weights

    plain = {"markup": False, "highlight": False, "soft_wrap": True}
    root = _weights.weights_dir()
    if args.weights_cmd == "list":
        rows = _weights.cache_rows()
        if not rows:
            console.print(f"No model weights in {root}. Fetch them with `epf weights fetch <Model>`.", **plain)
            return
        from rich.table import Table

        table = Table(title=f"Weights in {root}")
        for column in ("weights", "family", "files", "size", "licence", "fetched"):
            table.add_column(column, overflow="fold")
        for row in rows:
            table.add_row(*row)
        console.print(table)
        return
    if args.weights_cmd == "verify":
        findings = _weights.verify_cache()
        for finding in findings:
            detail = f": {finding.detail}" if finding.detail else ""
            console.print(f"{finding.status:8} {finding.directory / finding.name}{detail}", **plain)
        bad = [f for f in findings if f.status in ("changed", "missing")]
        unlisted = [f for f in findings if f.status == "unlisted"]
        if bad:
            console.print(
                f"[red]{len(bad)} file(s) changed or missing.[/red] Fetch them again with `epf weights fetch`."
            )
        if unlisted:
            console.print(
                f"[red]{len(unlisted)} file(s) no fetch recorded[/red] next to the weights: see where they came from, and "
                "remove them if they are not yours."
            )
        if bad or unlisted:
            sys.exit(1)
        console.print(f"[green]{len(findings)} file(s) verified[/green] in {escape(str(root))}.")
        return

    accepted = list(args.accept or [])
    failed = 0
    classes = _carded_classes(args.model)
    if not classes:
        console.print("No registered model has weights to fetch.", **plain)
        return
    for label, cls in classes:
        what, card = _card_of(label, cls, args.size)
        if card is None or card.weights is None:
            console.print(f"{what} has no weights to fetch: it trains on the run's data.", **plain)
            continue
        try:
            plan = _weights.fetch_plan(card, what, args.revision)
        except ValueError as exc:  # a revision that isn't a branch, tag or commit
            console.print(f"[red]{escape(str(exc))}[/red]")
            failed += 1
            continue
        console.print(
            f"\n{what}: {plan.repo}@{plan.revision} ({len(plan.files) or 'all'} file(s), {_weights.size_text(plan.bytes)})",
            **plain,
        )
        console.print(f"  licence:     {plan.licence} ({plan.licence_url})", **plain)
        if not card.commercial_use:
            console.print("  commercial use: not allowed", **plain)
        if plan.revenue_cap:
            console.print(f"  revenue cap: {plan.revenue_cap}", **plain)
        if plan.attribution:
            console.print(f"  attribution: {plan.attribution}", **plain)
        console.print(f"  destination: {plan.destination}", **plain)
        accept = card.licence if card.licence in accepted else (accepted[0] if len(accepted) == 1 else None)
        try:
            sidecar = _weights.fetch(card, what, revision=args.revision, accept=accept)
        except (_weights.LicenceNotAcceptedError, _weights.WeightsMismatchError, _weights.WeightsDownloadError) as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")  # and on to the next class of `all`
            failed += 1
            continue
        except ValueError as exc:  # a card that can't be fetched as it is (a restricted licence with no id)
            console.print(f"[red]{escape(str(exc))}[/red]")
            failed += 1
            continue
        except ImportError as exc:  # huggingface_hub not installed: no other class can be fetched either
            console.print(f"[red]{escape(str(exc))}[/red]")
            sys.exit(1)
        hashes = {
            True: "sha256 verified against the card",
            False: "sha256 differs from the card's, recorded (runs warn: not the reference weights)",
            None: "sha256 recorded (the card pins none to verify)",
        }[sidecar.get("reference")]
        console.print(
            f"  fetched {len(sidecar['files'])} file(s), {_weights.size_text(sidecar['bytes'])}, {hashes}", **plain
        )
        if plan.revision != card.weights.revision:
            console.print(f"  a model runs on them with weights='{plan.repo}@{plan.revision}'", **plain)
    if failed:
        console.print(f"[red]{failed} fetch(es) failed.[/red]")
        sys.exit(1)


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
#   models_per_task: 50   # with model_index: task i runs the 50 models from i*50 on; `epf array` prints the range
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

    # -- array --
    array_p = sub.add_parser(
        "array", help="Print the task range of a SLURM array over a workflow's models, for sbatch --array="
    )
    array_p.add_argument("yaml", metavar="experiment.yaml", help="Path to workflow YAML")

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
    _add_horizon_arguments(model_p, 7)
    _add_issue_days_argument(model_p, "Issue forecasts")
    model_p.add_argument(
        "--freq",
        default="1h",
        choices=["1h", "15min", "1D"],
        help="Resolution of the data and forecasts: one model per hour, per quarter-hour, or per day on a frame with one "
        "row per date at one time of day (default: 1h)",
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
    _add_horizon_arguments(verify_p, 1)
    _add_issue_days_argument(verify_p, "Sample issue days")
    verify_p.add_argument(
        "--freq",
        default="1h",
        choices=["1h", "15min", "1D"],
        help="Resolution of the data and forecasts: one model per hour, per quarter-hour, or per day on a frame with one "
        "row per date at one time of day (default: 1h)",
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
    verify_p.add_argument(
        "--device",
        metavar="DEVICE",
        help="Run every model with an attribute named device on this one (e.g. cpu, to verify a GPU model where it is "
        "deterministic) and name them; models without one run as they are",
    )
    verify_p.add_argument(
        "--rerun",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run each issue day twice first and call a model whose forecasts differ inconclusive (default: for models "
        "that predict together, name their items or have a card)",
    )
    verify_p.add_argument(
        "--co-batch",
        dest="co_batch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Probe again over runs widened to the other items of each issue day's call (default: as --rerun)",
    )

    # -- results --
    results_p = sub.add_parser(
        "results", help="Convert results between JSONL and Parquet, compact Parquet results, or verify a save_dir"
    )
    results_sub = results_p.add_subparsers(dest="results_cmd", required=True, metavar="<action>")
    convert_p = results_sub.add_parser(
        "convert", help="Convert every model's results in SAVE_DIR to one format, record for record"
    )
    convert_p.add_argument("save_dir", metavar="SAVE_DIR", help="A model pipeline's save_dir")
    convert_p.add_argument("--to", required=True, choices=["parquet", "jsonl"], help="The format to convert to")
    convert_p.add_argument("--model", metavar="NAME", help="Convert this model's results only")
    compact_p = results_sub.add_parser(
        "compact", help="Merge each Parquet model's parts of a year into one and remove files no catalog lists"
    )
    compact_p.add_argument("save_dir", metavar="SAVE_DIR", help="A model pipeline's save_dir")
    compact_p.add_argument("--model", metavar="NAME", help="Compact this model's results only")
    verify_results_p = results_sub.add_parser(
        "verify", help="Check catalogs, part checksums, orphan files and repeated forecast keys (exit 1 on an error)"
    )
    verify_results_p.add_argument("save_dir", metavar="SAVE_DIR", help="A model pipeline's save_dir")
    verify_results_p.add_argument("--model", metavar="NAME", help="Verify this model's results only")

    # -- weights --
    weights_p = sub.add_parser(
        "weights",
        help="List, fetch or re-hash the model weights in the cache (EPFTOOLBOX2_WEIGHTS_DIR); a run never downloads",
    )
    weights_sub = weights_p.add_subparsers(dest="weights_cmd", required=True, metavar="<action>")
    weights_sub.add_parser("list", help="What is in the cache: repository, revision, files, size, licence, fetch time")
    fetch_p = weights_sub.add_parser(
        "fetch", help="Download a model's weights from Hugging Face into the cache and verify their sha256"
    )
    fetch_p.add_argument(
        "model", metavar="MODEL", help="A registered model class (T0Model), module:Class for one of your own, or all"
    )
    fetch_p.add_argument("--size", metavar="SIZE", help="Which of the class's sizes (its _card_variants), e.g. 209m")
    fetch_p.add_argument(
        "--revision",
        metavar="REV",
        help="A revision given explicitly (the card's, or another one, which a model then uses with "
        "weights='<repo>@<rev>'): files whose sha256 differs from the card's are recorded, not refused",
    )
    fetch_p.add_argument(
        "--accept",
        metavar="LICENCE",
        action="append",
        help="Accept this licence (its id as the card names it): needed for a non-commercial or revenue-capped one",
    )
    weights_sub.add_parser(
        "verify", help="Re-hash every file in the cache against its fetch record (exit 1 on a change)"
    )

    # -- tune --
    tune_p = sub.add_parser(
        "tune",
        help="Run a hyper-parameter search script (BayesSearch with storage=) in parallel worker processes",
        description=(
            "Start the script in N processes that share its BayesSearch's trials through the journal file of storage=. "
            "Each process runs the whole script: the search, then whatever follows (result.save writes the same file "
            "from each). MAX_PROCESSES, if not set, is the usable CPUs divided by N for each worker's model runs."
        ),
    )
    tune_p.add_argument("script", metavar="search.py", help="Python script that runs a BayesSearch with storage=")
    tune_p.add_argument("--workers", type=int, default=1, metavar="N", help="Processes sharing the search (default 1)")
    tune_p.add_argument(
        "--requeue",
        action="store_true",
        help="With no worker running: mark the trials a killed worker left running as interrupted, so a rerun redoes them",
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
        "array": cmd_array,
        "data": cmd_data,
        "model": cmd_model,
        "verify": cmd_verify,
        "results": cmd_results,
        "weights": cmd_weights,
        "tune": cmd_tune,
        "validate": cmd_validate,
        "init": cmd_init,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
