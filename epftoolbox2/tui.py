"""Interactive wizard for the epf CLI.

Launched when `epf` is run with no subcommand. Guides the user through
building and running pipelines without writing YAML by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# ---------------------------------------------------------------------------
# Param schemas
# ---------------------------------------------------------------------------
# Each entry is a list of field dicts:
#   name, description, required, default, kind
# kind values:
#   "text"         – free text input
#   "int"          – integer input (validated)
#   "float"        – float input (validated)
#   "confirm"      – yes/no
#   "select"       – single choice from choices list
#   "multiselect"  – one or more choices (checkbox)
#   "comma_list"   – comma-separated text → list[str]
#   "column_select"– checkbox from available_columns (falls back to comma_list)

PARAM_SCHEMAS: dict[str, list[dict]] = {
    "EntsoeSource": [
        {"name": "country_code", "description": "2-letter country code (e.g. PL, DE)", "required": True, "default": None, "kind": "text"},
        {"name": "api_key", "description": "ENTSOE API key", "required": True, "default": None, "kind": "text"},
        {"name": "type", "description": "Data types to fetch", "required": True, "default": None, "kind": "multiselect",
         "choices": ["load", "generation", "price"]},
    ],
    "OpenMeteoSource": [
        {"name": "latitude", "description": "Latitude (-90 to 90)", "required": True, "default": None, "kind": "float"},
        {"name": "longitude", "description": "Longitude (-180 to 180)", "required": True, "default": None, "kind": "float"},
        {"name": "horizon", "description": "Forecast horizon in days", "required": False, "default": "7", "kind": "int"},
        {"name": "prefix", "description": "Column name prefix (optional)", "required": False, "default": "", "kind": "text"},
    ],
    "CsvSource": [
        {"name": "file_path", "description": "Path to CSV file", "required": True, "default": None, "kind": "text"},
        {"name": "datetime_column", "description": "Datetime column name", "required": False, "default": "datetime", "kind": "text"},
        {"name": "separator", "description": "CSV separator", "required": False, "default": ",", "kind": "text"},
        {"name": "prefix", "description": "Column name prefix (optional)", "required": False, "default": "", "kind": "text"},
    ],
    "CalendarSource": [
        {"name": "country", "description": "Country code", "required": True, "default": None, "kind": "select",
         "choices": ["PL", "DE", "FR", "ES", "IT", "NL", "BE", "AT", "CH", "CZ", "SK", "HU", "RO", "BG", "GR", "PT", "DK", "SE", "NO", "FI", "GB", "IE"]},
        {"name": "holidays", "description": "Holiday encoding", "required": False, "default": "binary", "kind": "select",
         "choices": ["binary", "onehot", "name", "false"]},
        {"name": "weekday", "description": "Weekday encoding", "required": False, "default": "number", "kind": "select",
         "choices": ["number", "onehot", "name", "false"]},
        {"name": "hour", "description": "Hour encoding", "required": False, "default": "false", "kind": "select",
         "choices": ["false", "number", "onehot"]},
        {"name": "month", "description": "Month encoding", "required": False, "default": "false", "kind": "select",
         "choices": ["false", "number", "onehot", "name"]},
        {"name": "daylight", "description": "Include sunrise/sunset/daylight hours", "required": False, "default": False, "kind": "confirm"},
        {"name": "prefix", "description": "Column name prefix (optional)", "required": False, "default": "", "kind": "text"},
    ],
    "TimezoneTransformer": [
        {"name": "target_tz", "description": "IANA timezone (e.g. Europe/Warsaw)", "required": True, "default": None, "kind": "text"},
    ],
    "ResampleTransformer": [
        {"name": "freq", "description": "Target frequency (e.g. 1h, 15min)", "required": False, "default": "1h", "kind": "text"},
        {"name": "method", "description": "Interpolation method", "required": False, "default": "linear", "kind": "select",
         "choices": ["linear", "ffill", "bfill"]},
    ],
    "LagTransformer": [
        {"name": "columns", "description": "Columns to lag", "required": True, "default": None, "kind": "column_select"},
        {"name": "lags", "description": "Lag values, comma-separated integers (e.g. 1,2,24 or -7,-1)", "required": False, "default": "1", "kind": "lag_list"},
        {"name": "freq", "description": "Frequency (e.g. 1h, day)", "required": False, "default": "1h", "kind": "text"},
    ],
    "ContinuityValidator": [
        {"name": "freq", "description": "Expected frequency (e.g. 1h)", "required": False, "default": "1h", "kind": "text"},
    ],
    "NullCheckValidator": [
        {"name": "columns", "description": "Columns to check (comma-separated, blank = all)", "required": False, "default": "", "kind": "column_select"},
        {"name": "allow_nulls", "description": "Allow nulls (just warn)?", "required": False, "default": False, "kind": "confirm"},
    ],
    "EdaValidator": [
        {"name": "columns", "description": "Columns to analyse (comma-separated, blank = auto)", "required": False, "default": "", "kind": "column_select"},
    ],
    "OLSModel": [
        {"name": "name", "description": "Model display name", "required": False, "default": "OLS", "kind": "text"},
        {"name": "predictors", "description": "Predictor column names", "required": True, "default": None, "kind": "column_select"},
        {"name": "training_window", "description": "Training window in days", "required": False, "default": "365", "kind": "int"},
    ],
    "LassoCVModel": [
        {"name": "name", "description": "Model display name", "required": False, "default": "LassoCV", "kind": "text"},
        {"name": "predictors", "description": "Predictor column names", "required": True, "default": None, "kind": "column_select"},
        {"name": "training_window", "description": "Training window in days", "required": False, "default": "365", "kind": "int"},
        {"name": "cv", "description": "Cross-validation folds", "required": False, "default": "5", "kind": "int"},
    ],
    "MAEEvaluator": [],
    "ExcelExporter": [
        {"name": "path", "description": "Output Excel file path", "required": True, "default": None, "kind": "text"},
        {"name": "sheets", "description": "Sheets to include", "required": False, "default": None, "kind": "multiselect",
         "choices": ["summary", "hour", "horizon", "hour_horizon", "year", "year_horizon", "run_weekday_horizon", "target_weekday_horizon", "details"]},
    ],
    "TerminalExporter": [
        {"name": "show", "description": "Sections to display", "required": False, "default": None, "kind": "multiselect",
         "choices": ["summary", "hour", "horizon", "hour_horizon", "year", "year_horizon"]},
    ],
}


# ---------------------------------------------------------------------------
# Low-level prompt helpers
# ---------------------------------------------------------------------------

def _ask_field(field: dict, available_columns: Optional[list[str]] = None) -> Any:
    """Prompt the user for a single parameter field. Returns the value or sentinel SKIP."""
    name = field["name"]
    desc = field["description"]
    kind = field["kind"]
    default = field["default"]
    required = field["required"]
    choices = field.get("choices", [])

    label = f"  {desc}" + ("" if required else f" (default: {default!r})")

    if kind == "confirm":
        val = questionary.confirm(label, default=bool(default)).ask()
        if val is None:
            _abort()
        return val

    if kind == "select":
        val = questionary.select(label, choices=choices, default=default).ask()
        if val is None:
            _abort()
        # "false" string → Python False
        return False if val == "false" else val

    if kind == "multiselect":
        val = questionary.checkbox(label, choices=choices).ask()
        if val is None:
            _abort()
        return val if val else None

    if kind == "lag_list":
        while True:
            raw = questionary.text(label, default=str(default) if default is not None else "1").ask()
            if raw is None:
                _abort()
            raw = raw.strip()
            if not raw:
                raw = str(default)
            try:
                result = [int(x.strip()) for x in raw.split(",") if x.strip()]
                if not result:
                    raise ValueError
                return result
            except ValueError:
                console.print("[red]Please enter comma-separated integers, e.g. 1,2,7 or -7,-1,0[/red]")

    if kind == "column_select":
        if available_columns:
            # Required fields (predictors/columns) must pick at least one
            val = questionary.checkbox(label, choices=available_columns).ask()
            if val is None:
                _abort()
            if not val and required:
                console.print("[red]At least one column is required.[/red]")
                return _ask_field(field, available_columns)
            return val if val else None
        else:
            # Fall back to comma-separated text
            val = questionary.text(label, default=default or "").ask()
            if val is None:
                _abort()
            val = val.strip()
            if not val:
                if required:
                    console.print("[red]At least one column is required.[/red]")
                    return _ask_field(field, available_columns)
                return None
            return [c.strip() for c in val.split(",") if c.strip()]

    # text / int / float
    raw = questionary.text(label, default=str(default) if default is not None else "").ask()
    if raw is None:
        _abort()
    raw = raw.strip()

    if not raw:
        if required:
            console.print("[red]This field is required.[/red]")
            return _ask_field(field, available_columns)
        return default

    if kind == "int":
        try:
            return int(raw)
        except ValueError:
            console.print("[red]Please enter a valid integer.[/red]")
            return _ask_field(field, available_columns)

    if kind == "float":
        try:
            return float(raw)
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")
            return _ask_field(field, available_columns)

    return raw


def _abort() -> None:
    console.print("\n[yellow]Cancelled.[/yellow]")
    sys.exit(0)


def _ask_date(label: str, default: str = "") -> str:
    import re
    while True:
        val = questionary.text(label, default=default).ask()
        if val is None:
            _abort()
        val = val.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", val):
            return val
        console.print("[red]Expected format: YYYY-MM-DD[/red]")


# ---------------------------------------------------------------------------
# Component configuration
# ---------------------------------------------------------------------------

def configure_component(class_name: str, available_columns: Optional[list[str]] = None) -> dict:
    """Ask the user to fill in all params for a component. Returns a params dict."""
    schema = PARAM_SCHEMAS.get(class_name, [])
    params: dict[str, Any] = {}
    for field in schema:
        val = _ask_field(field, available_columns=available_columns)
        if val is not None and val != "":
            params[field["name"]] = val
    return params


def _instantiate(class_name: str, params: dict) -> Any:
    """Import and instantiate a component by class name using the pipeline registries."""
    from .pipelines.data_pipeline import COMPONENT_REGISTRY as DR
    from .pipelines.model_pipeline import COMPONENT_REGISTRY as MR

    all_registries = {**DR.get("sources", {}), **DR.get("transformers", {}), **DR.get("validators", {}),
                      **MR.get("models", {}), **MR.get("evaluators", {}), **MR.get("exporters", {})}
    if class_name not in all_registries:
        raise ValueError(f"Unknown component: {class_name}")
    import importlib
    module = importlib.import_module(all_registries[class_name])
    return getattr(module, class_name)(**params)


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

_SOURCES = ["EntsoeSource", "OpenMeteoSource", "CsvSource", "CalendarSource"]
_TRANSFORMERS = ["TimezoneTransformer", "ResampleTransformer", "LagTransformer"]
_VALIDATORS = ["ContinuityValidator", "NullCheckValidator", "EdaValidator"]
_MODELS = ["OLSModel", "LassoCVModel"]
_EVALUATORS = ["MAEEvaluator"]
_EXPORTERS = ["TerminalExporter", "ExcelExporter"]


def _build_component_list(choices: list[str], label: str, available_columns: Optional[list[str]] = None) -> list[Any]:
    """Generic loop: select component type → configure → add another?"""
    components = []
    while True:
        class_name = questionary.select(f"  Select {label} type", choices=choices).ask()
        if class_name is None:
            _abort()
        console.print(f"  [bold]Configure {class_name}[/bold]")
        params = configure_component(class_name, available_columns=available_columns)
        components.append(_instantiate(class_name, params))
        again = questionary.confirm(f"  Add another {label}?", default=False).ask()
        if again is None:
            _abort()
        if not again:
            break
    return components


def build_data_pipeline_interactive():  # noqa: ANN201
    from .pipelines.data_pipeline import DataPipeline

    console.print("\n[bold cyan]── Data Pipeline ──────────────────────────[/bold cyan]")

    # Sources (at least one required)
    console.print("\n[bold]Step 1/6: Sources[/bold]")
    sources = _build_component_list(_SOURCES, "source")

    # Transformers (optional)
    console.print("\n[bold]Step 2/6: Transformers[/bold]")
    transformers = []
    if questionary.confirm("  Add transformers?", default=False).ask():
        transformers = _build_component_list(_TRANSFORMERS, "transformer")

    # Validators (optional)
    console.print("\n[bold]Step 3/6: Validators[/bold]")
    validators = []
    if questionary.confirm("  Add validators?", default=False).ask():
        validators = _build_component_list(_VALIDATORS, "validator")

    # Run config
    console.print("\n[bold]Step 4/6: Date range[/bold]")
    start = _ask_date("  Start date (YYYY-MM-DD)")
    end = _ask_date("  End date   (YYYY-MM-DD)")

    cache_enabled = questionary.confirm("  Enable source caching?", default=False).ask()
    if cache_enabled is None:
        _abort()
    cache: Any = False
    if cache_enabled:
        cache_path = questionary.text("  Cache path (leave blank for automatic)").ask()
        if cache_path is None:
            _abort()
        cache = cache_path.strip() if cache_path.strip() else True

    # Output
    console.print("\n[bold]Step 5/6: Output[/bold]")
    while True:
        output = questionary.text("  Output CSV path").ask()
        if output is None:
            _abort()
        output = output.strip()
        if output:
            break
        console.print("  [red]Output path is required.[/red]")

    dp = DataPipeline(sources=sources, transformers=transformers, validators=validators)

    # Save YAML?
    console.print("\n[bold]Step 6/6: Save & run[/bold]")
    if questionary.confirm("  Save pipeline YAML?", default=True).ask():
        yaml_path = questionary.text("  YAML path", default="data_pipeline.yaml").ask()
        if yaml_path is None:
            _abort()
        dp.save(yaml_path)
        console.print(f"  [green]Saved →[/green] {yaml_path}")

    if questionary.confirm("  Run now?", default=True).ask():
        df = dp.run(start=start, end=end, cache=cache)
        if df.empty:
            console.print("[yellow]Warning: pipeline returned an empty DataFrame.[/yellow]")
        df.to_csv(output)
        console.print(f"[green]Saved {len(df)} rows →[/green] {output}")


def build_model_pipeline_interactive() -> None:
    import pandas as pd
    from .pipelines.model_pipeline import ModelPipeline

    console.print("\n[bold cyan]── Model Pipeline ─────────────────────────[/bold cyan]")

    # Data CSV
    console.print("\n[bold]Step 1/7: Input data[/bold]")
    while True:
        data_path = questionary.text("  Input CSV path (from epf data)").ask()
        if data_path is None:
            _abort()
        p = Path(data_path.strip())
        if p.exists():
            break
        console.print(f"  [red]File not found: {p}[/red]")

    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    available_columns = list(df.columns)
    console.print(f"  Loaded {len(df)} rows, columns: {', '.join(available_columns)}")

    # Models
    console.print("\n[bold]Step 2/7: Models[/bold]")
    models = _build_component_list(_MODELS, "model", available_columns=available_columns)

    # Evaluators
    console.print("\n[bold]Step 3/7: Evaluators[/bold]")
    evaluators = []
    if questionary.confirm("  Add evaluators?", default=True).ask():
        evaluators = _build_component_list(_EVALUATORS, "evaluator")

    # Exporters
    console.print("\n[bold]Step 4/7: Exporters[/bold]")
    exporters = []
    if questionary.confirm("  Add exporters?", default=True).ask():
        exporters = _build_component_list(_EXPORTERS, "exporter")

    # Run config
    console.print("\n[bold]Step 5/7: Test period[/bold]")
    test_start = _ask_date("  Test start (YYYY-MM-DD)")
    test_end = _ask_date("  Test end   (YYYY-MM-DD)")

    target = questionary.select("  Target column", choices=available_columns).ask()
    if target is None:
        _abort()

    horizon_raw = questionary.text("  Forecast horizon (days)", default="7").ask()
    if horizon_raw is None:
        _abort()
    horizon = int(horizon_raw.strip())

    forecast_only = questionary.confirm("  Forecast-only mode (skip evaluation)?", default=False).ask()
    if forecast_only is None:
        _abort()

    save_dir_raw = questionary.text("  Save predictions to directory (blank to skip)").ask()
    if save_dir_raw is None:
        _abort()
    save_dir = save_dir_raw.strip() or None

    mp = ModelPipeline()
    for m in models:
        mp.add_model(m)
    for e in evaluators:
        mp.add_evaluator(e)
    for ex in exporters:
        mp.add_exporter(ex)

    # Save YAML?
    console.print("\n[bold]Step 6/7: Save YAML[/bold]")
    if questionary.confirm("  Save pipeline YAML?", default=True).ask():
        yaml_path = questionary.text("  YAML path", default="model_pipeline.yaml").ask()
        if yaml_path is None:
            _abort()
        mp.save(yaml_path)
        console.print(f"  [green]Saved →[/green] {yaml_path}")

    # Run
    console.print("\n[bold]Step 7/7: Run[/bold]")
    if questionary.confirm("  Run now?", default=True).ask():
        mp.run(
            data=df,
            test_start=test_start,
            test_end=test_end,
            target=target,
            horizon=horizon,
            save_dir=save_dir,
            forecast_only=forecast_only,
        )


# ---------------------------------------------------------------------------
# YAML-based runners (thin wrappers used from main menu)
# ---------------------------------------------------------------------------

def _run_experiment_from_yaml() -> None:
    path = questionary.text("  Workflow YAML path").ask()
    if path is None:
        _abort()
    from .pipelines.workflow import Workflow
    Workflow.load(path.strip()).run()


def _run_data_from_yaml() -> None:
    import pandas as pd
    from .pipelines.data_pipeline import DataPipeline

    path = questionary.text("  DataPipeline YAML path").ask()
    if path is None:
        _abort()
    start = _ask_date("  Start date (YYYY-MM-DD)")
    end = _ask_date("  End date   (YYYY-MM-DD)")
    output = questionary.text("  Output CSV path").ask()
    if output is None:
        _abort()

    dp = DataPipeline.load(path.strip())
    df = dp.run(start=start, end=end)
    df.to_csv(output.strip())
    console.print(f"[green]Saved {len(df)} rows →[/green] {output.strip()}")


def _run_model_from_yaml() -> None:
    import pandas as pd
    from .pipelines.model_pipeline import ModelPipeline

    path = questionary.text("  ModelPipeline YAML path").ask()
    if path is None:
        _abort()

    while True:
        data_path = questionary.text("  Input CSV path").ask()
        if data_path is None:
            _abort()
        p = Path(data_path.strip())
        if p.exists():
            break
        console.print(f"  [red]File not found: {p}[/red]")

    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)

    test_start = _ask_date("  Test start (YYYY-MM-DD)")
    test_end = _ask_date("  Test end   (YYYY-MM-DD)")
    target = questionary.text("  Target column", default="price").ask()
    if target is None:
        _abort()
    horizon_raw = questionary.text("  Forecast horizon (days)", default="7").ask()
    if horizon_raw is None:
        _abort()

    mp = ModelPipeline.load(path.strip())
    mp.run(data=df, test_start=test_start, test_end=test_end, target=target.strip(), horizon=int(horizon_raw.strip()))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_MENU_CHOICES = [
    questionary.Choice("Run experiment from YAML",             value="run_experiment"),
    questionary.Choice("Run data pipeline from YAML → CSV",    value="run_data"),
    questionary.Choice("Run model pipeline from YAML",         value="run_model"),
    questionary.Separator(),
    questionary.Choice("Build & run data pipeline interactively",  value="build_data"),
    questionary.Choice("Build & run model pipeline interactively", value="build_model"),
    questionary.Separator(),
    questionary.Choice("Exit",                                 value="exit"),
]

_DISPATCH = {
    "run_experiment": _run_experiment_from_yaml,
    "run_data":       _run_data_from_yaml,
    "run_model":      _run_model_from_yaml,
    "build_data":     build_data_pipeline_interactive,
    "build_model":    build_model_pipeline_interactive,
}


def run_interactive() -> None:
    banner = Text("EPF Toolbox 2", style="bold cyan", justify="center")
    console.print(Panel(banner, subtitle="electricity price forecasting", padding=(0, 4)))

    while True:
        action = questionary.select("What do you want to do?", choices=_MENU_CHOICES).ask()
        if action is None or action == "exit":
            console.print("[dim]Bye.[/dim]")
            break
        try:
            _DISPATCH[action]()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            break
        except SystemExit:
            raise
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
