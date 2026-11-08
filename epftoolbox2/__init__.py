"""Energy Price Forecasting Toolbox 2

A modern Python library for electricity price forecasting with modular data pipelines
and model evaluation.
"""

import sys
import multiprocessing
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from epftoolbox2.pipelines import DataPipeline, ModelPipeline
from epftoolbox2.data.sources import EntsoeSource, OpenMeteoSource, CalendarSource

__version__ = "2.3.0"


def _print_extras_and_weights(console: Console) -> None:
    """The extras (installed, the versions of their modules) and the weights in the cache (`epf weights list`): what a run of a
    family with a card would find. Imports no extra."""
    from epftoolbox2._extras import extras_rows
    from epftoolbox2.models._weights import cache_rows, weights_dir

    extras = Table(title="Extras")
    for column in ("extra", "installed", "modules"):
        extras.add_column(column)
    for name, installed, modules in extras_rows():
        extras.add_row(name, "yes" if installed else "no", modules)
    console.print(extras)
    rows = cache_rows()
    if not rows:
        console.print(f"No model weights in {weights_dir()} (fetch them with `epf weights fetch <Model>`).")
        return
    weights = Table(title=f"Weights in {weights_dir()}")
    for column in ("weights", "family", "files", "size", "licence", "fetched"):
        weights.add_column(column, overflow="fold")
    for row in rows:
        weights.add_row(*row)
    console.print(weights)


def verify():
    console = Console()
    table = Table(show_header=False)

    table.add_row("Python Version", sys.version)

    gil_status = "🔒 Locked (Standard)"
    gil_info = ""

    if hasattr(sys, "_is_gil_enabled"):
        if not sys._is_gil_enabled():
            gil_status = "🔓 Unlocked (Free-threading enabled)"
            gil_info = "Free-threading is active!"
        else:
            gil_status = "🔒 Locked (Free-threading build, but GIL enabled)"
            if sys.platform == "win32":
                gil_info = "Run Python with: $env:PYTHON_GIL=0; python script.py (PowerShell) or set PYTHON_GIL=0 && python script.py (CMD)"
            else:
                gil_info = "Run Python with: PYTHON_GIL=0 python script.py"
    elif sys.version_info >= (3, 13):
        gil_info = "This Python build may not have free-threading support"

    table.add_row("GIL Status", gil_status)

    cpu_count = multiprocessing.cpu_count()
    table.add_row("Available CPU Cores", str(cpu_count))
    table.add_row("epftoolbox2 Version", __version__)
    table.add_row("Platform", sys.platform)

    console.print()
    console.print(Panel(table, title="[bold]EPFToolbox2[/bold]", border_style="blue"))
    _print_extras_and_weights(console)

    if gil_info:
        console.print(f"\n💡 [yellow]Info:[/yellow] {gil_info}")

    console.print("\n📚 [bold cyan]Resources:[/bold cyan]")
    console.print(
        "  • Documentation: [link=https://dawidlinek.github.io/epftoolbox2]https://dawidlinek.github.io/epftoolbox2[/link]"
    )
    console.print(
        "  • GitHub: [link=https://github.com/dawidlinek/epftoolbox2]https://github.com/dawidlinek/epftoolbox2[/link]"
    )
    console.print(
        "  • Issues: [link=https://github.com/dawidlinek/epftoolbox2/issues]https://github.com/dawidlinek/epftoolbox2/issues[/link]"
    )

    if hasattr(sys, "_is_gil_enabled") and sys._is_gil_enabled() and cpu_count > 1:
        console.print("\n⚡ [bold yellow]Performance Tip:[/bold yellow]")
        console.print(f"  Your system has {cpu_count} cores. For better multithreading performance,")
        if sys.platform == "win32":
            console.print("  run with free-threading enabled:")
            console.print("    PowerShell: $env:PYTHON_GIL=0; python your_script.py")
            console.print("    CMD: set PYTHON_GIL=0 && python your_script.py")
        else:
            console.print("  run with free-threading enabled: PYTHON_GIL=0 python your_script.py")

    console.print()


__all__ = [
    "verify",
    "DataPipeline",
    "ModelPipeline",
    "EntsoeSource",
    "OpenMeteoSource",
    "CalendarSource",
    "__version__",
]
