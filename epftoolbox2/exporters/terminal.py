from typing import List
from rich.console import Console
from rich.table import Table

from .base import Exporter
from ..results.report import EvaluationReport


class TerminalExporter(Exporter):
    _SHOW_MAP = {
        "summary": ("Summary", lambda r: r.summary()),
        "hour": ("By Hour", lambda r: r.by_hour()),
        "horizon": ("By Horizon", lambda r: r.by_horizon()),
        "hour_horizon": ("By Hour × Horizon", lambda r: r.by_hour_horizon()),
        "year": ("By Year", lambda r: r.by_year()),
        "year_horizon": ("By Year × Horizon", lambda r: r.by_year_horizon()),
    }

    def __init__(self, show: List[str] = None):
        self.show = show or ["summary", "horizon"]

    def export(self, report: EvaluationReport) -> None:
        console = Console()
        if report.window != "test":
            console.print(f"\n[bold]The {report.window} window[/bold]")
        self._print_availability(console, report.availability())
        self._print_choices(console, report.choices())
        for key in self.show:
            if key in self._SHOW_MAP:
                title, fn = self._SHOW_MAP[key]
                console.print(f"\n[bold]{title}[/bold]")
                console.print(self._df_to_table(fn(report)))

    def _print_availability(self, console: Console, listing) -> None:
        """What the run can't vouch for comes first, whatever `show` asks for, and only if there is something to list."""
        if listing.empty:
            return
        console.print("\n[bold]Availability: inputs not known to be out at the issue time[/bold]")
        shown = listing.drop(columns="rule")  # `known` says when; the rule itself is in `report.availability()`
        shown = shown.loc[:, (shown != "").any()]  # `list` and `note` only where they say something
        if (shown["role"] == "predictor").all():
            shown = shown.drop(columns="role")
        table = Table()
        for col in shown.columns:
            table.add_column(str(col), overflow="fold")  # a predictor's name is no use cut short
        for _, row in shown.iterrows():
            table.add_row(*[str(value) for value in row])
        console.print(table)

    def _print_choices(self, console: Console, choices) -> None:
        """What was chosen on which days for which test period, whatever `show` asks for, if anything chose anything."""
        if choices.empty:
            return
        console.print("\n[bold]Choices made on validation[/bold]")
        shown = choices.assign(chosen=choices["chosen"].map(", ".join)).fillna("")
        table = Table()
        for col in shown.columns:
            table.add_column(str(col), overflow="fold")  # a member's name is no use cut short
        for _, row in shown.iterrows():
            table.add_row(*[self._format(value) for value in row])
        console.print(table)

    def _df_to_table(self, df) -> Table:
        table = Table()
        for col in df.columns:
            table.add_column(str(col))
        for _, row in df.iterrows():
            table.add_row(*[self._format(v) for v in row])
        return table

    def _format(self, value) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
