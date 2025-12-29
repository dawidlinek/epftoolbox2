from typing import List
from rich.console import Console
from rich.table import Table

from .base import Exporter
from ..results.report import EvaluationReport


class TerminalExporter(Exporter):
    def __init__(self, show: List[str] = None):
        """
        Args:
            show: Which tables to display. Options: "summary", "hour", "horizon",
                  "hour_horizon", "year", "year_horizon". Defaults to ["summary", "horizon"].
        """
        self.show = show or ["summary", "horizon"]

    def export(self, report: EvaluationReport) -> None:
        console = Console()

        if "summary" in self.show:
            console.print("\n[bold]Summary[/bold]")
            console.print(self._df_to_table(report.summary()))

        if "hour" in self.show:
            console.print("\n[bold]By Hour[/bold]")
            console.print(self._df_to_table(report.by_hour()))

        if "horizon" in self.show:
            console.print("\n[bold]By Horizon[/bold]")
            console.print(self._df_to_table(report.by_horizon()))

        if "hour_horizon" in self.show:
            console.print("\n[bold]By Hour × Horizon[/bold]")
            console.print(self._df_to_table(report.by_hour_horizon()))

        if "year" in self.show:
            console.print("\n[bold]By Year[/bold]")
            console.print(self._df_to_table(report.by_year()))

        if "year_horizon" in self.show:
            console.print("\n[bold]By Year × Horizon[/bold]")
            console.print(self._df_to_table(report.by_year_horizon()))

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
