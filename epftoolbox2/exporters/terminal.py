from typing import List
from rich.console import Console
from rich.table import Table

from .base import Exporter
from ..results.report import EvaluationReport


class TerminalExporter(Exporter):
    _SHOW_MAP = {
        "summary":      ("Summary",           lambda r: r.summary()),
        "hour":         ("By Hour",            lambda r: r.by_hour()),
        "horizon":      ("By Horizon",         lambda r: r.by_horizon()),
        "hour_horizon": ("By Hour × Horizon",  lambda r: r.by_hour_horizon()),
        "year":         ("By Year",            lambda r: r.by_year()),
        "year_horizon": ("By Year × Horizon",  lambda r: r.by_year_horizon()),
    }

    def __init__(self, show: List[str] = None):
        self.show = show or ["summary", "horizon"]

    def export(self, report: EvaluationReport) -> None:
        console = Console()
        for key in self.show:
            if key in self._SHOW_MAP:
                title, fn = self._SHOW_MAP[key]
                console.print(f"\n[bold]{title}[/bold]")
                console.print(self._df_to_table(fn(report)))

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
