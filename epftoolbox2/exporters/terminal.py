from typing import List

import numpy as np
from rich.console import Console
from rich.table import Table

from .base import Exporter
from ..results.availability import HISTORY_COLUMNS, is_flagged
from ..results.report import EvaluationReport
from ..results.vintages import is_revised_class


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
        self._print_revised(console, report.vintages())
        self._print_thin_training(console, report.training_days())
        self._print_choices(console, report.choices())
        self._print_provenance(console, report.provenance())
        for key in self.show:
            if key in self._SHOW_MAP:
                title, fn = self._SHOW_MAP[key]
                console.print(f"\n[bold]{title}[/bold]")
                console.print(self._df_to_table(fn(report)))

    def _print_availability(self, console: Console, listing) -> None:
        """What the run can't vouch for comes first, whatever `show` asks for, and only if there is something to list: rows late
        or without a rule (`results.availability.is_flagged`). A sequence model's series read in time follow in a table of
        their own, with how far each is read (`newest`)."""
        flagged = np.array([is_flagged(row) for row in listing.to_dict("records")], dtype=bool)
        self._print_flagged(console, listing[flagged])
        self._print_series(console, listing[~flagged])

    def _print_flagged(self, console: Console, listing) -> None:
        if listing.empty:
            return
        console.print("\n[bold]Availability: inputs not known to be out at the issue time[/bold]")
        shown = listing.drop(columns="rule")  # `known` says when; the rule itself is in `report.availability()`
        shown = shown.loc[
            :, (shown != "").any()
        ]  # `list`, `note`, `newest` and `vintage` only where they say something
        if (shown["role"] == "predictor").all():
            shown = shown.drop(columns="role")
        table = Table()
        for col in shown.columns:
            table.add_column(str(col), overflow="fold")  # a predictor's name is no use cut short
        for _, row in shown.iterrows():
            table.add_row(*[str(value) for value in row])
        console.print(table)

    def _print_series(self, console: Console, series) -> None:
        """The series sequence models read in time (`models._history`): to which step, relative to the issue day, and from what
        archive; nothing for a run without such a model."""
        if series.empty:
            return
        console.print("\n[bold]History: series read to the newest step known at the issue time[/bold]")
        shown = series[["model", "predictor", "role", "horizons", *HISTORY_COLUMNS]]
        table = Table()
        for col in shown.columns:
            table.add_column(str(col), overflow="fold")
        for _, row in shown.iterrows():
            table.add_row(*[str(value) for value in row])
        console.print(table)

    def _print_revised(self, console: Console, vintages) -> None:
        """The inputs read from an archive of revised values, next to what the rules put late, whatever `show` asks for, and
        only if there are any: a backtest on them is more accurate than a live run would have been."""
        revised = vintages[vintages["class"].map(is_revised_class)] if len(vintages) else vintages
        if revised.empty:
            return
        console.print("\n[bold]Vintages: inputs read from an archive of revised values[/bold]")
        table = Table()
        for col in revised.columns:
            table.add_column(str(col), overflow="fold")
        for _, row in revised.iterrows():
            table.add_row(*[str(value) for value in row])
        console.print(table)

    def _print_thin_training(self, console: Console, training_days) -> None:
        """Forecasts on no more days than predictors, whatever `show` asks for, and only if there are any."""
        thin = training_days[training_days["thin_forecasts"] > 0]
        if thin.empty:
            return
        console.print("\n[bold]train_on='matching': forecasts on no more training days than predictors[/bold]")
        table = Table()
        for col in thin.columns:
            table.add_column(str(col), overflow="fold")
        for _, row in thin.iterrows():
            table.add_row(*[self._format(value) for value in row])
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

    def _print_provenance(self, console: Console, provenance) -> None:
        """Each carded model's licence, attribution, weights, runtime and contamination verdicts, whatever `show` asks for, and
        only if some model has a card: every report says what its models are under."""
        if provenance.empty:
            return
        console.print("\n[bold]Model provenance[/bold]")
        shown = provenance.drop(columns="licence_url")  # the licence names it; the URL is in `report.provenance()`
        shown = shown.loc[:, shown.notna().any() & (shown != "").any()]  # a column only where some model says something
        table = Table()
        for col in shown.columns:
            table.add_column(str(col), overflow="fold")
        for _, row in shown.iterrows():
            table.add_row(*["" if value is None or value != value else str(value) for value in row])
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
