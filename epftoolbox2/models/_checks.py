"""Checks before export in a run (step 8f): `CheckGate` holds a member-day's forecasts until the whole day is back, checks it
(`results.checks.ForecastChecks`) and writes it, or writes `check_failed` skip records in its place.

A run's jobs bring a member-day back whole (a day, or a refit block) or in parts (an engine group's job is one period of a day),
so the gate counts what comes back of each (run day, horizon) against the run's tasks of it, read when the first job comes back:
`ShapeModel` and `ResidualModel` drop tasks in `_execute` before any job, so `_RunState.tasks` is final only then.

A member-day is checked when this run gives it a new record and it has a forecast, with the forecasts its keys have on file from
an earlier run into the same results (a resume after a crash, or keys a run skipped before): those are read once, for such days
alone. A failing day's forecasts of this run are never written, each key gets a `check_failed` skip record instead, and one on
file is overruled by a skip record one revision above it (`results.store._Store.overrule`): within one revision a forecast wins
over a skip record, so one written at its revision would leave the forecast read and a fallback unmoved. A day this run gives no
record isn't checked again: its records were checked when they were written, under the checks the manifest records and compares.
"""

from __future__ import annotations

import datetime
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from ..results.checks import CHECK_FAILED, ForecastChecks, failure_detail, planned_periods

if TYPE_CHECKING:  # pragma: no cover
    from .base import BaseModel, _RunState


class CheckGate:
    """The checks of one model's run: what each member-day has come back with, until it is whole."""

    def __init__(self, checks: ForecastChecks, freq: str, zone: Any, first_day: datetime.date):
        self.checks = checks
        self.planned = planned_periods(freq, zone)
        self.first_day = first_day
        self._expected: Optional[Counter] = None  # (day, horizon) -> the run's tasks of it, read at the first job
        self._returned: Counter = Counter()
        self._held: Dict[Tuple[int, int], List[Dict]] = {}
        self._on_file: Dict[Tuple[int, int], Dict[int, Optional[float]]] = {}

    def _run_date(self, day: int) -> datetime.date:
        return self.first_day + datetime.timedelta(days=day)

    def _start(self, model: "BaseModel", state: "_RunState", sink: Any) -> None:
        """What the run's tasks are of each member-day, and the forecasts on file of the member-days this run gives a record and
        that have some. Those without a task (every new record a skip decided before any fit) are checked by `finish`."""
        self._expected = Counter((task[2], task[1]) for task in state.tasks)
        store = state.store
        if store is None or not store.has_results():
            return
        pending: Dict[Tuple[int, int], Set[int]] = {}
        for period, horizon, day in state.pending:
            pending.setdefault((day, horizon), set()).add(period)
        wanted: Dict[Tuple[str, int], Tuple[Tuple[int, int], Set[int]]] = {}
        for (day, horizon), periods in pending.items():
            done = set(self.planned(self._run_date(day), horizon)) - periods
            if done:
                wanted[(self._run_date(day).isoformat(), horizon)] = ((day, horizon), done)
        if not wanted:
            return
        for record in store.iter_lines(cols=["hour", "horizon", "run_date", "prediction"]):
            found = wanted.get((record.get("run_date"), record.get("horizon")))
            if found is not None and record.get("hour") in found[1]:
                self._on_file.setdefault(found[0], {})[int(record["hour"])] = record.get("prediction")

    def collect(
        self, model: "BaseModel", state: "_RunState", results: List[Dict], skipped: List[Tuple[tuple, str]], sink: Any
    ) -> None:
        """What a job gave back (`BaseModel._collect`): its skips counted and written at once, its forecasts held by
        member-day, and each member-day this completes checked and written."""
        if self._expected is None:
            self._start(model, state, sink)
        touched = []
        for result in results:
            key = ((datetime.date.fromisoformat(result["run_date"]) - self.first_day).days, int(result["horizon"]))
            self._held.setdefault(key, []).append(result)
            self._returned[key] += 1
            touched.append(key)
        for (hour, hz, day), outcome in skipped:
            detail = outcome.partition(": ")[2]
            where = f"period {hour}, horizon {hz}, run date {self._run_date(day)}"
            state.skip((hour, hz, day), outcome, f"{where}: {detail}" if detail else where, sink)
            self._returned[(day, hz)] += 1
            touched.append((day, hz))
        for key in dict.fromkeys(touched):
            if self._returned[key] >= self._expected.get(key, 0):
                self._finish(key, state, sink)

    def finish(self, model: "BaseModel", state: "_RunState", sink: Any) -> None:
        """At the end of the run: every member-day still held checked with what it has (none is, when every task came back),
        and every one with forecasts on file and no task, from those alone."""
        if self._expected is None:
            self._start(model, state, sink)
        for key in sorted({*self._held, *self._on_file}):
            self._finish(key, state, sink)

    def _finish(self, key: Tuple[int, int], state: "_RunState", sink: Any) -> None:
        day, horizon = key
        held = self._held.pop(key, [])
        on_file = self._on_file.pop(key, {})
        predictions = dict(on_file)
        predictions.update({int(record["hour"]): record.get("prediction") for record in held})
        if not predictions:
            return
        found = self.checks.failures(predictions, self.planned(self._run_date(day), horizon))
        if not found:
            for record in held:
                sink.save(record)
            return
        outcome = f"{CHECK_FAILED}: {failure_detail(found)}"
        run_date = self._run_date(day)
        fresh = {int(record["hour"]) for record in held}
        for period in sorted(predictions):
            example = f"period {period}, horizon {horizon}, run date {run_date}: {failure_detail(found)}"
            state.skip((period, horizon, day), outcome, example, sink, overrule=period not in fresh)
