from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .store import OK_ONLY, STATUS_OK, is_parquet_path, iter_file_records, read_records, visible_revisions
from .windows import SKIP_TARGET_IN_TEST, TEST, VALIDATION, RunWindows

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

# What `_on_the_clock` reads of a record.
_CLOCK_FIELDS = ("hour", "run_date", "target_date", "freq")
# What `_records`' filter and `_as_read` read of a record besides the columns asked for (a Parquet directory reads these alone).
_FILTER_FIELDS = ("run_date", "target_date", "run_weekday", "horizon", "hour", "window", "freq", "day_in_test")
# 1970-01-01, day 0 of a Parquet date column, and its weekday (a Thursday).
_EPOCH = date(1970, 1, 1)


@dataclass
class ModelResultRef:
    """Where one model's results live and how many forecasts the run planned and produced.

    `count` is the number of planned forecasts (`IssueCalendar`: periods x horizons x test days, without the slots the clock
    doesn't have on an issue day or a target day). `produced` is how many exist after the run (including ones resumed from an
    earlier run), and `skipped` counts the tasks that gave no forecast by reason (e.g. `missing_test_predictors`). `produced` is None for refs built from
    raw result lists. `predictor_lists` describes each distinct list a predictor function of `ctx` returned, with how
    many forecasts use it where; it is None for fixed lists. `availability_listing` holds the rows of inputs known after
    the issue time (`epftoolbox2.results.availability`), which `EvaluationReport.availability()` shows. `matching_lists` holds,
    for a function with `train_on="matching"` lists, each such list's hash and number of predictors, which is what
    `EvaluationReport.training_days()` reads records for; it is None otherwise. A report merged from stored results reads
    `skipped` back from the manifests.

    In a run with validation (see `epftoolbox2.results.windows`) a ref of one window has `window`, and all of the above are
    about that window alone: `test_start` and `test_end` are its first and last issue day, and only records of that window
    are read. `target_before` is the first test issue day of a validation ref, whose forecasts for that day or later are
    left out and counted in `skipped` as `target_in_test`. `windows` holds what the run was made of, and `by_window` the
    refs of its windows on the ref of the whole run. All four stay at their defaults in a plain run.

    `test_only` marks an ensemble that has forecasts in the test window alone, which a report of a run with validation doesn't
    look for in `.validation` (see `epftoolbox2.ensembles`).

    `zone` is the time zone of the frame the forecasts came from, which says what periods a date has on the clock: the frame's
    `index.tz`, or a manifest's `data_timezone` in a report made from files. It is None for a naive index and for results that
    don't say, such as plain lists or a results file with no manifest. With a zone, a record of a slot the clock doesn't have
    on its run date or its target date is no forecast, as its target can't exist: `iter_records` leaves it out, and
    `off_the_clock` counts it. Only results written before training rows were chosen by date hold one.

    `horizon` is the largest horizon planned, and `horizons` the ones planned: a run's ref has them when they aren't every one
    from 1 to `horizon` (`run(horizons=[1, 3, 7])`), and a ref of `ModelPipeline.report()` always. None reads every horizon.
    Only records of those horizons are read: `count` plans no other, and a record of another would be compared with nothing.

    `issue_days` are the weekdays a run issued on (`run(issue_days=["friday"])`), as `date.weekday()` numbers, None for every
    day. Only records issued on them are read (their `run_weekday`), for the same reason.

    `vintages` holds the vintage class of each input (`epftoolbox2.results.vintages`), which `EvaluationReport.vintages()` shows
    and the comparisons read; None when no input has a class, or without an information set. An ensemble's holds its models'.

    `provenance` is a carded model's `card`, `runtime` and `weights` as its manifest records them (`models._cards.run_provenance`),
    which `EvaluationReport.provenance()` shows; None for a model without a card and for an ensemble.

    Results in a live directory (step 8c, `results.live`) are revisions, read once committed: `policy` says which of an issue
    day's committed revisions a key is read from ("latest", the default there, the newest; "first"; "delivered", up to the
    newest delivered one), of those committed at or before `cutoff` ("deadline", the default: the deadline `live.json` records,
    on each issue day; None for every committed revision; an instant, "HH:MM", a mapping or a function of the issue day). An
    uncommitted revision, or one committed after the cut-off, is never read. A `policy` on a backtest's results raises.
    """

    name: str
    path: Optional[Path]
    count: int
    test_start: str
    test_end: str
    horizon: int
    freq: str = "1h"
    _results: Optional[List[Dict]] = field(default=None, repr=False)
    produced: Optional[int] = None
    skipped: Dict[str, int] = field(default_factory=dict)
    predictor_lists: Optional[List[Dict]] = None
    availability_listing: Optional[List[Dict]] = None
    window: Optional[str] = None
    target_before: Optional[str] = None
    windows: Optional[RunWindows] = None
    by_window: Optional[Dict[str, "ModelResultRef"]] = field(default=None, repr=False)
    test_only: bool = False
    matching_lists: Optional[Dict[str, int]] = None
    zone: Any = None
    horizons: Optional[Tuple[int, ...]] = None
    issue_days: Optional[Tuple[int, ...]] = None
    vintages: Optional[List[Dict]] = None
    provenance: Optional[Dict[str, Any]] = None
    policy: Optional[str] = None
    cutoff: Any = "deadline"

    def frame(self, columns: Sequence[str]) -> Optional["pd.DataFrame"]:
        """The forecasts `iter_records(columns)` gives, as a DataFrame of `columns` read without a dict per record, for results
        in a Parquet directory (`results.parquet_store.read_frame`): the same rule, filters and clock, and each column as a
        list of the records' values would make it (`parquet_store._series`). None for any other results (a JSONL file, results
        held in memory), which `iter_records` reads. Internal: `EvaluationReport._model_frame` and the counts of
        `ModelPipeline.report` read through it."""
        return self._frame(columns, clock=True)

    def count_records(self) -> int:
        """How many forecasts `iter_records` gives, through `frame` where it can."""
        frame = self.frame(["hour"])
        return len(frame) if frame is not None else sum(1 for _ in self.iter_records(cols=["hour"]))

    def _frame(self, columns: Sequence[str], clock: bool) -> Optional["pd.DataFrame"]:
        committed = self._committed()
        if self.path is None or not is_parquet_path(self.path) or not Path(self.path).exists():
            return None
        from .parquet_store import read_frame

        return read_frame(
            Path(self.path),
            list(columns),
            self._keep_rows(clock),
            extra=_FILTER_FIELDS,
            visible=visible_revisions(committed),
        )

    def _committed(self) -> Optional[Dict[str, Dict[int, Any]]]:
        """The revisions this ref may read of a live directory's results (`policy`, `cutoff`; `results.live.live_reading`),
        for `read_records(committed=...)`; None for a backtest's or results held in memory."""
        from .live import live_reading

        return live_reading(self.path, self.policy, self.cutoff)

    def _keep_rows(self, clock: bool) -> Callable[[Any, Any], Any]:
        """`_records`' filter (and the clock's, with `clock` and a zone) over rows of a Parquet table: a boolean per row."""
        import numpy as np
        import pyarrow as pa
        import pyarrow.compute as pc

        def days_of(text: str) -> int:
            return (date.fromisoformat(text) - _EPOCH).days

        def keep(table: Any, rows: Any) -> Any:
            mask = np.ones(len(rows), dtype=bool)
            if not len(rows):
                return mask
            index = pa.array(rows, pa.int64())

            def column(name: str) -> Any:
                return table.column(name).take(index)

            def days(name: str) -> Any:
                return column(name).cast(pa.int32()).to_numpy(zero_copy_only=False)

            def text(name: str, missing: str) -> Any:
                return np.asarray(pc.fill_null(column(name).cast(pa.string()), missing).to_pylist(), dtype=object)

            run_days = days("run_date")
            if self.test_start and self.test_end:
                mask &= (run_days >= days_of(self.test_start)) & (run_days <= days_of(self.test_end))
            if self.horizons is not None:
                mask &= np.isin(column("horizon").to_numpy(zero_copy_only=False), list(self.horizons))
            if self.issue_days is not None:
                weekday = column("run_weekday").to_numpy(zero_copy_only=False)
                weekday = (
                    np.where(np.isnan(weekday), (run_days + 3) % 7, weekday) if weekday.dtype.kind == "f" else weekday
                )
                mask &= np.isin(weekday, list(self.issue_days))
            if self.window is not None:
                mask &= text("window", TEST) == self.window
            if self.window == VALIDATION and self.windows is not None and self.windows.validation_every != 1:
                # every k-th day of the validation window alone, as `_records` keeps them
                offset = run_days - days_of(self.windows.validation[0])
                mask &= offset % self.windows.validation_every == 0
            if self.target_before is not None:
                mask &= days("target_date") < days_of(self.target_before)
            if clock and self.zone is not None:
                mask &= self._on_the_clock_rows(column, days, text)
            return mask

        return keep

    def _on_the_clock_rows(self, column: Callable, days: Callable, text: Callable) -> Any:
        """`_on_the_clock` over rows: False where the record's period is one its run date or target date lacks."""
        import numpy as np

        from ..models._calendar import FREQ_STEPS, skipped_periods

        freqs = text("freq", self.freq)
        hours = column("hour").to_numpy(zero_copy_only=False)
        on = np.ones(len(freqs), dtype=bool)
        for freq in set(freqs.tolist()):
            if freq not in FREQ_STEPS:
                continue
            of_freq = freqs == freq
            for day_numbers in (days("run_date"), days("target_date")):
                for day in np.unique(day_numbers[of_freq]).tolist():
                    skipped = skipped_periods((_EPOCH + timedelta(days=int(day))).isoformat(), freq, self.zone)
                    if skipped:
                        on &= ~(of_freq & (day_numbers == day) & np.isin(hours, list(skipped)))
        return on

    def iter_records(self, cols: Optional[List[str]] = None) -> Iterator[Dict]:
        """The forecasts of this ref, wherever they live: records with `status` "ok" or without one, a skip record never, and from
        a file one per key (`results.store.read_records`: the highest revision, the first forecast).

        A resumed JSONL can hold rows from an earlier run with a wider or different test window (its rows are never
        deleted), and a run with validation holds both windows, so each ref keeps to its own. Refs built from plain result
        lists have no window (test_start/test_end are "") and are read as given.

        A window's ref gives `day_in_test` from the window's start as it is now: a record resumed after the start moved
        holds the old count in the file. A record of a slot the clock doesn't have is left out, and so is one of a horizon
        `horizons` doesn't list or issued on a weekday `issue_days` doesn't list (see the class docstring).
        """
        return (read for read, _ in self._records(cols, clock=self._on_the_clock()))

    def off_the_clock(self) -> int:
        """How many records of this ref are of a slot the clock doesn't have, which `iter_records` leaves out. Reads what the
        clock needs of each record, never its coefficients."""
        on_the_clock = self._on_the_clock()
        if on_the_clock is None:
            return 0
        every = self._frame(["hour"], clock=False)
        if every is not None:  # Parquet: the rows the clock leaves out
            return len(every) - len(self._frame(["hour"], clock=True))
        return sum(1 for r, _ in self._records(list(_CLOCK_FIELDS)) if not on_the_clock(r))

    def first_record(self, cols: Optional[List[str]] = None) -> Optional[Dict]:
        """The first forecast of this ref in file order, as `iter_records` reads it, without reading the rest of the file: for
        what every record of a file shares, such as `freq`. None if it has none."""
        read = next(self._records(cols, clock=self._on_the_clock(), first=True), None)
        return None if read is None else read[0]

    def _on_the_clock(self) -> Optional[Callable[[Dict], bool]]:
        if self.zone is None:
            return None
        from ..models._calendar import FREQ_STEPS, skipped_periods  # the models import this module

        def on_the_clock(r: Dict) -> bool:
            freq = r.get("freq", self.freq)
            if freq not in FREQ_STEPS:
                return True
            return not (
                r["hour"] in skipped_periods(r["run_date"], freq, self.zone)
                or r["hour"] in skipped_periods(r["target_date"], freq, self.zone)
            )

        return on_the_clock

    def _records(
        self,
        cols: Optional[List[str]],
        statuses: Optional[Iterable[str]] = OK_ONLY,
        clock: Optional[Callable[[Dict], bool]] = None,
        first: bool = False,
    ) -> Iterator[Tuple[Dict, Dict]]:
        """(the record as read, the whole record; the record as read again when `cols` are given) of each record of this ref's
        dates and window, and on the clock if `clock` (`_on_the_clock`) is given: a file's forecasts (`read_records`: one record a
        key, the highest revision, the first forecast; status "ok", or none) unless told other `statuses` (None for every key's),
        and the forecasts of results held in memory as they are. `first` reads a file's lines in order without the per-key rule, for
        a caller that stops at the first (`first_record`)."""
        filter_dates = bool(self.test_start and self.test_end)
        window_start = date.fromisoformat(self.test_start) if self.window is not None and self.test_start else None
        horizons = None if self.horizons is None else frozenset(self.horizons)
        weekdays = None if self.issue_days is None else frozenset(self.issue_days)
        # A validation window issued on every k-th day (`RunWindows.validation_every`): a record of another day is none of its.
        thinned = (
            self.windows
            if self.window == VALIDATION and self.windows is not None and self.windows.validation_every != 1
            else None
        )

        def keep(r: Dict) -> bool:
            if filter_dates and not self.test_start <= r["run_date"] <= self.test_end:
                return False
            if thinned is not None and not thinned.issues_on(r["run_date"]):
                return False
            if horizons is not None and r["horizon"] not in horizons:
                return False
            if weekdays is not None and _run_weekday(r) not in weekdays:
                return False
            if self.window is not None and r.get("window", TEST) != self.window:
                return False
            return self.target_before is None or r["target_date"] < self.target_before

        committed = self._committed()
        if self.path is not None:
            path = Path(self.path)
            if path.exists():

                def read(r: Dict) -> Optional[Tuple[Dict, Dict]]:
                    if not keep(r) or (clock is not None and not clock(r)):
                        return None
                    as_read = self._as_read(r, cols, window_start)
                    return as_read, (r if cols is None else as_read)  # a file's whole records aren't held

                if not first:
                    yield from read_records(
                        path,
                        statuses,
                        read,
                        columns=None if cols is None else [*cols, *_FILTER_FIELDS],
                        committed=committed,
                    )
                    return
                wanted = None if statuses is None else frozenset(statuses)
                visible = visible_revisions(committed)
                for r in iter_file_records(path):
                    if visible is not None and (r.get("run_date"), r.get("revision")) not in visible:
                        continue  # a revision this ref doesn't see
                    if wanted is None or r.get("status", STATUS_OK) in wanted:
                        out = read(r)
                        if out is not None:
                            yield out
        else:
            wanted = None if statuses is None else frozenset(statuses)
            for r in self._results or []:
                if (wanted is None or r.get("status", STATUS_OK) in wanted) and keep(r):
                    if clock is None or clock(r):
                        yield self._as_read(r, cols, window_start), r

    @staticmethod
    def _as_read(r: Dict, cols: Optional[List[str]], window_start: Optional[date]) -> Dict:
        if window_start is not None and "day_in_test" in r:
            day_in_window = (date.fromisoformat(r["run_date"]) - window_start).days
            if r["day_in_test"] != day_in_window:
                r = {**r, "day_in_test": day_in_window}
        return {k: r[k] for k in cols if k in r} if cols else r


def issued_mae(records: Iterable[Dict], period: Optional[Sequence[str]]) -> Optional[float]:
    """The MAE of the records issued from `period[0]` to `period[1]` with a finite prediction and actual, as an exact sum (the
    same whatever order the records are in); None without a period or such a record. What `choices()` shows as `test_mae`."""
    if not period:
        return None
    start, end = period
    errors = []
    for r in records:
        if not start <= r["run_date"] <= end:
            continue
        prediction, actual = r.get("prediction"), r.get("actual")
        if prediction is None or actual is None or not (math.isfinite(prediction) and math.isfinite(actual)):
            continue
        errors.append(abs(prediction - actual))
    return math.fsum(errors) / len(errors) if errors else None


def _run_weekday(r: Dict) -> int:
    """A record's `run_weekday`, which every record the models write carries; from its `run_date` for one without."""
    weekday = r.get("run_weekday")
    return date.fromisoformat(r["run_date"]).weekday() if weekday is None else int(weekday)


def split_by_window(
    ref: ModelResultRef,
    windows: RunWindows,
    calendar: Any,
    skipped: Optional[Mapping[str, Mapping[str, int]]] = None,
    predictor_lists: Optional[Mapping[str, List[Dict]]] = None,
    listings: Optional[Mapping[str, Optional[List[Dict]]]] = None,
    target_before: Optional[str] = None,
) -> Dict[str, ModelResultRef]:
    """One ref per window of `ref`, the ref of a whole run. `produced` is counted from the records and `count` from the
    window's days by `calendar` (an `IssueCalendar`), and what is left out of a validation report for its target day is counted
    in `skipped`."""
    produced: Counter = Counter()
    excluded = 0
    for r in ref.iter_records(cols=["run_date", "target_date", "window"]):
        window = windows.window_of(r["run_date"])
        if window is None or r.get("window", TEST) != window:
            continue
        if window == VALIDATION and target_before is not None and r["target_date"] >= target_before:
            excluded += 1
        else:
            produced[window] += 1

    refs = {}
    for window, (start, end) in windows.periods:
        counts = dict((skipped or {}).get(window, {}))
        is_validation = window == VALIDATION
        if is_validation and excluded:
            counts[SKIP_TARGET_IN_TEST] = excluded
        refs[window] = replace(
            ref,
            count=calendar.count(start, end, windows.validation_every if is_validation else 1),
            test_start=start,
            test_end=end,
            produced=produced[window],
            skipped=counts,
            predictor_lists=(predictor_lists or {}).get(window),
            availability_listing=(listings or {}).get(window),
            window=window,
            target_before=target_before if is_validation else None,
            windows=windows,
            by_window=None,
        )
    return refs


def warn_off_the_clock(refs: Mapping[str, ModelResultRef], stacklevel: int = 2) -> None:
    """One warning naming the refs with records of a slot the clock doesn't have, and how many: they are no forecasts."""
    found = {name: n for name, ref in refs.items() if (n := ref.off_the_clock())}
    if found:
        warnings.warn(
            f"Records of a slot the clock doesn't have, the hour a spring switch skips, on their issue day or target day "
            f"(per model: {found}): their target can't exist, so they are left out of every table and count. Only results "
            "written before training rows were chosen by date hold them.",
            UserWarning,
            stacklevel=stacklevel + 1,
        )
