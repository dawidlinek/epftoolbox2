from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

from .windows import SKIP_TARGET_IN_TEST, TEST, VALIDATION, RunWindows


@dataclass
class ModelResultRef:
    """Where one model's results live and how many forecasts the run planned and produced.

    `count` is the number of planned forecasts (`IssueCalendar`: periods x horizons x test days, without the slots the clock
    doesn't have on an issue day or a target day). `produced` is how many exist after the run (including ones resumed from an earlier run), and `skipped` counts the tasks that
    gave no forecast by reason (e.g. `missing_test_predictors`). `produced` is None for refs built from
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
    don't say, such as plain lists, a results file with no manifest or one older than `Manifest.training_cut`.
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

    def iter_records(self, cols: Optional[List[str]] = None) -> Iterator[Dict]:
        """The records of this ref, wherever they live.

        A resumed JSONL can hold rows from an earlier run with a wider or different test window (its rows are never
        deleted), and a run with validation holds both windows, so each ref keeps to its own. Refs built from plain result
        lists have no window (test_start/test_end are "") and are read as given.

        A window's ref gives `day_in_test` from the window's start as it is now: a record resumed after the start moved
        holds the old count in the file.
        """
        filter_dates = bool(self.test_start and self.test_end)
        window_start = date.fromisoformat(self.test_start) if self.window is not None and self.test_start else None

        def keep(r: Dict) -> bool:
            if filter_dates and not self.test_start <= r["run_date"] <= self.test_end:
                return False
            if self.window is not None and r.get("window", TEST) != self.window:
                return False
            return self.target_before is None or r["target_date"] < self.target_before

        if self.path is not None:
            path = Path(self.path)
            if path.exists():
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            r = json.loads(line)
                            if keep(r):
                                yield self._as_read(r, cols, window_start)
        else:
            for r in self._results or []:
                if keep(r):
                    yield self._as_read(r, cols, window_start)

    @staticmethod
    def _as_read(r: Dict, cols: Optional[List[str]], window_start: Optional[date]) -> Dict:
        if window_start is not None and "day_in_test" in r:
            day_in_window = (date.fromisoformat(r["run_date"]) - window_start).days
            if r["day_in_test"] != day_in_window:
                r = {**r, "day_in_test": day_in_window}
        return {k: r[k] for k in cols if k in r} if cols else r


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
            count=calendar.count(start, end),
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
