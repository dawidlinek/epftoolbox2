"""The validation and test windows of a run.

`ModelPipeline.run(df, validation=..., test=...)` forecasts the days of both windows in one run, into one results file and
one manifest. A forecast doesn't depend on the role it plays: the only things the window of a day decides are the tag on its
record, what its `day_in_test` counts from, and which day a list's binary-column mask is read from (each window's own first
day, so that adding a validation period can't move a test forecast). Days between the two windows are not forecast.

`validation="previous_year"` cuts the test period into consecutive 12-month segments, each served by the 12 months before
it. The run gains one year at the start, tagged "validation". The validation days of a later segment are the test days of
the one before, and they stay tagged "test". A segment says which forecasts may serve its choices (`Segment.serves`).

`validation_every=k` (`run(..., validation_every=k)`) issues on every k-th day of a fixed validation window alone, counted from its
first day, which is always one: the days a search scores its cheap rungs on (`epftoolbox2.tune`). Each forecast on such a day is
the one a run of every day gives it; the other days are neither forecast nor planned, nor read from a results file that holds
them. The test window is never thinned.

A manifest stores all of this (`RunWindows.to_dict`), `read_windows` in `epftoolbox2.results.manifest` reads it back, and
`EvaluationReport.windows` gives what a report was built on.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .._date_utils import resolve_date

VALIDATION = "validation"
TEST = "test"

FIXED = "fixed"
PREVIOUS_YEAR = "previous_year"

# In `forecast_counts()` of a validation report: forecasts issued in validation for a day at or after the first test issue
# day. They are in the results file, and left out of the report because choosing from them would be look-ahead.
SKIP_TARGET_IN_TEST = "target_in_test"

Period = Tuple[str, str]


def _iso_day(value: Any, label: str) -> str:
    value = resolve_date(value)
    if isinstance(value, datetime.datetime):  # a pd.Timestamp too
        value = value.date()
    if isinstance(value, datetime.date):
        return value.isoformat()
    try:
        return datetime.date.fromisoformat(value).isoformat()
    except (TypeError, ValueError):
        raise ValueError(f"The {label} must be a date such as '2024-01-31', got {value!r}") from None


def check_validation_every(value: Any) -> int:
    """`validation_every` as an int of at least 1; a `ValueError` otherwise."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"validation_every must be a whole number of at least 1, got {value!r}")
    return int(value)


def to_period(value: Any, label: str) -> Period:
    """`value` as (first day, last day) in ISO format; a `ValueError` naming `label` if it isn't such a pair."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"The {label} period must be (start, end), got {value!r}")
    start, end = _iso_day(value[0], f"{label} start"), _iso_day(value[1], f"{label} end")
    if start > end:
        raise ValueError(f"The {label} period is reversed: it starts on {start} and ends on {end}")
    return start, end


def add_years(day: datetime.date, years: int) -> datetime.date:
    """The same calendar day `years` years later, or 28 February for a 29 February in a year without one."""
    try:
        return day.replace(year=day.year + years)
    except ValueError:
        return day.replace(year=day.year + years, day=28)


def mask_key(window: str, period: int, horizon: int, member: Any = None) -> str:
    """The key of a binary-column mask's digest in a manifest: a test mask has the key a plain run gives it. A model with a list
    of training windows has a mask per member, under the key with `@` and the member (`@56`, `@expanding`); one with one window
    passes no member and keeps the key it always had."""
    key = f"{period}-{horizon}" if member is None else f"{period}-{horizon}@{member}"
    return key if window == TEST else f"{window}/{key}"


@dataclass(frozen=True)
class Segment:
    """A test period and the validation period that serves it.

    Attributes:
        validation: (first, last) issue day of the validation period.
        test: (first, last) issue day of the test period it serves.
    """

    validation: Period
    test: Period

    def serves(self, run_date: str, target_date: str) -> bool:
        """Whether a forecast may count for this segment's choices: issued in its validation period, for a day before
        its first test issue day, whatever the record's `window` tag says."""
        return self.validation[0] <= run_date <= self.validation[1] and target_date < self.test[0]

    def to_dict(self) -> Dict[str, List[str]]:
        return {VALIDATION: list(self.validation), TEST: list(self.test)}

    @classmethod
    def from_dict(cls, data: Dict[str, Sequence[str]]) -> "Segment":
        return cls(tuple(data[VALIDATION]), tuple(data[TEST]))


def yearly_segments(test: Period) -> Tuple[Segment, ...]:
    """`test` cut into 12-month segments, each boundary counted from the original start, so a 29 February start doesn't
    drift to the 28th for good. The last segment may be partial, and each one has the 12 months before it as validation."""
    first, last = datetime.date.fromisoformat(test[0]), datetime.date.fromisoformat(test[1])
    segments = []
    k = 0
    while (start := add_years(first, k)) <= last:
        next_start = add_years(first, k + 1)
        segments.append(
            Segment(
                (add_years(first, k - 1).isoformat(), (start - datetime.timedelta(days=1)).isoformat()),
                (start.isoformat(), min(next_start - datetime.timedelta(days=1), last).isoformat()),
            )
        )
        k += 1
    return tuple(segments)


@dataclass(frozen=True)
class RunWindows:
    """The windows of a run.

    Attributes:
        validation: (first, last) issue day tagged "validation", or None.
        test: (first, last) issue day tagged "test", or None.
        mode: `"fixed"`, or `"previous_year"` when `validation` was derived from `test`.
        segments: Each test period and what serves it: one for fixed windows (none with a single window), one per 12 months
            with `"previous_year"`.
        validation_every: k: the run issues on every k-th day of the validation window alone, counted from its first day (1,
            every day, for any run that doesn't ask for it). Only with a fixed validation window.
    """

    validation: Optional[Period] = None
    test: Optional[Period] = None
    mode: str = FIXED
    segments: Tuple[Segment, ...] = ()
    validation_every: int = 1

    @classmethod
    def resolve(cls, validation: Any = None, test: Any = None, validation_every: int = 1) -> "RunWindows":
        """The windows `run(validation=..., test=..., validation_every=...)` asks for; a `ValueError` if they make no sense."""
        every = check_validation_every(validation_every)
        test = None if test is None else to_period(test, TEST)
        if isinstance(validation, str):
            if validation != PREVIOUS_YEAR:
                raise ValueError(f"validation must be (start, end), {PREVIOUS_YEAR!r} or None, got {validation!r}")
            if test is None:
                raise ValueError(
                    f"validation={PREVIOUS_YEAR!r} is the year before a test period: pass test=(start, end)"
                )
            if every != 1:
                raise ValueError(
                    f"validation_every={every} thins a fixed validation window, and with validation={PREVIOUS_YEAR!r} the "
                    "later years are served by test days, which are never thinned: pass validation=(start, end)"
                )
            return cls.of_previous_year(test)
        validation = None if validation is None else to_period(validation, VALIDATION)
        if validation is None and test is None:
            raise ValueError("No window to forecast: pass test=(start, end), validation=(start, end) or both")
        return cls.fixed(validation, test, every)

    @classmethod
    def fixed(cls, validation: Optional[Period], test: Optional[Period], validation_every: int = 1) -> "RunWindows":
        if validation is not None and test is not None and validation[1] >= test[0]:
            raise ValueError(
                f"The validation period {validation} must end before the test period {test} starts: a test forecast must "
                "not be chosen on its own days"
            )
        every = check_validation_every(validation_every)
        if every != 1 and validation is None:
            raise ValueError(f"validation_every={every} thins the validation window: pass validation=(start, end)")
        segments = () if validation is None or test is None else (Segment(validation, test),)
        return cls(validation, test, FIXED, segments, every)

    @classmethod
    def of_previous_year(cls, test: Period) -> "RunWindows":
        segments = yearly_segments(test)
        return cls(segments[0].validation, test, PREVIOUS_YEAR, segments)

    @property
    def periods(self) -> List[Tuple[str, Period]]:
        """Each window and its (first, last) issue day, in time order."""
        found = [(VALIDATION, self.validation), (TEST, self.test)]
        return [(name, period) for name, period in found if period is not None]

    @property
    def first_day(self) -> str:
        return self.periods[0][1][0]

    @property
    def last_day(self) -> str:
        return self.periods[-1][1][1]

    @property
    def target_before(self) -> Optional[str]:
        """A validation forecast counts only if its target day is before this one, the first test issue day."""
        return None if self.test is None else self.test[0]

    def issue_days(self, weekdays: Optional[Sequence[int]] = None) -> List[str]:
        """The days a run forecasts, in order: both windows, once each, and nothing between them, and of the validation window
        every `validation_every`-th day alone. With `weekdays` (a run's `issue_days`, as `date.weekday()` numbers) only the
        days of those weekdays."""
        days: List[str] = []
        kept = None if weekdays is None else frozenset(weekdays)
        for name, (start, end) in self.periods:
            first, last = datetime.date.fromisoformat(start), datetime.date.fromisoformat(end)
            step = self.validation_every if name == VALIDATION else 1
            for d in range(0, (last - first).days + 1, step):
                day = first + datetime.timedelta(days=d)
                if kept is None or day.weekday() in kept:
                    days.append(day.isoformat())
        return days

    def issues_on(self, run_date: Any) -> bool:
        """Whether a day of the run's windows is an issue day by `validation_every` (weekdays aside): every day of the test
        window, every k-th of the validation window. False for a day in neither."""
        day = run_date.isoformat() if isinstance(run_date, datetime.date) else str(run_date)
        window = self._period_of(day)
        if window is None:
            return False
        if window != VALIDATION or self.validation_every == 1:
            return True
        offset = (datetime.date.fromisoformat(day) - datetime.date.fromisoformat(self.validation[0])).days
        return offset % self.validation_every == 0

    def _period_of(self, run_date: str) -> Optional[str]:
        for name, (start, end) in self.periods:
            if start <= run_date <= end:
                return name
        return None

    def window_of(self, run_date: str) -> Optional[str]:
        """The window of an issue day, or None for a day in neither, and for a validation day `validation_every` skips: a
        record of it is no forecast of this run."""
        window = self._period_of(run_date)
        if window == VALIDATION and self.validation_every != 1 and not self.issues_on(run_date):
            return None
        return window

    def would_drop(self, older: "RunWindows") -> Optional[Tuple[str, Period]]:
        """The window of `older` this run doesn't name and that can't stay next to what it names, or None."""
        validation, test = self.validation or older.validation, self.test or older.test
        if validation is None or test is None or validation[1] < test[0]:
            return None
        return (VALIDATION, older.validation) if self.validation is None else (TEST, older.test)

    def keeping(self, older: "RunWindows") -> "RunWindows":
        """What a manifest records after this run into `older`'s results: records outside a run's windows stay in the file,
        so a window this run doesn't name stays recorded. What `would_drop` names isn't kept, which a run refuses unless
        it is told to go on."""
        validation, test = self.validation or older.validation, self.test or older.test
        # The thinning of the validation window that is kept: this run's if it names one, else the one on file.
        every = self.validation_every if self.validation is not None else older.validation_every
        if self.would_drop(older) is not None:
            validation, test = self.validation, self.test
            every = self.validation_every
        keeps_mode = self.validation is None or (self.test is None and self.validation == older.validation)
        if (keeps_mode and older.mode or self.mode) == PREVIOUS_YEAR and test is not None:
            derived = RunWindows.of_previous_year(test)
            if derived.validation == validation:
                return derived
        return RunWindows.fixed(validation, test, every if validation is not None else 1)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            VALIDATION: None if self.validation is None else list(self.validation),
            TEST: None if self.test is None else list(self.test),
            "mode": self.mode,
            "segments": [segment.to_dict() for segment in self.segments],
        }
        if self.validation_every != 1:  # written only when set, so manifests of every other run don't move
            out["validation_every"] = self.validation_every
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunWindows":
        validation, test = data.get(VALIDATION), data.get(TEST)
        return cls(
            None if validation is None else tuple(validation),
            None if test is None else tuple(test),
            data.get("mode", FIXED),
            tuple(Segment.from_dict(segment) for segment in data.get("segments", [])),
            int(data.get("validation_every", 1)),
        )


__all__ = [
    "FIXED",
    "PREVIOUS_YEAR",
    "SKIP_TARGET_IN_TEST",
    "TEST",
    "VALIDATION",
    "RunWindows",
    "Segment",
    "add_years",
    "check_validation_every",
    "mask_key",
    "to_period",
    "yearly_segments",
]
