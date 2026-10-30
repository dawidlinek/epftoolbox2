"""The validation and test windows of a run.

`ModelPipeline.run(df, validation=..., test=...)` forecasts the days of both windows in one run, into one results file and
one manifest. A forecast doesn't depend on the role it plays: the only things the window of a day decides are the tag on its
record, what its `day_in_test` counts from, and which day a list's binary-column mask is read from (each window's own first
day, so that adding a validation period can't move a test forecast). Days between the two windows are not forecast.

`validation="previous_year"` cuts the test period into consecutive 12-month segments, each served by the 12 months before
it. The run gains one year at the start, tagged "validation". The validation days of a later segment are the test days of
the one before, and they stay tagged "test". A segment says which forecasts may serve its choices (`Segment.serves`).

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


def mask_key(window: str, period: int, horizon: int) -> str:
    """The key of a binary-column mask's digest in a manifest: a test mask has the key a plain run gives it."""
    key = f"{period}-{horizon}"
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
    """

    validation: Optional[Period] = None
    test: Optional[Period] = None
    mode: str = FIXED
    segments: Tuple[Segment, ...] = ()

    @classmethod
    def resolve(cls, validation: Any = None, test: Any = None) -> "RunWindows":
        """The windows `run(validation=..., test=...)` asks for; a `ValueError` if they make no sense."""
        test = None if test is None else to_period(test, TEST)
        if isinstance(validation, str):
            if validation != PREVIOUS_YEAR:
                raise ValueError(f"validation must be (start, end), {PREVIOUS_YEAR!r} or None, got {validation!r}")
            if test is None:
                raise ValueError(
                    f"validation={PREVIOUS_YEAR!r} is the year before a test period: pass test=(start, end)"
                )
            return cls.of_previous_year(test)
        validation = None if validation is None else to_period(validation, VALIDATION)
        if validation is None and test is None:
            raise ValueError("No window to forecast: pass test=(start, end), validation=(start, end) or both")
        return cls.fixed(validation, test)

    @classmethod
    def fixed(cls, validation: Optional[Period], test: Optional[Period]) -> "RunWindows":
        if validation is not None and test is not None and validation[1] >= test[0]:
            raise ValueError(
                f"The validation period {validation} must end before the test period {test} starts: a test forecast must "
                "not be chosen on its own days"
            )
        return cls(validation, test, FIXED, () if validation is None or test is None else (Segment(validation, test),))

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

    def issue_days(self) -> List[str]:
        """The days a run forecasts, in order: both windows, once each, and nothing between them."""
        days: List[str] = []
        for _, (start, end) in self.periods:
            first, last = datetime.date.fromisoformat(start), datetime.date.fromisoformat(end)
            days += [(first + datetime.timedelta(days=d)).isoformat() for d in range((last - first).days + 1)]
        return days

    def window_of(self, run_date: str) -> Optional[str]:
        """The window of an issue day, or None for a day in neither."""
        for name, (start, end) in self.periods:
            if start <= run_date <= end:
                return name
        return None

    def keeping(self, older: "RunWindows") -> "RunWindows":
        """What a manifest records after this run into `older`'s results: records outside a run's windows stay in the file,
        so a window this run doesn't name stays recorded. Nothing that would make the windows overlap is kept."""
        validation, test = self.validation or older.validation, self.test or older.test
        if validation is not None and test is not None and validation[1] >= test[0]:
            validation, test = self.validation, self.test
        keeps_mode = self.validation is None or (self.test is None and self.validation == older.validation)
        if (keeps_mode and older.mode or self.mode) == PREVIOUS_YEAR and test is not None:
            derived = RunWindows.of_previous_year(test)
            if derived.validation == validation:
                return derived
        return RunWindows.fixed(validation, test)

    def to_dict(self) -> Dict[str, Any]:
        return {
            VALIDATION: None if self.validation is None else list(self.validation),
            TEST: None if self.test is None else list(self.test),
            "mode": self.mode,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunWindows":
        validation, test = data.get(VALIDATION), data.get(TEST)
        return cls(
            None if validation is None else tuple(validation),
            None if test is None else tuple(test),
            data.get("mode", FIXED),
            tuple(Segment.from_dict(segment) for segment in data.get("segments", [])),
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
    "mask_key",
    "to_period",
    "yearly_segments",
]
