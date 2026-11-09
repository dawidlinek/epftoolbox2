"""The checks a column read from another pipeline's stored forecasts (`ForecastSource`) takes down the chain (step 7a).

`BaseModel._check_inputs` calls `check` once per model, after its own checks. It reads the information set's chained columns
(`InformationSet.chained_inputs`, which `with_source_rules` registers) and the predictors of the model that are built from one,
through their shifts, profiles and day aggregates (`InformationSet.chained_of`):

- **Coverage.** The rows the fits read (`TrainingRows`: each planned issue day's span, or with `refit` its anchor's, per horizon
  and window member, from the frame's first date) where such a predictor has no value drop out of training silently. A slot the
  clock doesn't have where a day shift lands (`x_d+1` at 02:00 the day before a spring switch) is no hole of the upstream and
  isn't counted. Any count is a warning at every level, `none` included, and an `InformationSetError` at `raise`; the counts go
  to the manifest's `chained_inputs[column]["coverage"]`, recorded, not compared.
- **Late upstream.** An upstream whose leaf models listed inputs as late or without a rule, or ran at level `none`, makes every
  use of its column a `late upstream` row of the listing at `warn` and a refusal at `raise`: a chain whose first stage read
  something after its own issue time is never an honest chain by accident.

The manifest records each chained column the model reads (`ChainedInput.record()` and its coverage), whose identity a resume
compares (`results.manifest.chained_differences`).
"""

from __future__ import annotations

import datetime
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .._date_utils import shift_calendar_days
from ..information_set import (
    ChainedInput,
    InformationSet,
    InformationSetError,
    TrainingRows,
    _built_from_name,
    parse_predictor,
)
from ..results.availability import late_upstream_rows
from ._calendar import skipped_periods
from ._contract import wall_clock

_STEP = {"1h": 60, "15min": 15}


@dataclass
class ChainedCheck:
    """What `check` found: the manifest's `chained_inputs` (None without a chained column) and the `late upstream` rows."""

    record: Optional[Dict[str, Dict[str, Any]]] = None
    listing: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class _Use:
    chained: ChainedInput
    cells: Set[Tuple[int, int]] = field(default_factory=set)  # (period, horizon)

    @property
    def horizons(self) -> List[int]:
        return sorted({hz for _, hz in self.cells})


def check(
    model_name: str,
    data: pd.DataFrame,
    lists: Dict[Tuple[int, int], Sequence[str]],
    information_set: Optional[InformationSet],
    availability: str,
    issue_days: Optional[Sequence[datetime.date]],
    rows: Optional[TrainingRows],
) -> ChainedCheck:
    """Coverage and late-upstream checks of the chained predictors in `lists` ((period, horizon) -> names a model reads) on the
    frame `data`, for a run issued on `issue_days` (None: no coverage can be told) whose fits read `rows`."""
    if information_set is None or not information_set.chained_inputs():
        return ChainedCheck()
    uses: Dict[str, _Use] = {}
    resolved: Dict[str, Optional[ChainedInput]] = {}  # each name once, however many lists hold it
    for (period, hz), names in lists.items():
        for name in names:
            if name not in resolved:
                resolved[name] = information_set.chained_of(name)
            chained = resolved[name]
            if chained is not None:
                uses.setdefault(name, _Use(chained)).cells.add((int(period), int(hz)))
    if not uses:
        return ChainedCheck()

    record: Dict[str, Dict[str, Any]] = {}
    for use in uses.values():
        record.setdefault(use.chained.column, {**use.chained.record(), "coverage": {}})
    problems: List[str] = []
    if issue_days and rows is not None and len(data.index):
        dates = _row_dates(data.index)
        for name, use in sorted(uses.items()):
            found = coverage(data, name, use, dates, issue_days, rows)
            record[use.chained.column]["coverage"][name] = found
            if found["uncovered"]:
                problems.append(_coverage_message(model_name, name, use.chained, found))
    for message in problems:
        if availability != "raise":
            warnings.warn(message, UserWarning, stacklevel=5)

    late = {name: use for name, use in uses.items() if use.chained.late_upstream}
    listing: List[Dict[str, Any]] = []
    if late and availability == "raise":
        problems += [_late_message(model_name, name, use.chained) for name, use in sorted(late.items())]
    elif late and availability == "warn":
        listing = late_upstream_rows(
            (name, period, hz, _rule_of(information_set, name), _late_note(use.chained))
            for name, use in late.items()
            for period, hz in sorted(use.cells)
        )
    if problems and availability == "raise":
        raise InformationSetError("\n".join(problems))
    return ChainedCheck(record, listing)


def coverage(
    data: pd.DataFrame,
    name: str,
    use: _Use,
    dates: np.ndarray,
    issue_days: Sequence[datetime.date],
    rows: TrainingRows,
) -> Dict[str, Any]:
    """How many rows the fits read (on the dates `read_dates` gives, of the frame's rows) hold no value of `name`, less those
    that can't hold one whatever the upstream does (`_off_the_clock`): {"read", "uncovered", "first", "last", "run_from"} (dates
    ISO, None with nothing uncovered). `run_from` is the first upstream issue day whose forecast the first uncovered row reads."""
    read = read_dates(rows, issue_days, use.horizons)
    on = np.isin(dates, read)
    values = data[name].to_numpy(dtype=np.float64) if name in data.columns else np.full(len(data.index), np.nan)
    hole = on & np.isnan(values)
    hole[hole] = ~_off_the_clock(data.index[hole], name, use.chained)
    uncovered = np.flatnonzero(hole)
    out: Dict[str, Any] = {"read": int(on.sum()), "uncovered": int(len(uncovered)), "first": None, "last": None}
    out["run_from"] = None
    if len(uncovered):
        first, last = int(dates[uncovered].min()), int(dates[uncovered].max())
        out["first"] = datetime.date.fromordinal(first).isoformat()
        out["last"] = datetime.date.fromordinal(last).isoformat()
        out["run_from"] = datetime.date.fromordinal(
            first + day_shift(name, use.chained.column) - use.chained.horizon
        ).isoformat()
    return out


def read_dates(rows: TrainingRows, issue_days: Sequence[datetime.date], horizons: Iterable[int]) -> np.ndarray:
    """The local dates (ordinals) of the rows the fits of these issue days read at these horizons: each fit's
    `training_span`, from its anchor with `refit`, of each window member, never before the frame's first date."""
    days = pd.DatetimeIndex(pd.to_datetime([day.isoformat() for day in issue_days]))
    ordinals = np.array([day.toordinal() for day in issue_days], dtype=np.int64)
    floor = None if rows.first_day is None else datetime.date.fromisoformat(rows.first_day).toordinal()
    intervals: List[Tuple[int, int]] = []
    for window, rule in rows.members or ((rows.window, rows.refit),):
        anchors = np.unique(ordinals - rows.lags(days, rule))
        for hz in horizons:
            span = rows.predictor_days(int(hz), 0, window)  # relative to the fit's own day, its anchor
            intervals.extend((int(a) + span.start, int(a) + span.stop) for a in anchors)
    if not intervals:
        return np.empty(0, dtype=np.int64)
    merged: List[List[int]] = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    out = np.concatenate([np.arange(start, stop, dtype=np.int64) for start, stop in merged])
    return out if floor is None else out[out >= floor]


def day_shift(name: str, column: str) -> int:
    """The days from a row's date to the date of the chained `column` value `name` holds on it: 1 for `x_d+1`, -1 for
    `x_h05_d-1` and `x_day_max_d-1`, the sum of every shift on the way from the name to the column, hours as whole days down."""
    minutes = 0.0
    current = name
    while True:
        chain = parse_predictor(current)
        for candidate, shifts in chain:
            if candidate == column:
                return math.floor((minutes + sum(shift.minutes for shift in shifts)) / 1440.0)
        base, shifts = chain[-1]
        minutes += sum(shift.minutes for shift in shifts)
        built = _built_from_name(base)
        if built is None:
            return math.floor(minutes / 1440.0)
        current = built


_EPOCH_ORDINAL = datetime.date(1970, 1, 1).toordinal()


def _row_dates(index: pd.DatetimeIndex) -> np.ndarray:
    """Each row's local date as an ordinal."""
    days = wall_clock(index).to_numpy().astype("datetime64[D]").astype(np.int64)
    return days + _EPOCH_ORDINAL


def _off_the_clock(index: pd.DatetimeIndex, name: str, chained: ChainedInput) -> np.ndarray:
    """Which of these rows can't hold a value of `name` whatever the upstream holds, for a plain day shift of the chained column
    (`x`, `x_d+1`, `x_d-1`): the row reads a wall time its frame's clock skips (`x_d+1` at 02:00 the day before a spring
    switch), or a slot the upstream never plans, one its clock skips on the issue day it would be issued on (the forecast for
    02:00 of the day after a switch at horizon 1). Every other name is judged as it stands: no row is let off."""
    out = np.zeros(len(index), dtype=bool)
    if not len(index) or getattr(index, "tz", None) is None:
        return out
    shifts = next((shifts for candidate, shifts in parse_predictor(name) if candidate == chained.column), None)
    if shifts is None or any(shift.unit != "d" for shift in shifts):
        return out
    days = sum(shift.offset for shift in shifts)
    read = shift_calendar_days(index, days) if days else index
    out |= np.asarray(read.isna())
    if chained.freq not in _STEP:  # a daily upstream's stamp is on every date's clock
        return out
    upstream = pd.DatetimeIndex(read[~read.isna()]).tz_convert(chained.data_timezone).tz_localize(None)
    periods = np.asarray((upstream.hour * 60 + upstream.minute) // _STEP[chained.freq], dtype=np.int64)
    issued = upstream.normalize() - pd.Timedelta(days=chained.horizon)
    unplanned = np.zeros(len(upstream), dtype=bool)
    for day in issued.unique():
        gaps = skipped_periods(day.date(), chained.freq, chained.data_timezone)
        if gaps:
            at = np.asarray(issued == day)
            unplanned[at] = np.isin(periods[at], list(gaps))
    out[~np.asarray(read.isna())] = unplanned
    return out


def _rule_of(information_set: InformationSet, name: str) -> str:
    rule = information_set.resolve(name).rule
    return "" if rule is None else rule.describe()


def _coverage_message(model_name: str, name: str, chained: ChainedInput, found: Dict[str, Any]) -> str:
    return (
        f"{model_name}: {name} has no value on {found['uncovered']} of the {found['read']} rows its fits read, from "
        f"{found['first']} to {found['last']}: those rows drop out of training. {chained.column} holds the forecasts of "
        f"{chained.model} (horizon {chained.horizon}) in {chained.path}; run {chained.model} from {found['run_from']} so that "
        "they cover the rows the fits read."
    )


def _late_note(chained: ChainedInput) -> str:
    parts = []
    if chained.late:
        shown = ", ".join(chained.late[:5]) + (f" and {len(chained.late) - 5} more" if len(chained.late) > 5 else "")
        parts.append(f"{chained.model} read inputs after its issue time or without a rule: {shown}")
    if chained.unchecked:
        parts.append(f"{', '.join(chained.unchecked)} ran at availability='none', so nothing checked their inputs")
    return "; ".join(parts)


def _late_message(model_name: str, name: str, chained: ChainedInput) -> str:
    return (
        f"{model_name}: {name} is a forecast of {chained.model} in {chained.path}, whose own inputs are not known to be out at "
        f"its issue time ({_late_note(chained)}). Run the upstream at availability='raise' without them, or list it with "
        "availability='warn'."
    )
