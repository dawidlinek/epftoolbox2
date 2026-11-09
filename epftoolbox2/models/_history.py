"""The history accessor (step 6h): the series a sequence model (a foundation model, 6i) reads for a forecast, cut by the
information set.

A sequence model reads series, not the predictor columns of a row. For the forecast issued on D at horizon h:

- `target` (T,): the target's last T = context x periods-per-day steps in physical time, on the UTC index, ending at its
  newest step known at the issue time by its rule. T is the same on every issue day: a context across a DST switch covers an
  hour more or less of local time, and no day is resampled.
- `past` {name: (T,)}: each past covariate on the same steps. After its own newest known step it is NaN where the backend
  accepts missing values (`HistorySpec.accepts_missing`); otherwise such a series is late.
- `known` {name: (T + S,)}: each known-ahead covariate on the same steps and the S steps ahead, from the target's first
  unknown step to the end of the target day D + h. One that isn't known through that end is late.
- `slots` (S,): the (local date, period) each step ahead forecasts, None for a step before the target day. The repeated autumn
  hour's period is its second step, as `_worker.issue_row` takes it; the hour a spring switch skips has no step.

So S is 24 for a price issued at 11:45 (D's prices are out at 13:00 on D-1), 23 or 25 when the target day is a DST day, and 38
for a load target whose actuals are out up to 10:00.

Which steps are known comes from `information_set.steps_known`, the per-cell reading `perturb_unknown` (and so `verify`)
perturbs by, through the reading the listing and the check use (`_available_array`): a second computation could leave a value
in the context that a probe perturbs. Each series is read from 31 days before D (`NEWEST_TARGET_FROM`, as
`InformationSet.newest_targets` reads a target) to the end of the target day, and its newest known step is the one before its
first unknown step; a known value there says every older one is known, unless its rule changes over time (`over_time`), whose
older steps the context holds are read too. `HistoryBatch.batches(batch_size)` groups the items by S and pads each batch to
`batch_size` with copies of its first item, so a backend call has one shape whatever else is run.

A model reads history if its class defines `_history_spec(target) -> HistorySpec` (`reads_history`). `verify` never cuts the
frame of such a model after the issue day (`information_set.cuts_at_issue_day`): its known-ahead values are on later rows.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..information_set import (
    _STEP_MINUTES,
    NEWEST_TARGET_FROM,
    InformationSet,
    InformationSetError,
    StepsKnown,
    over_time,
    parse_predictor,
    steps_known,
    vintage_of,
)
from ..results.availability import ROLE_HISTORY_TARGET, ROLE_KNOWN_AHEAD, ROLE_PAST
from ._calendar import FREQ_STEPS, periods_per_day

# (local date, period) of a step ahead, or None for a step before the target day (or the first of a repeated hour).
Slot = Optional[Tuple[datetime.date, int]]
_EPOCH = pd.Timestamp("1970-01-01")
_EPOCH_ORDINAL = datetime.date(1970, 1, 1).toordinal()
# Frequencies a context is counted in: steps of physical time. A daily row is no step of it.
_SUB_DAILY = ("1h", "15min")


@dataclass(frozen=True)
class HistorySpec:
    """What a sequence model reads: `context` local days of the target (T = context x periods-per-day steps), the past and
    known-ahead covariates as full base column names (`"load_forecast"`, never a shifted `"load_forecast_d+1"`), and the
    target. `accepts_missing` says the backend takes NaN: a past series then ends at its newest known step with NaN after it,
    instead of being late."""

    context: int
    known_ahead: Tuple[str, ...] = ()
    past: Tuple[str, ...] = ()
    target: str = "price"
    accepts_missing: bool = False

    def __post_init__(self):
        if isinstance(self.context, bool) or not isinstance(self.context, (int, np.integer)) or self.context < 1:
            raise ValueError(f"context is a whole number of local days from 1, got {self.context!r}")
        object.__setattr__(self, "context", int(self.context))
        for name in ("known_ahead", "past"):
            value = getattr(self, name)
            if isinstance(value, str):
                raise ValueError(f"{name} is a list of column names, got the string {value!r}: write [{value!r}]")
            object.__setattr__(self, name, tuple(value))
        for name in self.names:
            if not isinstance(name, str) or not name:
                raise ValueError(f"a series is a column name, got {name!r}")
            if len(parse_predictor(name)) > 1:
                raise ValueError(
                    f"{name!r} reads as a shifted column ({parse_predictor(name)[-1][0]!r} with a suffix): a sequence model "
                    "reads each series as it is on the index and cuts it itself, so give the base column"
                )
        repeated = sorted({name for name in self.names if self.names.count(name) > 1})
        if repeated:
            raise ValueError(f"{repeated} named twice among the target, past and known_ahead: each series has one role")

    @property
    def names(self) -> Tuple[str, ...]:
        """The target, then the past and the known-ahead series, in order."""
        return (self.target, *self.past, *self.known_ahead)

    def role_of(self, name: str) -> str:
        if name == self.target:
            return ROLE_HISTORY_TARGET
        return ROLE_PAST if name in self.past else ROLE_KNOWN_AHEAD


def reads_history(model: Any) -> bool:
    """Whether a model reads series through the history accessor: its class defines `_history_spec(target)`. No core model
    does, so nothing of theirs depends on this."""
    return callable(getattr(type(model), "_history_spec", None))


def history_spec_of(model: Any, target: str) -> Optional[HistorySpec]:
    """The model's `HistorySpec` for `target`, None for a model that reads no history."""
    if not reads_history(model):
        return None
    spec = model._history_spec(target)
    if not isinstance(spec, HistorySpec):
        raise TypeError(f"{type(model).__name__}._history_spec returned {type(spec).__name__}, not a HistorySpec")
    return spec


class Lateness(NamedTuple):
    """When the values a late series lacks are known, on the issue day: minutes since its midnight, and as labels."""

    known: float
    issue: float
    known_label: str
    issue_label: str


@dataclass(frozen=True)
class HistoryItem:
    """What a sequence model reads for the forecast issued on `issue_day` at `horizon` (see the module's docstring)."""

    issue_day: datetime.date
    horizon: int
    target: np.ndarray  # (T,)
    past: Dict[str, np.ndarray]  # each (T,)
    known: Dict[str, np.ndarray]  # each (T + S,)
    slots: Tuple[Slot, ...]  # (S,)
    first_step: pd.Timestamp  # the instant of the context's first step, in the frame's zone
    # Each series' newest step known at the issue time with every one before it: its instant, and its wall-clock minutes after
    # midnight of the issue day (D+1 01:00 is 1500), which a label shows.
    newest: Dict[str, pd.Timestamp]
    newest_minutes: Dict[str, float]
    late: Dict[str, Lateness]  # the series late on this issue day
    # Series holding a missing value of the data's (not a NaN the cut put after a past series' newest step).
    missing: Tuple[str, ...] = ()

    @property
    def steps_ahead(self) -> int:
        """S: the steps from the target's first unknown one to the end of the target day."""
        return len(self.slots)

    @property
    def periods(self) -> np.ndarray:
        """(S,) the period each step ahead forecasts, -1 where it forecasts none (`slots` None)."""
        return np.array([-1 if slot is None else slot[1] for slot in self.slots], dtype=np.int64)


class Batch(NamedTuple):
    """One backend call: items of one S (`items`, positions in `HistoryBatch.items`, the call's first rows), padded to the
    batch size with copies of the first."""

    steps_ahead: int
    items: Tuple[int, ...]
    target: np.ndarray  # (batch_size, T)
    past: Dict[str, np.ndarray]  # each (batch_size, T)
    known: Dict[str, np.ndarray]  # each (batch_size, T + S)


class SeriesInfo(NamedTuple):
    """A series of a `HistorySpec` as the information set resolves it: its role, its rule as described, its vintage class."""

    name: str
    role: str
    rule: str
    vintage: str


@dataclass(frozen=True)
class HistoryBatch:
    """The `HistoryItem`s of some issue days at one horizon, in date order, and how to call a backend on them."""

    spec: HistorySpec
    freq: str
    horizon: int
    steps: int  # T
    periods: int  # periods of an ordinary day, which every forecast's series serve
    series: Tuple[SeriesInfo, ...]
    items: Tuple[HistoryItem, ...] = field(default=())

    def late(self) -> Dict[str, List[datetime.date]]:
        """The issue days each series is late on, of those late on any."""
        out: Dict[str, List[datetime.date]] = {}
        for item in self.items:
            for name in item.late:
                out.setdefault(name, []).append(item.issue_day)
        return out

    def groups(self, which: Optional[Iterable[int]] = None) -> Dict[int, List[int]]:
        """The items (positions; `which`, every one if None) by S, smallest S first, each in date order."""
        out: Dict[int, List[int]] = {}
        for i in range(len(self.items)) if which is None else sorted(which):
            out.setdefault(self.items[i].steps_ahead, []).append(i)
        return dict(sorted(out.items()))

    def batches(self, batch_size: int, which: Optional[Iterable[int]] = None) -> List[Batch]:
        """The backend calls of these items (`which`, every one if None): each S's items in date order, `batch_size` at a time,
        each call padded to `batch_size` rows with copies of its first item. A call's shape depends on S and `batch_size`
        alone, and its real rows are the same whichever other days a run has."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError(f"batch_size is a whole number from 1, got {batch_size!r}")
        out = []
        for steps_ahead, members in self.groups(which).items():
            for start in range(0, len(members), batch_size):
                chunk = tuple(members[start : start + batch_size])
                rows = [self.items[i] for i in chunk]
                rows += [rows[0]] * (batch_size - len(rows))
                out.append(
                    Batch(
                        steps_ahead,
                        chunk,
                        np.stack([row.target for row in rows]),
                        {name: np.stack([row.past[name] for row in rows]) for name in self.spec.past},
                        {name: np.stack([row.known[name] for row in rows]) for name in self.spec.known_ahead},
                    )
                )
        return out


class _Grid(NamedTuple):
    """The regular steps of a frame in physical time (`steps`, in the index's zone), each step's local date as an ordinal and
    period, and the values of the series on them (NaN where the frame has no row)."""

    steps: pd.DatetimeIndex
    ordinal: np.ndarray
    period: np.ndarray
    minute: np.ndarray  # wall-clock minute of the day
    values: Dict[str, np.ndarray]


def _midnight(day: datetime.date, tz: Any) -> pd.Timestamp:
    """Local midnight of `day` in `tz` (naive if None), as `information_set._Context.origin` reads it."""
    stamp = pd.Timestamp(day)
    return stamp if tz is None else stamp.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")


def _grid(frame: pd.DataFrame, names: Sequence[str], freq: str, start: pd.Timestamp, end: pd.Timestamp) -> _Grid:
    """Every step of `freq` in physical time from `start` to `end` (or the frame's first and last rows, where those are
    wider), in the phase of the frame's rows. A row off that grid, a repeated timestamp or an unsorted index raises."""
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or not len(index):
        raise ValueError("the history accessor reads a frame with a DatetimeIndex and rows")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("the history accessor reads a frame whose index is sorted and has no repeated timestamp")
    step = FREQ_STEPS[freq]
    aware = index.tz is not None
    instants = index.tz_convert("UTC") if aware else index
    if aware:
        start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    first, last = instants[0], instants[-1]
    before = max(0, math.ceil((first - start) / step))
    after = max(0, math.ceil((end - last) / step))
    regular = pd.date_range(first - before * step, last + after * step, freq=step)
    if not instants.isin(regular).all():
        off = instants[~instants.isin(regular)][0]
        raise ValueError(
            f"a row at {off} is off the {freq} steps of the frame's first row {first}: the history accessor counts a context "
            "in steps of physical time, which needs every row on them"
        )
    steps = regular.tz_convert(index.tz) if aware else regular
    wall = steps.tz_localize(None) if aware else steps
    ordinal = np.asarray((wall.normalize() - _EPOCH) // pd.Timedelta(days=1), dtype=np.int64) + _EPOCH_ORDINAL
    minute = np.asarray(wall.hour * 60 + wall.minute, dtype=np.int64)
    period = minute // _STEP_MINUTES[freq]
    table = frame[list(names)].reindex(steps)
    values = {name: table[name].to_numpy(dtype=np.float64, copy=True) for name in names}
    return _Grid(steps, ordinal, period, minute, values)


def history_batch(
    spec: HistorySpec,
    frame: pd.DataFrame,
    issue_days: Iterable[Any],
    information_set: InformationSet,
    freq: str = "1h",
    horizon: int = 1,
) -> HistoryBatch:
    """The `HistoryItem` of each issue day (local dates of the frame's zone; each once, in date order) at `horizon`, read from
    `frame` (rows in its own zone, which define local dates and periods) and cut by `information_set`'s rules at its issue time
    on each day (see the module's docstring).

    Raises InformationSetError for a series without a rule (whatever `unknown` says: the cut decides numbers), and for a target
    not known from 31 days before an issue day on, or with an older value its context holds unknown under a rule over time.
    A known-ahead or past series not known as far as the item needs is late (`HistoryItem.late`): the arrays still hold its
    values, which a run lists (availability "warn") or refuses ("raise"), as it does a late predictor."""
    if freq not in _SUB_DAILY:
        raise ValueError(
            f"a context is counted in steps of physical time: freq is one of {list(_SUB_DAILY)}, got {freq!r}"
        )
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError(f"horizon is a whole number of days from 1, got {horizon!r}")
    horizon = int(horizon)
    absent = [name for name in spec.names if name not in frame.columns]
    if absent:
        raise ValueError(f"the frame has no column {absent}, which the history of {spec.target!r} reads")
    for name in spec.names:
        column = frame[name]
        if not pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
            raise ValueError(f"{name!r} isn't numeric, and a series is: convert it to float")
    rules = information_set.in_data_zone(frame.index)
    minutes = _STEP_MINUTES[freq]
    resolved = {name: rules._resolve_for(name, minutes) for name in spec.names}
    for name, col in resolved.items():
        if col.rule is None:
            raise InformationSetError(
                f"'{name}' has no availability rule, and a sequence model's context is cut where its rule says the values "
                f"are known: declare it, e.g. InformationSet(rules={{'{name}': Availability.realised('1h')}})."
            )
        if col.shifts:
            raise ValueError(f"the information set reads {name!r} as a shifted column, which a series can't be")
    periods = periods_per_day(freq)
    steps = spec.context * periods
    series = tuple(
        SeriesInfo(name, spec.role_of(name), col.rule.describe(), vintage_of(col.rule))
        for name, col in resolved.items()
    )
    days = sorted({pd.Timestamp(day).date() for day in issue_days})
    batch = HistoryBatch(spec, freq, horizon, steps, periods, series)
    if not days:
        return batch
    tz = frame.index.tz
    step = FREQ_STEPS[freq]
    start = _midnight(days[0] + datetime.timedelta(days=NEWEST_TARGET_FROM), tz) - steps * step
    end = _midnight(days[-1] + datetime.timedelta(days=horizon + 1), tz)
    grid = _grid(frame, spec.names, freq, start, end)
    over = {name for name, col in resolved.items() if over_time(col.rule)}
    items = tuple(_item(spec, grid, rules, freq, day, horizon, steps, over) for day in days)
    return HistoryBatch(spec, freq, horizon, steps, periods, series, items)


def _lateness(read: StepsKnown, where: np.ndarray) -> Lateness:
    """When the values at `where` (positions of `read`'s steps) are all known, as `listing_rows` shows a late use."""
    known = float(read.available[where].max())
    return Lateness(known, read.issue, read.label(known), read.label(read.issue))


def _item(
    spec: HistorySpec,
    grid: _Grid,
    rules: InformationSet,
    freq: str,
    day: datetime.date,
    horizon: int,
    steps: int,
    over: set,
) -> HistoryItem:
    ordinal = day.toordinal()
    walk = int(np.searchsorted(grid.ordinal, ordinal + NEWEST_TARGET_FROM, side="left"))
    end = int(np.searchsorted(grid.ordinal, ordinal + horizon, side="right")) - 1  # the target day's last step
    reads: Dict[str, StepsKnown] = {}
    newest: Dict[str, int] = {}
    for name in spec.names:
        read = steps_known(rules, name, grid.steps[walk : end + 1], day, freq)
        position = read.newest()
        if position < 0:
            raise InformationSetError(
                f"by its rule '{name}' of {grid.steps[walk]} isn't known at the issue time on {day} (it is "
                f"{read.label(read.available[0])}): a sequence model's history is read from {-NEWEST_TARGET_FROM} days before "
                "the issue day on, and a rule this slow leaves it nothing. Check the rule."
            )
        reads[name], newest[name] = read, walk + position
    cut = newest[spec.target]  # the target's newest known step: the context ends here
    first = cut - steps + 1
    late: Dict[str, Lateness] = {}
    for name in spec.names:
        if name not in over or first >= walk:
            continue
        # Under a rule over time an older value can be out after a newer one: every step the context holds before the walk.
        older = steps_known(rules, name, grid.steps[first:walk], day, freq)
        unknown = np.flatnonzero(~older.known)
        if not len(unknown):
            continue
        if name == spec.target:
            raise InformationSetError(
                f"by its rules over time the target '{name}' of {grid.steps[first + int(unknown[0])]}, in the context of the "
                f"issue day {day}, isn't known at its issue time: shorten the context or start the test where the rule allows"
            )
        late[name] = _lateness(older, unknown)

    target = grid.values[spec.target][first : cut + 1].copy()
    missing = [] if np.isfinite(target).all() else [spec.target]
    past = {}
    for name in spec.past:
        values = grid.values[name][first : cut + 1].copy()
        # Its newest known step in the context's positions: -1 or less where every step of the context is after it.
        own = max(newest[name] - first, -1)
        if own < steps - 1:
            if spec.accepts_missing:
                values[own + 1 :] = np.nan
            elif name not in late:
                late[name] = _lateness(reads[name], np.arange(newest[name] + 1, cut + 1) - walk)
        held = values[: own + 1] if spec.accepts_missing else values
        if not np.isfinite(held).all():
            missing.append(name)
        past[name] = values
    known = {}
    for name in spec.known_ahead:
        values = grid.values[name][first : end + 1].copy()
        if newest[name] < end and name not in late:
            late[name] = _lateness(reads[name], np.arange(newest[name] + 1, end + 1) - walk)
        if not np.isfinite(values).all():
            missing.append(name)
        known[name] = values

    target_day = day + datetime.timedelta(days=horizon)
    ahead = np.arange(cut + 1, end + 1)
    on_target = grid.ordinal[ahead] == target_day.toordinal()
    slots: List[Slot] = []
    for i, position in enumerate(ahead):
        period = int(grid.period[position])
        # A period whose step comes again later on the target day is the repeated autumn hour's first: the second forecasts it.
        again = on_target[i + 1 :] & (grid.period[ahead[i + 1 :]] == period)
        slots.append((target_day, period) if on_target[i] and not again.any() else None)

    def wall(position: int) -> float:
        return float((grid.ordinal[position] - ordinal) * 1440 + grid.minute[position])

    return HistoryItem(
        issue_day=day,
        horizon=horizon,
        target=target,
        past=past,
        known=known,
        slots=tuple(slots),
        first_step=grid.steps[first],
        newest={name: grid.steps[position] for name, position in newest.items()},
        newest_minutes={name: wall(position) for name, position in newest.items()},
        late=late,
        missing=tuple(missing),
    )


__all__ = [
    "Batch",
    "HistoryBatch",
    "HistoryItem",
    "HistorySpec",
    "Lateness",
    "SeriesInfo",
    "history_batch",
    "history_spec_of",
    "reads_history",
]
