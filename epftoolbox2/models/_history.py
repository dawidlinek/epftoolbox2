"""The history accessor (step 6h): the series a sequence model (a foundation model, 6i) reads for a forecast, cut by the
information set.

A sequence model reads series, not the predictor columns of a row. For the forecast issued on D at horizon h:

- `target` (T,): the target's last T = context x periods-per-day steps in physical time, on the UTC index, ending at its cut.
  With `newest_target="published"` that is its newest step known at the issue time by its rule, at the latest the last step of
  the day before the target day (as `InformationSet.newest_targets` caps a row model's at h - 1: at 14:00 D+1's prices are out,
  and the forecast at h = 1 is still of all of D+1); with `"d-1"`, the default of every model, the earlier of that and the last
  step of local D-1, as such a model's newest training target is D-1's. T is the
  same on every issue day: a context across a DST switch covers an hour more or less of local time, and no day is resampled.
- `past` {name: (T,)}: each past covariate on the same steps. After its own newest known step it is NaN where the backend
  accepts missing values (`HistorySpec.accepts_missing`); otherwise such a series is late.
- `known` {name: (T + S,)}: each known-ahead covariate on the same steps and the S steps ahead, from the target's cut to the
  end of the target day D + h. One that isn't known through that end is late.
- `slots` (S,): the (local date, period) each step ahead forecasts, None for a step before the target day. The repeated autumn
  hour's period is its second step, as `_worker.issue_row` takes it; the hour a spring switch skips has no step.

So S is 24 for a price issued at 11:45 under `"published"` (D's prices are out at 13:00 on D-1), 23 or 25 when the target day is
a DST day, 38 for a load target whose actuals are out up to 10:00; under `"d-1"` it is 48 for either at h = 1.

Which steps are known comes from `information_set.StepsReader`, the per-cell reading `perturb_unknown` (and so `verify`)
perturbs by, through the reading the listing and the check use (`_available_array`): a second computation could leave a value
in the context that a probe perturbs. Each series is read from 31 days before D (`NEWEST_TARGET_FROM`, as
`InformationSet.newest_targets` reads a target) to the end of the target day, and its newest known step is the one before its
first unknown step; a known value there says every older one is known, unless its rule changes over time (`over_time`), whose
older steps the context holds are read too. A late series keeps its values, which a run lists (availability "warn") or refuses
("raise"), as it does a late predictor: only then can an array hold a value the probe perturbs.

The arrays of an item are read-only views of one copy of the frame's series (a past series with a NaN tail is a copy of its
own), so items of many days and horizons cost little memory. `HistoryBatch.batches(batch_size)` groups the items by S and pads
each batch to `batch_size` with copies of its first item, so a backend call has one shape whatever else is run. With
`arrays=False` (`history_batches`) the items hold the cuts alone, for a listing or `explain`.

A model reads history if its class defines `_history_spec(target) -> HistorySpec` (`reads_history`). `verify` runs it on the
frame cut after what each forecast reads (`information_set.target_day_cut`), never after the issue day: its known-ahead values
are on later rows.
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
    StepsReader,
    over_time,
    parse_predictor,
    vintage_of,
)
from ..results.availability import ROLE_HISTORY_TARGET, ROLE_KNOWN_AHEAD, ROLE_PAST
from ._calendar import FREQ_STEPS, check_newest_target, periods_per_day

# (local date, period) of a step ahead, or None for a step before the target day (or the first of a repeated hour).
Slot = Optional[Tuple[datetime.date, int]]
_EPOCH = pd.Timestamp("1970-01-01")
_EPOCH_ORDINAL = datetime.date(1970, 1, 1).toordinal()
# Frequencies a context is counted in: steps of physical time. A daily row is no step of it.
_SUB_DAILY = ("1h", "15min")
_NONE = -1  # a step ahead that forecasts no period (`HistoryItem.slot_periods`)


@dataclass(frozen=True)
class HistorySpec:
    """What a sequence model reads: `context` local days of the target (T = context x periods-per-day steps), the past and
    known-ahead covariates as full base column names (`"load_forecast"`, never a shifted `"load_forecast_d+1"`), and the
    target. `accepts_missing` says the backend takes NaN in a context: a past series then ends at its newest known step with
    NaN after it, instead of being late, and a missing value of the data in any context doesn't keep an item out of a call.
    `newest_target` is the model's: `"d-1"` cuts the target's context at the end of D-1 at the latest, `"published"` where its
    rule says."""

    context: int
    known_ahead: Tuple[str, ...] = ()
    past: Tuple[str, ...] = ()
    target: str = "price"
    accepts_missing: bool = False
    newest_target: str = "d-1"

    def __post_init__(self):
        if isinstance(self.context, bool) or not isinstance(self.context, (int, np.integer)) or self.context < 1:
            raise ValueError(f"context is a whole number of local days from 1, got {self.context!r}")
        object.__setattr__(self, "context", int(self.context))
        check_newest_target(self.newest_target)
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


@dataclass(frozen=True)
class Lateness:
    """When the values a late series lacks are known, on the issue day: minutes since its midnight, and as labels (read when
    asked for: a listing shows one item's)."""

    known: float
    issue: float
    _reader: StepsReader = field(repr=False, compare=False)
    _day: int = field(repr=False, compare=False)  # the issue day's position in the reader
    _instants: Tuple[float, float] = field(repr=False, compare=False)  # known and issue on the reader's scale

    @property
    def known_label(self) -> str:
        return self._reader.label(self._instants[0], self._day)

    @property
    def issue_label(self) -> str:
        return self._reader.label(self._instants[1], self._day)


@dataclass(frozen=True, eq=False)
class HistoryItem:
    """What a sequence model reads for the forecast issued on `issue_day` at `horizon` (see the module's docstring). Arrays are
    None for an item of the cuts alone (`arrays=False`)."""

    issue_day: datetime.date
    horizon: int
    target: Optional[np.ndarray]  # (T,)
    past: Optional[Dict[str, np.ndarray]]  # each (T,)
    known: Optional[Dict[str, np.ndarray]]  # each (T + S,)
    # The period each step ahead forecasts on the target day, -1 where it forecasts none: (S,), which `slots` spells out.
    slot_periods: np.ndarray
    first_step: pd.Timestamp  # the instant of the context's first step, in the frame's zone
    # Each series' newest step it is read to, known at the issue time with every one before it: the target's cut, a past
    # series' newest known step up to the cut, a known-ahead one's up to the end of the target day. Its instant and its
    # wall-clock minutes after midnight of the issue day (D+1 01:00 is 1500), which a label shows.
    newest: Dict[str, pd.Timestamp]
    newest_minutes: Dict[str, float]
    late: Dict[str, Lateness]  # the series late on this issue day
    # Missing values of the data's, per series: (in the context, in the steps ahead); a NaN the cut put after a past series'
    # newest step isn't one. Empty for an item of the cuts alone.
    gaps: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    # The series the backend can't take as they are: any whose context holds no value at all (a past series known only before
    # it), a known-ahead one missing a value ahead, and without `accepts_missing` any missing a value anywhere.
    missing: Tuple[str, ...] = ()

    @property
    def target_day(self) -> datetime.date:
        return self.issue_day + datetime.timedelta(days=self.horizon)

    @property
    def steps_ahead(self) -> int:
        """S: the steps from the target's cut to the end of the target day."""
        return len(self.slot_periods)

    @property
    def slots(self) -> Tuple[Slot, ...]:
        """(S,): the (target day, period) each step ahead forecasts, None where it forecasts none."""
        day = self.target_day
        return tuple(None if period == _NONE else (day, int(period)) for period in self.slot_periods)


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


@dataclass(frozen=True, eq=False)
class HistoryBatch:
    """The `HistoryItem`s of some issue days at one horizon, in date order, and how to call a backend on them."""

    spec: HistorySpec
    freq: str
    horizon: int
    steps: int  # T
    periods_per_day: int  # of an ordinary day, which every forecast's series serve
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
        if any(item.target is None for item in self.items):
            raise ValueError("these items hold the cuts alone (arrays=False): read them with arrays to call a backend")
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
    """The regular steps of a frame in physical time (`steps`, in the index's zone) over what a call reads, each step's local
    date as an ordinal, period and wall-clock minute, and the values of the series on them (NaN where the frame has no row),
    read-only, with a running count of their missing values."""

    steps: pd.DatetimeIndex
    ordinal: np.ndarray
    period: np.ndarray
    minute: np.ndarray
    values: Dict[str, np.ndarray]
    gaps: Dict[str, np.ndarray]  # missing values before each position: gaps[name][b] - gaps[name][a] counts [a, b)


def _midnight(day: datetime.date, tz: Any) -> pd.Timestamp:
    """Local midnight of `day` in `tz` (naive if None), as `information_set._Context.origin` reads it."""
    stamp = pd.Timestamp(day)
    return stamp if tz is None else stamp.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")


def _grid(
    frame: pd.DataFrame, names: Sequence[str], freq: str, start: pd.Timestamp, end: pd.Timestamp, arrays: bool
) -> _Grid:
    """Every step of `freq` in physical time from `start` to `end`, in the phase of the frame's rows, with the rows of the frame
    between them (only those: a frame of ten years costs what the call reads). A repeated timestamp, an unsorted index or a row
    off the steps raises."""
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or not len(index):
        raise ValueError("the history accessor reads a frame with a DatetimeIndex and rows")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("the history accessor reads a frame whose index is sorted and has no repeated timestamp")
    step = FREQ_STEPS[freq]
    aware = index.tz is not None
    if aware:
        start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    instants = index.tz_convert("UTC") if aware else index
    anchor = instants[0]
    begin = anchor + math.floor((start - anchor) / step) * step
    finish = anchor + math.ceil((end - anchor) / step) * step
    regular = pd.date_range(begin, finish, freq=step)
    lo, hi = int(instants.searchsorted(regular[0])), int(instants.searchsorted(regular[-1], side="right"))
    inside = instants[lo:hi]
    if not inside.isin(regular).all():
        off = inside[~inside.isin(regular)][0]
        raise ValueError(
            f"a row at {off} is off the {freq} steps of the frame's first row {anchor}: the history accessor counts a context "
            "in steps of physical time, which needs every row on them"
        )
    steps = regular.tz_convert(index.tz) if aware else regular
    wall = steps.tz_localize(None) if aware else steps
    ordinal = np.asarray((wall.normalize() - _EPOCH) // pd.Timedelta(days=1), dtype=np.int64) + _EPOCH_ORDINAL
    minute = np.asarray(wall.hour * 60 + wall.minute, dtype=np.int64)
    values, gaps = {}, {}
    if arrays:
        table = frame.iloc[lo:hi][list(names)].reindex(steps)
        for name in names:
            column = table[name].to_numpy(dtype=np.float64, na_value=np.nan)
            column.setflags(write=False)
            values[name] = column
            gaps[name] = np.concatenate([[0], np.cumsum(~np.isfinite(column))])
    return _Grid(steps, ordinal, minute // _STEP_MINUTES[freq], minute, values, gaps)


def _issue_dates(issue_days: Iterable[Any], tz: Any) -> List[datetime.date]:
    """The local dates of these issue days, each once, in order: a stamp with a zone is read on the frame's clock (and refused
    on a frame without one), anything else as the date it names."""
    out = set()
    for day in issue_days:
        stamp = pd.Timestamp(day)
        if stamp.tz is not None:
            if tz is None:
                raise ValueError(f"the issue day {day!r} has a zone, and the frame has none to read its date on")
            stamp = stamp.tz_convert(tz)
        out.add(stamp.date())
    return sorted(out)


def history_batch(
    spec: HistorySpec,
    frame: pd.DataFrame,
    issue_days: Iterable[Any],
    information_set: InformationSet,
    freq: str = "1h",
    horizon: int = 1,
    arrays: bool = True,
) -> HistoryBatch:
    """`history_batches` at one horizon."""
    return history_batches(spec, frame, issue_days, information_set, freq, [horizon], arrays)[horizon]


def history_batches(
    spec: HistorySpec,
    frame: pd.DataFrame,
    issue_days: Iterable[Any],
    information_set: InformationSet,
    freq: str = "1h",
    horizons: Sequence[int] = (1,),
    arrays: bool = True,
) -> Dict[int, HistoryBatch]:
    """The `HistoryBatch` of each horizon: the `HistoryItem` of each issue day (local dates of the frame's zone, each once, in
    date order; a stamp with a zone read on the frame's clock), read from `frame` (rows in its own zone, which define local
    dates and periods) and cut by `information_set`'s rules at its issue time on each day (see the module's docstring). Each
    series is read once for every horizon. `arrays=False` reads the cuts alone: every array None, no gaps.

    Raises InformationSetError without an information set, for a series without a rule (whatever `unknown` says: the cut
    decides numbers), and for a target not known from 31 days before an issue day on, or with an older value its context holds
    unknown under a rule over time. A known-ahead or past series not known as far as the item needs is late
    (`HistoryItem.late`)."""
    if information_set is None:
        raise InformationSetError(
            "a sequence model's series are cut where the information set's rules say they are known, so they need one: pass "
            "information_set (or issue_time) to the ModelPipeline, or information_set to run()"
        )
    if freq not in _SUB_DAILY:
        raise ValueError(
            f"a context is counted in steps of physical time: freq is one of {list(_SUB_DAILY)}, got {freq!r}"
        )
    horizons = sorted(set(horizons))
    for horizon in horizons:
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError(f"horizon is a whole number of days from 1, got {horizon!r}")
    if not horizons:
        raise ValueError("history_batches reads one horizon at least")
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
    days = _issue_dates(issue_days, frame.index.tz)
    empty = {h: HistoryBatch(spec, freq, int(h), steps, periods, series) for h in horizons}
    if not days:
        return empty
    tz = frame.index.tz
    step = FREQ_STEPS[freq]
    start = _midnight(days[0] + datetime.timedelta(days=NEWEST_TARGET_FROM), tz) - steps * step
    end = _midnight(days[-1] + datetime.timedelta(days=horizons[-1] + 1), tz)
    grid = _grid(frame, spec.names, freq, start, end, arrays)
    over = {name for name, col in resolved.items() if over_time(col.rule)}
    cuts = _Cuts(spec, grid, StepsReader(rules, grid.steps, days, freq), days, horizons, steps, over)
    return {
        h: HistoryBatch(spec, freq, int(h), steps, periods, series, tuple(cuts.item(i, h) for i in range(len(days))))
        for h in horizons
    }


class _Cuts:
    """Each series read once for every issue day, to the end of the largest horizon's target day, and the items built from it."""

    def __init__(
        self,
        spec: HistorySpec,
        grid: _Grid,
        reader: StepsReader,
        days: List[datetime.date],
        horizons: List[int],
        steps: int,
        over: set,
    ):
        self.spec, self.grid, self.reader, self.days, self.steps, self.over = spec, grid, reader, days, steps, over
        ordinals = np.array([day.toordinal() for day in days], dtype=np.int64)
        self.ordinals = ordinals
        self.walk = np.searchsorted(grid.ordinal, ordinals + NEWEST_TARGET_FROM, side="left")
        # The last step of each target day (each horizon), and of D-1, where a "d-1" model's target ends at the latest.
        self.end = {h: np.searchsorted(grid.ordinal, ordinals + h, side="right") - 1 for h in horizons}
        self.before = np.searchsorted(grid.ordinal, ordinals, side="left") - 1
        # And of the day before each target day: the target's context ends there at the latest, as `newest_targets` caps a
        # "published" row model's newest target at h - 1, so a late issue time (prices of D+1 out by then) still forecasts.
        self.before_target = {h: np.searchsorted(grid.ordinal, ordinals + h - 1, side="right") - 1 for h in horizons}
        last = self.end[horizons[-1]]
        lengths = last - self.walk + 1
        self.starts = np.cumsum(lengths) - lengths
        owners = np.repeat(np.arange(len(days)), lengths)
        place = np.arange(int(lengths.sum())) - np.repeat(self.starts, lengths)
        self.rows = place + np.repeat(self.walk, lengths)
        self.owners = owners
        issue = reader.issue(owners)
        self.available: Dict[str, np.ndarray] = {}
        self.newest: Dict[str, np.ndarray] = {}
        for name in spec.names:
            available = reader.available(name, self.rows, owners)
            unknown = np.where(available > issue, place, np.iinfo(np.int64).max)
            first = np.minimum.reduceat(unknown, self.starts)
            stuck = np.flatnonzero(first == 0)
            if len(stuck):
                i = int(stuck[0])
                raise InformationSetError(
                    f"by its rule '{name}' of {grid.steps[self.walk[i]]} isn't known at the issue time on {days[i]} (it is "
                    f"{reader.label(available[self.starts[i]], i)}): a sequence model's history is read from "
                    f"{-NEWEST_TARGET_FROM} days before the issue day on, and a rule this slow leaves it nothing. Check the rule."
                )
            self.available[name] = available
            self.newest[name] = np.where(first == np.iinfo(np.int64).max, last, self.walk + first - 1)
        self._shared: Dict[tuple, Any] = {}

    def _lateness(self, name: str, i: int, positions: np.ndarray) -> Lateness:
        """When the values of `name` at these grid positions are all known on issue day i, as `listing_rows` shows a late use."""
        at = self.starts[i] + positions - self.walk[i]
        return self._late(float(self.available[name][at].max()), i)

    def _late(self, known: float, i: int) -> Lateness:
        """The lateness of a value known at `known` (the reader's scale) on issue day i."""
        issue = float(self.reader.issue(np.array([i]))[0])
        reader = self.reader
        return Lateness(reader.since_midnight(known, i), reader.since_midnight(issue, i), reader, i, (known, issue))

    def _older(self, name: str, i: int, first: int) -> Optional[Lateness]:
        """Under rules over time, the context's steps before the walk: when its unknown ones are known, None if none is."""
        key = ("older", name, i, first)
        if key not in self._shared:
            rows = np.arange(first, self.walk[i])
            available = self.reader.available(name, rows, np.full(len(rows), i))
            issue = self.reader.issue(np.array([i]))[0]
            unknown = np.flatnonzero(available > issue)
            found = None
            if len(unknown):
                found = (self._late(float(available[unknown].max()), i), self.grid.steps[first + int(unknown[0])])
            self._shared[key] = found
        return self._shared[key]

    def _gaps(self, name: str, first: int, last: int) -> int:
        """Missing values of `name` at grid positions first to last."""
        if last < first:
            return 0
        counts = self.grid.gaps[name]
        return int(counts[last + 1] - counts[first])

    def item(self, i: int, horizon: int) -> HistoryItem:
        spec, grid, day = self.spec, self.grid, self.days[i]
        end = int(self.end[horizon][i])
        newest = {name: min(int(self.newest[name][i]), end) for name in spec.names}
        cut = min(newest[spec.target], int(self.before_target[horizon][i]))
        if spec.newest_target == "d-1":
            cut = min(cut, int(self.before[i]))
        newest[spec.target] = cut
        first = cut - self.steps + 1
        late: Dict[str, Lateness] = {}
        for name in spec.names:
            if name not in self.over or first >= self.walk[i]:
                continue
            hole = self._older(name, i, first)
            if hole is None:
                continue
            if name == spec.target:
                raise InformationSetError(
                    f"by its rules over time the target '{name}' of {hole[1]}, in the context of the issue day {day}, isn't "
                    "known at its issue time: shorten the context or start the test where the rule allows"
                )
            late[name] = hole[0]
        for name in spec.past:
            if newest[name] < cut and not spec.accepts_missing and name not in late:
                late[name] = self._lateness(name, i, np.arange(newest[name] + 1, cut + 1))
        for name in spec.known_ahead:
            if newest[name] < end and name not in late:
                late[name] = self._lateness(name, i, np.arange(newest[name] + 1, end + 1))

        target = past = known = None
        gaps: Dict[str, Tuple[int, int]] = {}
        missing: List[str] = []
        if grid.values:
            target = grid.values[spec.target][first : cut + 1]
            gaps[spec.target] = (self._gaps(spec.target, first, cut), 0)
            past = {}
            for name in spec.past:
                # Its newest known step in the context's positions: -1 or less where every step of the context is after it.
                own = max(newest[name] - first, -1)
                values = grid.values[name][first : cut + 1]
                if own < self.steps - 1 and spec.accepts_missing:
                    key = ("past", name, first, cut, own)
                    if key not in self._shared:
                        tail = values.copy()
                        tail[own + 1 :] = np.nan
                        tail.setflags(write=False)
                        self._shared[key] = tail
                    values = self._shared[key]
                    gaps[name] = (self._gaps(name, first, first + own), 0)
                else:
                    gaps[name] = (self._gaps(name, first, cut), 0)
                if own < 0 and spec.accepts_missing:
                    gaps[name] = (0, 0)
                past[name] = values
            known = {}
            for name in spec.known_ahead:
                known[name] = grid.values[name][first : end + 1]
                gaps[name] = (self._gaps(name, first, cut), self._gaps(name, cut + 1, end))
            for name in spec.names:
                context, ahead = gaps[name]
                held = (past[name] if name in spec.past else known.get(name, target))[: self.steps]
                if not np.isfinite(held).any() or ahead or (context and not spec.accepts_missing):
                    missing.append(name)

        target_day = day + datetime.timedelta(days=horizon)
        ahead = np.arange(cut + 1, end + 1)
        on_target = grid.ordinal[ahead] == target_day.toordinal()
        periods = np.where(on_target, grid.period[ahead], _NONE)
        # A period whose step comes again later on the target day is the repeated autumn hour's first: the second forecasts it.
        later = np.flatnonzero(on_target)
        if len(later):
            _, last = np.unique(periods[later][::-1], return_index=True)
            keep = later[len(later) - 1 - last]
            periods[np.setdiff1d(later, keep)] = _NONE
        periods.setflags(write=False)

        def wall(position: int) -> float:
            return float((grid.ordinal[position] - self.ordinals[i]) * 1440 + grid.minute[position])

        read = {name: min(position, cut) if name in spec.past else position for name, position in newest.items()}

        return HistoryItem(
            issue_day=day,
            horizon=horizon,
            target=target,
            past=past,
            known=known,
            slot_periods=periods,
            first_step=grid.steps[first],
            newest={name: grid.steps[position] for name, position in read.items()},
            newest_minutes={name: wall(position) for name, position in read.items()},
            late=late,
            gaps=gaps,
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
    "history_batches",
    "history_spec_of",
    "reads_history",
]
