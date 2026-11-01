"""Predictor functions of the forecast's context.

A model's `predictors` may be a function with one parameter named `ctx`. It gets a `PredictorContext` for one planned
forecast (issue day, horizon, period) and returns the full list of column names for that forecast, or a `Use`.
`build_plan` evaluates it for every planned forecast before any fit and returns a `PredictorPlan`: the distinct lists and
which forecast uses which. Each forecast is still its own fit, with its own list; forecasts are grouped by list only for
the information-set check, the manifest and the records.

`Use(list, train_on="matching")` trains a forecast only on the past days for which the function returned that same list and
rule. So the plan also evaluates the function on those days, once per (day, period, horizon), each exactly as for a forecast
issued on that day: `ctx` answers as of that day's own issue time, never the time of the forecast being made. A plan with no
such list does none of this.

`ctx` never hands out the frame. `ctx.published(columns)` and `ctx.value(column)` read single cells of the forecast's own
row (issue day D, this period), and only cells the information set says are known at the issue time. They answer through
the same availability path as `InformationSet.check` (`_IssueClock`), so a profile column is judged at its slot, a day
aggregate at the last slot of its day, and either through the column it was built from.

`availability` is the pipeline's level. `published` answers alike at every level, so the plan and the forecasts of a model
that passes "raise" are the same in "warn" and "none"; only what happens around a value that can't be vouched for differs.
At "raise" `value` of a late column raises, and either method raises on a column without a rule. At "warn" they answer (a
column without a rule counts as known), and `PredictorPlan.listing` lists what they read. At "none" they answer and nothing
is listed.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib
import inspect
import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from ..information_set import (
    _LATE_ADVICE,
    _STEP_MINUTES,
    ROLE_PUBLISHED,
    ROLE_TARGET,
    ROLE_VALUE,
    InformationSet,
    InformationSetError,
    ResolvedColumn,
    Violation,
)
from ..results.availability import listing_rows, sort_rows
from ..results.windows import mask_key
from ._calendar import IssueCalendar
from ._contract import period_index, wall_clock
from ._worker import NOT_PLANNED, OTHER_LIST

_HORIZON = "{horizon}"
# How many lists an InformationSetError spells out; a late column often sits in every list.
_LISTS_SHOWN = 5
_STEP_PERIODS = {freq: 1440 // minutes for freq, minutes in _STEP_MINUTES.items()}

TRAIN_ON_ALL = "all"
TRAIN_ON_MATCHING = "matching"
TRAINING_DAY = "training day"  # the last part of a key of `PredictorPlan.read_cells` for a cell read on a training day


@dataclass(frozen=True)
class Use:
    """What a predictor function returns when a list needs a training rule.

    Args:
        predictors: The full list of column names, as a plain returned list would be.
        train_on: `"all"` trains on the window's rows where the list's columns exist, as a plain list does. `"matching"`
            then keeps only the days for which the function returned this same list and `train_on`, per (period, horizon),
            each evaluated as of its own issue time. The window isn't extended, so that is fewer rows.
    """

    predictors: Sequence[str]
    train_on: str = TRAIN_ON_ALL

    def __post_init__(self):
        if self.train_on not in (TRAIN_ON_ALL, TRAIN_ON_MATCHING):
            raise ValueError(f"train_on must be '{TRAIN_ON_ALL}' or '{TRAIN_ON_MATCHING}', got {self.train_on!r}")
        if isinstance(self.predictors, str):
            raise TypeError(f"Use(predictors=...) takes a list of column names, got the string {self.predictors!r}")
        object.__setattr__(self, "predictors", tuple(self.predictors))


def takes_ctx(fn: Any) -> bool:
    """True for a predictor function of the forecast's context: one parameter, named `ctx`."""
    try:
        return list(inspect.signature(fn).parameters) == ["ctx"]
    except (TypeError, ValueError):
        return False


def function_name(fn: Any) -> str:
    return getattr(fn, "__qualname__", repr(fn))


def function_path(fn: Any) -> str:
    """`package.module:qualname`, what a manifest records. Not necessarily importable: see `importable_path`."""
    return f"{getattr(fn, '__module__', '?')}:{function_name(fn)}"


def importable_path(fn: Any, where: str) -> str:
    """`function_path(fn)` if importing it gives `fn` back; a `TypeError` naming `where` otherwise."""
    module, qualname = getattr(fn, "__module__", None), getattr(fn, "__qualname__", None)
    if inspect.ismethod(fn):
        reason = "it is a bound method"
    elif not module or not qualname:
        reason = "it has no import path"
    elif "<lambda>" in qualname:
        reason = "it is a lambda"
    elif "<locals>" in qualname:
        reason = "it is a nested function"
    elif module == "__main__":
        reason = "it is defined in the script being run, which a later load can't import"
    else:
        try:
            reason = None if _import(module, qualname) is fn else f"importing {function_path(fn)} gives another object"
        except (ImportError, AttributeError) as exc:
            reason = f"{function_path(fn)} can't be imported ({exc})"
    if reason:
        raise TypeError(
            f"Cannot save {where}: its predictor function is saved by import path, and {reason}. Define it with `def` "
            "at the top level of a module that can be imported wherever the file is loaded."
        )
    return function_path(fn)


def load_function(path: Any) -> Callable:
    """The predictor function a YAML file names as `package.module:qualname`."""
    module, _, qualname = str(path).partition(":")
    if not module or not qualname:
        raise ValueError(f"A predictor function is saved as 'package.module:qualname', got {path!r}")
    try:
        fn = _import(module, qualname)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import predictor function '{path}': {exc}") from exc
    if not takes_ctx(fn):
        raise TypeError(f"Predictor function '{path}' must take one parameter named `ctx`")
    return fn


def _import(module: str, qualname: str) -> Any:
    found: Any = importlib.import_module(module)
    for part in qualname.split("."):
        found = getattr(found, part)
    return found


def function_fingerprint(fn: Any, information_set: Optional[InformationSet]) -> str:
    """A hash of the function's source and of the information set its `ctx` answers through.

    Provenance: what a function reads from outside (a list defined next to it) isn't in its source, so resuming compares
    the list each record used instead.
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = getattr(getattr(fn, "__code__", None), "co_code", b"").hex() or function_name(fn)
    rules = None
    if information_set is not None:
        sources = {name: rule.to_dict() for name, rule in information_set._source_rules.items()}
        rules = {**information_set.to_dict(), "source_rules": sources}
    payload = json.dumps([source, rules], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def list_hash(names: Sequence[str], train_on: str = "all") -> str:
    """What a record of a predictor function stores: a short hash of the ordered list and its training rule."""
    return hashlib.sha256(json.dumps([list(names), train_on]).encode()).hexdigest()[:12]


def mask_digest(mask: np.ndarray) -> str:
    """A short hash of which columns of a list are scaled."""
    return hashlib.sha256(np.asarray(mask, dtype=bool).tobytes()).hexdigest()[:8]


def _expand(name: Any, horizon: int) -> str:
    if not isinstance(name, str):
        raise TypeError(f"Column names must be strings, got {name!r}")
    return name.replace(_HORIZON, str(horizon))


class _Cells:
    """The cells of every planned forecast's row, by [period, issue day], and whether each is known at the issue time.

    A repeated autumn hour reads its second occurrence, the row the model forecasts from. A period the day doesn't have
    (spring) or a row the data lacks is NaN. Whether a cell is known is what `InformationSet.violations` says of it.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        information_set: Optional[InformationSet],
        freq: str,
        issue_days: Sequence[Any],
        availability: str = "raise",
    ):
        index = data.index
        self._data = data
        self._info = information_set
        self._level = availability
        self._step = _STEP_MINUTES[freq]
        self.freq = freq
        self.periods = 1440 // self._step
        self.zone = str(index.tz) if getattr(index, "tz", None) is not None else None
        days = pd.DatetimeIndex(issue_days).as_unit("ns")
        self._n_days = len(days)
        self._dates = [day.date() for day in days]
        self._planned: Dict[int, np.ndarray] = {}
        # In time order, so the later occurrence of a repeated hour is written last.
        order = np.argsort(index.asi8, kind="stable")
        ordered = index[order]
        position = days.get_indexer(wall_clock(ordered).normalize())
        planned = position >= 0
        self._rows = order[planned]
        self._row_cell = (period_index(ordered, freq)[planned], position[planned])
        self._periods = np.arange(self.periods)
        self._clock = information_set._issue_clock(days, index) if information_set is not None and len(days) else None
        self._columns: Dict[str, ResolvedColumn] = {}
        self._values: Dict[str, np.ndarray] = {}
        self._available: Dict[str, np.ndarray] = {}
        self._known: Dict[str, np.ndarray] = {}
        # (argument of `published`, horizon) -> the names it stands for, its answers, the cells it was asked about.
        self._published: Dict[Tuple[Any, int], Tuple[List[str], np.ndarray, np.ndarray]] = {}
        self._value_cells: Dict[Tuple[str, int], np.ndarray] = {}  # (column, horizon) -> the cells `value` read
        # Per argument of `published`, as it was written: the names it stood for, how often it was asked and whether it
        # ever answered True.
        self._asked: Dict[Any, List[Any]] = {}

    @property
    def queried(self) -> Set[str]:
        return set(self._values)

    def planned(self, horizon: int) -> np.ndarray:
        """Whether the key of each [period, issue day] at this horizon is planned (`IssueCalendar`). The function is asked about
        the others too, but nothing it reads there is a forecast's, so it is neither listed, refused nor compared by a probe."""
        grid = self._planned.get(horizon)
        if grid is None:
            calendar = IssueCalendar(self.freq, self._data.index.tz, [horizon])
            grid = self._planned[horizon] = calendar.planned_cells(self._dates)[horizon]
        return grid

    def has_row(self) -> np.ndarray:
        """Whether the data has a row at each [period, issue day]."""
        present = np.zeros((self.periods, self._n_days), dtype=bool)
        present[self._row_cell] = True
        return present

    def read_cells(self) -> Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]]:
        """(column, horizon) -> the cells whose value the function's answers depended on, and the column's cells.

        `published` doesn't depend on the value of a cell known only later: it answers False whatever the cell holds.
        """
        read: Dict[Tuple[str, int], np.ndarray] = dict(self._value_cells)
        for (_, horizon), (names, _, cells) in self._published.items():
            for name in names:
                known = cells & self.known(name)
                read[(name, horizon)] = read[(name, horizon)] | known if (name, horizon) in read else known
        return {key: (cells & self.planned(key[1]), self._values[key[0]]) for key, cells in read.items()}

    def values(self, name: str) -> np.ndarray:
        grid = self._values.get(name)
        if grid is None:
            if name not in self._data.columns:
                raise KeyError(
                    f"ctx: column '{name}' is not in the data. A misspelt name would make a predictor function switch "
                    "lists without a word, so it raises."
                )
            series = self._data[name]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                grid = np.full((self.periods, self._n_days), np.nan)
                column = series.to_numpy(dtype=float, na_value=np.nan)
            else:
                grid = np.full((self.periods, self._n_days), np.nan, dtype=object)
                column = series.astype(object).to_numpy()
            grid[self._row_cell] = column[self._rows]
            self._values[name] = grid
        return grid

    def column(self, name: str) -> ResolvedColumn:
        col = self._columns.get(name)
        if col is None:
            col = self._columns[name] = self._info._resolve_for(name, self._step)
        return col

    def available(self, name: str) -> np.ndarray:
        """The instant each cell of a column with a rule is known, in minutes since the first issue day's midnight."""
        grid = self._available.get(name)
        if grid is None:
            col = self.column(name)
            self._info._check_source_zone(col, self._data.index)
            grid = self._available[name] = self._clock.available(col, self._periods, self._step)
        return grid

    def known(self, name: str) -> np.ndarray:
        grid = self._known.get(name)
        if grid is None:
            if self._info is None:
                grid = np.ones((self.periods, self._n_days), dtype=bool)
            elif self.column(name).rule is not None:
                grid = self.available(name) <= self._clock.issue
            elif self._level == "raise":
                # Whatever `unknown` says about a predictor: what can't be told to be known isn't handed out.
                raise InformationSetError(
                    f"ctx: no availability rule for '{self.column(name).series}' (column '{name}'), so it can't be "
                    "told whether its value is known at the issue time. Declare one with InformationSet(rules={...})."
                )
            else:
                grid = np.ones((self.periods, self._n_days), dtype=bool)  # listed as "no rule" at the level "warn"
            self._known[name] = grid
        return grid

    def known_at(self, name: str, period: int, position: int) -> str:
        return self._clock.label(self.available(name)[period, position], position)

    def published(self, columns: Any, horizon: int, period: int, position: int) -> bool:
        entry = self._published.get((columns, horizon))
        if entry is None:
            entry = self._published_entry(columns, horizon)
        _, answers, cells = entry
        cells[period, position] = True
        asked = self._asked[columns]
        asked[1] += 1
        if answers[period, position]:
            asked[2] = True
            return True
        return False

    def _published_entry(self, columns: Any, horizon: int) -> Tuple[List[str], np.ndarray, np.ndarray]:
        names = [_expand(name, horizon) for name in ([columns] if isinstance(columns, str) else columns)]
        # Every name, whatever the earlier ones answer: a misspelt one must not hide behind a False.
        values = [self.values(name) for name in names]
        answers = np.ones((self.periods, self._n_days), dtype=bool)
        for name, column in zip(names, values):
            answers = answers & self.known(name) & self._present(column)
        entry = names, answers, np.zeros_like(answers)
        self._published[(columns, horizon)] = entry
        self._asked.setdefault(columns, [set(), 0, False])[0].update(names)
        return entry

    @staticmethod
    def _present(values: np.ndarray) -> np.ndarray:
        return pd.notna(values)

    def value(self, column: str, horizon: int, period: int, position: int, issue_day: datetime.date) -> Any:
        name = _expand(column, horizon)
        values = self.values(name)
        if (
            self._level == "raise"
            and not self.known(name)[period, position]
            and self.planned(horizon)[period, position]
        ):
            raise InformationSetError(
                f"ctx.value('{name}') on issue day {issue_day}, period {period}: by its availability rule it is known "
                f"{self.known_at(name, period, position)}, after the issue time ({self._info.issue_time} "
                f"{self._info.issue_timezone})."
            )
        cells = self._value_cells.get((name, horizon))
        if cells is None:
            cells = self._value_cells[(name, horizon)] = np.zeros(values.shape, dtype=bool)
        cells[period, position] = True
        value = values[period, position]
        return float(value) if values.dtype != object else value

    def late_reads(self, days: Any = None) -> List[Violation]:
        """Each (column, period, horizon) whose `value` was read though its rule puts it after the issue time, on `days` if
        given (a mask of the issue days, or `horizon -> mask of [period, issue day]`)."""
        out = []
        if self._info is None:
            return out
        for (name, horizon), read in self._value_cells.items():
            col = self.column(name)
            if col.rule is None:
                continue
            avail = self.available(name)
            read = self._on(read, horizon, days)
            late = self._info._late_periods(col, self._periods, avail, self._clock, self._step, used=read)
            for found in late:
                out.append(found.use(name, horizon, col, ROLE_VALUE))
        return out

    def no_rule_reads(self, days: Any = None) -> List[Violation]:
        """Each (column, period, horizon) read through `ctx` that has no rule, on `days` if given (see `late_reads`)."""
        out = []
        if self._info is None:
            return out
        for (name, horizon), read in self._value_cells.items():
            out.extend(self._without_rule(name, horizon, self._on(read, horizon, days), ROLE_VALUE))
        for (_, horizon), (names, _, asked) in self._published.items():
            for name in names:
                out.extend(self._without_rule(name, horizon, self._on(asked, horizon, days), ROLE_PUBLISHED))
        return out

    def _on(self, read: np.ndarray, horizon: int, days: Any) -> np.ndarray:
        read = read & self.planned(horizon)
        if days is None:
            return read
        return read & (days.get(horizon, False) if isinstance(days, dict) else days)

    def _without_rule(self, name: str, horizon: int, cells: np.ndarray, role: str) -> List[Violation]:
        col = self.column(name)
        if col.rule is not None:
            return []
        return [
            Violation(name, int(period), horizon, col, float("nan"), float("nan"), role)
            for period in np.flatnonzero(cells.any(axis=1))
        ]

    def never_published(self) -> List[Tuple[Any, int, str]]:
        """Each `published` argument that never answered True: (argument, times asked, why, as far as can be told)."""
        out = []
        for columns, (names, asked, answered) in self._asked.items():
            if not answered:
                out.append((columns, asked, self._why_not_published(sorted(names))))
        return out

    def never_published_on(self, days: np.ndarray) -> List[Tuple[Any, int, str]]:
        """Each `published` argument asked on `days` (a mask of the issue days) that never answered True there."""
        asked: Dict[Any, List[Any]] = {}
        for (columns, _), (names, answers, cells) in self._published.items():
            entry = asked.setdefault(columns, [set(), 0, False])
            entry[0].update(names)
            entry[1] += int((cells & days).sum())
            entry[2] = entry[2] or bool((answers & cells & days).any())
        return [
            (columns, n_asked, self._why_not_published(sorted(names)))
            for columns, (names, n_asked, answered) in asked.items()
            if n_asked and not answered
        ]

    def _why_not_published(self, names: List[str]) -> str:
        reasons = []
        for name in names:
            if not self.known(name).any():
                reasons.append(f"'{name}' is known {self.known_at(name, 0, 0)}, after the issue time")
            elif not self._present(self.values(name)).any():
                reasons.append(f"'{name}' has no value on any forecast's row")
        if reasons:
            shown = "; ".join(reasons[:3]) + (f"; and {len(reasons) - 3} more" if len(reasons) > 3 else "")
            return f"{shown}. Check the rule and the name's day offset"
        return "no forecast that asked had all of them known and present"


class PredictorContext:
    """What a predictor function may know about the forecast being planned. It never sees the frame.

    Attributes:
        horizon: Days ahead of the issue day, from 1.
        period: Period of the target day: 0-23, or 0-95 for 15-minute data. `hour` is the same number.
        issue_day: The local calendar date D the forecast is issued on.
        target_day: The local calendar date D + horizon it is for.
        target_weekday: Weekday of `target_day`, 0 = Monday.
        zone: The data's time zone name, or None for a naive index.
        freq: `"1h"` or `"15min"`.
    """

    __slots__ = ("horizon", "period", "issue_day", "target_day", "zone", "freq", "_cells", "_position")

    def __init__(
        self,
        cells: _Cells,
        position: int,
        period: int,
        horizon: int,
        issue_day: datetime.date,
        target_day: datetime.date,
    ):
        self._cells = cells
        self._position = position
        self.period = period
        self.horizon = horizon
        self.issue_day = issue_day
        self.target_day = target_day
        self.zone = cells.zone
        self.freq = cells.freq

    @property
    def hour(self) -> int:
        return self.period

    @property
    def target_weekday(self) -> int:
        return self.target_day.weekday()

    def published(self, columns: Union[str, Iterable[str]]) -> bool:
        """True if every column's value at this forecast's row (D, this period) is known at the issue time and not NaN.

        `columns` is one name or several, with `{horizon}` expanded. A column that isn't in the frame raises `KeyError`.
        Without an information set availability isn't checked, only NaN. In a backtest "not NaN" is the archive's state,
        not the server's at the issue time: a value that came late and was backfilled reads as published.

        The answer is the same at every `availability` level. A column without a rule raises `InformationSetError` at
        "raise"; at "warn" and "none" only NaN counts for it, and "warn" lists it as "no rule".
        """
        key = columns if isinstance(columns, str) else tuple(columns)
        return self._cells.published(key, self.horizon, self.period, self._position)

    def value(self, column: str) -> Any:
        """The value at this forecast's row (D, this period), NaN if missing.

        With `availability="raise"` it raises `InformationSetError` if the information set says it is known only after
        the issue time or has no rule for it. With "warn" it returns the value and the column is listed in the report, and
        with "none", or without an information set, it returns the value. A column that isn't in the frame raises `KeyError`.
        """
        return self._cells.value(column, self.horizon, self.period, self._position, self.issue_day)

    def __repr__(self) -> str:
        return (
            f"PredictorContext(issue_day={self.issue_day}, target_day={self.target_day}, horizon={self.horizon}, "
            f"period={self.period})"
        )


@dataclass(eq=False)
class PredictorPlan:
    """The distinct lists a predictor function returned and which planned forecast uses which.

    The pipeline builds one per model before any fit and passes it to the model's run. Nothing about it lives on the
    model, so a model can't run with a plan made for another frame or window.

    Attributes:
        function: The predictor function.
        issue_days: The planned issue days, in order.
        horizon: The largest horizon planned.
        freq: `"1h"` or `"15min"`.
        lists: Each distinct list, names expanded, in the order first returned. A list id is a position here.
        train_on: Each list's training rule.
        hashes: Each list's `list_hash`, as records store it.
        assignment: `(period, horizon)` -> the list id of each issue day.
        queried: The columns the function read through `ctx`.
        calls: How many times the function was called, on training days too.
        seconds: How long planning took.
        availability: The level it was planned at, which decides what `ctx` does around a value it can't vouch for.
        training_window: The window the training days of `train_on="matching"` lists were planned for, if any.
        first_grid_day: The first day of `day_lists`, whose entries are one per calendar day up to the last issue day.
        day_lists: `(period, horizon)` of a matching forecast -> the list id of each day of the grid, what a matching forecast
            filters its training rows on: `OTHER_LIST` for a list no forecast uses, which matches nothing, and
            `NOT_PLANNED` for a day no matching forecast of that pair reads. Empty without a matching forecast.
    """

    function: Any
    data: pd.DataFrame = field(repr=False)
    information_set: Optional[InformationSet] = field(repr=False)
    issue_days: List[datetime.date] = field(repr=False)
    horizon: int
    freq: str
    lists: List[Tuple[str, ...]] = field(default_factory=list, repr=False)
    train_on: List[str] = field(default_factory=list, repr=False)
    hashes: List[str] = field(default_factory=list, repr=False)
    assignment: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict, repr=False)
    queried: Set[str] = field(default_factory=set, repr=False)
    calls: int = 0
    seconds: float = 0.0
    _cells: Optional[_Cells] = field(default=None, repr=False)
    _checked: bool = field(default=False, repr=False)
    availability: str = "raise"
    _listings: Dict[Tuple[int, Optional[int]], List[Dict[str, Any]]] = field(default_factory=dict, repr=False)
    _positions: Dict[str, int] = field(init=False, repr=False)
    training_window: Optional[int] = None
    first_grid_day: Optional[datetime.date] = field(default=None, repr=False)
    day_lists: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict, repr=False)
    _training_cells: Optional[_Cells] = field(default=None, repr=False)
    _planned: Optional[Dict[Tuple[int, int], np.ndarray]] = field(default=None, repr=False)

    def __post_init__(self):
        self._positions = {day.isoformat(): position for position, day in enumerate(self.issue_days)}

    @property
    def uses_matching(self) -> bool:
        return TRAIN_ON_MATCHING in self.train_on

    @property
    def matching_ids(self) -> Set[int]:
        """The ids of the lists with `train_on="matching"`."""
        return {list_id for list_id, rule in enumerate(self.train_on) if rule == TRAIN_ON_MATCHING}

    def list_id_of(self, period: int, horizon: int, run_date: str) -> Optional[int]:
        """The list id of the forecast with this key, or None if this plan doesn't cover it."""
        position = self._positions.get(run_date)
        ids = self.assignment.get((period, horizon))
        return None if position is None or ids is None else int(ids[position])

    def planned(self) -> Dict[Tuple[int, int], np.ndarray]:
        """(period, horizon) -> whether the forecast of each issue day is planned: not where the clock doesn't have the period on
        the issue day or the target day (`IssueCalendar`). The function is called there too, so that the plan's calls don't
        depend on the clock, but what it returns or reads there is no forecast's: nothing counts, lists or checks it (`_Cells`
        leaves the reads out)."""
        if self._planned is None:
            calendar = IssueCalendar(self.freq, self.data.index.tz, range(1, self.horizon + 1))
            grids = calendar.planned_cells(self.issue_days)
            self._planned = {(period, horizon): grids[horizon][period] for period, horizon in self.assignment}
        return self._planned

    def names_by_pair(self) -> Dict[Tuple[int, int], List[str]]:
        """(period, horizon) -> every name any of its forecasts uses."""
        out = {}
        for pair, ids in self.assignment.items():
            names: Dict[str, None] = {}
            for list_id in np.unique(ids).tolist():
                names.update(dict.fromkeys(self.lists[list_id]))
            out[pair] = list(names)
        return out

    def position_of(self, day: str) -> int:
        """Where an issue day is in `issue_days`."""
        return self._positions[day]

    def describe(self, days: Tuple[int, Optional[int]] = (0, None)) -> List[Dict[str, Any]]:
        """One entry per distinct list: its hash, rule, names, and how many forecasts use it where.

        `days` is a (start, stop) slice of `issue_days`, for the part of the plan in one window: a list not used there has
        no entry.
        """
        issue_days = self.issue_days[days[0] : days[1]]
        uses: List[Dict[str, Any]] = [
            {"forecasts": 0, "horizons": set(), "periods": set(), "days": np.zeros(len(issue_days), dtype=bool)}
            for _ in self.lists
        ]
        for (period, horizon), all_ids in self.assignment.items():
            ids = all_ids[days[0] : days[1]]
            planned = self.planned()[(period, horizon)][days[0] : days[1]]
            for list_id in np.unique(ids).tolist():
                on = ids == list_id
                use = uses[list_id]
                use["forecasts"] += int((on & planned).sum())
                use["horizons"].add(horizon)
                use["periods"].add(period)
                use["days"] |= on
        out = []
        for list_id, use in enumerate(uses):
            on_days = np.flatnonzero(use["days"])
            if not len(on_days):
                continue
            out.append(
                {
                    "hash": self.hashes[list_id],
                    "train_on": self.train_on[list_id],
                    "n_predictors": len(self.lists[list_id]),
                    "predictors": list(self.lists[list_id]),
                    "forecasts": use["forecasts"],
                    "horizons": sorted(use["horizons"]),
                    "periods": sorted(use["periods"]),
                    "issue_days": len(on_days),
                    "first_issue_day": issue_days[on_days[0]].isoformat(),
                    "last_issue_day": issue_days[on_days[-1]].isoformat(),
                }
            )
        return out

    def function_record(self) -> Dict[str, str]:
        """What a manifest records about the function."""
        return {
            "path": function_path(self.function),
            "fingerprint": function_fingerprint(self.function, self.information_set),
        }

    def manifest_lists(self, masks: Dict[str, Dict[Tuple[int, int, int], np.ndarray]]) -> List[Dict[str, Any]]:
        """`describe()` with a digest of the binary-column mask of each (window, period, horizon) that uses the list."""
        entries = self.describe()
        for entry in entries:
            entry["binary_masks"] = {}
        for window, window_masks in masks.items():
            for (list_id, period, horizon), mask in window_masks.items():
                entries[list_id]["binary_masks"][mask_key(window, period, horizon)] = mask_digest(mask)
        return entries

    def _scan(self, target: str, pairs: Dict[Tuple[int, int], List[str]], days: Tuple[int, Optional[int]] = (0, None)):
        """The cells, the uses of `target` known after the issue time, the columns without a rule, and the names of the
        lists whose value is known after the issue time on some planned forecast or other.

        Availability is computed once per distinct column name, whatever the number of lists.
        """
        info = self.information_set
        cells = self._cells or _Cells(self.data, info, self.freq, self.issue_days, self.availability)
        scanned = self.issue_days[days[0] : days[1]]
        target_violations, unknown = info.violations(
            pairs, target, self.freq, self.data.index, issue_days=[day.isoformat() for day in scanned]
        )
        late_names: List[str] = []
        no_rule: List[str] = []
        in_use = self._lists_in_use()
        for name in dict.fromkeys(name for list_id in in_use for name in self.lists[list_id]):
            col = cells.column(name)
            if col.rule is None:
                no_rule.append(name)
                unknown.add(col.series)
            elif (cells.available(name) > cells._clock.issue).any():
                late_names.append(name)
        return cells, target_violations, unknown, late_names, no_rule

    def check(self, target: str, model_name: str) -> None:
        """Raise `InformationSetError` if a forecast reads a value not known at its issue time, or if the newest training
        target isn't known then.

        Availability is computed once per distinct column name, whatever the number of lists, and only the cells a list
        actually reads count: a list used on Fridays only may hold what is known on Fridays only. Violations are then
        reported per list, with the model, the list, the columns and the periods.
        """
        if self._checked or not self.issue_days:
            return
        info = self.information_set
        boundary = {(period, 1): [] for period in range(_STEP_PERIODS[self.freq])}
        cells, target_violations, unknown, late_names, _ = self._scan(target, boundary)
        info._report_unknown(unknown, model_name)
        if target_violations:
            raise InformationSetError(info._format(target_violations, model_name))

        by_list = self._late_uses(late_names, cells)
        if by_list:
            shown = sorted(by_list)[:_LISTS_SHOWN]
            names = {i: f"{model_name}, predictor list {i + 1} of {len(self.lists)} ({self.hashes[i]})" for i in shown}
            parts = [info._format(by_list[i], names[i], advice=False) for i in shown]
            if len(by_list) > len(shown):
                parts.append(f"... and {len(by_list) - len(shown)} more list(s) with late inputs.")
            raise InformationSetError("\n".join([*parts, _LATE_ADVICE]))
        info._log_assumptions(model_name)
        self._checked = True

    def listing(
        self, target: str, model_name: str = "Model", days: Tuple[int, Optional[int]] = (0, None)
    ) -> List[Dict[str, Any]]:
        """What `check` refuses, and every input without a rule, per list, as the rows of `availability="warn"`.

        Besides the lists it holds what the function only read: a `value` known after the issue time, and any column
        without a rule read through `ctx`. A column of a data source on an index in another zone still raises. Built once,
        so a run reuses what `ModelPipeline` found before any model ran. `days` is a (start, stop) slice of `issue_days`, for
        what one window of a run uses.
        """
        if days in self._listings:
            return self._listings[days]
        if self.information_set is None or not self.issue_days:
            self._listings[days] = []
            return self._listings[days]
        pairs = {(period, hz): [] for period, hz in self.assignment}
        cells, target_uses, _, late_names, no_rule = self._scan(target, pairs, days)
        on_days = None
        read_on, read_on_training = None, None
        if days != (0, None):
            on_days = np.zeros(len(self.issue_days), dtype=bool)
            on_days[days[0] : days[1]] = True
            read_on, read_on_training = self._read_by_forecasts_of(days, on_days)
        col = cells.column(target)
        if col.rule is None:
            target_uses = [Violation(target, p, h, col, float("nan"), float("nan"), ROLE_TARGET) for p, h in pairs]
        rows = listing_rows(target_uses)
        by_list = self._late_uses(late_names, cells, on_days)
        pairs_of_list = self._pairs_of_lists(on_days)
        for list_id, list_hash in enumerate(self.hashes):
            uses = list(by_list.get(list_id, []))
            names = set(self.lists[list_id])
            for name in no_rule:
                if name in names:
                    col = cells.column(name)
                    nan = float("nan")
                    uses.extend(Violation(name, p, h, col, nan, nan) for p, h in pairs_of_list[list_id])
            rows += listing_rows(uses, list_hash)
        reads = [*cells.late_reads(read_on), *cells.no_rule_reads(read_on)]
        if self._training_cells is not None:
            training = self._training_cells
            reads += [*training.late_reads(read_on_training), *training.no_rule_reads(read_on_training)]
        rows += listing_rows(reads)
        self.information_set._log_assumptions(model_name)
        self._listings[days] = sort_rows(rows)
        return self._listings[days]

    def _read_by_forecasts_of(self, days: Tuple[int, Optional[int]], on_days: np.ndarray) -> Tuple[Any, Any]:
        """What the forecasts of `days`, a (start, stop) slice of `issue_days`, had the function read: the mask to read
        `_cells` with and the one to read `_training_cells` with (see `_Cells.late_reads`).

        A window's forecasts read their own days and, through a matching list, the days they train on, also if those are days
        of another window: a plain run over this window evaluates them as training days and lists what they read.
        """
        if not self.day_lists:
            return on_days, None
        grid = self._grid_positions()
        is_training_day = np.ones(self._grid_size(), dtype=bool)
        is_training_day[grid] = False
        periods, horizons = _STEP_PERIODS[self.freq], range(1, self.horizon + 1)
        on_issue_days = {horizon: np.tile(on_days, (periods, 1)) for horizon in horizons}
        on_training_days = {
            horizon: np.zeros((periods, int(is_training_day.sum())), dtype=bool) for horizon in horizons
        }
        slice_of = slice(days[0], days[1])
        for (period, horizon), ids in self.assignment.items():
            read = self._days_read_by(ids[slice_of], grid[slice_of], horizon)
            if read is not None:
                on_issue_days[horizon][period] |= read[grid]
                on_training_days[horizon][period] = read[is_training_day]
        return on_issue_days, on_training_days

    def _days_read_by(self, ids: np.ndarray, grid: np.ndarray, horizon: int) -> Optional[np.ndarray]:
        """Which days of the grid these forecasts of one (period, horizon) can train on, or None if none of them is matching.
        Each reads `[D - training_window - horizon, D - 1]`, which holds its `training_span` and more: the calls of a plan don't
        depend on how far back from the issue day the rows end."""
        matching = np.isin(ids, sorted(self.matching_ids))
        if not matching.any():
            return None
        marks = np.zeros(self._grid_size() + 1, dtype=np.int64)
        np.add.at(marks, grid[matching] - self.training_window - horizon, 1)
        np.add.at(marks, grid[matching], -1)
        return np.cumsum(marks[:-1]) > 0

    def _grid_positions(self) -> np.ndarray:
        """Where each issue day is in the grid of `day_lists`."""
        first = self.first_grid_day.toordinal()
        return np.array([day.toordinal() - first for day in self.issue_days], dtype=np.int64)

    def _grid_size(self) -> int:
        return max(self.issue_days).toordinal() - self.first_grid_day.toordinal() + 1

    def grid_days(self) -> List[datetime.date]:
        """Every calendar day of `day_lists`, the first one first."""
        return [self.first_grid_day + datetime.timedelta(days=d) for d in range(self._grid_size())]

    def _pairs_of_lists(self, days: Optional[np.ndarray] = None) -> Dict[int, List[Tuple[int, int]]]:
        """List id -> the (period, horizon) pairs whose planned forecasts use it, on `days` (a mask of the issue days) if
        given."""
        out: Dict[int, List[Tuple[int, int]]] = {list_id: [] for list_id in range(len(self.lists))}
        for pair, ids in self.assignment.items():
            on = self.planned()[pair] if days is None else self.planned()[pair] & days
            for list_id in np.unique(ids[on]).tolist():
                out[list_id].append(pair)
        return out

    def _lists_in_use(self) -> List[int]:
        """The ids of the lists some planned forecast uses."""
        return sorted(list_id for list_id, pairs in self._pairs_of_lists().items() if pairs)

    def late_reads(self) -> List[Violation]:
        """The `value` reads of cells known only after the issue time of their own day, on a training day too, for
        `leakage_probe`, until `release`."""
        if self._cells is None:
            return []
        return [*self._cells.late_reads(), *(self._training_cells.late_reads() if self._training_cells else ())]

    def _late_uses(
        self, late_names: List[str], cells: _Cells, days: Optional[np.ndarray] = None
    ) -> Dict[int, List[Violation]]:
        """List id -> its uses of values known after the issue time, on the issue day each is latest, among `days` (a mask
        of the issue days) if given."""
        if not late_names:
            return {}
        info = self.information_set
        step = _STEP_MINUTES[self.freq]
        out: Dict[int, List[Violation]] = {}
        lists = [set(names) for names in self.lists]
        for name in late_names:
            col, avail = cells.column(name), cells.available(name)
            for list_id in (list_id for list_id, names in enumerate(lists) if name in names):
                for (period, horizon), ids in self.assignment.items():
                    used = (ids == list_id) & self.planned()[(period, horizon)]
                    if days is not None:
                        used = used & days
                    if not used.any():
                        continue
                    row = np.array([period])
                    late = info._late_periods(col, row, avail[row], cells._clock, step, used=used[np.newaxis])
                    for found in late:
                        use = found.use(name, horizon, col, hint=info._weather_cutoff_hint(col))
                        out.setdefault(list_id, []).append(use)
        return out

    def read_cells(self) -> Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]]:
        """What the function's answers depended on, for `leakage_probe`, until `release`: (column, horizon) -> the cells
        [period, issue day] it read, and the column's cells. What it read on a training day is under (column, horizon,
        `TRAINING_DAY`), on those days: a frame on which no forecast is matching has none to compare."""
        if self._cells is None:
            return {}
        read: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]] = dict(self._cells.read_cells())
        if self._training_cells is not None:
            read.update({(*key, TRAINING_DAY): cells for key, cells in self._training_cells.read_cells().items()})
        return read

    def release(self) -> None:
        """Drop the cells read while planning: a loop over many models would otherwise keep them all."""
        self._cells = None
        self._training_cells = None


def _names(result: Any, fn: Any) -> Tuple[Sequence[str], str]:
    if isinstance(result, Use):
        return result.predictors, result.train_on
    if isinstance(result, (list, tuple)):
        return result, "all"
    raise TypeError(
        f"Predictor function {function_name(fn)} returned {type(result).__name__}; expected a list of column names or "
        "Use(list, train_on=...)"
    )


def _expanded_list(names: Sequence[str], horizon: int, fn: Any) -> Tuple[str, ...]:
    if not names:
        raise ValueError(f"Predictor function {function_name(fn)} returned an empty list")
    bad = [name for name in names if not isinstance(name, str)]
    if bad:
        raise TypeError(f"Predictor function {function_name(fn)} returned something other than names: {bad[:3]}")
    expanded = tuple(_expand(name, horizon) for name in names)
    if len(set(expanded)) != len(expanded):
        repeated = sorted({name for name in expanded if expanded.count(name) > 1})
        raise ValueError(f"Predictor function {function_name(fn)} returned repeated names: {repeated[:3]}")
    return expanded


def build_plan(
    fn: Callable,
    data: pd.DataFrame,
    issue_days: Sequence[datetime.date],
    horizon: int,
    freq: str,
    information_set: Optional[InformationSet] = None,
    warn_unpublished: bool = True,
    availability: str = "raise",
    training_window: Optional[int] = None,
) -> PredictorPlan:
    """Evaluate `fn` for every planned forecast. `warn_unpublished` warns for each `published()` argument that was never
    True over these days, which is meaningless for a sample of days: a wrong rule would silently disable an input.
    `availability` is the level `ctx` answers at (see the module docstring).

    If a forecast gets `Use(list, train_on="matching")`, `fn` is also evaluated on the days it can train on, which needs
    `training_window`. What `fn` answers on a day is then the same whether that day is issued or only trained on.
    """
    started = time.perf_counter()
    days = list(issue_days)
    cells = _Cells(data, information_set, freq, days, availability)
    plan = PredictorPlan(fn, data, information_set, days, horizon, freq, _cells=cells, availability=availability)
    horizons = range(1, horizon + 1)
    periods = range(cells.periods)
    for hz in horizons:
        for period in periods:
            plan.assignment[(period, hz)] = np.full(len(days), -1, dtype=np.int32)

    returned: Dict[Tuple[Tuple[str, ...], str, int], int] = {}  # what the function returned, before validating it
    distinct: Dict[Tuple[Tuple[str, ...], str], int] = {}  # the same list, whatever horizon's template it came from
    for position, day in enumerate(days):
        for hz in horizons:
            target_day = day + datetime.timedelta(days=hz)
            for period in periods:
                result = _call(fn, PredictorContext(cells, position, period, hz, day, target_day))
                names, train_on = _names(result, fn)
                key = (tuple(names), train_on, hz)
                list_id = returned.get(key)
                if list_id is None:
                    expanded = _expanded_list(names, hz, fn)
                    list_id = distinct.get((expanded, train_on))
                    if list_id is None:
                        list_id = distinct[(expanded, train_on)] = len(plan.lists)
                        plan.lists.append(expanded)
                        plan.train_on.append(train_on)
                        plan.hashes.append(list_hash(expanded, train_on))
                    returned[key] = list_id
                plan.assignment[(period, hz)][position] = list_id
    plan.calls = len(days) * len(horizons) * len(periods)
    plan.queried = cells.queried
    only_on_training_days = _plan_training_days(plan, returned, distinct, training_window) if plan.uses_matching else []
    _warn_same_names_in_another_order([*plan.lists, *only_on_training_days], fn)
    if warn_unpublished:
        _warn_never_published(cells, fn)
    plan.seconds = time.perf_counter() - started
    return plan


def _call(fn: Callable, ctx: PredictorContext, training: bool = False) -> Any:
    try:
        return fn(ctx)
    except Exception as exc:
        as_what = "for training day" if training else "for issue day"
        exc.add_note(
            f"Raised by predictor function {function_name(fn)} {as_what} {ctx.issue_day}, horizon {ctx.horizon}, "
            f"period {ctx.period}."
        )
        raise


def _plan_training_days(
    plan: PredictorPlan,
    returned: Dict[Tuple[Tuple[str, ...], str, int], int],
    distinct: Dict[Tuple[Tuple[str, ...], str], int],
    training_window: Optional[int],
) -> List[Tuple[str, ...]]:
    """The list of each day a matching forecast can train on, into `plan.day_lists`. Returns the lists no forecast uses.

    A row on day D' at horizon h has issue day D' and target day D' + h, so the function runs for it exactly as for a
    forecast issued on D': over `_Cells` of those days, whose clock gives each its own issue time. A day that is an issue
    day already has its answer, and only a (day, period, horizon) with a row and a matching forecast to read it is asked.
    A list no forecast uses is validated like any other, and gets `OTHER_LIST`: no forecast can match it, and it must not
    show in the plan's lists, whose ids the manifest, the masks and the check go through.
    """
    fn = plan.function
    if training_window is None:
        raise ValueError(
            f"{function_name(fn)} returned Use(..., train_on='matching'), which needs the training_window to plan"
        )
    plan.training_window = training_window
    plan.first_grid_day = min(plan.issue_days) - datetime.timedelta(days=training_window + plan.horizon)
    grid = plan._grid_positions()
    is_issue_day = np.zeros(plan._grid_size(), dtype=bool)
    is_issue_day[grid] = True
    grid_days = plan.grid_days()
    training_days = [day for day, issued in zip(grid_days, is_issue_day) if not issued]
    cells = _Cells(plan.data, plan.information_set, plan.freq, training_days, plan.availability)
    plan._training_cells = cells
    has_row = cells.has_row()
    position_of = np.full(len(grid_days), -1, dtype=np.int64)
    position_of[~is_issue_day] = np.arange(len(training_days))

    asked: Dict[Tuple[int, int], np.ndarray] = {}  # (period, horizon) -> which training days its forecasts read
    for (period, hz), ids in plan.assignment.items():
        read = plan._days_read_by(ids, grid, hz)
        if read is None:
            continue
        plan.day_lists[(period, hz)] = lists = np.full(len(grid_days), NOT_PLANNED, dtype=np.int32)
        lists[grid] = ids
        asked_days = np.zeros(len(training_days), dtype=bool)
        asked_days[position_of[read & ~is_issue_day]] = True
        asked[(period, hz)] = asked_days & has_row[period]

    horizons = range(1, plan.horizon + 1)
    periods = range(cells.periods)
    first = plan.first_grid_day.toordinal()
    calls = 0
    only_here: List[Tuple[str, ...]] = []
    for position, day in enumerate(training_days):
        place = day.toordinal() - first
        for hz in horizons:
            target_day = day + datetime.timedelta(days=hz)
            for period in periods:
                pair = (period, hz)
                if pair not in asked or not asked[pair][position]:
                    continue
                result = _call(fn, PredictorContext(cells, position, period, hz, day, target_day), training=True)
                calls += 1
                names, train_on = _names(result, fn)
                key = (tuple(names), train_on, hz)
                list_id = returned.get(key)
                if list_id is None:
                    expanded = _expanded_list(names, hz, fn)
                    list_id = returned[key] = distinct.get((expanded, train_on), OTHER_LIST)
                    if list_id == OTHER_LIST:
                        only_here.append(expanded)
                plan.day_lists[pair][place] = list_id
    plan.calls += calls
    plan.queried = plan.queried | cells.queried
    return only_here


def _warn_same_names_in_another_order(lists: Iterable[Tuple[str, ...]], fn: Any) -> None:
    """The order of a list is part of it (`list_hash`), and `list(set(names))` has one that changes with the hash seed, and the
    numbers with it. Only a function that returns two orders in one run can be told by this."""
    orders: Dict[frozenset, Dict[Tuple[str, ...], None]] = {}
    for names in lists:
        orders.setdefault(frozenset(names), {})[names] = None
    reordered = [list(found) for found in orders.values() if len(found) > 1]
    if reordered:
        first, second = reordered[0][:2]
        warnings.warn(
            f"{function_name(fn)} returned the same names in another order, e.g. {list(first[:3])} ... and {list(second[:3])} "
            "... The order is part of a list: these are two lists, with their columns in another order and another hash, and "
            "train_on='matching' doesn't match one with the other. An order taken from a set also changes with the hash seed, "
            "and the numbers with it. Sort the names.",
            UserWarning,
            stacklevel=4,
        )


def _warn_never_published(cells: _Cells, fn: Any) -> None:
    for columns, asked, why in cells.never_published():
        shown = repr(columns) if isinstance(columns, str) else f"{list(columns)[:3]}" + (" ..." * (len(columns) > 3))
        warnings.warn(
            f"{function_name(fn)}: ctx.published({shown}) was never True on the {asked} forecast(s) that asked, so the "
            f"function never switched on what it guards: {why}.",
            UserWarning,
            stacklevel=4,
        )


def warn_never_published_in(plan: PredictorPlan, window: str, days: Tuple[int, int]) -> None:
    """What a plain run over one window warns about, for a run of several: a `published()` argument never True in that
    window, unless it wasn't True anywhere, which `build_plan` has said already."""
    cells = plan._cells
    on_days = np.zeros(len(plan.issue_days), dtype=bool)
    on_days[days[0] : days[1]] = True
    everywhere = {columns for columns, _, _ in cells.never_published()}
    for columns, asked, why in cells.never_published_on(on_days):
        if columns in everywhere:
            continue
        shown = repr(columns) if isinstance(columns, str) else f"{list(columns)[:3]}" + (" ..." * (len(columns) > 3))
        warnings.warn(
            f"{function_name(plan.function)}: ctx.published({shown}) was never True on the {asked} forecast(s) that asked "
            f"in the {window} window, so the function never switched on what it guards there: {why}.",
            UserWarning,
            stacklevel=4,
        )


__all__ = ["PredictorContext", "PredictorPlan", "Use", "build_plan", "takes_ctx"]
