"""Predictor functions of the forecast's context.

A model's `predictors` may be a function with one parameter named `ctx`. It gets a `PredictorContext` for one planned
forecast (issue day, horizon, period) and returns the full list of column names for that forecast, or a `Use`.
`build_plan` evaluates it for every planned forecast before any fit and returns a `PredictorPlan`: the distinct lists and
which forecast uses which. Each forecast is still its own fit, with its own list; forecasts are grouped by list only for
the information-set check, the manifest and the records.

`ctx` never hands out the frame. `ctx.published(columns)` and `ctx.value(column)` read single cells of the forecast's own
row (issue day D, this period), and only cells the information set says are known at the issue time. They answer through
the same availability path as `InformationSet.check` (`_IssueClock`), so a profile column is judged at its slot, a day
aggregate at the last slot of its day, and either through the column it was built from.
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
    InformationSet,
    InformationSetError,
    ResolvedColumn,
    Violation,
    _fmt_minute,
)
from ._contract import period_index, wall_clock

_HORIZON = "{horizon}"
# How many lists an InformationSetError spells out; a late column often sits in every list.
_LISTS_SHOWN = 5


@dataclass(frozen=True)
class Use:
    """What a predictor function returns when a list needs a training rule.

    Args:
        predictors: The full list of column names, as a plain returned list would be.
        train_on: `"all"` trains on the window's rows where the list's columns exist, as a plain list does.
    """

    predictors: Sequence[str]
    train_on: str = "all"

    def __post_init__(self):
        if self.train_on == "matching":
            raise NotImplementedError(
                "Use(..., train_on='matching') isn't available yet: training only on the past days that got the same "
                "list needs the function evaluated on every training day too. Use train_on='all' for now."
            )
        if self.train_on != "all":
            raise ValueError(f"train_on must be 'all', got {self.train_on!r}")
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

    A repeated autumn hour reads its second occurrence, the row the model forecasts from and `InformationSet.violations`
    judges. A period the day doesn't have (spring) or a row the data lacks is NaN.
    """

    def __init__(
        self, data: pd.DataFrame, information_set: Optional[InformationSet], freq: str, issue_days: Sequence[Any]
    ):
        index = data.index
        self._data = data
        self._info = information_set
        self._step = _STEP_MINUTES[freq]
        self.freq = freq
        self.periods = 1440 // self._step
        self.zone = str(index.tz) if getattr(index, "tz", None) is not None else None
        days = pd.DatetimeIndex(issue_days).as_unit("ns")
        self._n_days = len(days)
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
        self._published: Dict[Tuple[Any, int], np.ndarray] = {}
        # Per argument of `published`, as it was written: the names it stood for, how often it was asked and whether it
        # ever answered True.
        self._asked: Dict[Any, List[Any]] = {}

    @property
    def queried(self) -> Set[str]:
        return set(self._values)

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
        """When each cell of a column with a rule is known, in minutes after the first issue day's midnight."""
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
            else:
                if self.column(name).rule is None:
                    # Whatever `unknown` says about a predictor: what can't be told to be known isn't handed out.
                    raise InformationSetError(
                        f"ctx: no availability rule for '{self.column(name).series}' (column '{name}'), so it can't be "
                        "told whether its value is known at the issue time. Declare one with InformationSet(rules={...})."
                    )
                grid = self.available(name) <= self._clock.issue
            self._known[name] = grid
        return grid

    def known_at(self, name: str, period: int, position: int) -> str:
        return _fmt_minute(self._clock.minute_of_day(self.available(name)[period, position], position))

    def published(self, columns: Any, horizon: int, period: int, position: int) -> bool:
        grid = self._published.get((columns, horizon))
        if grid is None:
            grid = self._published_grid(columns, horizon)
        asked = self._asked[columns]
        asked[1] += 1
        if grid[period, position]:
            asked[2] = True
            return True
        return False

    def _published_grid(self, columns: Any, horizon: int) -> np.ndarray:
        names = [_expand(name, horizon) for name in ([columns] if isinstance(columns, str) else columns)]
        # Every name, whatever the earlier ones answer: a misspelt one must not hide behind a False.
        values = [self.values(name) for name in names]
        grid = np.ones((self.periods, self._n_days), dtype=bool)
        for name, column in zip(names, values):
            grid = grid & self.known(name) & self._present(column)
        self._published[(columns, horizon)] = grid
        self._asked.setdefault(columns, [set(), 0, False])[0].update(names)
        return grid

    @staticmethod
    def _present(values: np.ndarray) -> np.ndarray:
        return pd.notna(values)

    def value(self, column: str, horizon: int, period: int, position: int, issue_day: datetime.date) -> Any:
        name = _expand(column, horizon)
        values = self.values(name)
        if not self.known(name)[period, position]:
            raise InformationSetError(
                f"ctx.value('{name}') on issue day {issue_day}, period {period}: by its availability rule it is known "
                f"{self.known_at(name, period, position)}, after the issue time ({self._info.issue_time} "
                f"{self._info.issue_timezone})."
            )
        value = values[period, position]
        return float(value) if values.dtype != object else value

    def never_published(self) -> List[Tuple[Any, int, str]]:
        """Each `published` argument that never answered True: (argument, times asked, why, as far as can be told)."""
        out = []
        for columns, (names, asked, answered) in self._asked.items():
            if not answered:
                out.append((columns, asked, self._why_not_published(sorted(names))))
        return out

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
        """
        key = columns if isinstance(columns, str) else tuple(columns)
        return self._cells.published(key, self.horizon, self.period, self._position)

    def value(self, column: str) -> Any:
        """The value at this forecast's row (D, this period), NaN if missing.

        Raises `InformationSetError` if the information set says it is known only after the issue time, and `KeyError`
        for a column that isn't in the frame. Without an information set it returns the value.
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
        calls: How many times the function was called.
        seconds: How long planning took.
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
    _positions: Dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self._positions = {day.isoformat(): position for position, day in enumerate(self.issue_days)}

    def list_id_of(self, period: int, horizon: int, run_date: str) -> Optional[int]:
        """The list id of the forecast with this key, or None if this plan doesn't cover it."""
        position = self._positions.get(run_date)
        ids = self.assignment.get((period, horizon))
        return None if position is None or ids is None else int(ids[position])

    def names_by_pair(self) -> Dict[Tuple[int, int], List[str]]:
        """(period, horizon) -> every name any of its forecasts uses."""
        out = {}
        for pair, ids in self.assignment.items():
            names: Dict[str, None] = {}
            for list_id in np.unique(ids).tolist():
                names.update(dict.fromkeys(self.lists[list_id]))
            out[pair] = list(names)
        return out

    def describe(self) -> List[Dict[str, Any]]:
        """One entry per distinct list: its hash, rule, names, and how many forecasts use it where."""
        uses: List[Dict[str, Any]] = [
            {"forecasts": 0, "horizons": set(), "periods": set(), "days": np.zeros(len(self.issue_days), dtype=bool)}
            for _ in self.lists
        ]
        for (period, horizon), ids in self.assignment.items():
            for list_id in np.unique(ids).tolist():
                on = ids == list_id
                use = uses[list_id]
                use["forecasts"] += int(on.sum())
                use["horizons"].add(horizon)
                use["periods"].add(period)
                use["days"] |= on
        out = []
        for list_id, use in enumerate(uses):
            days = np.flatnonzero(use["days"])
            out.append(
                {
                    "hash": self.hashes[list_id],
                    "train_on": self.train_on[list_id],
                    "n_predictors": len(self.lists[list_id]),
                    "predictors": list(self.lists[list_id]),
                    "forecasts": use["forecasts"],
                    "horizons": sorted(use["horizons"]),
                    "periods": sorted(use["periods"]),
                    "issue_days": len(days),
                    "first_issue_day": self.issue_days[days[0]].isoformat(),
                    "last_issue_day": self.issue_days[days[-1]].isoformat(),
                }
            )
        return out

    def function_record(self) -> Dict[str, str]:
        """What a manifest records about the function."""
        return {
            "path": function_path(self.function),
            "fingerprint": function_fingerprint(self.function, self.information_set),
        }

    def manifest_lists(self, masks: Dict[Tuple[int, int, int], np.ndarray]) -> List[Dict[str, Any]]:
        """`describe()` with a digest of the binary-column mask of each (period, horizon) that uses the list."""
        entries = self.describe()
        for entry in entries:
            entry["binary_masks"] = {}
        for (list_id, period, horizon), mask in masks.items():
            entries[list_id]["binary_masks"][f"{period}-{horizon}"] = mask_digest(mask)
        return entries

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
        cells = self._cells or _Cells(self.data, info, self.freq, self.issue_days)
        first, last = self.issue_days[0].isoformat(), self.issue_days[-1].isoformat()
        boundary = {(period, 1): [] for period in range(cells.periods)}
        target_violations, unknown = info.violations(boundary, target, self.freq, self.data.index, first, last)

        late_names: List[str] = []
        for name in dict.fromkeys(name for names in self.lists for name in names):
            col = cells.column(name)
            if col.rule is None:
                unknown.add(col.series)
            elif (cells.available(name) > cells._clock.issue).any():
                late_names.append(name)
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

    def _late_uses(self, late_names: List[str], cells: _Cells) -> Dict[int, List[Violation]]:
        """List id -> its uses of values known after the issue time, on the issue day each is latest."""
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
                    used = ids == list_id
                    if not used.any():
                        continue
                    row = np.array([period])
                    late = info._late_periods(col, row, avail[row], cells._clock, step, used=used[np.newaxis])
                    for at, known, issue, hint in late:
                        hint = hint or info._weather_cutoff_hint(col)
                        out.setdefault(list_id, []).append(Violation(name, at, horizon, col, known, issue, hint=hint))
        return out

    def release(self) -> None:
        """Drop the cells read while planning: a loop over many models would otherwise keep them all."""
        self._cells = None


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
) -> PredictorPlan:
    """Evaluate `fn` for every planned forecast. `warn_unpublished` warns for each `published()` argument that was never
    True over these days, which is meaningless for a sample of days: a wrong rule would silently disable an input."""
    started = time.perf_counter()
    days = list(issue_days)
    cells = _Cells(data, information_set, freq, days)
    plan = PredictorPlan(fn, data, information_set, days, horizon, freq, _cells=cells)
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
                try:
                    result = fn(PredictorContext(cells, position, period, hz, day, target_day))
                except Exception as exc:
                    exc.add_note(
                        f"Raised by predictor function {function_name(fn)} for issue day {day}, horizon {hz}, "
                        f"period {period}."
                    )
                    raise
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
    if warn_unpublished:
        _warn_never_published(cells, fn)
    plan.seconds = time.perf_counter() - started
    return plan


def _warn_never_published(cells: _Cells, fn: Any) -> None:
    for columns, asked, why in cells.never_published():
        shown = repr(columns) if isinstance(columns, str) else f"{list(columns)[:3]}" + (" ..." * (len(columns) > 3))
        warnings.warn(
            f"{function_name(fn)}: ctx.published({shown}) was never True on the {asked} forecast(s) that asked, so the "
            f"function never switched on what it guards: {why}.",
            UserWarning,
            stacklevel=4,
        )


__all__ = ["PredictorContext", "PredictorPlan", "Use", "build_plan", "takes_ctx"]
