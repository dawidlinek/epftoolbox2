"""The listing `availability="warn"` makes: every input a run uses though its rule puts it after the issue time, and every
input without a rule, per model, with the periods and horizons concerned.

One row says "these periods at these horizons", and both are exact: a predictor used at other periods on other horizons
gets one row per distinct set of periods, so the rows of a key add up to exactly its uses. A manifest stores the rows, so a
report built later (`ModelPipeline.report` after array jobs) shows what the run listed. `merge_listings` keeps them when a
narrower run into the same directory rewrites the manifest: the records outside its test period stay in the file.

A sequence model (`models._history`) reads series, not lagged columns: it has one row per series it reads (`history_rows`),
late or not, with the newest step known at the issue time and the series' vintage class in two more columns, `newest` and
`vintage`, which a frame has only when such a row is in it. Only late rows and rows without a rule are warned about.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

from ..information_set import (
    NEVER,
    ROLE_PREDICTOR,
    ROLE_PUBLISHED,
    ROLE_TARGET,
    ROLE_VALUE,
    Violation,
    _fmt_minute,
    _ranges,
)

LATE = "late"
NO_RULE = "no rule"
# A column read from another pipeline's forecasts (`ForecastSource`) whose upstream used inputs after its own issue time, or ran
# at level "none" (`models._chained`): its forecasts are no honest input, whatever this run's rule says of them.
LATE_UPSTREAM = "late upstream"
# How a manifest's listing compared times. Listings without it were written on the wall clock, which next to a clock change
# lists other periods than a fresh run would: such a listing is replaced on resume (`manifest._keep_listing`) and read with
# a warning (`read_listing`).
LISTING_TIMES = "instants"
# A sequence model's series (`models._history`), by role: the target's context, a past covariate, a known-ahead one; and the
# status of a series read in time, which a listing of a model of predictors never has.
ROLE_HISTORY_TARGET = "target"
ROLE_PAST = "past"
ROLE_KNOWN_AHEAD = "known_ahead"
ON_TIME = "ok"

# The columns of `EvaluationReport.availability()`.
COLUMNS = ["model", "predictor", "role", "status", "list", "periods", "horizons", "rule", "known", "issued", "note"]
# What a sequence model's rows add (`history_rows`), shown only when a listing has such a row.
HISTORY_COLUMNS = ["newest", "vintage"]
# The range of a series' newest known steps under its `newest` label, wall-clock minutes after the issue day's midnight: kept in
# a manifest's rows, so a merged listing spans both runs' days.
_NEWEST_RANGE = ("newest_from", "newest_to")

_ROLE_ORDER = {
    ROLE_PREDICTOR: 0,
    ROLE_VALUE: 1,
    ROLE_PUBLISHED: 2,
    ROLE_TARGET: 3,
    ROLE_HISTORY_TARGET: 4,
    ROLE_PAST: 5,
    ROLE_KNOWN_AHEAD: 6,
}
_SHOWN_PER_MODEL = 5  # how many rows of a model a warning spells out

_Key = Tuple[str, str, str, str]  # (predictor, role, status, list)
_Cell = Tuple[int, int]  # (horizon, period)


def listing_rows(uses: Iterable[Violation], list_hash: str = "") -> List[Dict[str, Any]]:
    """One row per predictor and distinct set of periods, with the latest known time of its uses."""
    cells: Dict[_Key, Dict[_Cell, Dict[str, Any]]] = {}
    for use in uses:
        status = NO_RULE if use.column.rule is None else LATE
        details = _details(use) if status == LATE else _NOTHING_KNOWN
        _keep_latest(cells, (use.predictor, use.role, status, list_hash), (use.horizon, use.period), details)
    return _rows(cells)


def late_upstream_rows(uses: Iterable[Tuple[str, int, int, str, str]]) -> List[Dict[str, Any]]:
    """The rows of chained columns whose upstream is not vouched for (`models._chained`), from (predictor, period, horizon, rule,
    note) of each use: one row per predictor and distinct set of periods, status `late upstream`, the upstream's late inputs or
    unchecked models in `note`."""
    cells: Dict[_Key, Dict[_Cell, Dict[str, Any]]] = {}
    for predictor, period, horizon, rule, note in uses:
        details = dict(_NOTHING_KNOWN, rule=rule, note=note)
        _keep_latest(cells, (predictor, ROLE_PREDICTOR, LATE_UPSTREAM, ""), (horizon, period), details)
    return _rows(cells)


def history_rows(batches: Iterable[Any], list_hash: str = "") -> List[Dict[str, Any]]:
    """The rows of a sequence model's series, from its `models._history.HistoryBatch`es over a run's issue days (one per horizon,
    or several of a horizon, such as a validation and a test window's): one row per series and status with the horizons it has
    there, not one per lagged step. A series is "late" at a horizon where it is late on some issue day of any batch (a
    known-ahead series not known through the end of the target day, a past one not known as far as the target's context and
    not NaN after its newest step), with `known` and `issued` of the issue day it is latest on, as a late predictor's; else
    "ok". `newest` is its newest step known at the issue time with every one before it (the target's: its cut), relative to the
    issue day D (`D 23:00`, or the range over the issue days and horizons of the row), and `vintage` the class of its rule.
    Every period of the day reads each series."""
    found: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]] = {}  # (series, role) -> horizon -> what its batches say
    infos: Dict[Tuple[str, str], Any] = {}
    periods = 0
    for batch in batches:
        periods = max(periods, batch.periods_per_day)
        items = list(batch.items)
        for info in batch.series if items else ():
            infos[(info.name, info.role)] = info
            held = found.setdefault((info.name, info.role), {}).setdefault(batch.horizon, {"late": None, "newest": []})
            held["newest"].extend(item.newest_minutes[info.name] for item in items)
            for item in items:
                lateness = item.late.get(info.name)
                if lateness is not None and (held["late"] is None or _late_by(lateness) > _late_by(held["late"])):
                    held["late"] = lateness
    rows = []
    for (name, role), by_horizon in found.items():
        info = infos[(name, role)]
        for status in (LATE, ON_TIME):
            horizons = sorted(h for h, held in by_horizon.items() if (held["late"] is not None) == (status == LATE))
            if not horizons:
                continue
            details = dict(_NOTHING_KNOWN, rule=info.rule, vintage=info.vintage)
            if status == LATE:
                lateness = max((by_horizon[h]["late"] for h in horizons), key=_late_by)
                minutes = _late_by(lateness)
                details.update(
                    known=lateness.known_label,
                    issued=lateness.issue_label,
                    minutes_late=minutes if math.isfinite(minutes) else None,
                )
            newest = [minute for h in horizons for minute in by_horizon[h]["newest"]]
            details.update(dict(zip(_NEWEST_RANGE, (min(newest), max(newest)))), newest=newest_label(newest))
            rows.append(
                {
                    "predictor": name,
                    "role": role,
                    "status": status,
                    "list": list_hash,
                    "periods": _ranges(list(range(periods))),
                    "horizons": _ranges(horizons),
                    **details,
                }
            )
    return sort_rows(rows)


def _late_by(lateness: Any) -> float:
    return float(lateness.known - lateness.issue)


def is_flagged(row: Mapping[str, Any]) -> bool:
    """Whether a listing's row says something is wrong (late, or without a rule): every row but a sequence model's series read
    in time (status "ok", `history_rows`). What `warn_about` warns of and the terminal exporter lists as not known in time."""
    return row["status"] != ON_TIME


def newest_label(minutes: Iterable[float]) -> str:
    """Newest known steps, wall-clock minutes after the issue day's midnight (D+1 01:00 is 1500), as a label, `D 23:00`, or
    their range over issue days, `D-1 23:00 … D 23:00`."""
    minutes = list(minutes)
    lowest, highest = min(minutes), max(minutes)
    first = _fmt_minute(lowest)
    return first if lowest == highest else f"{first} … {_fmt_minute(highest)}"


def merge_listings(earlier: Iterable[Mapping[str, Any]], later: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The uses of both listings, in the same shape, whichever lists a use first."""
    cells: Dict[_Key, Dict[_Cell, Dict[str, Any]]] = {}
    for row in [*earlier, *later]:
        key = (row["predictor"], row["role"], row["status"], row["list"])
        details = {name: row[name] for name in _NOTHING_KNOWN}
        # A sequence model's row: its label, class and the range of its newest steps.
        details.update({name: row[name] for name in (*HISTORY_COLUMNS, *_NEWEST_RANGE) if name in row})
        for horizon in _parse_ranges(row["horizons"]):
            for period in _parse_ranges(row["periods"]):
                _keep_latest(cells, key, (horizon, period), details)
    # A series read in time by one run and late in another, at a horizon, is late there: its "ok" cells go into the late row's.
    for (predictor, role, status, list_hash), held in list(cells.items()):
        late = cells.get((predictor, role, LATE, list_hash)) if status == ON_TIME else None
        for cell in [cell for cell in held if late is not None and cell in late]:
            late[cell] = _with_newest(late[cell], held.pop(cell))
        if not held:
            del cells[(predictor, role, status, list_hash)]
    return _rows(cells)


_NOTHING_KNOWN: Dict[str, Any] = {"rule": "", "known": "", "issued": "", "minutes_late": None, "note": ""}


def _details(use: Violation) -> Dict[str, Any]:
    late = float(use.available - use.issue)
    return {
        "rule": use.column.rule.describe(),
        "known": use.available_label or _fmt_minute(use.available),
        "issued": use.issue_label or _fmt_minute(use.issue),
        # A value no rule holds for is never known: null, as JSON has no infinity, and `known` says never.
        "minutes_late": late if math.isfinite(late) else None,
        "note": use.hint,
    }


def _lateness(details: Mapping[str, Any]) -> float:
    """How late a use is, for keeping the latest: a never known one is the latest of all, one without a rule the earliest."""
    late = details["minutes_late"]
    if late is None:
        return math.inf if str(details.get("known", "")) == NEVER else -math.inf
    return late


def _keep_latest(cells, key: _Key, cell: _Cell, details: Dict[str, Any]) -> None:
    held = cells.setdefault(key, {})
    current = held.get(cell)
    if current is None:
        held[cell] = details
    elif _lateness(details) > _lateness(current):
        held[cell] = _with_newest(details, current)
    elif _NEWEST_RANGE[0] in details:
        held[cell] = _with_newest(current, details)


def _with_newest(details: Dict[str, Any], other: Mapping[str, Any]) -> Dict[str, Any]:
    """`details` with the range of newest steps of both, and its label, where both have one (a sequence model's row)."""
    if _NEWEST_RANGE[0] not in details or _NEWEST_RANGE[0] not in other:
        return details
    lowest = min(details[_NEWEST_RANGE[0]], other[_NEWEST_RANGE[0]])
    highest = max(details[_NEWEST_RANGE[1]], other[_NEWEST_RANGE[1]])
    return {**details, _NEWEST_RANGE[0]: lowest, _NEWEST_RANGE[1]: highest, "newest": newest_label([lowest, highest])}


def _rows(cells: Dict[_Key, Dict[_Cell, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for (predictor, role, status, list_hash), held in cells.items():
        by_horizon: Dict[int, List[int]] = {}
        for horizon, period in sorted(held):
            by_horizon.setdefault(horizon, []).append(period)
        same_periods: Dict[Tuple[int, ...], List[int]] = {}
        for horizon, periods in by_horizon.items():
            same_periods.setdefault(tuple(periods), []).append(horizon)
        for periods, horizons in same_periods.items():
            uses = [held[(horizon, period)] for horizon in horizons for period in periods]
            latest = max(uses, key=_lateness)
            for use in uses:  # a sequence model's row: the range of every use's newest steps
                latest = _with_newest(latest, use)
            rows.append(
                {
                    "predictor": predictor,
                    "role": role,
                    "status": status,
                    "list": list_hash,
                    "periods": _ranges(list(periods)),
                    "horizons": _ranges(horizons),
                    **latest,
                }
            )
    return sort_rows(rows)


def sort_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows in the order they are listed: the same however they were found."""

    def key(row):
        horizons, periods = _parse_ranges(row["horizons"]), _parse_ranges(row["periods"])
        return _ROLE_ORDER.get(row["role"], len(_ROLE_ORDER)), row["predictor"], horizons, periods, row["list"]

    return sorted(rows, key=key)


def _parse_ranges(text: str) -> List[int]:
    """The numbers `_ranges` wrote as `"0-3, 7"`."""
    numbers: List[int] = []
    for part in str(text).split(", "):
        first, _, last = part.partition("-")
        numbers.extend(range(int(first), int(last or first) + 1))
    return numbers


def listing_frame(listings: Mapping[str, Optional[Iterable[Mapping[str, Any]]]]) -> pd.DataFrame:
    """The rows of every model as `EvaluationReport.availability()` shows them; no rows gives the same columns. With a sequence
    model's row (`history_rows`) the frame also has `newest` and `vintage`, empty on the other models' rows."""
    rows = [{"model": model, **row} for model, listing in listings.items() for row in listing or []]
    history = any(name in row for row in rows for name in HISTORY_COLUMNS)
    columns = COLUMNS + HISTORY_COLUMNS if history else COLUMNS
    return pd.DataFrame(
        [{name: "" for name in HISTORY_COLUMNS} | row for row in rows] if history else rows, columns=columns
    )


def warn_about(listings: Mapping[str, Optional[Iterable[Mapping[str, Any]]]], stacklevel: int = 2) -> None:
    """One warning for all models, if any of them has something late or without a rule listed (a sequence model's series read
    in time is listed too, and isn't warned about)."""
    flagged = {model: [row for row in listing or [] if is_flagged(row)] for model, listing in listings.items()}
    by_model = {model: rows for model, rows in flagged.items() if rows}
    if not by_model:
        return
    lines = [
        f"Availability: {sum(len(rows) for rows in by_model.values())} input(s) of {len(by_model)} model(s) are not known "
        "to be out at the issue time. The forecasts use them anyway; report.availability() lists them all."
    ]
    for model, rows in by_model.items():
        shown = "; ".join(_describe(row) for row in rows[:_SHOWN_PER_MODEL])
        more = f"; and {len(rows) - _SHOWN_PER_MODEL} more" if len(rows) > _SHOWN_PER_MODEL else ""
        lines.append(f"  {model}: {shown}{more}")
    lines.append('availability="raise" refuses them before any fit; availability="none" turns the listing off.')
    warnings.warn("\n".join(lines), UserWarning, stacklevel=stacklevel)


def _describe(row: Mapping[str, Any]) -> str:
    where = f"periods {row['periods']}, horizons {row['horizons']}"
    if row["status"] == NO_RULE:
        return f"{row['predictor']} has no rule ({where})"
    if row["status"] == LATE_UPSTREAM:
        return f"{row['predictor']} is a forecast whose upstream is late ({where}; {row['note']})"
    return f"{row['predictor']} is late ({where}; known {row['known']}, issued {row['issued']})"


def read_listing(path: Path, window: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """The rows a manifest holds, or None if it has none (no manifest, level "none", or results that predate them).

    `window` gives the rows of one window of a run with validation.
    """
    path = Path(path)
    if not path.exists():
        return None
    availability = json.loads(path.read_text()).get("availability")
    if availability is None:
        return None
    if availability.get("times") != LISTING_TIMES:
        warnings.warn(
            f"{path.name}: this availability listing was written when the check compared wall-clock times, so next to a "
            "clock change it may list other periods than a fresh run. A run at availability='warn' lists it anew: into "
            "this directory where its manifest lets it resume, refitting nothing, else into a new one.",
            UserWarning,
            stacklevel=3,
        )
    if window is not None:
        return list(availability.get("windows", {}).get(window, []))
    return list(availability.get("listing", []))


__all__ = [
    "COLUMNS",
    "HISTORY_COLUMNS",
    "LATE",
    "LATE_UPSTREAM",
    "LISTING_TIMES",
    "NO_RULE",
    "ON_TIME",
    "ROLE_HISTORY_TARGET",
    "ROLE_KNOWN_AHEAD",
    "ROLE_PAST",
    "history_rows",
    "is_flagged",
    "late_upstream_rows",
    "listing_frame",
    "listing_rows",
    "merge_listings",
    "newest_label",
    "read_listing",
    "warn_about",
]
