"""The listing `availability="warn"` makes: every input a run uses though its rule puts it after the issue time, and every
input without a rule, per model, with the periods and horizons concerned.

One row says "these periods at these horizons", and both are exact: a predictor used at other periods on other horizons
gets one row per distinct set of periods, so the rows of a key add up to exactly its uses. A manifest stores the rows, so a
report built later (`ModelPipeline.report` after array jobs) shows what the run listed. `merge_listings` keeps them when a
narrower run into the same directory rewrites the manifest: the records outside its test period stay in the file.
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
# How a manifest's listing compared times. Listings without it were written on the wall clock, which next to a clock change
# lists other periods than a fresh run would: such a listing is replaced on resume (`manifest._keep_listing`) and read with
# a warning (`read_listing`).
LISTING_TIMES = "instants"

# The columns of `EvaluationReport.availability()`.
COLUMNS = ["model", "predictor", "role", "status", "list", "periods", "horizons", "rule", "known", "issued", "note"]

_ROLE_ORDER = {ROLE_PREDICTOR: 0, ROLE_VALUE: 1, ROLE_PUBLISHED: 2, ROLE_TARGET: 3}
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


def merge_listings(earlier: Iterable[Mapping[str, Any]], later: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The uses of both listings, in the same shape, whichever lists a use first."""
    cells: Dict[_Key, Dict[_Cell, Dict[str, Any]]] = {}
    for row in [*earlier, *later]:
        key = (row["predictor"], row["role"], row["status"], row["list"])
        details = {name: row[name] for name in _NOTHING_KNOWN}
        for horizon in _parse_ranges(row["horizons"]):
            for period in _parse_ranges(row["periods"]):
                _keep_latest(cells, key, (horizon, period), details)
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
    if current is None or _lateness(details) > _lateness(current):
        held[cell] = details


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
    """The rows of every model as `EvaluationReport.availability()` shows them; no rows gives the same columns."""
    rows = [{"model": model, **row} for model, listing in listings.items() for row in listing or []]
    return pd.DataFrame(rows, columns=COLUMNS)


def warn_about(listings: Mapping[str, Optional[Iterable[Mapping[str, Any]]]], stacklevel: int = 2) -> None:
    """One warning for all models, if any of them has something listed."""
    by_model = {model: list(listing) for model, listing in listings.items() if listing}
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
    "LATE",
    "LISTING_TIMES",
    "NO_RULE",
    "listing_frame",
    "listing_rows",
    "merge_listings",
    "read_listing",
    "warn_about",
]
