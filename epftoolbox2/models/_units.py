"""Unit kinds: what one fit of a model on the refit calendar holds (step 6f).

A model's `_unit_kind` says what a unit (`_calendar.FitUnit`) fits:

- `"period"`, every core model's: the rows of one period of the day, one target column (`_worker._fit_unit`).
- `"day"`: rows are local dates, and every predictor is a day-level column, one value per local date (a daily profile such as
  `price_h05_d-1`, a calendar flag such as `is_monday_d+1`); the targets are the P periods of the target day, `{target}_d+{h}` of
  each period's row, NaN where the slot has none (the hour a spring switch skips, the repeated autumn hour, a missing value) and
  masked. One fit, P outputs: `_fit_days(x, y, y_mask)` and `_predict_days(fitted, rows)` (`_worker._fit_day_unit`).
- `"pooled"`: rows are the (date, period) rows of every period, one output: each predictor is read on its period's own row (so
  `price_d-1` is that period's price and `price_h05_d-1` the date's 05:00 price), and `period` (0 to P-1) is a feature only
  when a list names it. `_fit_model` and `_predict_block` over the stacked rows (`_worker._fit_pooled_unit`).

A day or pooled unit is `FitUnit(None, horizon, anchor, list id, member)`: every period at once. Both need `refit` (the fixed
calendar of step 3: rows by date, `training_span(anchor, W, h, newest)`), and each predicts every issue day of its block that
the frame has, in one call of the block's nominal length, padded with copies of the first, so a forecast is the same alone, in
its block, after a resume and whatever the run's `test_start` or the data's end. Records are those of every planned (period,
horizon, run date), keyed as a period unit's. A list may change with the horizon and the issue day, never with the period of
one issue day: one fit reads one list.
"""

import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..information_set import parse_predictor
from ..scalers.target import MultiTargetAsinh, RawTarget
from . import _contract
from ._calendar import FREQ_STEPS, periods_per_day

PERIOD, DAY, POOLED = "period", "day", "pooled"
UNIT_KINDS = (PERIOD, DAY, POOLED)

# How a unit transforms its targets: each period's own Invariant (the research's per-hour one, `ztree.py:104-105`), one over
# the rows of every period (a pooled unit's only), or none, the raw target (`scalers.target.RawTarget`, the identity there and
# back). A day unit takes each period's own or none.
PER_PERIOD, POOLED_TARGET, NO_TRANSFORM = "per_period", "pooled", "none"
TARGET_TRANSFORMS = (PER_PERIOD, POOLED_TARGET, NO_TRANSFORM)
DAY_TARGET_TRANSFORMS = (PER_PERIOD, NO_TRANSFORM)

# The virtual column of a pooled unit: the period of each row, a feature only where a list names it.
PERIOD_FEATURE = "period"

# What one class (not BaseModel) must define for a unit of each kind: what it fits and forecasts with.
UNIT_METHODS = {
    PERIOD: ("_fit_predict", "_fit_model", "_predict_fitted"),
    DAY: ("_fit_days", "_predict_days"),
    POOLED: ("_fit_model", "_predict_block"),
}


def unit_kind(model: Any) -> str:
    """The model's unit kind, `"period"` for a model that says none; anything else raises."""
    kind = getattr(model, "_unit_kind", PERIOD)
    if kind not in UNIT_KINDS:
        raise ValueError(f"{type(model).__name__}._unit_kind must be one of {list(UNIT_KINDS)}, got {kind!r}")
    return kind


def check_target_transform(value: Any) -> str:
    """`value` if it is one of `TARGET_TRANSFORMS`, else a ValueError."""
    if not isinstance(value, str) or value not in TARGET_TRANSFORMS:
        raise ValueError(
            f"target_transform must be one of {list(TARGET_TRANSFORMS)}, got {value!r}: 'per_period' fits each period's "
            "targets an Invariant of its own (asinh of median and MAD), 'pooled' one over the rows of every period, 'none' "
            "the raw target"
        )
    return value


def check_periods_target_transform(model: Any, value: Any, periods: str) -> str:
    """`value` for a tree class of `periods` ("pooled" or "each"; `HistGBModel`, the boosted trees): any of
    `TARGET_TRANSFORMS` for a pooled fit, the default "per_period" alone for `periods="each"`, whose period units transform
    their targets with the model's `scaler`; anything else a ValueError, never a setting silently left unread."""
    check_target_transform(value)
    if periods == "each" and value != PER_PERIOD:
        raise ValueError(
            f"{type(model).__name__}: target_transform={value!r} is how a pooled fit (periods='pooled') transforms its "
            "targets; with periods='each' each period's target goes through scaler= (the Invariant by default), so "
            "target_transform takes its default 'per_period' there"
        )
    return value


def target_transform_of(model: Any) -> str:
    """How a day or pooled unit of `model` transforms its targets: its `target_transform` (`"per_period"` if it has none),
    which for a day unit is `"per_period"` or `"none"` (a ValueError for `"pooled"`, a pooled unit's alone)."""
    value = check_target_transform(getattr(model, "target_transform", PER_PERIOD))
    if unit_kind(model) == DAY and value not in DAY_TARGET_TRANSFORMS:
        raise ValueError(
            f"{type(model).__name__}: a day unit fits each period of the day as an output of its own, so its "
            f"target_transform is one of {list(DAY_TARGET_TRANSFORMS)}, got {value!r} (one transform over the rows of "
            "every period is a pooled unit's)"
        )
    return value


def target_of(transform: str) -> Any:
    """A fresh target transform of a unit's `transform` (`units_config`'s): `MultiTargetAsinh`, each period's Invariant or one
    over every row as the unit fits it, or `RawTarget`, the identity, for `"none"`."""
    return RawTarget() if transform == NO_TRANSFORM else MultiTargetAsinh()


def units_config(model: Any, periods: int) -> Optional[Dict[str, Any]]:
    """What a worker reads of a day or pooled model (`config["units"]`): the kind, the periods of a day and the target
    transform; None for a period model, whose config doesn't change."""
    kind = unit_kind(model)
    if kind == PERIOD:
        return None
    return {"kind": kind, "periods": periods, "target_transform": target_transform_of(model)}


def kind_of(cfg: Dict) -> str:
    """The unit kind of a worker's config: `"period"` without `config["units"]`."""
    units = cfg.get("units")
    return PERIOD if units is None else units["kind"]


def manifest_fields(model: Any) -> Dict[str, Any]:
    """`unit_kind` and `target_transform` for a manifest of a day or pooled model (compared on resume), nothing for a period
    model, whose manifest doesn't change."""
    kind = unit_kind(model)
    if kind == PERIOD:
        return {}
    return {"unit_kind": kind, "target_transform": target_transform_of(model)}


def check_unit_settings(model: Any, data: pd.DataFrame, lists: Iterable[Sequence[str]] = ()) -> None:
    """A ValueError for what a day or pooled unit of `model` can't run with in this step: a target transform its kind doesn't
    take (`target_transform_of`), no `refit` (a unit is a fit on the refit calendar; `refit="1D"` fits every issue day), and
    for a pooled unit a frame column named `period` where a list names the virtual column (which of the two would it be?).
    Nothing for a period model."""
    kind = unit_kind(model)
    if kind == PERIOD:
        return
    target_transform_of(model)
    if model.refit is None:
        raise ValueError(
            f"{model.name}: a {kind} unit is a fit on the refit calendar, so {type(model).__name__} needs refit= ('7D' fits "
            "weekly, '1D' every issue day); refit=None fits one period's forecast at a time, which a "
            f"{kind} unit isn't"
        )
    if kind == POOLED and PERIOD_FEATURE in data.columns and any(PERIOD_FEATURE in names for names in lists):
        raise ValueError(
            f"{model.name}: the frame has a column named {PERIOD_FEATURE!r}, which a pooled unit's lists name as the period of "
            "each row (0 to P-1): rename the column"
        )


def check_plan_for_units(model: Any, plan: Any) -> None:
    """A ValueError for a predictor function's plan a day or pooled unit can't use in this step: a `train_on="matching"`
    list (its days are a period's), and the virtual `period` in a function's list (fixed lists only)."""
    kind = unit_kind(model)
    if kind == PERIOD or plan is None:
        return
    if plan.uses_matching:
        raise ValueError(
            f"{model.name}: train_on='matching' filters the rows of one period by the days with its list, and a {kind} unit "
            "fits every period at once: return the list with train_on='all'"
        )
    if kind == POOLED and any(PERIOD_FEATURE in names for names in plan.lists):
        raise ValueError(
            f"{model.name}: the virtual {PERIOD_FEATURE!r} feature is for fixed lists in this step: give the predictors as a "
            "list, or leave it out of the function's lists"
        )


def check_lists_by_period(
    model_name: str,
    horizons: Sequence[int],
    periods: int,
    fixed: Optional[Dict[Tuple[int, int], List[str]]] = None,
    plan: Any = None,
) -> None:
    """A ValueError if a list differs between two periods of one horizon (fixed lists, a `(horizon, hour)` function) or of
    one issue day (a predictor function of `ctx`, on its planned keys): a day or pooled unit fits every period of its days in
    one fit, from one list. A list may change with the horizon and the issue day."""
    for horizon in horizons:
        if fixed is not None:
            first = list(fixed[(0, horizon)])
            for period in range(1, periods):
                other = list(fixed[(period, horizon)])
                if other != first:
                    raise ValueError(
                        f"{model_name}: a day or pooled unit fits every period of a day from one list, but at horizon "
                        f"{horizon} period 0 has {_diff(first, other)} and period {period} {_diff(other, first)}: a "
                        "predictor that changes with the period (a `(horizon, hour)` function) can't feed one fit"
                    )
            continue
        if plan is None:
            continue
        planned = plan.planned()
        ids = np.stack([plan.assignment[(period, horizon)] for period in range(periods)])
        on = np.stack([planned[(period, horizon)] for period in range(periods)])
        highest = np.where(on, ids, -1).max(axis=0)
        lowest = np.where(on, ids, np.iinfo(ids.dtype).max).min(axis=0)
        differs = on.any(axis=0) & (highest != lowest)
        if differs.any():
            position = int(np.flatnonzero(differs)[0])
            column = ids[:, position]
            first_period = int(np.flatnonzero(on[:, position])[0])
            other_period = int(np.flatnonzero(on[:, position] & (column != column[first_period]))[0])
            first_list = list(plan.lists[int(column[first_period])])
            other_list = list(plan.lists[int(column[other_period])])
            raise ValueError(
                f"{model_name}: a day or pooled unit fits every period of an issue day from one list, but the predictor "
                f"function returned two lists on {plan.issue_days[position]} at horizon {horizon}: period {first_period} "
                f"{_diff(first_list, other_list)} and period {other_period} {_diff(other_list, first_list)}. A list may change "
                "with the horizon and the issue day, not with the period (ctx.period, ctx.hour)."
            )


def _diff(names: List[str], other: List[str]) -> str:
    """What `names` has that `other` lacks, or its order, for an error message."""
    extra = [name for name in names if name not in other]
    if extra:
        return f"{extra[:3]}{' ...' if len(extra) > 3 else ''}"
    return f"its names in the order {names[:4]}{' ...' if len(names) > 4 else ''}"


def without_feature(lists: Dict[Any, List[str]]) -> Dict[Any, List[str]]:
    """These lists without the virtual `period`: what the frame's and the information set's checks read of a pooled unit's
    lists, the period of a row being known by construction."""
    return {key: [name for name in names if name != PERIOD_FEATURE] for key, names in lists.items()}


def profile_names(name: str, freq: str) -> str:
    """The daily profile names a day unit reads in place of the per-period column `name`, as a fix for an error message:
    `price_d-1` -> `price_h00_d-1 … price_h23_d-1` (`DailyProfileTransformer(columns=["price"])`, then the same shift), or at
    15 minutes `price_s00_d-1 … price_s95_d-1`."""
    base, _ = parse_predictor(name)[-1]
    suffix = name[len(base) :]
    periods = periods_per_day(freq)
    letter = "h" if FREQ_STEPS[freq] == pd.Timedelta(hours=1) else "s"
    width = max(2, len(str(periods - 1)))
    first, last = f"{base}_{letter}{0:0{width}d}{suffix}", f"{base}_{letter}{periods - 1:0{width}d}{suffix}"
    return f"{first} … {last} (DailyProfileTransformer(columns=[{base!r}]), then the shift)"


def check_day_level(
    data: pd.DataFrame,
    names: Sequence[str],
    freq: str,
    model_name: str,
    span: Optional[Tuple[datetime.date, datetime.date]] = None,
) -> None:
    """A ValueError if a column of `names` holds two different finite values on one local date of `span` (first, last; every
    date of the frame if None): a day unit reads one value per date, and a column that changes with the period of the day
    (`price_d-1`) would silently give it one period's. A NaN row holds no value for its date: a date with a value on some rows
    and NaN on others is fine. The error names the column, a date and its profile names, the fix.

    A run checks the dates its fits train on (`BaseModel._check_day_level`): the probes of `verify` perturb what is unknown, or
    fresh, at the issue time cell by cell, which a later date's or the issue date's rows hold, while the dates a fit trains on
    stay as they are. So a plain run reads the last row's value (`day_arrays`) of a column that changes with the period only on
    dates no fit trains on, such as an issue date. A check with no test period (`verify` checks each model so first) reads the
    whole frame."""
    if periods_per_day(freq) == 1 or not len(names):
        return
    dates = _contract.wall_clock(data.index).normalize()
    rows = np.ones(len(data), dtype=bool)
    if span is not None:
        first, last = (pd.Timestamp(day) for day in span)
        rows = np.asarray((dates >= first) & (dates <= last))
    if not rows.any():
        return
    columns = list(dict.fromkeys(names))
    frame = data.loc[rows, columns].astype(np.float64)
    frame = frame.where(np.isfinite(frame.to_numpy()))
    grouped = frame.groupby(dates[rows])
    lowest, highest = grouped.min(), grouped.max()
    differs = (lowest != highest) & lowest.notna()
    if not differs.to_numpy().any():
        return
    column = next(name for name in columns if differs[name].any())
    date = differs.index[differs[column].to_numpy()][0]
    count = int(differs[column].sum())
    raise ValueError(
        f"{model_name}: a day unit reads one value of each predictor per local date, but {column!r} holds values from "
        f"{float(lowest.at[date, column])!r} to {float(highest.at[date, column])!r} on {date.date()} ({count} date"
        f"{'s' if count > 1 else ''} like it): it changes with the period of the day. Give its daily profile instead: "
        f"{profile_names(column, freq)}"
    )


def day_arrays(
    hour_arrays: Dict[int, Dict],
    periods: int,
    names: Sequence[str],
    first: int,
    last: int,
    horizon: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """(has a row, inputs, targets) of the local dates with day ids `first` to `last`, both included, from the period arrays.

    A date has a row when any period has one on it. Its input of a column is that column's value on the date's last row, in
    period order (the second of a repeated hour's two), with a finite value: the date's one value where the column is
    day-level, which `check_day_level` holds the dates a fit trains on to. Its target of period p, with `horizon`, is
    `{target}_d+{horizon}` of p's issue row on the date (the last row, as a forecast is issued from it), NaN where p has no row
    on the date (the hour a spring switch skips), so a slot is never shifted to another period."""
    n = max(last - first + 1, 0)
    has_row = np.zeros(n, dtype=bool)
    x = np.full((n, len(names)), np.nan)
    y = None if horizon is None else np.full((n, periods), np.nan)
    for period in range(periods):
        arrays = hour_arrays[period]
        days = arrays["days"]
        lo, hi = int(np.searchsorted(days, first, side="left")), int(np.searchsorted(days, last, side="right"))
        if lo == hi:
            continue
        at = days[lo:hi] - first
        has_row[at] = True
        values = arrays["x"][lo:hi][:, [arrays["col_index"][name] for name in names]]
        # A period has at most two rows on a date (a repeated autumn hour), one after the other: the earlier first, so that the
        # later one's value stays where both have one.
        later = np.zeros(len(at), dtype=bool)
        later[1:] = at[1:] == at[:-1]
        for part in (~later, later):
            where, found = at[part], values[part]
            x[where] = np.where(np.isfinite(found), found, x[where])
        if y is not None:
            issue = np.ones(len(at), dtype=bool)
            issue[:-1] = at[:-1] != at[1:]
            y[at[issue], period] = arrays["y"][lo:hi][issue, horizon - 1]
    return has_row, x, y


def pooled_values(arrays: Dict, names: Sequence[str], rows: Any, period: int) -> np.ndarray:
    """The values of `names` on these rows of one period's arrays, the virtual `period` its id."""
    index = np.arange(len(arrays["days"]))[rows]
    out = np.empty((len(index), len(names)))
    for i, name in enumerate(names):
        out[:, i] = period if name == PERIOD_FEATURE else arrays["x"][index, arrays["col_index"][name]]
    return out
