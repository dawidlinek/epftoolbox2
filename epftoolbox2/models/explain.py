"""`ModelPipeline.explain`: what each predictor of one forecast holds, when it is known, and what a function returns.

Column suffixes are relative to the row's local date D, and papers write them relative to the target day T = D + horizon.
The table shows them that way: at horizon 2, `price_d-1` is `T-3` and `x_d+2` is `T`. What a name holds comes from
`information_set` (`parse_predictor`, `ResolvedColumn`, `_delivered_span`), the same reading the check and the probe use,
and when it is known from the same clock.

A sequence model (`models._history`) reads series instead: `series` has one row per series, with the newest step it reads,
from the history accessor's own cut, not one per lagged step.

A model with a per-row target (`models._row_target`) has a `row_target` table: its `normalise_by` and `zero_when` columns at the
horizon shown, judged as predictors are (they are inputs the model reads, not regressors), with their role.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..information_set import (
    _PROFILE_SLOT,
    _STEP_MINUTES,
    InformationSet,
    TrainingRows,
    _RowsKnown,
    _built_from_name,
    _delivered_span,
    _fmt_minute,
    _IssueClock,
    _profile_slot,
    _ranges,
    frame_first_day,
    over_time,
    parse_predictor,
)
from .._date_utils import fmt_minutes
from ..results.availability import newest_label
from . import _contract, _row_target
from ._calendar import refit_anchor
from ._history import HistorySpec, history_batch, history_spec_of
from .context import TRAIN_ON_MATCHING, PredictorPlan, function_name

COLUMNS = ["predictor", "target-relative", "rule", "known", "status"]
# The table of a per-row target's columns (`row_target_table`): a predictor table's, after the column's role.
ROW_TARGET_COLUMNS = ["role", *COLUMNS]
# The table of a sequence model's series (`history_table`).
SERIES_COLUMNS = ["series", "role", "rule", "vintage", "steps", "newest", "status"]
_TRAINING_DAY_COLUMNS = ("smallest days", "median days", "largest days")
_DASH = "-"
# Names a run of slots or a list of added names spells out before it shortens them.
_NAMES_SHOWN = 3
# Reads names, nothing else: with no rules every suffix counts, while a rule on a shifted name would stop the reading there.
_NAMES = InformationSet(use_defaults=False)


def _target_label(minutes: float) -> str:
    """Minutes after midnight of the target day as `T-2 12:00`."""
    return _fmt_minute(minutes).replace("D", "T", 1)


def _span_label(start: float, end: float) -> str:
    first = _target_label(start)
    if start == end:
        return first
    last = _target_label(end)
    same_day = first.rsplit(" ", 1)[0] == last.rsplit(" ", 1)[0]
    return f"{first} … {last.rsplit(' ', 1)[1] if same_day else last}"


def target_relative(name: str, period: int, horizon: int, freq: str, clock: _IssueClock, used: np.ndarray) -> str:
    """When the value `name` holds on the row (D, `period`) was delivered, relative to T = D + `horizon`.

    Most days agree. An hour lag runs in elapsed time, so next to a DST switch it lands on another hour; the label is then
    the most common one and says how many days differ. `used` picks the issue days to read.
    """
    step = _STEP_MINUTES[freq]
    col = _NAMES._resolve_for(name, step)
    if col.built_from is None and _built_from_name(col.base) is not None:
        return "?"  # a `_sNN` slot on hourly data doesn't say how long a slot is
    days = clock.offset[used]
    minute = clock.row_minutes(np.array([period]), step)[0]  # a daily row's is the time it is stamped at
    start, end = _delivered_span(col, days, np.full(len(days), minute), step, clock.ctx)
    target = (days + horizon) * 1440.0
    counts = Counter(zip((start - target).tolist(), (end - target).tolist()))
    (first, last), common = counts.most_common(1)[0]
    label = _span_label(first, last)
    return label if len(counts) == 1 else f"{label} (else on {len(days) - common} of {len(days)} days)"


def _judge(
    name: str,
    period: int,
    data: pd.DataFrame,
    info: Optional[InformationSet],
    freq: str,
    clock: _IssueClock,
    used,
    training_rows: Optional[TrainingRows] = None,
    horizon: int = 1,
) -> Tuple[str, str, str]:
    """The (rule, known, status) of `name` at `period` over the issue days `used`. A column whose rule changes over time is
    judged on the rows its fit at `horizon` reads too (`training_rows`), as the check judges it."""
    if name not in data.columns:
        return _DASH, _DASH, "not in data"
    if info is None:
        return _DASH, _DASH, "no issue time"
    step = _STEP_MINUTES[freq]
    col = info._resolve_for(name, step)
    if col.rule is None:
        return _DASH, _DASH, "no rule"
    info._check_source_zone(col, data.index)
    used = used & ~clock.skipped(np.array([period]), step)[0]  # a period the clocks skip has no row to judge
    if not used.any():
        return col.rule.describe(), _DASH, "no row"
    available = clock.available(col, np.array([period]), step)[0]
    if training_rows is not None and over_time(col.rule):
        known = _RowsKnown(col, np.array([period]), clock, step)
        available = np.maximum(available, training_rows.latest(known, horizon)[0])
    available = available[used]
    excess = available - clock.issue[used]
    worst = int(excess.argmax())
    known = clock.label(available[worst], int(np.flatnonzero(used)[worst]))
    late = int((excess > 0).sum())
    status = "ok" if not late else "late" if late == len(excess) else f"late on {late} of {len(excess)} days"
    return col.rule.describe(), known, status


def predictor_table(
    names: List[str],
    data: pd.DataFrame,
    info: Optional[InformationSet],
    horizon: int,
    period: int,
    freq: str,
    clock: _IssueClock,
    used: Optional[np.ndarray] = None,
    training_rows: Optional[TrainingRows] = None,
) -> pd.DataFrame:
    """One row per predictor of the forecast of (`horizon`, `period`) over the issue days `used` (default: all). With
    `training_rows` a column whose rule changes over time is judged on the rows its fit reads too."""
    used = np.ones(len(clock.days), dtype=bool) if used is None else used
    rows = []
    for name in names:
        rule, known, status = _judge(name, period, data, info, freq, clock, used, training_rows, horizon)
        label = target_relative(name, period, horizon, freq, clock, used)
        rows.append(dict(zip(COLUMNS, (name, label, rule, known, status))))
    return pd.DataFrame(rows, columns=COLUMNS)


def row_target_table(
    model: Any,
    data: pd.DataFrame,
    info: Optional[InformationSet],
    horizon: int,
    period: int,
    freq: str,
    clock: _IssueClock,
    used: Optional[np.ndarray] = None,
    training_rows: Optional[TrainingRows] = None,
) -> Optional[pd.DataFrame]:
    """One row per column of the model's per-row target at `horizon`: its role (`normalise_by`, `zero_when`, or both), then what
    `predictor_table` says of it; None for a model without such a column."""
    names = _row_target.columns_at(model, horizon)
    if not names:
        return None
    table = predictor_table(names, data, info, horizon, period, freq, clock, used, training_rows)
    roles = []
    for name in names:
        roles.append(
            ", ".join(
                key
                for key in ("normalise_by", "zero_when")
                if getattr(model, key, None) is not None and _row_target.expand(getattr(model, key), horizon) == name
            )
        )
    table.insert(0, "role", roles)
    return table[ROW_TARGET_COLUMNS]


def row_target_note(model: Any) -> str:
    """What the model's per-row target does to a forecast, in a line; empty without one."""
    parts = []
    if getattr(model, "normalise_by", None) is not None:
        parts.append(
            f"the target divided by {getattr(model, 'normalise_by', None)} on each training row, the forecast multiplied by the issue row's"
        )
    if getattr(model, "bounds", None) is not None:
        low, high = getattr(model, "bounds", None)
        unit = " times the scale" if getattr(model, "normalise_by", None) is not None else ""
        parts.append(
            f"the forecast clipped to [{'-inf' if low is None else low}, {'inf' if high is None else high}]{unit}"
        )
    if getattr(model, "zero_when", None) is not None:
        parts.append(f"0 with no fit where {getattr(model, 'zero_when', None)} is 1 on the issue row")
    return "; ".join(parts)


def history_table(
    spec: HistorySpec,
    data: pd.DataFrame,
    info: Optional[InformationSet],
    days: pd.DatetimeIndex,
    horizon: int,
    freq: str,
) -> pd.DataFrame:
    """One row per series of a sequence model's forecasts at `horizon` on the issue days `days`, from the history accessor's cuts
    (`models._history.history_batch(arrays=False)`): its role, rule and vintage class, how many steps it reads (T, or T + S for
    a known-ahead series, S over the days), its newest step known at the issue time relative to D (the target's: its cut; the
    range over the days), and `ok`, `late` or `late on 3 of 10 days`. A series without a rule is `no rule`, as a predictor is,
    and with a target without one the others can't be cut (`not cut`); without an information set every row is `no issue
    time`."""

    def blank(name: str, status: str) -> tuple:
        return (name, spec.role_of(name), _DASH, _DASH, _DASH, _DASH, status)

    if info is None:
        return pd.DataFrame([blank(name, "no issue time") for name in spec.names], columns=SERIES_COLUMNS)
    step = _STEP_MINUTES[freq]
    ruleless = {name for name in spec.names if info.in_data_zone(data.index)._resolve_for(name, step).rule is None}
    if spec.target in ruleless:
        rows = [
            blank(name, "no rule" if name in ruleless else "not cut (the target has no rule)") for name in spec.names
        ]
        return pd.DataFrame(rows, columns=SERIES_COLUMNS)
    kept = HistorySpec(
        spec.context,
        tuple(name for name in spec.known_ahead if name not in ruleless),
        tuple(name for name in spec.past if name not in ruleless),
        spec.target,
        spec.accepts_missing,
        spec.newest_target,
    )
    batch = history_batch(kept, data, days, info, freq, horizon, arrays=False)
    late = batch.late()
    ahead = sorted({item.steps_ahead for item in batch.items})
    found = {series.name: series for series in batch.series}
    rows = []
    for name in spec.names:
        if name in ruleless:
            rows.append(blank(name, "no rule"))
            continue
        series = found[name]
        steps = str(batch.steps)
        if name in spec.known_ahead and ahead:
            steps = f"{batch.steps} + {ahead[0] if len(ahead) == 1 else f'{ahead[0]}…{ahead[-1]}'}"
        count = len(late.get(name, []))
        status = "late" if count == len(batch.items) else f"late on {count} of {len(batch.items)} days"
        status = "ok" if not count else status
        newest = newest_label(item.newest_minutes[name] for item in batch.items) if batch.items else _DASH
        rows.append((name, series.role, series.rule, series.vintage, steps, newest, status))
    return pd.DataFrame(rows, columns=SERIES_COLUMNS)


def _slot_of(name: str) -> Optional[Tuple[Tuple[str, str, str], int]]:
    """(what the name is without its slot, the slot's number), for a name with a slot."""
    base = parse_predictor(name)[-1][0]
    if _profile_slot(base) is None:
        return None
    match = _PROFILE_SLOT.match(base)
    return (match["column"], match["unit"], name[len(base) :]), int(match["n"])


def grouped_table(table: pd.DataFrame) -> pd.DataFrame:
    """`table` with each run of neighbouring slots that agree on everything else as one row."""
    groups: List[List[Dict[str, Any]]] = []
    previous = None
    for row in table.to_dict("records"):
        label = row["target-relative"]
        slot = None if "…" in label or "(" in label or label == "?" else _slot_of(row["predictor"])
        key = None if slot is None else (slot[0], row["rule"], row["status"], label.split(" ", 1)[0])
        if key is not None and previous == (key, slot[1] - 1):
            groups[-1].append(row)
        else:
            groups.append([row])
        previous = None if key is None else (key, slot[1])
    return pd.DataFrame([_group_row(group) for group in groups], columns=COLUMNS)


def _group_row(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    first, last = group[0], group[-1]
    if len(group) == 1:
        return first
    known = list(dict.fromkeys(row["known"] for row in group))
    return {
        "predictor": f"{first['predictor']} … {last['predictor']} ({len(group)})",
        "target-relative": f"{first['target-relative']} … {last['target-relative'].rsplit(' ', 1)[-1]}",
        "rule": first["rule"],
        "known": known[0] if len(known) == 1 else f"{known[0]} … {known[-1]}",
        "status": first["status"],
    }


def _short(names: List[str]) -> str:
    if len(names) <= _NAMES_SHOWN:
        return ", ".join(names)
    return f"{names[0]}, {names[1]} … {names[-1]} ({len(names)})"


def list_table(
    plan: PredictorPlan, horizons: Sequence[int], training_days: Optional[Dict[Tuple[int, int], np.ndarray]] = None
) -> pd.DataFrame:
    """One row per (horizon, list that horizon uses), and what each adds to or drops from that horizon's most used list.

    Compared within a horizon only: a `{horizon}` in a name makes the lists of two horizons differ anyway.

    A plan with `train_on="matching"` lists has three more columns, from `training_days` (`BaseModel._matching_day_counts`):
    the smallest, median and largest number of days the forecasts of such a list train on, and a dash for any other.
    """
    columns = [
        "horizon",
        "list",
        "hash",
        "forecasts",
        "issue days",
        "first issue day",
        "last issue day",
        "periods",
        "predictors",
        "train_on",
        "adds",
        "drops",
        *(_TRAINING_DAY_COLUMNS if training_days is not None else ()),
    ]
    rows = []
    for horizon in horizons:
        pairs = {period: ids for (period, hz), ids in plan.assignment.items() if hz == horizon}
        grid = np.stack([pairs[period] for period in sorted(pairs)])
        counts = {int(list_id): int((grid == list_id).sum()) for list_id in np.unique(grid)}
        # Ties go to the list returned first, so the same plan always compares against the same list.
        reference = plan.lists[min(counts, key=lambda list_id: (-counts[list_id], list_id))]
        for list_id in sorted(counts):
            names = plan.lists[list_id]
            days = np.flatnonzero((grid == list_id).any(axis=0))
            rows.append(
                {
                    "horizon": horizon,
                    "list": list_id + 1,
                    "hash": plan.hashes[list_id],
                    "forecasts": counts[list_id],
                    "issue days": len(days),
                    "first issue day": plan.issue_days[days[0]].isoformat(),
                    "last issue day": plan.issue_days[days[-1]].isoformat(),
                    "periods": _ranges(np.flatnonzero((grid == list_id).any(axis=1)).tolist()),
                    "predictors": len(names),
                    "train_on": plan.train_on[list_id],
                    "adds": _DASH if names == reference else _short([n for n in names if n not in reference]),
                    "drops": _DASH if names == reference else _short([n for n in reference if n not in names]),
                    **(_training_day_columns(plan, list_id, horizon, pairs, training_days) if training_days else {}),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _training_day_columns(
    plan: PredictorPlan,
    list_id: int,
    horizon: int,
    pairs: Dict[int, np.ndarray],
    training_days: Dict[Tuple[int, int], np.ndarray],
) -> Dict[str, Any]:
    counts = [
        training_days[(period, horizon)][ids == list_id]
        for period, ids in pairs.items()
        if (period, horizon) in training_days
    ]
    days = np.concatenate(counts) if counts else np.array([], dtype=np.int64)
    days = days[days >= 0]
    if plan.train_on[list_id] != TRAIN_ON_MATCHING or not len(days):
        return dict.fromkeys(_TRAINING_DAY_COLUMNS, _DASH)
    return dict(zip(_TRAINING_DAY_COLUMNS, (int(days.min()), float(np.median(days)), int(days.max()))))


def _matching_counts(
    model: Any, plan: PredictorPlan, data: pd.DataFrame, target: str, horizon: int, freq: str
) -> Optional[Dict[Tuple[int, int], np.ndarray]]:
    """How many days the forecasts of matching lists train on, before any fit, as a run counts them. A list holding a column
    `run` would refuse has no count: the predictor table says what is wrong with it."""
    if not plan.uses_matching or target not in data.columns:
        return None
    usable = [
        name
        for name in dict.fromkeys(name for names in plan.lists for name in names)
        if name in data.columns and _contract._is_numeric(data[name])
    ]
    model._load_periods(data, list(dict.fromkeys([target, *usable])), horizon, target)
    model._data = None
    absent = _contract.missing_test_rows(data.index, freq, plan.issue_days)
    return model._matching_day_counts(plan, target, {(period, day.toordinal()) for period, day in absent})


def explain_model(
    model: Any,
    data: pd.DataFrame,
    info: Optional[InformationSet],
    test: Optional[Tuple[str, str]],
    horizon: int,
    hour: int,
    target: str,
    freq: str,
    echo: bool,
    horizons: Optional[Sequence[int]] = None,
) -> Dict[str, pd.DataFrame]:
    """The tables of `ModelPipeline.explain`. `model` is the caller's own copy, since planning leaves state on it.

    `horizon` is the forecast shown, and `horizons` the ones a predictor function is planned for, every one from 1 to `horizon` if
    not given; `horizon` is their largest."""
    model._set_freq(freq)
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError(f"horizon must be a whole number of days from 1, got {horizon!r}")
    horizons = tuple(range(1, horizon + 1)) if horizons is None else tuple(horizons)
    periods = model._periods_per_day
    if not isinstance(hour, (int, np.integer)) or not 0 <= hour < periods:
        raise ValueError(f"hour is a period of the day: 0-{periods - 1} with freq={freq!r}, got {hour!r}")
    if test is not None and pd.Timestamp(test[0]) > pd.Timestamp(test[1]):
        raise ValueError(f"test starts after it ends: {test}")
    if model._ctx_fn is not None and test is None:
        raise ValueError(f"{model.name}: explain() of a predictor function of `ctx` needs test=(start, end) to plan it")

    start, end = test if test is not None else (None, None)
    days = InformationSet._issue_days(data.index, start, end)
    newest = model._newest_targets(data, target, horizon, freq, info, [day.date() for day in days])
    # What `run` refuses it can't explain: the default freq on 15-minute data would read period 10 as 10:00, not 02:30.
    model._check_frame(data, target, horizon, freq, start, end, newest=newest)
    model._newest = newest
    clock = (info or _NAMES)._issue_clock(days, data.index, freq)
    out: Dict[str, pd.DataFrame] = {}
    if model._ctx_fn is not None:
        # At "warn", so that a late `ctx.value()` is a thing to show rather than an error.
        level = "none" if info is None else "warn"
        plan = model._plan_predictors(
            data, target, horizon, freq, info, start, end, warn_unpublished=False, availability=level, horizons=horizons
        )
        out["lists"] = list_table(plan, horizons, _matching_counts(model, plan, data, target, horizon, freq))
        ids = plan.assignment[(hour, horizon)]
        chosen = int(
            pd.Series(ids).value_counts(sort=False).sort_index().idxmax()
        )  # ties go to the list returned first
        used = ids == chosen
        names = list(plan.lists[chosen])
        subject = (
            f"{model.name}: list {chosen + 1} ({plan.hashes[chosen]}), on {int(used.sum())} of {len(ids)} issue days"
        )
    else:
        used = np.ones(len(days), dtype=bool)
        names = model._expand_predictors(horizon, hour=hour)
        subject = model.name
    rows = model._training_rows(
        newest, model._reach_window(data, [day.date() for day in days]), frame_first_day(data.index)
    )
    out["predictors"] = predictor_table(names, data, info, horizon, hour, freq, clock, used, rows)
    out["grouped"] = grouped_table(out["predictors"])
    row_target = row_target_table(model, data, info, horizon, hour, freq, clock, used, rows)
    if row_target is not None:
        out["row_target"] = row_target
    spec = history_spec_of(model, target)
    if spec is not None:
        out["series"] = history_table(spec, data, info, days[used], horizon, freq)
    if newest is not None:
        out["newest_target"] = newest_target_table(
            _newest_by_issue_day(model, newest, days, hour, horizon)[used], horizon
        )
    if echo:
        minute = clock.row_minutes(np.array([hour]), _STEP_MINUTES[freq])[0]
        _print(
            out,
            subject,
            info,
            days,
            horizon,
            hour,
            freq,
            used,
            data.index,
            model._ctx_fn,
            minute,
            row_target_note(model),
        )
    return out


def _newest_by_issue_day(model: Any, newest: Any, days: pd.DatetimeIndex, hour: int, horizon: int) -> np.ndarray:
    """The newest target of the forecast of each issue day, relative to it. With `refit` it was read on the anchor a, so it is
    `a + min(k(a), horizon - 1)`, relative to the issue day; with `refit="auto"` the newest of its windows' (each on the anchor of
    its own rule), as its record's `newest_target_day`."""
    if model.refit is None:
        return newest.k[hour]
    dates = [day.date() for day in days]
    rules = list(dict.fromkeys(model._refit_rules))
    return np.array(
        [
            max(
                (a - d).days + min(int(newest.at(hour, a.toordinal())), horizon - 1)
                for a in (refit_anchor(d, rule) for rule in rules)
            )
            for d in dates
        ],
        dtype=np.int64,
    )


def newest_target_table(newest: np.ndarray, horizon: int) -> pd.DataFrame:
    """For a model with `newest_target="published"`: the newest target of the forecast shown and the newest row it trains on,
    relative to the issue day D, and on how many issue days; `newest` has the day of that target per issue day. A row is never
    dated D or later, so beyond D-1 + horizon a target known earlier doesn't train."""
    days = Counter(min(int(k), horizon - 1) for k in newest)
    rows = [
        {"newest target": _day_label(k), "newest row": _day_label(k - horizon), "issue days": count}
        for k, count in sorted(days.items(), reverse=True)
    ]
    return pd.DataFrame(rows, columns=["newest target", "newest row", "issue days"])


def _day_label(offset: int) -> str:
    return "D" if offset == 0 else f"D{offset:+d}"


def _print(out, subject, info, days, horizon, hour, freq, used, index, function, minute, row_note="") -> None:
    if function is not None:
        print(f"{function_name(function)}: {out['lists']['list'].nunique()} distinct list(s)")
        print(out["lists"].to_string(index=False))
        print()
    issued = f"issued {info.issue_time} {info.issue_timezone} on D" if info is not None else "no issue time"
    time_of_day = fmt_minutes(minute)
    print(
        f"{subject}: horizon {horizon}, period {hour} ({time_of_day}); T = D + {horizon}; {issued}; "
        f"issue days {days[0].date()} … {days[-1].date()}"
    )
    if not ("series" in out and out["grouped"].empty):  # a sequence model may read series alone
        print(out["grouped"].to_string(index=False))
    if row_note:
        print(f"per-row target: {row_note}")
    if "row_target" in out:
        print(out["row_target"].to_string(index=False))
    if "series" in out:
        print("the series it reads, cut at the newest step known at the issue time:")
        print(out["series"].to_string(index=False))
    if "newest_target" in out:
        print("newest_target='published': the newest target it trains on, by its rule")
        print(out["newest_target"].to_string(index=False))
    note = _no_row_note(index, freq, hour, days[used])
    if note:
        print(note)


def _no_row_note(index: pd.DatetimeIndex, freq: str, period: int, days: pd.DatetimeIndex) -> str:
    """Which issue days have no row at this period, so no forecast to explain: a gap, or an hour a spring switch skips."""
    if not len(days):
        return ""
    local = _contract.wall_clock(index)
    present = local[_contract.period_index(index, freq) == period].normalize()
    absent = days[~days.isin(present)]
    if not len(absent):
        return ""
    shown = ", ".join(str(day.date()) for day in absent[:3]) + (" ..." if len(absent) > 3 else "")
    return f"No row at period {period} on {len(absent)} of the issue days ({shown}): no forecast comes from it."
