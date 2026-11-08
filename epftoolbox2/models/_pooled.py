"""How a `PooledModel` runs (step 7g): one fit per pooled unit over every zone of a run, each zone's forecasts in its own store.

`ModelPipeline.run(data={zone: frame})` makes a **member** per zone (`member_of`: a copy of the pooled model's inner model under
the pooled model's name, with `_pooled`, what that zone's manifest records), checks each on its own frame as it checks any model
(`_checked_inputs`), then judges every zone's training rows as of the earliest issue instant of their day over the zones
(`shared_fit_checks`), since one fit serves them all and is made by the first issue. `run_pooled` then calls each member's
`_prepare_run` (its masks, store, manifest, pending tasks and the skips decided before any fit, exactly as its own run has them),
reads each zone's period arrays and worker inputs (`_build_hour_arrays`, `_worker_inputs`), and runs one pool whose jobs are
(issue day or refit anchor, period) and hold every pending task of every zone there.

A **unit** is (period, issue day or anchor, horizon), or (period, issue day or anchor) when the model pools horizons. In the
worker (`fit_unit`) each zone's (and horizon's) rows are read as the model's own path reads them (`_worker.member_rows`, or
`unit_rows` with refit), filtered (`finite_rows`), and scaled by a scaler fitted on them alone, the target too (`member_fit`):
every zone takes part in every unit, whether or not it has a pending task there, so a fit never depends on which forecasts a
run still has to make. The scaled rows are stacked zone by zone in the run's order, horizon by horizon, with the zone feature
(and with pooled horizons a one-hot horizon), fitted once (`_fit_model`), and each pending task's issue row, scaled by its own
member's scaler, is forecast on its own (`_predict_fitted`) and inverted by that scaler. With one zone (and one horizon) the
stack is that zone's rows and nothing is added, so a record is the model's own to the bit.
"""

import copy
import datetime
import multiprocessing
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from rich.progress import Progress

from ..information_set import InformationSet, InformationSetError, frame_first_day, over_time
from ..results.availability import listing_rows, merge_listings
from ..results.ref import ModelResultRef
from ..scalers.registry import make_scaler
from ..scalers.standard import StandardScaler
from ._calendar import FitUnit, listed_days, refit_anchor
from ._engine import run_arguments
from ._worker import (
    SKIP_FIT_ERROR,
    SKIP_MISSING_TEST_PREDICTORS,
    SKIP_MISSING_TEST_ROW,
    SKIP_NO_FINITE_TRAINING_ROWS,
    SKIP_TOO_FEW_ROWS,
    TaskFailed,
    _limit_native_threads,
    _list_hash_of,
    finite_rows,
    forecast_record,
    issue_row,
    member_rows,
    task_mask,
    task_place,
    unit_rows,
)
from .base import _WINDOWS_MAX_WORKERS, BaseModel, _MemorySink, _RunState
from .pooled import CODE, MISSING_RAISE, ONEHOT, PooledModel

# The minutes of a day, for the issue instants compared in `earlier_issue_zones`.
_NS_PER_MINUTE = 60 * 10**9


def member_of(pooled: PooledModel, identity: Optional[Dict[str, Any]]) -> BaseModel:
    """The member of one zone: a copy of the pooled model's inner model, named as the pooled model, whose manifest records
    `identity` as `pooled` (`results.manifest.build_manifest` reads `_pooled`)."""
    member = copy.deepcopy(pooled.model)
    member.name = pooled.name
    member._pooled = identity
    return member


def needed_columns(member: BaseModel, freq: str, horizons: Sequence[int]) -> List[str]:
    """Every predictor name the member's lists hold at these horizons, each once, in first use."""
    member._set_freq(freq)
    expanded = member._precompute_expanded_predictors(horizons)
    return list(dict.fromkeys(name for names in expanded.values() for name in names))


def with_missing_columns(pooled: PooledModel, zone: str, frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """`frame`, or with `missing="zero"` a copy that holds a column of zeros for each of `columns` it lacks. With
    `missing="raise"` a ValueError naming them, before any fit."""
    lacking = [name for name in columns if name not in frame.columns]
    if not lacking:
        return frame
    if pooled.missing == MISSING_RAISE:
        raise ValueError(
            f"PooledModel {pooled.name!r}: zone {zone!r} has no column(s) {lacking[:10]}"
            f"{f' and {len(lacking) - 10} more' if len(lacking) > 10 else ''}. Add them to its frame, or pass "
            "missing='zero' to give them 0 there, their standardised mean."
        )
    return frame.assign(**{name: 0.0 for name in lacking})


def check_horizon_lists(pooled: PooledModel, member: BaseModel, freq: str, horizons: Sequence[int]) -> None:
    """With pooled horizons the lists of every horizon are stacked by place: a ValueError unless they have one length."""
    if not pooled.pools_horizons:
        return
    member._set_freq(freq)
    lengths = {hz: len(names) for (_, hz), names in member._precompute_expanded_predictors(horizons).items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"PooledModel {pooled.name!r} pools horizons, which stacks each horizon's predictors by their place in the list, "
            f"but the lists have different lengths by horizon ({lengths}). Give every horizon a list of the same length."
        )


# -- one fit serves every zone: its rows are judged as of the earliest issue -------------------------------------------------


def issue_instants(information_set: InformationSet, days: Sequence[str]) -> np.ndarray:
    """The issue instant of each of these days (ISO dates) on the information set's clock, as UTC nanoseconds."""
    naive = pd.to_datetime([f"{day} {information_set.issue_time}" for day in days])
    zoned = naive.tz_localize(information_set.issue_timezone, ambiguous=True, nonexistent="shift_forward")
    return zoned.asi8


def earlier_issue_zones(infos: Dict[str, Optional[InformationSet]], days: Sequence[str]) -> Dict[str, List[str]]:
    """Per zone, the other zones whose forecasts are issued before its own on some of these days: the fit a pooled unit shares
    is made by the earliest of them, so a zone's training rows must be known by then. Zones issuing on one clock have none."""
    if not days or any(info is None for info in infos.values()):
        return {zone: [] for zone in infos}
    instants = {zone: issue_instants(info, days) for zone, info in infos.items()}
    return {
        zone: [
            other
            for other in infos
            if other != zone
            and infos[other].issue_timezone != infos[zone].issue_timezone
            and bool((instants[other] < instants[zone]).any())
        ]
        for zone in infos
    }


def check_days(
    test_start: Optional[str], test_end: Optional[str], weekdays: Optional[Sequence[int]]
) -> Optional[List[str]]:
    """The issue days a check reads, as `BaseModel._check_inputs` reads them in a plain run: the listed weekdays' days, or None
    for every day of [test_start, test_end]."""
    if weekdays is None or test_start is None or test_end is None:
        return None
    first, last = datetime.date.fromisoformat(test_start), datetime.date.fromisoformat(test_end)
    every = [first + datetime.timedelta(days=d) for d in range((last - first).days + 1)]
    return [day.isoformat() for day in listed_days(every, weekdays)]


def all_days(test_start: Optional[str], test_end: Optional[str], days: Optional[Sequence[str]]) -> List[str]:
    """The issue days themselves: `days`, or every day of [test_start, test_end]."""
    if days is not None:
        return list(days)
    if test_start is None or test_end is None:
        return []
    first, last = datetime.date.fromisoformat(test_start), datetime.date.fromisoformat(test_end)
    return [(first + datetime.timedelta(days=d)).isoformat() for d in range((last - first).days + 1)]


def shared_fit_uses(
    pooled: PooledModel,
    zone: str,
    member: BaseModel,
    frame: pd.DataFrame,
    expanded: Dict[Tuple[int, int], List[str]],
    target: str,
    freq: str,
    other: str,
    information_set: InformationSet,
    test_start: Optional[str],
    test_end: Optional[str],
    days: Optional[Sequence[str]],
    level: str,
) -> List[Any]:
    """Zone `zone`'s training rows judged on the clock of `other`, whose issue is earlier (`information_set` is other's): its
    newest training target (on the refit anchors with `refit`) and, for a column whose rule changes over time, its predictors
    on the rows a fit reads. At `level` "raise" an InformationSetError names both zones; at "warn" the uses are returned, each
    with a note saying why. Its issue row isn't judged here: it serves `zone`'s own forecast, issued on `zone`'s clock."""
    step = 15 if freq == "15min" else 60
    names = sorted({name for names in expanded.values() for name in names})
    changing = {name for name in names if over_time(information_set._resolve_for(name, step).rule)}
    boundary = {pair: [name for name in names if name in changing] for pair, names in expanded.items()}
    issued = member._days_of(test_start, test_end, None)
    window = member._reach_window(frame, issued)
    rows = member._training_rows(None, window, frame_first_day(frame.index))
    method = information_set.check if level == "raise" else information_set.late_uses
    label = f"{pooled.name} ({zone})"
    try:
        uses = method(
            boundary,
            target,
            freq,
            frame.index,
            test_start,
            test_end,
            model_name=label,
            issue_days=days,
            check_target=member.refit is None,
            training_rows=rows,
        )
        if member.refit is not None:  # a unit's rows are cut as of the issue time on its anchor
            on_anchors = member._target_on_anchors(
                method, expanded, target, freq, frame.index, test_start, test_end, days, window
            )
            if level != "raise":
                uses = [*uses, *on_anchors]
    except InformationSetError as exc:
        raise InformationSetError(
            f"PooledModel {pooled.name!r} fits zone {zone!r} with zone {other!r}, whose forecasts are issued earlier "
            f"({information_set.issue_time} {information_set.issue_timezone}): the fit they share is made by then, so "
            f"{zone!r}'s training rows must be known by then too. {exc}"
        ) from None
    if level == "raise":
        return []
    note = (
        f"pooled with zone {other!r}, issued earlier ({information_set.issue_time} {information_set.issue_timezone}): the "
        "shared fit is made by then"
    )
    return [replace(use, hint=note) for use in uses if use.column.rule is not None]


def shared_fit_checks(
    pooled: PooledModel,
    members: Dict[str, BaseModel],
    frames: Dict[str, pd.DataFrame],
    checks: Dict[str, Any],
    infos: Dict[str, Optional[InformationSet]],
    target: str,
    freq: str,
    test_start: Optional[str],
    test_end: Optional[str],
    weekdays: Optional[Sequence[int]],
    level: str,
) -> Dict[str, Any]:
    """Every zone's training rows judged as of the earliest issue instant of their day over the zones (`earlier_issue_zones`),
    before any fit: at "raise" a late one raises, at "warn" it joins the zone's listing (the check `checks[zone]` with it, which
    its manifest records), at "none" nothing is looked at. Returns the checks."""
    if level == "none":
        return checks
    days = check_days(test_start, test_end, weekdays)
    earlier = earlier_issue_zones(infos, all_days(test_start, test_end, days))
    out = dict(checks)
    for zone, others in earlier.items():
        found = []
        for other in others:
            found += shared_fit_uses(
                pooled,
                zone,
                members[zone],
                frames[zone],
                checks[zone].expanded,
                target,
                freq,
                other,
                _on_clock_of(infos[zone], infos[other]),
                test_start,
                test_end,
                days,
                level,
            )
        if found:
            out[zone] = checks[zone]._replace(listing=merge_listings(checks[zone].listing, listing_rows(found)))
    return out


def _on_clock_of(rules: InformationSet, clock: InformationSet) -> InformationSet:
    """`rules` (a zone's own information set, its source rules included) issued at `clock`'s time in `clock`'s zone."""
    out = rules.copy()
    out.issue_time = clock.issue_time
    out.issue_timezone = clock.issue_timezone
    return out


# -- the run --------------------------------------------------------------------------------------------------------------


class ZoneInputs(NamedTuple):
    """One zone's part of a pooled pool's inputs: its period arrays (`BaseModel._build_hour_arrays`), expanded lists keyed
    (period, horizon) and worker config (`BaseModel._worker_inputs`), as that zone's member would run its own tasks with."""

    zone: str
    hour_arrays: Dict
    expanded: Dict
    config: Dict


class PoolSettings(NamedTuple):
    """What a pooled unit is and how its rows are told apart: the zones in stack order, the planned horizons, whether a unit
    pools them, the zone feature, the refit rule (None: a unit per issue day) and the run's first day."""

    zones: Tuple[str, ...]
    horizons: Tuple[int, ...]
    pools_horizons: bool
    zone_feature: Optional[str]
    refit: Optional[str]
    test_start: str


class PooledInputs(NamedTuple):
    """Everything a pooled job reads: each zone's inputs, the settings and the model, rebuilt as a worker rebuilds it
    (`model_cls(predictors=[], **model_kwargs)`)."""

    zones: Tuple[ZoneInputs, ...]
    settings: PoolSettings
    model: Any


_P_INPUTS: Optional[PooledInputs] = None


def pooled_inputs(
    zones: Sequence[ZoneInputs], settings: PoolSettings, model_cls: type, model_kwargs: Dict
) -> PooledInputs:
    return PooledInputs(tuple(zones), settings, model_cls(predictors=[], **model_kwargs))


def _pooled_init(zones, settings, model_cls, model_kwargs) -> None:
    """Initialiser of a pooled pool's worker: limit native threads, then load the inputs (`_worker._pool_init`'s rule)."""
    _limit_native_threads()
    _pooled_worker_init(zones, settings, model_cls, model_kwargs)


def _pooled_worker_init(zones, settings, model_cls, model_kwargs) -> None:
    """Load a pooled pool's inputs into this process (`_worker._worker_init`'s rule: no thread limit, for a caller's own
    process, such as a test's pool that runs here)."""
    global _P_INPUTS
    _P_INPUTS = pooled_inputs(zones, settings, model_cls, model_kwargs)


def unit_key(settings: PoolSettings, task: tuple) -> Tuple[int, Any, Optional[int]]:
    """(period, issue day or refit anchor, horizon or None when horizons are pooled) of a task (period, horizon, day in test)."""
    period, hz, day = task
    if settings.refit is None:
        when: Any = day
    else:
        when = refit_anchor(
            datetime.date.fromisoformat(settings.test_start) + datetime.timedelta(days=day), settings.refit
        )
    return period, when, None if settings.pools_horizons else hz


class Member(NamedTuple):
    """One zone's (and horizon's) part of a unit: its rows scaled by its own scaler, which also scaled its pending tasks' issue
    rows, each kept as (task, row in the period arrays, the scaled row)."""

    zone: int
    horizon: int
    train_x: np.ndarray
    train_y: np.ndarray
    scaler: Any
    rows: List[Tuple[tuple, int, np.ndarray]]


def member_rows_of(inputs: PooledInputs, z: int, period: int, when: Any, hz: int) -> slice:
    """The rows of zone `z` a unit (period, `when`, `hz`) trains on, before any filter: those of a forecast issued on `when` in
    that zone's own day ids (`config["offset"]`, its frame's first local date), as its own model's path cuts them."""
    zone = inputs.zones[z]
    cfg = zone.config
    days = zone.hour_arrays[period]["days"]
    if inputs.settings.refit is not None:
        return unit_rows(cfg, days, FitUnit(period, hz, when, None, 0))[0]
    return member_rows(days, cfg["offset"] + when, cfg["training_window"], hz, -1)


def member_fit(
    inputs: PooledInputs, z: int, period: int, when: Any, hz: int, tasks: Sequence[tuple]
) -> Tuple[Optional[Member], List[Tuple[tuple, str]]]:
    """Zone `z`'s part of a unit at horizon `hz` and the skips of its `tasks` (all of that zone and horizon): its rows, filtered
    and scaled as the model's own path does it, in the same order of checks (a missing issue row, no row, no finite row, a
    missing issue-row predictor), or None with no finite row. The binary-column mask is the zone's own: its parent's mask of the
    (period, horizon) without refit, else the rows' own before any filter, as `_fit_unit` reads it."""
    zone = inputs.zones[z]
    cfg = zone.config
    arrays = zone.hour_arrays[period]
    days = arrays["days"]
    predictors = zone.expanded[(period, hz)]
    col_idx = [arrays["col_index"][c] for c in predictors]

    skipped: List[Tuple[tuple, str]] = []
    issued = []
    for task in tasks:
        test = issue_row(days, cfg["offset"] + task[2])
        if test is None:
            skipped.append((task, SKIP_MISSING_TEST_ROW))
        else:
            issued.append((task, test))

    train = member_rows_of(inputs, z, period, when, hz)
    train_x = arrays["x"][train][:, col_idx]
    train_y = arrays["y"][train][:, hz - 1]
    if train_x.shape[0] == 0:
        return None, skipped + [(task, SKIP_TOO_FEW_ROWS) for task, _ in issued]
    mask = None if inputs.settings.refit is not None else task_mask(cfg, None, (period, hz))
    if mask is None:
        mask = StandardScaler.get_scalable_mask(
            train_x
        )  # before any filter, as `_fit_unit` and `binary_mask="fit"` read it

    finite = finite_rows(train_x, train_y)
    train_x = train_x[finite]
    train_y = train_y[finite]
    if train_x.shape[0] == 0:
        return None, skipped + [(task, SKIP_NO_FINITE_TRAINING_ROWS) for task, _ in issued]

    forecast = []
    for task, test in issued:
        if np.isfinite(arrays["x"][test, col_idx]).all():
            forecast.append((task, test))
        else:
            skipped.append((task, SKIP_MISSING_TEST_PREDICTORS))
    scaler = make_scaler(cfg.get("scaler", "standard"))
    train_x, train_y, test_x = scaler.fit_transform(
        train_x.copy(),
        train_y.copy(),
        arrays["x"][[test for _, test in forecast]][:, col_idx],
        scalable_mask=mask,
        columns=predictors,
    )
    rows = [(task, test, test_x[i : i + 1]) for i, (task, test) in enumerate(forecast)]
    return Member(z, hz, train_x, train_y, scaler, rows), skipped


def features(settings: PoolSettings, z: int, hz: int) -> Optional[np.ndarray]:
    """The columns that tell a row of zone `z` and horizon `hz` apart in a fit, or None: a one-hot zone (or its code) with more
    than one zone, and with pooled horizons a one-hot horizon with more than one."""
    parts: List[float] = []
    n_zones = len(settings.zones)
    if n_zones > 1 and settings.zone_feature == ONEHOT:
        parts += [1.0 if i == z else 0.0 for i in range(n_zones)]
    elif n_zones > 1 and settings.zone_feature == CODE:
        parts.append(float(z))
    if settings.pools_horizons and len(settings.horizons) > 1:
        parts += [1.0 if h == hz else 0.0 for h in settings.horizons]
    return np.array(parts, dtype=np.float64) if parts else None


def _with_features(x: np.ndarray, extra: Optional[np.ndarray]) -> np.ndarray:
    if extra is None:
        return x
    return np.hstack([x, np.broadcast_to(extra, (x.shape[0], extra.size))])


def fit_unit(
    inputs: PooledInputs, key: Tuple[int, Any, Optional[int]], tasks: Dict[int, List[tuple]]
) -> List[Tuple[int, tuple, Union[Dict, str]]]:
    """Fit one unit over every zone and forecast the tasks of `tasks` (zone index -> its tasks of the unit): (zone index, task,
    record or skip reason) of each. Every zone's rows train, whatever tasks it has here."""
    settings = inputs.settings
    period, when, hz_key = key
    horizons = settings.horizons if hz_key is None else (hz_key,)
    members: List[Member] = []
    outcomes: List[Tuple[int, tuple, Union[Dict, str]]] = []
    for z in range(len(inputs.zones)):
        mine = tasks.get(z, [])
        for hz in horizons:
            member, skipped = member_fit(inputs, z, period, when, hz, [task for task in mine if task[1] == hz])
            outcomes += [(z, task, reason) for task, reason in skipped]
            if member is not None:
                members.append(member)
    forecast = [member for member in members if member.rows]
    if not forecast:
        return outcomes

    extra = {(m.zone, m.horizon): features(settings, m.zone, m.horizon) for m in members}
    if len(members) == 1 and extra[(members[0].zone, members[0].horizon)] is None:
        train_x, train_y = members[0].train_x, members[0].train_y  # one zone and horizon: the model's own arrays
    else:
        train_x = np.vstack([_with_features(m.train_x, extra[(m.zone, m.horizon)]) for m in members])
        train_y = np.concatenate([m.train_y for m in members])
    model = inputs.model
    try:
        fitted = model._fit_model(train_x, train_y)
    except ValueError as exc:
        return outcomes + [(m.zone, task, f"{SKIP_FIT_ERROR}: {exc}") for m in forecast for task, _, _ in m.rows]

    for m in forecast:
        zone = inputs.zones[m.zone]
        cfg = zone.config
        arrays = zone.hour_arrays[period]
        for task, test, row in m.rows:
            # One row at a time (decision 16 of the step 3 spec): a product over several rows can differ in the last bits.
            pred, coefs = model._predict_fitted(fitted, _with_features(row, extra[(m.zone, m.horizon)]))
            window, day_in_window, _ = task_place(cfg, task[2])
            run_date = datetime.date.fromisoformat(cfg["test_start"]) + datetime.timedelta(days=task[2])
            record = forecast_record(
                run_date,
                period,
                m.horizon,
                day_in_window,
                m.scaler.inverse(float(pred)),
                float(arrays["y"][test, m.horizon - 1]),
                coefs,
                cfg.get("freq", "1h"),
                _list_hash_of(cfg, zone.expanded, period, m.horizon, None),
                window=window,
                refit_anchor=None if settings.refit is None else when.isoformat(),
            )
            outcomes.append((m.zone, task, record))
    return outcomes


def run_pooled_job(
    tasks: Dict[int, List[tuple]], inputs: Optional[PooledInputs] = None
) -> List[Tuple[int, List[Dict], List[Tuple[tuple, str]]]]:
    """One job of a pooled pool, the tasks of one (issue day or anchor, period) by zone index, with `inputs` (the pool's if None):
    each of its units fitted once (`fit_unit`). Returns (zone index, records, [(task, skip reason)]) per zone; an unexpected
    error raises TaskFailed naming the unit."""
    inputs = _P_INPUTS if inputs is None else inputs
    if inputs is None:
        raise ValueError("a pooled job's inputs: pass PooledInputs, or run it in a process `_pooled_init` set up")
    units: Dict[Tuple[int, Any, Optional[int]], Dict[int, List[tuple]]] = {}
    for z, zone_tasks in tasks.items():
        for task in zone_tasks:
            units.setdefault(unit_key(inputs.settings, task), {}).setdefault(z, []).append(task)
    by_zone: Dict[int, Tuple[List[Dict], List[Tuple[tuple, str]]]] = {z: ([], []) for z in tasks}
    for key, unit_tasks in units.items():
        try:
            outcomes = fit_unit(inputs, key, unit_tasks)
        except Exception as exc:
            raise TaskFailed(
                f"pooled unit (period={key[0]}, {'anchor' if inputs.settings.refit else 'day_in_test'}={key[1]}, "
                f"horizon={'pooled' if key[2] is None else key[2]}) raised {exc!r}\n{traceback.format_exc()}"
            ) from None
        for z, task, outcome in outcomes:
            results, skipped = by_zone[z]
            (skipped.append((task, outcome)) if isinstance(outcome, str) else results.append(outcome))
    return [(z, results, skipped) for z, (results, skipped) in by_zone.items()]


def run_pooled(pooled: PooledModel, zones: Dict[str, Tuple[BaseModel, Dict[str, Any]]]) -> Dict[str, ModelResultRef]:
    """Run a pooled model over its zones, `zones[zone]` = (that zone's member, what its `run` is called with), and return each
    zone's ref. Each member's `_prepare_run` and `_finish_run` are its own run's: its store, manifest, skips and ref. A zone
    whose preparation raises stops the run before any fit, its stores closed."""
    states: List[Tuple[str, BaseModel, _RunState]] = []
    try:
        for zone, (member, arguments) in zones.items():
            states.append((zone, member, member._prepare_run(**run_arguments(**arguments))))
        _execute(pooled, states)
    except BaseException:
        _close(states)
        raise
    refs: Dict[str, ModelResultRef] = {}
    error: Optional[BaseException] = None
    for zone, member, state in states:
        try:
            refs[zone] = member._finish_run(state)
        except Exception as exc:  # every zone's counts and warnings, then the first failure
            error = error or exc
    _close(states)
    if error is not None:
        raise error
    return refs


def _close(states: List[Tuple[str, BaseModel, _RunState]]) -> None:
    for _, member, state in states:
        member._release()
        state.close()


def _execute(pooled: PooledModel, states: List[Tuple[str, BaseModel, _RunState]]) -> None:
    """The pool of a pooled run: every zone's arrays and worker inputs, jobs of (issue day or anchor, period) holding every
    zone's pending tasks there, each job's records into its zone's store (or in-memory list) as it comes back."""
    try:
        if not any(state.tasks for _, _, state in states):
            return
        first = states[0][1]
        threads, n_processes = first._executor_config()
        zone_inputs = []
        for zone, member, state in states:
            # Targets for every horizon from 1 to the largest, whatever is planned: the fit reads the column of `hz - 1`.
            hour_arrays = member._build_hour_arrays(max(state.horizons))
            expanded, config = member._worker_inputs(state, threads)
            zone_inputs.append(ZoneInputs(zone, hour_arrays, expanded, config))
        settings = PoolSettings(
            zones=tuple(zone for zone, _, _ in states),
            horizons=tuple(states[0][2].horizons),
            pools_horizons=pooled.pools_horizons,
            zone_feature=pooled.zone_feature,
            refit=first._refit_block,
            test_start=states[0][2].test_start,
        )
        jobs: Dict[Tuple[Any, int], Dict[int, List[tuple]]] = {}
        for z, (_, _, state) in enumerate(states):
            for task in state.tasks:
                period, when, _ = unit_key(settings, task)
                jobs.setdefault((when, period), {}).setdefault(z, []).append(task)
        sinks = [state.store if state.store else _MemorySink(state.in_memory) for _, _, state in states]
        n_processes = max(1, min(n_processes, len(jobs)))
        if sys.platform == "win32":
            n_processes = min(n_processes, _WINDOWS_MAX_WORKERS)
        mp_context = multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn")
        with ProcessPoolExecutor(
            max_workers=n_processes,
            mp_context=mp_context,
            initializer=_pooled_init,
            initargs=(zone_inputs, settings, type(first), first._model_kwargs),
        ) as pool:
            try:
                futures = {pool.submit(run_pooled_job, job): key for key, job in jobs.items()}
                with Progress() as progress:
                    bar = progress.add_task(
                        f"[cyan]{pooled.name} ({len(states)} zones)", total=sum(len(s.tasks) for _, _, s in states)
                    )
                    for future in as_completed(futures):
                        key = futures.pop(future)
                        job = jobs.pop(key)
                        try:
                            outcomes = future.result()
                        except Exception as exc:
                            when, period = key
                            what = f"the refit block from {when}" if settings.refit else f"day {when} of the test"
                            raise RuntimeError(
                                f"{pooled.name}: pooled forecasting failed for {what}, period {period}: {exc}"
                            ) from exc
                        for z, results, skipped in outcomes:
                            _, member, state = states[z]
                            member._collect(state, results, skipped, sinks[z])
                        progress.advance(bar, sum(len(tasks) for tasks in job.values()))
            except BaseException:
                pool.shutdown(wait=True, cancel_futures=True)
                raise
    finally:
        for _, member, state in states:
            if state.store:
                state.store.flush()
            member._cleanup()
