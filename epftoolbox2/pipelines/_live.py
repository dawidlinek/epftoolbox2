"""One live call of `ModelPipeline.forecast()` (step 8e of the forecasting layer): the call flow of the step 8 spec, "One call of
`forecast()`", over the pieces 8a-8g built.

1. Take the live directory's `live.lock` without waiting: held, the status is `busy` (exit 75) and nothing is written. The issue
   day is resolved in the issue zone on the clock (`epftoolbox2.clock`).
2. `ensure=True`: a committed revision of D with status ok or fallback, committed by the deadline, makes the call a `noop` with
   that revision's status; nothing is written.
3. Refresh the data (a `DataPipeline`, with the refresh budget of 8b) and poll: while a waited-for column lacks a cell on the
   issue rows and its give-up time hasn't passed, sleep `poll` on the clock and refresh again; every poll goes to `waits.json`.
4. Plan the issue: each model's run of D alone (`models._fits.plan_issue`), so its keys are those `IssueCalendar` plans and a
   predictor function's lists those its plan gives on the refreshed frame. Its `ctx.published(c)` is the backtest's: true when
   the information set says c is known by the issue time and the frame holds a value now (decision 7): a revision before an
   input is out uses the list without it, and no list uses a column the backtest couldn't.
5. Serve each model: a model whose fits are stored (OLS, Ridge, LassoCV, LEAR) from its stored fits or inline fits by the
   backtest's own lines (`models._fits.serve`), any other through its own run of D. An issue row with a missing input is a
   `missing_test_predictors` skip record in every path, never a NaN into a model.
6. The checks before export over the members' records (`results.checks.check_records`), then every ensemble, a
   `FallbackEnsemble` among them, computed from the records in memory as a backtest computes it from its files.
7. Past the deadline by the clock: status `late` (exit 30), a call line without a revision, nothing else. Forecasts equal to the
   latest committed revision's (every member's and ensemble's predictions and statuses): no revision, `unchanged`, with that
   revision's status. Otherwise the revision is written (`results.live.LiveRevision`): every model's and ensemble's records, the
   snapshot files in `issues/<D>/r<n>/` (`inputs.parquet`, `quality.json`, `checks.json`, `refresh.json`, `waits.json`,
   `fits.json`), the live manifests, then the commit line.
8. The issue status: `skipped` when a planned key of the delivered model has no forecast, else `fallback` when a target day of it
   came from a rung above 0, else `ok` (`results.forecast_result`).

A dry run does all of it and writes nothing anywhere (decision 6). An exception writes an `error` call line (exit 1) when it can
and is raised.
"""

from __future__ import annotations

import datetime
import json
import re
import warnings
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._clock import current_clock
from .._date_utils import resolve_date
from ..information_set import DATA_ZONE, InformationSet
from ..models._calendar import (
    EXPANDING,
    IssueCalendar,
    horizon_arguments,
    issue_day_names,
    resolve_horizons,
    resolve_issue_days,
)
from ..models._engine import ENGINE_OFF
from ..models._fits import FIT_MISSING, serve, stores_fits
from ..models._worker import pool_native_threads
from ..models.base import empty_runs_kept
from ..results.checks import check_records, planned_periods
from ..results.forecast_result import (
    BUSY,
    ERROR,
    FALLBACK,
    LATE,
    OK,
    SKIPPED,
    ForecastResult,
    exit_code,
)
from ..results.live import LiveBusyError, LiveLateError, LiveStore, local_instant
from ..results.manifest import Manifest, check_and_write_live, manifest_path
from ..results.ref import ModelResultRef
from ..results.store import SAVE_FORMATS, STATUS_OK, check_save_format, read_records
from . import _issue_records as records_io

REFRESH = ("auto", "never")
# Days more than the oldest row a fit reads and the transformers' longest lookback: the newest targets' reach and a margin
# (`span` overrides the whole range).
LOOKBACK_PAD_DAYS = 31
FITS_FILE = "fits.json"
_LIVE_KEYS = ("revision", "issued_at", "mode", "snapshot")
_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(s|sec|min|h)?\s*$")
_HH_MM = re.compile(r"^\s*([01]\d|2[0-3]):([0-5]\d)\s*$")


# --- arguments --------------------------------------------------------------------------------------------------------------


def seconds(value: Any, name: str) -> Optional[float]:
    """A duration as seconds: a number of seconds, a `Timedelta`, or text such as "120s", "5min", "2h" (None stays None)."""
    if value is None:
        return None
    if isinstance(value, (pd.Timedelta, datetime.timedelta)):
        out = pd.Timedelta(value).total_seconds()
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out = float(value)
    elif isinstance(value, str) and _DURATION.match(value):
        number, unit = _DURATION.match(value).groups()
        out = float(number) * {None: 1, "s": 1, "sec": 1, "min": 60, "h": 3600}[unit]
    else:
        raise ValueError(f"{name} is a duration such as '120s', '5min' or '2h', or seconds, got {value!r}")
    if not np.isfinite(out) or out < 0:
        raise ValueError(f"{name} must be a duration of 0 or more, got {value!r}")
    return out


def instant_on(day: datetime.date, value: Any, zone: str, name: str) -> Optional[pd.Timestamp]:
    """A time of the issue day as an instant (UTC): "HH:MM" on `day` in `zone`, the issue zone, or an aware instant. A naive
    instant raises: whose clock it is can't be told."""
    if value is None:
        return None
    if isinstance(value, str) and _HH_MM.match(value):
        return local_instant(day.isoformat(), value.strip(), zone)
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} is a time of the issue day 'HH:MM' or an aware instant, got {value!r}") from None
    if stamp is pd.NaT or stamp.tzinfo is None:
        raise ValueError(f"{name} is a time of the issue day 'HH:MM' or an aware instant, got {value!r}")
    return stamp.tz_convert("UTC")


def _is_pipeline(data: Any) -> bool:
    return not isinstance(data, pd.DataFrame) and callable(getattr(data, "run", None))


def issue_zone(info: Optional[InformationSet], data: Any) -> str:
    """The zone the issue day, the deadline and the give-up times are read in: the information set's `issue_timezone`, or with
    "data" the frame's (a `DataPipeline`'s `timezone`)."""
    if info is None:
        raise ValueError(
            "forecast() issues at the information set's issue time: pass issue_time or information_set to ModelPipeline"
        )
    if info.issue_timezone != DATA_ZONE:
        return info.issue_timezone
    zone = getattr(data.index, "tz", None) if isinstance(data, pd.DataFrame) else getattr(data, "timezone", None)
    if zone is None:
        raise ValueError(
            "issue_timezone='data' reads the issue zone from the data, which has none: give the frame (or the "
            "DataPipeline) a time zone, or the information set an issue_timezone"
        )
    return str(zone)


def resolve_issue(issue: Any, zone: str) -> datetime.date:
    """`issue` as the issue day: "today" (or `today_d±k`) in the issue zone on the clock, a date or its ISO text."""
    if isinstance(issue, datetime.datetime):
        raise TypeError(f"issue is a day, a date or 'today', got the instant {issue!r}")
    if isinstance(issue, datetime.date):
        return issue
    try:
        return datetime.date.fromisoformat(str(resolve_date(issue, zone))[:10])
    except ValueError:
        raise ValueError(f"issue must be a date or 'today', got {issue!r}") from None


def _named_base(model: Any, models: Sequence[Any]) -> Any:
    """A `ResidualModel`'s base as a model: its own, the one it was bound to, or the pipeline's model its name names."""
    base = getattr(model, "base", None)
    if base is None or not isinstance(base, str):
        return base
    bound = getattr(model, "_named_base", None)
    return bound if bound is not None else next((other for other in models if other.name == base), None)


def history_days(model: Any, day: datetime.date, models: Sequence[Any] = ()) -> int:
    """How many days before the issue day D the oldest row a forecast of `model` on D reads is: its fit's anchor less its largest
    training window, and for a refit model the newest older fit `fit_missing="previous"` may serve (`previous_cap`), whose rows
    the stale check reads. A `ResidualModel` with a dated `warmup` reads its base's forecasts from that date, each with the base's
    own history. An `"expanding"` window, or a residual model without a dated warm-up, has no oldest row: ValueError."""
    from ..models._fits import previous_cap, stores_fits
    from ..models.residual import ResidualModel

    if isinstance(model, ResidualModel):
        if not isinstance(model.warmup, str):
            raise ValueError(
                f"{model.name}: a ResidualModel without a dated warmup reads base forecasts from where its run starts: pass "
                "span=(start, end) for the data pipeline's run"
            )
        start = datetime.date.fromisoformat(model.warmup)
        base = _named_base(model, models)
        return (day - start).days + (0 if base is None else history_days(base, start, models))
    members = list(getattr(model, "_members", None) or [getattr(model, "training_window", 0) or 0])
    if EXPANDING in members:
        raise ValueError(
            f"{model.name}: an 'expanding' training window reads the whole history, which forecast() can't bound: pass "
            "span=(start, end) for the data pipeline's run"
        )
    start = model._fit_start(day.isoformat()) if hasattr(model, "_fit_start") else day.isoformat()
    anchor = datetime.date.fromisoformat(str(start)[:10])
    back = (day - anchor).days + max(int(member) for member in members)
    if getattr(model, "refit", None) is not None and stores_fits(model) is None:
        back += previous_cap(model)  # an older stored fit served in its place reads rows that far back
    return back


def transformer_lookback(data: Any) -> Tuple[int, List[str]]:
    """The longest lookback (days) a data pipeline's transformers state (`Transformer.lookback_days`), and the names of those
    that can't state one."""
    longest, unknown = 0, []
    for transformer in getattr(data, "transformers", None) or []:
        days = transformer.lookback_days() if hasattr(transformer, "lookback_days") else None
        if days is None:
            unknown.append(type(transformer).__name__)
        else:
            longest = max(longest, int(days))
    return longest, unknown


def data_span(models: Sequence[Any], day: datetime.date, horizons: Sequence[int], data: Any = None) -> Tuple[str, str]:
    """The range a refresh asks the data pipeline for: from the oldest row any forecast of D reads (`history_days`), less the
    horizon, the transformers' longest lookback (`transformer_lookback`) and `LOOKBACK_PAD_DAYS`, to the last target day. A
    transformer that can't state its lookback is warned about: the range may then cut a lag or a window, and `span=` sets it."""
    back = max((history_days(model, day, models) for model in models), default=0)
    lookback, unknown = transformer_lookback(data)
    if unknown:
        warnings.warn(
            f"forecast(): {', '.join(dict.fromkeys(unknown))} can't say how far back it reads, so the data range asked for "
            f"(the fits' rows plus {lookback + LOOKBACK_PAD_DAYS} days) may cut what it computes on the oldest training rows, "
            "which would then differ from the backtest's: pass span=(start, end) to set the range",
            UserWarning,
            stacklevel=4,
        )
    first = day - datetime.timedelta(days=back + max(horizons) + lookback + LOOKBACK_PAD_DAYS)
    last = day + datetime.timedelta(days=max(horizons) + 1)
    return first.isoformat(), last.isoformat()


# --- refresh and waits ------------------------------------------------------------------------------------------------------


class _Data:
    """The data of a call: a frame used as given, or a data pipeline refreshed on each `frame()` within the call's limits."""

    def __init__(self, data: Any, span: Tuple[str, str], options: Dict[str, Any], refresh: str, deadline: Any):
        self.data = data
        self.span = span
        self.options = options
        self.refresh = refresh
        self.deadline = deadline
        self.last = None
        self.refreshes = 0

    @property
    def refreshable(self) -> bool:
        return _is_pipeline(self.data)

    def frame(self) -> pd.DataFrame:
        if not self.refreshable:
            return self.data
        options = dict(self.options)
        if self.refresh == "never":  # the per-source caches alone: no time for any request
            options.update(cache=True, budget=0, on_failure="cache")
        if self.deadline is not None:
            options["deadline"] = self.deadline
        options = {key: value for key, value in options.items() if value is not None}
        try:
            return self.data.run(*self.span, **options)
        finally:
            self.last = getattr(self.data, "last_refresh", None)
            self.refreshes += 1


def lacking_cells(frame: pd.DataFrame, column: str, day: datetime.date, zone: str) -> int:
    """How many cells of `column` on the issue rows of D (its local date in `zone`, the issue zone) hold no value."""
    if column not in frame.columns:
        raise KeyError(f"wait_for: the frame has no column {column!r}")
    rows = records_io.issue_rows(frame, day, zone)
    return int(np.count_nonzero(~pd.notna(frame[column]).to_numpy()[rows]))


def read_columns(
    models: Sequence[Any],
    frame: pd.DataFrame,
    day: datetime.date,
    target: str,
    horizons: Sequence[int],
    freq: str,
    info: Any,
) -> List[str]:
    """Every column of the frame a model's lists of D may read: a fixed list's names at every horizon and period, and what a
    predictor function's plan of D returns or reads through `ctx`."""
    from ..models._calendar import periods_per_day

    names: Dict[str, None] = {}
    for model in models:
        if getattr(model, "_ctx_fn", None) is not None:
            plan = model._plan_predictors(
                frame,
                target,
                max(horizons),
                freq,
                info,
                day.isoformat(),
                day.isoformat(),
                warn_unpublished=False,
                availability="none",
                horizons=list(horizons),
            )
            if plan is not None:
                for names_of_list in plan.lists:
                    names.update(dict.fromkeys(names_of_list))
                names.update(dict.fromkeys(sorted(plan.queried)))
                plan.release()
            continue
        expand = getattr(model, "_expand_predictors", None)
        if expand is None:
            continue
        for horizon in horizons:
            for period in range(periods_per_day(freq)):
                names.update(dict.fromkeys(str(name) for name in expand(horizon, period)))
    return [name for name in names if name in frame.columns]


def rule_give_ups(
    frame: pd.DataFrame, columns: Sequence[str], info: InformationSet, day: datetime.date, freq: str, grace: float
) -> Dict[str, pd.Timestamp]:
    """`wait_for="rules"`: per column with a rule, its give-up instant, the latest publication time its rule gives a cell of the
    issue rows that it knows by the issue time, plus `grace`. A column without a rule, or late at the issue time on every row, is
    not waited for (no list of D can use it)."""
    from ..models.context import _Cells

    zone = frame.index.tz
    cells = _Cells(frame, info, freq, [pd.Timestamp(day)], availability="none")
    midnight = pd.Timestamp(day).tz_localize(zone, ambiguous=True, nonexistent="shift_forward")
    out: Dict[str, pd.Timestamp] = {}
    for column in columns:
        try:
            resolved = cells.column(column)
        except Exception:  # a name the rules can't parse: not waited for
            continue
        if resolved.rule is None:
            continue
        minutes = np.asarray(cells.available(column), dtype=float)
        known = minutes[np.isfinite(minutes) & (minutes <= cells._clock.issue)]
        if not known.size:
            continue
        out[column] = (midnight + pd.Timedelta(minutes=float(known.max())) + pd.Timedelta(seconds=grace)).tz_convert(
            "UTC"
        )
    return out


def wait(
    source: _Data,
    give_ups: Mapping[str, pd.Timestamp],
    day: datetime.date,
    zone: str,
    poll: float,
    deadline: Optional[pd.Timestamp],
    columns_of: Any = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Refresh, and while a waited-for column lacks a cell on the issue rows and its give-up time hasn't passed, sleep `poll`
    seconds on the clock and refresh again; never past the latest give-up time or the deadline. Returns the last frame and every
    poll (`waits.json`). `columns_of(frame)` gives the give-up times from a frame, for `wait_for="rules"`."""
    clock = current_clock()
    polls: List[Dict[str, Any]] = []
    while True:
        frame = source.frame()
        if columns_of is not None:
            give_ups = columns_of(frame)
            columns_of = None
        now = clock.now()
        # The issue day's rows as the models select them: by local date in the frame's zone, which may differ from the issue's.
        rows_zone = str(frame.index.tz) if getattr(frame.index, "tz", None) is not None else zone
        lacking = {column: lacking_cells(frame, column, day, rows_zone) for column in give_ups}
        lacking = {column: count for column, count in lacking.items() if count}
        waiting = sorted(column for column in lacking if now < give_ups[column])
        entry: Dict[str, Any] = {
            "poll": len(polls) + 1,
            "at": now.isoformat(),
            "lacking": lacking,
            "waiting": waiting,
            "refresh": _refresh_summary(source.last),
        }
        polls.append(entry)
        if not lacking:
            entry["outcome"] = "in" if give_ups else "nothing to wait for"
            return frame, polls
        if not waiting:
            entry["outcome"] = "gave up"
            entry["gave_up"] = sorted(lacking)
            return frame, polls
        limit = max(give_ups[column] for column in waiting)
        if deadline is not None:
            limit = min(limit, deadline)
        wake = min(now + pd.Timedelta(seconds=poll), limit)
        if wake <= now or not source.refreshable:
            entry["outcome"] = "gave up" if source.refreshable else "nothing to refresh"
            entry["gave_up"] = sorted(lacking)
            return frame, polls
        clock.sleep((wake - now).total_seconds())


def _refresh_summary(record: Any) -> Any:
    if record is None:
        return None
    sources = getattr(record, "sources", ()) or ()
    return {getattr(source, "name", str(i)): getattr(source, "status", None) for i, source in enumerate(sources)}


# --- serving ----------------------------------------------------------------------------------------------------------------


def _skip_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for record in records:
        status = record.get("status", STATUS_OK)
        if status != STATUS_OK:
            reason = str(status).split(":")[0]
            out[reason] = out.get(reason, 0) + 1
    return out


def _run_lists(manifest: Optional[Dict[str, Any]]) -> List[str]:
    """The columns a model run through its own run read, from its manifest: its predictors and every list of its plan."""
    if not manifest:
        return []
    names: Dict[str, None] = dict.fromkeys(str(name) for name in manifest.get("predictors") or [])
    for entry in manifest.get("predictor_lists") or []:
        names.update(dict.fromkeys(str(name) for name in entry.get("predictors", []) or []))
    return list(names)


def serve_models(
    pipeline: Any,
    frame: pd.DataFrame,
    day: datetime.date,
    save_dir: Optional[Path],
    *,
    target: str,
    horizon: int,
    horizons: Sequence[int],
    freq: str,
    info: Any,
    issue_days: Optional[Sequence[str]],
    fit_missing: str,
) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Each model's records of D (by name, in the pipeline's order), how they were made, its lists (for the snapshot) and its live
    manifest. A model whose fits are stored is served (`models._fits.serve`); any other runs as a backtest of D alone, through a
    pipeline of those models, into a scratch directory that is removed (its manifest read first)."""
    engine = pipeline.engine != ENGINE_OFF
    records: Dict[str, List[Dict]] = {}
    made: Dict[str, Dict[str, Any]] = {}
    lists: Dict[str, Any] = {}
    manifests: Dict[str, Any] = {}
    run_models = []
    for model in pipeline.models:
        if stores_fits(model) is not None:
            run_models.append(model)
            continue
        served = serve(
            model,
            frame,
            day,
            save_dir,
            target=target,
            horizons=horizons,
            freq=freq,
            information_set=info,
            availability=pipeline.availability,
            issue_days=issue_days,
            engine=engine,
            fit_missing=fit_missing,
        )
        records[model.name] = served.records
        fits = {
            anchor: {**entry, **served.identity.get(anchor, {})} for anchor, entry in served.fits.items()
        }  # with what each fit was made from
        made[model.name] = {"source": "stored fits", "fits": fits, "notes": served.notes}
        lists[model.name] = served.lists
        manifests[model.name] = served.manifest
    if run_models:
        from .model_pipeline import ModelPipeline

        for model in run_models:  # decision 5, as the stored fits' path checks every fit it makes now
            refuse_too_early(
                model,
                frame,
                day,
                target=target,
                horizons=horizons,
                freq=freq,
                info=info,
                availability=pipeline.availability,
                issue_days=issue_days,
            )

        sub = ModelPipeline(information_set=pipeline.information_set, engine=pipeline._engine)
        sub._availability = pipeline._availability
        sub._bases_from_the_whole_list = True  # a ResidualModel's named base is bound already, to the pipeline's model
        for model in run_models:
            sub.add_model(model)
        with tempfile.TemporaryDirectory(prefix="epf-forecast-") as scratch, empty_runs_kept():
            sub.run(
                frame,
                day.isoformat(),
                day.isoformat(),
                target=target,
                freq=freq,
                save_dir=scratch,
                **horizon_arguments(horizon, horizons),
                **({} if issue_days is None else {"issue_days": issue_days}),
            )
            for model in run_models:
                path = sub.results_path(scratch, model)
                lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
                records[model.name] = [json.loads(line) for line in lines if line.strip()]
                manifest = json.loads(manifest_path(path).read_text(encoding="utf-8"))
                manifests[model.name] = Manifest.from_dict(manifest)
                lists[model.name] = _run_lists(manifest)
                made[model.name] = {"source": "run", "reason": stores_fits(model)}
    ordered = {model.name: records[model.name] for model in pipeline.models}
    for name, held in ordered.items():
        made[name]["forecasts"] = sum(1 for record in held if record.get("status", STATUS_OK) == STATUS_OK)
        made[name]["skipped"] = _skip_counts(held)
    return ordered, {name: made[name] for name in ordered}, lists, manifests


def refuse_too_early(
    model: Any,
    frame: pd.DataFrame,
    day: datetime.date,
    *,
    target: str,
    horizons: Sequence[int],
    freq: str,
    info: Any,
    availability: str,
    issue_days: Optional[Sequence[str]],
) -> None:
    """Decision 5 for a model that fits at serve time through its own run: refuse (`FitTooEarlyError`) a call made before a
    training target, or an input of the newest training row, that its fits of D read is out by its rule at the clock's now.
    The issue day is planned as the stored fits' path plans it (`plan_issue`) and judged by its check (`check_published`) for
    every anchor, since every fit is made now. A model that path can't plan (a day or pooled unit, a foundation model) is held
    to the target of D - 1 being out, the newest any `newest_target="d-1"` fit of D reads."""
    from ..information_set import _STEP_MINUTES
    from ..models._fits import FitTooEarlyError, check_published, plan_issue

    if getattr(model, "_fits_nothing", False):
        return  # a NaiveModel trains on nothing
    now = current_clock().now()
    try:
        plan = plan_issue(
            model,
            frame,
            day,
            target=target,
            horizon=max(horizons),
            horizons=horizons,
            freq=freq,
            information_set=info,
            availability=availability,
            issue_days=issue_days,
            engine=False,
        )
    except Exception:  # its run reports what is wrong with it; here only when its targets are out is asked
        plan = None
    if plan is not None:
        check_published(plan, list(plan.items), now, frame)
        return
    step = _STEP_MINUTES[freq]
    rule = info._resolve_for(target, step)
    if rule.rule is None:
        return
    clock = info._issue_clock(pd.DatetimeIndex([pd.Timestamp(day)]), frame.index, freq)
    from ..models._calendar import periods_per_day

    periods = np.arange(periods_per_day(freq), dtype=np.int64)
    # D - 1's targets; with newest_target="published" also D's that the rule puts out by the issue time.
    days = (-1, 0) if getattr(model, "newest_target", "d-1") == "published" else (-1,)
    for offset in days:
        minutes = np.asarray(clock.available(rule, periods, step, row_day=offset)[:, 0], dtype=float)
        minutes = minutes[np.isfinite(minutes) & ((offset < 0) | (minutes <= float(np.ravel(clock.issue)[0])))]
        if not minutes.size:
            continue
        out = clock.ctx.origin + pd.Timedelta(minutes=float(minutes.max()))
        if out > now:
            raise FitTooEarlyError(
                f"{model.name}: its fits of {day} train on '{target}' of {day + datetime.timedelta(days=offset)}, which its "
                f"rule publishes at {out.tz_convert('UTC'):%Y-%m-%d %H:%M} UTC, after the clock's now "
                f"({now:%Y-%m-%d %H:%M} UTC): a fit made now would train on fewer targets than the backtest's. Forecast once it "
                "is out."
            )


def compute_ensembles(
    pipeline: Any,
    records: Dict[str, List[Dict]],
    day: datetime.date,
    horizons: Sequence[int],
    freq: str,
    zone: Any,
    weekdays: Optional[Tuple[int, ...]],
) -> Dict[str, List[Dict]]:
    """Every ensemble of the pipeline from the members' records of D held in memory, as a backtest computes it from its files
    (`ModelPipeline._compute_ensembles` with no directory): the records of each, by name."""
    if not pipeline.ensembles:
        return {}
    every = list(horizons) == list(range(1, max(horizons) + 1))
    calendar = IssueCalendar(freq, zone, horizons, weekdays)
    count = calendar.count(day.isoformat(), day.isoformat())
    refs = {
        name: ModelResultRef(
            name=name,
            path=None,
            count=count,
            test_start=day.isoformat(),
            test_end=day.isoformat(),
            horizon=max(horizons),
            freq=freq,
            _results=held,
            zone=zone,
            horizons=None if every else tuple(horizons),
            issue_days=weekdays,
        )
        for name, held in records.items()
    }
    computed, _ = pipeline._compute_ensembles(pipeline._ordered_ensembles(), refs, None, None)
    return {
        ensemble.name: list(computed[ensemble.name]._results or [])
        for ensemble in pipeline.ensembles
        if ensemble.name in computed
    }


def planned_keys(
    day: datetime.date, horizons: Sequence[int], freq: str, zone: Any, weekdays: Any
) -> List[Tuple[int, int]]:
    """(period, horizon) of every key D plans (`IssueCalendar`)."""
    return [(int(p), int(h)) for p, h, _ in IssueCalendar(freq, zone, horizons, weekdays).tasks(day, [0])]


def issue_status(
    records: Sequence[Dict[str, Any]], planned: Sequence[Tuple[int, int]]
) -> Tuple[str, int, Dict[str, int]]:
    """The status of an issue from its delivered model's records: `skipped` when a planned key has no forecast, else `fallback`
    when a target day came from a rung above 0, else `ok`; with how many planned keys lack one, and each target day's rung."""
    have = set()
    rungs: Dict[str, int] = {}
    for record in records:
        prediction = record.get("prediction")
        if record.get("status", STATUS_OK) != STATUS_OK or prediction is None or not np.isfinite(float(prediction)):
            continue
        have.add((int(record["hour"]), int(record["horizon"])))
        if "rung" in record:
            day = str(record["target_date"])
            rungs[day] = max(rungs.get(day, 0), int(record["rung"]))
    missing = sum(1 for key in planned if key not in have)
    if missing:
        return SKIPPED, missing, rungs
    if any(rung > 0 for rung in rungs.values()):
        return FALLBACK, 0, rungs
    return OK, 0, rungs


def _canonical(records: Mapping[str, Sequence[Dict[str, Any]]], day: str) -> Dict[str, List[Any]]:
    """What decides whether two revisions are the same: per model and ensemble, each key's prediction, status, detail and rung,
    sorted by key."""
    out = {}
    for name, held in records.items():
        rows = []
        for record in held:
            if str(record.get("run_date")) != day:
                continue
            prediction = record.get("prediction")
            rows.append(
                (
                    int(record["hour"]),
                    int(record["horizon"]),
                    None if prediction is None else repr(float(prediction)),
                    record.get("status", STATUS_OK),
                    record.get("detail"),
                    record.get("rung"),
                    record.get("member"),
                )
            )
        out[name] = sorted(rows, key=lambda row: (row[1], row[0]))
    return out


def latest_committed(store: LiveStore, day: str) -> Optional[Tuple[int, Any]]:
    """The latest committed revision of D (its number and `Commit`), None without one."""
    revisions = store.committed(day).get(day, {})
    if not revisions:
        return None
    head = max(revisions)
    return head, revisions[head]


def committed_records(
    store: LiveStore, day: str, names: Sequence[str], formats: Mapping[str, str]
) -> Dict[str, List[Dict]]:
    """Each model's and ensemble's records of the latest committed revision of D, by name (none for a name without a file)."""
    seen = store.visible("latest", None)
    seen = {day: seen[day]} if day in seen else {}
    out: Dict[str, List[Dict]] = {}
    for name in names:
        stem = name.lower().replace(" ", "_")
        path = store.save_dir / f"{stem}{SAVE_FORMATS[formats[name]]}"
        if not (path.exists()):
            out[name] = []
            continue
        out[name] = [
            record for record in read_records(path, statuses=None, committed=seen) if str(record.get("run_date")) == day
        ]
    return out


# --- the call -----------------------------------------------------------------------------------------------------------------


def _strip(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in record.items() if key not in _LIVE_KEYS}


def forecast(
    pipeline: Any,
    data: Any,
    issue: Any,
    save_dir: Any,
    *,
    wait_for: Any,
    deadline: Any,
    ensure: bool,
    refresh: str,
    budget: Any,
    timeout: Any,
    retries: Any,
    on_failure: str,
    cache: Any,
    poll: Any,
    grace: Any,
    fit_missing: str,
    deliver: Optional[str],
    dry_run: bool,
    target: str,
    horizon: Optional[int],
    horizons: Optional[Sequence[int]],
    freq: str,
    issue_days: Optional[Sequence[str]],
    save_format: str,
    span: Optional[Tuple[Any, Any]],
) -> ForecastResult:
    """`ModelPipeline.forecast` (see there, and the module docstring)."""
    clock = current_clock()
    started = clock.now()
    call = uuid.uuid4().hex[:12]
    # Every argument is checked before the lock is taken, so a mistake never leaves a call line.
    if not pipeline.models:
        raise ValueError("At least one model is required")
    pipeline._refuse_pooled("forecast")
    if isinstance(data, Mapping):
        raise ValueError(
            "forecast() takes one frame or one DataPipeline: live forecasts of a run over zones aren't supported"
        )
    if not isinstance(data, pd.DataFrame) and not _is_pipeline(data):
        raise TypeError(f"forecast() takes a DataFrame or a DataPipeline, got {type(data).__name__}")
    if refresh not in REFRESH:
        raise ValueError(f"refresh must be one of {list(REFRESH)}, got {refresh!r}")
    if fit_missing not in FIT_MISSING:
        raise ValueError(f"fit_missing must be one of {list(FIT_MISSING)}, got {fit_missing!r}")
    if ensure and save_dir is None:
        raise ValueError("ensure=True looks for a committed revision in the live directory: pass save_dir")
    check_save_format(save_format)
    poll_s = seconds(poll, "poll")
    grace_s = seconds(grace, "grace")
    budget_s = seconds(budget, "budget")
    timeout_s = seconds(timeout, "timeout")
    if poll_s is None or poll_s <= 0:
        raise ValueError(f"poll must be a duration above 0, got {poll!r}")
    if wait_for is not None and isinstance(data, pd.DataFrame):
        raise ValueError(
            "wait_for polls the data by refreshing it: pass a DataPipeline, not a DataFrame (a frame is used as given)"
        )
    if wait_for is not None and wait_for != "rules" and not isinstance(wait_for, Mapping):
        raise ValueError(f"wait_for is a mapping of column -> give-up time ('HH:MM'), or 'rules', got {wait_for!r}")
    pipeline._check_model_names()
    pipeline._check_availability()
    names = [model.name for model in pipeline.models] + [ensemble.name for ensemble in pipeline.ensembles]
    for ensemble in pipeline.ensembles:
        if ensemble.needs_validation:
            raise ValueError(
                f"{ensemble.name} chooses its members on validation (top=...): a live call can't. Freeze its choice first: "
                "pipeline.frozen(save_dir_of_the_study).forecast(...)"
            )
    from ..models.residual import ResidualModel

    for model in pipeline.models:
        if isinstance(model, ResidualModel) and not isinstance(model.warmup, str):
            raise ValueError(
                f"{model.name}: a ResidualModel with warmup={model.warmup!r} starts its base's forecasts where its run starts, "
                "so each forecast depends on the backtest's first issue day and no live forecast can equal one. Give it a "
                "dated warm-up, warmup='YYYY-MM-DD', the day its base's forecasts start on, before it runs live."
            )
    pipeline._resolve_bases()  # a named base (`base="a"`) is the pipeline's model of that name
    pipeline._ordered_ensembles()  # members that exist, no cycle
    delivered = deliver if deliver is not None else names[-1]
    if delivered not in names:
        raise ValueError(f"deliver={deliver!r} names no model or ensemble of the pipeline: {names}")
    horizon, horizons = resolve_horizons(horizon, horizons, 7)
    weekdays = resolve_issue_days(issue_days)
    day_names = issue_day_names(weekdays) if weekdays is not None else None
    info = pipeline.information_set
    zone = issue_zone(info, data)
    day = resolve_issue(issue, zone)
    if weekdays is not None and day.weekday() not in weekdays:
        raise ValueError(f"{day} is a {day.strftime('%A')}, not an issue day of {day_names}")
    deadline_at = instant_on(day, deadline, zone, "deadline")
    formats = {model.name: save_format for model in pipeline.models}
    formats.update({ensemble.name: "jsonl" for ensemble in pipeline.ensembles})  # an ensemble's records are JSONL
    if span is None and _is_pipeline(data):
        span = data_span(pipeline.models, day, horizons, data)
    options = dict(cache=cache, budget=budget_s, timeout=timeout_s, retries=retries, on_failure=on_failure)
    source = _Data(data, span, options, refresh, deadline_at)
    store = None if save_dir is None else LiveStore(save_dir)

    def result(status: str, **fields: Any) -> ForecastResult:
        return ForecastResult(
            issue=day.isoformat(),
            status=status,
            exit_code=exit_code(status),
            dry_run=dry_run,
            delivered=delivered,
            freq=freq,
            **fields,
        )

    if store is None:
        return _issue(pipeline, source, None, day, zone, deadline_at, call, started, result, locals_of(locals()))
    try:
        with store.hold(create=not dry_run):
            return _issue(pipeline, source, store, day, zone, deadline_at, call, started, result, locals_of(locals()))
    except LiveBusyError:
        return result(BUSY)


def locals_of(scope: Dict[str, Any]) -> Dict[str, Any]:
    """The call's settings `_issue` reads, by name."""
    keys = (
        "wait_for",
        "deadline",
        "ensure",
        "grace_s",
        "poll_s",
        "fit_missing",
        "dry_run",
        "target",
        "horizon",
        "horizons",
        "freq",
        "weekdays",
        "day_names",
        "info",
        "formats",
        "delivered",
        "names",
        "save_format",
    )
    return {key: scope[key] for key in keys}


def _issue(
    pipeline: Any,
    source: _Data,
    store: Optional[LiveStore],
    day: datetime.date,
    zone: str,
    deadline_at: Optional[pd.Timestamp],
    call: str,
    started: pd.Timestamp,
    result: Any,
    settings: Dict[str, Any],
) -> ForecastResult:
    """Steps 2-8 of the module docstring, under the lock when there is a live directory."""
    clock = current_clock()
    dry_run = settings["dry_run"]
    issue = day.isoformat()
    writes = store is not None and not dry_run
    revision = None
    try:
        if store is not None and store.is_live():
            recorded = store.zone()
            if recorded is not None and recorded != zone:
                raise ValueError(
                    f"{store.save_dir} issues in {recorded}, this pipeline in {zone}: use another save_dir for it"
                )
        # 2. ensure: an issue with a committed revision, ok or fallback, by the deadline is done.
        if settings["ensure"] and store is not None and store.is_live():
            for number, commit in sorted(store.committed(issue).get(issue, {}).items(), reverse=True):
                limit = deadline_at if deadline_at is not None else commit.deadline
                status = commit.line.get("status")
                if status in (OK, FALLBACK) and (limit is None or commit.finished <= limit):
                    return result(status, revision=number, noop=True)
        if deadline_at is not None and clock.now() > deadline_at:
            return _late(store, writes, issue, call, started, deadline_at, result)
        if writes:
            identity: Dict[str, Any] = {"issue_timezone": zone}
            if isinstance(settings["deadline"], str) and _HH_MM.match(settings["deadline"]):
                identity["deadline"] = settings["deadline"].strip()
            if pipeline.frozen_choices:
                identity["frozen_choices"] = pipeline.frozen_choices
            store.init(**identity)

        # 3. refresh and poll.
        info = settings["info"]
        horizons, freq, target = settings["horizons"], settings["freq"], settings["target"]
        wait_for = settings["wait_for"]
        if wait_for is None:
            frame, polls = source.frame(), []
        elif wait_for == "rules":

            def give_ups(frame: pd.DataFrame) -> Dict[str, pd.Timestamp]:
                columns = read_columns(
                    pipeline.models, frame, day, target, horizons, freq, info.in_data_zone(frame.index)
                )
                return rule_give_ups(frame, columns, info.in_data_zone(frame.index), day, freq, settings["grace_s"])

            frame, polls = wait(source, {}, day, zone, settings["poll_s"], deadline_at, columns_of=give_ups)
        else:
            give = {column: instant_on(day, when, zone, f"wait_for[{column!r}]") for column, when in wait_for.items()}
            frame, polls = wait(source, give, day, zone, settings["poll_s"], deadline_at)
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError(
                "forecast() needs a frame whose index holds instants (a time zone): the issue day is local"
            )
        data_zone = str(frame.index.tz)
        run_info = pipeline._information_set_for(frame)

        # 4-5. plan and serve each model.
        with pool_native_threads():
            held, made, lists, manifests = serve_models(
                pipeline,
                frame,
                day,
                None if store is None or not store.is_live() else store.save_dir,
                target=target,
                horizon=settings["horizon"],
                horizons=horizons,
                freq=freq,
                info=run_info,
                issue_days=settings["day_names"],
                fit_missing=settings["fit_missing"],
            )
        # 6. checks before export, then the ensembles.
        checks_record = None
        if pipeline.checks is not None:
            checked = check_records(held, pipeline.checks, planned_periods(freq, data_zone), freq=freq, zone=data_zone)
            held = checked.records
            checks_record = checked.to_dict(pipeline.checks)
        ensembles = compute_ensembles(pipeline, held, day, horizons, freq, data_zone, settings["weekdays"])
        every = {**held, **ensembles}
        planned = planned_keys(day, horizons, freq, data_zone, settings["weekdays"])
        status, missing, rungs = issue_status(every[settings["delivered"]], planned)

        # The issue's records: the snapshot of its inputs, their quality, the fits.
        # The issue day's rows as the models read them: local dates in the frame's zone (the issue zone's may be another day's).
        snap = records_io.snapshot(frame, list(held), lists, day, zone=data_zone)
        now = clock.now()
        quality = records_io.quality(
            frame, source.last, run_info, day, zone=data_zone, columns=list(snap.inputs.columns), issued_at=now
        )
        quality = {**quality, "issue_timezone": zone, "data_timezone": data_zone}
        fits = {name: entry.get("fits", {}) for name, entry in made.items()}
        outcome = dict(
            records={name: [_strip(record) for record in held_records] for name, held_records in every.items()},
            models=made,
            refresh=source.last,
            waits=polls,
            checks=checks_record,
            snapshot=snap.hash,
            planned=len(planned),
            missing=missing,
            rungs=rungs,
            planned_keys=planned,
            zone=data_zone,
        )

        # 7. late, unchanged, or a new revision.
        if deadline_at is not None and clock.now() > deadline_at:
            return _late(store, writes, issue, call, started, deadline_at, result, **outcome)
        if store is not None and store.is_live():
            latest = latest_committed(store, issue)
            if latest is not None:
                number, commit = latest
                before = committed_records(store, issue, settings["names"], settings["formats"])
                if _canonical(before, issue) == _canonical(every, issue):
                    line = None
                    head_status = commit.line.get("status", status)
                    if writes:
                        line = store.record_call(
                            issue,
                            head_status,
                            exit_code(head_status),
                            revision=number,
                            call=call,
                            started=started,
                            unchanged=True,
                        )
                    return result(head_status, revision=number, unchanged=True, call=line, **outcome)
        if not writes:  # a dry run, or no live directory: what a commit would be
            number = store.next_revision(issue) if store is not None and store.is_live() else (1 if store else None)
            return result(status, revision=number, **outcome)

        for name, manifest in manifests.items():
            if manifest is None:
                continue
            if pipeline.checks is not None:
                manifest.checks = pipeline.checks.to_dict()
            path = pipeline.results_path(store.save_dir, _Named(name), settings["formats"][name])
            check_and_write_live(manifest_path(path), manifest, model_name=name, save_format=settings["formats"][name])
        revision = store.begin(issue, issued_at=clock.now(), snapshot=snap.hash, call=call, started=started)
        for name, held_records in every.items():
            writer = revision.open(name, settings["formats"][name])
            for record in held_records:
                if record.get("status", STATUS_OK) == STATUS_OK:
                    writer.save(record)
                else:
                    writer.save_skip(record)
        directory = revision.directory
        records_io.write_snapshot(snap, directory)
        records_io.write_json(directory, records_io.QUALITY_FILE, quality)
        records_io.write_json(directory, records_io.CHECKS_FILE, checks_record)
        records_io.write_json(directory, records_io.REFRESH_FILE, records_io.refresh_record(source.last))
        records_io.write_json(directory, records_io.WAITS_FILE, records_io.waits_record(polls))
        records_io.write_json(directory, FITS_FILE, {"issue": issue, "models": fits})
        try:
            line = revision.commit(
                status,
                exit_code(status),
                deadline=deadline_at,
                delivered=settings["delivered"],
                planned=len(planned),
                missing=missing,
                issue_timezone=zone,
                data_timezone=data_zone,
            )
        except LiveLateError:
            line = revision.fail(LATE, exit_code(LATE), deadline=deadline_at.isoformat())
            return result(LATE, call=line, **outcome)
        return result(status, revision=revision.revision, call=line, **outcome)
    except BaseException as exc:
        if writes:
            _record_error(store, revision, issue, call, started, exc)
        raise


class _Named:
    """A name where `ModelPipeline.results_path` reads a model's."""

    def __init__(self, name: str):
        self.name = name


def _late(
    store: Optional[LiveStore],
    writes: bool,
    issue: str,
    call: str,
    started: pd.Timestamp,
    deadline_at: pd.Timestamp,
    result: Any,
    **outcome: Any,
) -> ForecastResult:
    line = None
    if writes:
        line = store.record_call(
            issue, LATE, exit_code(LATE), call=call, started=started, deadline=deadline_at.isoformat()
        )
    return result(LATE, call=line, **outcome)


def _record_error(
    store: LiveStore, revision: Any, issue: str, call: str, started: pd.Timestamp, exc: BaseException
) -> None:
    """The call line of a call that raised (status `error`, exit 1), with the revision it had begun if any; never raises over
    the error itself."""
    text = f"{type(exc).__name__}: {exc}"[:500]
    try:
        if revision is not None and not revision._done:
            revision.fail(ERROR, exit_code(ERROR), error=text)
        else:
            store.record_call(issue, ERROR, exit_code(ERROR), call=call, started=started, error=text)
    except BaseException:  # the error raised is the call's, not this line's
        pass


__all__ = ["forecast", "seconds", "instant_on", "data_span", "LOOKBACK_PAD_DAYS"]
