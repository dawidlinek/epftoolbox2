from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, List, Dict, Callable, Iterable, Sequence, Union, Tuple, Optional

import datetime
import gc
import inspect
import multiprocessing
import os
import sys
import warnings

import numpy as np
import pandas as pd
from rich.progress import Progress

from ..scalers.standard import StandardScaler
from ..scalers.registry import ScalerSpec, scaler_spec
from ..results.store import ResultStore
from ..results.ref import ModelResultRef, split_by_window, warn_off_the_clock
from ..results.availability import LISTING_TIMES, listing_rows, warn_about
from ..results.matching import training_day_rows, warn_about_thin
from ..results.manifest import (
    PAST_THE_FRAME,
    TRAINING_CUT_BY_DATE,
    ManifestMismatchError,
    PlannedForecast,
    FrameHashes,
    build_manifest,
    check_and_write,
    check_stored_lists,
    manifest_path,
    read_windows,
    record_skips,
)
from ..results.windows import TEST, RunWindows, mask_key
from ..results.vintages import vintage_rows
from ..logging import get_logger
from . import _contract
from ._calendar import (
    EXPANDING,
    FREQ_STEPS,
    IssueCalendar,
    NewestTargets,
    REFIT_EPOCH,
    check_issue_days_in,
    check_newest_target,
    check_refit,
    every_horizon,
    listed_days,
    mask_members,
    member_window,
    periods_per_day,
    refit_anchor,
    refit_anchors,
    resolve_horizons,
    resolve_issue_days,
    resolve_training_window,
    training_span,
)
from .context import (
    TRAIN_ON_MATCHING,
    PredictorPlan,
    _Cells,
    build_plan,
    mask_digest,
    takes_ctx,
    warn_never_published_in,
)
from ._worker import (
    NOT_PLANNED,
    SKIP_MISSING_TEST_ROW,
    SKIP_NOT_PUBLISHED,
    _pool_init,
    _run_block,
    _run_day,
    issue_row,
    matching_digest,
    matching_rows,
    training_row_days,
    training_rows,
)
from .._date_utils import is_utc_index, resolve_date, shift_calendar_days, zone_name
from ..information_set import _STEP_MINUTES, InformationSetError, TrainingRows, check_level, frame_first_day, rule_dict

if TYPE_CHECKING:  # pragma: no cover
    from ..information_set import InformationSet

logger = get_logger(__name__)

# Periods of an ordinary day: one model per period. "1D" has one, a row per local date at one time of day.
_FREQ_TO_PERIODS: Dict[str, int] = {freq: periods_per_day(freq) for freq in FREQ_STEPS}

# A model's own fit, and the two methods a refit forecasts from: `_can_refit` requires them to agree.
_REFIT_METHODS = ("_fit_predict", "_fit_model", "_predict_fitted")

# ProcessPoolExecutor refuses more workers than this on Windows.
_WINDOWS_MAX_WORKERS = 61


def _test_days(test_start: str, test_end: str) -> List[datetime.date]:
    first, last = datetime.date.fromisoformat(test_start), datetime.date.fromisoformat(test_end)
    return [first + datetime.timedelta(days=d) for d in range((last - first).days + 1)]


def _usable_cpu_count() -> int:
    """CPUs this process may run on: a SLURM or taskset allocation, not the whole machine."""
    if hasattr(os, "process_cpu_count"):  # Python 3.13+
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


class BaseModel(ABC):
    # Parameters a saved pipeline writes only when not at these values (`ModelPipeline._serialize_component`), so files saved
    # before they existed don't move. A subclass adds its own.
    _WRITTEN_WHEN_SET: Dict[str, Any] = {"newest_target": "d-1", "refit": None}
    # True for a model that fits nothing and forecasts a column's value on the issue row (`NaiveModel`): its tasks read that row
    # alone (`_worker._row_value`), with no training rows, scaler or fit.
    _fits_nothing: bool = False

    def __init__(
        self,
        predictors: Union[List[Union[str, Callable]], Callable],
        training_window: Union[int, str, List[Union[int, str]]] = 365,
        name: str = "Model",
        scaler: ScalerSpec = "standard",
        newest_target: str = "d-1",
        refit: Optional[str] = None,
    ):
        """`newest_target` is the newest target a forecast issued on D trains on: `"d-1"` the target of D-1, whatever the
        information set says, and `"published"` every target its rule says is known by the issue time on D, per period
        (`InformationSet.newest_targets`), which needs an information set with a rule for the target. A row dated D or later
        never trains. See the docs page Time and Calendars.

        `refit` is how often a model is fitted: None fits every forecast on its own, `"kD"` once every k days and `"kMS"` on the
        first of every k-th month, on a calendar counted from Monday 1970-01-05 (`_calendar.refit_anchor`). A forecast issued on
        D then uses the fit of its anchor a, the refit day on or before D: rows cut as for a forecast issued on a, with the
        newest target as of a's issue time, so nothing after a trains, and each forecast predicted on its own from that fit. Only
        a model with `_fit_model` and `_predict_fitted` (OLS, LassoCV, Ridge) can.

        `training_window` is how many days of rows a forecast trains on (`_calendar.training_span`), `"expanding"` every row from
        the frame's first local date, or a list of them, such as `[56, 84, 728, "expanding"]`: each member is fitted on its own
        rows, with its own scaler and binary-column mask, exactly as a model with that window alone, and the forecast is their
        mean in target units, made only when every member's is. A list is sorted (ints ascending, then `"expanding"`), and `[W]`
        is `W` in everything (`_calendar.resolve_training_window`)."""
        # A function of `ctx` is planned per forecast (epftoolbox2.models.context); a `(horizon)` or `(horizon, hour)`
        # function gives one list per (period, horizon).
        self._ctx_fn: Optional[Callable] = predictors if callable(predictors) and takes_ctx(predictors) else None
        if self._ctx_fn is not None:
            self._predictors_fn = None
            self.predictors = []
        elif callable(predictors):
            self._predictors_fn = predictors
            self.predictors = []
        else:
            self._predictors_fn = None
            self.predictors = predictors
        self.training_window = training_window
        self.name = name
        self.scaler = scaler  # validated and normalised in __setattr__
        self.newest_target = newest_target  # validated in __setattr__
        self.refit = refit  # validated in __setattr__

        self._data: Optional[pd.DataFrame] = None
        self._hour_data: Dict[int, pd.DataFrame] = {}
        self._hour_days: Dict[int, np.ndarray] = {}
        self._offset: int = 0
        self._last_day: int = 0  # the day id of the run's last calendar day, where a mask's re-read stops
        self._first_ordinal: int = 0  # the date of day id 0
        self._target: str = ""
        self._freq: str = "1h"
        self._periods_per_day: int = 24
        self._used_predictors: List[str] = []
        self._availability_listing: List[Dict] = []
        self._window_listings: Dict[str, List[Dict]] = {}
        # A run's newest targets with newest_target="published", by issue day and period; None otherwise.
        self._newest: Optional[NewestTargets] = None

    def __setattr__(self, key, value):
        # The scaler is kept as a name or {"name", "params"} dict, also when assigned after construction, so a
        # typo fails here rather than in every worker and a scaler instance can't be dropped by save().
        if key == "scaler":
            value = scaler_spec(value)
        elif key == "training_window":
            # The normalised form stays the attribute, what YAML and the manifest write; the members are what a run fits.
            value, members = resolve_training_window(value)
            super().__setattr__("_members", members)
        elif key == "newest_target":
            value = check_newest_target(value)
        elif key == "refit":
            value = check_refit(value)
            if value is not None and not self._can_refit():
                owners = {name: self._owner_of(name).__name__ for name in _REFIT_METHODS}
                raise ValueError(
                    f"{type(self).__name__} can't refit on a calendar (refit={value!r}): its `_fit_predict` is "
                    f"{owners['_fit_predict']}'s, and refit forecasts from `_fit_model` and `_predict_fitted`, which are "
                    f"{owners['_fit_model']}'s and {owners['_predict_fitted']}'s, so refit=None and refit would fit different "
                    "models. One class must define all three; a subclass that overrides any of them takes refit only if it "
                    "overrides all three, else leave refit=None."
                )
        super().__setattr__(key, value)

    @classmethod
    def _owner_of(cls, name: str) -> type:
        """The class of the model's method resolution order that defines `name`."""
        return next(klass for klass in cls.__mro__ if name in vars(klass))

    @classmethod
    def _can_refit(cls) -> bool:
        """Whether the model fits once and forecasts from the fit (`_fit_model`, `_predict_fitted`), which `refit` needs, and
        that fit is the one `refit=None` makes: one class (not BaseModel) defines all three of `_fit_predict`, `_fit_model` and
        `_predict_fitted`. A subclass of OLSModel that overrides `_fit_predict` alone would be refitted with OLS, and one that
        overrides the other two alone would refit with its own estimator while `refit=None` fits OLS."""
        owners = {cls._owner_of(name) for name in _REFIT_METHODS}
        return len(owners) == 1 and BaseModel not in owners

    @property
    def _model_kwargs(self) -> Dict:
        return {}

    @property
    def _known_only(self) -> bool:
        """Whether a forecast whose input its rule puts after the issue time is no forecast (`not_published`), rather than an
        input the information-set check refuses or lists: a benchmark that is a published forecast (`NaiveModel`)."""
        return False

    def _resolve_day(self, date_str: str, label: str) -> int:
        """The day id of a local date the data has a row on. By date, not by `.loc[date_str]`: on a daily frame stamped at
        midnight pandas reads that as the one row's timestamp rather than its date."""
        day = datetime.date.fromisoformat(date_str).toordinal() - self._first_ordinal
        if not (self._data["day"].to_numpy() == day).any():
            last = self._data.index[-1].date()
            raise ValueError(
                f"'{label}' date '{date_str}' is not in the data "
                f"(data ends at {last}). Fetch data through at least '{date_str}'."
            )
        return int(day)

    def run(
        self,
        data: pd.DataFrame,
        test_start: str = None,
        test_end: str = None,
        target: str = "price",
        horizon: Optional[int] = None,
        freq: str = "1h",
        save_to: str = None,
        forecast_only: bool = False,
        information_set: Optional["InformationSet"] = None,
        allow_manifest_mismatch: bool = False,
        require_manifest: bool = False,
        plan: Optional[PredictorPlan] = None,
        availability: str = "raise",
        warn_listing: bool = True,
        windows: Optional[RunWindows] = None,
        frame_hashes: Optional[FrameHashes] = None,
        cut_rules: Optional["InformationSet"] = None,
        horizons: Optional[Sequence[int]] = None,
        issue_days: Optional[Sequence[str]] = None,
    ) -> ModelResultRef:
        """Forecast every period and horizon the clock has on each day in [test_start, test_end] and on its target day.

        A forecast (period, h, D) trains on the rows of its period dated in `training_span` and forecasts from the last row dated
        D: see `epftoolbox2.models._calendar`.

        `horizon=H` forecasts every horizon from 1 to H, 7 if neither it nor `horizons` is given. `horizons` forecasts the ones it
        lists, such as `[1, 3, 7]` or `range(3, 10)`, and nothing else; giving both raises (`resolve_horizons`). Each forecast
        (period, h, D) reads its own rows whatever else is listed, so its record is what a run with `horizon=max(horizons)`
        gives it, and `range(1, H + 1)` is `horizon=H` in everything. Targets are built for 1 to the largest horizon.

        `issue_days` forecasts on the weekdays it names only, such as `["friday"]` (any case; all seven are every day). Training
        is untouched: every date's row trains, and a list's binary-column mask is read from the window's first calendar day as
        in a run of every day, so a Friday's record is that of a run of every day. A day of another weekday is no issue day:
        nothing is planned on it, missing or skipped. A window without one of these weekdays raises.

        With `information_set`, `availability` says what happens to a predictor known after the issue time (see
        epftoolbox2.information_set): "raise", the default here, refuses it before any fit; "warn" goes on and lists it in
        a warning, the returned ref and the manifest; "none" does neither. It doesn't change a forecast (see
        `ModelPipeline`). `ModelPipeline` defaults to "warn" and passes `warn_listing=False`, having warned for all its
        models at once.

        With `save_to`, a manifest is written next to the results (see epftoolbox2.results.manifest) and,
        if one is already there, compared to this run's configuration before anything is resumed.
        `allow_manifest_mismatch` overwrites a disagreeing manifest instead of raising; `require_manifest`
        raises instead of warning when `save_to` has results but no manifest.

        `plan` is what `ModelPipeline` built for a predictor function of `ctx` before any model ran, for this data,
        test period, horizon and freq; without it the run builds its own. A model with fixed lists has none. If a list has
        `train_on="matching"` the plan also holds the list of each day its forecasts can train on, and this run warns before
        any fit about forecasts with no more such days than predictors.

        `windows` (what `ModelPipeline.run(validation=..., test=...)` resolves to, see `epftoolbox2.results.windows`)
        forecasts the days of its windows in place of [test_start, test_end]. The returned ref covers the whole run and its
        `by_window` holds a ref per window. Its records carry `window`, their `day_in_test` counts from the start of it, and
        each window reads its binary-column masks from its own first day, which leaves every forecast what a plain run over
        that window gives. A plain run into results made with windows is treated as a run of the test window alone, so that
        its records and masks are checked and the windows on file stay recorded.

        `frame_hashes` is what `ModelPipeline.run` shares between its models, for this `data`, so that each column of a frame is
        hashed once per call.

        A model with `newest_target="published"` reads its newest training target from the target's rule in `information_set`.
        `cut_rules` is an information set read for that alone, when `information_set` isn't given: nothing is checked or listed
        with it, and a predictor function doesn't answer through it. `leakage_probe` runs models so. A `known_only` model
        (`NaiveModel`) reads from either which of its forecasts are published by the issue time: the others are `not_published`.
        """
        self._set_freq(freq)
        check_level(availability)
        horizon, horizons = resolve_horizons(horizon, horizons, 7)
        weekdays = resolve_issue_days(issue_days)
        rules = self._cut_rules(information_set, cut_rules)

        if windows is not None:
            test_start, test_end = windows.first_day, windows.last_day
        elif forecast_only:
            test_start = resolve_date(test_start or "today")
            test_end = resolve_date(test_end or "today")
        else:
            if test_start is None or test_end is None:
                raise ValueError("test_start and test_end are required unless forecast_only=True")
            test_start = resolve_date(test_start)
            test_end = resolve_date(test_end)
        if windows is None and save_to:
            on_file = read_windows(save_to)
            if on_file is not None:
                windows = self._as_run_of_test_window(
                    on_file, test_start, test_end, forecast_only, allow_manifest_mismatch
                )

        self._check_issue_days(test_start, test_end, windows, weekdays)

        self._target = target
        # The newest targets of every calendar day, as a run of every day reads them: a list's mask is read on the window's first
        # day, whatever its weekday, and the reach of the training rows mustn't depend on the weekdays.
        calendar_days = self._days_of(test_start, test_end, windows)
        newest = self._newest_targets(data, target, horizon, freq, rules, calendar_days)
        if plan is None:
            plan = self._plan_predictors(
                data,
                target,
                horizon,
                freq,
                information_set,
                test_start,
                test_end,
                availability=availability,
                windows=windows,
                cut_rules=cut_rules,
                horizons=horizons,
                weekdays=weekdays,
            )
        else:
            self._check_plan_is_for(
                plan,
                data,
                horizon,
                freq,
                information_set,
                test_start,
                test_end,
                availability,
                windows,
                newest,
                horizons,
                weekdays,
            )
        expanded = self._check_inputs(
            data,
            target,
            horizon,
            freq,
            information_set,
            test_start,
            test_end,
            plan=plan,
            availability=availability,
            windows=windows,
            cut_rules=cut_rules,
            horizons=horizons,
            weekdays=weekdays,
        )
        listing = self._availability_listing
        window_listings = self._window_listings
        if warn_listing:
            warn_about({self.name: listing}, stacklevel=3)
        used_columns = list(dict.fromkeys([target, *self._used_predictors]))
        # What the archive of each input holds, by its rule: recorded, read by nothing that makes a number.
        vintages = vintage_rows(information_set, freq, self._used_predictors, () if plan is None else plan.queried)

        self._load_periods(data, used_columns, horizon, target)
        self._newest = newest

        first_day = datetime.date.fromisoformat(test_start)
        if windows is None:
            self._offset = self._resolve_day(test_start, "test_start")
            test_end_day = self._last_day = self._resolve_day(test_end, "test_end")
            run_days = [
                day
                for day in range(test_end_day - self._offset + 1)
                if weekdays is None or (first_day + datetime.timedelta(days=day)).weekday() in weekdays
            ]
        else:
            for name, (start, end) in windows.periods:
                self._resolve_day(start, f"{name} start")
                self._resolve_day(end, f"{name} end")
            self._offset = self._resolve_day(test_start, "first day of the run")
            self._last_day = self._offset + (datetime.date.fromisoformat(test_end) - first_day).days
            run_days = [(datetime.date.fromisoformat(day) - first_day).days for day in windows.issue_days(weekdays)]
        n_days = len(run_days)
        test_days = {d: first_day + datetime.timedelta(days=d) for d in run_days}
        # A run of every day calls it as it did before `issue_days`, so a test that stands in for it still binds.
        calendar = (
            IssueCalendar(freq, data.index.tz, horizons)
            if weekdays is None
            else IssueCalendar(freq, data.index.tz, horizons, weekdays)
        )
        absent_slots = {
            (period, (day.date() - first_day).days)
            for period, day in _contract.missing_test_rows(self._data.index, freq, test_days.values())
        }
        self._data = None

        all_tasks = calendar.tasks(first_day, run_days)

        window_days = None if windows is None else self._window_days(windows, first_day, weekdays)
        # What a worker reads a task's day by: its window, days since the window's start, and place among the issue days, which
        # a plan and the newest targets are indexed by. A plain run of every day has no entry, its place being its day.
        run_positions = window_days
        if windows is None and weekdays is not None:
            run_positions = {day: (None, day, position) for position, day in enumerate(run_days)}
        if self.refit is not None:
            # A unit reads its mask from its own rows, in the worker (`_worker._fit_unit`): no window or first day decides it.
            masks = {} if windows is None else {name: {} for name, _ in windows.periods}
        elif windows is not None:
            masks = self._window_masks(plan, windows, horizons, first_day)
        elif plan is not None:
            masks = self._plan_scalable_masks(plan)
        else:
            masks = self._precompute_scalable_masks(horizons)
        stored_windows = None

        store = ResultStore(save_to, test_start=test_start) if save_to else None
        if store and data.index.tz is not None and zone_name(data.index.tz) is None:
            warnings.warn(
                f"{self.name}: the data's time zone {data.index.tz!r} has no name that reads back, so a report() from the files in "
                f"{store.path.parent} won't know which periods a date has on the clock: it plans every period of every day and "
                "judges a whole day by its records. Give the index a named zone, such as tz_convert('Europe/Warsaw').",
                UserWarning,
                stacklevel=2,
            )
        if store:
            frame_hashes = FrameHashes(data) if frame_hashes is None else frame_hashes
            manifest = build_manifest(
                model=self,
                data=data,
                used_columns=used_columns,
                expanded=expanded,
                target=target,
                horizon=horizon,
                horizons=horizons,
                freq=freq,
                test_start=test_start,
                test_end=test_end,
                information_set=information_set,
                predictor_function=None if plan is None else plan.function_record(),
                predictor_lists=None if plan is None else plan.manifest_lists(masks if windows else {TEST: masks}),
                availability=self._manifest_availability(information_set, availability, listing, window_listings),
                windows=windows,
                binary_masks=self._fixed_mask_digests(plan, windows, masks, horizons),
                read_columns=() if plan is None else sorted(plan.queried),
                frame_hashes=frame_hashes,
                training_cut=self._training_cut(rules, target, freq),
                issue_days=weekdays,
                vintages=vintages,
            )

            def check_stored():
                check_stored_lists(
                    manifest_path(store.path),
                    manifest,
                    store.iter_lines(
                        cols=[
                            "hour",
                            "horizon",
                            "run_date",
                            "predictor_list",
                            "matching_days",
                            "window",
                            "actual",
                            "refit_anchor",
                        ]
                    ),
                    self._stored_forecast_lookup(plan, masks, windows, weekdays),
                    allow_mismatch=allow_manifest_mismatch,
                    model_name=self.name,
                    stored_actual=None if windows is None else self._stored_actual_lookup(first_day),
                )

            if plan is not None and windows is None:
                check_stored()
            check_and_write(
                manifest_path(store.path),
                manifest,
                has_existing_results=store.path.exists() and store.path.stat().st_size > 0,
                allow_mismatch=allow_manifest_mismatch,
                require_manifest=require_manifest,
                model_name=self.name,
                frame_hashes=frame_hashes,
                # A plain run of fixed lists has nothing per record to compare, except a refit's anchors.
                check_stored=check_stored if windows is not None or (plan is None and self.refit is not None) else None,
            )
            if availability != "none" and manifest.availability is not None:
                listing = manifest.availability["listing"]  # with what earlier runs listed for other test periods
                window_listings = manifest.availability.get("windows", window_listings)
            vintages = manifest.vintages  # with what earlier runs read for other test periods
            stored_windows = None if manifest.windows is None else RunWindows.from_dict(manifest.windows)
        matching_lists = None
        if plan is not None and plan.uses_matching:
            matching_lists = self._matching_lists(plan)
            absent = {(period, first_day.toordinal() + day) for period, day in absent_slots}
            counts = self._matching_day_counts(plan, target, absent)
            warn_about_thin(
                training_day_rows(self.name, self._matching_uses(plan, counts), matching_lists), stacklevel=2
            )
        pending = store.get_missing(all_tasks) if store else all_tasks
        unpublished = self._unpublished(data, rules, freq, expanded, plan, test_days, run_days, pending)

        skipped: Counter = Counter()
        window_skipped: Dict[str, Counter] = {} if windows is None else {name: Counter() for name, _ in windows.periods}
        skip_examples: Dict[str, str] = {}
        tasks = []
        for task in pending:
            if (task[0], task[2]) in absent_slots:
                skipped[SKIP_MISSING_TEST_ROW] += 1
                if window_days is not None:
                    window_skipped[window_days[task[2]][0]][SKIP_MISSING_TEST_ROW] += 1
                skip_examples.setdefault(SKIP_MISSING_TEST_ROW, f"period {task[0]} on {test_days[task[2]]}")
            elif task in unpublished:
                skipped[SKIP_NOT_PUBLISHED] += 1
                if window_days is not None:
                    window_skipped[window_days[task[2]][0]][SKIP_NOT_PUBLISHED] += 1
                skip_examples.setdefault(
                    SKIP_NOT_PUBLISHED, f"period {task[0]}, horizon {task[1]}, run date {test_days[task[2]]}"
                )
            else:
                tasks.append(task)

        in_memory = None if save_to else []
        try:
            if tasks:
                self._run_tasks(
                    tasks,
                    horizons,
                    test_start,
                    n_days,
                    store,
                    in_memory,
                    skipped,
                    skip_examples,
                    plan,
                    masks,
                    window_days,
                    window_skipped,
                    run_positions,
                    None if weekdays is None else [test_days[day] for day in run_days],
                )
        finally:
            if store:
                store.flush()
            self._cleanup()
        if store:  # not after a crash: what it skipped so far is no count of anything
            record_skips(manifest_path(store.path), manifest, {TEST: skipped} if windows is None else window_skipped)

        if not pending:
            print(f"All {len(all_tasks)} tasks completed")
        produced = len(all_tasks) - len(store.get_missing(all_tasks)) if store else len(in_memory)
        self._report_counts(len(all_tasks), produced, skipped, skip_examples)

        ref = ModelResultRef(
            name=self.name,
            path=store.path if store else None,
            count=len(all_tasks),
            test_start=test_start,
            test_end=test_end,
            horizon=horizon,
            freq=self._freq,
            _results=in_memory,
            produced=produced,
            skipped=dict(skipped),
            predictor_lists=None if plan is None else plan.describe(),
            availability_listing=listing,
            windows=windows,
            matching_lists=matching_lists,
            zone=data.index.tz,
            horizons=None if every_horizon(horizon, horizons) else horizons,
            issue_days=weekdays,
            vintages=vintages,
        )
        if store:
            warn_off_the_clock({self.name: ref}, stacklevel=2)
        if windows is not None:
            ref.by_window = split_by_window(
                ref,
                windows,
                calendar,
                skipped=window_skipped,
                predictor_lists=None if plan is None else self._window_lists(plan, windows),
                listings=window_listings,
                target_before=(stored_windows or windows).target_before,
            )
            empty = [name for name, part in ref.by_window.items() if part.produced == 0]
            if empty:
                warnings.warn(
                    f"{self.name}: no forecasts in the {' and '.join(empty)} window of this run. See forecast_counts().",
                    UserWarning,
                    stacklevel=2,
                )
        return ref

    def _cut_rules(
        self, information_set: Optional["InformationSet"], cut_rules: Optional["InformationSet"]
    ) -> Optional["InformationSet"]:
        """The information set the training cut is read from: `information_set`, or `cut_rules` without one."""
        if information_set is not None and cut_rules is not None and cut_rules is not information_set:
            raise ValueError(f"{self.name}: pass information_set or cut_rules, not both")
        return information_set if information_set is not None else cut_rules

    @staticmethod
    def _check_issue_days(
        test_start: str, test_end: str, windows: Optional[RunWindows], weekdays: Optional[Sequence[int]]
    ) -> None:
        """A ValueError if a window of the run (or its test period) has none of the listed weekdays."""
        for name, (start, end) in windows.periods if windows is not None else [("test", (test_start, test_end))]:
            check_issue_days_in(start, end, weekdays, name)

    def _newest_targets(
        self,
        data: pd.DataFrame,
        target: str,
        horizon: int,
        freq: str,
        rules: Optional["InformationSet"],
        issue_days: Optional[Sequence[datetime.date]],
    ) -> Optional[NewestTargets]:
        """The newest target of each forecast on these issue days, for `newest_target="published"`; None for `"d-1"`, which
        reads no information set. Raises without one, or without a rule for the target, whatever the level: the rule decides
        numbers. `issue_days` None only checks that it could be read.

        With `refit` they are read on the anchors of these days instead, as of the issue time on each anchor: a unit's rows are
        cut there, so a target published between the anchor and the issue day doesn't train (`_newest_of` then takes anchors)."""
        if self.newest_target == "d-1":
            return None
        if rules is None:
            raise InformationSetError(
                f"{self.name}: newest_target='published' trains on every target its rule says is known by the issue time, "
                "so it needs an information set: pass information_set (or issue_time) to the ModelPipeline, or "
                "information_set to run()."
            )
        # The window whose targets are checked under a rule over time: the largest a fit reads (`_reach_window`).
        window = self._reach_window(data, issue_days)
        if issue_days is None:
            rules.newest_targets(target, freq, data.index, [], horizon, window)
            return None
        days = list(issue_days) if self.refit is None else refit_anchors(issue_days, self.refit)
        return NewestTargets(days, rules.newest_targets(target, freq, data.index, days, horizon, window))

    def _training_cut(self, rules: Optional["InformationSet"], target: str, freq: str) -> Dict[str, Any]:
        """What a manifest's `training_cut` says of this run: rows by date, with an `"expanding"` training window the frame's
        first local date its rows start on (`expanding_from`; read after `_load_periods`), with `refit` the rule and its epoch, and
        with `newest_target="published"` what the newest target is read from. The information set isn't compared on resume, but
        these decide numbers."""
        cut = dict(TRAINING_CUT_BY_DATE)
        if EXPANDING in self._members:
            # Where an expanding window's rows start: the frame's first local date, day id 0 (the data hash covers it too).
            cut["expanding_from"] = datetime.date.fromordinal(self._first_ordinal).isoformat()
        if self.refit is not None:
            # Which fit a forecast uses: its anchor depends on the rule and the epoch, and nothing else.
            cut.update(refit=self.refit, refit_epoch=REFIT_EPOCH)
        if self._known_only and rules is not None:
            # Which forecasts exist: the information set isn't compared, but its rules and issue time decide these.
            cut["known_only"] = self._published_by(rules, freq)
        if self.newest_target == "d-1":
            return cut
        rule = rules._resolve_for(target, _STEP_MINUTES[freq]).rule
        return {
            **cut,
            "newest_target": self.newest_target,
            # Without the note and the class, which decide nothing: a list of rules with their periods for a `RuleChain`.
            "target_rule": rule_dict(rule, ("note", "vintage")),
            "issue_time": rules.issue_time,
            "issue_timezone": rules.issue_timezone,
        }

    def _newest_of(self, period: int, day: Any) -> Any:
        """The newest target, relative to each issue day, of these day ids of a run: -1 with `newest_target="d-1"`. With
        `refit` the day ids are anchors, the only days it was read on."""
        if self._newest is None:
            return -1
        return self._newest.at(period, np.asarray(day, dtype=np.int64) + self._first_ordinal)

    def _published_by(self, rules: "InformationSet", freq: str) -> Dict[str, Any]:
        """What decides which forecasts of a `known_only` model exist: the issue time and each input's resolved rule (without the
        note and the vintage class, which decide nothing, as `target_rule`; None for a column without one, which counts as known)."""
        step = _STEP_MINUTES[freq]
        resolved = {}
        for name in self._used_predictors:
            rule = rules._resolve_for(name, step).rule
            resolved[name] = None if rule is None else rule_dict(rule, ("note", "vintage"))
        return {"issue_time": rules.issue_time, "issue_timezone": rules.issue_timezone, "rules": resolved}

    def _unpublished(
        self,
        data: pd.DataFrame,
        rules: Optional["InformationSet"],
        freq: str,
        expanded: Dict[Tuple[int, int], List[str]],
        plan: Optional[PredictorPlan],
        test_days: Dict[int, datetime.date],
        run_days: Sequence[int],
        tasks: Iterable[Tuple[int, int, int]],
    ) -> set:
        """The tasks of a `known_only` model whose one input its rule puts after the issue time on their issue day: no forecast
        existed yet, so the task is `not_published`, at every availability level. Empty for any other model, or without an
        information set. The cell is read through `_Cells.known`, the path of `ctx.published` and `violations`; a column
        without a rule counts as known, as at the level "warn" (at "raise" the check has already refused it, if `unknown`
        says so)."""
        tasks = list(tasks)
        if not self._known_only or rules is None or not tasks:
            return set()
        position = {day: i for i, day in enumerate(run_days)}
        cells = _Cells(data, rules, freq, [test_days[day] for day in run_days], "warn")
        out = set()
        for period, hz, day in tasks:
            at = position[day]
            names = expanded[(period, hz)] if plan is None else plan.lists[int(plan.assignment[(period, hz)][at])]
            if not cells.known(names[0])[period, at]:
                out.add((period, hz, day))
        return out

    def _issue_time_checks(self, information_set: "InformationSet") -> Tuple[Callable, Callable]:
        """`information_set.check` and `information_set.late_uses`, as this model's fixed lists are judged by them. A cell a
        `known_only` model's rule puts after the issue time is no forecast (`_unpublished`), so it is neither refused nor
        listed; a column without a rule still is, as `unknown` and the level say."""
        if not self._known_only:
            return information_set.check, information_set.late_uses

        def check(expanded, target, freq, index, test_start=None, test_end=None, model_name="Model", **kwargs):
            _, unknown = information_set.violations(expanded, target, freq, index, test_start, test_end, **kwargs)
            information_set._report_unknown(unknown, model_name)
            information_set._log_assumptions(model_name)

        def late_uses(*args, **kwargs):
            return [use for use in information_set.late_uses(*args, **kwargs) if use.column.rule is None]

        return check, late_uses

    def _as_run_of_test_window(
        self, on_file: RunWindows, test_start: str, test_end: str, forecast_only: bool, allow_mismatch: bool
    ) -> RunWindows:
        """A plain run into results made with windows is a run of the test window alone, if that is the window on file: it
        can't say anything about the others, and must not replace it without being asked to."""
        period = (test_start, test_end)
        if not allow_mismatch and (on_file.test not in (None, period) or (forecast_only and on_file.test is None)):
            raise ManifestMismatchError(
                f"{self.name}: these results were made with the windows {on_file.periods}, and a run on {period} with no "
                "validation= would change them. Pass validation= and test= to say what the windows are now, use a "
                "different save_dir, or pass allow_manifest_mismatch=True to resume anyway."
            )
        return RunWindows.fixed(None, period)

    def _stored_actual_lookup(self, first_day: datetime.date) -> Callable[[int, int, str], Any]:
        """What a fresh run's `actual` is for a key: the `y` of the worker's row on the issue day.

        `PAST_THE_FRAME` for an issue day after the frame's last day. The frame's and not the period's: a period's last row can
        come before the last day, as the hour skipped on a spring switch does. NaN for a day with no such row, where a fresh run
        has no forecast to compare: a stored number there is no forecast of this frame, such as an old record of a slot the clock
        doesn't have, built from the day before. None if nothing can be said."""
        columns: Dict[Tuple[int, int], np.ndarray] = {}
        last_day = max((int(days[-1]) for days in self._hour_days.values() if len(days)), default=None)

        def lookup(period: int, horizon: int, run_date: str) -> Any:
            day = self._offset + (datetime.date.fromisoformat(run_date) - first_day).days
            days = self._hour_days.get(period)
            name = f"{self._target}_d+{horizon}"
            if days is None or name not in self._hour_data[period].columns:
                return None
            if last_day is not None and day > last_day:
                return PAST_THE_FRAME
            if len(days) == 0 or day < days[0]:
                return None
            row = issue_row(days, day)
            if row is None:
                return float("nan")
            key = (period, horizon)
            if key not in columns:
                columns[key] = self._hour_data[period][name].to_numpy(dtype=np.float64)
            return float(columns[key][row])

        return lookup

    @staticmethod
    def _manifest_availability(
        information_set, availability: str, listing: List[Dict], window_listings: Optional[Dict[str, List[Dict]]] = None
    ) -> Optional[Dict]:
        """What a manifest records about the issue-time check; level "none" adds nothing, as if it had not run."""
        if information_set is None or availability == "none":
            return None
        recorded = {"level": availability, "times": LISTING_TIMES, "listing": listing}
        if window_listings:
            recorded["windows"] = window_listings
        return recorded

    @staticmethod
    def _window_days(
        windows: RunWindows, first_day: datetime.date, weekdays: Optional[Sequence[int]] = None
    ) -> Dict[int, Tuple[str, int, int]]:
        """Each issue day of the run, counted from `first_day` -> (its window, calendar days since the window's start, place
        among the run's issue days). With `weekdays` only the days of those weekdays are issue days."""
        days = {}
        position = 0
        for name, (start, end) in windows.periods:
            first = (datetime.date.fromisoformat(start) - first_day).days
            last = (datetime.date.fromisoformat(end) - first_day).days
            for day in range(first, last + 1):
                if weekdays is not None and (first_day + datetime.timedelta(days=day)).weekday() not in weekdays:
                    continue
                days[day] = (name, day - first, position)
                position += 1
        return days

    def _window_masks(
        self, plan, windows: RunWindows, horizons: Sequence[int], first_day: datetime.date
    ) -> Dict[str, Dict]:
        """Window -> its binary-column masks, as a plain run over that window alone computes them, from its first day and
        no day after its last."""
        masks = {}
        for name, (start, end) in windows.periods:
            offset = self._offset + (datetime.date.fromisoformat(start) - first_day).days
            last = self._offset + (datetime.date.fromisoformat(end) - first_day).days
            if plan is None:
                masks[name] = self._precompute_scalable_masks(horizons, offset, last)
            else:
                masks[name] = self._plan_scalable_masks(plan, plan.days_between(start, end), offset, last)
        return masks

    def _fixed_mask_digests(
        self, plan: Optional[PredictorPlan], windows: Optional[RunWindows], masks: Dict, horizons: Sequence[int]
    ) -> Optional[Dict[str, str]]:
        """The `binary_masks` of a manifest: a fixed list's digests per window in a run with windows, a predictor function's
        none (they stay with each list). A plain run of fixed lists records its digests, under the test window's key, only where
        a column with no value on the first day was read again on a later day and that changed a mask: results scaled before
        that re-read existed have none, and `check_and_write` refuses to resume them (`unrecorded_masks`). Every other plain
        manifest is as it was. A model with `refit` has none: a unit's mask is a function of its rows, which the data hash
        covers, and its records carry their anchors."""
        if plan is not None or self.refit is not None:
            return None
        if windows is not None:
            return self._mask_digests(masks)
        unread = self._precompute_scalable_masks(horizons, self._offset, self._offset)
        if all(np.array_equal(mask, unread[key]) for key, mask in masks.items()):
            return None
        return self._mask_digests({TEST: masks})

    @staticmethod
    def _mask_digests(masks: Dict[str, Dict[Tuple, np.ndarray]]) -> Dict[str, str]:
        """Digests by `mask_key`: masks keyed (period, horizon), or (period, horizon, member) for a list of training windows."""
        return {
            mask_key(window, period, horizon, *member): mask_digest(mask)
            for window, window_masks in masks.items()
            for (period, horizon, *member), mask in window_masks.items()
        }

    @staticmethod
    def _window_lists(plan: PredictorPlan, windows: RunWindows) -> Dict[str, List[Dict]]:
        return {name: plan.describe(plan.days_between(start, end)) for name, (start, end) in windows.periods}

    def _run_tasks(
        self,
        tasks,
        horizons,
        test_start,
        n_days,
        store,
        in_memory,
        skipped,
        skip_examples,
        plan=None,
        masks=None,
        window_days=None,
        window_skipped=None,
        run_positions=None,
        newest_days=None,
    ) -> None:
        # Targets for every horizon from 1 to the largest, whatever is planned: the worker reads the column of `hz - 1`.
        hour_arrays = self._build_hour_arrays(max(horizons))
        if plan is None:
            expanded_preds = self._precompute_expanded_predictors(horizons)
            scalable_masks = self._precompute_scalable_masks(horizons) if masks is None else masks
        else:
            expanded_preds = {list_id: list(names) for list_id, names in enumerate(plan.lists)}
            scalable_masks = masks

        first_day = datetime.date.fromisoformat(test_start)
        # A pool job is a day, or with `refit` a block: the issue days of one anchor, whose units it fits once each.
        tasks_by_day: Dict[Any, list] = {}
        for t in tasks:
            job = t[2] if self.refit is None else refit_anchor(first_day + datetime.timedelta(days=t[2]), self.refit)
            tasks_by_day.setdefault(job, []).append(t)
        days_with_tasks = len({t[2] for t in tasks})

        threads_per_process, n_processes = self._executor_config()
        n_processes = min(n_processes, len(tasks_by_day))  # a worker runs whole days (blocks); more would sit idle
        if sys.platform == "win32":
            n_processes = min(n_processes, _WINDOWS_MAX_WORKERS)

        config = {
            "offset": self._offset,
            "test_start": test_start,
            "training_window": self.training_window,
            "scalable_masks": scalable_masks,
            # A list of training windows: each member fitted on its own rows (`_worker._fit_member`), their forecasts averaged.
            **({"members": self._members} if len(self._members) > 1 else {}),
            # Passed in config rather than _model_kwargs, so subclasses that override _model_kwargs can't drop it.
            "scaler": self.scaler,
            "threads_per_process": threads_per_process,
            "freq": self._freq,
        }
        if plan is not None:
            config["task_lists"] = plan.assignment
            config["list_hashes"] = plan.hashes
            if plan.uses_matching:
                config["matching"] = {
                    "lists": plan.day_lists,
                    "start": self._grid_start(plan),
                    "first_ordinal": self._first_ordinal,
                    "ids": plan.matching_ids,
                }
        if run_positions is not None:
            config["run_days"] = run_positions
        if self._fits_nothing:
            config["fits"] = False  # each task reads its issue row alone (`_worker._row_value`)
        if self.refit is not None:
            # The newest targets were read on the anchors (`_newest_targets`), and a unit looks its own up by date.
            config["refit"] = {"rule": self.refit, "first_ordinal": self._first_ordinal, "newest": self._newest}
        elif self._newest is not None:
            # By a task's place among the issue days: with `issue_days`, among the days of the listed weekdays alone.
            config["newest"] = self._newest.by_position(newest_days)

        mp_context = multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn")
        with ProcessPoolExecutor(
            max_workers=n_processes,
            mp_context=mp_context,
            initializer=_pool_init,
            initargs=(
                hour_arrays,
                expanded_preds,
                config,
                type(self),
                self._model_kwargs,
            ),
        ) as pool:
            run_job = _run_day if self.refit is None else _run_block
            futures = {pool.submit(run_job, day_tasks): day for day, day_tasks in tasks_by_day.items()}
            with Progress() as progress:
                task_id = progress.add_task(
                    f"[cyan]{self.name}",
                    total=n_days,
                    completed=n_days - days_with_tasks,
                )
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        day_results, day_skipped = future.result()
                    except Exception as exc:
                        # Results already saved stay in the store, so a fixed rerun resumes from here.
                        pool.shutdown(wait=True, cancel_futures=True)
                        what = (
                            f"run date {first_day + datetime.timedelta(days=job)}"
                            if self.refit is None
                            else f"the refit block from {job}"
                        )
                        raise RuntimeError(f"{self.name}: forecasting failed for {what}: {exc}") from exc
                    for result in day_results:
                        if store:
                            store.save(result)
                        else:
                            in_memory.append(result)
                    for (hour, hz, day), outcome in day_skipped:
                        reason, _, detail = outcome.partition(": ")
                        skipped[reason] += 1
                        if window_days is not None:
                            window_skipped[window_days[day][0]][reason] += 1
                        where = f"period {hour}, horizon {hz}, run date {first_day + datetime.timedelta(days=day)}"
                        skip_examples.setdefault(reason, f"{where}: {detail}" if detail else where)
                    progress.advance(task_id, len({t[2] for t in tasks_by_day[job]}))

    def _report_counts(self, planned: int, produced: int, skipped: Counter, examples: Dict[str, str]) -> None:
        """Make a shortfall of forecasts loud: a warning with the reasons, an error if nothing was produced."""
        if produced == planned:
            return
        reasons = "; ".join(f"{reason} x{n} (e.g. {examples.get(reason, '?')})" for reason, n in skipped.items())
        if produced == 0 and planned > 0:
            raise RuntimeError(f"{self.name}: no forecasts produced ({planned} planned). Skipped: {reasons}")
        logger.warning(f"{self.name}: produced {produced} of {planned} planned forecasts. Skipped: {reasons}")

    def _set_freq(self, freq: str) -> None:
        if freq not in _FREQ_TO_PERIODS:
            raise ValueError(f"Unsupported freq '{freq}'. Supported: {list(_FREQ_TO_PERIODS)}")
        self._freq = freq
        self._periods_per_day = _FREQ_TO_PERIODS[freq]

    def _check_inputs(
        self,
        data: pd.DataFrame,
        target: str,
        horizon: int,
        freq: str,
        information_set: Optional["InformationSet"] = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        plan: Optional[PredictorPlan] = None,
        availability: str = "raise",
        windows: Optional[RunWindows] = None,
        cut_rules: Optional["InformationSet"] = None,
        horizons: Optional[Sequence[int]] = None,
        weekdays: Optional[Sequence[int]] = None,
    ) -> Dict[Tuple[int, int], List[str]]:
        """Check the data contract before any fit and return the expanded predictors per (period, horizon).

        `horizon` is the largest horizon and `horizons` the ones planned, every one from 1 to `horizon` if not given (see
        `run`). Targets, and the names refused as inputs, go up to `horizon` whatever is planned.

        Raises ValueError on a frequency mismatch, duplicate timestamps, reserved or colliding column names,
        a UTC index with existing day-lag (`_d-<k>`) columns, and missing or non-numeric predictors. `ModelPipeline` calls
        it for every model before running any. The row spacing is checked only on the days this run can read
        (`_contract.read_days`), or on the whole frame without a test window.

        When `information_set` is given, `availability` says what a predictor not known at issue time does: "raise" raises
        InformationSetError (a ValueError), "warn" keeps it for `_availability_listing`, and "none" doesn't look.

        A predictor function of `ctx` is checked through its `plan`, which `_plan_predictors` built after the same data
        contract checks; without one, this builds one to check and drops it.

        `windows` checks the days of its windows, once, in place of [test_start, test_end], and keeps what `availability`
        "warn" lists for each of them in `_window_listings`.

        A model with `newest_target="published"` raises without an information set (or `cut_rules`, see `run`) and without a
        rule for the target, at every level. Its training target isn't checked: what it trains on is known by construction.

        `weekdays` (`resolve_issue_days` of a run's `issue_days`) checks and lists the issue days of those weekdays alone: what a
        forecast of another weekday would read late is no use of this run's.
        """
        self._set_freq(freq)
        horizons = tuple(range(1, horizon + 1)) if horizons is None else tuple(horizons)
        self._availability_listing = []
        self._window_listings = {}
        issued = self._days_of(test_start, test_end, windows)
        newest = self._newest_targets(
            data,
            target,
            horizon,
            freq,
            self._cut_rules(information_set, cut_rules),
            issued,
        )
        self._check_reaches_back(data, test_start, test_end, windows, weekdays, horizons, newest)
        # A model that fits nothing has no training target to check.
        check_target = self.newest_target == "d-1" and not self._fits_nothing
        if self._ctx_fn is not None:
            if plan is None:
                plan = self._plan_predictors(
                    data,
                    target,
                    horizon,
                    freq,
                    information_set,
                    test_start,
                    test_end,
                    availability=availability,
                    windows=windows,
                    cut_rules=cut_rules,
                    horizons=horizons,
                    weekdays=weekdays,
                )
            expanded = plan.names_by_pair()
            self._used_predictors = _contract.check_predictors(data, expanded, target, horizon)
            _contract.warn_about_a_midnight_stamp(
                data.index, freq, self._used_predictors, self._cut_rules(information_set, cut_rules), self.name
            )
            if information_set is not None and availability == "raise":
                plan.check(target, self.name)
            elif information_set is not None and availability == "warn":
                self._availability_listing = plan.listing(target, self.name)
                if windows is not None:
                    self._window_listings = self._plan_window_listings(plan, target, windows)
            plan.release()
            return expanded
        self._check_frame(data, target, horizon, freq, test_start, test_end, windows, newest)
        expanded = self._precompute_expanded_predictors(horizons)
        self._used_predictors = _contract.check_predictors(data, expanded, target, horizon)
        _contract.warn_about_a_midnight_stamp(
            data.index, freq, self._used_predictors, self._cut_rules(information_set, cut_rules), self.name
        )
        days = None if windows is None else windows.issue_days(weekdays)
        if days is None and weekdays is not None and test_start is not None and test_end is not None:
            days = [day.isoformat() for day in listed_days(_test_days(test_start, test_end), weekdays)]
        # With `refit` the newest training target is checked on the anchors, where a unit's rows are cut.
        on_anchors = check_target and self.refit is not None
        check_target = check_target and self.refit is None
        window = self._reach_window(data, issued)
        rows = self._training_rows(newest, window, frame_first_day(data.index))
        if information_set is not None:
            check, late_uses = self._issue_time_checks(information_set)
        if information_set is not None and availability == "raise":
            check(
                expanded,
                target,
                freq,
                data.index,
                test_start,
                test_end,
                model_name=self.name,
                issue_days=days,
                check_target=check_target,
                training_rows=rows,
            )
            if on_anchors:
                self._target_on_anchors(
                    information_set.check, expanded, target, freq, data.index, test_start, test_end, days, window
                )
        elif information_set is not None and availability == "warn":
            uses = late_uses(
                expanded,
                target,
                freq,
                data.index,
                test_start,
                test_end,
                model_name=self.name,
                issue_days=days,
                check_target=check_target,
                training_rows=rows,
            )
            if on_anchors:
                uses += self._target_on_anchors(
                    information_set.late_uses, expanded, target, freq, data.index, test_start, test_end, days, window
                )
            self._availability_listing = listing_rows(uses)
            for name, (start, end) in windows.periods if windows is not None else ():
                listed = None if weekdays is None else RunWindows.fixed(None, (start, end)).issue_days(weekdays)
                uses = late_uses(
                    expanded,
                    target,
                    freq,
                    data.index,
                    start,
                    end,
                    model_name=self.name,
                    check_target=check_target,
                    training_rows=rows,
                    **({} if listed is None else {"issue_days": listed}),
                )
                if on_anchors:
                    uses += self._target_on_anchors(
                        information_set.late_uses, expanded, target, freq, data.index, start, end, listed, window
                    )
                self._window_listings[name] = listing_rows(uses)
        return expanded

    def _check_reaches_back(
        self,
        data: pd.DataFrame,
        test_start: Optional[str],
        test_end: Optional[str],
        windows: Optional[RunWindows],
        weekdays: Optional[Sequence[int]],
        horizons: Sequence[int],
        newest: Optional[NewestTargets],
    ) -> None:
        """With `refit`, a ValueError if the frame starts after the earliest row a unit of the run trains on.

        A unit (p, h, a) trains on the rows dated in `training_span(a, W, h, k)`, k its newest target read on the anchor (-1 with
        `newest_target="d-1"`), and the first anchor can be up to a year before the test starts. The earliest first row over the
        planned anchors, periods and horizons is the day the frame must start on, exactly: a frame starting a day later drops a
        row some unit trains on. A frame sized for `refit=None` (from test_start - W - h) would train that block on fewer days,
        silently: its forecasts move and a comparison keeps identical keys and counts. So that frame is refused, naming the day.
        A matching list filters the same span, so it needs no more. Without `refit` nothing is checked, as before.

        With a list of training windows each member's units are checked, which the largest int member decides; an `"expanding"`
        member trains on whatever the frame has from its first date, so it asks for nothing."""
        widths = [member for member in self._members if member != EXPANDING]
        if self.refit is None or not widths:
            return
        if windows is not None:
            issued = windows.issue_days(weekdays)
        elif test_start is None or test_end is None:
            return
        else:
            issued = [day.isoformat() for day in listed_days(_test_days(test_start, test_end), weekdays)]
        if not issued or not len(data.index):
            return
        first = min(datetime.date.fromisoformat(day) for day in issued)
        anchors = np.array([day.toordinal() for day in refit_anchors(issued, self.refit)], dtype=np.int64)
        earliest = []
        for period in range(self._periods_per_day) if newest is not None else [None]:
            newest_of = -1 if newest is None else newest.at(period, anchors)
            for horizon in horizons:
                first_rows, _ = training_span(anchors, max(widths), int(horizon), newest_of)
                earliest.append(int(np.min(first_rows)))
        needed = datetime.date.fromordinal(min(earliest))
        starts = _contract.wall_clock(data.index).min().date()
        if starts > needed:
            raise ValueError(
                f"{self.name}: with refit={self.refit!r} the forecasts issued from {first} on are fitted on their refit days, the "
                f"first {refit_anchor(first, self.refit)}, and the earliest row they train on is dated {needed}, but the frame starts "
                f"on {starts}: that block would train on fewer days. Start the frame on {needed} or earlier, or the test later."
            )

    def _target_on_anchors(
        self,
        method: Callable,
        expanded: Dict[Tuple[int, int], List[str]],
        target: str,
        freq: str,
        index: pd.DatetimeIndex,
        test_start: Optional[str],
        test_end: Optional[str],
        issue_days: Optional[Sequence[str]],
        window: Optional[int] = None,
    ) -> Any:
        """`method` (an information set's `check` or `late_uses`) of the newest training target alone, for a model with `refit`:
        on the anchors of the issue days (`issue_days`, else every day from `test_start` to `test_end`), since a unit's rows are
        cut as of the issue time on its anchor. Without days to read anchors from, on every day of the frame, as before.
        `window` is the one a fit reads back (`_reach_window`), the model's int window if not given; no target before the frame's
        first local date is judged, having no row there."""
        boundary = {pair: [] for pair in expanded}
        # The targets of a unit's rows, back from its anchor.
        rows = TrainingRows(max(self._members) if window is None else window, first_day=frame_first_day(index))
        if issue_days is None and (test_start is None or test_end is None):
            return method(boundary, target, freq, index, test_start, test_end, model_name=self.name, training_rows=rows)
        issued = issue_days if issue_days is not None else [day.isoformat() for day in _test_days(test_start, test_end)]
        anchors = [day.isoformat() for day in refit_anchors(issued, self.refit)]
        return method(boundary, target, freq, index, model_name=self.name, issue_days=anchors, training_rows=rows)

    def _training_rows(
        self, newest: Optional[NewestTargets], window: Optional[int] = None, first_day: Optional[str] = None
    ) -> TrainingRows:
        """The rows a fit of this model reads (`TrainingRows`), for checking a column whose rule changes over time on them too:
        the window, a newest target before D - 1 under `"published"`, and with `refit` the refit day, whose fit a forecast uses:
        its rows are those of a forecast issued on it (`training_span(a, W, h, k(a))`), as of the issue time on D. `window` is the
        one a fit reads back (`_reach_window`), the model's int window if not given; `first_day` the frame's first local date
        (`frame_first_day`), before which no row exists to judge."""
        return TrainingRows(
            max(self._members) if window is None else window,
            reach=0 if newest is None else newest.reach,
            to_issue=self.newest_target == "published",
            refit=self.refit,
            first_day=first_day,
        )

    def _reach_window(self, data: pd.DataFrame, issue_days: Optional[Sequence[Any]] = None) -> int:
        """The window of the rows a fit of this model can read, as one int, for whatever asks which days a fit reads (the checks
        of rules over time, a predictor function's plan): the model's window, or with a list the largest int member. With an
        `"expanding"` member at least the days from the frame's first local date to the last of `issue_days` (the frame's last
        date if not given), plus one, which puts every span's start before the frame. What selects rows takes each member's
        own (`member_window`)."""
        widths = [member for member in self._members if member != EXPANDING]
        if EXPANDING not in self._members:
            return max(widths)
        local = _contract.wall_clock(data.index)
        if not len(local):
            return max(widths, default=1)
        days = [pd.Timestamp(day).date() for day in issue_days] if issue_days else [local.max().date()]
        return max([*widths, (max(days) - local.min().date()).days + 1])

    @staticmethod
    def _days_of(
        test_start: Optional[str], test_end: Optional[str], windows: Optional[RunWindows]
    ) -> Optional[List[datetime.date]]:
        """The issue days of a run, or None if it doesn't say."""
        if windows is not None:
            return [datetime.date.fromisoformat(day) for day in windows.issue_days()]
        if test_start is None or test_end is None:
            return None
        return _test_days(test_start, test_end)

    def _plan_window_listings(self, plan: PredictorPlan, target: str, windows: RunWindows) -> Dict[str, List[Dict]]:
        """What each window uses. A run of one window uses what the whole plan does, whether or not its cells are gone."""
        if len(windows.periods) == 1:
            return {windows.periods[0][0]: plan.listing(target, self.name)}
        return {
            name: plan.listing(target, self.name, plan.days_between(start, end))
            for name, (start, end) in windows.periods
        }

    def _check_frame(
        self, data, target, horizon, freq, test_start=None, test_end=None, windows=None, newest=None
    ) -> None:
        reach = 0 if newest is None else newest.reach
        # The days a fit can read go back by the largest member; an `"expanding"` one reads the whole frame, which `read_days`
        # can't bound, so the spacing checks cover it all.
        window = EXPANDING if EXPANDING in self._members else max(self._members)
        if windows is None:
            days = _contract.read_days(self._fit_start(test_start), test_end, window, horizon, reach)
        else:
            periods = [(self._fit_start(start), end) for _, (start, end) in windows.periods]
            days = _contract.read_ranges(periods, window, horizon, reach)
        _contract.check_index(data, freq, days)
        _contract.check_timezone_with_day_lags(data)
        _contract.check_names(data, target, horizon)

    def _fit_start(self, start: Optional[str]) -> Optional[str]:
        """The first day a forecast issued from `start` on is fitted on: `start`, or with `refit` its anchor, from which a
        unit's rows are cut, so that the rows a run reads (`_contract.read_days`) go back from it."""
        if self.refit is None or start is None:
            return start
        try:
            return refit_anchor(datetime.date.fromisoformat(start), self.refit).isoformat()
        except (TypeError, ValueError):  # `read_days` says what it can't read
            return start

    def _fit_days(self, days: Any) -> Any:
        """The day id each of these issue day ids is fitted on: itself, or with `refit` its anchor."""
        if self.refit is None:
            return days
        ordinals = np.atleast_1d(np.asarray(days, dtype=np.int64)) + self._first_ordinal
        anchors = np.array(
            [refit_anchor(datetime.date.fromordinal(int(o)), self.refit).toordinal() for o in ordinals], dtype=np.int64
        )
        anchors -= self._first_ordinal
        return int(anchors[0]) if np.ndim(days) == 0 else anchors

    def _plan_predictors(
        self,
        data: pd.DataFrame,
        target: str,
        horizon: int,
        freq: str,
        information_set: Optional["InformationSet"] = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        issue_days: Optional[Sequence[str]] = None,
        warn_unpublished: bool = True,
        availability: str = "raise",
        windows: Optional[RunWindows] = None,
        cut_rules: Optional["InformationSet"] = None,
        horizons: Optional[Sequence[int]] = None,
        weekdays: Optional[Sequence[int]] = None,
    ) -> Optional[PredictorPlan]:
        """The plan of a predictor function of `ctx` for this data and test period, or None for a model with fixed lists.

        `issue_days` plans these days instead of [test_start, test_end]: `verify` samples a few to learn the inputs.
        `windows` plans the days of a run with validation, once each and none between them; the function doesn't see them.
        `availability` is the level `ctx` answers at (see `epftoolbox2.models.context`). `cut_rules` is as in `run`.
        `horizons` are the horizons planned, every one from 1 to `horizon` (the largest) if not given. `weekdays` plans the days of
        those weekdays alone (a run's `issue_days`), not with `issue_days`, which are already the days to plan.
        """
        if self._ctx_fn is None:
            return None
        self._set_freq(freq)
        if windows is not None:
            calendar_days = [datetime.date.fromisoformat(day) for day in windows.issue_days()]
            days = listed_days(calendar_days, weekdays)
        elif issue_days is not None:
            days = calendar_days = [datetime.date.fromisoformat(day) for day in issue_days]
        elif test_start is None or test_end is None:
            raise ValueError(f"{self.name}: a predictor function of `ctx` needs the test period to plan its lists")
        else:
            calendar_days = _test_days(test_start, test_end)
            days = listed_days(calendar_days, weekdays)
        # Of every calendar day, as `run` reads them, so that the reach of a matching list's training days is that run's.
        rules = self._cut_rules(information_set, cut_rules)
        newest = self._newest_targets(data, target, horizon, freq, rules, calendar_days)
        self._check_frame(data, target, horizon, freq, test_start, test_end, windows, newest)
        plan = build_plan(
            self._ctx_fn,
            data,
            days,
            horizon,
            freq,
            information_set,
            warn_unpublished,
            availability,
            # The model's int window, which a matching list plans its training days for; else the reach of every member.
            training_window=self._reach_window(data, calendar_days),
            newest_target=self.newest_target,
            reach=0 if newest is None else newest.reach,
            horizons=horizons,
            refit=self.refit,
            known_only=self._known_only,
            trains=not self._fits_nothing,
            matching_refused=None
            if isinstance(self.training_window, int)
            else (
                f"{self.name}: train_on='matching' isn't supported yet with training_window={self.training_window!r}: the "
                "matching days, their counts and the resume checks are kept for one int window. Use an int training_window "
                "for this model, or train_on='all'."
            ),
        )
        if windows is not None and warn_unpublished and len(windows.periods) > 1:
            for name, (start, end) in windows.periods:
                warn_never_published_in(plan, name, plan.days_between(start, end))
        return plan

    def _check_plan_is_for(
        self,
        plan,
        data,
        horizon,
        freq,
        information_set,
        test_start,
        test_end,
        availability="raise",
        windows=None,
        newest=None,
        horizons=None,
        weekdays=None,
    ) -> None:
        """A plan answers for one function, frame and test period; on another run it would give wrong lists silently."""
        if plan.function is not self._ctx_fn:
            raise ValueError(f"{self.name}: this plan was built for another predictor function")
        problems = []
        if plan.data is not data:
            problems.append("data")
        planned_days = listed_days(self._days_of(test_start, test_end, windows), weekdays)
        if plan.issue_days != planned_days:
            problems.append("test period or issue days")
        if plan.horizon != horizon:
            problems.append("horizon")
        elif plan.horizons != (tuple(range(1, horizon + 1)) if horizons is None else tuple(horizons)):
            problems.append("horizons")
        if plan.freq != freq:
            problems.append("freq")
        if information_set is not None and plan.information_set is not information_set:
            problems.append("information set")
        if information_set is not None and plan.availability != availability:
            problems.append("availability level")
        if plan.uses_matching and plan.training_window != self.training_window:
            problems.append("training window")
        if plan.newest_target != self.newest_target or (
            plan.uses_matching and plan.reach != (0 if newest is None else newest.reach)
        ):
            problems.append("newest target")
        if plan.refit != self.refit:
            problems.append("refit")
        if problems:
            raise ValueError(f"{self.name}: this plan was built for another {', '.join(problems)}")

    def _plan_scalable_masks(
        self,
        plan: PredictorPlan,
        days: Tuple[int, Optional[int]] = (0, None),
        offset: Optional[int] = None,
        last: Optional[int] = None,
    ) -> Dict[Tuple[int, int, int], np.ndarray]:
        """(list id, period, horizon) -> which of the list's columns are scaled, for each list a planned forecast uses.

        All of them, pending or not, so a resumed run computes what a fresh one does. `days` is a (start, stop) slice of the
        plan's issue days, `offset` the day id of the first calendar day of the window they are in and `last` that of its last
        (the run's if not given), when masks are kept per window: what happens in another window then can't move them.

        Only planned keys count: what the function returns where the clock doesn't have the period on the issue day or the
        target day (`PredictorPlan.planned`) is no forecast's, so it decides no mask and a list returned only there gets none.
        A list's mask day is the window's first calendar day if the list is that of the window's first planned issue day of the
        (period, horizon), whatever that day's weekday, as a fixed list's is, else the first planned issue day that uses it. It
        is no later than any of the list's forecasts and depends on nothing after it: not on later issue days, their answers or
        the end of the test period. So a function returning one list on every issue day, or on Fridays alone in a run of
        Fridays, is identical to that fixed list. Issue days are the plan's, by date: with `issue_days` they aren't consecutive.
        A column with no value on the mask day is read again as `_scalable_mask` says.

        A model with a list of training windows has a mask per member, keyed (list id, period, horizon, member), each read with
        the member's window exactly as the model with that window alone reads it, so a member is that model to the bit.
        """
        offset = self._offset if offset is None else offset
        last = self._last_day if last is None else last
        day_ids = np.array([day.toordinal() - self._first_ordinal for day in plan.issue_days], dtype=np.int64)
        planned = plan.planned()
        masks = {}
        for (hour, hz), all_ids in plan.assignment.items():
            on_clock = planned[(hour, hz)][days[0] : days[1]]
            ids = all_ids[days[0] : days[1]][on_clock]
            ids_days = day_ids[days[0] : days[1]][on_clock]
            for list_id in np.unique(ids).tolist():
                first = offset if ids[0] == list_id else int(ids_days[np.argmax(ids == list_id)])
                for member in mask_members(self.training_window):
                    key = (list_id, hour, hz) if member is None else (list_id, hour, hz, member)
                    masks[key] = self._scalable_mask(hour, hz, list(plan.lists[list_id]), first, last, member)
        return masks

    def _scalable_mask(
        self, hour: int, hz: int, predictors: List[str], first: int, last: int, member: Any = None
    ) -> np.ndarray:
        """Which of `predictors` are scaled at (period `hour`, horizon `hz`), read from the training rows of day id `first`,
        the mask day, by `StandardScaler.get_scalable_mask`. The rows are those of the training window `member` (the model's one
        window if None), so a member of a list reads its mask as the model with that window alone does.

        A column with no value in those rows can't tell whether it is binary. It is read on the first calendar day d after
        `first`, up to `last`, whose training rows have a value in it, and the scan stops there: for every list, fixed or a
        function's, however often it is used, and over calendar days, not issue days, so that a run of Fridays reads it where a
        run of every day does. No forecast issued on D reads past D through it. The mask day is D or earlier. A forecast of the
        list with a finite training row has a value of the column in its own rows, so d <= D, and d is decided by the rows of
        the days up to d, dated d - 1 - h or earlier (d - 1 at the latest with `newest_target="published"`). A forecast issued
        before d has no value of the column in its rows, so no finite row, and is skipped whatever its mask says.
        """
        member = self._members[0] if member is None else member
        sample = self._training_sample(hour, first, hz, predictors, member)
        mask = StandardScaler.get_scalable_mask(sample)
        without = np.flatnonzero(np.isnan(sample).all(axis=0))
        if not len(without) or last <= first:
            return mask
        days = self._hour_days[hour]
        values = self._hour_data[hour][[predictors[i] for i in without]].to_numpy(dtype=np.float64)
        # The scan stops at the window's last day. That moves no forecast: one with a finite row finds d at its issue day or
        # before (d <= D <= last), and a column found only after `last` belongs to forecasts with no finite row, which are
        # skipped. It keeps the mask of a window from reading another window's days, and from the end of the data.
        for day in range(first + 1, last + 1):
            rows = values[training_rows(days, day, member_window(member, day), hz, self._newest_of(hour, day))]
            found = ~np.isnan(rows).all(axis=0)
            if found.any():
                mask[without[found]] = StandardScaler.get_scalable_mask(rows[:, found])
                without, values = without[~found], values[:, ~found]
                if not len(without):
                    break
        return mask

    def _load_periods(self, data: pd.DataFrame, used_columns: Sequence[str], horizon: int, target: str) -> None:
        """Read `data` into the frames of the periods, whose rows the tasks are cut from."""
        self._data = self._preprocess(data[list(used_columns)], horizon, target)
        self._first_ordinal = self._data.index[0].date().toordinal()
        for hour in range(self._periods_per_day):
            hour_df = self._data[self._data["hour"] == hour].copy()
            self._hour_data[hour] = hour_df
            self._hour_days[hour] = hour_df["day"].values

    def _grid_start(self, plan: PredictorPlan) -> int:
        """The day id of the first day of `plan.day_lists`, which may be before the frame's."""
        return plan.first_grid_day.toordinal() - self._first_ordinal

    @staticmethod
    def _matching_lists(plan: PredictorPlan) -> Dict[str, int]:
        """The hash of each list with `train_on="matching"` -> how many predictors it has."""
        return {plan.hashes[list_id]: len(plan.lists[list_id]) for list_id in sorted(plan.matching_ids)}

    def _matching_days_of(self, plan: PredictorPlan, period: int, horizon: int, run_date: str, list_id: int) -> str:
        """The record's `matching_days` of a forecast: the worker's own cut and filter, on the rows the worker has."""
        day = self._fit_days(datetime.date.fromisoformat(run_date).toordinal() - self._first_ordinal)
        rows = training_row_days(
            self._hour_days[period], day, self.training_window, horizon, self._newest_of(period, day)
        )
        lists = plan.day_lists[(period, horizon)]
        return matching_digest(rows[matching_rows(rows, lists, self._grid_start(plan), list_id)], self._first_ordinal)

    def _matching_day_counts(
        self, plan: PredictorPlan, target: str, absent: Iterable[Tuple[int, int]] = ()
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """(period, horizon) of a matching forecast -> the days each issue day's forecast would train on, before any fit, -1
        if it isn't one of a matching list, isn't planned or has no row to forecast from. They are what its record's `training_days`
        will say, so after dropping rows with a missing value; a list holding a column the frame lacks can't say, and keeps -1.

        `absent` has (period, date ordinal) of what the run skips for lack of a row.
        """
        absent = set(absent)
        # The days the forecasts are fitted on: the issue days, or with `refit` their anchors.
        issue_days = self._fit_days(
            np.array([day.toordinal() - self._first_ordinal for day in plan.issue_days], dtype=np.int64)
        )
        start = self._grid_start(plan)
        finite: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
        counts: Dict[Tuple[int, int], np.ndarray] = {}
        for (period, horizon), lists in plan.day_lists.items():
            ids = plan.assignment[(period, horizon)]
            frame = self._hour_data[period]
            days = self._hour_days[period]
            on_grid = (days >= start) & (days < start + len(lists))
            row_lists = np.full(len(days), NOT_PLANNED, dtype=np.int32)
            row_lists[on_grid] = lists[days[on_grid] - start]
            finite_y = np.isfinite(frame[f"{target}_d+{horizon}"].to_numpy(dtype=np.float64))
            # The rows of this period dated in the span of each issue day.
            if self._newest is None:
                first_day, last_day = training_span(issue_days, self.training_window, horizon)
            else:
                newest = self._newest_of(period, issue_days)
                first_day, last_day = training_span(issue_days, self.training_window, horizon, newest)
            first_row = np.searchsorted(days, first_day, side="left")
            last_row = np.searchsorted(days, last_day, side="right")
            out = counts[(period, horizon)] = np.full(len(ids), -1, dtype=np.int64)
            for list_id in plan.matching_ids.intersection(np.unique(ids).tolist()):
                key = (period, list_id)
                if key not in finite:
                    names = list(plan.lists[list_id])
                    finite[key] = (
                        np.isfinite(frame[names].to_numpy(dtype=np.float64)).all(axis=1)
                        if set(names) <= set(frame.columns)
                        else None
                    )
                if finite[key] is None:
                    continue
                kept = (row_lists == list_id) & finite_y & finite[key]
                # A doubled autumn row counts once.
                new_day = kept.copy()
                new_day[1:] &= ~((days[1:] == days[:-1]) & kept[:-1])
                total = np.concatenate(([0], np.cumsum(new_day)))
                on = ids == list_id
                out[on] = total[last_row[on]] - total[first_row[on]]
            if absent:
                out[[(period, day.toordinal()) in absent for day in plan.issue_days]] = -1
        calendar = IssueCalendar(plan.freq, plan.data.index.tz, plan.horizons)
        for period, horizon, position in calendar.unplanned(plan.issue_days):
            if (period, horizon) in counts:
                counts[(period, horizon)][position] = -1
        return counts

    @staticmethod
    def _matching_uses(
        plan: PredictorPlan, counts: Dict[Tuple[int, int], np.ndarray]
    ) -> List[Tuple[str, int, int, int]]:
        """(list hash, period, horizon, days it trains on) of each forecast of a matching list that will have a fit."""
        uses = []
        for (period, horizon), days in counts.items():
            fitted = days > 0
            ids = plan.assignment[(period, horizon)][fitted]
            uses.extend(
                (plan.hashes[list_id], period, horizon, count)
                for list_id, count in zip(ids.tolist(), days[fitted].tolist())
            )
        return uses

    def _stored_forecast_lookup(
        self,
        plan: Optional[PredictorPlan],
        masks: Dict,
        windows: Optional[RunWindows] = None,
        weekdays: Optional[Sequence[int]] = None,
    ) -> Callable[[int, int, str], Optional[PlannedForecast]]:
        """What a fresh run would give a stored forecast's key, or None if it isn't planned, as on a weekday `weekdays` doesn't
        list. `masks` are per window in a run with windows. With `refit` that is its anchor and no mask digest: a unit reads its
        mask from its own rows. A model with a list of training windows has a digest per member (`member_masks`)."""
        by_window = {TEST: masks} if windows is None else masks
        digests = {
            window: {key: mask_digest(mask) for key, mask in window_masks.items()}
            for window, window_masks in by_window.items()
        }
        members = mask_members(self.training_window)

        def lookup(period: int, horizon: int, run_date: str) -> Optional[PlannedForecast]:
            window = TEST if windows is None else windows.window_of(run_date)
            if window is None or (weekdays is not None and not listed_days([run_date], weekdays)):
                return None
            list_id = None if plan is None else plan.list_id_of(period, horizon, run_date)
            key = (period, horizon) if plan is None else (list_id, period, horizon)
            each = [digests[window].get(key if member is None else (*key, member)) for member in members]
            digest = each[0]
            anchor = None if self.refit is None else refit_anchor(run_date, self.refit).isoformat()
            if digest is None and (anchor is None or (plan is not None and list_id is None)):
                return None
            matching = plan is not None and plan.train_on[list_id] == TRAIN_ON_MATCHING
            return PlannedForecast(
                None if plan is None else plan.hashes[list_id],
                digest if members == (None,) else None,
                mask_key(window, period, horizon),
                None if windows is None else window,
                self._matching_days_of(plan, period, horizon, run_date, list_id) if matching else None,
                anchor,
                ()
                if members == (None,) or digest is None
                else tuple((mask_key(window, period, horizon, member), found) for member, found in zip(members, each)),
            )

        return lookup

    def _executor_config(self) -> tuple:
        """(threads per process, processes) from THREADS_PER_PROCESS and MAX_PROCESSES.

        Defaults: one task thread per process and one process per usable CPU. Threads in one process share its GIL
        (on, unless a free-threaded build runs with PYTHON_GIL=0), so they are slower than as many processes.
        """
        threads = int(os.environ.get("THREADS_PER_PROCESS", "1"))
        processes = int(os.environ.get("MAX_PROCESSES", max(1, _usable_cpu_count() // threads)))
        return threads, processes

    def _cleanup(self) -> None:
        self._data = None
        self._newest = None
        self._hour_data.clear()
        self._hour_days.clear()
        gc.collect()

    def _preprocess(self, data: pd.DataFrame, horizon: int, target: str) -> pd.DataFrame:
        # The worker finds a period's rows by their date and takes the last of a date, so rows must be in time order.
        # Duplicates and reserved column names were rejected by _check_inputs.
        data = data.sort_index()
        new_cols = {}
        for h in range(1, horizon + 1):
            # Use calendar-day shifts (not elapsed 24h) to keep local wall-clock alignment across DST.
            future_idx = self._shift_index_by_calendar_days(data.index, h)
            aligned = data[target].reindex(future_idx)
            aligned.index = data.index
            new_cols[f"{target}_d+{h}"] = aligned

        # Build day ids from local calendar dates to avoid DST-induced day duplication.
        local_dates = pd.Series(data.index.date, index=data.index)
        base_ordinal = local_dates.iloc[0].toordinal()
        day_ids = local_dates.map(lambda d: d.toordinal() - base_ordinal).astype(np.int64)
        new_cols["day"] = day_ids

        new_cols["hour"] = pd.Series(self._compute_period_index(data.index), index=data.index)
        return pd.concat([data, pd.DataFrame(new_cols)], axis=1)

    def _shift_index_by_calendar_days(self, index: pd.DatetimeIndex, days: int) -> pd.DatetimeIndex:
        # Same wall-clock time `days` local calendar days later, NaT on the hour skipped in spring. A target
        # landing on the hour repeated in autumn from a day where that hour occurs once stays NaT (one row
        # per horizon per year): which of the two prices is "the" target is undefined, and a mean of them would
        # be an actual no market ever cleared. `aggregate()` drops that day for it, and nothing else depends on it.
        if is_utc_index(index):
            warnings.warn(
                "The model's data has a UTC index, so its days and hours are UTC ones, not local market days, and "
                "day-lag columns built in local time won't line up with the targets around DST. Keep the market's "
                "time zone (e.g. after reading a CSV: df.index = pd.to_datetime(df.index, utc=True).tz_convert(zone)).",
                UserWarning,
                stacklevel=3,
            )
        return shift_calendar_days(index, days, ambiguous="NaT")

    def _compute_period_index(self, index: pd.DatetimeIndex) -> np.ndarray:
        return _contract.period_index(index, self._freq)

    def _build_hour_arrays(self, horizon: int) -> Dict[int, Dict]:
        target_cols = _contract.target_columns(self._target, horizon)
        # Only the predictors the model uses are converted to float and sent to the workers.
        feature_cols = list(self._used_predictors)
        hour_arrays = {}
        for hour in range(self._periods_per_day):
            hour_df = self._hour_data[hour]
            hour_arrays[hour] = {
                "x": hour_df[feature_cols].to_numpy(dtype=np.float64, copy=True),
                "y": hour_df[target_cols].to_numpy(dtype=np.float64, copy=True),
                "days": self._hour_days[hour].copy(),
                "timestamps": hour_df.index.values.copy(),
                "col_index": {c: i for i, c in enumerate(feature_cols)},
            }
        return hour_arrays

    def _precompute_scalable_masks(
        self, horizons: Sequence[int], offset: Optional[int] = None, last: Optional[int] = None
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """(period, horizon) -> which of a fixed list's columns are scaled, for each horizon planned, read on the window's first
        calendar day `offset` (a column with no value there as `_scalable_mask` says, up to `last`, the window's last). With a list
        of training windows, (period, horizon, member) -> each member's."""
        offset = self._offset if offset is None else offset
        last = self._last_day if last is None else last
        return {
            (hour, hz) if member is None else (hour, hz, member): self._scalable_mask(
                hour, hz, self._expand_predictors(hz, hour=hour), offset, last, member
            )
            for hour in range(self._periods_per_day)
            for hz in horizons
            for member in mask_members(self.training_window)
        }

    def _training_sample(self, hour: int, day: int, hz: int, predictors: List[str], member: Any = None) -> np.ndarray:
        """The training rows of the task (hour, hz, day) in these columns, before dropping missing values, with the training
        window `member` (the model's one window if None)."""
        window = member_window(self._members[0] if member is None else member, day)
        rows = training_rows(self._hour_days[hour], day, window, hz, self._newest_of(hour, day))
        return self._hour_data[hour].iloc[rows][predictors].to_numpy(dtype=np.float64)

    def _precompute_expanded_predictors(self, horizons: Sequence[int]) -> Dict[Tuple[int, int], List[str]]:
        """(period, horizon) -> a fixed list's names, `{horizon}` and callables expanded, for each horizon planned."""
        return {
            (hour, hz): self._expand_predictors(hz, hour=hour)
            for hour in range(self._periods_per_day)
            for hz in horizons
        }

    def _expand_predictors(self, horizon: int, hour: int = 0) -> List[str]:
        result = []
        if self._predictors_fn is not None:
            try:
                n_params = len(inspect.signature(self._predictors_fn).parameters)
            except (ValueError, TypeError):
                n_params = 1
            predictors = self._predictors_fn(horizon, hour) if n_params >= 2 else self._predictors_fn(horizon)
        else:
            predictors = self.predictors
        for col in predictors:
            if callable(col):
                try:
                    n_params = len(inspect.signature(col).parameters)
                except (ValueError, TypeError):
                    n_params = 1
                result.append(col(horizon, hour) if n_params >= 2 else col(horizon))
            elif "{horizon}" in str(col):
                result.append(str(col).replace("{horizon}", str(horizon)))
            else:
                result.append(col)
        return result

    @abstractmethod
    def _fit_predict(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> Tuple[float, list]:
        pass

    def _fit_model(self, train_x: np.ndarray, train_y: np.ndarray) -> Any:
        """Fit on standardised data and return the fit, which `_predict_fitted` forecasts from: the seam of `refit`.

        A model that keeps nothing between fits implements both, and its `_predict_fitted(_fit_model(x, y), test_x)` is
        `_fit_predict(x, y, test_x)` to the bit. A model with `_fit_predict` alone runs with `refit=None` only."""
        raise NotImplementedError(f"{type(self).__name__} has no _fit_model: it runs with refit=None only")

    def _predict_fitted(self, fitted: Any, test_x: np.ndarray) -> Tuple[float, list]:
        """The forecast of the one standardised row `test_x` from `fitted`, and its coefficients, as `_fit_predict` returns
        them. Called once per forecast, never on several rows at once: a product of a block of rows can differ from that of one
        row in the last bits, and a forecast must not depend on which others share its fit."""
        raise NotImplementedError(f"{type(self).__name__} has no _predict_fitted: it runs with refit=None only")
