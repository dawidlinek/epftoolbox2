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
from ..results.availability import listing_rows, warn_about
from ..results.matching import training_day_rows, warn_about_thin
from ..results.manifest import (
    PAST_THE_FRAME,
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
from ..logging import get_logger
from . import _contract
from ._calendar import IssueCalendar, training_span
from .context import TRAIN_ON_MATCHING, PredictorPlan, build_plan, mask_digest, takes_ctx, warn_never_published_in
from ._worker import (
    NOT_PLANNED,
    SKIP_MISSING_TEST_ROW,
    _pool_init,
    _run_day,
    issue_row,
    matching_digest,
    matching_rows,
    training_row_days,
    training_rows,
)
from .._date_utils import is_utc_index, resolve_date, shift_calendar_days, zone_name
from ..information_set import check_level

if TYPE_CHECKING:  # pragma: no cover
    from ..information_set import InformationSet

logger = get_logger(__name__)

_FREQ_TO_PERIODS: Dict[str, int] = {"1h": 24, "15min": 96}

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
    def __init__(
        self,
        predictors: Union[List[Union[str, Callable]], Callable],
        training_window: int = 365,
        name: str = "Model",
        scaler: ScalerSpec = "standard",
    ):
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

        self._data: Optional[pd.DataFrame] = None
        self._hour_data: Dict[int, pd.DataFrame] = {}
        self._hour_days: Dict[int, np.ndarray] = {}
        self._offset: int = 0
        self._first_ordinal: int = 0  # the date of day id 0
        self._target: str = ""
        self._freq: str = "1h"
        self._periods_per_day: int = 24
        self._used_predictors: List[str] = []
        self._availability_listing: List[Dict] = []
        self._window_listings: Dict[str, List[Dict]] = {}

    def __setattr__(self, key, value):
        # The scaler is kept as a name or {"name", "params"} dict, also when assigned after construction, so a
        # typo fails here rather than in every worker and a scaler instance can't be dropped by save().
        if key == "scaler":
            value = scaler_spec(value)
        super().__setattr__(key, value)

    @property
    def _model_kwargs(self) -> Dict:
        return {}

    def _resolve_day(self, date_str: str, label: str) -> int:
        if date_str not in self._data.index:
            last = self._data.index[-1].date()
            raise ValueError(
                f"'{label}' date '{date_str}' is not in the data "
                f"(data ends at {last}). Fetch data through at least '{date_str}'."
            )
        return int(self._data.loc[date_str, "day"].iloc[0])

    def run(
        self,
        data: pd.DataFrame,
        test_start: str = None,
        test_end: str = None,
        target: str = "price",
        horizon: int = 7,
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
    ) -> ModelResultRef:
        """Forecast every period and horizon the clock has on each day in [test_start, test_end] and on its target day.

        A forecast (period, h, D) trains on the rows of its period dated in `training_span` and forecasts from the last row dated
        D: see `epftoolbox2.models._calendar`.

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
        """
        self._set_freq(freq)
        check_level(availability)

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

        self._target = target
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
            )
        else:
            self._check_plan_is_for(
                plan, data, horizon, freq, information_set, test_start, test_end, availability, windows
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
        )
        listing = self._availability_listing
        window_listings = self._window_listings
        if warn_listing:
            warn_about({self.name: listing}, stacklevel=3)
        used_columns = list(dict.fromkeys([target, *self._used_predictors]))

        self._load_periods(data, used_columns, horizon, target)

        first_day = datetime.date.fromisoformat(test_start)
        if windows is None:
            self._offset = self._resolve_day(test_start, "test_start")
            test_end_day = self._resolve_day(test_end, "test_end")
            run_days = list(range(test_end_day - self._offset + 1))
        else:
            for name, (start, end) in windows.periods:
                self._resolve_day(start, f"{name} start")
                self._resolve_day(end, f"{name} end")
            self._offset = self._resolve_day(test_start, "first day of the run")
            run_days = [(datetime.date.fromisoformat(day) - first_day).days for day in windows.issue_days()]
        n_days = len(run_days)
        test_days = {d: first_day + datetime.timedelta(days=d) for d in run_days}
        calendar = IssueCalendar(freq, data.index.tz, range(1, horizon + 1))
        absent_slots = {
            (period, (day.date() - first_day).days)
            for period, day in _contract.missing_test_rows(self._data.index, freq, test_days.values())
        }
        self._data = None

        all_tasks = calendar.tasks(first_day, run_days)

        window_days = None if windows is None else self._window_days(windows, first_day)
        if windows is not None:
            masks = self._window_masks(plan, windows, horizon, first_day)
        else:
            masks = None if plan is None else self._plan_scalable_masks(plan)
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
                freq=freq,
                test_start=test_start,
                test_end=test_end,
                information_set=information_set,
                predictor_function=None if plan is None else plan.function_record(),
                predictor_lists=None if plan is None else plan.manifest_lists(masks if windows else {TEST: masks}),
                availability=self._manifest_availability(information_set, availability, listing, window_listings),
                windows=windows,
                binary_masks=None if windows is None or plan is not None else self._mask_digests(masks),
                read_columns=() if plan is None else sorted(plan.queried),
                frame_hashes=frame_hashes,
            )

            def check_stored():
                check_stored_lists(
                    manifest_path(store.path),
                    manifest,
                    store.iter_lines(
                        cols=["hour", "horizon", "run_date", "predictor_list", "matching_days", "window", "actual"]
                    ),
                    self._stored_forecast_lookup(plan, masks, windows),
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
                check_stored=None if windows is None else check_stored,
            )
            if availability != "none" and manifest.availability is not None:
                listing = manifest.availability["listing"]  # with what earlier runs listed for other test periods
                window_listings = manifest.availability.get("windows", window_listings)
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
            else:
                tasks.append(task)

        in_memory = None if save_to else []
        try:
            if tasks:
                self._run_tasks(
                    tasks,
                    horizon,
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
        recorded = {"level": availability, "listing": listing}
        if window_listings:
            recorded["windows"] = window_listings
        return recorded

    @staticmethod
    def _window_days(windows: RunWindows, first_day: datetime.date) -> Dict[int, Tuple[str, int, int]]:
        """Each day of the run, counted from `first_day` -> (its window, days since the window's start, place among the
        run's issue days)."""
        days = {}
        position = 0
        for name, (start, end) in windows.periods:
            first = (datetime.date.fromisoformat(start) - first_day).days
            last = (datetime.date.fromisoformat(end) - first_day).days
            for day in range(first, last + 1):
                days[day] = (name, day - first, position)
                position += 1
        return days

    def _window_masks(self, plan, windows: RunWindows, horizon: int, first_day: datetime.date) -> Dict[str, Dict]:
        """Window -> its binary-column masks, as a plain run over that window alone computes them, from its first day."""
        masks = {}
        for name, (start, end) in windows.periods:
            offset = self._offset + (datetime.date.fromisoformat(start) - first_day).days
            if plan is None:
                masks[name] = self._precompute_scalable_masks(horizon, offset)
            else:
                days = (plan.position_of(start), plan.position_of(end) + 1)
                masks[name] = self._plan_scalable_masks(plan, days, offset)
        return masks

    @staticmethod
    def _mask_digests(masks: Dict[str, Dict[Tuple[int, int], np.ndarray]]) -> Dict[str, str]:
        return {
            mask_key(window, period, horizon): mask_digest(mask)
            for window, window_masks in masks.items()
            for (period, horizon), mask in window_masks.items()
        }

    @staticmethod
    def _window_lists(plan: PredictorPlan, windows: RunWindows) -> Dict[str, List[Dict]]:
        return {
            name: plan.describe((plan.position_of(start), plan.position_of(end) + 1))
            for name, (start, end) in windows.periods
        }

    def _run_tasks(
        self,
        tasks,
        horizon,
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
    ) -> None:
        hour_arrays = self._build_hour_arrays(horizon)
        if plan is None:
            expanded_preds = self._precompute_expanded_predictors(horizon)
            scalable_masks = self._precompute_scalable_masks(horizon) if masks is None else masks
        else:
            expanded_preds = {list_id: list(names) for list_id, names in enumerate(plan.lists)}
            scalable_masks = masks

        tasks_by_day: Dict[int, list] = {}
        for t in tasks:
            tasks_by_day.setdefault(t[2], []).append(t)

        threads_per_process, n_processes = self._executor_config()
        n_processes = min(n_processes, len(tasks_by_day))  # a worker runs whole days; more would sit idle
        if sys.platform == "win32":
            n_processes = min(n_processes, _WINDOWS_MAX_WORKERS)

        config = {
            "offset": self._offset,
            "test_start": test_start,
            "training_window": self.training_window,
            "scalable_masks": scalable_masks,
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
        if window_days is not None:
            config["run_days"] = window_days
        first_day = datetime.date.fromisoformat(test_start)

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
            futures = {pool.submit(_run_day, day_tasks): day for day, day_tasks in tasks_by_day.items()}
            with Progress() as progress:
                task_id = progress.add_task(
                    f"[cyan]{self.name}",
                    total=n_days,
                    completed=n_days - len(tasks_by_day),
                )
                for future in as_completed(futures):
                    run_date = first_day + datetime.timedelta(days=futures[future])
                    try:
                        day_results, day_skipped = future.result()
                    except Exception as exc:
                        # Results already saved stay in the store, so a fixed rerun resumes from here.
                        pool.shutdown(wait=True, cancel_futures=True)
                        raise RuntimeError(f"{self.name}: forecasting failed for run date {run_date}: {exc}") from exc
                    for result in day_results:
                        if store:
                            store.save(result)
                        else:
                            in_memory.append(result)
                    for (hour, hz, _), outcome in day_skipped:
                        reason, _, detail = outcome.partition(": ")
                        skipped[reason] += 1
                        if window_days is not None:
                            window_skipped[window_days[futures[future]][0]][reason] += 1
                        where = f"period {hour}, horizon {hz}, run date {run_date}"
                        skip_examples.setdefault(reason, f"{where}: {detail}" if detail else where)
                    progress.advance(task_id)

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
    ) -> Dict[Tuple[int, int], List[str]]:
        """Check the data contract before any fit and return the expanded predictors per (period, horizon).

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
        """
        self._set_freq(freq)
        self._availability_listing = []
        self._window_listings = {}
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
                )
            expanded = plan.names_by_pair()
            self._used_predictors = _contract.check_predictors(data, expanded, target, horizon)
            if information_set is not None and availability == "raise":
                plan.check(target, self.name)
            elif information_set is not None and availability == "warn":
                self._availability_listing = plan.listing(target, self.name)
                if windows is not None:
                    self._window_listings = self._plan_window_listings(plan, target, windows)
            plan.release()
            return expanded
        self._check_frame(data, target, horizon, freq, test_start, test_end, windows)
        expanded = self._precompute_expanded_predictors(horizon)
        self._used_predictors = _contract.check_predictors(data, expanded, target, horizon)
        days = None if windows is None else windows.issue_days()
        if information_set is not None and availability == "raise":
            information_set.check(
                expanded, target, freq, data.index, test_start, test_end, model_name=self.name, issue_days=days
            )
        elif information_set is not None and availability == "warn":
            uses = information_set.late_uses(
                expanded, target, freq, data.index, test_start, test_end, model_name=self.name, issue_days=days
            )
            self._availability_listing = listing_rows(uses)
            for name, (start, end) in windows.periods if windows is not None else ():
                self._window_listings[name] = listing_rows(
                    information_set.late_uses(expanded, target, freq, data.index, start, end, model_name=self.name)
                )
        return expanded

    def _plan_window_listings(self, plan: PredictorPlan, target: str, windows: RunWindows) -> Dict[str, List[Dict]]:
        """What each window uses. A run of one window uses what the whole plan does, whether or not its cells are gone."""
        if len(windows.periods) == 1:
            return {windows.periods[0][0]: plan.listing(target, self.name)}
        return {
            name: plan.listing(target, self.name, (plan.position_of(start), plan.position_of(end) + 1))
            for name, (start, end) in windows.periods
        }

    def _check_frame(self, data, target, horizon, freq, test_start=None, test_end=None, windows=None) -> None:
        if windows is None:
            days = _contract.read_days(test_start, test_end, self.training_window, horizon)
        else:
            days = _contract.read_ranges([p for _, p in windows.periods], self.training_window, horizon)
        _contract.check_index(data, freq, days)
        _contract.check_timezone_with_day_lags(data)
        _contract.check_names(data, target, horizon)

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
    ) -> Optional[PredictorPlan]:
        """The plan of a predictor function of `ctx` for this data and test period, or None for a model with fixed lists.

        `issue_days` plans these days instead of [test_start, test_end]: `verify` samples a few to learn the inputs.
        `windows` plans the days of a run with validation, once each and none between them; the function doesn't see them.
        `availability` is the level `ctx` answers at (see `epftoolbox2.models.context`).
        """
        if self._ctx_fn is None:
            return None
        self._set_freq(freq)
        self._check_frame(data, target, horizon, freq, test_start, test_end, windows)
        if windows is not None:
            days = [datetime.date.fromisoformat(day) for day in windows.issue_days()]
        elif issue_days is not None:
            days = [datetime.date.fromisoformat(day) for day in issue_days]
        elif test_start is None or test_end is None:
            raise ValueError(f"{self.name}: a predictor function of `ctx` needs the test period to plan its lists")
        else:
            days = _test_days(test_start, test_end)
        plan = build_plan(
            self._ctx_fn,
            data,
            days,
            horizon,
            freq,
            information_set,
            warn_unpublished,
            availability,
            training_window=self.training_window,
        )
        if windows is not None and warn_unpublished and len(windows.periods) > 1:
            for name, (start, end) in windows.periods:
                warn_never_published_in(plan, name, (plan.position_of(start), plan.position_of(end) + 1))
        return plan

    def _check_plan_is_for(
        self, plan, data, horizon, freq, information_set, test_start, test_end, availability="raise", windows=None
    ) -> None:
        """A plan answers for one function, frame and test period; on another run it would give wrong lists silently."""
        if plan.function is not self._ctx_fn:
            raise ValueError(f"{self.name}: this plan was built for another predictor function")
        problems = []
        if plan.data is not data:
            problems.append("data")
        planned_days = (
            _test_days(test_start, test_end)
            if windows is None
            else [datetime.date.fromisoformat(day) for day in windows.issue_days()]
        )
        if plan.issue_days != planned_days:
            problems.append("test period")
        if plan.horizon != horizon:
            problems.append("horizon")
        if plan.freq != freq:
            problems.append("freq")
        if information_set is not None and plan.information_set is not information_set:
            problems.append("information set")
        if information_set is not None and plan.availability != availability:
            problems.append("availability level")
        if plan.uses_matching and plan.training_window != self.training_window:
            problems.append("training window")
        if problems:
            raise ValueError(f"{self.name}: this plan was built for another {', '.join(problems)}")

    def _plan_scalable_masks(
        self, plan: PredictorPlan, days: Tuple[int, Optional[int]] = (0, None), offset: Optional[int] = None
    ) -> Dict[Tuple[int, int, int], np.ndarray]:
        """(list id, period, horizon) -> which of the list's columns are scaled, from all test days of that list.

        All of them, pending or not, so a resumed run computes what a fresh one does. `days` is a (start, stop) slice of the
        plan's issue days and `offset` the day id of its first day, when masks are kept per window: what happens in another
        window then can't move them.

        A list's mask is read from the window of its first test day, so nothing later than a forecast's own issue day
        decides how it is scaled. A list in use on every test day of a (period, horizon) is a fixed list there and gets
        exactly what a fixed list gets, which is what makes a function returning that list identical to it. Any other
        list may hold a column that only starts with it, and a column with no value can't tell whether it is binary
        (`get_scalable_mask` would leave it unscaled for good): it is read from the first of the list's test days that has
        a value for it. Before that day the list's forecasts have no training row, since none has the column.
        """
        offset = self._offset if offset is None else offset
        masks = {}
        for (hour, hz), all_ids in plan.assignment.items():
            ids = all_ids[days[0] : days[1]]
            for list_id in np.unique(ids).tolist():
                predictors = list(plan.lists[list_id])
                test_days = np.flatnonzero(ids == list_id)
                sample = self._training_sample(hour, offset + test_days[0], hz, predictors)
                mask = StandardScaler.get_scalable_mask(sample)
                if len(test_days) < len(ids):
                    self._read_columns_without_values(
                        mask, np.isnan(sample).all(axis=0), hour, hz, predictors, test_days, offset
                    )
                masks[(list_id, hour, hz)] = mask
        return masks

    def _read_columns_without_values(self, mask, without_values, hour, hz, predictors, test_days, offset) -> None:
        for day in test_days[1:]:
            if not without_values.any():
                return
            sample = self._training_sample(hour, offset + day, hz, predictors)
            found = without_values & ~np.isnan(sample).all(axis=0)
            if found.any():
                mask[found] = StandardScaler.get_scalable_mask(sample)[found]
                without_values &= ~found

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
        day = datetime.date.fromisoformat(run_date).toordinal() - self._first_ordinal
        rows = training_row_days(self._hour_days[period], day, self.training_window, horizon)
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
        issue_days = np.array([day.toordinal() - self._first_ordinal for day in plan.issue_days], dtype=np.int64)
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
            first_day, last_day = training_span(issue_days, self.training_window, horizon)
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
        calendar = IssueCalendar(plan.freq, plan.data.index.tz, range(1, plan.horizon + 1))
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
        self, plan: Optional[PredictorPlan], masks: Dict, windows: Optional[RunWindows] = None
    ) -> Callable[[int, int, str], Optional[PlannedForecast]]:
        """What a fresh run would give a stored forecast's key, or None if it isn't planned. `masks` are per window in a
        run with windows."""
        by_window = {TEST: masks} if windows is None else masks
        digests = {
            window: {key: mask_digest(mask) for key, mask in window_masks.items()}
            for window, window_masks in by_window.items()
        }

        def lookup(period: int, horizon: int, run_date: str) -> Optional[PlannedForecast]:
            window = TEST if windows is None else windows.window_of(run_date)
            if window is None:
                return None
            list_id = None if plan is None else plan.list_id_of(period, horizon, run_date)
            key = (period, horizon) if plan is None else (list_id, period, horizon)
            digest = digests[window].get(key)
            if digest is None:
                return None
            matching = plan is not None and plan.train_on[list_id] == TRAIN_ON_MATCHING
            return PlannedForecast(
                None if plan is None else plan.hashes[list_id],
                digest,
                mask_key(window, period, horizon),
                None if windows is None else window,
                self._matching_days_of(plan, period, horizon, run_date, list_id) if matching else None,
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
        self, horizon: int, offset: Optional[int] = None
    ) -> Dict[Tuple[int, int], np.ndarray]:
        offset = self._offset if offset is None else offset
        masks = {}
        for hour in range(self._periods_per_day):
            for hz in range(1, horizon + 1):
                preds = self._expand_predictors(hz, hour=hour)
                sample_x = self._training_sample(hour, offset, hz, preds)
                masks[(hour, hz)] = StandardScaler.get_scalable_mask(sample_x)
        return masks

    def _training_sample(self, hour: int, day: int, hz: int, predictors: List[str]) -> np.ndarray:
        """The training rows of the task (hour, hz, day) in these columns, before dropping missing values."""
        rows = training_rows(self._hour_days[hour], day, self.training_window, hz)
        return self._hour_data[hour].iloc[rows][predictors].to_numpy(dtype=np.float64)

    def _precompute_expanded_predictors(self, horizon: int) -> Dict[Tuple[int, int], List[str]]:
        return {
            (hour, hz): self._expand_predictors(hz, hour=hour)
            for hour in range(self._periods_per_day)
            for hz in range(1, horizon + 1)
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
