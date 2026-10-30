from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, List, Dict, Callable, Sequence, Union, Tuple, Optional

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
from ..results.ref import ModelResultRef
from ..results.availability import listing_rows, warn_about
from ..results.manifest import build_manifest, check_and_write, check_stored_lists, manifest_path
from ..logging import get_logger
from . import _contract
from .context import PredictorPlan, build_plan, mask_digest, takes_ctx
from ._worker import _pool_init, _run_day
from .._date_utils import is_utc_index, resolve_date, shift_calendar_days
from ..information_set import check_level

if TYPE_CHECKING:  # pragma: no cover
    from ..information_set import InformationSet

logger = get_logger(__name__)

# The issue day has no row for this period (a gap in the data); without the row there is nothing to forecast from.
SKIP_MISSING_TEST_ROW = "missing_test_row"

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
        self._target: str = ""
        self._freq: str = "1h"
        self._periods_per_day: int = 24
        self._used_predictors: List[str] = []
        self._availability_listing: List[Dict] = []

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
    ) -> ModelResultRef:
        """Forecast every period and horizon for each day in [test_start, test_end].

        With `information_set`, `availability` says what happens to a predictor known after the issue time (see
        epftoolbox2.information_set): "raise", the default here, refuses it before any fit; "warn" goes on and lists it in
        a warning, the returned ref and the manifest; "none" does neither. It never changes a forecast. `ModelPipeline`
        defaults to "warn" and passes `warn_listing=False`, having warned for all its models at once.

        With `save_to`, a manifest is written next to the results (see epftoolbox2.results.manifest) and,
        if one is already there, compared to this run's configuration before anything is resumed.
        `allow_manifest_mismatch` overwrites a disagreeing manifest instead of raising; `require_manifest`
        raises instead of warning when `save_to` has results but no manifest.

        `plan` is what `ModelPipeline` built for a predictor function of `ctx` before any model ran, for this data,
        test period, horizon and freq; without it the run builds its own. A model with fixed lists has none.
        """
        self._set_freq(freq)
        check_level(availability)

        if forecast_only:
            test_start = resolve_date(test_start or "today")
            test_end = resolve_date(test_end or "today")
        else:
            if test_start is None or test_end is None:
                raise ValueError("test_start and test_end are required unless forecast_only=True")
            test_start = resolve_date(test_start)
            test_end = resolve_date(test_end)

        self._target = target
        if plan is None:
            plan = self._plan_predictors(
                data, target, horizon, freq, information_set, test_start, test_end, availability=availability
            )
        else:
            self._check_plan_is_for(plan, data, horizon, freq, information_set, test_start, test_end, availability)
        expanded = self._check_inputs(
            data, target, horizon, freq, information_set, test_start, test_end, plan=plan, availability=availability
        )
        listing = self._availability_listing
        if warn_listing:
            warn_about({self.name: listing}, stacklevel=3)
        used_columns = list(dict.fromkeys([target, *self._used_predictors]))

        self._data = self._preprocess(data[used_columns], horizon, target)

        for hour in range(self._periods_per_day):
            hour_df = self._data[self._data["hour"] == hour].copy()
            self._hour_data[hour] = hour_df
            self._hour_days[hour] = hour_df["day"].values

        self._offset = self._resolve_day(test_start, "test_start")
        test_end_day = self._resolve_day(test_end, "test_end")
        n_days = test_end_day - self._offset + 1
        first_day = datetime.date.fromisoformat(test_start)
        test_days = [first_day + datetime.timedelta(days=d) for d in range(n_days)]
        absent_slots = {
            (period, (day.date() - first_day).days)
            for period, day in _contract.missing_test_rows(self._data.index, freq, test_days)
        }
        self._data = None

        all_tasks = [
            (hour, h, d) for d in range(n_days) for h in range(1, horizon + 1) for hour in range(self._periods_per_day)
        ]

        plan_masks = None if plan is None else self._plan_scalable_masks(plan)

        store = ResultStore(save_to, test_start=test_start) if save_to else None
        if store:
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
                predictor_lists=None if plan is None else plan.manifest_lists(plan_masks),
                availability=self._manifest_availability(information_set, availability, listing),
            )
            if plan is not None:
                check_stored_lists(
                    manifest_path(store.path),
                    manifest,
                    store.iter_lines(cols=["hour", "horizon", "run_date", "predictor_list"]),
                    self._stored_list_lookup(plan, plan_masks),
                    allow_mismatch=allow_manifest_mismatch,
                    model_name=self.name,
                )
            check_and_write(
                manifest_path(store.path),
                manifest,
                has_existing_results=store.path.exists() and store.path.stat().st_size > 0,
                allow_mismatch=allow_manifest_mismatch,
                require_manifest=require_manifest,
                model_name=self.name,
            )
            if manifest.availability is not None:
                listing = manifest.availability["listing"]  # with what earlier runs listed for other test periods
        pending = store.get_missing(all_tasks) if store else all_tasks

        skipped: Counter = Counter()
        skip_examples: Dict[str, str] = {}
        tasks = []
        for task in pending:
            if (task[0], task[2]) in absent_slots:
                skipped[SKIP_MISSING_TEST_ROW] += 1
                skip_examples.setdefault(SKIP_MISSING_TEST_ROW, f"period {task[0]} on {test_days[task[2]]}")
            else:
                tasks.append(task)

        in_memory = None if save_to else []
        try:
            if tasks:
                self._run_tasks(
                    tasks, horizon, test_start, n_days, store, in_memory, skipped, skip_examples, plan, plan_masks
                )
        finally:
            if store:
                store.flush()
            self._cleanup()

        if not pending:
            print(f"All {len(all_tasks)} tasks completed")
        produced = len(all_tasks) - len(store.get_missing(all_tasks)) if store else len(in_memory)
        self._report_counts(len(all_tasks), produced, skipped, skip_examples)

        return ModelResultRef(
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
        )

    @staticmethod
    def _manifest_availability(information_set, availability: str, listing: List[Dict]) -> Optional[Dict]:
        """What a manifest records about the issue-time check; level "none" records nothing, as if it had not run."""
        if information_set is None or availability == "none":
            return None
        return {"level": availability, "listing": listing}

    def _run_tasks(
        self, tasks, horizon, test_start, n_days, store, in_memory, skipped, skip_examples, plan=None, plan_masks=None
    ) -> None:
        hour_arrays = self._build_hour_arrays(horizon)
        if plan is None:
            expanded_preds = self._precompute_expanded_predictors(horizon)
            scalable_masks = self._precompute_scalable_masks(horizon)
        else:
            expanded_preds = {list_id: list(names) for list_id, names in enumerate(plan.lists)}
            scalable_masks = plan_masks

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
        """
        self._set_freq(freq)
        self._availability_listing = []
        if self._ctx_fn is not None:
            if plan is None:
                plan = self._plan_predictors(
                    data, target, horizon, freq, information_set, test_start, test_end, availability=availability
                )
            expanded = plan.names_by_pair()
            self._used_predictors = _contract.check_predictors(data, expanded, target, horizon)
            if information_set is not None and availability == "raise":
                plan.check(target, self.name)
            elif information_set is not None and availability == "warn":
                self._availability_listing = plan.listing(target, self.name)
            plan.release()
            return expanded
        self._check_frame(data, target, horizon, freq, test_start, test_end)
        expanded = self._precompute_expanded_predictors(horizon)
        self._used_predictors = _contract.check_predictors(data, expanded, target, horizon)
        if information_set is not None and availability == "raise":
            information_set.check(expanded, target, freq, data.index, test_start, test_end, model_name=self.name)
        elif information_set is not None and availability == "warn":
            uses = information_set.late_uses(
                expanded, target, freq, data.index, test_start, test_end, model_name=self.name
            )
            self._availability_listing = listing_rows(uses)
        return expanded

    def _check_frame(self, data, target, horizon, freq, test_start=None, test_end=None) -> None:
        _contract.check_index(data, freq, _contract.read_days(test_start, test_end, self.training_window, horizon))
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
    ) -> Optional[PredictorPlan]:
        """The plan of a predictor function of `ctx` for this data and test period, or None for a model with fixed lists.

        `issue_days` plans these days instead of [test_start, test_end]: `verify` samples a few to learn the inputs.
        `availability` is the level `ctx` answers at (see `epftoolbox2.models.context`).
        """
        if self._ctx_fn is None:
            return None
        self._set_freq(freq)
        self._check_frame(data, target, horizon, freq, test_start, test_end)
        if issue_days is not None:
            days = [datetime.date.fromisoformat(day) for day in issue_days]
        elif test_start is None or test_end is None:
            raise ValueError(f"{self.name}: a predictor function of `ctx` needs the test period to plan its lists")
        else:
            days = _test_days(test_start, test_end)
        return build_plan(self._ctx_fn, data, days, horizon, freq, information_set, warn_unpublished, availability)

    def _check_plan_is_for(
        self, plan, data, horizon, freq, information_set, test_start, test_end, availability="raise"
    ) -> None:
        """A plan answers for one function, frame and test period; on another run it would give wrong lists silently."""
        if plan.function is not self._ctx_fn:
            raise ValueError(f"{self.name}: this plan was built for another predictor function")
        problems = []
        if plan.data is not data:
            problems.append("data")
        if plan.issue_days != _test_days(test_start, test_end):
            problems.append("test period")
        if plan.horizon != horizon:
            problems.append("horizon")
        if plan.freq != freq:
            problems.append("freq")
        if information_set is not None and plan.information_set is not information_set:
            problems.append("information set")
        if information_set is not None and plan.availability != availability:
            problems.append("availability level")
        if problems:
            raise ValueError(f"{self.name}: this plan was built for another {', '.join(problems)}")

    def _plan_scalable_masks(self, plan: PredictorPlan) -> Dict[Tuple[int, int, int], np.ndarray]:
        """(list id, period, horizon) -> which of the list's columns are scaled, from all test days of that list.

        All of them, pending or not, so a resumed run computes what a fresh one does.

        A list's mask is read from the window of its first test day, so nothing later than a forecast's own issue day
        decides how it is scaled. A list in use on every test day of a (period, horizon) is a fixed list there and gets
        exactly what a fixed list gets, which is what makes a function returning that list identical to it. Any other
        list may hold a column that only starts with it, and a column with no value can't tell whether it is binary
        (`get_scalable_mask` would leave it unscaled for good): it is read from the first of the list's test days that has
        a value for it. Before that day the list's forecasts have no training row, since none has the column.
        """
        masks = {}
        for (hour, hz), ids in plan.assignment.items():
            for list_id in np.unique(ids).tolist():
                predictors = list(plan.lists[list_id])
                test_days = np.flatnonzero(ids == list_id)
                sample = self._training_sample(hour, self._offset + test_days[0], hz, predictors)
                mask = StandardScaler.get_scalable_mask(sample)
                if len(test_days) < len(ids):
                    self._read_columns_without_values(
                        mask, np.isnan(sample).all(axis=0), hour, hz, predictors, test_days
                    )
                masks[(list_id, hour, hz)] = mask
        return masks

    def _read_columns_without_values(self, mask, without_values, hour, hz, predictors, test_days) -> None:
        for day in test_days[1:]:
            if not without_values.any():
                return
            sample = self._training_sample(hour, self._offset + day, hz, predictors)
            found = without_values & ~np.isnan(sample).all(axis=0)
            if found.any():
                mask[found] = StandardScaler.get_scalable_mask(sample)[found]
                without_values &= ~found

    @staticmethod
    def _stored_list_lookup(plan: PredictorPlan, masks: Dict[Tuple[int, int, int], np.ndarray]) -> Callable:
        """What a fresh run would give a stored forecast's key: (list hash, mask digest), or None if it isn't planned."""
        digests = {key: mask_digest(mask) for key, mask in masks.items()}

        def lookup(period: int, horizon: int, run_date: str) -> Optional[Tuple[str, str]]:
            list_id = plan.list_id_of(period, horizon, run_date)
            return None if list_id is None else (plan.hashes[list_id], digests[(list_id, period, horizon)])

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
        # The worker treats the last row of a period's window as the issue day's row, so rows must be in time order.
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
        # per horizon per year): which of the two prices is "the" target is ambiguous, and with the training
        # cut counting rows (see .claude/rules/models.md) a value there would put a target dated on the
        # issue day into training for the hour-2 forecasts on the fall-back day.
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

    def _precompute_scalable_masks(self, horizon: int) -> Dict[Tuple[int, int], np.ndarray]:
        masks = {}
        for hour in range(self._periods_per_day):
            for hz in range(1, horizon + 1):
                preds = self._expand_predictors(hz, hour=hour)
                sample_x = self._training_sample(hour, self._offset, hz, preds)
                masks[(hour, hz)] = StandardScaler.get_scalable_mask(sample_x)
        return masks

    def _training_sample(self, hour: int, day: int, hz: int, predictors: List[str]) -> np.ndarray:
        """The training rows of the task (hour, hz, day) in these columns, before dropping missing values."""
        days = self._hour_days[hour]
        mask = (days >= day - self.training_window - hz) & (days <= day)
        return self._hour_data[hour][mask][predictors].to_numpy(dtype=np.float64)[: -(1 + hz)]

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
