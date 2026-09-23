from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Callable, Union, Tuple, Optional

import datetime
import gc
import inspect
import multiprocessing
import os
import sys

import numpy as np
import pandas as pd
from rich.progress import Progress

from ..scalers.standard import StandardScaler
from ..results.store import ResultStore
from ..results.ref import ModelResultRef
from ..logging import get_logger
from . import _contract
from ._worker import _worker_init, _run_day
from .._date_utils import resolve_date

logger = get_logger(__name__)

# The issue day has no row for this period (a gap in the data); without the row there is nothing to forecast from.
SKIP_MISSING_TEST_ROW = "missing_test_row"

_FREQ_TO_PERIODS: Dict[str, int] = {"1h": 24, "15min": 96}


class BaseModel(ABC):
    def __init__(
        self,
        predictors: List[Union[str, Callable]],
        training_window: int = 365,
        name: str = "Model",
    ):
        if callable(predictors):
            self._predictors_fn = predictors
            self.predictors = []
        else:
            self._predictors_fn = None
            self.predictors = predictors
        self.training_window = training_window
        self.name = name

        self._data: Optional[pd.DataFrame] = None
        self._hour_data: Dict[int, pd.DataFrame] = {}
        self._hour_days: Dict[int, np.ndarray] = {}
        self._offset: int = 0
        self._target: str = ""
        self._freq: str = "1h"
        self._periods_per_day: int = 24
        self._used_predictors: List[str] = []

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
    ) -> ModelResultRef:
        self._set_freq(freq)

        if forecast_only:
            test_start = resolve_date(test_start or "today")
            test_end = resolve_date(test_end or "today")
        else:
            if test_start is None or test_end is None:
                raise ValueError("test_start and test_end are required unless forecast_only=True")
            test_start = resolve_date(test_start)
            test_end = resolve_date(test_end)

        self._target = target
        self._check_inputs(data, target, horizon, freq)
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

        store = ResultStore(save_to) if save_to else None
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
                self._run_tasks(tasks, horizon, test_start, n_days, store, in_memory, skipped, skip_examples)
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
        )

    def _run_tasks(self, tasks, horizon, test_start, n_days, store, in_memory, skipped, skip_examples) -> None:
        hour_arrays = self._build_hour_arrays(horizon)
        expanded_preds = self._precompute_expanded_predictors(horizon)
        scalable_masks = self._precompute_scalable_masks(horizon)

        tasks_by_day: Dict[int, list] = {}
        for t in tasks:
            tasks_by_day.setdefault(t[2], []).append(t)

        threads_per_process, n_processes = self._executor_config()

        config = {
            "offset": self._offset,
            "test_start": test_start,
            "training_window": self.training_window,
            "scalable_masks": scalable_masks,
            "threads_per_process": threads_per_process,
            "freq": self._freq,
        }
        first_day = datetime.date.fromisoformat(test_start)

        mp_context = multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn")
        with ProcessPoolExecutor(
            max_workers=n_processes,
            mp_context=mp_context,
            initializer=_worker_init,
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
        self, data: pd.DataFrame, target: str, horizon: int, freq: str
    ) -> Dict[Tuple[int, int], List[str]]:
        """Check the data contract before any fit and return the expanded predictors per (period, horizon).

        Raises ValueError on a frequency mismatch, duplicate timestamps, reserved or colliding column names,
        and missing or non-numeric predictors. `ModelPipeline` calls it for every model before running any.
        """
        self._set_freq(freq)
        _contract.check_index(data, freq)
        _contract.check_names(data, target, horizon)
        expanded = self._precompute_expanded_predictors(horizon)
        self._used_predictors = _contract.check_predictors(data, expanded, target, horizon)
        return expanded

    def _executor_config(self) -> tuple:
        threads = int(os.environ.get("THREADS_PER_PROCESS", os.environ.get("MAX_THREADS", "16")))
        processes = int(os.environ.get("MAX_PROCESSES", max(1, (os.cpu_count() or 1) // threads)))
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
        if index.tz is None:
            return index + pd.Timedelta(days=days)

        shifted_local_naive = index.tz_localize(None) + pd.Timedelta(days=days)
        try:
            return shifted_local_naive.tz_localize(index.tz, ambiguous="infer", nonexistent="NaT")
        except Exception:
            return shifted_local_naive.tz_localize(index.tz, ambiguous="NaT", nonexistent="NaT")

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
            days = self._hour_days[hour]
            day = self._offset
            hour_df = self._hour_data[hour]
            for hz in range(1, horizon + 1):
                day_min = day - self.training_window - hz
                mask = (days >= day_min) & (days <= day)
                preds = self._expand_predictors(hz, hour=hour)
                sample_x = hour_df[mask][preds].to_numpy(dtype=np.float64)[: -(1 + hz)]
                masks[(hour, hz)] = StandardScaler.get_scalable_mask(sample_x)
        return masks

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
