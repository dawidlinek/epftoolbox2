from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from threading import Lock
from typing import List, Dict, Callable, Union, Tuple, Optional
from datetime import date
import gc
import os

import numpy as np
import pandas as pd
from rich.progress import Progress

from ..scalers.standard import StandardScaler
from ..results.store import ResultStore
from ..results.ref import ModelResultRef


def _batched(iterable, n: int):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


class BaseModel(ABC):
    def __init__(
        self,
        predictors: List[Union[str, Callable]],
        training_window: int = 365,
        name: str = "Model",
    ):
        self.predictors = predictors
        self.training_window = training_window
        self.name = name

        self._data: Optional[pd.DataFrame] = None
        self._hour_data: Dict[int, pd.DataFrame] = {}
        self._hour_days: Dict[int, np.ndarray] = {}
        self._offset: int = 0
        self._target: str = ""
        self._run_date: str = ""

        self._scalable_mask: Optional[np.ndarray] = None
        self._mask_lock = Lock()

    def run(
        self,
        data: pd.DataFrame,
        test_start: str,
        test_end: str,
        target: str = "price",
        horizon: int = 7,
        save_to: str = None,
    ) -> ModelResultRef:
        self._target = target
        self._run_date = date.today().isoformat()

        self._data = self._preprocess(data, horizon, target)

        for hour in range(24):
            hour_df = self._data[self._data["hour"] == hour].copy()
            self._hour_data[hour] = hour_df
            self._hour_days[hour] = hour_df["day"].values

        self._offset = int(self._data.loc[test_start, "day"].iloc[0])
        test_end_day = int(self._data.loc[test_end, "day"].iloc[0])

        self._data = None

        all_tasks = [
            (hour, h, d)
            for d in range(test_end_day - self._offset + 1)
            for h in range(1, horizon + 1)
            for hour in range(24)
        ]

        store = ResultStore(save_to) if save_to else None
        tasks = store.get_missing(all_tasks) if store else all_tasks
        n_completed = len(all_tasks) - len(tasks)

        if not tasks:
            print(f"All {len(all_tasks)} tasks completed")
            if store:
                store.flush()
            self._cleanup()
            return ModelResultRef(
                name=self.name,
                path=store.path if store else None,
                count=len(all_tasks),
                test_start=test_start,
                test_end=test_end,
                horizon=horizon,
            )

        n_jobs = int(os.environ.get("MAX_THREADS", os.cpu_count() or 1))
        batch_size = max(n_jobs * 8, 256)

        in_memory: Optional[List[Dict]] = None if save_to else []

        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            with Progress() as progress:
                task_id = progress.add_task(
                    f"[cyan]{self.name}", total=len(all_tasks), completed=n_completed
                )
                for batch in _batched(tasks, batch_size):
                    futures = {pool.submit(self._fit_one, *task): task for task in batch}
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                        except Exception as e:
                            t = futures[future]
                            print(
                                f"Error running task "
                                f"(hour={t[0]}, horizon={t[1]}, day={t[2]}): {e}"
                            )
                            continue
                        if store:
                            store.save(result)
                        else:
                            in_memory.append(result)
                        progress.advance(task_id)

        if store:
            store.flush()

        self._cleanup()

        return ModelResultRef(
            name=self.name,
            path=store.path if store else None,
            count=len(all_tasks),
            test_start=test_start,
            test_end=test_end,
            horizon=horizon,
            _results=in_memory,
        )

    def _cleanup(self) -> None:
        self._data = None
        self._hour_data.clear()
        self._hour_days.clear()
        self._scalable_mask = None
        gc.collect()

    def _preprocess(self, data: pd.DataFrame, horizon: int, target: str) -> pd.DataFrame:
        new_cols: Dict[str, pd.Series] = {}
        for h in range(1, horizon + 1):
            new_cols[f"{target}_d+{h}"] = data[target].shift(-24 * h)
        new_cols["day"] = pd.Series(np.arange(len(data)) // 24, index=data.index)
        new_cols["hour"] = pd.Series(data.index.hour, index=data.index)
        return pd.concat([data, pd.DataFrame(new_cols, index=data.index)], axis=1)

    def _get_slice(self, hour: int, day: int, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        hour_df = self._hour_data[hour]
        days = self._hour_days[hour]
        day_min = day - self.training_window - horizon
        mask = (days >= day_min) & (days <= day)
        filtered = hour_df[mask]
        return filtered.iloc[: -1 - horizon], filtered.iloc[-1:]

    def _fit_one(self, hour: int, horizon: int, day_in_test: int) -> Dict:
        day = self._offset + day_in_test
        target_col = f"{self._target}_d+{horizon}"
        predictors = self._expand_predictors(horizon)
        train, test = self._get_slice(hour, day, horizon)
        actual = float(test[target_col].iloc[0])

        run_date = test.index[0]
        target_date = run_date + pd.Timedelta(days=horizon)

        train_x = train[predictors].values
        train_y = train[target_col].values
        test_x = test[predictors].values

        if self._scalable_mask is None:
            with self._mask_lock:
                if self._scalable_mask is None:
                    self._scalable_mask = StandardScaler.get_scalable_mask(train_x)

        scaler = StandardScaler()
        train_x, train_y, test_x = scaler.fit_transform(
            train_x, train_y, test_x, scalable_mask=self._scalable_mask
        )
        pred, coefs = self._fit_predict(train_x, train_y, test_x)

        return {
            "run_date": run_date.strftime("%Y-%m-%d"),
            "target_date": target_date.strftime("%Y-%m-%d"),
            "run_weekday": run_date.dayofweek,
            "hour": hour,
            "horizon": horizon,
            "day_in_test": day_in_test,
            "prediction": scaler.inverse(float(pred)),
            "actual": actual,
            "coefficients": coefs,
        }

    def _expand_predictors(self, horizon: int) -> List[str]:
        result = []
        for col in self.predictors:
            if callable(col):
                result.append(col(horizon))
            elif "{horizon}" in str(col):
                result.append(str(col).replace("{horizon}", str(horizon)))
            else:
                result.append(col)
        return result

    @abstractmethod
    def _fit_predict(
        self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
    ) -> Tuple[float, list]:
        pass
