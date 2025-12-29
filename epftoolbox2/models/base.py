from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable, Union, Tuple
from datetime import date
import os
import numpy as np
import pandas as pd
from rich.progress import Progress

from ..scalers.standard import StandardScaler
from ..results.store import ResultStore


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

        self._data: pd.DataFrame = None
        self._groups = None
        self._offset: int = 0
        self._target: str = ""
        self._run_date: str = ""

    def run(
        self,
        data: pd.DataFrame,
        test_start: str,
        test_end: str,
        target: str = "price",
        horizon: int = 7,
        save_to: str = None,
    ) -> List[Dict]:
        self._target = target
        self._run_date = date.today().isoformat()

        self._data = self._preprocess(data, horizon, target)
        self._groups = self._data.groupby(["hour", "day"])
        self._offset = int(self._data.loc[test_start, "day"].iloc[0])
        test_end_day = int(self._data.loc[test_end, "day"].iloc[0])

        all_tasks = [(hour, h, d) for d in range(test_end_day - self._offset + 1) for h in range(1, horizon + 1) for hour in range(24)]

        store = ResultStore(save_to) if save_to else None
        tasks = store.get_missing(all_tasks) if store else all_tasks

        if not tasks:
            print(f"All {len(all_tasks)} tasks completed")
            return store.load_all()

        print(f"Running {len(tasks)}/{len(all_tasks)} tasks")

        n_jobs = int(os.environ.get("MAX_THREADS", os.cpu_count() or 1))
        results = [] if not save_to else None

        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(self._fit_one, *task): task for task in tasks}
            with Progress() as progress:
                task_id = progress.add_task(f"[cyan]{self.name}", total=len(tasks))
                for future in as_completed(futures):
                    result = future.result()
                    if store:
                        store.save(result)
                    else:
                        results.append(result)
                    progress.advance(task_id)

        return store.load_all() if save_to else results

    def _preprocess(self, data: pd.DataFrame, horizon: int, target: str) -> pd.DataFrame:
        data = data.copy()
        for h in range(1, horizon + 1):
            data[f"{target}_d+{h}"] = data[target].shift(-24 * h)
        data["day"] = np.arange(len(data)) // 24
        data["hour"] = data.index.hour
        return data

    def _get_slice(self, hour: int, day: int, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        chunks = []
        for d in range(day - self.training_window - horizon, day + 1):
            if (hour, d) in self._groups.groups:
                chunks.append(self._groups.get_group((hour, d)))
        filtered = pd.concat(chunks)
        return filtered.iloc[: -1 - horizon], filtered.iloc[-1:]

    def _fit_one(self, hour: int, horizon: int, day_in_test: int) -> Dict:
        day = self._offset + day_in_test
        target_col = f"{self._target}_d+{horizon}"
        predictors = self._expand_predictors(horizon)

        train, test = self._get_slice(hour, day, horizon)
        actual = float(test[target_col].iloc[0])

        run_date = test.index[0]
        target_date = run_date + pd.Timedelta(days=horizon)

        scaler = StandardScaler()
        train, test = scaler.fit_transform(train, test, predictors, target_col)

        pred, coefs = self._fit_predict(
            train[predictors],
            train[target_col],
            test[predictors],
        )

        return {
            "run_date": run_date.strftime("%Y-%m-%d"),
            "target_date": target_date.strftime("%Y-%m-%d"),
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
    def _fit_predict(self, train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> Tuple[float, list]:
        pass
