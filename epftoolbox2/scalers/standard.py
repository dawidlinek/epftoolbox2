import pandas as pd
from typing import List, Tuple


class StandardScaler:
    def __init__(self):
        self._target_mean = 0.0
        self._target_std = 1.0

    def fit_transform(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        predictors: List[str],
        target: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train, test = train.copy(), test.copy()

        for col in predictors:
            if self._is_scalable(train[col]):
                mean = train[col].mean()
                std = train[col].std() or 1
                train[col] = (train[col] - mean) / std
                test[col] = (test[col] - mean) / std

        self._target_mean = train[target].mean()
        self._target_std = train[target].std() or 1
        train[target] = (train[target] - self._target_mean) / self._target_std

        return train, test

    def _is_scalable(self, series: pd.Series) -> bool:
        unique = series.dropna().unique()
        if len(unique) <= 2 and set(unique).issubset({0, 1}):
            return False
        if series.dtype.name == "category":
            return False
        return True

    def inverse(self, value: float) -> float:
        return value * self._target_std + self._target_mean
