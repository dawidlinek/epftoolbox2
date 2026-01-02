import numpy as np
from typing import Tuple


class StandardScaler:
    def __init__(self):
        self._target_mean = 0.0
        self._target_std = 1.0

    def fit_transform(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_x = train_x.astype(np.float64, copy=True)
        test_x = test_x.astype(np.float64, copy=True)

        scalable_mask = self.get_scalable_mask(train_x)

        for i in range(train_x.shape[1]):
            if not scalable_mask[i]:
                continue
            col = train_x[:, i]
            mean = np.nanmean(col)
            std = np.nanstd(col, ddof=1)
            if std == 0 or np.isnan(std):
                std = 1.0
            train_x[:, i] = (col - mean) / std
            test_x[:, i] = (test_x[:, i] - mean) / std

        self._target_mean = np.nanmean(train_y)
        self._target_std = np.nanstd(train_y, ddof=1)
        if self._target_std == 0 or np.isnan(self._target_std):
            self._target_std = 1.0
        train_y_scaled = (train_y - self._target_mean) / self._target_std

        return train_x, train_y_scaled, test_x

    @staticmethod
    def get_scalable_mask(arr: np.ndarray) -> np.ndarray:
        mask = np.ones(arr.shape[1], dtype=bool)
        for i in range(arr.shape[1]):
            col = arr[:, i]
            unique = np.unique(col[~np.isnan(col)])
            if len(unique) <= 2 and np.all((unique == 0) | (unique == 1)):
                mask[i] = False
        return mask

    def inverse(self, value: float) -> float:
        return value * self._target_std + self._target_mean
