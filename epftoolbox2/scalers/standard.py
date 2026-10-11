import numpy as np
from typing import Optional, Sequence, Tuple

from .base import BaseScaler


class StandardScaler(BaseScaler):
    """Standardise every non-binary predictor and the target to mean 0, standard deviation 1 (the default)."""

    name = "standard"

    def __init__(self):
        self._target_mean = 0.0
        self._target_std = 1.0

    def fit_transform(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        scalable_mask: Optional[np.ndarray] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if train_x.dtype == np.float64:
            train_x = train_x.copy()
        else:
            train_x = train_x.astype(np.float64)

        if test_x.dtype == np.float64:
            test_x = test_x.copy()
        else:
            test_x = test_x.astype(np.float64)

        if scalable_mask is None:
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

    def inverse(self, value: float) -> float:
        return value * self._target_std + self._target_mean
