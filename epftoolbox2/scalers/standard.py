import numpy as np
from typing import Optional, Sequence, Tuple

from .base import BaseScaler, mean_std, scale_training_column


class StandardScaler(BaseScaler):
    """Standardise every non-binary predictor and the target to mean 0, standard deviation 1 (the default).

    A predictor that is constant up to rounding noise (standard deviation at or below 1e-12 of its largest absolute
    value) has training values of exactly 0.0, and its test row is centred and divided by 1.0, not by about 1e-15. The
    target is centred and divided by 1.0 in that case. A column with fewer than two values has the scale 1.0 as well.
    """

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
            mean, std, constant = mean_std(col)
            train_x[:, i] = scale_training_column(col, col, mean, std, constant)
            test_x[:, i] = (test_x[:, i] - mean) / std

        self._target_mean, self._target_std, _ = mean_std(train_y)
        train_y_scaled = (train_y - self._target_mean) / self._target_std

        return train_x, train_y_scaled, test_x

    def inverse(self, value: float) -> float:
        return value * self._target_std + self._target_mean
