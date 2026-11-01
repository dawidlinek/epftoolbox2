from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

# A column whose sample standard deviation is at or below this fraction of its largest absolute value is constant
# up to rounding noise: 0.3 with floating-point error has a standard deviation of about 1e-16, not 0, and dividing
# by it would turn the error into a regressor. Real predictors spread over many orders of magnitude more.
CONSTANT_STD_RTOL = 1e-12


def _negligible(std: float, values: np.ndarray) -> bool:
    """Whether a defined standard deviation is at or below ``CONSTANT_STD_RTOL`` of the largest finite ``|value|``."""
    finite = values[np.isfinite(values)]
    return finite.size > 0 and bool(std <= CONSTANT_STD_RTOL * np.abs(finite).max())


def is_constant(values: np.ndarray) -> bool:
    """Whether the non-NaN values are constant up to rounding noise (two or more, see ``CONSTANT_STD_RTOL``)."""
    present = values[~np.isnan(values)]
    if present.size < 2:
        return False
    # The sample standard deviation is at least range / sqrt(2 (n - 1)), so a constant column has a range of at most
    # CONSTANT_STD_RTOL * sqrt(2 (n - 1)) of its largest value. Most columns leave here, before the standard deviation.
    spread = present.max() - present.min()
    if not spread <= 4 * CONSTANT_STD_RTOL * np.sqrt(present.size) * np.abs(present).max():
        return False
    std = np.nanstd(values, ddof=1)
    return not np.isnan(std) and _negligible(std, values)


def mean_std(values: np.ndarray) -> Tuple[float, float]:
    """Mean and sample standard deviation of the non-NaN values: the pair every scaler standardises with.

    The standard deviation is 1.0 where it is undefined (fewer than two values) or the column is constant up to
    rounding noise (``CONSTANT_STD_RTOL``), as for an exactly constant or all-zero column, so that such a column is
    centred and left at its scale instead of being divided by about 1e-15.
    """
    mean = np.nanmean(values)
    std = np.nanstd(values, ddof=1)
    if np.isnan(std) or _negligible(std, values):
        std = 1.0
    return mean, std


class BaseScaler(ABC):
    """Scales one task's training and test data before the fit and maps the prediction back.

    A fresh instance is built for every (period, horizon, run day) task and fitted on that task's training rows
    only; the test row is transformed with the training parameters. Constructor parameters are stored as
    same-named public attributes so that a scaler can be saved to YAML as ``{"name": ..., "params": ...}``.
    """

    name: str = ""

    @abstractmethod
    def fit_transform(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        scalable_mask: Optional[np.ndarray] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit on ``train_x``/``train_y`` and return transformed copies of all three arrays.

        ``scalable_mask`` marks the predictor columns that may be scaled (binary 0/1 columns are left untouched);
        ``columns`` holds the predictor names, for scalers that treat columns differently by name.
        """

    @abstractmethod
    def inverse(self, value: float) -> float:
        """Map a prediction from the scaled target space back to the original target scale."""

    def get_params(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    @staticmethod
    def get_scalable_mask(arr: np.ndarray) -> np.ndarray:
        mask = np.ones(arr.shape[1], dtype=bool)
        for i in range(arr.shape[1]):
            col = arr[:, i]
            unique = np.unique(col[~np.isnan(col)])
            if len(unique) <= 2 and np.all((unique == 0) | (unique == 1)):
                mask[i] = False
        return mask
