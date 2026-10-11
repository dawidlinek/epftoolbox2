from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


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
