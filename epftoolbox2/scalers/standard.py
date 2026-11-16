import numpy as np
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .base import BaseScaler, mean_std, scale_training_column


class StandardScaler(BaseScaler):
    """Standardise every non-binary predictor and the target to mean 0, standard deviation 1 (the default).

    `scale_binary=True` standardises the binary 0/1 predictors too, ignoring the mask of columns that may be scaled, as a
    study that standardises its whole design matrix does (the week-ahead study's Lasso, with no intercept). The default
    False leaves them as they are.

    A predictor that is constant up to rounding noise (standard deviation at or below 1e-12 of its largest absolute
    value) has training values of exactly 0.0, and its test row is centred and divided by 1.0, not by about 1e-15. The
    target is centred and divided by 1.0 in that case. A column with fewer than two values has the scale 1.0 as well.
    """

    name = "standard"

    def __init__(self, *, scale_binary: bool = False):
        self.scale_binary = scale_binary
        self._target_mean = 0.0
        self._target_std = 1.0
        # The last fit's columns (`transform`, `state`): which it scaled, with what centre and divisor; None before a fit.
        self._x_scaled: Optional[np.ndarray] = None
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None

    def get_params(self) -> Dict[str, Any]:
        # The default is left out, so the spec of `StandardScaler()` stays "standard" and no manifest or YAML file moves.
        return {"scale_binary": True} if self.scale_binary else {}

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

        if self.scale_binary:
            scalable_mask = np.ones(train_x.shape[1], dtype=bool)
        elif scalable_mask is None:
            scalable_mask = self.get_scalable_mask(train_x)

        self._x_scaled = np.zeros(train_x.shape[1], dtype=bool)
        self._x_mean = np.zeros(train_x.shape[1])
        self._x_std = np.ones(train_x.shape[1])
        for i in range(train_x.shape[1]):
            if not scalable_mask[i]:
                continue
            col = train_x[:, i]
            mean, std, constant = mean_std(col)
            train_x[:, i] = scale_training_column(col, col, mean, std, constant)
            test_x[:, i] = (test_x[:, i] - mean) / std
            self._x_scaled[i], self._x_mean[i], self._x_std[i] = True, mean, std

        self._target_mean, self._target_std, _ = mean_std(train_y)
        train_y_scaled = (train_y - self._target_mean) / self._target_std

        return train_x, train_y_scaled, test_x

    def inverse(self, value: float) -> float:
        return value * self._target_std + self._target_mean

    def transform(self, test_x: np.ndarray) -> np.ndarray:
        test_x = self._check_columns(test_x, None if self._x_scaled is None else self._x_scaled.size)
        for i in np.flatnonzero(self._x_scaled).tolist():
            test_x[:, i] = (test_x[:, i] - self._x_mean[i]) / self._x_std[i]
        return test_x

    def state(self) -> Dict[str, np.ndarray]:
        if self._x_scaled is None:
            raise ValueError("StandardScaler wasn't fitted: it has no state")
        return {
            **self._spec_state(),
            "x_scaled": self._x_scaled.copy(),
            "x_mean": self._x_mean.copy(),
            "x_std": self._x_std.copy(),
            "y_mean": np.array(self._target_mean, dtype=np.float64),
            "y_std": np.array(self._target_std, dtype=np.float64),
        }

    def _load_state(self, state: Mapping[str, np.ndarray]) -> None:
        self._x_scaled = np.array(state["x_scaled"], dtype=bool)
        self._x_mean = np.array(state["x_mean"], dtype=np.float64)
        self._x_std = np.array(state["x_std"], dtype=np.float64)
        if not self._x_scaled.shape == self._x_mean.shape == self._x_std.shape or self._x_scaled.ndim != 1:
            raise ValueError("a StandardScaler state's columns disagree in length")
        self._target_mean = np.float64(state["y_mean"])
        self._target_std = np.float64(state["y_std"])
