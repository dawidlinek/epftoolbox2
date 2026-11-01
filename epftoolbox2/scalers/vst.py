"""Variance-stabilising transformations (VSTs) for spiky prices.

References: Uniejewski, Weron & Ziel (2018), "Variance stabilizing transformations for electricity spot price
forecasting", IEEE Transactions on Power Systems 33(2); Uniejewski (2026), "Variance stabilizing transformations
for electricity price forecasting in periods of increased volatility", Electric Power Systems Research 257
(arXiv:2511.13603).
"""

import fnmatch
import functools
import warnings
from abc import abstractmethod
from statistics import NormalDist
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from .base import BaseScaler, is_constant, mean_std, scale_training_column

# MAD / _Z75 estimates the standard deviation of normally distributed data.
_Z75 = NormalDist().inv_cdf(0.75)


def _as_exactly_constant(values: np.ndarray) -> np.ndarray:
    """The finite values of a column that is constant up to rounding noise, replaced by their median."""
    finite = values[np.isfinite(values)]
    return np.full(finite.size, np.median(finite))


@functools.lru_cache(maxsize=None)
def _match_columns(columns: Tuple[str, ...], patterns: Tuple[str, ...]) -> np.ndarray:
    matched = np.array([any(fnmatch.fnmatchcase(c, p) for p in patterns) for c in columns], dtype=bool)
    if not matched.any():
        warnings.warn(
            f"Scaler apply_to patterns {list(patterns)} match none of the predictors {list(columns)}; "
            "only the target is transformed",
            stacklevel=2,
        )
    matched.flags.writeable = False
    return matched


@functools.lru_cache(maxsize=64)
def _normal_scores(n: int) -> np.ndarray:
    """Standard normal quantiles at m / (2(n+1)), m = 0..2(n+1), indexed by twice a (mid)rank in 1..n."""
    denom = 2 * (n + 1)
    normal = NormalDist()
    scores = np.empty(denom + 1)
    scores[0], scores[-1] = -np.inf, np.inf
    scores[1:-1] = [normal.inv_cdf(m / denom) for m in range(1, denom)]
    scores.flags.writeable = False
    return scores


class _VSTScaler(BaseScaler):
    """Per-column VST fitted on training rows, followed by standardisation of the transformed values.

    The target is always transformed. ``apply_to`` selects the predictors that are transformed too:
    ``"all"`` (every non-binary predictor), ``"target"`` (none) or a list of glob patterns matched against the
    expanded predictor names (e.g. ``["price", "price_d-*"]``). Other non-binary predictors are standardised as
    by StandardScaler; binary 0/1 predictors are never changed. Each column gets its own parameters.

    A column that is constant up to rounding noise (standard deviation at or below 1e-12 of its largest absolute
    value, ``scalers.base.CONSTANT_STD_RTOL``) is treated as the exactly constant column it stands for: the VST is
    fitted as for that constant and the standardising scale is 1.0, so its noise isn't stretched into a regressor.
    A predictor like that has training values of exactly 0.0 (NaNs stay NaN); its test row and the target are
    transformed and centred as usual.
    """

    def __init__(self, apply_to: Union[str, List[str]] = "all"):
        if isinstance(apply_to, str):
            if apply_to not in ("all", "target"):
                raise ValueError(
                    f"apply_to must be 'all', 'target' or a list of column patterns, got {apply_to!r} "
                    f"(for a single pattern use [{apply_to!r}])"
                )
        elif isinstance(apply_to, (list, tuple)) and apply_to and all(isinstance(p, str) for p in apply_to):
            apply_to = list(apply_to)
        else:
            raise ValueError(
                f"apply_to must be 'all', 'target' or a non-empty list of column patterns, got {apply_to!r}"
            )
        self.apply_to = apply_to

        self._column_params: List[Optional[Tuple[Any, float, float]]] = []
        self._target_params: Any = None
        self._target_mean = 0.0
        self._target_std = 1.0

    @abstractmethod
    def _fit(self, values: np.ndarray) -> Any:
        """Transformation parameters estimated from one column of training values."""

    @abstractmethod
    def _forward(self, params: Any, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _backward(self, params: Any, z: np.ndarray) -> np.ndarray:
        pass

    def _vst_mask(self, n_cols: int, columns: Optional[Sequence[str]]) -> np.ndarray:
        if self.apply_to == "all":
            return np.ones(n_cols, dtype=bool)
        if self.apply_to == "target":
            return np.zeros(n_cols, dtype=bool)
        if columns is None or len(columns) != n_cols:
            raise ValueError(f"{type(self).__name__}(apply_to=[...]) needs one predictor name per column")
        return _match_columns(tuple(columns), tuple(self.apply_to))

    def fit_transform(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        scalable_mask: Optional[np.ndarray] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_x = np.array(train_x, dtype=np.float64)
        test_x = np.array(test_x, dtype=np.float64)
        train_y = np.array(train_y, dtype=np.float64)

        if scalable_mask is None:
            scalable_mask = self.get_scalable_mask(train_x)
        vst_mask = self._vst_mask(train_x.shape[1], columns)

        self._column_params = []
        for i in range(train_x.shape[1]):
            if not scalable_mask[i]:
                self._column_params.append(None)
                continue
            raw = train_x[:, i]
            params, mean, std, constant, (train_col, test_col) = self._standardise(raw, [test_x[:, i]], vst_mask[i])
            train_x[:, i] = scale_training_column(raw, train_col, mean, std, constant)
            test_x[:, i] = (test_col - mean) / std
            self._column_params.append((params, mean, std))

        self._target_params, self._target_mean, self._target_std, _, (train_z,) = self._standardise(train_y, [], True)
        return train_x, (train_z - self._target_mean) / self._target_std, test_x

    def _standardise(
        self, train: np.ndarray, others: List[np.ndarray], vst: bool
    ) -> Tuple[Any, float, float, bool, List[np.ndarray]]:
        """VST parameters (``vst``), mean, scale and constant flag of one column, and the VST of it and of ``others``.

        The mean and scale are those of the transformed training values. If the raw column is constant up to rounding
        noise, the VST is fitted on its median repeated, as for an exactly constant column, and the scale is 1.0. It
        takes the raw column to tell: the transformed noise is as small as the noise was, but not small next to its
        own largest value.
        """
        constant = vst and is_constant(train)
        params = None
        if vst:
            params = self._fit(_as_exactly_constant(train) if constant else train)
            train, others = self._forward(params, train), [self._forward(params, o) for o in others]
        mean, std, flat = mean_std(train)
        constant = constant or flat
        return params, mean, 1.0 if constant else std, constant, [train, *others]

    def inverse(self, value: float) -> float:
        z = np.asarray(value, dtype=np.float64) * self._target_std + self._target_mean
        restored = self._backward(self._target_params, z)
        return float(restored) if np.ndim(restored) == 0 else restored


class AsinhScaler(_VSTScaler):
    """Median/MAD normalisation followed by asinh; epftoolbox's "Invariant" scaler.

    ``x -> asinh((x - median) / (MAD / 0.6745))`` with median and MAD from the training rows. If the MAD is 0 (more
    than half of the training values are equal) the sample standard deviation is used instead, and 1.0 if that is 0
    too, or at most rounding noise (``scalers.base.CONSTANT_STD_RTOL``). Defined for negative prices; the inverse is
    ``median + scale * sinh(z)``.
    """

    name = "asinh"

    def _fit(self, values: np.ndarray) -> Tuple[float, float]:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0, 1.0
        center = float(np.median(values))
        scale = float(np.median(np.abs(values - center))) / _Z75
        if not scale > 0:
            scale = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            if not scale > 0 or is_constant(values):
                scale = 1.0
        return center, scale

    def _forward(self, params: Tuple[float, float], x: np.ndarray) -> np.ndarray:
        center, scale = params
        return np.arcsinh((x - center) / scale)

    def _backward(self, params: Tuple[float, float], z: np.ndarray) -> np.ndarray:
        center, scale = params
        return center + scale * np.sinh(z)


class NPITScaler(_VSTScaler):
    """Normal probability integral transform: ``x -> Phi^-1(F(x))`` with F the training rows' empirical CDF.

    The i-th smallest of n training values gets F = i / (n + 1) (tied values share their mid-rank), so the knots
    map to finite normal quantiles. Between knots the map is linear, and values outside the training range are
    clamped to the extreme knots: a transformed predictor can't exceed the training range, and a back-transformed
    forecast stays within [min, max] of the training target.
    """

    name = "npit"

    def _fit(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        values = np.sort(values[np.isfinite(values)])
        if values.size == 0:
            return np.zeros(1), np.zeros(1)
        knots_x, first, counts = np.unique(values, return_index=True, return_counts=True)
        # 1-based positions first+1 .. first+counts; twice their mean indexes _normal_scores.
        knots_z = _normal_scores(values.size)[2 * first + counts + 1]
        return knots_x, knots_z

    def _forward(self, params: Tuple[np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
        knots_x, knots_z = params
        return np.interp(x, knots_x, knots_z)

    def _backward(self, params: Tuple[np.ndarray, np.ndarray], z: np.ndarray) -> np.ndarray:
        knots_x, knots_z = params
        return np.interp(z, knots_z, knots_x)
