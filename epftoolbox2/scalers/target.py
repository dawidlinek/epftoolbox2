"""Target transforms of a fit with several outputs: one Invariant per output.

A day unit (`BaseModel._unit_kind = "day"`, `models/_units.py`) fits the P periods of its days at once, and a pooled unit
with `target_transform="per_period"` the rows of every period together. Each period's targets then get their own
transformation, fitted on that period's training targets alone, exactly as a period unit's
`AsinhScaler(apply_to="target", standardise=False)` fits it (epftoolbox's "Invariant"), and a forecast of period p is mapped
back with p's: a column is that scaler's to the bit. With `target_transform="none"` a unit fits the raw targets
(`RawTarget`, the identity there and back).
"""

from typing import Optional, Tuple

import numpy as np

from .base import is_constant
from .vst import _as_exactly_constant, _median_mad


def invariant_params(values: np.ndarray) -> Tuple[float, float]:
    """(median, scale) of the Invariant of these training values, as `AsinhScaler` fits its target: the median and
    MAD / 0.6745 of the finite values, the sample standard deviation where the MAD is 0, and 1.0 where that is 0 too; a column
    constant up to rounding noise (`scalers.base.CONSTANT_STD_RTOL`) as the exactly constant column it stands for (its median,
    scale 1.0). No value at all is (0.0, 1.0)."""
    values = np.asarray(values, dtype=np.float64)
    if is_constant(values):
        values = _as_exactly_constant(values)
    return _median_mad(values)


class MultiTargetAsinh:
    """`asinh((y - median) / (MAD / 0.6745))` with a median and MAD of each output column (or group of rows) of its own.

    `fit(y, mask)` takes the training targets as (n, P), one column per output (a period of the day), and fits each column on
    its finite values where `mask` is True (a slot the clock skips, the repeated autumn hour, a missing value or a target not
    yet published by the issue time is masked); `fit_groups(y, groups, n_groups)` fits stacked targets (N,) by their group (a
    pooled unit's rows by their period), and `fit_groups(y, None, 1)` all of them as one. `transform` and `inverse` take the
    same shapes: an (n, P) array, or a 1-D one with each value's column in `groups`. A column with no training value is (0, 1):
    a fit has nothing to learn there, and the unit skips that period's forecasts (`no_finite_training_rows`).
    """

    name = "multi_asinh"

    def __init__(self):
        self.center = np.zeros(0)
        self.scale = np.ones(0)

    def fit(self, y: np.ndarray, mask: Optional[np.ndarray] = None) -> "MultiTargetAsinh":
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError(f"MultiTargetAsinh.fit takes the targets as (rows, outputs), got shape {y.shape}")
        valid = np.isfinite(y) if mask is None else np.asarray(mask, dtype=bool) & np.isfinite(y)
        params = [invariant_params(y[valid[:, column], column]) for column in range(y.shape[1])]
        self.center = np.array([center for center, _ in params], dtype=np.float64)
        self.scale = np.array([scale for _, scale in params], dtype=np.float64)
        return self

    def fit_groups(self, y: np.ndarray, groups: Optional[np.ndarray], n_groups: int) -> "MultiTargetAsinh":
        y = np.asarray(y, dtype=np.float64)
        if groups is None:
            groups = np.zeros(len(y), dtype=np.int64)
        groups = np.asarray(groups)
        finite = np.isfinite(y)
        params = [invariant_params(y[finite & (groups == group)]) for group in range(n_groups)]
        self.center = np.array([center for center, _ in params], dtype=np.float64)
        self.scale = np.array([scale for _, scale in params], dtype=np.float64)
        return self

    def _params(self, shape: Tuple[int, ...], groups: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if groups is not None:
            groups = np.asarray(groups, dtype=np.int64)
            return self.center[groups], self.scale[groups]
        if len(shape) == 2:
            return self.center[None, :], self.scale[None, :]
        if len(self.center) == 1:  # one group, stacked values
            return self.center, self.scale
        raise ValueError("a 1-D transform of several columns needs each value's column (groups)")

    def transform(self, y: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        center, scale = self._params(y.shape, groups)
        return np.arcsinh((y - center) / scale)

    def inverse(self, z: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
        # `* 1.0 + 0.0` as `AsinhScaler.inverse` computes it with standardise=False, so that a column is that scaler's to the bit
        # (the sign of a zero included).
        z = np.asarray(z, dtype=np.float64) * 1.0 + 0.0
        center, scale = self._params(z.shape, groups)
        return center + scale * np.sinh(z)


class RawTarget:
    """The raw target in `MultiTargetAsinh`'s interface (`target_transform="none"`): nothing is fitted, `transform` and
    `inverse` give back their values as float64, unchanged (no centring, no scale, no asinh), so a unit's learner fits and
    predicts in target units."""

    name = "none"

    def fit(self, y: np.ndarray, mask: Optional[np.ndarray] = None) -> "RawTarget":
        return self

    def fit_groups(self, y: np.ndarray, groups: Optional[np.ndarray], n_groups: int) -> "RawTarget":
        return self

    def transform(self, y: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
        return np.asarray(y, dtype=np.float64)

    def inverse(self, z: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
        return np.asarray(z, dtype=np.float64)
