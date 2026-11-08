import numpy as np
from sklearn.linear_model import RidgeCV

from . import _stats
from ._stats import ScaledStats
from .base import BaseModel

# The grid `alphas=None` stands for: `np.logspace(-3, 4, 36)`, 36 penalties from 1e-3 to 1e4, a factor of 10^0.2 = 1.585 apart,
# the quality study's grid, so its Ridge row can be reproduced. Written out, not computed, so that a last-bit difference of
# `np.logspace` between numpy builds can't change what a manifest records and refuse a resume on another machine; the values are
# the ones numpy 2.4.0 computes here (a test compares them with `np.logspace`, to the bit on this build and to 1e-15 elsewhere).
DEFAULT_ALPHAS = [
    0.001,
    0.001584893192461114,
    0.0025118864315095794,
    0.003981071705534973,
    0.00630957344480193,
    0.01,
    0.01584893192461114,
    0.025118864315095808,
    0.039810717055349734,
    0.06309573444801933,
    0.1,
    0.15848931924611143,
    0.25118864315095824,
    0.3981071705534973,
    0.6309573444801936,
    1.0,
    1.584893192461114,
    2.5118864315095824,
    3.981071705534973,
    6.309573444801936,
    10.0,
    15.848931924611142,
    25.11886431509582,
    39.81071705534978,
    63.09573444801943,
    100.0,
    158.48931924611142,
    251.18864315095823,
    398.1071705534977,
    630.9573444801943,
    1000.0,
    1584.893192461114,
    2511.886431509582,
    3981.0717055349774,
    6309.573444801943,
    10000.0,
]

# The smallest penalty accepted. scikit-learn's leave-one-out route loses digits in proportion to 1/alpha: against an exact SVD
# ridge on correlated standardised columns (n 364, p 247, coefficients of about 9) the largest coefficient error is 1.7e-10 at
# 1e-3, 1.7e-7 at 1e-6, 1.8e-4 at 1e-9 and 0.17 at 1e-12.
MIN_ALPHA = 1e-6


def _check_alphas(value):
    """None, or the grid as a list of Python floats; anything else that isn't a non-empty 1-D sequence of positive, finite
    penalties raises here, not in the first fit of every worker."""
    if value is None:
        return None
    message = (
        f"alphas must be None (the default grid) or a non-empty sequence of positive, finite penalties, got {value!r}"
    )
    too_small = (
        f"alphas below {MIN_ALPHA:g} are refused: scikit-learn's leave-one-out route loses digits in proportion to 1/alpha "
        "there. Use OLSModel for an unpenalised fit."
    )
    if isinstance(value, (str, bytes)) or any(
        isinstance(a, (bool, np.bool_)) for a in np.ravel(np.asarray(value, dtype=object))
    ):
        raise ValueError(message)
    try:
        grid = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if grid.ndim != 1 or grid.size == 0 or not (np.isfinite(grid).all() and (grid > 0).all()):
        raise ValueError(message)
    if (grid < MIN_ALPHA).any():
        raise ValueError(f"{too_small} Got {value!r}")
    return grid.tolist()


class RidgeModel(BaseModel):
    # What the step-5 engine must put in the `ScaledStats` it hands `_fit_stats`: the scaled rows too, for the leverages.
    _ENGINE_NEEDS = frozenset({"rows"})
    # No bound in standard deviations: an issue row goes to the model path where RidgeCV's own rounding at it exceeds the budget
    # it has at the training rows (`_stats.issue_rounding_ok`, `RIDGE_ROUNDING_MAX`), which is how far out a row may be.
    _ENGINE_ISSUE_Z = float("inf")
    # Saved to YAML only when not at these values: `alphas=None` is the default grid, which a manifest records in full.
    _WRITTEN_WHEN_SET = {**BaseModel._WRITTEN_WHEN_SET, "alphas": None, "fit_intercept": True}

    def __init__(self, predictors, training_window=365, alphas=None, *, fit_intercept=True, **kwargs):
        """Ridge regression whose penalty is chosen on every fit by scikit-learn's `RidgeCV` with `cv=None`: the leave-one-out
        error of each penalty on the training rows, in closed form, and the first penalty with the lowest one. The rows are
        scaled first, but the default scaler leaves 0/1 columns as they are, so a dummy is shrunk harder than a standardised
        column (a rare one most): `StandardScaler(scale_binary=True)` standardises them too, for the same penalty on every column.

        `alphas` is the grid, None for `np.logspace(-3, 4, 36)`; a manifest records the grid in full. `fit_intercept` is
        scikit-learn's: True (the default) fits an intercept on the standardised data, which the penalty doesn't shrink."""
        super().__init__(predictors, training_window, **kwargs)
        self.alphas = alphas  # checked and normalised in __setattr__
        self.fit_intercept = fit_intercept

    def __setattr__(self, key, value):
        if key == "alphas":
            value = _check_alphas(value)
        super().__setattr__(key, value)

    @property
    def _model_kwargs(self):
        # The resolved grid, not None: a manifest then says what was searched, and a resume after the default changes refuses.
        return {
            "alphas": list(DEFAULT_ALPHAS if self.alphas is None else self.alphas),
            "fit_intercept": self.fit_intercept,
        }

    def _estimator(self):
        return RidgeCV(**self._model_kwargs, cv=None)

    def _fit_predict(self, train_x, train_y, test_x):
        return self._predict_fitted(self._fit_model(train_x, train_y), test_x)

    def _fit_model(self, train_x, train_y):
        if len(train_y) < 3:
            # Leave-one-out refits on n-1 rows, and with an intercept one or two rows leave every row fitted exactly: the error of
            # each penalty is 0/0. scikit-learn then warns, skips the NaN penalties or, when the first is NaN, keeps the first and
            # returns a forecast that looks like any other. A ValueError is a counted skip (`fit_error`).
            raise ValueError(
                f"RidgeModel needs at least 3 training rows to choose its penalty by leave-one-out, got {len(train_y)}"
            )
        model = self._estimator()
        model.fit(train_x, train_y)
        return model

    def _predict_fitted(self, fitted, test_x):
        return fitted.predict(test_x)[0], fitted.coef_.tolist()

    def _fit_stats(self, stats: ScaledStats):
        """`RidgeCV(alphas, fit_intercept, cv=None)` from a unit's scaled sums and rows (`_stats.ridge_fit`: the eigendecomposition
        of the centred Gram and exact leave-one-out leverages), or None where the model path must fit it: n <= p (scikit-learn's
        `eigen` mode), fewer than 3 rows, or scikit-learn's own rounding at the chosen penalty above `_stats.RIDGE_ROUNDING_MAX`
        (it errs by about `_stats.ridgecv_rounding`/α, so no exact fit can stay within 1e-9 of it there), and wherever
        `_estimator()` isn't exactly that `RidgeCV` with valid parameters (`_stats.estimator_params`). The settings are the
        estimator's own."""
        params = _stats.estimator_params(self._estimator(), RidgeCV, ("alphas", "fit_intercept"), {"cv": None})
        if params is None:
            return None
        return _stats.ridge_fit(stats, params["alphas"], bool(params["fit_intercept"]))
