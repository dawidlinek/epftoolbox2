import numpy as np
from sklearn.linear_model import RidgeCV

from .base import BaseModel

# The grid `alphas=None` stands for: 36 penalties from 1e-3 to 1e4, a factor of 1.38 apart. On standardised columns that runs from
# no shrinkage to a ridge that leaves the intercept alone. The quality study's grid, so its Ridge row can be reproduced.
DEFAULT_ALPHAS = np.logspace(-3, 4, 36).tolist()


def _check_alphas(value):
    """None, or the grid as a list of Python floats; anything else that isn't a non-empty 1-D sequence of positive, finite
    penalties raises here, not in the first fit of every worker."""
    if value is None:
        return None
    message = (
        f"alphas must be None (the default grid) or a non-empty sequence of positive, finite penalties, got {value!r}"
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
    return grid.tolist()


class RidgeModel(BaseModel):
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
