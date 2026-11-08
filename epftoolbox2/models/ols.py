from sklearn.linear_model import LinearRegression

from . import _stats
from ._stats import ScaledStats
from .base import BaseModel


class OLSModel(BaseModel):
    # What the step-5 engine must put in the `ScaledStats` it hands `_fit_stats`: the sums alone.
    _ENGINE_NEEDS = frozenset()
    # Saved to YAML only when not at these values, so files saved before the option existed don't move.
    _WRITTEN_WHEN_SET = {**BaseModel._WRITTEN_WHEN_SET, "fit_intercept": False}

    def __init__(self, predictors, training_window=365, *, fit_intercept=False, **kwargs):
        """`fit_intercept` is scikit-learn's: False (the default) fits none, since the target is standardised and one-hot
        dummies carry the level; True fits one, as an expert model with a few dummies that aren't a one-hot needs."""
        super().__init__(predictors, training_window, **kwargs)
        self.fit_intercept = fit_intercept

    @property
    def _model_kwargs(self):
        # The default is left out, so a manifest made before the option existed stays as it was.
        return {"fit_intercept": True} if self.fit_intercept else {}

    def _estimator(self):
        return LinearRegression(fit_intercept=self.fit_intercept)

    def _fit_predict(self, train_x, train_y, test_x):
        model = self._estimator()
        model.fit(train_x, train_y)
        return model.predict(test_x)[0], model.coef_.tolist()

    def _fit_model(self, train_x, train_y):
        model = self._estimator()
        model.fit(train_x, train_y)
        return model

    def _predict_fitted(self, fitted, test_x):
        return fitted.predict(test_x)[0], fitted.coef_.tolist()

    def _fit_stats(self, stats: ScaledStats):
        """The fit from a unit's scaled sums by the normal equations (`_stats.ols_fit`), or None where the model path must fit
        it: n <= p + 1, a failed Cholesky factorisation, either guard (`_stats.GUARD`, `_stats.CONDITION_MAX`), or
        `_estimator()` not exactly that `LinearRegression` with valid parameters (`_stats.estimator_params`)."""
        params = _stats.estimator_params(self._estimator(), LinearRegression, ("fit_intercept",))
        if params is None:
            return None
        return _stats.ols_fit(stats, bool(params["fit_intercept"]))
