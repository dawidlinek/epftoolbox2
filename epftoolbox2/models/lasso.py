import numbers

from sklearn.linear_model import LassoCV

from . import _cv_stats
from ._lars import CV_FOLDS
from ._stats import ScaledStats
from .base import BaseModel


class LassoCVModel(BaseModel):
    # What the step-5 engine must put in the `ScaledStats` it hands `_fit_stats`: the CV blocks (`_cv_stats.block_stats`).
    _ENGINE_NEEDS = frozenset({"blocks"})
    # Saved to YAML only when not at these values, so files saved before the option existed don't move.
    _WRITTEN_WHEN_SET = {**BaseModel._WRITTEN_WHEN_SET, "fit_intercept": True}

    def __init__(self, predictors, training_window=365, cv=5, max_iter=10000, *, fit_intercept=True, **kwargs):
        """`fit_intercept` is scikit-learn's: True (the default) fits an intercept on the standardised data, False fits
        none, as a study that centres the target itself does (the week-ahead study's Lasso)."""
        super().__init__(predictors, training_window, **kwargs)
        self.cv = cv
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept

    @property
    def _model_kwargs(self):
        # The default is left out, so a manifest made before the option existed stays as it was.
        return {"cv": self.cv, "max_iter": self.max_iter, **({} if self.fit_intercept else {"fit_intercept": False})}

    def _estimator(self):
        return LassoCV(cv=self.cv, max_iter=self.max_iter, n_jobs=1, fit_intercept=self.fit_intercept)

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
        """`LassoCV(cv, max_iter, n_jobs=1, fit_intercept)` from a unit's scaled sums and its KFold blocks
        (`_cv_stats.lassocv_fit`: scikit-learn's own coordinate descent on each fold's Gram through `enet_path`), or None
        where the model path must fit it: `cv` not a number of folds, no blocks or another number of them, a fold whose
        training part has no more rows than columns (n <= p among them), or anything that isn't finite."""
        folds = CV_FOLDS if self.cv is None else self.cv
        if isinstance(folds, bool) or not isinstance(folds, numbers.Integral):
            return None  # a splitter object: its folds aren't KFold's blocks
        return _cv_stats.lassocv_fit(stats, self.fit_intercept, self.max_iter, folds=int(folds))
