import numbers

from sklearn.linear_model import LassoCV

from . import _cv_stats, _stats
from ._lars import CV_FOLDS
from ._stats import ScaledStats
from .base import BaseModel


class LassoCVModel(BaseModel):
    # What the step-5 engine must put in the `ScaledStats` it hands `_fit_stats`: the CV blocks (`_cv_stats.block_stats`).
    _ENGINE_NEEDS = frozenset({"blocks"})
    # The engine hands back a fit two of whose columns are near twins, 1 - |cos| of their centred values at most this (not
    # exact twins): the coordinate descent stops at a tolerance, so along such a pair it follows its Gram's last bits (to
    # 2e-8 EUR/MWh apart with a pair 1e-3 of a standard deviation apart, `engine_vst_cv/sweep.py`; 4.2e-8 at 1 - |cos| 5e-6 in
    # short windows, `sweep_dense.py`); within it the dense sweep's worst kept gap is 1.9e-9 against the engine's 1e-8
    # (`_engine_worker.EngineFits`, `near_twin`).
    _ENGINE_NEAR_TWIN = 1e-5
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
        where the model path must fit it: `_estimator()` not exactly that `LassoCV` with valid parameters
        (`_stats.estimator_params`), `cv` not a number of folds, no blocks or another number of them, a fold whose training
        part has no more rows than columns (n <= p among them), or anything that isn't finite. The settings are the
        estimator's own."""
        params = _stats.estimator_params(self._estimator(), LassoCV, ("cv", "max_iter", "fit_intercept"), {"n_jobs": 1})
        if params is None:
            return None
        folds = CV_FOLDS if params["cv"] is None else params["cv"]
        if isinstance(folds, bool) or not isinstance(folds, numbers.Integral):
            return None  # a splitter object: its folds aren't KFold's blocks
        return _cv_stats.lassocv_fit(stats, bool(params["fit_intercept"]), int(params["max_iter"]), folds=int(folds))
