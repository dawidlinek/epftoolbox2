from sklearn.linear_model import LinearRegression
from .base import BaseModel


class OLSModel(BaseModel):
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

    def _fit_predict(self, train_x, train_y, test_x):
        model = LinearRegression(fit_intercept=self.fit_intercept)
        model.fit(train_x, train_y)
        return model.predict(test_x)[0], model.coef_.tolist()

    def _fit_model(self, train_x, train_y):
        model = LinearRegression(fit_intercept=self.fit_intercept)
        model.fit(train_x, train_y)
        return model

    def _predict_fitted(self, fitted, test_x):
        return fitted.predict(test_x)[0], fitted.coef_.tolist()
