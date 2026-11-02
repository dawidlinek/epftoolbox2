from sklearn.linear_model import LinearRegression
from .base import BaseModel


class OLSModel(BaseModel):
    def _fit_predict(self, train_x, train_y, test_x):
        model = LinearRegression(fit_intercept=False)
        model.fit(train_x, train_y)
        return model.predict(test_x)[0], model.coef_.tolist()

    def _fit_model(self, train_x, train_y):
        model = LinearRegression(fit_intercept=False)
        model.fit(train_x, train_y)
        return model

    def _predict_fitted(self, fitted, test_x):
        return fitted.predict(test_x)[0], fitted.coef_.tolist()
