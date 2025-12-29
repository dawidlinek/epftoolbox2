from sklearn.linear_model import LinearRegression
from .base import BaseModel


class OLSModel(BaseModel):
    def _fit_predict(self, train_x, train_y, test_x):
        model = LinearRegression()
        model.fit(train_x, train_y)
        return model.predict(test_x)[0], model.coef_.tolist()
