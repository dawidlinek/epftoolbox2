from sklearn.linear_model import LassoCV
from .base import BaseModel


class LassoCVModel(BaseModel):
    def __init__(self, predictors, training_window=365, cv=5, max_iter=10000, **kwargs):
        super().__init__(predictors, training_window, **kwargs)
        self.cv = cv
        self.max_iter = max_iter

    def _fit_predict(self, train_x, train_y, test_x):
        model = LassoCV(cv=self.cv, max_iter=self.max_iter, n_jobs=1)
        model.fit(train_x, train_y)
        return model.predict(test_x)[0], model.coef_.tolist()
