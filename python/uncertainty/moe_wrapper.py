from sklearn.base import BaseEstimator, RegressorMixin
from ..ml.moe_model import MOEModel

class MOEModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, moe_model: MOEModel, preprocess: bool = False):
        self.moe_model = moe_model
        self.preprocess = preprocess
        self._estimator_type = "regressor"
        self.fitted_ = self.moe_model.is_trained_total()

    def fit(self, data, target):
        return self

    def predict(self, data):
        return self.moe_model.predict_labels_blind_without_cv(data, preprocess=self.preprocess)
