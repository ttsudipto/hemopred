from typing import Dict, Tuple
from .moe_wrapper import MOEModelWrapper
from mapie.regression import SplitConformalRegressor
from copy import deepcopy
import numpy as np

class MAPIEModel:
    def __init__(
            self,
            estimator: MOEModelWrapper,
            data: np.ndarray,
            target: np.ndarray
    ) -> None:
        self.estimator = estimator
        self.data = data
        self.target = target
        self.trained = False
        self.uncertainty_estimator = None

    def is_trained(self) -> bool:
        return self.trained

    def _create_estimator(self, params: Dict) -> SplitConformalRegressor:
        uncertainty_estimator = SplitConformalRegressor(
            estimator=self.estimator,
            prefit=True,
            n_jobs=8,
            verbose=0,
            **params)
        return uncertainty_estimator

    def learn(self, params: Dict) -> None:
        self.trained = False
        self.uncertainty_estimator = None
        uncertainty_estimator = self._create_estimator(params)
        uncertainty_estimator.conformalize(self.data, self.target)
        self.uncertainty_estimator = deepcopy(uncertainty_estimator)
        self.trained = True

    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred, y_pis = self.uncertainty_estimator.predict_interval(data)
        return y_pred, y_pis

    def predict_prediction_intervals(self, data: np.ndarray) -> np.ndarray:
        _, y_pis = self.uncertainty_estimator.predict_interval(data)
        return y_pis

